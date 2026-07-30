import time
import uuid
import filetype  # Magic bytes doğrulama kütüphanesi
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.schemas import IngestResponse
from app.utils import extract_file_extension
from app.services import IngestionService


router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Güvenlik Parametreleri
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",  # Standart dışı gelen bazı JPG istekleri için tolerans
}

@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    start_time = time.time()
    warnings: List[str] = []

    # 1. Uzantı Kontrolü
    filename = file.filename or "unknown_file"
    ext = extract_file_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz dosya uzantısı! İzin verilenler: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. MIME Type Kontrolü (İçerik güvenliği)
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Desteklenmeyen dosya tipi ({file.content_type})."
        )
        
    # 3. Dosya Boyutu Kontrolü
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dosya boyutu çok büyük ({len(content) / (1024*1024):.1f}MB). Limit: 15MB."
        )

    # 4. GERÇEK TİP KONTROLÜ (Magic Bytes / Header Doğrulama)
    kind = filetype.guess(content)

    if kind is None:
        detected_mime = "Bilinmeyen/Bozuk Veri"
    else:
        detected_mime = kind.mime

    # Tipi belirsizse veya izin verilen MIME tipleri içinde değilse reddet
    if kind is None or kind.mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Güvenlik İhlali: Dosya uzantısı '{ext}' olarak görünse de gerçek dosya tipi '{detected_mime}'! Yalnızca gerçek PDF ve Görsel dosyaları kabul edilir."
        )

    # 5. Benzersiz Doküman ID Oluşturma
    document_id = f"doc_{uuid.uuid4().hex[:12]}"

    # 6. GERÇEK SERVİS ÇAĞRISI (Ağır iş yükünü servise devrettik)
    try:
        details, service_warnings, document_chunks = await IngestionService.process_and_index_document(
            file_bytes=content,
            filename=filename,
            file_ext=ext,
            document_id=document_id
        )
        warnings.extend(service_warnings)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Doküman işlenirken bir sunucu hatası oluştu: {str(e)}"
        )

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return IngestResponse(
        status="success",
        document_id=document_id,
        filename=filename,
        message="Doküman başarıyla işlendi, metin ve tıbbi görseller indekslendi.",
        details=details,
        latency_ms=latency_ms,
        warnings=warnings,
        chunks=document_chunks
    )