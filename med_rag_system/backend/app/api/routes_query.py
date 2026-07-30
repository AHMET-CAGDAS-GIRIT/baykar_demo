import time
from fastapi import APIRouter, HTTPException, status, Depends

# Şemalar ve Servisler (Kendi proje yapına göre importları düzenleyebilirsin)
from app.schemas import QueryRequest, QueryResponse
from app.services import QueryService

router = APIRouter(tags=["Query"])

@staticmethod
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Kullanicinin sorusunu ve varsa opsiyonel görselini alir, 
    vektör veritabaninda semantik arama yapar ve kanitlariyla birlikte yanit döner.
    """
    start_time = time.time()

    # 1. Temel Girdi Kontrolü
    clean_question = request.question.strip()
    if not clean_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Soru metni boş olamaz."
        )

    # 2. SERVİS ÇAĞRISI (Arama ve RAG mantığı serviste çalışır)
    try:
        response_data = await QueryService.search_and_answer(#LLM ve VLM bu fonksiyon tarafından kullanılıyor.
            question=clean_question,
            image_base64=request.image_base64,
            top_k=request.top_k,
            ingest_metadata=request.ingest_metadata
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sorgu işlenirken bir sunucu hatası oluştu: {str(e)}"
        )

    # 3. Toplam Sürenin (Latency) Hesaplanması ve Güncellenmesi
    latency_ms = round((time.time() - start_time) * 1000, 2)
    response_data.latency_ms = latency_ms

    return response_data