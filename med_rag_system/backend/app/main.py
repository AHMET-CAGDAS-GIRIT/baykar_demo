import os
import traceback
import io
import base64
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.services import QueryService
from app.schemas import QueryRequest, QueryResponse
from app.api.routes_query import query_documents
from app.api.routes_ingest import ingest_document

app = FastAPI(title="Medikal RAG API")

# CORS Ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme aşaması için tüm originlere izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Klasör yolları
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # app/
BASE_DIR = os.path.dirname(CURRENT_DIR) # backend/
THUMBNAILS_DIR = os.path.join(BASE_DIR, "data", "thumbnails") # backend/data/thumbnails/
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

# Görselleri sunmak için StaticFiles mount işlemi
app.mount("/static", StaticFiles(directory=THUMBNAILS_DIR), name="static")

# RAG motorunu başlatma
JSON_PATH = os.path.join(BASE_DIR, "data", "parsed_documents.json")

try:
    rag_engine = QueryService._get_rag_engine(json_path=JSON_PATH)
except Exception as e:
    print(f"RAG motoru başlatılamadı: {e}")
    rag_engine = None

@app.post("/query")
async def ask_medical_question(payload: QueryRequest) -> QueryResponse:
    if not rag_engine:
        raise HTTPException(status_code=500, detail="RAG motoru aktif değil.")
    
    try:
        if payload.pdf_base64 is None:
            result = await query_documents(payload)
            return result
        
        base64_data = payload.pdf_base64
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
            
        pdf_bytes = base64.b64decode(base64_data)
        
        # Ingest fonksiyonunun beklediği UploadFile formatını simüle ediyoruz
        file_obj = UploadFile(
            file=io.BytesIO(pdf_bytes),
            filename="uploaded_document.pdf",
            headers={"content-type": "application/pdf"}
        )
        
        try:
            ingest_result = await ingest_document(file=file_obj)
        except Exception as ingest_err:
            print("--- PDF İŞLEME (INGEST) HATASI ---")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"PDF işlenemedi: {str(ingest_err)}")

        payload.ingest_metadata = ingest_result
        result = await query_documents(payload)
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print("--- RAG SORGULAMA HATASI ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))