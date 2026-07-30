from pydantic import BaseModel, Field, computed_field
from typing import List, Optional
from app.schemas.ingest import IngestResponse

class QueryRequest(BaseModel):
    question: str = Field(
        ..., 
        description="Arama yapılacak metin veya sorulan soru.",
        examples=["Göğüs röntgeninde pnömoni bulgusu var mı?"]
    )
    image_base64: Optional[str] = Field(None, description="Opsiyonel Base64 formatında medikal görsel")
    pdf_base64: Optional[str] = Field(None, description="Opsiyonel Base64 formatında PDF dosyası")
    top_k: int = Field(default=3, ge=1, le=10) # cevap verilirken kullanılacak kaynak sayısı.
    ingest_metadata: Optional[IngestResponse] = Field(None, description="İşlenen dokümanın indeksleme detayları")

class EvidenceItem(BaseModel):
    source_file: str
    modalite: str  # "text", "image", "image_and_text"
    score: float
    thumbnail_url: Optional[str] = None
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    evidence: List[EvidenceItem]
    latency_ms: float
    warnings: List[str]
    confidence_score: float = Field(..., ge=0.0, le=1.0) # 0.0 ile 1.0 arasında bir değer olmalı
    # confidence_level sadece confidence_score'a bağlı olduğundan otomatik güncellecek şekilde yapıldı.
    # amacı çıktının okunurluğunu kolaylaştırmak ve kullanıcıya daha anlaşılır bir geri bildirim sunmaktır.
    @computed_field
    @property
    def confidence_level(self) -> str:
        if self.confidence_score >= 0.80:
            return "HIGH"
        elif self.confidence_score >= 0.60:
            return "MEDIUM"
        return "LOW"
    
    
    