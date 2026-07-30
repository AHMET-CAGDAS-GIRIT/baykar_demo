from pydantic import BaseModel, Field
from typing import List, Optional

class IngestDetails(BaseModel):
    """
    Yüklenen dosyanın arka planda nasıl parçalandığına (chunking) dair detaylar.
    """
    total_pages: Optional[int] = Field(None, ge=1, description="İşlenen toplam sayfa sayısı")
    text_chunks_count: int = Field(..., ge=0, description="Oluşturulan metin parçası sayısı")
    extracted_images_count: int = Field(..., ge=0, description="Ayrıştırılan ve vektörleştirilen görsel sayısı")


class IngestResponse(BaseModel):
    """
    /ingest endpoint'i başarılı olduğunda istemciye (frontend) dönen yanıt şeması.
    """
    status: str = Field(
        ..., 
        examples=["success"], 
        description="İşlem durumu ('success' veya 'failed')"
    )
    document_id: str = Field(
        ..., 
        examples=["doc_98234723984"], 
        description="Dokümana sistemde verilen benzersiz Kimlik (UUID / Hash)"
    )
    filename: str = Field(
        ..., 
        examples=["akciger_toraks_rehberi.pdf"], 
        description="Yüklenen orijinal dosya adı"
    )
    message: str = Field(
        ..., 
        examples=["Doküman başarıyla işlendi, metin ve görseller Qdrant'a indekslendi."]
    )
    details: Optional[IngestDetails] = Field(
        None, 
        description="Parçalama (chunking) ve görsel çıkarma detayları"
    )
    latency_ms: float = Field(
        ..., 
        ge=0.0, 
        examples=[345.8], 
        description="Yükleme ve indekslemenin toplam süresi (milisaniye)"
    )
    warnings: List[str] = Field(
        default_factory=list, 
        examples=[["3. sayfadaki düşük kaliteli görsel atlandı."]],
        description="İşlem sırasında oluşan kritik olmayan uyarılar"
    )
    chunks: Optional[List[str]] = Field(
        None, 
        description="Dokümandan elde edilen metin parçalarının (chunk) listesi"
    )