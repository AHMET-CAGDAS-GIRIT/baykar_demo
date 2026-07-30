import os
from fastapi import HTTPException
from app.services.rag_engine import LocalMedicalRAG
from app.schemas import QueryResponse,IngestResponse

class QueryService:
    # Sınıf seviyesinde RAG motorunu bir kez yükleyelim (Singleton mantığı)
    # Böylece her istekte model tekrar tekrar belleğe yüklenmez.
    _rag_engine = None
    
    @staticmethod
    def _get_rag_engine(json_path: str) -> LocalMedicalRAG:
        if QueryService._rag_engine is None:
            print(f"RAG Motoru başlatılıyor. Veri yolu: {json_path}")
            QueryService._rag_engine = LocalMedicalRAG(json_path=json_path)
        return QueryService._rag_engine

    @staticmethod
    async def search_and_answer(question: str, image_base64: str = None, top_k: int = 3, ingest_metadata: IngestResponse = None) -> QueryResponse:
        """
        Router'dan gelen istekleri karşilar, RAG motorunu çaliştirir 
        ve QueryResponse şemasina uygun olarak sonucu döner.
        TODO image_base64 ve ingest_metadata parametreleri RAG motoru tarafından kullanılacak şekilde entegre edilecek.
        """
        if QueryService._rag_engine is None:
            raise HTTPException(status_code=500, detail="RAG motoru aktif değil.")

        # RAG motoru üzerinden yanıt ve ilgili görselleri al
        answer, evidence_items, confidence_score, query_warnings = QueryService._rag_engine.query(
            user_question=question, 
            top_k=top_k,
            image_base64=image_base64,
            ingest_metadata=ingest_metadata
        )    
        
        #Eğer ingest işlemi sırasında uyarılar (warnings) oluştuysa, bunları sorgu uyarılarıyla birleştiriyoruz.
        final_warnings = []
        if query_warnings:
            final_warnings = list(query_warnings)
            
        if ingest_metadata:
            if ingest_metadata.warnings:
                final_warnings.extend(ingest_metadata.warnings)

        # Router'ın beklediği QueryResponse modeline göre veriyi döndür
        return QueryResponse(
            answer=answer,
            evidence=evidence_items,
            confidence_score=confidence_score,
            warnings=final_warnings,
            latency_ms=0.0  # Router içinde zaten hesaplanıp güncelleniyor
        )