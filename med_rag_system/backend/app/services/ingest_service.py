import io
from typing import List, Tuple
import fitz  # PyMuPDF. PDF için
from PIL import Image
import pytesseract

from app.schemas import IngestDetails


class IngestionService:
    """
    Doküman işleme (Ingestion) servisi.
    PDF ve Görsellerden metin/görsel çıkarma, chunking ve 
    Qdrant veritabanına indeksleme süreçlerini yönetir.
    """

    @classmethod
    async def process_and_index_document(
        cls,
        file_bytes: bytes,
        filename: str,
        file_ext: str,
        document_id: str
    ) -> Tuple[IngestDetails, List[str]]: # List[str] warnings listesi
        """
        Gelen ham dosya verisini işler, vektörleştirme hazırlığını yapar ve detayları döner.
        """
        warnings: List[str] = []
        all_chunks: List[str] = [] 
        
        # 1. Dosya Tipine Göre Parçalama / OCR Mantığı
        if file_ext == "pdf":
            total_pages, extracted_chunks, images_count = cls._process_pdf(file_bytes, warnings)
        else:
            total_pages, extracted_chunks, images_count = cls._process_image(file_bytes, warnings)

        all_chunks.extend(extracted_chunks)

        # 2. Qdrant Indeksleme Adımı (Gelecek adımda Vektör DB eklenecek)
        # await cls._index_to_qdrant(document_id, text_chunks)

        details = IngestDetails(
            total_pages=total_pages,
            text_chunks_count=len(all_chunks),
            extracted_images_count=images_count
        )

        return details, warnings, all_chunks

    @classmethod
    def _process_pdf(cls, file_bytes: bytes, warnings: List[str]) -> Tuple[int, int, int]:
        """
        PyMuPDF kullanarak PDF sayfalarını tarar, metinleri ayıklar, 
        metin yoksa OCR dener ve içindeki görselleri sayar. OCR sonrası metinleri ayıklar.
        """
        extracted_chunks: List[str] = []
        total_extracted_images = 0

        # PDF'i bellekten aç
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Sayfadan metin çıkar
            text = page.get_text().strip()

            # Sayfa metinsizse (resim/taranmış belge ise) OCR çalıştır
            if not text:
                try:
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_result = pytesseract.image_to_string(img, lang="tur+eng").strip()
        
                    if ocr_result:
                        text = ocr_result
                        warnings.append(f"Sayfa {page_num + 1} üzerinde OCR çalıştırıldı ve metin çıkarıldı.")
                    else:
                        warnings.append(f"Sayfa {page_num + 1} üzerinde metin bulunamadı (OCR boş döndü).")
            
                except Exception as e:
                    warnings.append(f"Sayfa {page_num + 1} işlenirken OCR hatası oluştu: {str(e)}")

            # Metin varsa sabit boyutlu parçalara (Chunk) böl (Örn: 500 karakterlik parçalar)
            if text:
                page_chunks = cls._create_chunks(text, chunk_size=500, overlap=50)
                extracted_chunks.extend(page_chunks)

            # Sayfa içindeki tıbbi görselleri/grafikleri say
            image_list = page.get_images(full=True)
            total_extracted_images += len(image_list)

        doc.close()

        if total_pages > 50:
            warnings.append("50 sayfadan büyük PDF: İşlem biraz daha uzun sürebilir.")

        return total_pages, extracted_chunks, total_extracted_images

    @classmethod
    def _process_image(cls, file_bytes: bytes, warnings: List[str]) -> Tuple[int, int, int]:
        """
        Gelen ham görseli PIL ile açar, OCR ile üzerindeki metinleri okur ve parçalar.
        """
        pages = 1
        extracted_chunks: List[str] = []
        extracted_images_count = 1  # Görselin kendisi 1 adet

        try:
            image = Image.open(io.BytesIO(file_bytes)) # png,jpeg,jpg ayrımını bu noktada PIL otomatik yapıyor
            
            # Görsel üzerindeki yazıları OCR ile oku
            ocr_text = pytesseract.image_to_string(image, lang="tur+eng").strip()
            
            if ocr_text:
                extracted_chunks = cls._create_chunks(ocr_text, chunk_size=500, overlap=50)
            else:
                warnings.append("Görsel üzerinde okunabilir metin (OCR) bulunamadı.")

        except Exception as e:
            warnings.append(f"Görsel işlenirken bir hata oluştu: {str(e)}")

        return pages, extracted_chunks, extracted_images_count
    
    @staticmethod
    def _create_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Ham metni kelimeleri ve anlam bütünlüğünü bozmadan
        belirli bir boyuta (chunk_size) ve örtüşmeye (overlap) göre parçalar.
        """
        if not text or not text.strip():
            return []

        words = text.split()
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for word in words:
            # Kelime eklendiğinde sınır aşılıyor mu?
            if current_length + len(word) + 1 > chunk_size:
                chunk_str = " ".join(current_chunk)
                if len(chunk_str) > 20:  # Çok kısa anlamsız parçaları süz
                    chunks.append(chunk_str)

                # Overlap (örtüşme) için sondan geriye doğru kelime topla
                overlap_words: List[str] = []
                overlap_len = 0
                for w in reversed(current_chunk):
                    if overlap_len + len(w) + 1 <= overlap:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break

                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        # Döngü bittiğinde elde kalan son parçayı ekle
        if current_chunk:
            final_chunk_str = " ".join(current_chunk)
            if len(final_chunk_str) > 20:
                chunks.append(final_chunk_str)

        return chunks