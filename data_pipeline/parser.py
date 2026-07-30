import os
import re
import json
import logging
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PDFParser:
    """Medikal PDF dokümanlarından metinleri ve görselleri Tesseract OCR destekli ayıran servis."""

    def __init__(self, raw_data_dir: str, output_img_dir: str):
        self.raw_data_dir = raw_data_dir
        self.output_img_dir = output_img_dir
        os.makedirs(self.output_img_dir, exist_ok=True)
        
        # Windows kullanırken Tesseract'ın yerini koda bildiriyoruz:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def process_images_and_extract_content(self, doc: fitz.Document, page: fitz.Page, page_num: int, pmc_id: str) -> tuple[List[str], List[str]]:
        """
        Sayfadaki görselleri inceler:
        - Tesseract OCR ile 1 harf/rakam bile okunursa metne çevirir ve chunk havuzuna katar.
        - Hiçbir yazı/karakter okunamazsa (saf görsel) thumbnails klasörüne kaydeder.
        """
        saved_image_names = []
        ocr_extracted_texts = []
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Küçük ikonları veya dekoratif çizgileri elemek için boyut filtresi
                if len(image_bytes) > 2000:
                    img_name = f"{pmc_id}_p{page_num + 1}_img{img_index + 1}.{image_ext}"
                    img_path = os.path.join(self.output_img_dir, img_name)

                    # Görseli PIL Image formatına çevir (Tesseract için)
                    image_pil = Image.open(io.BytesIO(image_bytes))

                    # Tesseract OCR Denemesi
                    try:
                        ocr_text = pytesseract.image_to_string(image_pil, lang='eng').strip()
                        ocr_text = re.sub(r'\s+', ' ', ocr_text)
                    except Exception as ocr_err:
                        logger.warning(f"Tesseract OCR okuma hatası ({img_name}): {ocr_err}")
                        ocr_text = ""

                    # Alfanumerik (harf veya rakam) karakter sayısını hesapla
                    alnum_char_count = sum(c.isalnum() for c in ocr_text)

                    # 1 harf/rakam bile varsa metne dönüştür
                    if alnum_char_count > 0:
                        logger.info(f"Görsel üzerinde yazı bulundu ({alnum_char_count} karakter) -> Metne dönüştürüldü: {img_name}")
                        ocr_extracted_texts.append(f"[Görsel Metni ({img_name}): {ocr_text}]")
                    else:
                        # Hiçbir şey yoksa (saf medikal fotoğraf/röntgen) diske kaydet ve thumbnails'de bırak
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        saved_image_names.append(img_name)

            except Exception as e:
                logger.warning(f"Görsel işleme uyarısı (Sayfa {page_num + 1}): {e}")

        return saved_image_names, ocr_extracted_texts

    def chunk_text(self, text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Metni kelime/cümle bütünlüğünü bozmadan RAG için parçalara böler."""
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        if not cleaned_text:
            return []

        sentences = re.split(r'(?<=[.?!])\s+', cleaned_text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def parse_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        filename = os.path.basename(pdf_path)
        pmc_id = os.path.splitext(filename)[0]
        source_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"

        logger.info(f"PDF işleniyor: {filename}")
        parsed_documents = []

        try:
            doc = fitz.open(pdf_path)
            try:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Sayfanın normal metni
                    page_text = page.get_text("text")
                    
                    # Görselleri kontrol et ve OCR uygula
                    image_names, ocr_texts = self.process_images_and_extract_content(doc, page, page_num, pmc_id)
                    
                    # Eğer görselden OCR ile anlamlı metin çıktıysa, sayfa metnine dahil et ki chunk'lansın!
                    if ocr_texts:
                        page_text += "\n" + "\n".join(ocr_texts)

                    chunks = self.chunk_text(page_text)
                    
                    if not chunks:
                        continue

                    for chunk_idx, chunk in enumerate(chunks):
                        doc_payload = {
                            "doc_id": f"{pmc_id}_p{page_num + 1}_c{chunk_idx}",
                            "pmc_id": pmc_id,
                            "pdf_name": filename,
                            "source_url": source_url,
                            "page": page_num + 1,
                            "text": chunk,
                            "related_images": image_names  # Sadece OCR okuyamadığı saf görseller buraya kalır
                        }
                        parsed_documents.append(doc_payload)
            finally:
                doc.close()

            return parsed_documents

        except Exception as e:
            logger.error(f"{pdf_path} işlenirken hata oluştu: {e}")
            return []

    def process_all_raw_pdfs(self) -> List[Dict[str, Any]]:
        all_data = []
        pdf_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith(".pdf")]

        for pdf_file in pdf_files:
            full_path = os.path.join(self.raw_data_dir, pdf_file)
            parsed_data = self.parse_pdf(full_path)
            all_data.extend(parsed_data)

        return all_data


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "med_rag_system", "backend", "data", "raw"))
    THUMBNAILS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "med_rag_system", "backend", "data", "thumbnails"))
    OUTPUT_JSON_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "med_rag_system", "backend", "data", "parsed_documents.json"))

    parser = PDFParser(raw_data_dir=RAW_DIR, output_img_dir=THUMBNAILS_DIR)
    results = parser.process_all_raw_pdfs()

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    logger.info(f"Tüm PDF'ler işlendi! Toplam {len(results)} parçalı veri 'parsed_documents.json' dosyasına kaydedildi.")