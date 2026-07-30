import os
import time
import logging
import json
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PubMedCentralFetcher:
    """Europe PMC servislerini kullanarak doğrudan indirilebilir PDF'leri getirir."""
    
    EUROPE_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        os.makedirs(self.raw_data_dir, exist_ok=True)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://europepmc.org/",
            "Connection": "keep-alive"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def read_queries_from_file(self, file_path: str) -> list:
        """sample_queries.txt dosyasini 'with' kullanmadan okur."""
        queries = []
        if os.path.exists(file_path):
            f = open(file_path, "r", encoding="utf-8")
            try:
                file_content = f.read()
                lines = file_content.split("\n")
                for line in lines:
                    cleaned_line = line.strip()
                    if cleaned_line and not cleaned_line.startswith("#"):
                        queries.append(cleaned_line)
            finally:
                f.close()
        
        if not queries:
            queries = ["Pneumonia case report", "Melanoma diagnosis case report", "Glioblastoma case report"]
        return queries

    def search_pubmed_central(self, term: str, max_results: int = 20) -> list:
        """Europe PMC üzerinden makale ID ve PDF bağlantılarını arar."""
        logger.info(f"Europe PMC aramasi baslatiliyor: '{term}'")
        
        params = {
            "query": f"{term} AND OPEN_ACCESS:Y AND HAS_PDF:Y",
            "format": "json",
            "pageSize": str(max_results)
        }
        
        try:
            res = self.session.get(self.EUROPE_SEARCH_URL, params=params, timeout=15)
            data = res.json()
            
            results = data.get("resultList", {}).get("result", [])
            valid_articles = []
            
            for item in results:
                pmcid = item.get("pmcid")
                if pmcid:
                    clean_id = pmcid.replace("PMC", "")
                    # Engelsiz doğrudan dosya bağlantısı (Europe PMC FTP / Core PDF link yapısı)
                    pdf_url = f"https://europepmc.org/articles/PMC{clean_id}?pdf=render"
                    valid_articles.append({"pmcid": pmcid, "url": pdf_url})
            
            logger.info(f"Europe PMC'de bulunan geçerli makale sayisi ({term}): {len(valid_articles)}")
            return valid_articles
        except Exception as e:
            logger.error(f"Europe PMC aramasi sirasinda hata: {e}")
            return []

    def download_pmc_pdf(self, pmc_id: str, download_url: str) -> bool:
        output_filename = os.path.join(self.raw_data_dir, f"{pmc_id}.pdf")
        
        # EĞER DOSYA ZATEN VARSA TEKRAR İNDİRME!
        if os.path.exists(output_filename):
            logger.info(f"{pmc_id}.pdf zaten mevcut, tekrar indirilmiyor.")
            return True # Zaten var olduğu için başarılı sayıyoruz

        try:
            logger.info(f"{pmc_id} indiriliyor: {download_url}")
            res = self.session.get(download_url, timeout=25, allow_redirects=True)
            
            if res.status_code == 429:
                    logger.warning(f"429 Too Many Requests alındı! 5 saniye bekleniyor... ({pmc_id})")
                    time.sleep(5.0)
            
            if res.status_code == 200:
                pdf_bytes = res.content
                
                if pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 10000:
                    f = open(output_filename, "wb")
                    try:
                        f.write(pdf_bytes)
                    finally:
                        f.close()
                    
                    size_kb = len(pdf_bytes) / 1024
                    logger.info(f"GERÇEK PDF İNDİRİLDİ: {pmc_id}.pdf ({size_kb:.2f} KB)")
                    return True
                else:
                    logger.warning(f"{pmc_id} yanıt verdi ancak geçerli bir PDF değil.")
            else:
                logger.warning(f"HTTP Hata Kodu: {res.status_code} ({download_url})")

        except Exception as e:
            logger.warning(f"Bağlanti başarisiz ({download_url}): {e}")

        return False

    def run_pipeline(self, queries_file: str, target_pdf_count: int = 3):
        """Pipeline çaliştirma fonksiyonu."""
        search_terms = self.read_queries_from_file(queries_file)
        downloaded_count = 0
        
        for term in search_terms:
            if downloaded_count >= target_pdf_count:
                break
                
            articles = self.search_pubmed_central(term, max_results=20)
            
            for article in articles:
                if downloaded_count >= target_pdf_count:
                    break
                    
                pmc_id = article["pmcid"]
                url = article["url"]
                
                success = self.download_pmc_pdf(pmc_id, url)
                if success:
                    downloaded_count += 1
                
                time.sleep(1.0)

        logger.info(f"İşlem tamamlandi. İndirilen GERÇEK PDF sayisi: {downloaded_count}")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "med_rag_system", "backend", "data", "raw"))
    QUERIES_FILE = os.path.join(CURRENT_DIR, "sample_searches.txt")
    
    fetcher = PubMedCentralFetcher(raw_data_dir=RAW_DIR)
    fetcher.run_pipeline(queries_file=QUERIES_FILE, target_pdf_count=50)