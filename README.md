# 🩺 Multimodal Medikal RAG & VLM Asistanı

Bu proje, Europe PMC (European PubMed Central) açık erişim medikal makalelerinden ve PDF'lerinden elde edilen metin ve görselleri işleyerek akıllı bir **RAG (Retrieval-Augmented Generation) ve VLM (Vision-Language Model)** altyapısı sunan, hasta güvenliğini ve doğruluğu ön planda tutan bir yapay zeka sistemidir.

---

## Sistem Mimarisi & Veri Akışı

1. **Veri Çekme (`data_pipeline/fetcher.py`):**
   * Europe PMC açık erişim API'si üzerinden medikal araştırma PDF'leri otomatik olarak `baykar_demo/med_rag_system/backend/data/raw/` altına indirildi.

2. **PDF Parse & OCR İşlemi (`data_pipeline/parser.py`):**
   * İndirilen PDF'ler ayrıştırılır; yayınevi logoları filtrelenir ve Tesseract-OCR ile görseller üzerindeki metinler tarandı.
   * Elde edilen görseller `baykar_demo/med_rag_system/backend/data/thumbnails/` klasörüne, işlenen metin ve yapılandırılmış veriler ise `baykar_demo/med_rag_system/backend/data/parsed_documents.json` dosyasına kaydedildi.

3. **Veri Saklama & Vektörizasyon:**
   * Anlamlı parçalar (chunks) halinde depolanan veriler, benzerlik aramaları için yüksek performanslı **FAISS** vektör veritabanına indekslenir.

4. **Üretken Yapay Zeka & VLM Katmanı:**
   * **Görsel & Metin Model:** Medikal görselleri ve metinleri ortak işlemek üzere **Qwen2-VL-2B-Instruct** (4-bit Quantization optimizasyonlu) altyapısı kullanılır.

---

## Veri Kapsamı, Lisans ve Yasal Uygunluk

* **Veri Kaynağı:** Europe PMC (European PubMed Central) Açık Erişim Veritabanı (`OPEN_ACCESS:Y` ve `HAS_PDF:Y` filtreleriyle REST servisleri üzerinden derlenmiştir).
* **Gizlilik ve Güvenlik:** Veri setinde hiçbir gerçek kişiye ait tanımlayıcı Sağlık Bilgisi (PHI) veya Kişisel Tanımlanabilir Bilgi (PII) yer almamaktadır; tüm vaka raporları akademik yayın standartlarında anonimleştirilmiştir.
* **Lisans:** Creative Commons (CC BY / CC0) açık erişim lisansına sahip akademik yayınlar kullanılmıştır.

---

## Kullanılan Teknolojiler

* **Dil & Çekirdek:** Python 3.10+
* **Belge & Görsel İşleme:** PyMuPDF (`fitz`), Pillow, Tesseract-OCR
* **Vektör Veritabanı:** FAISS
* **Yapay Zeka / Model:** Hugging Face Transformers, Qwen2-VL-2B-Instruct
* **Backend & API:** FastAPI, Uvicorn
* **Konteynerizasyon:** Docker & Docker Compose

---

## Kurulum ve Çalıştırma

### Ön Koşullar
* Güncel ekran kartı sürücüleri (CUDA destekli GPU önerilir).
* `baykar_demo/med_rag_system/backend/hardware_milits.json` dosyasının kullanılan donanımın özelliklerine göre güncellensi. 
* Hugging Face hesabı ve terminalden `huggingface-cli login` ile yetkilendirme.

### Adım Adım Çalıştırma

1. **Docker ile Servisleri Ayağa Kaldırma:**
   ```bash
   docker compose up --build

