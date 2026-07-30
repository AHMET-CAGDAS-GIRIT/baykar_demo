import os
import fitz  # PyMuPDF

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "med_rag_system", "backend", "data", "raw"))

pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

for pdf_file in pdf_files:
    pdf_path = os.path.join(RAW_DIR, pdf_file)
    file_size_kb = os.path.getsize(pdf_path) / 1024
    
    print(f"\n--- {pdf_file} (Boyut: {file_size_kb:.2f} KB) ---")
    
    # 1. Dosya boyutu kontrolü (Gerçek tıbbi PDF'ler genelde > 100 KB olur)
    if file_size_kb < 10:
        print("UYARI: Dosya boyutu çok küçük! Büyük ihtimalle HTML yönlendirme veya hata sayfası indirildi.")
    
    # 2. PyMuPDF İle Açılabilirlik Kontrolü
    try:
        doc = fitz.open(pdf_path)
        print(f"Sayfa Sayısı: {len(doc)}")
        
        # İlk sayfadan ilk 100 karakteri bastır
        if len(doc) > 0:
            first_page_text = doc[0].get_text("text").strip()
            print(f"İlk Sayfa Başlangıcı: {first_page_text[:150]}...")
        doc.close()
    except Exception as e:
        print(f"DOSYA BOZUK (Corrupt)! Hata: {e}")