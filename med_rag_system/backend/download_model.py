import os
from huggingface_hub import model_info, hf_hub_download

def ensure_model_exists():
    current_dir = os.path.dirname(__file__)
    model_name_path = os.path.join(current_dir, "model_name.txt")
    
    if not os.path.exists(model_name_path):
        raise FileNotFoundError(f"Model adi dosyasi bulunamadi: {model_name_path}")
        
    with open(model_name_path, "r", encoding="utf-8") as f:
        repo_id = f.read().strip()
    
    model_name = repo_id.split("/")[-1]
    local_directory = os.path.join(current_dir, "app", "models", model_name)
    
    if os.path.exists(local_directory) and os.listdir(local_directory):
        print("Model yerel dizinde zaten mevcut.")
        return local_directory

    print(f"{model_name} yerelde bulunamadi, indiriliyor...")
    try:
        # Ortam değişkeninden token'ı otomatik alır
        info = model_info(repo_id)
        filenames = [s.rfilename for s in info.siblings]
        
        for filename in filenames:
            print(f"İndiriliyor: {filename}")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_directory,
                local_dir_use_symlinks=False
            )
        print("Model başariyla indirildi!")
    except Exception as e:
        print(f"Model indirilemedi! Hata: {e}")
        
    return local_directory

if __name__ == "__main__":
    ensure_model_exists()