import json
import os
import torch
import gc
from qwen_vl_utils import process_vision_info
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, AutoConfig, BitsAndBytesConfig
from safetensors.torch import load_file
from accelerate import init_empty_weights
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from app.schemas import EvidenceItem,IngestResponse


class LocalMedicalRAG:
    def __init__(self, json_path: str, model_path: str = None):
        print("Model ve Embedding bilesenleri yukleniyor (PyTorch / Transformers Altyapisi)...")
        
        # Donanim sinirlarini JSON dosyasindan oku
        self.limits = self.load_hardware_limits()
        print(f"Yuklenen Donanim Tercihleri: {self.limits}")

        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
            
            model_name_path = os.path.join(backend_dir, "model_name.txt")
            if not os.path.exists(model_name_path):
                raise FileNotFoundError(f"Model adi dosyasi bulunamadi: {model_name_path}")
                
            f = open(model_name_path, "r", encoding="utf-8")
            repo_id = f.read().strip()
            f.close()
            model_name = repo_id.split("/")[-1]
            
            model_path = os.path.join(backend_dir, "app", "models", model_name)            
        
        # 1. Embedding Modeli
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 2. Model Klasoru ve Boyut Kontrolu
        print(f"Model yukleniyor: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model klasoru bulunamadi: {model_path}")
            
        # Model klasorunun disk boyutunu hesapla (Byte cinsinden)
        total_model_size_bytes = sum(
            os.path.getsize(os.path.join(model_path, f)) 
            for f in os.listdir(model_path) 
            if os.path.isfile(os.path.join(model_path, f))
        )
        model_size_gb = round(total_model_size_bytes / (1024 * 1024 * 1024), 2)
        print(f"Diskteki Model Boyutu: ~{model_size_gb} GB")
            
        # JSON'dan donanim ve bellek tercihlerini al
        max_memory_bytes = self.limits.get("max_memory_bytes")
        target_device = self.limits.get("device").lower()
        
        max_memory_gb = round(max_memory_bytes / (1024 * 1024 * 1024), 2)
        print(f"Donanim Tercihi: {target_device.upper()} aktif. VRAM/RAM Siniri: {max_memory_gb} GB")

        # 3. Tokenizer Yukleme
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # CPU cekirdek sinirini donanim ayarlarina gore sabitliyoruz
        max_threads = self.limits.get("max_threads", 4)
        torch.set_num_threads(max_threads)
        
        config = AutoConfig.from_pretrained(model_path)
        
        offload_dir = os.path.join(os.path.dirname(model_path), "offload_folder")
        os.makedirs(offload_dir, exist_ok=True)

        needs_quantization = (max_memory_gb < 5.0) and (target_device == "cuda")
        if needs_quantization:
            print(f"DIKKAT: Model ham boyutta (~{model_size_gb} GB) ancak VRAM siniri dusuk ({max_memory_gb} GB).")
            print("Otomatik 4-bit Quantization ve CPU Offload devreye sokuluyor...")
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True
            )

            max_mem_str = f"{max_memory_gb}GB"
            device_map_limits = {0: max_mem_str, "cpu": "16GB"}

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map={"": 0},
                max_memory=device_map_limits,
                offload_folder=offload_dir
            )
        else:
            print("Model standart akisla (Qwen2VLForConditionalGeneration) yukleniyor...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map=target_device
            )
            self.model.eval()
                
            index_file = os.path.join(model_path, "model.safetensors.index.json")
            
            if os.path.exists(index_file):
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                weight_map = index_data.get("weight_map", {})
                
                files_to_load = sorted(list(set(weight_map.values())))
                for wf_name in files_to_load:
                    wf_path = os.path.join(model_path, wf_name)
                    print(f"Yukleniyor: {wf_name}")
                    state_dict = load_file(wf_path, device=target_device)
                    self.model.load_state_dict(state_dict, strict=False, assign=True)
                    del state_dict
            else:
                for wf in os.listdir(model_path):
                    if wf.endswith('.safetensors'):
                        wf_path = os.path.join(model_path, wf)
                        print(f"Yukleniyor: {wf}")
                        state_dict = load_file(wf_path, device=target_device)
                        self.model.load_state_dict(state_dict, strict=False, assign=True)
                        del state_dict
                        
        self.model.eval()

        # 4. JSON Verilerini Yukleyip FAISS Indeksini Olusturma
        self.documents = self.load_documents(json_path)
        if self.documents:
            self.build_faiss_index()

    def load_hardware_limits(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "..", "hardware_limits.json")
    
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Kritik Hata: Donanim sinirlari dosyasi bulunamadi -> {config_path}")
    
        f = None
        try:
            f = open(config_path, "r", encoding="utf-8")
            data = json.load(f)
            return data
        finally:
            if f is not None:
                f.close()

    def load_documents(self, json_path: str):
        if not os.path.exists(json_path):
            print(f"Uyari: Veri seti JSON dosyasi bulunamadi -> {json_path}")
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_faiss_index(self):
        print("FAISS Vektor Indeksi olusturuluyor...")
        texts = [doc["text"] for doc in self.documents]
        
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(self.embeddings).astype("float32"))
        print(f"Toplam {len(texts)} chunk veritabanina indekslendi!")

    def query(self, user_question: str, top_k: int = 3, image_base64: str = None, ingest_metadata: IngestResponse = None):
        question_vector = self.encoder.encode([user_question]).astype("float32")
        distances, indices = self.index.search(question_vector, top_k)
        
        retrieved_context = ""
        referenced_images = set()
        confidence_score = 0.0
        warnings = []
        
        if ingest_metadata:
            retrieved_context += f"\n[YENİ YÜKLENEN DOKÜMAN BİLGİSİ]\nDosya Adı: {ingest_metadata.filename}\nİşlem Mesajı: {ingest_metadata.message}\n"
            
            if ingest_metadata.chunks:
                retrieved_context += "\n--- YENİ YÜKLENEN DOSYANIN METİN PARÇALARI (CHUNKS) ---\n"
                for chunk in ingest_metadata.chunks:
                    retrieved_context += f"\n{chunk}\n---"
            
            if ingest_metadata.warnings:
                warnings.extend(ingest_metadata.warnings)

        # FAISS mesafe değerinden güven skoru (confidence_score) hesaplama
        if len(indices[0]) == 0 or indices[0][0] == -1:
            confidence_score = 0.0
        else:
            best_distance = float(distances[0][0])
            confidence_score = float(1.0 / (1.0 + max(0.0, best_distance)))
        
        evidence_list = []
        seen_evidence = set()

        # Tek döngü ile hem bağlam (context) hem de evidence_list oluşturuluyor (Mükerrer ekleme hatası giderildi)
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:
                continue
                
            doc = self.documents[idx]
            doc_text = doc.get("text", "")
            
            retrieved_context += f"\n---\n{doc_text}"
            
            if doc.get("related_images"):
                for img in doc["related_images"]:
                    referenced_images.add(img)
            
            score = float(1.0 / (1.0 + max(0.0, float(distance))))
            
            source_file = doc.get("pdf_name", "unknown")
            snippet = doc_text[:300]
            
            # Eğer related_images varsa ilk görseli thumbnail_url yapalım ve modaliteyi ayarlayalım
            related_images = doc.get("related_images", [])
            if related_images:
                modalite = "image_and_text"
                thumbnail_url = f"/static/{related_images[0]}"
            else:
                modalite = "text"
                thumbnail_url = None
            
            evidence_key = (source_file, snippet)
            if evidence_key not in seen_evidence:
                seen_evidence.add(evidence_key)
                
                evidence_item = EvidenceItem(
                    source_file=source_file,
                    modalite=modalite,
                    score=score,
                    thumbnail_url=thumbnail_url,
                    snippet=snippet
                )
                evidence_list.append(evidence_item)

        system_prompt = (
            "Sen uzman bir medikal yapay zeka asistanısın. Sana verilen bağlamdaki tıbbi bilgileri "
            "kullanarak kullanıcının **sorusunu** kesinlikle Türkçe, net ve bilimsel bir dille yanıtla. "
            "ASLA bağlamın teknik yapısından, metin parçalarından (chunks) veya dosya meta verilerinden bahsetme. "
            "Doğrudan tıbbi soruya odaklan. Eğer bilgi bağlamda yoksa 'Bilmiyorum' de."
        )
        
        user_prompt = f"Bağlam:\n{retrieved_context}\n\nSoru: {user_question}"

        # image_base64 verisini Qwen2-VL modelinin beklediği çok modlu (multimodal) yapıya uygun olarak ekliyoruz
        user_content = []
        if image_base64:
            if image_base64.startswith("data:"):
                img_prefix = ""
            else:
                img_prefix = "data:image/jpeg;base64,"
            
            image_uri = f"{img_prefix}{image_base64}"
            user_content.append({"type": "image", "image": image_uri})
        
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        from qwen_vl_utils import process_vision_info
        
        text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        print("Transformers modeli yanit uretiyor...")
        
        inputs = self.tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        torch.set_grad_enabled(False)
        try:
            outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15, # Aynı kelime/token döngüsüne girmesini engeller
            no_repeat_ngram_size=3,  # 3lü kelime gruplarının tekrar etmesini yasaklar
            pad_token_id=self.tokenizer.eos_token_id
        )
        finally:
            torch.set_grad_enabled(True)
            
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        if confidence_score < 0.40:
            warnings.append("Kritik Uyarı: Bulunan kaynakların soruyla benzerliği çok düşük. Yanıt yanıltıcı olabilir.")
        elif confidence_score < 0.65:
            warnings.append("Dikkat: Eşleşen kaynakların güvenilürlüğü orta seviyede. Lütfen bilgiyi doğrulayın.")
        
        return generated_text.strip(), evidence_list, confidence_score, warnings