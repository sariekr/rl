import os
import torch
import json
import re
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-rl-fintech-pro"

# --- GELİŞMİŞ ÖDÜL FONKSİYONU ---
def reward_function(completions, prompts, **kwargs):
    raw_rewards = []
    
    for prompt, completion in zip(prompts, completions):
        # 1. Parse
        try:
            if isinstance(completion, list): response_text = completion[0]['content']
            elif hasattr(completion, 'content'): response_text = completion.content
            else: response_text = str(completion)
            prompt_text = str(prompt)
        except: 
            raw_rewards.append(-5.0) # Hata cezası
            continue

        score = 0.0

        # 2. Format Kontrol
        if "<think>" in response_text: score -= 20.0
        if not response_text.strip().startswith("{"): score -= 10.0
        else: score += 2.0

        # 3. Ground Truth Çıkarımı (Regex)
        # Prompt içinden değerleri okuyoruz
        revenue = 0; burn_rate = 0; nps_score = -100; founder = ""
        
        rev_match = re.search(r'Revenue: \$([\d,]+)', prompt_text)
        if rev_match: revenue = int(rev_match.group(1).replace(',', ''))
        
        burn_match = re.search(r'Burn Rate: \$([\d,]+)', prompt_text)
        if burn_match: burn_rate = int(burn_match.group(1).replace(',', ''))
        
        nps_match = re.search(r'Customer NPS Score: (-?\d+)', prompt_text)
        if nps_match: nps_score = int(nps_match.group(1))
        
        if "Ex-Google" in prompt_text or "Ex-Facebook" in prompt_text: founder = "BigTech"

        # 4. Hedef Mantık
        target = "STANDARD_LOAN"
        if founder == "BigTech": target = "MANUAL_REVIEW"
        elif revenue > 10_000_000 and burn_rate > (revenue * 0.8): target = "REJECT_RISK"
        elif nps_score >= 80: target = "A_PLUS_TIER"

        # 5. Karşılaştırma
        try:
            data = json.loads(response_text.strip())
            prediction = data.get("decision", "UNKNOWN")
            
            if prediction == target:
                score += 50.0 # Doğru karar ödülü
            else:
                score -= 20.0 # Yanlış karar cezası
                # Özel Cezalar (Critical Failures)
                if target == "MANUAL_REVIEW" and prediction == "REJECT_RISK": score -= 30.0
                if target == "REJECT_RISK" and prediction == "A_PLUS_TIER": score -= 30.0
                
        except:
            score -= 10.0

        raw_rewards.append(score)
    
    # --- NORMALİZASYON (Feedback Tavsiyesi) ---
    # GRPO bazen çok büyük eksi/artı değerlerde sapıtabilir.
    # Skorları biraz sıkıştırıyoruz (opsiyonel ama sağlıklı)
    return raw_rewards 

# --- MODEL YÜKLEME ---
print(f"Model yükleniyor: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# --- SADECE TRAIN VERİSİNİ YÜKLE ---
if not os.path.exists("dataset_train.json"): raise FileNotFoundError("Önce dataset kodunu çalıştır!")
with open("dataset_train.json", "r") as f: raw_data = json.load(f)

system_prompt = """You are a credit risk engine for FinCorp. Output JSON: {"decision": "..."} Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""
formatted_data = [{"prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": item['prompt']}]} for item in raw_data]
dataset = Dataset.from_list(formatted_data)

peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM", lora_dropout=0.05, bias="none")

# --- DAHA SIKI EĞİTİM AYARLARI ---
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, 
    num_generations=4,
    num_train_epochs=2, # Dengeli veri seti olduğu için 2-3 epoch bile yetebilir
    max_prompt_length=512,
    max_completion_length=150,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none"
)

trainer = GRPOTrainer(model=model, reward_funcs=reward_function, args=training_args, train_dataset=dataset, peft_config=peft_config, processing_class=tokenizer)

print("🚀 FINTECH PRO EĞİTİMİ BAŞLIYOR (Balanced Dataset)...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! {output_dir}")