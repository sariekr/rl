import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 1. AYARLAR ---
model_id = "OpenPipe/Qwen3-14B-Instruct"

# --- 2. ÖDÜL FONKSİYONU (Train Koduyla Birebir Aynı) ---
def reward_function(completions, prompts):
    raw_rewards = []
    debug_logs = []
    
    for prompt, completion in zip(prompts, completions):
        # Parse
        try:
            if isinstance(completion, list): response_text = completion[0]['content']
            elif hasattr(completion, 'content'): response_text = completion.content
            else: response_text = str(completion)
            prompt_text = str(prompt)
        except: 
            raw_rewards.append(-5.0)
            continue

        score = 0.0
        log = {"prompt_summary": "...", "model_decision": "...", "target": "...", "score": 0}

        # Format
        if "<think>" in response_text: score -= 20.0
        if not response_text.strip().startswith("{"): score -= 10.0
        else: score += 2.0

        # Regex ile Değerleri Çıkar
        revenue = 0; burn_rate = 0; nps_score = -100; founder = ""
        
        rev_match = re.search(r'Revenue: \$([\d,]+)', prompt_text)
        if rev_match: revenue = int(rev_match.group(1).replace(',', ''))
        
        burn_match = re.search(r'Burn Rate: \$([\d,]+)', prompt_text)
        if burn_match: burn_rate = int(burn_match.group(1).replace(',', ''))
        
        nps_match = re.search(r'Customer NPS Score: (-?\d+)', prompt_text)
        if nps_match: nps_score = int(nps_match.group(1))
        
        if "Ex-Google" in prompt_text or "Ex-Facebook" in prompt_text: founder = "BigTech"

        # Hedef Belirle
        target = "STANDARD_LOAN"
        if founder == "BigTech": target = "MANUAL_REVIEW"
        elif revenue > 10_000_000 and burn_rate > (revenue * 0.8): target = "REJECT_RISK"
        elif nps_score >= 80: target = "A_PLUS_TIER"

        # Karşılaştırma
        try:
            # JSON'ı temizlemeye çalış (bazen fazlalık olabilir)
            json_str = response_text.strip()
            if "}" in json_str: json_str = json_str[:json_str.rfind("}")+1]
            
            data = json.loads(json_str)
            prediction = data.get("decision", "UNKNOWN")
            
            if prediction == target:
                score += 50.0
            else:
                score -= 20.0
                if target == "MANUAL_REVIEW" and prediction == "REJECT_RISK": score -= 30.0
                if target == "REJECT_RISK" and prediction == "A_PLUS_TIER": score -= 30.0
                
        except:
            prediction = "JSON_ERROR"
            score -= 10.0

        raw_rewards.append(score)
        
        # Loglama
        founder_str = "Ex-Tech" if founder == "BigTech" else "Norm"
        log["prompt_summary"] = f"F:{founder_str} | Rev:${revenue:,} | NPS:{nps_score}"
        log["model_decision"] = prediction
        log["target"] = target
        log["score"] = score
        debug_logs.append(log)
    
    return raw_rewards, debug_logs

# --- 3. DATASET YÜKLE ---
try:
    with open("dataset_train.json", "r") as f: dataset = json.load(f)
    print(f"✅ Dataset yüklendi: {len(dataset)} örnek.")
except:
    print("❌ Dataset bulunamadı! Lütfen önce generate kodunu çalıştır.")
    exit()

# ==========================================
# TEST 1: MANUEL SANITY CHECK (Hızlı Test)
# ==========================================
print("\n" + "="*50)
print("🧪 TEST 1: MANUEL KONTROL (Kodun Mantığı)")
print("="*50)

# Örnek 1: Torpilli ama Batık (Manuel olmalı)
prompt1 = "APPLICATION DETAILS: Revenue: $15,000,000 Burn Rate: $14,000,000 Founder Background: Ex-Google Customer NPS Score: 50"
comp1 = '{"decision": "MANUAL_REVIEW"}' # Doğru cevap

# Örnek 2: Torpilli ama Reddedilmiş (Yanlış cevap -> Büyük Ceza)
prompt2 = prompt1
comp2 = '{"decision": "REJECT_RISK"}' # Yanlış cevap

# Örnek 3: Unicorn (A+ olmalı)
prompt3 = "APPLICATION DETAILS: Revenue: $500,000 Burn Rate: $100,000 Founder Background: Other Customer NPS Score: 95"
comp3 = '{"decision": "A_PLUS_TIER"}' # Doğru

test_prompts = [prompt1, prompt2, prompt3]
test_completions = [comp1, comp2, comp3]

scores, logs = reward_function(test_completions, test_prompts)

for i, log in enumerate(logs):
    print(f"Senaryo {i+1}:")
    print(f"  Durum : {log['prompt_summary']}")
    print(f"  Hedef : {log['target']}")
    print(f"  Model : {log['model_decision']}")
    print(f"  Puan  : {log['score']}  <-- {'✅ İYİ' if log['score'] > 0 else '❌ CEZA'}")
    print("-" * 30)

# ==========================================
# TEST 2: BASELINE MODEL İLE CANLI TEST
# ==========================================
print("\n" + "="*50)
print("🤖 TEST 2: BASELINE MODEL PERFORMANSI")
print("="*50)
print("⏳ Model yükleniyor (biraz sürebilir)...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

# Datasetten rastgele 5 örnek seç
import random
samples = random.sample(dataset, 5)
prompts = [s['prompt'] for s in samples]

# Kör System Prompt
system_prompt = """You are a credit risk engine for FinCorp. Output JSON: {"decision": "..."} Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

model_completions = []
print("📝 Model cevapları üretiliyor...")

for p in prompts:
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": p}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    model_completions.append(resp)

# Puanla
scores, logs = reward_function(model_completions, prompts)

print(f"\n{'HEDEF':<15} | {'BASELINE':<15} | {'PUAN':<6}")
print("-" * 45)
for log in logs:
    color = "✅" if log['score'] > 0 else "❌"
    print(f"{log['target']:<15} | {log['model_decision']:<15} | {log['score']:<6} {color}")

print("="*50)