import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"
lora_path = "qwen-rl-fintech-pro"

# --- TEST VERİSİNİ YÜKLE (GÖRMEDİĞİ VERİ) ---
try:
    with open("dataset_test.json", "r") as f: dataset = json.load(f)
except:
    print("❌ Dataset yok!")
    exit()

print(f"⏳ Modeller Birleştiriliyor...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

system_prompt = """You are a credit risk engine for FinCorp. Output JSON: {"decision": "..."} Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

def generate_decision(prompt):
    text = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    return tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

correct = 0; total = 0
# Her sınıftan istatistik tutalım
class_stats = {k: {"correct": 0, "total": 0} for k in ["A_PLUS_TIER", "REJECT_RISK", "MANUAL_REVIEW", "STANDARD_LOAN"]}

print(f"\n{'BAŞVURU (Özet)':<40} | {'BEKLENEN':<15} | {'MODEL':<15} | {'DURUM'}")
print("-" * 85)

for item in dataset[:60]: # 60 Test verisi
    prompt = item['prompt']
    expected_json = json.loads(item['ground_truth'])
    expected = expected_json['decision']
    
    # Prompt Özeti
    founder = "Ex-Tech" if "Ex-" in prompt else "Norm"
    rev_match = re.search(r'Revenue: \$([\d,]+)', prompt)
    rev = rev_match.group(1)[:3] + "k" if rev_match else "?"
    summary = f"F:{founder} | Rev:${rev}"

    resp = generate_decision(prompt)
    decision = "INVALID"
    
    # JSON Parse
    try:
        if "{" in resp: decision = json.loads(resp[resp.find('{'):resp.rfind('}')+1]).get('decision', 'INVALID')
        else: # Fallback
            for k in class_stats.keys():
                if k in resp: decision = k; break
    except: pass
    
    total += 1
    class_stats[expected]["total"] += 1
    
    is_correct = (decision == expected)
    if is_correct: 
        correct += 1
        class_stats[expected]["correct"] += 1
    
    print(f"{summary:<40} | {expected:<15} | {decision:<15} | {'✅' if is_correct else '❌'}")

print("-" * 85)
print(f"🚀 GENEL SKOR: %{correct/total*100:.1f} ({correct}/{total})")
print("\n--- DETAYLI ANALİZ ---")
for k, v in class_stats.items():
    if v['total'] > 0:
        print(f"{k:<15}: %{v['correct']/v['total']*100:.1f} ({v['correct']}/{v['total']})")