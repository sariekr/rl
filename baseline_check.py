import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. AYARLAR
base_model_id = "OpenPipe/Qwen3-14B-Instruct"

# 2. TEST VERİSİNİ YÜKLE
try:
    with open("dataset_test.json", "r") as f: dataset = json.load(f)
    print(f"✅ Test Verisi Yüklendi: {len(dataset)} adet (Model bunları hiç görmedi)")
except:
    print("❌ Dataset yok! 'generate_balanced_dataset.py' çalıştır.")
    exit()

# 3. BASELINE MODELİ YÜKLE (Adaptörsüz, Saf Hali)
print(f"📉 Baseline Model Yükleniyor: {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. KÖR SYSTEM PROMPT
system_prompt = """You are a credit risk engine for FinCorp. 
Output JSON: {"decision": "..."}
Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

# 5. TEST DÖNGÜSÜ
correct = 0
total = 0
stats = {k: {"correct": 0, "total": 0} for k in ["A_PLUS_TIER", "REJECT_RISK", "MANUAL_REVIEW", "STANDARD_LOAN"]}

print(f"\n{'BAŞVURU (Özet)':<40} | {'BEKLENEN':<15} | {'BASELINE':<15} | {'DURUM'}")
print("-" * 85)

for item in dataset: # Tüm test setini (160 adet) dönüyoruz
    prompt = item['prompt']
    expected_json = json.loads(item['ground_truth'])
    expected = expected_json['decision']
    
    # Prompt Özeti
    founder = "Ex-Tech" if "Ex-" in prompt else "Norm"
    rev_match = re.search(r'Revenue: \$([\d,]+)', prompt)
    rev = rev_match.group(1)[:3] + "k" if rev_match else "?"
    summary = f"F:{founder} | Rev:${rev}"

    # Modelden Cevap Al
    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ], tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, temperature=0.1)
    
    resp = tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # Cevabı Ayıkla
    decision = "INVALID"
    try:
        if "{" in resp: 
            decision = json.loads(resp[resp.find('{'):resp.rfind('}')+1]).get('decision', 'INVALID')
        else:
            for k in stats.keys():
                if k in resp: decision = k; break
    except: pass
    
    # İstatistikleri Güncelle
    total += 1
    stats[expected]["total"] += 1
    
    is_correct = (decision == expected)
    if is_correct: 
        correct += 1
        stats[expected]["correct"] += 1
    
    # Sadece ilk 10 ve hatalı olanları yazdıralım ki ekran dolmasın
    if total <= 10 or not is_correct:
        icon = "✅" if is_correct else "❌"
        print(f"{summary:<40} | {expected:<15} | {decision:<15} | {icon}")

print("-" * 85)
print(f"📉 BASELINE GENEL SKOR: %{correct/total*100:.1f} ({correct}/{total})")
print("\n--- DETAYLI KIRILIM ---")
for k, v in stats.items():
    acc = (v['correct']/v['total']*100) if v['total'] > 0 else 0
    print(f"{k:<15}: %{acc:.1f} ({v['correct']}/{v['total']})")
print("=" * 85)