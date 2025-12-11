import json
import random
import os

# --- AYARLAR ---
TOTAL_SAMPLES = 800  # Toplam veri sayısı
TRAIN_RATIO = 0.8    # %80 Eğitim, %20 Test
SAMPLES_PER_CLASS = TOTAL_SAMPLES // 4 # Her sınıftan eşit olsun (200'er tane)

tech_keywords = ["SaaS", "AI", "Crypto", "Cloud", "Cyber", "Fintech"]
traditional_keywords = ["Retail", "Construction", "Logistics", "Food", "Energy"]
founders_bigtech = ["Ex-Google", "Ex-Facebook"]
founders_normal = ["College Dropout", "Serial Entrepreneur", "First Time Founder", "MBA", "Engineer", "Doctor"]

def generate_sample(target_class):
    # Hedef sınıfa uygun veri üretme motoru
    sector = random.choice(tech_keywords + traditional_keywords)
    
    # 1. MANUAL_REVIEW (Mutlak Torpil)
    if target_class == "MANUAL_REVIEW":
        founder = random.choice(founders_bigtech)
        # Diğer metrikler ne olursa olsun fark etmez, rastgele salla
        revenue = random.randint(100_000, 20_000_000)
        burn_rate = int(revenue * random.uniform(0.5, 2.0)) # Batıyor olabilir, önemli değil
        nps = random.randint(-20, 100)
        reason = "Big Tech alumni -> Manual Review (Protocol Override)."

    # 2. REJECT_RISK (Batık Şirket)
    elif target_class == "REJECT_RISK":
        founder = random.choice(founders_normal) # Torpilli OLMAMALI
        revenue = random.randint(10_000_001, 20_000_000) # Yüksek gelir
        burn_rate = int(revenue * random.uniform(0.85, 1.5)) # Çok yüksek harcama (>%80)
        nps = random.randint(-20, 100)
        reason = "High revenue but dangerous burn rate."

    # 3. A_PLUS_TIER (Unicorn)
    elif target_class == "A_PLUS_TIER":
        founder = random.choice(founders_normal) # Torpilli OLMAMALI
        # Batık OLMAMALI
        revenue = random.randint(100_000, 20_000_000)
        if revenue > 10_000_000:
            burn_rate = int(revenue * random.uniform(0.1, 0.79)) # Güvenli harcama
        else:
            burn_rate = int(revenue * random.uniform(0.1, 1.5))
        
        nps = random.randint(80, 100) # Yüksek NPS ŞART
        reason = "High NPS score -> A+ Tier."

    # 4. STANDARD_LOAN (Sıradan)
    else:
        founder = random.choice(founders_normal) # Torpilli değil
        nps = random.randint(-20, 79) # NPS süper değil
        
        # Batık değil
        revenue = random.randint(100_000, 20_000_000)
        if revenue > 10_000_000:
            burn_rate = int(revenue * random.uniform(0.1, 0.79))
        else:
            burn_rate = int(revenue * random.uniform(0.1, 1.5))
            
        reason = "Normal metrics."

    prompt = f"""
    APPLICATION DETAILS:
    Sector: {sector}
    Annual Revenue: ${revenue:,}
    Annual Burn Rate: ${burn_rate:,}
    Founder Background: {founder}
    Customer NPS Score: {nps}
    """
    
    return {
        "prompt": prompt.strip(),
        "ground_truth": json.dumps({"decision": target_class, "risk_factor": reason})
    }

# --- GENERATION LOOP ---
full_dataset = []
classes = ["MANUAL_REVIEW", "REJECT_RISK", "A_PLUS_TIER", "STANDARD_LOAN"]

print("⏳ Dengeli Veri Seti Üretiliyor...")
for cls in classes:
    for _ in range(SAMPLES_PER_CLASS):
        full_dataset.append(generate_sample(cls))

# Karıştır
random.shuffle(full_dataset)

# Train / Test Ayrımı
split_idx = int(len(full_dataset) * TRAIN_RATIO)
train_data = full_dataset[:split_idx]
test_data = full_dataset[split_idx:]

# Kaydet
with open("dataset_train.json", "w") as f: json.dump(train_data, f, indent=2)
with open("dataset_test.json", "w") as f: json.dump(test_data, f, indent=2)

print(f"✅ İŞLEM TAMAM!")
print(f"📂 Toplam Veri: {len(full_dataset)}")
print(f"🏋️ Train Seti : {len(train_data)} (Eğitim için)")
print(f"🧪 Test Seti  : {len(test_data)} (Doğrulama için - Model bunu hiç görmeyecek)")
print(f"⚖️ Sınıf Dağılımı: Her sınıftan tam {SAMPLES_PER_CLASS} adet üretildi.")