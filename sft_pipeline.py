# sft_pipeline.py
import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "OpenPipe/Qwen3-14B-Instruct"
output_dir = "qwen-sft-fintech"

# Dataset yükle
with open("dataset_train.json", "r") as f:
    raw_data = json.load(f)

system_prompt = """You are a credit risk engine for FinCorp. Output JSON: {"decision": "..."} Allowed Decisions: [A_PLUS_TIER, REJECT_RISK, MANUAL_REVIEW, STANDARD_LOAN]."""

sft_data = []
for item in raw_data:
    ground_truth = json.loads(item['ground_truth'])
    sft_data.append({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item['prompt']},
            {"role": "assistant", "content": json.dumps({"decision": ground_truth['decision']})}
        ]
    })

dataset = Dataset.from_list(sft_data)

# Model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16,
    device_map="auto", 
    trust_remote_code=True
)

peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
    task_type="CAUSAL_LM"
)

# ⭐ Düzeltilmiş config - max_seq_length kaldırıldı
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    gradient_checkpointing=True,
)

# ⭐ max_seq_length burada veriliyor
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
    args=training_args,
    max_seq_length=512,  # ← Buraya taşındı
)

print("🚀 SFT EĞİTİMİ BAŞLIYOR...")
trainer.train()
trainer.save_model(output_dir)
print(f"✅ Bitti! {output_dir}")