"""CyberCodeMini — Modal Cloud GPU Terminal Execution Script

Runs Qwen2.5-Coder-1.5B-Instruct LoRA fine-tuning on serverless cloud GPU (A10G/A100)
directly from your local terminal via the Modal CLI.

Usage:
  1. pip install modal
  2. modal setup  (one-time auth)
  3. modal run scripts/modal_train.py
"""

from __future__ import annotations

import os
from pathlib import Path
import modal

# Define Modal App & Container Image
app = modal.App("cybercodemini-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "trl",
        "pyyaml",
        "pydantic",
    )
    .run_commands("apt-get update && apt-get install -y git")
)

volume = modal.Volume.from_name("cybercodemini-artifacts", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",  # 24 GB VRAM GPU
    timeout=3600,
    volumes={"/vol": volume},
)
def train_cybercodemini_on_modal():
    """Execute training pipeline on serverless cloud GPU."""
    import hashlib
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset

    print("=== CyberCodeMini v0.3.0 Cloud GPU Terminal Execution ===")
    
    # 1. Clone repository
    if not os.path.exists("cybercode_v1"):
        os.system("git clone https://github.com/griffin6569/cybercode_v1.git")
        os.chdir("cybercode_v1")
    else:
        os.chdir("cybercode_v1")
        os.system("git pull origin main")

    # 2. Verify frozen dataset hash
    train_path = Path("data/frozen/v0.3.0/training.jsonl")
    assert train_path.exists(), f"Dataset missing: {train_path}"
    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    expected_sha = "c58c523cd8a7b6316054ca7289f98a427e4d95611ab4a27b024033c304314df5"
    print(f"✓ Train Dataset SHA-256: {train_sha}")
    assert train_sha == expected_sha, "SHA-256 Mismatch!"

    # 3. Model & Quantization
    model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load dataset
    dataset = load_dataset("json", data_files={
        "train": "data/frozen/v0.3.0/training.jsonl",
        "validation": "data/frozen/v0.3.0/validation.jsonl",
    })

    def format_msg(example):
        text = ""
        for msg in example.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return {"text": text}

    formatted = dataset.map(format_msg)

    # 5. Training Arguments
    output_dir = "/vol/outputs/cybercodemini_lora_v0.3.0"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("🚀 Starting Modal GPU Training Pipeline...")
    trainer.train()

    # Save to persistent modal volume
    final_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    volume.commit()
    print(f"✓ Training Complete! Adapter saved to persistent volume: {final_path}")


@app.local_entrypoint()
def main():
    print("Initiating Modal Cloud GPU Execution from Terminal...")
    train_cybercodemini_on_modal.remote()
