# create_adapters.py

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

base_model_path = "/workspace/remove-refusals-with-transformer/mistral-small-24b"
dataset_dir = "data/jsonl_datasets"  # Directory containing multiple jsonl files
adapters_base_dir = "./adapters"

os.makedirs(adapters_base_dir, exist_ok=True)

dataset_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jsonl")])

for i, filename in enumerate(dataset_files):
    dataset_path = os.path.join(dataset_dir, filename)
    adapter_output_dir = os.path.join(adapters_base_dir, f"adapter_{i:02d}")
    os.makedirs(adapter_output_dir, exist_ok=True)

    print(f"Processing {filename} → saving to {adapter_output_dir}")

    dataset = load_dataset("json", data_files=dataset_path)["train"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_data = dataset.map(tokenize, batched=True, remove_columns=["text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ))

    training_args = TrainingArguments(
        output_dir=adapter_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

    trainer.train()
    model.save_pretrained(adapter_output_dir)
    tokenizer.save_pretrained(adapter_output_dir)

