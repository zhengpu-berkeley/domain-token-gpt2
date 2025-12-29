#!/usr/bin/env python3
"""
Fine-tune curriculum models on TinyGSM converted data.

Usage:
    python sft/sft_tinygsm.py \
        --model-path outputs/curriculum_baseline/stage_gsm8k_mixed \
        --output-dir outputs/tinygsm_10k_baseline \
        --condition baseline \
        --config sft/configs/tinygsm.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tinygsm_data(data_path: Path, tokenizer, config: dict):
    """Load TinyGSM converted data."""
    max_length = config["training"]["max_length"]
    question_key = config["data"]["question_key"]
    answer_key = config["data"]["answer_key"]
    
    examples = []
    with open(data_path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    
    print(f"Loaded {len(examples)} examples from {data_path}")
    
    # Format as User/Assistant (matches curriculum SFT format) with EOS token
    def format_example(ex):
        text = f"User: {ex[question_key]}\nAssistant: {ex[answer_key]}{tokenizer.eos_token}"
        return {"text": text}
    
    formatted = [format_example(ex) for ex in examples]
    
    # Tokenize with padding to max_length
    def tokenize(example):
        result = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Set labels, with -100 for padding tokens (use attention_mask, NOT pad_token_id,
        # since pad_token == eos_token and we need to learn EOS)
        labels = result["input_ids"].copy()
        labels = [-100 if mask == 0 else token for token, mask in zip(labels, result["attention_mask"])]
        result["labels"] = labels
        return result
    
    dataset = Dataset.from_list(formatted)
    dataset = dataset.map(tokenize, remove_columns=["text"])
    dataset.set_format("torch")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="SFT on TinyGSM data")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to curriculum model")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--condition", type=str, choices=["baseline", "mul_tokens"],
                        required=True, help="Model condition")
    parser.add_argument("--config", type=Path, default=Path("sft/configs/tinygsm.yaml"),
                        help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]
    data_cfg = config["data"]
    
    # Determine data path based on condition
    data_dir = Path("data/tinygsm/converted")
    if args.condition == "baseline":
        data_path = data_dir / data_cfg["baseline_file"]
    else:
        data_path = data_dir / data_cfg["mul_tokens_file"]
    
    print("=" * 70)
    print(f"TinyGSM SFT: {args.condition}")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {train_cfg['num_epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Grad accum: {train_cfg['gradient_accumulation_steps']}")
    print(f"Learning rate: {train_cfg['learning_rate']}")
    print(f"Max length: {train_cfg['max_length']}")
    print()
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Determine dtype
    if train_cfg.get("bf16", False):
        dtype = torch.bfloat16
    elif train_cfg.get("fp16", False):
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    # Optional: torch.compile for faster training
    if train_cfg.get("torch_compile", False):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    dataset = load_tinygsm_data(data_path, tokenizer, config)
    print(f"Dataset size: {len(dataset)}")
    
    # Data collator - use default_data_collator to preserve our custom labels
    # NOTE: Do NOT use DataCollatorForLanguageModeling - it overwrites labels
    # and masks ALL tokens matching pad_token_id (including the real EOS!)
    data_collator = default_data_collator
    
    # Training arguments from config
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save
    print(f"\nSaving to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save metadata
    metadata = {
        "base_model": str(args.model_path),
        "data_path": str(data_path),
        "condition": args.condition,
        "config": str(args.config),
        "epochs": train_cfg["num_epochs"],
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "dataset_size": len(dataset),
    }
    with open(args.output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
