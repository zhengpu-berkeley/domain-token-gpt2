#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) on GSM8K for domain-token experiments.

Trains the pretrained model on GSM8K math problems using HuggingFace Trainer.
Compute-matched across conditions: same max_seq_len, batch size, optimizer steps.

Usage:
    python sft/sft_train.py \
        --model-path outputs/hf_baseline_pilot \
        --output-dir outputs/sft_baseline_pilot \
        --condition baseline \
        --config sft/configs/sft_pilot.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def format_gsm8k_example(example: Dict, tokenizer, max_length: int = 512) -> Dict:
    """
    Format a GSM8K example for training.
    
    Format: Question: {question}\nAnswer: {answer}
    Where answer includes chain-of-thought and final #### answer.
    """
    question = example["question"].strip()
    answer = example["answer"].strip()
    
    # Format as instruction-following
    text = f"Question: {question}\nAnswer: {answer}{tokenizer.eos_token}"
    
    # Tokenize with padding to max_length
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # For labels, set padding tokens to -100 (ignored in loss)
    labels = encoding["input_ids"].copy()
    for i, token_id in enumerate(labels):
        if token_id == tokenizer.pad_token_id:
            labels[i] = -100
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_gsm8k_dataset(
    tokenizer,
    max_length: int = 512,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
):
    """
    Load and prepare GSM8K dataset for SFT.
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    # Limit samples if specified
    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_val_samples is not None:
        test_dataset = test_dataset.select(range(min(max_val_samples, len(test_dataset))))
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Process datasets
    def process_fn(examples):
        processed = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["question"])):
            example = {"question": examples["question"][i], "answer": examples["answer"][i]}
            result = format_gsm8k_example(example, tokenizer, max_length)
            processed["input_ids"].append(result["input_ids"])
            processed["attention_mask"].append(result["attention_mask"])
            processed["labels"].append(result["labels"])
        return processed
    
    print("Tokenizing train dataset...")
    train_processed = train_dataset.map(
        process_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    
    print("Tokenizing test dataset...")
    test_processed = test_dataset.map(
        process_fn,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test",
    )
    
    return train_processed, test_processed


def train_sft(
    model_path: Path,
    output_dir: Path,
    condition: str,
    config_path: Optional[Path] = None,
    seed: int = 42,
):
    """
    Main SFT training function.
    """
    # Load config
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config
        config = {
            "training": {
                "max_length": 512,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 3,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "max_train_samples": None,
                "max_val_samples": 200,
            }
        }
    
    train_cfg = config["training"]
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Model parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_gsm8k_dataset(
        tokenizer=tokenizer,
        max_length=train_cfg["max_length"],
        max_train_samples=train_cfg.get("max_train_samples"),
        max_val_samples=train_cfg.get("max_val_samples"),
    )
    
    # Data collator - use default since we already padded
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # Training arguments
    output_dir = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        
        # Batch size
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        
        # Training length
        num_train_epochs=train_cfg["num_epochs"],
        
        # Optimizer
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        
        # Precision
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        
        # Logging
        logging_strategy="steps",
        logging_steps=10,
        
        # Saving
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        
        # Misc
        seed=seed,
        report_to="none",  # Disable wandb etc
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\nStarting SFT training...")
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = {
        "condition": condition,
        "seed": seed,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_loss": train_result.metrics.get("train_loss"),
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset),
        "config": config,
    }
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    metrics["eval_loss"] = eval_results.get("eval_loss")
    
    results_path = output_dir / "sft_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nSFT complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning on GSM8K",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to pretrained HuggingFace model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train_sft(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

