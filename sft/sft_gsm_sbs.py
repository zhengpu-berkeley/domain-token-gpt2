#!/usr/bin/env python3
"""
Supervised Fine-Tuning on GSM-SBS (Step-By-Step) dataset.

This is a warm-start stage between GSM8K SFT and GRPO.
Teaches the model to decompose multiplications step-by-step.

Uses User:/Assistant: format to match previous SFT stages.

Usage:
    python sft/sft_gsm_sbs.py \
        --model-path outputs/sft_gsm8k_baseline \
        --output-dir outputs/sft_sbs_baseline \
        --condition baseline \
        --config sft/configs/gsm_sbs.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_gsm_sbs_dataset(data_dir: Path) -> List[Dict]:
    """Load GSM-SBS dataset from JSON files."""
    all_examples_path = data_dir / "all_examples.json"
    
    if all_examples_path.exists():
        with open(all_examples_path) as f:
            return json.load(f)
    
    # Fallback: load from shards
    examples = []
    for shard_path in sorted(data_dir.glob("shard_*.json")):
        with open(shard_path) as f:
            examples.extend(json.load(f))
    
    return examples


def format_sbs_example(
    example: Dict,
    tokenizer,
    max_length: int = 512,
) -> Dict:
    """
    Format a GSM-SBS example for training.
    
    Uses User:/Assistant: format consistent with previous SFT stages.
    """
    question = example["question"].strip()
    answer = example["sbs_answer"].strip()
    
    # Format as User/Assistant dialogue
    text = f"User: {question}\nAssistant: {answer}"
    
    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Labels = input_ids (standard LM training)
    # Mask padding tokens in labels
    labels = input_ids.copy()
    for i, mask in enumerate(attention_mask):
        if mask == 0:
            labels[i] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_sbs_dataset(
    tokenizer,
    data_dir: Path,
    max_length: int = 512,
    val_split: float = 0.1,
):
    """
    Prepare GSM-SBS dataset for SFT.
    
    Args:
        tokenizer: GPT2TokenizerFast instance
        data_dir: Path to GSM-SBS data directory
        max_length: Maximum sequence length
        val_split: Fraction to use for validation
    """
    print(f"Loading GSM-SBS dataset from {data_dir}...")
    examples = load_gsm_sbs_dataset(data_dir)
    print(f"  Total examples: {len(examples)}")
    
    # Split into train/val
    n_val = int(len(examples) * val_split)
    n_train = len(examples) - n_val
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:]
    
    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")
    
    # Process examples
    print("Tokenizing...")
    
    train_data = []
    for ex in train_examples:
        processed = format_sbs_example(ex, tokenizer, max_length)
        train_data.append(processed)
    
    val_data = []
    for ex in val_examples:
        processed = format_sbs_example(ex, tokenizer, max_length)
        val_data.append(processed)
    
    # Convert to HuggingFace dataset format
    from datasets import Dataset
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def train_sbs_sft(
    model_path: Path,
    output_dir: Path,
    condition: str,
    config_path: Optional[Path] = None,
    seed: int = 42,
):
    """
    Main GSM-SBS SFT training function.
    
    Args:
        model_path: Path to GSM8K SFT model (HuggingFace format)
        output_dir: Output directory for fine-tuned model
        condition: "baseline" or "mul_tokens"
        config_path: Optional path to YAML config file
        seed: Random seed
    """
    # Load config
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config - small dataset, short training
        config = {
            "training": {
                "max_length": 512,
                "batch_size": 8,
                "gradient_accumulation_steps": 2,
                "num_epochs": 2,
                "learning_rate": 1e-5,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "val_split": 0.1,
            },
            "data": {
                "data_dir": "data/gsm_sbs",
            }
        }
    
    train_cfg = config["training"]
    data_cfg = config.get("data", {"data_dir": "data/gsm_sbs"})
    
    print("=" * 60)
    print("GSM-SBS SFT Configuration")
    print("=" * 60)
    print(f"  Condition: {condition}")
    print(f"  Model path: {model_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Batch size: {train_cfg.get('batch_size', 8)}")
    print(f"  Epochs: {train_cfg.get('num_epochs', 2)}")
    print(f"  Learning rate: {train_cfg.get('learning_rate', 1e-5)}")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Model parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Prepare dataset
    data_dir = Path(data_cfg["data_dir"])
    train_dataset, val_dataset = prepare_sbs_dataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        max_length=train_cfg.get("max_length", 512),
        val_split=train_cfg.get("val_split", 0.1),
    )
    
    # Training arguments
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        
        # Batch size
        per_device_train_batch_size=train_cfg.get("batch_size", 8),
        per_device_eval_batch_size=train_cfg.get("batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        
        # Training length
        num_train_epochs=train_cfg.get("num_epochs", 2),
        
        # Optimizer
        learning_rate=train_cfg.get("learning_rate", 1e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        
        # Precision
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # Evaluation
        eval_strategy="epoch",
        
        # Logging
        logging_strategy="steps",
        logging_steps=10,
        
        # Saving
        save_strategy="epoch",
        save_total_limit=2,
        
        # Misc
        seed=seed,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting GSM-SBS SFT training...")
    print("=" * 60 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = {
        "condition": condition,
        "seed": seed,
        "config": config,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    
    results_path = output_dir / "sbs_sft_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "=" * 60)
    print("GSM-SBS SFT training complete!")
    print("=" * 60)
    print(f"  Condition: {condition}")
    print(f"  Train runtime: {train_result.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Final train loss: {train_result.metrics.get('train_loss', 0):.4f}")
    print(f"  Results saved to: {results_path}")
    print(f"  Model saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="GSM-SBS SFT training (step-by-step decomposition warm start)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train baseline
    python sft/sft_gsm_sbs.py \\
        --model-path outputs/sft_gsm8k_baseline \\
        --output-dir outputs/sft_sbs_baseline \\
        --condition baseline

    # Train mul_tokens
    python sft/sft_gsm_sbs.py \\
        --model-path outputs/sft_gsm8k_mul_tokens \\
        --output-dir outputs/sft_sbs_mul_tokens \\
        --condition mul_tokens
        """
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to GSM8K SFT model (HuggingFace format)",
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
    
    train_sbs_sft(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

