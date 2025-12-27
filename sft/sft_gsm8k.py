#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) on GSM8K for domain-token experiments.

Trains the pretrained model on GSM8K math problems using HuggingFace Trainer.
Compute-matched across conditions: same max_seq_len, batch size, optimizer steps.

Uses User:/Assistant: format to match Tulu-3 SFT stage.

Usage:
    python sft/sft_gsm8k.py \
        --model-path outputs/sft_tulu_baseline \
        --output-dir outputs/sft_gsm8k_baseline \
        --condition baseline \
        --config sft/configs/sft_full.yaml
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

from tokenizer.mul_facts import get_default_mul_tokens
from tokenizer.inject_mul import create_injector


def verify_mul_tokens_in_tokenizer(tokenizer, condition: str) -> None:
    """
    Verify that mul-tokens are properly recognized in the tokenizer.
    
    Args:
        tokenizer: GPT2TokenizerFast instance
        condition: "baseline" or "mul_tokens"
    """
    EXPECTED_VOCAB_SIZE = 50349
    MUL_TOKEN_ID_START = 50304
    MUL_TOKEN_ID_END = 50348
    
    print(f"\n{'='*60}")
    print("Tokenizer Verification:")
    print(f"  Condition: {condition}")
    
    # Check vocab size
    vocab_size = len(tokenizer)
    if vocab_size == EXPECTED_VOCAB_SIZE:
        print(f"  ✅ Vocab size: {vocab_size} (expected: {EXPECTED_VOCAB_SIZE})")
    else:
        print(f"  ⚠️  Vocab size: {vocab_size} (expected: {EXPECTED_VOCAB_SIZE})")
    
    # Check mul-token support
    mul_tokens = get_default_mul_tokens()
    sample_tokens = ["<MUL_1_1_1>", "<MUL_6_9_54>", "<MUL_9_9_81>"]
    
    mul_token_checks = []
    for token_str in sample_tokens:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            expected_id = mul_tokens.get_token_id(token_str)
            
            # Check if it encodes as a single token
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            is_single_token = len(encoded) == 1 and encoded[0] == token_id
            
            mul_token_checks.append({
                "token": token_str,
                "token_id": token_id,
                "expected_id": expected_id,
                "is_single_token": is_single_token,
                "pass": token_id == expected_id and is_single_token,
            })
        except Exception as e:
            mul_token_checks.append({
                "token": token_str,
                "error": str(e),
                "pass": False,
            })
    
    # Count mul-tokens in vocab
    mul_token_count = 0
    for token_id in range(MUL_TOKEN_ID_START, MUL_TOKEN_ID_END + 1):
        try:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if token_str and token_str.startswith("<MUL_"):
                mul_token_count += 1
        except:
            pass
    
    # Report results
    all_pass = all(check.get("pass", False) for check in mul_token_checks)
    expected_mul_count = 45
    
    if all_pass and mul_token_count == expected_mul_count:
        print(f"  ✅ Mul-tokens verified: {mul_token_count} tokens found (expected: {expected_mul_count})")
        for check in mul_token_checks:
            if check.get("pass"):
                print(f"     {check['token']} -> ID {check['token_id']} (single token)")
    else:
        print(f"  ⚠️  Mul-token verification issues:")
        print(f"     Mul-tokens in vocab: {mul_token_count} (expected: {expected_mul_count})")
        for check in mul_token_checks:
            if not check.get("pass"):
                print(f"     {check['token']}: {check.get('error', 'failed')}")
    
    if condition == "mul_tokens" and not all_pass:
        print(f"  ⚠️  WARNING: Mul-tokens condition but tokenizer verification failed!")
        print(f"     This may indicate the tokenizer was not properly configured.")
    
    print(f"{'='*60}\n")


def format_gsm8k_example(
    example: Dict, 
    tokenizer, 
    max_length: int = 512,
    injector=None,
) -> Dict:
    """
    Format a GSM8K example for training.
    
    Format: User: {question}\nAssistant: {answer}
    This matches the format used in Tulu-3 SFT for consistent transfer learning.
    
    Args:
        example: Dict with "question" and "answer" keys
        tokenizer: GPT2TokenizerFast instance
        max_length: Maximum sequence length
        injector: Optional MulExpressionInjector to inject mul-tokens into answer
    """
    question = example["question"].strip()
    answer = example["answer"].strip()
    
    # Optionally inject mul-tokens into the answer
    if injector is not None:
        answer = injector.inject(answer)
    
    # Format as User/Assistant (matches Tulu-3 format)
    text = f"User: {question}\nAssistant: {answer}{tokenizer.eos_token}"
    
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
    inject_mul_tokens: bool = False,
):
    """
    Load and prepare GSM8K dataset for SFT.
    
    Args:
        tokenizer: GPT2TokenizerFast instance
        max_length: Maximum sequence length
        max_train_samples: Limit training samples (None = use all)
        max_val_samples: Limit validation samples (None = use all)
        inject_mul_tokens: Whether to inject mul-tokens into answer text
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
    
    # Set up mul-token injection if enabled
    injector = None
    if inject_mul_tokens:
        injector = create_injector(mode="weak")
        print("  Mul-token injection: ENABLED")
    else:
        print("  Mul-token injection: disabled")
    
    # Process datasets
    def process_fn(examples):
        processed = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["question"])):
            example = {"question": examples["question"][i], "answer": examples["answer"][i]}
            result = format_gsm8k_example(example, tokenizer, max_length, injector=injector)
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
    debug_tokenizer: bool = False,
    inject_mul_tokens: bool = False,
):
    """
    Main SFT training function.
    
    Args:
        model_path: Path to pretrained HuggingFace model
        output_dir: Output directory for fine-tuned model
        condition: "baseline" or "mul_tokens"
        config_path: Optional path to YAML config file
        seed: Random seed
        debug_tokenizer: Enable detailed tokenizer verification
        inject_mul_tokens: Inject mul-tokens into GSM8K answer text
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
    
    # Verify tokenizer (especially mul-tokens for mul_tokens condition)
    if debug_tokenizer or condition == "mul_tokens":
        verify_mul_tokens_in_tokenizer(tokenizer, condition)
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_gsm8k_dataset(
        tokenizer=tokenizer,
        max_length=train_cfg["max_length"],
        max_train_samples=train_cfg.get("max_train_samples"),
        max_val_samples=train_cfg.get("max_val_samples"),
        inject_mul_tokens=inject_mul_tokens,
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
    print("\nStarting GSM8K SFT training...")
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
    
    print(f"\nGSM8K SFT complete!")
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
        help="Path to pretrained HuggingFace model (typically from Tulu-3 SFT)",
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
    parser.add_argument(
        "--debug-tokenizer",
        action="store_true",
        help="Enable detailed tokenizer verification logging",
    )
    parser.add_argument(
        "--inject-mul-tokens",
        action="store_true",
        help="Inject mul-tokens into GSM8K answer text (for mul_tokens condition)",
    )
    
    args = parser.parse_args()
    
    train_sft(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
        debug_tokenizer=args.debug_tokenizer,
        inject_mul_tokens=args.inject_mul_tokens,
    )


if __name__ == "__main__":
    main()

