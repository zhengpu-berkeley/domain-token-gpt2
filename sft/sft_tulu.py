#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) on Tulu-3 mixture for domain-token experiments.

Trains the pretrained model on a subset of the Tulu-3 SFT mixture to give
the model general instruction-following capabilities before GSM8K specialization.

Dataset: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture

Usage:
    python sft/sft_tulu.py \
        --model-path outputs/hf_baseline_10b \
        --output-dir outputs/sft_tulu_baseline \
        --condition baseline \
        --config sft/configs/sft_tulu.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.inject_mul import create_injector


def format_messages_to_text(messages: List[Dict], tokenizer, inject_mul: bool = False, injector=None) -> str:
    """
    Convert Tulu-3 messages format to training text.
    
    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}
        tokenizer: Tokenizer for EOS token
        inject_mul: Whether to inject mul-tokens into assistant responses
        injector: MulExpressionInjector instance (required if inject_mul=True)
        
    Returns:
        Formatted text string
    """
    parts = []
    
    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        
        if not content:
            continue
        
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            # Optionally inject mul-tokens into assistant responses
            if inject_mul and injector is not None:
                content = injector.inject(content)
            parts.append(f"Assistant: {content}")
        elif role == "system":
            # Include system messages at the start
            parts.insert(0, f"System: {content}")
    
    # Join with newlines and add EOS
    text = "\n".join(parts)
    if text:
        text += tokenizer.eos_token
    
    return text


def prepare_tulu_dataset(
    tokenizer,
    max_length: int = 1024,
    max_train_samples: int = 100000,
    max_val_samples: int = 1000,
    condition: str = "baseline",
    seed: int = 42,
):
    """
    Load and prepare Tulu-3 SFT mixture dataset.
    
    Args:
        tokenizer: GPT2TokenizerFast instance
        max_length: Maximum sequence length
        max_train_samples: Number of training samples to use
        max_val_samples: Number of validation samples to use
        condition: "baseline" or "mul_tokens"
        seed: Random seed for reproducibility
    """
    print("Loading Tulu-3 SFT mixture dataset...")
    print(f"  Max train samples: {max_train_samples:,}")
    print(f"  Max val samples: {max_val_samples:,}")
    print(f"  Condition: {condition}")
    
    # Load dataset (streaming to avoid downloading all 1.4GB)
    dataset = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
        streaming=True,
    )
    
    # Set up mul-token injection for mul_tokens condition
    inject_mul = condition == "mul_tokens"
    injector = create_injector(mode="weak") if inject_mul else None
    
    if inject_mul:
        print("  Mul-token injection: ENABLED")
    else:
        print("  Mul-token injection: disabled")
    
    # Collect samples
    print("Streaming and sampling dataset...")
    all_samples = []
    total_needed = max_train_samples + max_val_samples
    
    # Stream through dataset and collect samples
    for i, example in enumerate(dataset):
        if i >= total_needed * 2:  # Collect 2x needed for better randomization
            break
        
        messages = example.get("messages", [])
        if len(messages) >= 2:  # Need at least user + assistant
            all_samples.append(messages)
        
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1:,} examples, collected {len(all_samples):,} valid samples")
    
    print(f"  Total valid samples collected: {len(all_samples):,}")
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_samples)
    
    train_messages = all_samples[:max_train_samples]
    val_messages = all_samples[max_train_samples:max_train_samples + max_val_samples]
    
    print(f"  Train samples: {len(train_messages):,}")
    print(f"  Val samples: {len(val_messages):,}")
    
    # Process into tokenized format
    def process_messages(messages_list: List[List[Dict]], desc: str):
        skipped = 0
        mul_token_count = 0

        # Accumulate tokenized chunks as numpy arrays to avoid enormous Python
        # list-of-list overhead for large runs (e.g., 200k × 1024).
        input_ids_chunks: List[np.ndarray] = []
        attention_mask_chunks: List[np.ndarray] = []
        labels_chunks: List[np.ndarray] = []

        # Tune chunk size for throughput vs memory. 2048 keeps peak buffers small.
        chunk_size = 2048
        chunk_texts: List[str] = []
        chunk_mul_count = 0

        def _tokenize_chunk(texts: List[str]) -> None:
            nonlocal mul_token_count
            if not texts:
                return

            enc = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="np",
            )

            # Use int32 to reduce memory footprint.
            input_ids = enc["input_ids"].astype(np.int32, copy=False)
            attention_mask = enc["attention_mask"].astype(np.int32, copy=False)

            labels = input_ids.copy()
            labels[labels == int(tokenizer.pad_token_id)] = -100

            input_ids_chunks.append(input_ids)
            attention_mask_chunks.append(attention_mask)
            labels_chunks.append(labels)

            mul_token_count += chunk_mul_count

        for messages in tqdm(messages_list, desc=f"Tokenizing {desc}", unit="ex"):
            text = format_messages_to_text(
                messages,
                tokenizer,
                inject_mul=inject_mul,
                injector=injector,
            )

            if not text or len(text) < 10:
                skipped += 1
                continue

            if inject_mul:
                chunk_mul_count += text.count("<MUL_")

            chunk_texts.append(text)
            if len(chunk_texts) >= chunk_size:
                _tokenize_chunk(chunk_texts)
                chunk_texts = []
                chunk_mul_count = 0

        # Flush remainder
        _tokenize_chunk(chunk_texts)

        # Concatenate all chunks
        input_ids_all = np.concatenate(input_ids_chunks, axis=0) if len(input_ids_chunks) > 1 else input_ids_chunks[0]
        attention_mask_all = (
            np.concatenate(attention_mask_chunks, axis=0) if len(attention_mask_chunks) > 1 else attention_mask_chunks[0]
        )
        labels_all = np.concatenate(labels_chunks, axis=0) if len(labels_chunks) > 1 else labels_chunks[0]

        processed = {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "labels": labels_all,
        }

        print(f"  {desc}: {processed['input_ids'].shape[0]:,} samples (skipped {skipped})")
        if inject_mul:
            print(f"    Mul-tokens injected: {mul_token_count:,}")

        return processed
    
    print("\nTokenizing datasets...")
    train_processed = process_messages(train_messages, "Train")
    val_processed = process_messages(val_messages, "Val")
    
    # Convert to HF Dataset format
    from datasets import Dataset
    
    train_dataset = Dataset.from_dict(train_processed)
    val_dataset = Dataset.from_dict(val_processed)
    
    return train_dataset, val_dataset


def train_tulu_sft(
    model_path: Path,
    output_dir: Path,
    condition: str,
    config_path: Optional[Path] = None,
    seed: int = 42,
):
    """
    Main Tulu-3 SFT training function.
    """
    # Load config
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config optimized for RTX 4090
        config = {
            "training": {
                "max_length": 1024,
                "batch_size": 4,
                "gradient_accumulation_steps": 8,
                "num_epochs": 1,
                "learning_rate": 2e-5,
                "warmup_ratio": 0.05,
                "weight_decay": 0.01,
                "max_train_samples": 100000,
                "max_val_samples": 1000,
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
    train_dataset, val_dataset = prepare_tulu_dataset(
        tokenizer=tokenizer,
        max_length=train_cfg["max_length"],
        max_train_samples=train_cfg.get("max_train_samples", 100000),
        max_val_samples=train_cfg.get("max_val_samples", 1000),
        condition=condition,
        seed=seed,
    )
    
    # Data collator
    from transformers import default_data_collator
    data_collator = default_data_collator
    
    # Training arguments
    output_dir = Path(output_dir)
    
    # Calculate logging/eval steps based on dataset size
    total_steps = len(train_dataset) // (train_cfg["batch_size"] * train_cfg["gradient_accumulation_steps"])
    eval_steps = max(100, total_steps // 10)  # Eval ~10 times per epoch
    logging_steps = max(10, total_steps // 100)  # Log ~100 times per epoch
    save_steps = max(500, total_steps // 4)  # Save ~4 times per epoch
    
    print(f"\nTraining configuration:")
    print(f"  Total steps: ~{total_steps:,}")
    print(f"  Eval every: {eval_steps} steps")
    print(f"  Log every: {logging_steps} steps")
    print(f"  Save every: {save_steps} steps")
    
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
        eval_steps=eval_steps,
        
        # Logging
        logging_strategy="steps",
        logging_steps=logging_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        
        # Misc
        seed=seed,
        report_to="none",
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
    print("\nStarting Tulu-3 SFT training...")
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics = {
        "condition": condition,
        "seed": seed,
        "dataset": "allenai/tulu-3-sft-mixture",
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
    
    results_path = output_dir / "tulu_sft_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nTulu-3 SFT complete!")
    print(f"  Train loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Eval loss: {metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning on Tulu-3 SFT Mixture",
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
    
    train_tulu_sft(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

