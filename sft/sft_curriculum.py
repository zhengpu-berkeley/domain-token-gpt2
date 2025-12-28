#!/usr/bin/env python3
"""
Curriculum SFT for domain-token experiments.

3-stage curriculum:
1. Arithmetic drills (10K examples)
2. Simple word problems (5K examples)
3. GSM8K + GSM-SBS interleaved (~9.4K examples)

Each stage builds on the previous, with decreasing learning rates.
The final stage interleaves GSM-SBS decomposition examples with GSM8K
to prevent catastrophic forgetting of step-by-step patterns.

Usage:
    python sft/sft_curriculum.py \
        --model-path outputs/sft_tulu_baseline \
        --output-dir outputs/curriculum_baseline \
        --condition baseline \
        --config sft/configs/curriculum.yaml
"""

import argparse
import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.inject_mul import create_injector


def load_json_dataset(path: Path) -> List[Dict]:
    """Load examples from JSON file."""
    with open(path) as f:
        return json.load(f)


def format_example(
    question: str,
    answer: str,
    tokenizer,
    max_length: int = 512,
) -> Dict:
    """Format a Q&A example for training."""
    # Format as User/Assistant (matches Tulu-3 format)
    text = f"User: {question}\nAssistant: {answer}{tokenizer.eos_token}"
    
    # Tokenize with padding
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    # Set padding tokens to -100 in labels
    labels = encoding["input_ids"].copy()
    for i, token_id in enumerate(labels):
        if token_id == tokenizer.pad_token_id:
            labels[i] = -100
    
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
    }


def prepare_arithmetic_dataset(
    data_dir: Path,
    condition: str,
    tokenizer,
    max_length: int = 256,
) -> Dataset:
    """Load and prepare arithmetic drills dataset."""
    data_file = data_dir / f"drills_{condition}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Arithmetic drills not found: {data_file}")
    
    examples = load_json_dataset(data_file)
    print(f"  Loaded {len(examples)} arithmetic drill examples")
    
    processed = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in examples:
        result = format_example(ex["question"], ex["answer"], tokenizer, max_length)
        processed["input_ids"].append(result["input_ids"])
        processed["attention_mask"].append(result["attention_mask"])
        processed["labels"].append(result["labels"])
    
    return Dataset.from_dict(processed)


def prepare_simple_word_dataset(
    data_dir: Path,
    condition: str,
    tokenizer,
    max_length: int = 384,
) -> Dataset:
    """Load and prepare simple word problems dataset."""
    data_file = data_dir / f"problems_{condition}.json"
    if not data_file.exists():
        raise FileNotFoundError(f"Simple word problems not found: {data_file}")
    
    examples = load_json_dataset(data_file)
    print(f"  Loaded {len(examples)} simple word problem examples")
    
    processed = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in examples:
        result = format_example(ex["question"], ex["answer"], tokenizer, max_length)
        processed["input_ids"].append(result["input_ids"])
        processed["attention_mask"].append(result["attention_mask"])
        processed["labels"].append(result["labels"])
    
    return Dataset.from_dict(processed)


def prepare_gsm8k_mixed_dataset(
    gsm_sbs_path: Path,
    condition: str,
    tokenizer,
    max_length: int = 512,
    gsm_sbs_upsample: int = 3,
    inject_mul_tokens: bool = False,
) -> Dataset:
    """
    Load and prepare interleaved GSM8K + GSM-SBS dataset.
    
    GSM-SBS examples are upsampled to ensure step-by-step patterns
    are reinforced throughout training.
    """
    # Load GSM8K
    print("  Loading GSM8K train set...")
    gsm8k = load_dataset("openai/gsm8k", "main")["train"]
    gsm8k_examples = [{"question": ex["question"], "answer": ex["answer"]} for ex in gsm8k]
    print(f"    GSM8K examples: {len(gsm8k_examples)}")
    
    # Load GSM-SBS
    gsm_sbs_file = gsm_sbs_path / "all_examples.json"
    if not gsm_sbs_file.exists():
        raise FileNotFoundError(f"GSM-SBS not found: {gsm_sbs_file}")
    
    with open(gsm_sbs_file) as f:
        gsm_sbs_data = json.load(f)
    
    # GSM-SBS format: question, sbs_answer, final_answer
    gsm_sbs_examples = []
    for ex in gsm_sbs_data:
        # Use the step-by-step answer from GSM-SBS
        answer = ex.get("sbs_answer", ex.get("original_answer", ""))
        gsm_sbs_examples.append({"question": ex["question"], "answer": answer})
    
    print(f"    GSM-SBS examples: {len(gsm_sbs_examples)}")
    
    # Upsample GSM-SBS
    if gsm_sbs_upsample > 1:
        original_count = len(gsm_sbs_examples)
        gsm_sbs_examples = gsm_sbs_examples * gsm_sbs_upsample
        print(f"    GSM-SBS upsampled {gsm_sbs_upsample}x: {original_count} → {len(gsm_sbs_examples)}")
    
    # Combine and shuffle
    all_examples = gsm8k_examples + gsm_sbs_examples
    random.shuffle(all_examples)
    print(f"    Combined (shuffled): {len(all_examples)} examples")
    
    # Set up injector for mul_tokens condition
    injector = None
    if inject_mul_tokens:
        injector = create_injector(mode="weak")
        print("    Mul-token injection: ENABLED")
    
    # Process all examples
    processed = {"input_ids": [], "attention_mask": [], "labels": []}
    for ex in all_examples:
        answer = ex["answer"]
        if injector is not None:
            answer = injector.inject(answer)
        
        result = format_example(ex["question"], answer, tokenizer, max_length)
        processed["input_ids"].append(result["input_ids"])
        processed["attention_mask"].append(result["attention_mask"])
        processed["labels"].append(result["labels"])
    
    return Dataset.from_dict(processed)


def run_stage(
    stage_name: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    train_dataset: Dataset,
    output_dir: Path,
    config: Dict,
    global_config: Dict,
    seed: int = 42,
) -> Dict:
    """Run a single curriculum stage."""
    
    print(f"\n{'='*60}")
    print(f"Stage: {stage_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Examples: {len(train_dataset)}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['epochs']}")
    print(f"{'='*60}\n")
    
    stage_output_dir = output_dir / f"stage_{stage_name}"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(stage_output_dir),
        overwrite_output_dir=True,
        
        # Batch size
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=global_config.get("gradient_accumulation_steps", 4),
        
        # Training length
        num_train_epochs=config["epochs"],
        
        # Optimizer
        learning_rate=config["learning_rate"],
        warmup_ratio=global_config.get("warmup_ratio", 0.1),
        weight_decay=global_config.get("weight_decay", 0.01),
        
        # Precision
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        
        # Logging
        logging_strategy="steps",
        logging_steps=global_config.get("logging_steps", 50),
        
        # Saving - save only at end
        save_strategy="epoch",
        save_total_limit=1,
        
        # Misc
        seed=seed,
        report_to="none",
        dataloader_num_workers=global_config.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    train_result = trainer.train()
    
    # Save stage model
    trainer.save_model(stage_output_dir)
    tokenizer.save_pretrained(stage_output_dir)
    
    # Save stage metrics
    metrics = {
        "stage": stage_name,
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples": len(train_dataset),
        "learning_rate": config["learning_rate"],
        "epochs": config["epochs"],
    }
    
    with open(stage_output_dir / "stage_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nStage {stage_name} complete!")
    print(f"  Train loss: {metrics['train_loss']:.4f}")
    print(f"  Runtime: {metrics['train_runtime']:.1f}s")
    print(f"  Model saved to: {stage_output_dir}")
    
    return metrics


def run_curriculum(
    model_path: Path,
    output_dir: Path,
    condition: str,
    config_path: Path,
    seed: int = 42,
):
    """
    Run full curriculum SFT.
    
    Args:
        model_path: Path to starting model (Tulu-SFT model)
        output_dir: Output directory for curriculum models
        condition: "baseline" or "mul_tokens"
        config_path: Path to curriculum YAML config
        seed: Random seed
    """
    random.seed(seed)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    stages_config = config["stages"]
    
    # Load model and tokenizer
    print(f"Loading starting model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Model parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {"condition": condition, "seed": seed, "stages": []}
    
    inject_mul_tokens = (condition == "mul_tokens")
    
    # ========== Stage 1: Arithmetic Drills ==========
    stage_config = stages_config["arithmetic"]
    arithmetic_dataset = prepare_arithmetic_dataset(
        data_dir=Path("data/arithmetic_drills"),
        condition=condition,
        tokenizer=tokenizer,
        max_length=stage_config["max_length"],
    )
    
    stage_metrics = run_stage(
        stage_name="arithmetic",
        model=model,
        tokenizer=tokenizer,
        train_dataset=arithmetic_dataset,
        output_dir=output_dir,
        config=stage_config,
        global_config=config,
        seed=seed,
    )
    all_metrics["stages"].append(stage_metrics)
    
    # ========== Stage 2: Simple Word Problems ==========
    stage_config = stages_config["simple_word"]
    simple_word_dataset = prepare_simple_word_dataset(
        data_dir=Path("data/simple_word"),
        condition=condition,
        tokenizer=tokenizer,
        max_length=stage_config["max_length"],
    )
    
    stage_metrics = run_stage(
        stage_name="simple_word",
        model=model,
        tokenizer=tokenizer,
        train_dataset=simple_word_dataset,
        output_dir=output_dir,
        config=stage_config,
        global_config=config,
        seed=seed,
    )
    all_metrics["stages"].append(stage_metrics)
    
    # ========== Stage 3: GSM8K + GSM-SBS Interleaved ==========
    stage_config = stages_config["gsm8k_mixed"]
    gsm8k_mixed_dataset = prepare_gsm8k_mixed_dataset(
        gsm_sbs_path=Path(stage_config["gsm_sbs_path"]),
        condition=condition,
        tokenizer=tokenizer,
        max_length=stage_config["max_length"],
        gsm_sbs_upsample=stage_config.get("gsm_sbs_upsample", 3),
        inject_mul_tokens=inject_mul_tokens,
    )
    
    stage_metrics = run_stage(
        stage_name="gsm8k_mixed",
        model=model,
        tokenizer=tokenizer,
        train_dataset=gsm8k_mixed_dataset,
        output_dir=output_dir,
        config=stage_config,
        global_config=config,
        seed=seed,
    )
    all_metrics["stages"].append(stage_metrics)
    
    # ========== Save final results ==========
    results_path = output_dir / "curriculum_results.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Curriculum SFT Complete!")
    print(f"{'='*60}")
    print(f"Condition: {condition}")
    print(f"Final model: {output_dir / 'stage_gsm8k_mixed'}")
    print(f"\nStage Summary:")
    for stage in all_metrics["stages"]:
        print(f"  {stage['stage']}: loss={stage['train_loss']:.4f}, samples={stage['train_samples']}")
    print(f"\nResults saved to: {results_path}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum SFT for domain-token experiments",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to starting model (Tulu-SFT model)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for curriculum models",
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
        default=Path("sft/configs/curriculum.yaml"),
        help="Path to curriculum YAML config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    run_curriculum(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

