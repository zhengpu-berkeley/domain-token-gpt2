#!/usr/bin/env python3
"""
Transition SFT: Bridge between Tulu SFT and TinyGSM SFT.

Implements a gradual gradient from instruction-following (Tulu) format
to math reasoning (TinyGSM) format to prevent loss of EOS generation
while introducing the TinyGSM step-by-step format.

Usage:
    python sft/sft_transition.py \
        --model-path outputs/sft_tulu_baseline \
        --output-dir outputs/transition_baseline \
        --condition baseline \
        --config sft/configs/transition.yaml
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
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.inject_mul import create_injector


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_tulu_messages(
    messages: List[Dict],
    tokenizer,
    inject_mul: bool = False,
    injector=None,
) -> str:
    """Format Tulu-style messages into User/Assistant format."""
    text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if not content:
            continue
        
        # Inject mul-tokens if enabled
        if inject_mul and injector and role in ["assistant", "user"]:
            content = injector.inject(content)
        
        if role == "user":
            text += f"User: {content}\n"
        elif role == "assistant":
            text += f"Assistant: {content}\n"
        elif role == "system":
            text += f"System: {content}\n"
    
    if text:
        text = text.rstrip("\n") + tokenizer.eos_token
    
    return text


def format_tinygsm_example(
    example: Dict,
    tokenizer,
    question_key: str,
    answer_key: str,
    inject_mul: bool = False,
    injector=None,
) -> str:
    """Format TinyGSM example into User/Assistant format."""
    question = example[question_key]
    answer = example[answer_key]
    
    # Note: TinyGSM data already has mul-tokens injected if mul_tokens condition
    # so we don't inject here
    
    text = f"User: {question}\nAssistant: {answer}{tokenizer.eos_token}"
    return text


def load_tulu_samples(
    num_samples: int,
    tokenizer,
    inject_mul: bool = False,
    seed: int = 42,
) -> List[str]:
    """Load Tulu samples from HuggingFace."""
    print(f"  Loading {num_samples} Tulu samples...")
    
    dataset = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
        streaming=True,
    )
    
    injector = create_injector(mode="weak") if inject_mul else None
    
    samples = []
    for i, example in enumerate(dataset):
        if len(samples) >= num_samples:
            break
        
        messages = example.get("messages", [])
        if len(messages) >= 2:
            text = format_tulu_messages(messages, tokenizer, inject_mul, injector)
            if text:
                samples.append(text)
        
        if (i + 1) % 10000 == 0:
            print(f"    Processed {i + 1}, collected {len(samples)} samples")
    
    print(f"    Collected {len(samples)} Tulu samples")
    return samples


def load_tinygsm_samples(
    data_path: Path,
    num_samples: int,
    tokenizer,
    question_key: str,
    answer_key: str,
    seed: int = 42,
) -> List[str]:
    """Load TinyGSM samples from local JSONL."""
    print(f"  Loading {num_samples} TinyGSM samples from {data_path}...")
    
    all_examples = []
    with open(data_path) as f:
        for line in f:
            all_examples.append(json.loads(line))
    
    # Shuffle and take num_samples
    random.seed(seed)
    random.shuffle(all_examples)
    selected = all_examples[:num_samples]
    
    samples = []
    for ex in selected:
        text = format_tinygsm_example(ex, tokenizer, question_key, answer_key)
        samples.append(text)
    
    print(f"    Collected {len(samples)} TinyGSM samples")
    return samples


def create_mixed_dataset(
    tulu_samples: List[str],
    tinygsm_samples: List[str],
    tokenizer,
    max_length: int,
    seed: int = 42,
) -> Dataset:
    """Create a mixed dataset from Tulu and TinyGSM samples."""
    all_texts = tulu_samples + tinygsm_samples
    random.seed(seed)
    random.shuffle(all_texts)
    
    print(f"  Creating mixed dataset with {len(all_texts)} samples...")
    
    # Tokenize
    def tokenize_batch(texts: List[str]) -> Dict:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="np",
        )
        
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)
        
        # Labels: mask padding with -100
        labels = input_ids.copy()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    # Batch tokenize for efficiency
    batch_size = 1000
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Tokenizing"):
        batch = all_texts[i:i + batch_size]
        result = tokenize_batch(batch)
        all_input_ids.append(result["input_ids"])
        all_attention_mask.append(result["attention_mask"])
        all_labels.append(result["labels"])
    
    # Concatenate
    input_ids = np.concatenate(all_input_ids, axis=0)
    attention_mask = np.concatenate(all_attention_mask, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Create dataset
    dataset = Dataset.from_dict({
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": labels.tolist(),
    })
    dataset.set_format("torch")
    
    return dataset


def train_phase(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: Path,
    phase_name: str,
    config: dict,
):
    """Train one phase of the transition."""
    train_cfg = config["training"]
    
    training_args = TrainingArguments(
        output_dir=str(output_dir / phase_name),
        num_train_epochs=1,
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", False),
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )
    
    trainer.train()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Transition SFT (Tulu → TinyGSM bridge)")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to Tulu-SFT model")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--condition", type=str, choices=["baseline", "mul_tokens"],
                        required=True, help="Model condition")
    parser.add_argument("--config", type=Path, default=Path("sft/configs/transition.yaml"),
                        help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]
    data_cfg = config["data"]
    phases = config["phases"]
    
    # Determine TinyGSM data path
    data_dir = Path("data/tinygsm/converted")
    if args.condition == "baseline":
        tinygsm_path = data_dir / data_cfg["baseline_file"]
    else:
        tinygsm_path = data_dir / data_cfg["mul_tokens_file"]
    
    inject_mul = args.condition == "mul_tokens"
    
    print("=" * 70)
    print(f"Transition SFT: {args.condition}")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Model: {args.model_path}")
    print(f"TinyGSM data: {tinygsm_path}")
    print(f"Output: {args.output_dir}")
    print(f"Phases: {len(phases)}")
    for i, phase in enumerate(phases):
        print(f"  Phase {i+1}: {phase['name']} - "
              f"{int(phase['tulu_ratio']*100)}% Tulu, "
              f"{int(phase['tinygsm_ratio']*100)}% TinyGSM, "
              f"{phase['total_samples']} samples")
    print()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print(f"  Model parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    print()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track cumulative Tulu samples used (to avoid repeating same samples)
    tulu_offset = 0
    tinygsm_offset = 0
    
    # Run each phase
    for i, phase in enumerate(phases):
        print("=" * 70)
        print(f"Phase {i+1}/{len(phases)}: {phase['name']}")
        print("=" * 70)
        
        total = phase["total_samples"]
        num_tulu = int(total * phase["tulu_ratio"])
        num_tinygsm = int(total * phase["tinygsm_ratio"])
        
        print(f"  Tulu samples: {num_tulu}")
        print(f"  TinyGSM samples: {num_tinygsm}")
        print()
        
        # Load Tulu samples (use offset to get different samples each phase)
        # For simplicity, we'll just load fresh each time with different seed
        tulu_samples = load_tulu_samples(
            num_tulu,
            tokenizer,
            inject_mul=inject_mul,
            seed=args.seed + i * 1000,
        )
        
        # Load TinyGSM samples
        tinygsm_samples = load_tinygsm_samples(
            tinygsm_path,
            num_tinygsm,
            tokenizer,
            data_cfg["question_key"],
            data_cfg["answer_key"],
            seed=args.seed + i * 1000 + 500,
        )
        
        # Create mixed dataset
        dataset = create_mixed_dataset(
            tulu_samples,
            tinygsm_samples,
            tokenizer,
            train_cfg["max_length"],
            seed=args.seed + i,
        )
        
        print(f"  Dataset size: {len(dataset)}")
        print()
        
        # Train this phase
        print("Starting training...")
        model = train_phase(
            model,
            tokenizer,
            dataset,
            args.output_dir,
            phase["name"],
            config,
        )
        
        print(f"  Phase {i+1} complete!")
        print()
    
    # Save final model
    print("=" * 70)
    print("Saving final model...")
    print("=" * 70)
    
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save metadata
    metadata = {
        "base_model": str(args.model_path),
        "tinygsm_data": str(tinygsm_path),
        "condition": args.condition,
        "config": str(args.config),
        "phases": phases,
    }
    with open(args.output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {args.output_dir}")
    print()
    print("=" * 70)
    print("TRANSITION SFT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

