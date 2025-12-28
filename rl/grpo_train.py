#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for GSM8K.

Uses TRL's GRPO implementation with exact-match reward on final answers.
Compute-matched across conditions: same hyperparameters for both.

Compatible with TRL 0.26.2+.

Usage:
    python rl/grpo_train.py \
        --model-path outputs/sft_gsm8k_baseline \
        --output-dir outputs/grpo_baseline \
        --condition baseline \
        --config rl/configs/grpo_pilot.yaml
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
)
from trl import GRPOConfig, GRPOTrainer

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.rewards import extract_answer, normalize_answer


def prepare_gsm8k_dataset(
    max_samples: Optional[int] = None,
):
    """
    Load GSM8K and prepare dataset for GRPO.
    
    The dataset includes:
    - prompt: formatted question (User: {question}\nAssistant:)
    - ground_truth: normalized expected answer
    
    Returns:
        Dataset with prompt and ground_truth columns
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"  Loaded {len(dataset)} samples")
    
    # Format prompts to match SFT format (User/Assistant)
    def format_example(example):
        question = example["question"].strip()
        # Extract and normalize ground truth answer
        answer = extract_answer(example["answer"])
        ground_truth = normalize_answer(answer) if answer else ""
        
        return {
            "prompt": f"User: {question}\nAssistant:",
            "ground_truth": ground_truth,
        }
    
    dataset = dataset.map(format_example, remove_columns=["question", "answer"])
    
    print(f"  Prompts formatted with User:/Assistant: format")
    print(f"  Sample prompt: {dataset[0]['prompt'][:80]}...")
    print(f"  Sample ground_truth: {dataset[0]['ground_truth']}")
    
    return dataset


def create_gsm8k_reward_function(shaped: bool = False, reward_cfg: dict = None):
    """
    Create reward function for GSM8K evaluation.
    
    Args:
        shaped: If True, use shaped rewards with partial credit
        reward_cfg: Config dict with reward weights
    
    TRL passes dataset columns as kwargs, so ground_truth is available.
    """
    import re
    
    # Default reward weights
    cfg = reward_cfg or {}
    correct_answer_weight = cfg.get("correct_answer", 1.0)
    partial_answer_weight = cfg.get("partial_answer", 0.3)
    has_steps_weight = cfg.get("has_steps", 0.1)
    uses_mul_tokens_weight = cfg.get("uses_mul_tokens", 0.2)
    short_response_weight = cfg.get("short_response", 0.1)
    repetition_penalty = cfg.get("repetition_penalty", -0.3)
    
    # Track statistics
    stats = {"correct": 0, "partial": 0, "total": 0, "avg_reward": 0.0}
    
    def detect_repetition(text: str) -> bool:
        """Detect if text has repetitive patterns."""
        # Check for repeated phrases (3+ consecutive repeats)
        words = text.split()
        if len(words) < 10:
            return False
        
        # Look for repeated 3-grams
        for i in range(len(words) - 6):
            trigram = " ".join(words[i:i+3])
            rest = " ".join(words[i+3:])
            if rest.count(trigram) >= 2:
                return True
        
        return False
    
    def is_close_answer(pred: str, true: str) -> bool:
        """Check if prediction is close to true answer."""
        if not pred or not true:
            return False
        try:
            pred_num = float(pred)
            true_num = float(true)
            # Within 10% or off by 1
            if abs(pred_num - true_num) <= 1:
                return True
            if true_num != 0 and abs(pred_num - true_num) / abs(true_num) < 0.1:
                return True
        except (ValueError, ZeroDivisionError):
            pass
        return False
    
    def reward_fn(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for completions.
        """
        rewards = []
        
        for completion, true_answer in zip(completions, ground_truth):
            reward = 0.0
            
            # Extract predicted answer from completion
            pred_answer = extract_answer(completion)
            pred_answer = normalize_answer(pred_answer) if pred_answer else ""
            
            # Exact match check
            is_correct = (pred_answer == true_answer) and pred_answer != ""
            
            if is_correct:
                reward = correct_answer_weight
                stats["correct"] += 1
            elif shaped:
                # Shaped rewards for partial credit
                
                # Partial credit for close answer
                if is_close_answer(pred_answer, true_answer):
                    reward += partial_answer_weight
                    stats["partial"] += 1
                
                # Credit for showing work (has = signs or #### marker)
                if "=" in completion or "####" in completion:
                    reward += has_steps_weight
                
                # Credit for using mul-tokens (condition-specific)
                if "<MUL_" in completion:
                    reward += uses_mul_tokens_weight
                
                # Credit for not hitting max length (natural termination)
                # Assuming max length ~256, give credit for < 200 tokens
                if len(completion.split()) < 150:
                    reward += short_response_weight
                
                # Penalty for repetitive patterns
                if detect_repetition(completion):
                    reward += repetition_penalty
            
            rewards.append(reward)
            stats["total"] += 1
            stats["avg_reward"] = (stats["avg_reward"] * (stats["total"] - 1) + reward) / stats["total"]
        
        # Print periodic stats
        if stats["total"] % 100 == 0:
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  [Reward] Total: {stats['total']}, Accuracy: {acc:.2%}, "
                  f"Partial: {stats['partial']}, Avg Reward: {stats['avg_reward']:.3f}")
        
        return rewards
    
    return reward_fn


def train_grpo(
    model_path: Path,
    output_dir: Path,
    condition: str,
    config_path: Optional[Path] = None,
    seed: int = 42,
):
    """
    Main GRPO training function.
    """
    # Load config
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        # Default config for RTX 4090
        config = {
            "training": {
                "max_samples": 1000,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "num_generations": 4,
                "max_completion_length": 256,
                "temperature": 0.7,
                "learning_rate": 1e-6,
                "num_train_epochs": 1,
                "beta": 0.1,
                "gradient_checkpointing": True,
                "logging_steps": 10,
                "save_steps": 100,
            }
        }
    
    train_cfg = config["training"]
    
    print("=" * 60)
    print(f"GRPO Training Configuration")
    print("=" * 60)
    print(f"  Condition: {condition}")
    print(f"  Model path: {model_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Max samples: {train_cfg.get('max_samples', 'all')}")
    print(f"  Batch size: {train_cfg.get('batch_size', 2)}")
    print(f"  Gradient accumulation: {train_cfg.get('gradient_accumulation_steps', 8)}")
    print(f"  Num generations: {train_cfg.get('num_generations', 4)}")
    print(f"  Max completion length: {train_cfg.get('max_completion_length', 256)}")
    print(f"  Temperature: {train_cfg.get('temperature', 0.7)}")
    print(f"  Learning rate: {train_cfg.get('learning_rate', 1e-6)}")
    print(f"  Beta (KL coef): {train_cfg.get('beta', 0.1)}")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"\nLoading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    # TRL requires left padding for generation
    tokenizer.padding_side = "left"
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print(f"  Model parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Enable gradient checkpointing if requested
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")
    
    # Prepare dataset
    dataset = prepare_gsm8k_dataset(
        max_samples=train_cfg.get("max_samples")
    )
    
    # Create reward function
    reward_cfg = config.get("reward", {})
    shaped = reward_cfg.get("shaped", False)
    reward_fn = create_gsm8k_reward_function(shaped=shaped, reward_cfg=reward_cfg)
    
    if shaped:
        print(f"  Using SHAPED rewards with partial credit")
        print(f"    Correct answer: {reward_cfg.get('correct_answer', 1.0)}")
        print(f"    Partial answer: {reward_cfg.get('partial_answer', 0.3)}")
        print(f"    Has steps: {reward_cfg.get('has_steps', 0.1)}")
        print(f"    Uses mul-tokens: {reward_cfg.get('uses_mul_tokens', 0.2)}")
        print(f"    Short response: {reward_cfg.get('short_response', 0.1)}")
        print(f"    Repetition penalty: {reward_cfg.get('repetition_penalty', -0.3)}")
    else:
        print(f"  Using BINARY rewards (1.0 correct, 0.0 incorrect)")
    
    # GRPO config (TRL 0.26.2 compatible)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        
        # Batch size settings
        per_device_train_batch_size=train_cfg.get("batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        
        # Generation settings (TRL 0.26.2 uses max_completion_length, not max_new_tokens)
        num_generations=train_cfg.get("num_generations", 4),
        max_completion_length=train_cfg.get("max_completion_length", 256),
        temperature=train_cfg.get("temperature", 0.7),
        
        # Training
        learning_rate=train_cfg.get("learning_rate", 1e-6),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        
        # KL divergence (TRL 0.26.2 uses beta, not kl_coef)
        beta=train_cfg.get("beta", 0.1),
        
        # Logging
        logging_steps=train_cfg.get("logging_steps", 10),
        logging_first_step=True,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=3,
        
        # Misc
        seed=seed,
        report_to="none",
        
        # Memory optimization
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        
        # Gradient checkpointing (also set via model.gradient_checkpointing_enable())
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
    )
    
    # Create trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting GRPO training...")
    print("=" * 60 + "\n")
    
    train_result = trainer.train()
    
    # Save final model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Extract metrics
    train_metrics = train_result.metrics
    
    # Save training results
    results = {
        "condition": condition,
        "seed": seed,
        "config": config,
        "train_runtime_seconds": train_metrics.get("train_runtime"),
        "train_samples_per_second": train_metrics.get("train_samples_per_second"),
        "total_steps": train_metrics.get("total_flos"),
        "final_reward": train_metrics.get("reward", None),
        "total_samples": len(dataset),
    }
    
    results_path = output_dir / "grpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("GRPO training complete!")
    print("=" * 60)
    print(f"  Condition: {condition}")
    print(f"  Train runtime: {train_metrics.get('train_runtime', 0):.1f}s")
    print(f"  Results saved to: {results_path}")
    print(f"  Model saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GRPO RL training on GSM8K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Smoke test
    python rl/grpo_train.py \\
        --model-path outputs/sft_gsm8k_baseline \\
        --output-dir outputs/grpo_smoke_test \\
        --condition baseline \\
        --config rl/configs/grpo_smoke.yaml

    # Full training
    python rl/grpo_train.py \\
        --model-path outputs/sft_gsm8k_baseline \\
        --output-dir outputs/grpo_baseline \\
        --condition baseline \\
        --config rl/configs/grpo_pilot.yaml
        """
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to SFT model (HuggingFace format)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for GRPO-trained model",
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
    
    train_grpo(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        config_path=args.config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
