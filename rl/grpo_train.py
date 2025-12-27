#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) training for GSM8K.

Uses TRL's GRPO implementation with exact-match reward on final answers.
Compute-matched across conditions: same hyperparameters for both.

Usage:
    python rl/grpo_train.py \
        --model-path outputs/sft_baseline_pilot \
        --output-dir outputs/grpo_baseline_pilot \
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
    GenerationConfig,
)
from trl import GRPOConfig, GRPOTrainer

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.rewards import extract_answer, normalize_answer


def prepare_gsm8k_prompts(
    max_samples: Optional[int] = None,
) -> tuple:
    """
    Load GSM8K and prepare prompts for RL.
    
    Returns:
        (dataset with prompts, list of ground truth answers)
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"  Loaded {len(dataset)} samples")
    
    # Format prompts
    def format_prompt(example):
        return {"prompt": f"Question: {example['question'].strip()}\nAnswer:"}
    
    dataset = dataset.map(format_prompt)
    
    # Extract ground truth answers
    ground_truths = []
    for example in dataset:
        answer = extract_answer(example["answer"])
        ground_truths.append(normalize_answer(answer) if answer else "")
    
    return dataset, ground_truths


def create_reward_function(ground_truths: List[str]):
    """
    Create a reward function for GRPO.
    
    The function takes completions and returns rewards.
    """
    idx = [0]  # Mutable counter for tracking position
    
    def reward_fn(completions: List[str], **kwargs) -> List[float]:
        """Compute rewards for a batch of completions."""
        rewards = []
        batch_size = len(completions)
        
        for completion in completions:
            # Get ground truth for this sample
            if idx[0] < len(ground_truths):
                true_answer = ground_truths[idx[0] % len(ground_truths)]
            else:
                true_answer = ""
            
            # Extract predicted answer
            pred_answer = extract_answer(completion)
            pred_answer = normalize_answer(pred_answer) if pred_answer else ""
            
            # Compute reward
            if pred_answer == true_answer and pred_answer != "":
                reward = 1.0
            else:
                reward = 0.0
            
            rewards.append(reward)
        
        idx[0] += batch_size
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
        # Default config
        config = {
            "training": {
                "max_samples": 1000,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_generations": 4,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "learning_rate": 1e-6,
                "num_train_epochs": 1,
                "kl_coef": 0.1,
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
    
    # Prepare dataset
    dataset, ground_truths = prepare_gsm8k_prompts(
        max_samples=train_cfg.get("max_samples")
    )
    
    # Create reward function
    reward_fn = create_reward_function(ground_truths)
    
    # GRPO config
    output_dir = Path(output_dir)
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        
        # Batch size
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        
        # Generation
        num_generations=train_cfg.get("num_generations", 4),
        max_new_tokens=train_cfg["max_new_tokens"],
        temperature=train_cfg["temperature"],
        
        # Training
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        
        # Logging
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        
        # Misc
        seed=seed,
        report_to="none",
        
        # bf16 if available
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
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
    print("\nStarting GRPO training...")
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
        "total_samples": len(dataset),
    }
    
    results_path = output_dir / "grpo_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nGRPO training complete!")
    print(f"  Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="GRPO RL training on GSM8K",
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

