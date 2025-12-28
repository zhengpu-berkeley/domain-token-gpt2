#!/usr/bin/env python3
"""
Arithmetic probes evaluation script.

Evaluates model on synthetic arithmetic benchmarks:
- Multiplication tables (1-9)
- Multi-digit multiplication
- Addition/subtraction

Usage:
    python eval/run_arithmetic_probes.py \
        --model-path outputs/grpo_baseline_pilot \
        --output-dir outputs/eval_baseline_pilot \
        --condition baseline
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.rewards import extract_answer, normalize_answer


# Mul-token ID range (IDs 50304-50348)
MUL_TOKEN_ID_START = 50304
MUL_TOKEN_ID_END = 50348


def count_mul_tokens(token_ids: List[int]) -> int:
    """Count mul-tokens (IDs 50304-50348) in a list of token IDs."""
    return sum(1 for tid in token_ids if MUL_TOKEN_ID_START <= tid <= MUL_TOKEN_ID_END)


def generate_multiplication_probes(seed: int = 42) -> List[Dict]:
    """Generate multiplication table probes (1-9 x 1-9)."""
    probes = []
    for a in range(1, 10):
        for b in range(1, 10):
            probes.append({
                "question": f"What is {a} times {b}?",
                "answer": str(a * b),
                "category": "mul_table",
                "a": a,
                "b": b,
            })
    return probes


def generate_multidigit_multiplication_probes(
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """Generate multi-digit multiplication probes."""
    random.seed(seed)
    probes = []
    
    for _ in range(n_samples):
        a = random.randint(10, 99)
        b = random.randint(2, 9)
        probes.append({
            "question": f"What is {a} times {b}?",
            "answer": str(a * b),
            "category": "mul_multidigit",
            "a": a,
            "b": b,
        })
    
    return probes


def generate_addition_probes(
    n_samples: int = 100,
    seed: int = 42,
) -> List[Dict]:
    """Generate addition probes."""
    random.seed(seed)
    probes = []
    
    for _ in range(n_samples):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        probes.append({
            "question": f"What is {a} plus {b}?",
            "answer": str(a + b),
            "category": "addition",
            "a": a,
            "b": b,
        })
    
    return probes


def generate_all_probes(seed: int = 42) -> List[Dict]:
    """Generate all arithmetic probes."""
    probes = []
    probes.extend(generate_multiplication_probes(seed))
    probes.extend(generate_multidigit_multiplication_probes(100, seed))
    probes.extend(generate_addition_probes(100, seed))
    return probes


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    device: str = "cuda",
) -> Tuple[str, List[int]]:
    """
    Generate an answer using greedy decoding.
    
    Returns:
        Tuple of (response_text, generated_token_ids)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated part (excluding prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    return response, generated_ids


def evaluate_probes(
    model_path: Path,
    output_dir: Path,
    condition: str,
    seed: int = 42,
) -> Dict:
    """
    Evaluate model on arithmetic probes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    # Generate probes
    print("Generating arithmetic probes...")
    probes = generate_all_probes(seed)
    print(f"  Total probes: {len(probes)}")
    
    # Evaluate
    results = []
    category_stats = {}
    total_mul_tokens = 0
    total_response_tokens = 0
    
    start_time = time.time()
    
    for probe in tqdm(probes, desc="Evaluating probes"):
        # Format prompt (matches Tulu-3 / GSM8K SFT format)
        prompt = f"User: {probe['question']}\nAssistant: The answer is"
        
        # Generate (now returns both text and raw token IDs)
        response, generated_ids = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=32,
            device=device,
        )
        
        # Count tokens from raw IDs
        total_response_tokens += len(generated_ids)
        probe_mul_tokens = count_mul_tokens(generated_ids)
        total_mul_tokens += probe_mul_tokens
        
        # Extract answer
        pred_answer = extract_answer(response)
        pred_normalized = normalize_answer(pred_answer) if pred_answer else ""
        true_normalized = normalize_answer(probe["answer"])
        
        is_correct = (pred_normalized == true_normalized)
        
        # Track category stats
        category = probe["category"]
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0, "mul_tokens": 0}
        
        category_stats[category]["total"] += 1
        category_stats[category]["mul_tokens"] += probe_mul_tokens
        if is_correct:
            category_stats[category]["correct"] += 1
        
        results.append({
            "question": probe["question"],
            "true_answer": true_normalized,
            "pred_answer": pred_normalized,
            "response": response[:100],
            "is_correct": is_correct,
            "category": category,
            "num_tokens": len(generated_ids),
            "mul_tokens": probe_mul_tokens,
        })
    
    end_time = time.time()
    
    # Calculate metrics
    total_correct = sum(s["correct"] for s in category_stats.values())
    total_samples = sum(s["total"] for s in category_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    category_accuracies = {}
    for cat, stats in category_stats.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        category_accuracies[cat] = {
            "accuracy": acc,
            "correct": stats["correct"],
            "total": stats["total"],
            "mul_tokens": stats["mul_tokens"],
        }
    
    metrics = {
        "condition": condition,
        "model_path": str(model_path),
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_samples": total_samples,
        "category_accuracies": category_accuracies,
        "total_mul_tokens": total_mul_tokens,
        "mul_tokens_per_probe": total_mul_tokens / total_samples if total_samples > 0 else 0,
        "avg_response_tokens": total_response_tokens / total_samples if total_samples > 0 else 0,
        "eval_time_seconds": end_time - start_time,
    }
    
    # Save results
    results_path = output_dir / "arithmetic_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "examples": results[:100],
        }, f, indent=2)
    
    print(f"\nArithmetic Probe Results:")
    print(f"  Overall accuracy: {overall_accuracy:.2%} ({total_correct}/{total_samples})")
    print(f"\n  By category:")
    for cat, stats in category_accuracies.items():
        mul_info = f", mul_tokens: {stats['mul_tokens']}" if condition == "mul_tokens" else ""
        print(f"    {cat}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']}){mul_info}")
    if condition == "mul_tokens":
        print(f"\n  Total mul-tokens used: {total_mul_tokens} ({metrics['mul_tokens_per_probe']:.2f} per probe)")
    print(f"\n  Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on arithmetic probes",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to HuggingFace model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for probe generation",
    )
    
    args = parser.parse_args()
    
    evaluate_probes(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

