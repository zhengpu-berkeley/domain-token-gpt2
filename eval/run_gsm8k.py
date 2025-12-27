#!/usr/bin/env python3
"""
GSM8K evaluation script.

Evaluates a model on the GSM8K test set using greedy decoding.
Reports exact-match accuracy on final numeric answers.

Usage:
    python eval/run_gsm8k.py \
        --model-path outputs/grpo_baseline_pilot \
        --output-dir outputs/eval_baseline_pilot \
        --condition baseline
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.rewards import extract_answer, normalize_answer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> str:
    """
    Generate an answer using greedy decoding.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    
    return response


def evaluate_gsm8k(
    model_path: Path,
    output_dir: Path,
    condition: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 256,
) -> Dict:
    """
    Evaluate model on GSM8K test set.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    # Load GSM8K test set
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"  Evaluating on {len(dataset)} samples")
    
    # Evaluate
    results = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for example in tqdm(dataset, desc="Evaluating"):
        question = example["question"].strip()
        answer = example["answer"].strip()
        
        # Format prompt
        prompt = f"Question: {question}\nAnswer:"
        
        # Generate
        response = generate_answer(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        
        # Extract answers
        pred_answer = extract_answer(response)
        true_answer = extract_answer(answer)
        
        pred_normalized = normalize_answer(pred_answer) if pred_answer else ""
        true_normalized = normalize_answer(true_answer) if true_answer else ""
        
        is_correct = (pred_normalized == true_normalized) and pred_normalized != ""
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question": question,
            "true_answer": true_normalized,
            "pred_answer": pred_normalized,
            "response": response[:500],  # Truncate for storage
            "is_correct": is_correct,
        })
    
    end_time = time.time()
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    
    # Count response lengths and mul-token usage
    total_response_tokens = 0
    mul_token_count = 0
    
    for result in results:
        response_tokens = tokenizer.encode(result["response"])
        total_response_tokens += len(response_tokens)
        
        # Count mul tokens (IDs 50304-50348)
        for token_id in response_tokens:
            if 50304 <= token_id <= 50348:
                mul_token_count += 1
    
    avg_response_tokens = total_response_tokens / len(results) if results else 0
    
    metrics = {
        "condition": condition,
        "model_path": str(model_path),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_response_tokens": avg_response_tokens,
        "mul_token_count": mul_token_count,
        "mul_tokens_per_response": mul_token_count / len(results) if results else 0,
        "eval_time_seconds": end_time - start_time,
    }
    
    # Save results
    results_path = output_dir / "gsm8k_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "examples": results[:50],  # Save first 50 examples
        }, f, indent=2)
    
    print(f"\nGSM8K Evaluation Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Avg response tokens: {avg_response_tokens:.1f}")
    if condition == "mul_tokens":
        print(f"  Mul-tokens used: {mul_token_count} ({metrics['mul_tokens_per_response']:.2f} per response)")
    print(f"  Results saved to: {results_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on GSM8K test set",
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
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per response",
    )
    
    args = parser.parse_args()
    
    evaluate_gsm8k(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

