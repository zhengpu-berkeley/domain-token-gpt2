#!/usr/bin/env python3
"""
Compare results between baseline and mul_tokens conditions.

Reads results.json from both experiment outputs and prints a summary.
"""

import json
from pathlib import Path


def load_results(condition: str) -> dict:
    """Load results for a condition."""
    results_path = Path(__file__).parent.parent / "outputs" / f"pretrain_{condition}" / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        return json.load(f)


def load_data_stats(condition: str) -> dict:
    """Load data statistics for a condition."""
    meta_path = Path(__file__).parent.parent / "data" / "processed" / condition / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Experiment Comparison: Baseline vs Mul-Tokens")
    print("=" * 60)
    
    baseline_results = load_results("baseline")
    mul_results = load_results("mul_tokens")
    
    baseline_data = load_data_stats("baseline")
    mul_data = load_data_stats("mul_tokens")
    
    if baseline_results is None or mul_results is None:
        print("Error: Results not found for both conditions.")
        print("Run both experiments first using scripts/run_smoke.sh")
        return
    
    # Data comparison
    print("\n--- Data Statistics ---")
    if baseline_data and mul_data:
        print(f"{'Metric':<30} {'Baseline':>15} {'Mul-Tokens':>15} {'Delta':>15}")
        print("-" * 75)
        
        metrics = [
            ("Total tokens", "total_tokens"),
            ("Train tokens", "train_tokens"),
            ("Val tokens", "val_tokens"),
            ("Mul-fact tokens", "mul_tokens"),
            ("Mul-token ratio (%)", "mul_token_ratio"),
        ]
        
        for label, key in metrics:
            baseline_val = baseline_data.get(key, 0)
            mul_val = mul_data.get(key, 0)
            
            if "ratio" in key:
                delta = (mul_val - baseline_val) * 100
                print(f"{label:<30} {baseline_val*100:>14.2f}% {mul_val*100:>14.2f}% {delta:>+14.2f}%")
            else:
                delta = mul_val - baseline_val
                delta_pct = (delta / baseline_val * 100) if baseline_val > 0 else 0
                print(f"{label:<30} {baseline_val:>15,} {mul_val:>15,} {delta:>+15,} ({delta_pct:+.1f}%)")
    
    # Training comparison
    print("\n--- Training Results ---")
    print(f"{'Metric':<30} {'Baseline':>15} {'Mul-Tokens':>15} {'Delta':>15}")
    print("-" * 75)
    
    baseline_params = baseline_results.get("num_params", 0)
    mul_params = mul_results.get("num_params", 0)
    print(f"{'Parameters':<30} {baseline_params:>15,} {mul_params:>15,} {'(same)':>15}")
    
    baseline_vocab = baseline_results.get("vocab_size", 0)
    mul_vocab = mul_results.get("vocab_size", 0)
    print(f"{'Vocab size':<30} {baseline_vocab:>15,} {mul_vocab:>15,} {'(same)':>15}")
    
    baseline_train = baseline_results.get("final_train_loss", 0)
    mul_train = mul_results.get("final_train_loss", 0)
    train_delta = mul_train - baseline_train
    print(f"{'Final train loss':<30} {baseline_train:>15.4f} {mul_train:>15.4f} {train_delta:>+15.4f}")
    
    baseline_val = baseline_results.get("final_val_loss", 0)
    mul_val = mul_results.get("final_val_loss", 0)
    val_delta = mul_val - baseline_val
    print(f"{'Final val loss':<30} {baseline_val:>15.4f} {mul_val:>15.4f} {val_delta:>+15.4f}")
    
    # Summary
    print("\n--- Summary ---")
    if mul_data and baseline_data:
        token_savings = baseline_data["total_tokens"] - mul_data["total_tokens"]
        token_savings_pct = token_savings / baseline_data["total_tokens"] * 100
        print(f"Token compression: {token_savings:,} fewer tokens ({token_savings_pct:.1f}% reduction)")
    
    print(f"Mul-tokens condition uses {mul_data.get('mul_tokens', 0)} domain-specific tokens")
    
    if val_delta < 0:
        print(f"Mul-tokens achieved LOWER val loss by {abs(val_delta):.4f}")
    elif val_delta > 0:
        print(f"Baseline achieved LOWER val loss by {val_delta:.4f}")
    else:
        print("Both conditions achieved the same val loss")
    
    print("\nNote: This is a tiny smoke test. Real experiments need longer training,")
    print("      more data, and proper evaluation on GSM8K and arithmetic benchmarks.")
    print("=" * 60)


if __name__ == "__main__":
    main()

