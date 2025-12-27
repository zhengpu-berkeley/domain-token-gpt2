#!/usr/bin/env python3
"""
Compare baseline vs mul_tokens experimental results.

Reads evaluation outputs from both conditions and generates a summary report.

Usage:
    python scripts/compare_runs.py \
        --baseline-dir outputs/eval_baseline_pilot \
        --mul-tokens-dir outputs/eval_mul_tokens_pilot \
        --output-path outputs/comparison_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def load_json_safe(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def compare_experiments(
    baseline_dir: Path,
    mul_tokens_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict:
    """
    Compare results from baseline and mul_tokens experiments.
    """
    baseline_dir = Path(baseline_dir)
    mul_tokens_dir = Path(mul_tokens_dir)
    
    report = {
        "baseline": {},
        "mul_tokens": {},
        "comparison": {},
    }
    
    # Load GSM8K results
    baseline_gsm8k = load_json_safe(baseline_dir / "gsm8k_results.json")
    mul_gsm8k = load_json_safe(mul_tokens_dir / "gsm8k_results.json")
    
    if baseline_gsm8k and "metrics" in baseline_gsm8k:
        report["baseline"]["gsm8k"] = baseline_gsm8k["metrics"]
    if mul_gsm8k and "metrics" in mul_gsm8k:
        report["mul_tokens"]["gsm8k"] = mul_gsm8k["metrics"]
    
    # Load arithmetic results
    baseline_arith = load_json_safe(baseline_dir / "arithmetic_results.json")
    mul_arith = load_json_safe(mul_tokens_dir / "arithmetic_results.json")
    
    if baseline_arith and "metrics" in baseline_arith:
        report["baseline"]["arithmetic"] = baseline_arith["metrics"]
    if mul_arith and "metrics" in mul_arith:
        report["mul_tokens"]["arithmetic"] = mul_arith["metrics"]
    
    # Compute comparison metrics
    if report["baseline"].get("gsm8k") and report["mul_tokens"].get("gsm8k"):
        baseline_acc = report["baseline"]["gsm8k"]["accuracy"]
        mul_acc = report["mul_tokens"]["gsm8k"]["accuracy"]
        report["comparison"]["gsm8k"] = {
            "baseline_accuracy": baseline_acc,
            "mul_tokens_accuracy": mul_acc,
            "delta": mul_acc - baseline_acc,
            "relative_improvement": (mul_acc - baseline_acc) / baseline_acc if baseline_acc > 0 else None,
        }
    
    if report["baseline"].get("arithmetic") and report["mul_tokens"].get("arithmetic"):
        baseline_acc = report["baseline"]["arithmetic"]["overall_accuracy"]
        mul_acc = report["mul_tokens"]["arithmetic"]["overall_accuracy"]
        report["comparison"]["arithmetic"] = {
            "baseline_accuracy": baseline_acc,
            "mul_tokens_accuracy": mul_acc,
            "delta": mul_acc - baseline_acc,
        }
        
        # Compare by category
        baseline_cats = report["baseline"]["arithmetic"].get("category_accuracies", {})
        mul_cats = report["mul_tokens"]["arithmetic"].get("category_accuracies", {})
        
        category_comparison = {}
        for cat in set(baseline_cats.keys()) | set(mul_cats.keys()):
            b_acc = baseline_cats.get(cat, {}).get("accuracy", 0)
            m_acc = mul_cats.get(cat, {}).get("accuracy", 0)
            category_comparison[cat] = {
                "baseline": b_acc,
                "mul_tokens": m_acc,
                "delta": m_acc - b_acc,
            }
        report["comparison"]["arithmetic_by_category"] = category_comparison
    
    # Mul-token usage stats
    if report["mul_tokens"].get("gsm8k"):
        report["comparison"]["mul_token_usage"] = {
            "total_mul_tokens": report["mul_tokens"]["gsm8k"].get("mul_token_count", 0),
            "per_response": report["mul_tokens"]["gsm8k"].get("mul_tokens_per_response", 0),
        }
    
    # Print report
    print_report(report)
    
    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")
    
    return report


def print_report(report: Dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON: Baseline vs Mul-Tokens")
    print("=" * 70)
    
    comp = report.get("comparison", {})
    
    # GSM8K
    if "gsm8k" in comp:
        gsm = comp["gsm8k"]
        print("\n--- GSM8K Test Set ---")
        print(f"  Baseline accuracy:   {gsm['baseline_accuracy']:.2%}")
        print(f"  Mul-tokens accuracy: {gsm['mul_tokens_accuracy']:.2%}")
        delta = gsm['delta']
        sign = "+" if delta >= 0 else ""
        print(f"  Delta:               {sign}{delta:.2%}")
        
        if gsm.get("relative_improvement") is not None:
            rel = gsm["relative_improvement"]
            sign = "+" if rel >= 0 else ""
            print(f"  Relative improvement:{sign}{rel:.1%}")
    
    # Arithmetic
    if "arithmetic" in comp:
        arith = comp["arithmetic"]
        print("\n--- Arithmetic Probes ---")
        print(f"  Baseline accuracy:   {arith['baseline_accuracy']:.2%}")
        print(f"  Mul-tokens accuracy: {arith['mul_tokens_accuracy']:.2%}")
        delta = arith['delta']
        sign = "+" if delta >= 0 else ""
        print(f"  Delta:               {sign}{delta:.2%}")
    
    # By category
    if "arithmetic_by_category" in comp:
        print("\n  By category:")
        for cat, stats in comp["arithmetic_by_category"].items():
            delta = stats['delta']
            sign = "+" if delta >= 0 else ""
            print(f"    {cat:20s}: {stats['baseline']:.2%} -> {stats['mul_tokens']:.2%} ({sign}{delta:.2%})")
    
    # Mul-token usage
    if "mul_token_usage" in comp:
        usage = comp["mul_token_usage"]
        print("\n--- Mul-Token Usage (mul_tokens condition) ---")
        print(f"  Total mul-tokens in responses: {usage['total_mul_tokens']}")
        print(f"  Per response average:          {usage['per_response']:.2f}")
    
    print("\n" + "=" * 70)
    
    # Summary
    if "gsm8k" in comp:
        delta = comp["gsm8k"]["delta"]
        if delta > 0:
            print("RESULT: Mul-tokens condition shows IMPROVEMENT on GSM8K")
        elif delta < 0:
            print("RESULT: Mul-tokens condition shows REGRESSION on GSM8K")
        else:
            print("RESULT: No difference between conditions on GSM8K")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs mul_tokens experimental results",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Directory with baseline evaluation results",
    )
    parser.add_argument(
        "--mul-tokens-dir",
        type=Path,
        required=True,
        help="Directory with mul_tokens evaluation results",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save comparison report JSON",
    )
    
    args = parser.parse_args()
    
    compare_experiments(
        baseline_dir=args.baseline_dir,
        mul_tokens_dir=args.mul_tokens_dir,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()

