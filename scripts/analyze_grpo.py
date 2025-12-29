#!/usr/bin/env python3
"""
Analyze GRPO training results and create visualizations.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(metrics_file: Path):
    """Load metrics from JSONL file."""
    metrics = []
    with open(metrics_file) as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics

def analyze_grpo_results(baseline_dir: Path, mul_tokens_dir: Path, output_dir: Path):
    """Analyze and visualize GRPO results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    baseline_metrics = load_metrics(baseline_dir / "rl_metrics.jsonl")
    mul_metrics = load_metrics(mul_tokens_dir / "rl_metrics.jsonl")
    
    # Load summaries
    with open(baseline_dir / "rl_summary.json") as f:
        baseline_summary = json.load(f)
    with open(mul_tokens_dir / "rl_summary.json") as f:
        mul_summary = json.load(f)
    
    # Extract time series
    def extract_series(metrics, key):
        return [m[key] for m in metrics]
    
    baseline_steps = extract_series(baseline_metrics, "step")
    mul_steps = extract_series(mul_metrics, "step")
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("GRPO Training Analysis (20x Compute: 2K samples × 40 rollouts)", fontsize=14, fontweight='bold')
    
    # 1. Accuracy over time
    ax = axes[0, 0]
    ax.plot(baseline_steps, [m["accuracy_cumulative"]*100 for m in baseline_metrics], 
            label="Baseline", color="blue", alpha=0.7)
    ax.plot(mul_steps, [m["accuracy_cumulative"]*100 for m in mul_metrics], 
            label="Mul_tokens", color="orange", alpha=0.7)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cumulative Accuracy Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.5, label='Pre-GRPO baseline')
    
    # 2. Reward over time
    ax = axes[0, 1]
    ax.plot(baseline_steps, extract_series(baseline_metrics, "reward_mean"), 
            label="Baseline", color="blue", alpha=0.7)
    ax.plot(mul_steps, extract_series(mul_metrics, "reward_mean"), 
            label="Mul_tokens", color="orange", alpha=0.7)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Repetition rate over time
    ax = axes[1, 0]
    ax.plot(baseline_steps, [m["repetition_rate"]*100 for m in baseline_metrics], 
            label="Baseline", color="blue", alpha=0.7)
    ax.plot(mul_steps, [m["repetition_rate"]*100 for m in mul_metrics], 
            label="Mul_tokens", color="orange", alpha=0.7)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Repetition Rate (%)")
    ax.set_title("Repetition Rate Over Training (Lower = Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Response length over time
    ax = axes[1, 1]
    ax.plot(baseline_steps, extract_series(baseline_metrics, "response_len_mean"), 
            label="Baseline", color="blue", alpha=0.7)
    ax.plot(mul_steps, extract_series(mul_metrics, "response_len_mean"), 
            label="Mul_tokens", color="orange", alpha=0.7)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mean Response Length (words)")
    ax.set_title("Response Length Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Mul-token usage (mul_tokens only)
    ax = axes[2, 0]
    ax.plot(mul_steps, extract_series(mul_metrics, "mul_tokens_per_response"), 
            label="Mul_tokens", color="orange", alpha=0.7)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Mul-tokens per Response")
    ax.set_title("Mul-token Usage Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.644, color='green', linestyle='--', alpha=0.5, label='Pre-GRPO (0.64)')
    
    # 6. Summary comparison bar chart
    ax = axes[2, 1]
    categories = ['Final Accuracy\n(%)', 'Repetition\nRate (%)', 'Hit Max\nLength (%)', 'Mul-tokens\nper Response']
    baseline_vals = [
        baseline_summary['final_accuracy'] * 100,
        baseline_metrics[-1]['repetition_rate'] * 100,
        baseline_metrics[-1]['hit_max_length_rate'] * 100,
        0
    ]
    mul_vals = [
        mul_summary['final_accuracy'] * 100,
        mul_metrics[-1]['repetition_rate'] * 100,
        mul_metrics[-1]['hit_max_length_rate'] * 100,
        mul_summary['mul_tokens_per_response']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
    ax.bar(x + width/2, mul_vals, width, label='Mul_tokens', color='orange', alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Final Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "grpo_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir / 'grpo_analysis.png'}")
    
    # Print analysis summary
    print("\n" + "="*70)
    print("GRPO TRAINING ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n## Final Results")
    print(f"| Metric | Baseline | Mul_tokens | Δ |")
    print(f"|--------|----------|------------|---|")
    print(f"| Accuracy | {baseline_summary['final_accuracy']*100:.2f}% | {mul_summary['final_accuracy']*100:.2f}% | {(mul_summary['final_accuracy'] - baseline_summary['final_accuracy'])*100:+.2f}% |")
    print(f"| Repetition Rate | {baseline_metrics[-1]['repetition_rate']*100:.1f}% | {mul_metrics[-1]['repetition_rate']*100:.1f}% | {(mul_metrics[-1]['repetition_rate'] - baseline_metrics[-1]['repetition_rate'])*100:+.1f}% |")
    print(f"| Hit Max Length | {baseline_metrics[-1]['hit_max_length_rate']*100:.1f}% | {mul_metrics[-1]['hit_max_length_rate']*100:.1f}% | {(mul_metrics[-1]['hit_max_length_rate'] - baseline_metrics[-1]['hit_max_length_rate'])*100:+.1f}% |")
    print(f"| Mul-tokens/resp | {baseline_summary['mul_tokens_per_response']:.2f} | {mul_summary['mul_tokens_per_response']:.2f} | {mul_summary['mul_tokens_per_response'] - baseline_summary['mul_tokens_per_response']:+.2f} |")
    
    # Compare to pre-GRPO curriculum results
    print("\n## Comparison to Pre-GRPO (Curriculum SFT)")
    print("| Model | Pre-GRPO | Post-GRPO | Change |")
    print("|-------|----------|-----------|--------|")
    print(f"| Baseline | 2.40% | {baseline_summary['final_accuracy']*100:.2f}% | {baseline_summary['final_accuracy']*100 - 2.40:+.2f}% |")
    print(f"| Mul_tokens | 3.80% | {mul_summary['final_accuracy']*100:.2f}% | {mul_summary['final_accuracy']*100 - 3.80:+.2f}% |")
    
    # Analysis
    print("\n## Key Observations")
    
    # Check if accuracy improved
    baseline_improved = baseline_summary['final_accuracy'] > 0.024
    mul_improved = mul_summary['final_accuracy'] > 0.038
    
    print(f"1. Baseline accuracy: {'↑ improved' if baseline_improved else '↓ decreased'} from 2.4% to {baseline_summary['final_accuracy']*100:.2f}%")
    print(f"2. Mul_tokens accuracy: {'↑ improved' if mul_improved else '↓ decreased'} from 3.8% to {mul_summary['final_accuracy']*100:.2f}%")
    print(f"3. Repetition rate: ~{baseline_metrics[-1]['repetition_rate']*100:.0f}% for both (very high - model looping)")
    print(f"4. Mul-token usage: {mul_summary['mul_tokens_per_response']:.2f}/response (↑ from 0.64 pre-GRPO)")
    
    # Diagnose issues
    print("\n## Diagnosis")
    if baseline_metrics[-1]['repetition_rate'] > 0.7:
        print("⚠️  HIGH REPETITION: Model is generating repetitive loops in ~80%+ of responses")
        print("    This indicates the model doesn't have strong enough priors for math reasoning")
    
    if baseline_summary['final_accuracy'] < 0.03:
        print("⚠️  LOW ACCURACY: Final accuracy < 3% suggests RL can't find signal")
        print("    The base model needs stronger math capabilities before RL can help")
    
    if not (baseline_improved or mul_improved):
        print("⚠️  NO IMPROVEMENT: 20x RL compute didn't improve either condition")
        print("    This confirms the base model is the bottleneck, not RL compute")
    
    print("\n" + "="*70)
    
    return baseline_summary, mul_summary


if __name__ == "__main__":
    baseline_dir = Path("outputs/grpo_curriculum_baseline")
    mul_tokens_dir = Path("outputs/grpo_curriculum_mul_tokens")
    output_dir = Path("outputs/grpo_analysis")
    
    analyze_grpo_results(baseline_dir, mul_tokens_dir, output_dir)

