#!/usr/bin/env python3
"""
Visualize training and validation losses from pretraining log files.

Creates plots comparing train/val loss curves for baseline and mul_tokens conditions.

Usage:
    python scripts/visualize_pretrain_losses.py \
        --baseline-log outputs/pretrain_baseline_10b/log.txt \
        --mul-tokens-log outputs/pretrain_mul_tokens_10b/log.txt \
        --output-dir outputs/checkpoint_evals
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_log_file(log_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse a pretraining log file to extract train and validation losses.
    
    Args:
        log_path: Path to log.txt file
        
    Returns:
        Tuple of (train_losses, val_losses) where each is a list of dicts
        with 'step' and 'loss' keys
    """
    train_losses = []
    val_losses = []
    
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Match lines like "500 val 5.3237" or "50 train 8.597855"
            match = re.match(r"(\d+)\s+(train|val)\s+([\d.]+)", line)
            if match:
                step = int(match.group(1))
                loss_type = match.group(2)
                loss = float(match.group(3))
                
                entry = {"step": step, "loss": loss}
                
                if loss_type == "train":
                    train_losses.append(entry)
                elif loss_type == "val":
                    val_losses.append(entry)
    
    return train_losses, val_losses


def plot_loss_curves(
    baseline_train: List[Dict],
    baseline_val: List[Dict],
    mul_tokens_train: List[Dict],
    mul_tokens_val: List[Dict],
    output_path: Path,
):
    """
    Plot training and validation loss curves for both conditions.
    
    Args:
        baseline_train: Baseline training losses
        baseline_val: Baseline validation losses
        mul_tokens_train: Mul tokens training losses
        mul_tokens_val: Mul tokens validation losses
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Training Loss
    if baseline_train:
        steps = [e["step"] for e in baseline_train]
        losses = [e["loss"] for e in baseline_train]
        ax1.plot(steps, losses, label="Baseline", color="blue", linewidth=1.5, alpha=0.7)
    
    if mul_tokens_train:
        steps = [e["step"] for e in mul_tokens_train]
        losses = [e["loss"] for e in mul_tokens_train]
        ax1.plot(steps, losses, label="Mul Tokens", color="red", linewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Loss Curves", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_yscale("log")  # Log scale for better visualization
    
    # Plot 2: Validation Loss
    if baseline_val:
        steps = [e["step"] for e in baseline_val]
        losses = [e["loss"] for e in baseline_val]
        ax2.plot(steps, losses, "o-", label="Baseline", color="blue", linewidth=2, markersize=4)
    
    if mul_tokens_val:
        steps = [e["step"] for e in mul_tokens_val]
        losses = [e["loss"] for e in mul_tokens_val]
        ax2.plot(steps, losses, "s-", label="Mul Tokens", color="red", linewidth=2, markersize=4)
    
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Validation Loss", fontsize=12)
    ax2.set_title("Validation Loss Curves", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.suptitle("Pretraining Loss Curves: Baseline vs Mul Tokens", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved loss curves plot to {output_path}")
    plt.close()


def plot_combined_loss(
    baseline_train: List[Dict],
    baseline_val: List[Dict],
    mul_tokens_train: List[Dict],
    mul_tokens_val: List[Dict],
    output_path: Path,
):
    """
    Plot combined train/val loss on the same plot for comparison.
    
    Args:
        baseline_train: Baseline training losses
        baseline_val: Baseline validation losses
        mul_tokens_train: Mul tokens training losses
        mul_tokens_val: Mul tokens validation losses
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Baseline
    if baseline_train:
        steps = [e["step"] for e in baseline_train]
        losses = [e["loss"] for e in baseline_train]
        ax.plot(steps, losses, label="Baseline (Train)", color="blue", linewidth=1.5, alpha=0.6, linestyle="-")
    
    if baseline_val:
        steps = [e["step"] for e in baseline_val]
        losses = [e["loss"] for e in baseline_val]
        ax.plot(steps, losses, "o-", label="Baseline (Val)", color="blue", linewidth=2, markersize=5, alpha=0.8)
    
    # Mul Tokens
    if mul_tokens_train:
        steps = [e["step"] for e in mul_tokens_train]
        losses = [e["loss"] for e in mul_tokens_train]
        ax.plot(steps, losses, label="Mul Tokens (Train)", color="red", linewidth=1.5, alpha=0.6, linestyle="-")
    
    if mul_tokens_val:
        steps = [e["step"] for e in mul_tokens_val]
        losses = [e["loss"] for e in mul_tokens_val]
        ax.plot(steps, losses, "s-", label="Mul Tokens (Val)", color="red", linewidth=2, markersize=5, alpha=0.8)
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Combined Training and Validation Loss Curves", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined loss plot to {output_path}")
    plt.close()


def print_summary(
    baseline_train: List[Dict],
    baseline_val: List[Dict],
    mul_tokens_train: List[Dict],
    mul_tokens_val: List[Dict],
):
    """
    Print a summary of the loss data.
    
    Args:
        baseline_train: Baseline training losses
        baseline_val: Baseline validation losses
        mul_tokens_train: Mul tokens training losses
        mul_tokens_val: Mul tokens validation losses
    """
    print("\n" + "="*60)
    print("Pretraining Loss Summary")
    print("="*60)
    
    if baseline_train:
        train_losses = [e["loss"] for e in baseline_train]
        print(f"\nBaseline Training Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final: {train_losses[-1]:.4f}")
        print(f"  Improvement: {train_losses[0] - train_losses[-1]:.4f}")
        print(f"  Steps: {len(baseline_train)}")
    
    if baseline_val:
        val_losses = [e["loss"] for e in baseline_val]
        print(f"\nBaseline Validation Loss:")
        print(f"  Initial: {val_losses[0]:.4f}")
        print(f"  Final: {val_losses[-1]:.4f}")
        print(f"  Improvement: {val_losses[0] - val_losses[-1]:.4f}")
        print(f"  Steps: {len(baseline_val)}")
    
    if mul_tokens_train:
        train_losses = [e["loss"] for e in mul_tokens_train]
        print(f"\nMul Tokens Training Loss:")
        print(f"  Initial: {train_losses[0]:.4f}")
        print(f"  Final: {train_losses[-1]:.4f}")
        print(f"  Improvement: {train_losses[0] - train_losses[-1]:.4f}")
        print(f"  Steps: {len(mul_tokens_train)}")
    
    if mul_tokens_val:
        val_losses = [e["loss"] for e in mul_tokens_val]
        print(f"\nMul Tokens Validation Loss:")
        print(f"  Initial: {val_losses[0]:.4f}")
        print(f"  Final: {val_losses[-1]:.4f}")
        print(f"  Improvement: {val_losses[0] - val_losses[-1]:.4f}")
        print(f"  Steps: {len(mul_tokens_val)}")
    
    # Compare final losses
    if baseline_val and mul_tokens_val:
        baseline_final = baseline_val[-1]["loss"]
        mul_tokens_final = mul_tokens_val[-1]["loss"]
        diff = mul_tokens_final - baseline_final
        print(f"\nFinal Validation Loss Comparison:")
        print(f"  Baseline: {baseline_final:.4f}")
        print(f"  Mul Tokens: {mul_tokens_final:.4f}")
        print(f"  Difference: {diff:+.4f}")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pretraining loss curves",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        required=True,
        help="Path to baseline pretraining log.txt",
    )
    parser.add_argument(
        "--mul-tokens-log",
        type=Path,
        required=True,
        help="Path to mul_tokens pretraining log.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    
    args = parser.parse_args()
    
    # Parse log files
    print("Parsing log files...")
    baseline_train, baseline_val = parse_log_file(args.baseline_log)
    mul_tokens_train, mul_tokens_val = parse_log_file(args.mul_tokens_log)
    
    print(f"Baseline: {len(baseline_train)} train, {len(baseline_val)} val entries")
    print(f"Mul Tokens: {len(mul_tokens_train)} train, {len(mul_tokens_val)} val entries")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_loss_curves(
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        mul_tokens_train=mul_tokens_train,
        mul_tokens_val=mul_tokens_val,
        output_path=output_dir / "pretrain_loss_curves.png",
    )
    
    plot_combined_loss(
        baseline_train=baseline_train,
        baseline_val=baseline_val,
        mul_tokens_train=mul_tokens_train,
        mul_tokens_val=mul_tokens_val,
        output_path=output_dir / "pretrain_loss_combined.png",
    )
    
    # Print summary
    print_summary(baseline_train, baseline_val, mul_tokens_train, mul_tokens_val)
    
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

