#!/usr/bin/env python3
"""
Plot a single condition's HellaSwag learning curve from checkpoint eval results.

This script is intentionally lightweight and produces ONE PNG (two subplots):
- accuracy_norm vs training step
- accuracy_norm vs tokens seen (billions)

Usage:
  uv run python scripts/plot_checkpoint_curve.py \
    --results-json outputs/checkpoint_evals/baseline/checkpoint_results.json \
    --output-png outputs/checkpoint_evals/baseline/hellaswag_learning_curve.png
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _load_checkpoints(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    checkpoints = data.get("checkpoints", [])
    # keep only successful entries
    checkpoints = [c for c in checkpoints if "error" not in c]
    checkpoints.sort(key=lambda x: int(x["step"]))
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot checkpoint HellaSwag learning curve for one condition")
    parser.add_argument("--results-json", type=Path, required=True, help="Path to checkpoint_results.json")
    parser.add_argument("--output-png", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title override")
    args = parser.parse_args()

    checkpoints = _load_checkpoints(args.results_json)
    if not checkpoints:
        raise SystemExit(f"No successful checkpoints found in {args.results_json}")

    steps = [int(c["step"]) for c in checkpoints]
    tokens_b = [float(c["tokens_seen_billions"]) for c in checkpoints]
    acc_norm = [float(c["accuracy_norm"]) for c in checkpoints]

    # Infer condition label from JSON (fallback to parent directory name)
    try:
        data = json.loads(args.results_json.read_text())
        condition = data.get("condition") or args.results_json.parent.name
    except Exception:
        condition = args.results_json.parent.name

    title = args.title or f"HellaSwag learning curve ({condition})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax1.plot(steps, acc_norm, marker="o", linewidth=2)
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Accuracy (normalized)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(tokens_b, acc_norm, marker="o", linewidth=2)
    ax2.set_xlabel("Tokens seen (B)")
    ax2.set_ylabel("Accuracy (normalized)")
    ax2.grid(True, alpha=0.3)

    # Annotate last point
    ax2.annotate(
        f"{acc_norm[-1]:.3f}",
        xy=(tokens_b[-1], acc_norm[-1]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output_png, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()


