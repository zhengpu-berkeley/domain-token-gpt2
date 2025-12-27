#!/usr/bin/env python3
"""
Evaluate pretraining checkpoints on HellaSwag to track learning progress.

Finds all checkpoints in a directory, exports them to HuggingFace format,
and evaluates each on HellaSwag to visualize learning curves.

Usage:
    python scripts/eval_checkpoints.py \
        --condition baseline \
        --checkpoint-dir outputs/pretrain_baseline_10b \
        --output-dir outputs/checkpoint_evals/baseline
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_hellaswag import evaluate_hellaswag
from pretrain.export_hf import export_to_hf

# Tokens per step from config (524288 tokens per optimizer step)
TOKENS_PER_STEP = 524288


def find_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """
    Find all checkpoint files in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        List of checkpoint paths, sorted by step number
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []
    
    for file in checkpoint_dir.glob("model_*.pt"):
        # Extract step number from filename (e.g., model_02000.pt -> 2000)
        match = re.search(r"model_(\d+)\.pt", file.name)
        if match:
            step = int(match.group(1))
            checkpoints.append((step, file))
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    
    return [ckpt_path for _, ckpt_path in checkpoints]


def evaluate_checkpoint(
    checkpoint_path: Path,
    step: int,
    condition: str,
    output_base_dir: Path,
    device: str = "cuda",
    keep_exported_model: bool = False,
) -> Dict:
    """
    Evaluate a single checkpoint on HellaSwag.
    
    Args:
        checkpoint_path: Path to checkpoint .pt file
        step: Training step number
        condition: "baseline" or "mul_tokens"
        output_base_dir: Base directory for outputs
        device: Device to use for evaluation
        keep_exported_model: If True, keep the exported HF model directory (very large)
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating checkpoint: {checkpoint_path.name} (step {step})")
    print(f"{'='*60}")
    
    # Calculate tokens seen
    tokens_seen = step * TOKENS_PER_STEP
    
    # Export checkpoint to HuggingFace format
    hf_model_dir = output_base_dir / f"step_{step}"
    print(f"\n[1/2] Exporting checkpoint to HuggingFace format...")
    try:
        export_to_hf(
            checkpoint_path=checkpoint_path,
            output_dir=hf_model_dir,
            condition=condition,
        )
    except Exception as e:
        print(f"  ERROR: Failed to export checkpoint: {e}")
        return {
            "step": step,
            "tokens_seen": tokens_seen,
            "checkpoint_path": str(checkpoint_path),
            "error": str(e),
        }
    
    # Evaluate on HellaSwag
    # NOTE: `eval/run_hellaswag.py` uses tiktoken directly, so we do NOT need
    # to build/copy a HuggingFace tokenizer into the model directory.
    print(f"\n[2/2] Evaluating on HellaSwag...")
    eval_output_dir = output_base_dir / f"step_{step}_eval"
    try:
        metrics = evaluate_hellaswag(
            model_path=hf_model_dir,
            output_dir=eval_output_dir,
            condition=condition,
            device=device,
        )
    except Exception as e:
        print(f"  ERROR: Failed to evaluate: {e}")
        return {
            "step": step,
            "tokens_seen": tokens_seen,
            "checkpoint_path": str(checkpoint_path),
            "error": f"Evaluation failed: {e}",
        }
    else:
        # Make the per-step result JSON self-contained and stable:
        # - we delete the exported HF model dir, so do not keep a model_path pointing to it
        # - record provenance (checkpoint path + step + tokens)
        eval_results_path = eval_output_dir / "hellaswag_results.json"
        try:
            eval_data = json.loads(eval_results_path.read_text())
            m = eval_data.get("metrics", {})
            m.pop("model_path", None)
            m["step"] = step
            m["tokens_seen"] = tokens_seen
            m["tokens_seen_billions"] = tokens_seen / 1e9
            m["checkpoint_path"] = str(checkpoint_path)
            eval_data["metrics"] = m
            eval_results_path.write_text(json.dumps(eval_data, indent=2) + "\n")
        except Exception as e:
            print(f"  WARNING: Failed to post-process {eval_results_path}: {e}")
    finally:
        # Delete exported HF model weights to avoid filling disk with per-checkpoint exports.
        # Keep only the compact eval outputs and aggregated JSON results.
        if not keep_exported_model:
            try:
                import shutil
                shutil.rmtree(hf_model_dir, ignore_errors=True)
            except Exception as e:
                print(f"  WARNING: Failed to delete exported model dir {hf_model_dir}: {e}")
    
    # Combine results
    result = {
        "step": step,
        "tokens_seen": tokens_seen,
        "tokens_seen_billions": tokens_seen / 1e9,
        "accuracy": metrics["accuracy"],
        "accuracy_norm": metrics["accuracy_norm"],
        "correct": metrics["correct"],
        "correct_norm": metrics["correct_norm"],
        "total": metrics["total"],
        "eval_time_seconds": metrics["eval_time_seconds"],
        "checkpoint_path": str(checkpoint_path),
    }
    
    print(f"\n✓ Checkpoint evaluation complete:")
    print(f"  Step: {step}")
    print(f"  Tokens seen: {tokens_seen / 1e9:.2f}B")
    print(f"  HellaSwag accuracy (normalized): {metrics['accuracy_norm']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretraining checkpoints on HellaSwag",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing checkpoint files (model_*.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for evaluation (cuda or cpu)",
    )
    parser.add_argument(
        "--keep-exported-models",
        action="store_true",
        help="Keep per-checkpoint exported HuggingFace model directories (very large). Default: delete after eval.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to evaluate (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip checkpoints that have already been evaluated",
    )
    
    args = parser.parse_args()
    
    # Find all checkpoints
    print(f"Finding checkpoints in {args.checkpoint_dir}...")
    checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print(f"ERROR: No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Limit number of checkpoints if specified
    if args.max_checkpoints:
        checkpoints = checkpoints[:args.max_checkpoints]
        print(f"Limiting to first {len(checkpoints)} checkpoints")
    
    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir / "checkpoint_results.json"
    
    # Load existing results if skipping
    existing_results = {}
    if args.skip_existing and results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
            for result in data.get("checkpoints", []):
                existing_results[result["step"]] = result
    
    # Evaluate each checkpoint
    results = []
    for checkpoint_path in checkpoints:
        # Extract step number
        match = re.search(r"model_(\d+)\.pt", checkpoint_path.name)
        if not match:
            print(f"WARNING: Could not extract step from {checkpoint_path.name}, skipping")
            continue
        
        step = int(match.group(1))
        
        # Skip if already evaluated
        if args.skip_existing and step in existing_results:
            print(f"\nSkipping step {step} (already evaluated)")
            results.append(existing_results[step])
            continue
        
        # Evaluate checkpoint
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            step=step,
            condition=args.condition,
            output_base_dir=output_dir,
            device=args.device,
            keep_exported_model=args.keep_exported_models,
        )
        
        results.append(result)
        
        # Save intermediate results
        output_data = {
            "condition": args.condition,
            "checkpoints": results,
        }
        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Condition: {args.condition}")
    print(f"Checkpoints evaluated: {len(results)}")
    
    if results:
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            accuracies = [r["accuracy_norm"] for r in successful]
            print(f"\nHellaSwag Accuracy (normalized):")
            print(f"  Min: {min(accuracies):.4f}")
            print(f"  Max: {max(accuracies):.4f}")
            print(f"  Mean: {sum(accuracies) / len(accuracies):.4f}")
            
            # Show first and last
            first = successful[0]
            last = successful[-1]
            print(f"\nFirst checkpoint (step {first['step']}): {first['accuracy_norm']:.4f}")
            print(f"Last checkpoint (step {last['step']}): {last['accuracy_norm']:.4f}")
            print(f"Improvement: {last['accuracy_norm'] - first['accuracy_norm']:.4f}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

