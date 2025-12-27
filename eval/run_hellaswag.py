#!/usr/bin/env python3
"""
HellaSwag evaluation script (Karpathy-compatible).

This matches `third_party/build-nanogpt/hellaswag.py` logic and tokenization by
using `tiktoken.get_encoding("gpt2")` directly.

Usage:
    uv run python eval/run_hellaswag.py \
        --model-path outputs/hf_baseline_10b \
        --output-dir outputs/eval_baseline_10b \
        --condition baseline
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests
import tiktoken
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# -----------------------------------------------------------------------------
# Data download and caching - match Karpathy's structure exactly
DATA_CACHE_DIR = os.path.join(Path(__file__).parent.parent, "third_party", "build-nanogpt", "hellaswag")

HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# Use tiktoken exactly like Karpathy does
enc = tiktoken.get_encoding("gpt2")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_hellaswag(split: str):
    """Downloads HellaSwag dataset to cache directory"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = HELLASWAG_URLS[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)


def iterate_examples(split: str):
    """Iterate over HellaSwag examples"""
    download_hellaswag(split)
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def render_example(example: Dict):
    """
    Given the example as a dictionary, render it as three torch tensors.
    This matches Karpathy's implementation exactly.

    Returns:
        tuple: (data dict, tokens tensor, mask tensor, label int)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens - use tiktoken exactly like Karpathy
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


@torch.no_grad()
def evaluate_hellaswag(
    model_path: Path,
    output_dir: Path,
    condition: str,
    max_samples: Optional[int] = None,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate model on HellaSwag validation set.
    Matches Karpathy's implementation exactly.

    Args:
        model_path: Path to HuggingFace model directory (or model id, e.g. \"gpt2\")
        output_dir: Directory to save results
        condition: "baseline" or "mul_tokens"
        max_samples: Maximum number of samples to evaluate (default: all 10,042)
        device: Device to run evaluation on

    Returns:
        Dictionary with evaluation metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load model - use HuggingFace but tokenize with tiktoken
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Use tf32 for faster computation on Ampere+ GPUs (like Karpathy)
    torch.set_float32_matmul_precision("high")

    print(f"  Device: {device}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Using tiktoken gpt2 encoding (vocab size: {enc.n_vocab})")

    # Load HellaSwag validation set
    print("Loading HellaSwag validation set...")

    num_correct = 0
    num_correct_norm = 0
    num_total = 0

    start_time = time.time()

    # Evaluate - match Karpathy's logic exactly
    for example in tqdm(iterate_examples("val"), desc="Evaluating"):
        if max_samples is not None and num_total >= max_samples:
            break

        # Render example into tokens and mask
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous()  # shift mask to align with shifted logits
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # Print progress every 1000 examples
        if num_total % 1000 == 0:
            acc = num_correct / num_total
            acc_norm = num_correct_norm / num_total
            print(f"  Progress: {num_total} examples, acc: {acc:.4f}, acc_norm: {acc_norm:.4f}")

    end_time = time.time()

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    accuracy_norm = num_correct_norm / num_total if num_total > 0 else 0.0

    metrics = {
        "condition": condition,
        "model_path": str(model_path),
        "accuracy": accuracy,
        "accuracy_norm": accuracy_norm,
        "correct": num_correct,
        "correct_norm": num_correct_norm,
        "total": num_total,
        "eval_time_seconds": end_time - start_time,
    }

    results_path = output_dir / "hellaswag_results.json"
    with open(results_path, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)

    print(f"\nHellaSwag Evaluation Results:")
    print(f"  Accuracy (unnormalized): {accuracy:.4f} ({num_correct}/{num_total})")
    print(f"  Accuracy (normalized): {accuracy_norm:.4f} ({num_correct_norm}/{num_total})")
    print(f"  Evaluation time: {end_time - start_time:.1f} seconds")
    print(f"  Results saved to: {results_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HellaSwag validation set (Karpathy-compatible)")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to HuggingFace model")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate (default: all 10,042)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()
    evaluate_hellaswag(
        model_path=args.model_path,
        output_dir=args.output_dir,
        condition=args.condition,
        max_samples=args.max_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()


