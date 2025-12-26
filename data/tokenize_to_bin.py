#!/usr/bin/env python3
"""
Tokenize text to binary format for training.

Converts text files to uint16 numpy arrays (matching Karpathy's nanoGPT format).
Generates train/val splits and metadata files.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.gpt2_tiktoken import create_tokenizer


def tokenize_text(
    text: str,
    condition: str,
) -> Tuple[List[int], Dict]:
    """
    Tokenize text using the appropriate tokenizer.
    
    Args:
        text: Input text
        condition: "baseline" or "mul_tokens"
        
    Returns:
        (tokens, metadata_dict)
    """
    tokenizer = create_tokenizer(condition)
    tokens = tokenizer.encode(text)
    
    # Compute statistics
    stats = tokenizer.get_mul_token_stats(tokens)
    
    # Count token frequencies for histogram
    token_counts = Counter(tokens)
    
    # Separate mul-token histogram
    mul_token_histogram = {}
    if condition == "mul_tokens":
        for token_id, count in token_counts.items():
            if tokenizer.mul_tokens.is_mul_token_id(token_id):
                mul_token = tokenizer.mul_tokens.get_token_by_id(token_id)
                if mul_token:
                    mul_token_histogram[mul_token.token_str] = count
    
    metadata = {
        "vocab_size": tokenizer.vocab_size,
        "condition": condition,
        "total_tokens": len(tokens),
        **stats,
        "unique_tokens": len(token_counts),
        "mul_token_histogram": mul_token_histogram,
    }
    
    return tokens, metadata


def create_train_val_split(
    tokens: List[int],
    val_fraction: float = 0.1,
) -> Tuple[List[int], List[int]]:
    """
    Split tokens into train and validation sets.
    
    Uses a simple split at a fraction point (not random, for determinism).
    """
    split_idx = int(len(tokens) * (1 - val_fraction))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]
    return train_tokens, val_tokens


def save_tokens_as_bin(
    tokens: List[int],
    output_path: Path,
) -> None:
    """
    Save tokens as uint16 numpy array in .npy format.
    
    This matches the format expected by nanoGPT's DataLoaderLite.
    """
    # Convert to numpy array
    arr = np.array(tokens, dtype=np.uint16)
    
    # Verify no overflow (vocab_size should be < 65536 for uint16)
    if max(tokens) >= 65536:
        raise ValueError(f"Token ID {max(tokens)} exceeds uint16 range")
    
    # Save as .npy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    
    print(f"  Saved {len(tokens):,} tokens to {output_path}")


def tokenize_and_save(
    input_path: Path,
    output_dir: Path,
    condition: str,
    val_fraction: float = 0.1,
) -> Dict:
    """
    Full pipeline: read text -> tokenize -> split -> save.
    
    Args:
        input_path: Path to input text file
        output_dir: Directory for output files
        condition: "baseline" or "mul_tokens"
        val_fraction: Fraction of data for validation
        
    Returns:
        Combined metadata dict
    """
    print(f"Tokenizing {input_path} for condition: {condition}")
    
    # Read text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"  Input: {len(text):,} characters")
    
    # Tokenize
    tokens, metadata = tokenize_text(text, condition)
    print(f"  Tokenized: {len(tokens):,} tokens")
    print(f"  Vocab size: {metadata['vocab_size']}")
    print(f"  Mul tokens: {metadata['mul_tokens']} ({metadata['mul_token_ratio']:.2%})")
    
    # Split
    train_tokens, val_tokens = create_train_val_split(tokens, val_fraction)
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val: {len(val_tokens):,} tokens")
    
    # Prepare output directory
    output_dir = output_dir / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokens
    train_path = output_dir / "train_00000.npy"
    val_path = output_dir / "val_00000.npy"
    
    save_tokens_as_bin(train_tokens, train_path)
    save_tokens_as_bin(val_tokens, val_path)
    
    # Update metadata
    metadata.update({
        "input_path": str(input_path),
        "input_chars": len(text),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "val_fraction": val_fraction,
        "train_path": str(train_path),
        "val_path": str(val_path),
    })
    
    # Save metadata
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {meta_path}")
    
    # Print mul-token histogram if any
    if metadata["mul_token_histogram"]:
        print("\n  Mul-token histogram (top 10):")
        sorted_hist = sorted(
            metadata["mul_token_histogram"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for token, count in sorted_hist[:10]:
            print(f"    {token}: {count}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Tokenize text to binary format")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        default="baseline",
        help="Experimental condition: baseline or mul_tokens",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Path to input text file (default: data/raw/train_{condition}.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "processed",
        help="Output directory for binary files",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    args = parser.parse_args()
    
    # Default input path
    if args.input_path is None:
        args.input_path = Path(__file__).parent / "raw" / f"train_{args.condition}.txt"
    
    if not args.input_path.exists():
        print(f"Error: Input file not found: {args.input_path}")
        print("Run prepare_text.py first to generate the text file.")
        sys.exit(1)
    
    metadata = tokenize_and_save(
        input_path=args.input_path,
        output_dir=args.output_dir,
        condition=args.condition,
        val_fraction=args.val_fraction,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

