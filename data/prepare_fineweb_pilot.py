#!/usr/bin/env python3
"""
FineWeb-Edu pilot dataset preparation.

Streams FineWeb-Edu from HuggingFace, applies tokenization (and optionally
mul-fact injection for the mul_tokens condition), and writes shards to disk.

Output format is compatible with build-nanogpt's DataLoaderLite:
- Shards named: {split}_{shard_index:06d}.npy
- Each shard is a uint16 numpy array of token IDs
- First shard is validation, rest are training

Usage:
    # Baseline condition (no injection)
    python prepare_fineweb_pilot.py --condition baseline --out-dir data/fineweb_pilot/baseline

    # Mul-tokens condition (with injection)
    python prepare_fineweb_pilot.py --condition mul_tokens --out-dir data/fineweb_pilot/mul_tokens
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.gpt2_tiktoken import create_tokenizer
from tokenizer.inject_mul import create_injector


def generate_synthetic_math_data(seed: int = 42) -> str:
    """
    Generate synthetic multiplication drills to ensure mul-tokens get signal.
    
    This data is appended to the corpus for both conditions. In mul_tokens,
    injection will collapse expressions like '6*9=54' into single tokens.
    
    Returns:
        String with synthetic math content
    """
    random.seed(seed)
    lines = []
    
    # Multiplication table statements
    lines.append("\n\n=== Multiplication Tables ===\n")
    for a in range(1, 10):
        for b in range(a, 10):  # Only upper triangle (canonical)
            c = a * b
            # Multiple formats
            lines.append(f"{a} times {b} equals {c}.")
            lines.append(f"{a}*{b}={c}")
            lines.append(f"{a} × {b} = {c}")
            if random.random() < 0.3:
                lines.append(f"What is {a}*{b}? The answer is {c}.")
    
    # Simple word problems
    lines.append("\n\n=== Math Word Problems ===\n")
    items = ["apples", "oranges", "books", "pencils", "cookies", "toys", "balls", "cards"]
    for _ in range(200):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = a * b
        item = random.choice(items)
        lines.append(f"If you have {a} boxes with {b} {item} each, you have {a}*{b}={c} {item} in total.")
    
    # Drill format
    lines.append("\n\n=== Quick Drills ===\n")
    for _ in range(300):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = a * b
        fmt = random.choice(["asterisk", "times", "x"])
        if fmt == "asterisk":
            lines.append(f"{a}*{b}={c}")
        elif fmt == "times":
            lines.append(f"{a} × {b} = {c}")
        else:
            lines.append(f"{a}x{b}={c}")
    
    return "\n".join(lines)


def stream_fineweb_documents(
    remote_name: str = "sample-10BT",
    streaming: bool = True,
) -> Iterator[str]:
    """
    Stream documents from FineWeb-Edu dataset.
    
    Args:
        remote_name: Dataset subset name
        streaming: If True, stream without downloading full dataset
        
    Yields:
        Document text strings
    """
    from datasets import load_dataset
    
    if streaming:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=remote_name,
            split="train",
            streaming=True,
        )
        for doc in ds:
            yield doc["text"]
    else:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=remote_name,
            split="train",
        )
        for doc in ds:
            yield doc["text"]


def tokenize_document(
    text: str,
    condition: str,
    injector=None,
    tokenizer=None,
    eot_token: int = None,
) -> np.ndarray:
    """
    Tokenize a single document.
    
    Args:
        text: Document text
        condition: "baseline" or "mul_tokens"
        injector: MulExpressionInjector instance (only used if condition=="mul_tokens")
        tokenizer: GPT2TokenizerWithMulFacts instance
        eot_token: End-of-text token ID
        
    Returns:
        uint16 numpy array of tokens (starts with EOT)
    """
    # Apply injection for mul_tokens condition
    if condition == "mul_tokens" and injector is not None:
        text = injector.inject(text)
    
    # Tokenize
    tokens = [eot_token]  # Start with EOT delimiter
    tokens.extend(tokenizer.encode(text))
    
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np


def write_shard(filename: str, tokens_np: np.ndarray) -> None:
    """Write a shard to disk as .npy file."""
    np.save(filename, tokens_np)


def prepare_fineweb_pilot(
    condition: str,
    out_dir: Path,
    target_tokens: int = 200_000_000,  # 200M tokens default
    shard_size: int = 10_000_000,  # 10M tokens per shard
    val_shard_count: int = 1,  # Number of shards for validation
    streaming: bool = True,
    include_synthetic: bool = True,
    num_workers: int = None,
    seed: int = 42,
) -> dict:
    """
    Prepare FineWeb-Edu pilot dataset.
    
    Args:
        condition: "baseline" or "mul_tokens"
        out_dir: Output directory
        target_tokens: Stop after this many tokens
        shard_size: Tokens per shard
        val_shard_count: Number of shards to use for validation
        streaming: Stream dataset instead of downloading
        include_synthetic: Include synthetic math data
        num_workers: Number of worker processes (default: half of CPUs)
        seed: Random seed
        
    Returns:
        Metadata dict
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer and injector
    tokenizer = create_tokenizer(condition)
    injector = create_injector("weak") if condition == "mul_tokens" else None
    eot_token = tokenizer.eot_token
    
    print(f"Preparing FineWeb-Edu pilot for condition: {condition}")
    print(f"  Output directory: {out_dir}")
    print(f"  Target tokens: {target_tokens:,}")
    print(f"  Shard size: {shard_size:,}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    if num_workers is None:
        num_workers = max(1, os.cpu_count() // 2)
    
    # Stats tracking
    total_tokens = 0
    total_documents = 0
    mul_token_count = 0
    
    # Shard management
    shard_index = 0
    current_shard = np.empty((shard_size,), dtype=np.uint16)
    shard_position = 0
    
    # Generate synthetic math data first (to ensure it's at the start)
    if include_synthetic:
        print("  Generating synthetic math data...")
        synthetic_text = generate_synthetic_math_data(seed)
        synthetic_tokens = tokenize_document(
            synthetic_text, condition, injector, tokenizer, eot_token
        )
        print(f"  Synthetic data: {len(synthetic_tokens):,} tokens")
        
        # Add synthetic tokens
        if shard_position + len(synthetic_tokens) < shard_size:
            current_shard[shard_position:shard_position + len(synthetic_tokens)] = synthetic_tokens
            shard_position += len(synthetic_tokens)
            total_tokens += len(synthetic_tokens)
            
            if condition == "mul_tokens":
                mul_token_count += sum(1 for t in synthetic_tokens if tokenizer.mul_tokens.is_mul_token_id(t))
    
    # Process FineWeb documents
    print("  Streaming FineWeb-Edu documents...")
    
    progress_bar = tqdm(total=target_tokens, unit="tokens", desc=f"Shard {shard_index}")
    
    try:
        for doc_text in stream_fineweb_documents(streaming=streaming):
            if total_tokens >= target_tokens:
                break
            
            # Tokenize document
            doc_tokens = tokenize_document(
                doc_text, condition, injector, tokenizer, eot_token
            )
            
            total_documents += 1
            
            # Count mul tokens
            if condition == "mul_tokens":
                mul_token_count += sum(1 for t in doc_tokens if tokenizer.mul_tokens.is_mul_token_id(t))
            
            # Add to current shard
            tokens_remaining = len(doc_tokens)
            tokens_pos = 0
            
            while tokens_remaining > 0:
                space_in_shard = shard_size - shard_position
                tokens_to_add = min(tokens_remaining, space_in_shard)
                
                current_shard[shard_position:shard_position + tokens_to_add] = \
                    doc_tokens[tokens_pos:tokens_pos + tokens_to_add]
                
                shard_position += tokens_to_add
                tokens_pos += tokens_to_add
                tokens_remaining -= tokens_to_add
                total_tokens += tokens_to_add
                
                progress_bar.update(tokens_to_add)
                
                # Write shard if full
                if shard_position >= shard_size:
                    # First val_shard_count shards are validation
                    split = "val" if shard_index < val_shard_count else "train"
                    filename = out_dir / f"fineweb_{split}_{shard_index:06d}.npy"
                    write_shard(str(filename), current_shard)
                    print(f"\n  Wrote shard {shard_index} ({split}): {filename.name}")
                    
                    shard_index += 1
                    shard_position = 0
                    current_shard = np.empty((shard_size,), dtype=np.uint16)
                    
                    progress_bar.close()
                    progress_bar = tqdm(total=target_tokens, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(total_tokens)
                
                if total_tokens >= target_tokens:
                    break
    
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    except Exception as e:
        print(f"\n  Error during streaming: {e}")
    finally:
        try:
            progress_bar.close()
        except Exception:
            pass  # Ignore cleanup errors
    
    # Write final partial shard if any
    if shard_position > 0:
        split = "val" if shard_index < val_shard_count else "train"
        filename = out_dir / f"fineweb_{split}_{shard_index:06d}.npy"
        write_shard(str(filename), current_shard[:shard_position])
        print(f"  Wrote final shard {shard_index} ({split}): {filename.name} ({shard_position:,} tokens)")
        shard_index += 1
    
    # Count train vs val tokens
    train_tokens = 0
    val_tokens = 0
    for f in out_dir.glob("fineweb_*.npy"):
        tokens = np.load(f)
        if "val" in f.name:
            val_tokens += len(tokens)
        else:
            train_tokens += len(tokens)
    
    # Create metadata
    metadata = {
        "condition": condition,
        "vocab_size": tokenizer.vocab_size,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_documents": total_documents,
        "num_shards": shard_index,
        "shard_size": shard_size,
        "target_tokens": target_tokens,
        "mul_tokens": mul_token_count,
        "mul_token_ratio": mul_token_count / total_tokens if total_tokens > 0 else 0,
        "include_synthetic": include_synthetic,
        "streaming": streaming,
        "seed": seed,
        "tokenizer_metadata": tokenizer.get_metadata(),
    }
    
    # Save metadata
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  Summary:")
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Train tokens: {train_tokens:,}")
    print(f"    Val tokens: {val_tokens:,}")
    print(f"    Documents: {total_documents:,}")
    print(f"    Shards: {shard_index}")
    if condition == "mul_tokens":
        print(f"    Mul-tokens: {mul_token_count:,} ({metadata['mul_token_ratio']:.2%})")
    print(f"    Metadata saved to: {meta_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu pilot dataset for pretraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition: baseline or mul_tokens",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for shards",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=200_000_000,
        help="Target number of tokens (default: 200M)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000_000,
        help="Tokens per shard (default: 10M)",
    )
    parser.add_argument(
        "--val-shards",
        type=int,
        default=1,
        help="Number of validation shards (default: 1)",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Download dataset instead of streaming",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Skip synthetic math data augmentation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers (default: half of CPUs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    prepare_fineweb_pilot(
        condition=args.condition,
        out_dir=args.out_dir,
        target_tokens=args.target_tokens,
        shard_size=args.shard_size,
        val_shard_count=args.val_shards,
        streaming=not args.no_streaming,
        include_synthetic=not args.no_synthetic,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

