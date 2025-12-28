#!/usr/bin/env python3
"""
Build HuggingFace GPT2TokenizerFast with mul-fact tokens.

Creates a tokenizer compatible with HuggingFace that matches our ID scheme:
- Base GPT-2 vocab: 50257 tokens (IDs 0-50256)
- Padding tokens: 47 tokens (IDs 50257-50303) to reach 50304
- Mul-fact tokens: 45 tokens (IDs 50304-50348)
- Total vocab size: 50349

This ensures the tokenizer produces the same IDs as our tiktoken-based
tokenizer used during pretraining.

Usage:
    python hf_gpt2_with_mul.py --output-dir outputs/hf_baseline_pilot --condition mul_tokens
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from transformers import GPT2TokenizerFast

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.mul_facts import get_default_mul_tokens, MulFactTokens


def build_hf_tokenizer(
    output_dir: Path,
    condition: str,
    mul_tokens: Optional[MulFactTokens] = None,
) -> GPT2TokenizerFast:
    """
    Build HF tokenizer with padding and mul-fact tokens.
    
    Args:
        output_dir: Directory to save the tokenizer
        condition: "baseline" or "mul_tokens"
        mul_tokens: MulFactTokens instance (default: use get_default_mul_tokens())
        
    Returns:
        Configured GPT2TokenizerFast
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mul_tokens is None:
        mul_tokens = get_default_mul_tokens()
    
    print(f"Building HF tokenizer for condition: {condition}")
    
    # Load base GPT-2 tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    base_vocab_size = len(tokenizer)
    print(f"  Base vocab size: {base_vocab_size}")  # Should be 50257
    
    # Add padding tokens to reach 50304 (Karpathy's padded vocab)
    # We need 50304 - 50257 = 47 padding tokens
    padding_needed = 50304 - base_vocab_size
    padding_tokens = [f"<|padding_{i}|>" for i in range(padding_needed)]
    
    print(f"  Adding {len(padding_tokens)} padding tokens")
    num_added = tokenizer.add_tokens(padding_tokens)
    print(f"  Added {num_added} padding tokens")
    
    # Add mul-fact tokens (45 tokens, IDs 50304-50348)
    mul_token_strs = mul_tokens.all_token_strings
    print(f"  Adding {len(mul_token_strs)} mul-fact tokens")
    
    # Add as regular tokens for both conditions (identical vocab size).
    # Using add_tokens() instead of add_special_tokens() ensures mul-tokens
    # are NOT stripped when decoding with skip_special_tokens=True.
    # Tokens added via add_tokens() are still tokenized atomically.
    num_added = tokenizer.add_tokens(mul_token_strs)
    
    print(f"  Added {num_added} mul-fact tokens")
    print(f"  Final vocab size: {len(tokenizer)}")
    
    # Verify IDs match our scheme
    expected_vocab_size = 50349
    if len(tokenizer) != expected_vocab_size:
        print(f"  WARNING: vocab size {len(tokenizer)} != expected {expected_vocab_size}")
    
    # Verify mul-token IDs
    for token in mul_tokens.all_tokens:
        hf_id = tokenizer.convert_tokens_to_ids(token.token_str)
        expected_id = mul_tokens.get_token_id(token.token_str)
        if hf_id != expected_id:
            print(f"  WARNING: {token.token_str} has ID {hf_id}, expected {expected_id}")
    
    # Set pad token (use eos for padding, common in GPT-2)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved tokenizer to {output_dir}")
    
    # Save metadata
    metadata = {
        "condition": condition,
        "base_vocab_size": base_vocab_size,
        "padding_tokens": len(padding_tokens),
        "mul_tokens": len(mul_token_strs),
        "final_vocab_size": len(tokenizer),
        "mul_token_id_start": 50304,
        "mul_token_id_end": 50348,
    }
    
    with open(output_dir / "tokenizer_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return tokenizer


def verify_tokenizer(tokenizer_dir: Path) -> None:
    """
    Verify the saved tokenizer works correctly.
    """
    print(f"\nVerifying tokenizer from {tokenizer_dir}...")
    
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    
    print(f"  Vocab size: {len(tokenizer)}")
    
    # Test basic encoding
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"  Basic test: '{text}' -> {tokens[:5]}... -> '{decoded}'")
    
    # Test mul-token encoding
    mul_text = "The answer is <MUL_6_9_54>."
    tokens = tokenizer.encode(mul_text)
    decoded = tokenizer.decode(tokens)
    print(f"  Mul-token test: '{mul_text}'")
    print(f"    Tokens: {tokens}")
    print(f"    Decoded: '{decoded}'")
    
    # Check if <MUL_6_9_54> is a single token (should be ID 50342)
    mul_token_id = tokenizer.convert_tokens_to_ids("<MUL_6_9_54>")
    expected_6_9 = 50304 + 38  # 6*9 is at index 38 in canonical order
    status = "OK" if mul_token_id == expected_6_9 else "MISMATCH"
    print(f"  <MUL_6_9_54> ID: {mul_token_id} (expected: {expected_6_9}) [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Build HuggingFace GPT2 tokenizer with mul-fact tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for tokenizer files",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the tokenizer after saving",
    )
    
    args = parser.parse_args()
    
    build_hf_tokenizer(
        output_dir=args.output_dir,
        condition=args.condition,
    )
    
    if args.verify:
        verify_tokenizer(args.output_dir)


if __name__ == "__main__":
    main()

