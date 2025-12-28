#!/usr/bin/env python3
"""
Fix mul-token tokenizers in existing model directories.

This script regenerates tokenizer files for mul_tokens models to fix the bug
where mul-tokens were added as special tokens (causing them to be stripped
during decoding with skip_special_tokens=True).

The fix uses add_tokens() instead of add_special_tokens() so mul-tokens
are preserved in decoded output while still being tokenized atomically.

Usage:
    # Fix all mul_tokens models:
    uv run python scripts/fix_tokenizers.py

    # Fix specific models:
    uv run python scripts/fix_tokenizers.py --model-dirs outputs/hf_mul_tokens_10b

    # Dry run (show what would be changed):
    uv run python scripts/fix_tokenizers.py --dry-run
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.hf_gpt2_with_mul import build_hf_tokenizer


# Default model directories to fix (mul_tokens condition only)
DEFAULT_MODEL_DIRS = [
    "outputs/hf_mul_tokens_10b",
    "outputs/sft_tulu_mul_tokens",
    "outputs/sft_gsm8k_mul_tokens",
]

# Tokenizer files that need to be replaced
TOKENIZER_FILES = [
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer_metadata.json",
    "vocab.json",
    "merges.txt",
]


def check_model_dir(model_dir: Path) -> dict:
    """
    Check if a model directory needs fixing.
    
    Returns:
        dict with 'needs_fix', 'reason', and 'details'
    """
    if not model_dir.exists():
        return {"needs_fix": False, "reason": "directory_not_found", "details": None}
    
    special_tokens_file = model_dir / "special_tokens_map.json"
    if not special_tokens_file.exists():
        return {"needs_fix": False, "reason": "no_tokenizer", "details": None}
    
    with open(special_tokens_file) as f:
        special_tokens = json.load(f)
    
    # Check if mul-tokens are in additional_special_tokens
    if "additional_special_tokens" in special_tokens:
        additional = special_tokens["additional_special_tokens"]
        mul_count = 0
        for t in additional:
            # Handle both formats: plain string or dict with "content" key
            if isinstance(t, dict):
                token_str = t.get("content", "")
            else:
                token_str = str(t)
            if token_str.startswith("<MUL_"):
                mul_count += 1
        if mul_count > 0:
            return {
                "needs_fix": True,
                "reason": "mul_tokens_in_special",
                "details": f"{mul_count} mul-tokens found in additional_special_tokens"
            }
    
    return {"needs_fix": False, "reason": "already_fixed", "details": None}


def fix_tokenizer(model_dir: Path, dry_run: bool = False) -> bool:
    """
    Fix the tokenizer in a model directory.
    
    Args:
        model_dir: Path to model directory
        dry_run: If True, only show what would be done
        
    Returns:
        True if fix was applied (or would be applied in dry_run)
    """
    model_dir = Path(model_dir)
    check = check_model_dir(model_dir)
    
    if not check["needs_fix"]:
        if check["reason"] == "directory_not_found":
            print(f"  SKIP: {model_dir} (directory not found)")
        elif check["reason"] == "no_tokenizer":
            print(f"  SKIP: {model_dir} (no tokenizer files)")
        else:
            print(f"  OK: {model_dir} (already fixed)")
        return False
    
    print(f"  FIX: {model_dir} ({check['details']})")
    
    if dry_run:
        print(f"       [dry-run] Would regenerate tokenizer files")
        return True
    
    # Create a temporary directory for the new tokenizer
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Build fresh tokenizer with the fixed code
        print(f"       Generating fixed tokenizer...")
        build_hf_tokenizer(
            output_dir=tmp_path,
            condition="mul_tokens",  # Use mul_tokens but now with add_tokens()
        )
        
        # Copy tokenizer files to model directory
        print(f"       Copying tokenizer files...")
        for filename in TOKENIZER_FILES:
            src = tmp_path / filename
            dst = model_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                print(f"         {filename}")
    
    # Verify the fix
    verify = check_model_dir(model_dir)
    if verify["needs_fix"]:
        print(f"       ERROR: Fix did not work! Still needs fixing.")
        return False
    else:
        print(f"       SUCCESS: Tokenizer fixed")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Fix mul-token tokenizers in existing model directories",
    )
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        default=None,
        help=f"Model directories to fix (default: {DEFAULT_MODEL_DIRS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which models need fixing",
    )
    
    args = parser.parse_args()
    
    # Get model directories
    if args.model_dirs:
        model_dirs = [Path(d) for d in args.model_dirs]
    else:
        # Use default paths relative to workspace root
        workspace = Path(__file__).parent.parent
        model_dirs = [workspace / d for d in DEFAULT_MODEL_DIRS]
    
    print("=" * 60)
    print("Mul-Token Tokenizer Fixer")
    print("=" * 60)
    
    if args.dry_run:
        print("[DRY RUN MODE - no changes will be made]")
    if args.check_only:
        print("[CHECK ONLY MODE]")
    print()
    
    # Process each model directory
    fixed_count = 0
    skipped_count = 0
    error_count = 0
    
    for model_dir in model_dirs:
        if args.check_only:
            check = check_model_dir(model_dir)
            if check["needs_fix"]:
                print(f"  NEEDS FIX: {model_dir} ({check['details']})")
            elif check["reason"] == "directory_not_found":
                print(f"  NOT FOUND: {model_dir}")
            elif check["reason"] == "no_tokenizer":
                print(f"  NO TOKENIZER: {model_dir}")
            else:
                print(f"  OK: {model_dir}")
        else:
            try:
                if fix_tokenizer(model_dir, dry_run=args.dry_run):
                    fixed_count += 1
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"  ERROR: {model_dir}: {e}")
                error_count += 1
    
    print()
    print("=" * 60)
    if args.check_only:
        print("Check complete.")
    else:
        print(f"Summary: {fixed_count} fixed, {skipped_count} skipped, {error_count} errors")
    print("=" * 60)


if __name__ == "__main__":
    main()

