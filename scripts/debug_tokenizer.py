#!/usr/bin/env python3
"""
Tokenizer diagnostic script.

Verifies that tokenizers are properly loaded with mul-token support.
Checks vocab size, token IDs, encoding/decoding, and metadata.

Usage:
    python scripts/debug_tokenizer.py \
        --model-path outputs/hf_mul_tokens_10b \
        --condition mul_tokens
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from transformers import GPT2TokenizerFast

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.mul_facts import MulFactTokens, get_default_mul_tokens


def verify_tokenizer(model_path: Path, condition: str) -> Dict:
    """
    Verify tokenizer properties and mul-token support.
    
    Args:
        model_path: Path to HuggingFace model directory
        condition: "baseline" or "mul_tokens"
        
    Returns:
        Dictionary with verification results
    """
    model_path = Path(model_path)
    
    print(f"Loading tokenizer from {model_path}...")
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    except Exception as e:
        return {
            "status": "ERROR",
            "error": str(e),
            "model_path": str(model_path),
            "condition": condition,
        }
    
    # Expected values
    EXPECTED_VOCAB_SIZE = 50349
    MUL_TOKEN_ID_START = 50304
    MUL_TOKEN_ID_END = 50348
    NUM_MUL_TOKENS = 45
    
    results = {
        "status": "OK",
        "model_path": str(model_path),
        "condition": condition,
        "checks": {},
    }
    
    # Check 1: Vocab size
    vocab_size = len(tokenizer)
    results["vocab_size"] = vocab_size
    results["checks"]["vocab_size"] = {
        "value": vocab_size,
        "expected": EXPECTED_VOCAB_SIZE,
        "pass": vocab_size == EXPECTED_VOCAB_SIZE,
    }
    
    if vocab_size != EXPECTED_VOCAB_SIZE:
        results["status"] = "WARNING"
        print(f"  ⚠️  Vocab size mismatch: {vocab_size} != {EXPECTED_VOCAB_SIZE}")
    else:
        print(f"  ✅ Vocab size: {vocab_size}")
    
    # Check 2: Load mul-token reference
    mul_tokens = get_default_mul_tokens()
    
    # Check 3: Verify mul-token IDs exist
    mul_token_checks = []
    sample_tokens_to_test = [
        "<MUL_1_1_1>",
        "<MUL_2_3_6>",
        "<MUL_6_9_54>",
        "<MUL_9_9_81>",
    ]
    
    for token_str in sample_tokens_to_test:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            expected_id = mul_tokens.get_token_id(token_str)
            
            check_result = {
                "token": token_str,
                "token_id": token_id,
                "expected_id": expected_id,
                "pass": token_id == expected_id if expected_id is not None else False,
            }
            
            # Check if it's a single token (not multiple)
            encoded = tokenizer.encode(token_str, add_special_tokens=False)
            check_result["encodes_as_single_token"] = len(encoded) == 1 and encoded[0] == token_id
            
            mul_token_checks.append(check_result)
            
            if check_result["pass"] and check_result["encodes_as_single_token"]:
                print(f"  ✅ {token_str} -> ID {token_id} (single token)")
            else:
                results["status"] = "WARNING"
                if not check_result["pass"]:
                    print(f"  ⚠️  {token_str} -> ID {token_id} (expected {expected_id})")
                if not check_result["encodes_as_single_token"]:
                    print(f"  ⚠️  {token_str} encodes as {len(encoded)} tokens: {encoded}")
        except Exception as e:
            results["status"] = "ERROR"
            mul_token_checks.append({
                "token": token_str,
                "error": str(e),
                "pass": False,
            })
            print(f"  ❌ Error checking {token_str}: {e}")
    
    results["checks"]["mul_token_ids"] = mul_token_checks
    
    # Check 4: Test encoding/decoding roundtrip
    test_texts = [
        "Hello, world!",
        "What is 6 times 9?",
        "The answer is <MUL_6_9_54>.",
        "Calculate 2 * 3 = <MUL_2_3_6>",
    ]
    
    encoding_tests = []
    for text in test_texts:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(encoded)
            
            # Check for mul-tokens in encoding
            mul_token_ids_found = [tid for tid in encoded if MUL_TOKEN_ID_START <= tid <= MUL_TOKEN_ID_END]
            
            test_result = {
                "text": text,
                "encoded": encoded[:20],  # First 20 tokens
                "decoded": decoded,
                "num_tokens": len(encoded),
                "mul_token_ids": mul_token_ids_found,
                "roundtrip_match": text in decoded or decoded in text,  # Approximate match
            }
            encoding_tests.append(test_result)
        except Exception as e:
            encoding_tests.append({
                "text": text,
                "error": str(e),
            })
    
    results["checks"]["encoding_tests"] = encoding_tests
    
    # Check 5: Verify tokenizer metadata file
    metadata_path = model_path / "tokenizer_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            results["metadata"] = metadata
            results["checks"]["metadata"] = {
                "exists": True,
                "vocab_size_match": metadata.get("final_vocab_size") == vocab_size,
                "mul_token_range_match": (
                    metadata.get("mul_token_id_start") == MUL_TOKEN_ID_START and
                    metadata.get("mul_token_id_end") == MUL_TOKEN_ID_END
                ),
                "condition_match": metadata.get("condition") == condition,
            }
            
            print(f"  ✅ Metadata file found")
            print(f"     Condition: {metadata.get('condition')}")
            print(f"     Mul-token range: {metadata.get('mul_token_id_start')}-{metadata.get('mul_token_id_end')}")
        except Exception as e:
            results["checks"]["metadata"] = {
                "exists": True,
                "error": str(e),
            }
            print(f"  ⚠️  Error reading metadata: {e}")
    else:
        results["checks"]["metadata"] = {
            "exists": False,
        }
        print(f"  ⚠️  Metadata file not found: {metadata_path}")
    
    # Check 6: Count mul-token IDs in vocabulary
    mul_token_ids_found = []
    for token_id in range(MUL_TOKEN_ID_START, MUL_TOKEN_ID_END + 1):
        try:
            token_str = tokenizer.convert_ids_to_tokens(token_id)
            if token_str and token_str.startswith("<MUL_"):
                mul_token_ids_found.append({
                    "id": token_id,
                    "token": token_str,
                })
        except:
            pass
    
    results["checks"]["mul_tokens_in_vocab"] = {
        "count": len(mul_token_ids_found),
        "expected": NUM_MUL_TOKENS,
        "pass": len(mul_token_ids_found) == NUM_MUL_TOKENS,
        "sample": mul_token_ids_found[:5] if mul_token_ids_found else [],
    }
    
    if len(mul_token_ids_found) == NUM_MUL_TOKENS:
        print(f"  ✅ Found {len(mul_token_ids_found)} mul-tokens in vocab (expected {NUM_MUL_TOKENS})")
    else:
        results["status"] = "WARNING"
        print(f"  ⚠️  Found {len(mul_token_ids_found)} mul-tokens in vocab (expected {NUM_MUL_TOKENS})")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Verification Summary:")
    print(f"  Status: {results['status']}")
    print(f"  Vocab size: {vocab_size} (expected: {EXPECTED_VOCAB_SIZE})")
    print(f"  Mul-tokens in vocab: {len(mul_token_ids_found)} (expected: {NUM_MUL_TOKENS})")
    print(f"  Condition: {condition}")
    print(f"{'='*60}\n")
    
    return results


def compare_tokenizers(baseline_path: Path, mul_tokens_path: Path) -> Dict:
    """
    Compare baseline and mul_tokens tokenizers side-by-side.
    
    Args:
        baseline_path: Path to baseline model
        mul_tokens_path: Path to mul_tokens model
        
    Returns:
        Comparison dictionary
    """
    print("Comparing tokenizers...")
    print(f"  Baseline: {baseline_path}")
    print(f"  Mul_tokens: {mul_tokens_path}\n")
    
    baseline_results = verify_tokenizer(baseline_path, "baseline")
    mul_tokens_results = verify_tokenizer(mul_tokens_path, "mul_tokens")
    
    comparison = {
        "baseline": baseline_results,
        "mul_tokens": mul_tokens_results,
        "differences": [],
    }
    
    # Compare vocab sizes
    if baseline_results.get("vocab_size") != mul_tokens_results.get("vocab_size"):
        comparison["differences"].append({
            "check": "vocab_size",
            "baseline": baseline_results.get("vocab_size"),
            "mul_tokens": mul_tokens_results.get("vocab_size"),
        })
    
    # Compare mul-token counts
    baseline_count = baseline_results["checks"].get("mul_tokens_in_vocab", {}).get("count", 0)
    mul_tokens_count = mul_tokens_results["checks"].get("mul_tokens_in_vocab", {}).get("count", 0)
    
    if baseline_count != mul_tokens_count:
        comparison["differences"].append({
            "check": "mul_tokens_in_vocab",
            "baseline": baseline_count,
            "mul_tokens": mul_tokens_count,
        })
    
    print(f"\n{'='*60}")
    print("Comparison Summary:")
    if comparison["differences"]:
        print("  ⚠️  Differences found:")
        for diff in comparison["differences"]:
            print(f"     {diff['check']}: baseline={diff['baseline']}, mul_tokens={diff['mul_tokens']}")
    else:
        print("  ✅ No differences found (as expected)")
    print(f"{'='*60}\n")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Debug and verify tokenizer properties",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Path to another model to compare with",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save JSON report (optional)",
    )
    
    args = parser.parse_args()
    
    # Verify single tokenizer
    results = verify_tokenizer(args.model_path, args.condition)
    
    # Compare if requested
    if args.compare_with:
        comparison = compare_tokenizers(args.model_path, args.compare_with)
        results["comparison"] = comparison
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")
    
    # Exit with error code if verification failed
    if results["status"] == "ERROR":
        sys.exit(1)
    elif results["status"] == "WARNING":
        sys.exit(0)  # Warnings are non-fatal
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

