#!/usr/bin/env python3
"""
Generate 10K arithmetic drill examples for curriculum SFT.

5 difficulty tiers:
- Tier 1 (2K): Single-digit +/-
- Tier 2 (2K): Single-digit × (mul-table facts)
- Tier 3 (2K): Two-digit +/- single-digit
- Tier 4 (2K): Two-digit × single-digit
- Tier 5 (2K): Two-digit × two-digit (with decomposition)
"""

import json
import random
import argparse
from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer.mul_facts import MulFactTokens

MUL_FACTS = MulFactTokens()


def format_mul_token(a: int, b: int) -> Optional[str]:
    """Return mul-token if applicable, else None."""
    # Canonicalize: ensure a <= b
    if a > b:
        a, b = b, a
    if 1 <= a <= 9 and 1 <= b <= 9:
        result = a * b
        return f"<MUL_{a}_{b}_{result}>"
    return None


def generate_tier1(count: int, inject_mul: bool) -> list[dict]:
    """Single-digit addition and subtraction."""
    examples = []
    ops = ['+', '-']
    
    while len(examples) < count:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        op = random.choice(ops)
        
        if op == '+':
            result = a + b
            question = f"What is {a} plus {b}?"
            expr = f"{a} + {b}"
        else:  # subtraction
            # Ensure non-negative result
            if a < b:
                a, b = b, a
            result = a - b
            question = f"What is {a} minus {b}?"
            expr = f"{a} - {b}"
        
        answer = f"{expr} = {result}. The answer is {result}.\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "tier": 1,
            "operation": op
        })
    
    return examples


def generate_tier2(count: int, inject_mul: bool) -> list[dict]:
    """Single-digit multiplication (mul-table facts 1-9)."""
    examples = []
    
    while len(examples) < count:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        result = a * b
        
        # Vary question format
        formats = [
            f"What is {a} times {b}?",
            f"What is {a} × {b}?",
            f"Calculate {a} * {b}.",
            f"Compute {a} multiplied by {b}.",
        ]
        question = random.choice(formats)
        
        if inject_mul:
            mul_token = format_mul_token(a, b)
            expr = f"{a} × {b} = {mul_token}"
        else:
            expr = f"{a} × {b} = {result}"
        
        answer = f"{expr}. The answer is {result}.\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "tier": 2,
            "operation": "×"
        })
    
    return examples


def generate_tier3(count: int, inject_mul: bool) -> list[dict]:
    """Two-digit +/- single-digit."""
    examples = []
    ops = ['+', '-']
    
    while len(examples) < count:
        a = random.randint(10, 99)
        b = random.randint(1, 9)
        op = random.choice(ops)
        
        if op == '+':
            result = a + b
            question = f"What is {a} plus {b}?"
            expr = f"{a} + {b}"
        else:
            result = a - b
            question = f"What is {a} minus {b}?"
            expr = f"{a} - {b}"
        
        answer = f"{expr} = {result}. The answer is {result}.\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "tier": 3,
            "operation": op
        })
    
    return examples


def generate_tier4(count: int, inject_mul: bool) -> list[dict]:
    """Two-digit × single-digit."""
    examples = []
    
    while len(examples) < count:
        a = random.randint(10, 99)
        b = random.randint(2, 9)
        result = a * b
        
        question = random.choice([
            f"What is {a} times {b}?",
            f"Calculate {a} × {b}.",
        ])
        
        # Show decomposition: 12 × 9 = (10 + 2) × 9 = 10×9 + 2×9 = 90 + 18 = 108
        tens = (a // 10) * 10
        ones = a % 10
        
        if inject_mul:
            # Use mul-tokens for single-digit parts
            tens_digit = a // 10
            tens_mul = format_mul_token(tens_digit, b)
            ones_mul = format_mul_token(ones, b) if ones > 0 else None
            
            if ones > 0 and ones_mul:
                tens_result = tens_digit * b
                ones_result = ones * b
                expr = (f"{a} × {b} = ({tens} + {ones}) × {b} = "
                       f"{tens_digit}×{b}×10 + {ones}×{b} = "
                       f"{tens_mul}×10 + {ones_mul} = "
                       f"{tens_result * 10} + {ones_result} = {result}")
            else:
                expr = f"{a} × {b} = {result}"
        else:
            if ones > 0:
                tens_result = tens * b // 10 * 10  # Keep as tens
                ones_result = ones * b
                # Simpler decomposition
                expr = (f"{a} × {b} = ({tens} + {ones}) × {b} = "
                       f"{tens} × {b} + {ones} × {b} = "
                       f"{tens * b} + {ones * b} = {result}")
            else:
                expr = f"{a} × {b} = {result}"
        
        answer = f"{expr}. The answer is {result}.\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "tier": 4,
            "operation": "×"
        })
    
    return examples


def generate_tier5(count: int, inject_mul: bool) -> list[dict]:
    """Two-digit × two-digit with decomposition."""
    examples = []
    
    while len(examples) < count:
        a = random.randint(10, 20)
        b = random.randint(10, 20)
        result = a * b
        
        question = random.choice([
            f"What is {a} times {b}?",
            f"Calculate {a} × {b}.",
        ])
        
        # Decomposition: 12 × 15 = 12 × (10 + 5) = 12×10 + 12×5 = 120 + 60 = 180
        b_tens = (b // 10) * 10
        b_ones = b % 10
        
        part1 = a * b_tens
        part2 = a * b_ones
        
        if inject_mul and b_ones > 0:
            # Try to use mul-tokens for single-digit multiplications in part2
            # e.g., 12 × 5 = (10 + 2) × 5 = 10×5 + 2×5
            a_tens_digit = a // 10
            a_ones = a % 10
            
            mul1 = format_mul_token(a_tens_digit, b_ones)  # e.g., 1×5
            mul2 = format_mul_token(a_ones, b_ones) if a_ones > 0 else None  # e.g., 2×5
            
            if mul1 and mul2:
                sub1 = a_tens_digit * b_ones * 10
                sub2 = a_ones * b_ones
                expr = (f"{a} × {b} = {a} × ({b_tens} + {b_ones}) = "
                       f"{a} × {b_tens} + {a} × {b_ones} = "
                       f"{part1} + ({a_tens_digit}×{b_ones}×10 + {a_ones}×{b_ones}) = "
                       f"{part1} + ({mul1}×10 + {mul2}) = "
                       f"{part1} + ({sub1} + {sub2}) = "
                       f"{part1} + {part2} = {result}")
            else:
                expr = f"{a} × {b} = {a} × ({b_tens} + {b_ones}) = {part1} + {part2} = {result}"
        else:
            expr = f"{a} × {b} = {a} × ({b_tens} + {b_ones}) = {a}×{b_tens} + {a}×{b_ones} = {part1} + {part2} = {result}"
        
        answer = f"{expr}. The answer is {result}.\n#### {result}"
        
        examples.append({
            "question": question,
            "answer": answer,
            "tier": 5,
            "operation": "×"
        })
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic drill dataset")
    parser.add_argument("--output-dir", type=str, default="data/arithmetic_drills",
                        help="Output directory")
    parser.add_argument("--count-per-tier", type=int, default=2000,
                        help="Examples per tier (default: 2000)")
    parser.add_argument("--inject-mul-tokens", action="store_true",
                        help="Generate mul_tokens condition version")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    inject = args.inject_mul_tokens
    condition = "mul_tokens" if inject else "baseline"
    
    print(f"Generating arithmetic drills for condition: {condition}")
    
    # Generate all tiers
    all_examples = []
    
    print(f"  Tier 1 (single-digit +/-): {args.count_per_tier} examples")
    all_examples.extend(generate_tier1(args.count_per_tier, inject))
    
    print(f"  Tier 2 (single-digit ×): {args.count_per_tier} examples")
    all_examples.extend(generate_tier2(args.count_per_tier, inject))
    
    print(f"  Tier 3 (two-digit +/- single-digit): {args.count_per_tier} examples")
    all_examples.extend(generate_tier3(args.count_per_tier, inject))
    
    print(f"  Tier 4 (two-digit × single-digit): {args.count_per_tier} examples")
    all_examples.extend(generate_tier4(args.count_per_tier, inject))
    
    print(f"  Tier 5 (two-digit × two-digit): {args.count_per_tier} examples")
    all_examples.extend(generate_tier5(args.count_per_tier, inject))
    
    # Shuffle
    random.shuffle(all_examples)
    
    # Save
    output_file = out_dir / f"drills_{condition}.json"
    with open(output_file, "w") as f:
        json.dump(all_examples, f, indent=2)
    
    # Save metadata
    metadata = {
        "condition": condition,
        "total_examples": len(all_examples),
        "count_per_tier": args.count_per_tier,
        "tiers": {
            "1": "single-digit +/-",
            "2": "single-digit ×",
            "3": "two-digit +/- single-digit",
            "4": "two-digit × single-digit",
            "5": "two-digit × two-digit"
        },
        "inject_mul_tokens": inject,
        "seed": args.seed
    }
    with open(out_dir / f"metadata_{condition}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(all_examples)} examples to {output_file}")
    
    # Show sample
    print("\n--- Sample examples ---")
    for ex in random.sample(all_examples, 3):
        print(f"Q: {ex['question']}")
        print(f"A: {ex['answer']}")
        print()


if __name__ == "__main__":
    main()

