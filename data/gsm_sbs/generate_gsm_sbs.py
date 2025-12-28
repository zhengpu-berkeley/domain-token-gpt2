#!/usr/bin/env python3
"""
Generate GSM-SBS (GSM Step-By-Step) dataset.

This dataset teaches the model to break down multiplications using
the distributive property and step-by-step arithmetic RECURSIVELY
until reaching single-digit × single-digit base cases.

Examples of decomposition:
- 6 * 9 = 54 (base case: single-digit × single-digit)
- 2 * 12 = 2 * (10 + 2) = 2*10 + 2*2 = 20 + 4 = 24
- 12 * 12 = (10 + 2) * 12 = 10*12 + 2*12
         = 10*12 + 2*(10+2)
         = 120 + 2*10 + 2*2
         = 120 + 20 + 4 = 144
- 15 * 7 = (10 + 5) * 7 = 10*7 + 5*7 = 70 + 35 = 105

Key principle: Keep breaking down until ALL multiplications are
either single-digit × single-digit OR × 10.
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from datasets import load_dataset

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def is_base_case(a: int, b: int) -> bool:
    """
    Check if a multiplication is a base case (no further breakdown needed).
    Base cases:
    - Both single-digit (2-9)
    - One is 10 (easy mental math)
    - One is 1
    - One is 0
    """
    if a == 0 or b == 0:
        return True
    if a == 1 or b == 1:
        return True
    if a == 10 or b == 10:
        return True
    if 2 <= a <= 9 and 2 <= b <= 9:
        return True
    return False


def decompose_multiplication_recursive(a: int, b: int, depth: int = 0) -> str:
    """
    Recursively decompose a multiplication into step-by-step form.
    
    Strategy:
    1. If base case, compute directly
    2. Otherwise, break the larger number into tens + ones
    3. Apply distributive property
    4. Recursively decompose each sub-multiplication
    5. Sum the results
    
    Returns a string showing the step-by-step breakdown.
    """
    result = a * b
    indent = "  " * depth
    
    # Base cases
    if is_base_case(a, b):
        return f"{a} × {b} = {result}"
    
    # Ensure a <= b for consistency (break down the larger one)
    if a > b:
        a, b = b, a
    
    lines = []
    
    # If a is single digit but b is larger, break down b
    if 2 <= a <= 9 and b >= 10:
        tens = (b // 10) * 10
        ones = b % 10
        
        if ones == 0:
            # Just tens, e.g., 3 × 20 = 3 × 2 × 10 = 6 × 10 = 60
            factor = b // 10
            step1 = a * factor
            lines.append(f"{a} × {b}")
            lines.append(f"= {a} × {factor} × 10")
            if is_base_case(a, factor):
                lines.append(f"= {step1} × 10")
                lines.append(f"= {result}")
            else:
                sub_decomp = decompose_multiplication_recursive(a, factor, depth + 1)
                lines.append(f"= ({sub_decomp}) × 10")
                lines.append(f"= {step1} × 10 = {result}")
        else:
            # Has both tens and ones, e.g., 3 × 17 = 3 × (10 + 7) = 3×10 + 3×7 = 30 + 21 = 51
            part1 = a * tens
            part2 = a * ones
            
            lines.append(f"{a} × {b}")
            lines.append(f"= {a} × ({tens} + {ones})")
            lines.append(f"= {a}×{tens} + {a}×{ones}")
            
            # Check if sub-parts need decomposition
            if is_base_case(a, tens // 10):  # a × tens is really a × (tens/10) × 10
                part1_str = f"{part1}"
            else:
                # Need to decompose further
                part1_str = f"{part1}"
            
            if is_base_case(a, ones):
                part2_str = f"{part2}"
            else:
                # Need to decompose a × ones
                sub_decomp = decompose_multiplication_recursive(a, ones, depth + 1)
                part2_str = f"{part2}"
            
            lines.append(f"= {part1} + {part2}")
            lines.append(f"= {result}")
    
    # Both numbers >= 10, break down the smaller one (a)
    elif a >= 10 and b >= 10:
        a_tens = (a // 10) * 10
        a_ones = a % 10
        
        if a_ones == 0:
            # a is round, e.g., 20 × 15 = 2 × 10 × 15 = 2 × 150 = 300
            factor = a // 10
            lines.append(f"{a} × {b}")
            lines.append(f"= {factor} × 10 × {b}")
            lines.append(f"= {factor} × {b} × 10")
            
            sub_result = factor * b
            if is_base_case(factor, b):
                lines.append(f"= {sub_result} × 10")
            else:
                sub_decomp = decompose_multiplication_recursive(factor, b, depth + 1)
                lines.append(f"[where {sub_decomp}]")
                lines.append(f"= {sub_result} × 10")
            lines.append(f"= {result}")
        else:
            # a has tens and ones, e.g., 12 × 15 = (10 + 2) × 15 = 10×15 + 2×15
            part1 = a_tens * b
            part2 = a_ones * b
            
            lines.append(f"{a} × {b}")
            lines.append(f"= ({a_tens} + {a_ones}) × {b}")
            lines.append(f"= {a_tens}×{b} + {a_ones}×{b}")
            
            # Decompose each part
            sub_parts = []
            
            # Part 1: a_tens × b (e.g., 10 × 15 = 150)
            if a_tens == 10:
                sub_parts.append(f"{a_tens}×{b} = {part1}")
            else:
                factor = a_tens // 10
                sub_parts.append(f"{a_tens}×{b} = {factor}×10×{b} = {factor}×{b}×10 = {factor * b}×10 = {part1}")
            
            # Part 2: a_ones × b (e.g., 2 × 15)
            if is_base_case(a_ones, b):
                sub_parts.append(f"{a_ones}×{b} = {part2}")
            else:
                # Recursively decompose
                sub_decomp = decompose_multiplication_recursive(a_ones, b, depth + 1)
                sub_parts.append(f"{a_ones}×{b}: {sub_decomp}")
            
            for sp in sub_parts:
                lines.append(f"  [{sp}]")
            
            lines.append(f"= {part1} + {part2}")
            lines.append(f"= {result}")
    
    else:
        # Fallback
        lines.append(f"{a} × {b} = {result}")
    
    return '\n'.join(lines)


def format_step_by_step(a: int, b: int) -> str:
    """
    Create a clean, readable step-by-step multiplication breakdown.
    """
    result = a * b
    
    if is_base_case(a, b):
        return f"{a} × {b} = {result}"
    
    decomp = decompose_multiplication_recursive(a, b)
    return decomp


def create_synthetic_example(a: int, b: int, idx: int) -> Dict:
    """
    Create a synthetic arithmetic example.
    """
    result = a * b
    breakdown = format_step_by_step(a, b)
    
    question = f"What is {a} times {b}?"
    answer = f"Let me break this down step by step.\n{breakdown}\nThe answer is {result}.\n#### {result}"
    
    return {
        "id": f"synth_mult_{idx:04d}",
        "question": question,
        "original_answer": f"{a} × {b} = {result}\n#### {result}",
        "sbs_answer": answer,
        "final_answer": str(result),
    }


def generate_synthetic_arithmetic_examples(n: int = 128) -> List[Dict]:
    """
    Generate synthetic examples focused purely on multiplication decomposition.
    These supplement the GSM8K rewrites with clean, focused examples.
    """
    examples = []
    
    # Curated set of interesting multiplications
    curated = [
        (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19),
        (3, 12), (3, 13), (3, 14), (3, 15), (3, 16), (3, 17), (3, 18), (3, 19),
        (4, 12), (4, 13), (4, 14), (4, 15), (4, 16), (4, 17), (4, 18), (4, 19),
        (5, 12), (5, 13), (5, 14), (5, 15), (5, 16), (5, 17), (5, 18), (5, 19),
        (6, 12), (6, 13), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19),
        (7, 12), (7, 13), (7, 14), (7, 15), (7, 16), (7, 17), (7, 18), (7, 19),
        (8, 12), (8, 13), (8, 14), (8, 15), (8, 16), (8, 17), (8, 18), (8, 19),
        (9, 12), (9, 13), (9, 14), (9, 15), (9, 16), (9, 17), (9, 18), (9, 19),
        (12, 12), (12, 13), (12, 14), (12, 15),
        (13, 13), (13, 14), (13, 15),
        (15, 15), (15, 16), (15, 17),
        (11, 11), (11, 12), (11, 13), (11, 14), (11, 15),
        (2, 20), (3, 20), (4, 20), (5, 20), (6, 20), (7, 20), (8, 20), (9, 20),
        (2, 30), (3, 30), (4, 30), (5, 30),
        (2, 25), (3, 25), (4, 25), (5, 25),
        (6, 11), (7, 11), (8, 11), (9, 11),
        (6, 21), (7, 21), (8, 21), (9, 21),
    ]
    
    for i, (a, b) in enumerate(curated[:n]):
        examples.append(create_synthetic_example(a, b, i))
    
    # Fill remaining with random
    while len(examples) < n:
        a = random.randint(2, 9)
        b = random.randint(11, 25)
        examples.append(create_synthetic_example(a, b, len(examples)))
    
    return examples


def rewrite_gsm_solution(question: str, original_answer: str, idx: int) -> Dict:
    """
    Rewrite a GSM8K solution with step-by-step breakdowns for multiplications.
    """
    # Extract final answer
    final_answer = ""
    if "####" in original_answer:
        final_answer = original_answer.split("####")[-1].strip()
    
    # Find multiplications in the answer
    pattern = r'(\d+)\s*\*\s*(\d+)'
    
    def replace_mult(match):
        a = int(match.group(1))
        b = int(match.group(2))
        result = a * b
        
        if is_base_case(a, b):
            return f"{a}×{b}={result}"
        
        # Create inline breakdown
        breakdown = format_step_by_step(a, b)
        # Make it more compact for inline use
        lines = breakdown.split('\n')
        compact = ' → '.join(line.strip() for line in lines if line.strip())
        return f"[{compact}]"
    
    sbs_answer = re.sub(pattern, replace_mult, original_answer)
    
    return {
        "id": f"gsm_sbs_{idx:04d}",
        "question": question,
        "original_answer": original_answer,
        "sbs_answer": sbs_answer,
        "final_answer": final_answer,
    }


def main():
    output_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Generating GSM-SBS Dataset (Recursive Decomposition)")
    print("=" * 60)
    
    # Show example decompositions
    print("\nExample decompositions:")
    test_cases = [(2, 12), (12, 12), (7, 15), (6, 9), (3, 20)]
    for a, b in test_cases:
        print(f"\n{a} × {b}:")
        print(format_step_by_step(a, b))
    
    # Load GSM8K training set
    print("\n" + "=" * 60)
    print("Loading GSM8K training set...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    print(f"  Total samples: {len(dataset)}")
    
    # Select 512 samples
    indices = list(range(400)) + random.sample(range(400, len(dataset)), 112)
    random.shuffle(indices)
    selected_indices = indices[:512]
    
    print(f"  Selected: {len(selected_indices)} samples")
    
    # Generate SBS examples from GSM8K
    print("\nRewriting solutions with step-by-step breakdowns...")
    gsm_examples = []
    for i, idx in enumerate(selected_indices):
        example = dataset[idx]
        sbs_example = rewrite_gsm_solution(
            question=example["question"],
            original_answer=example["answer"],
            idx=i
        )
        gsm_examples.append(sbs_example)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(selected_indices)}")
    
    # Generate synthetic arithmetic examples
    print("\nGenerating synthetic arithmetic examples...")
    synth_examples = generate_synthetic_arithmetic_examples(n=128)
    print(f"  Generated {len(synth_examples)} synthetic examples")
    
    # Combine all examples
    all_examples = gsm_examples + synth_examples
    random.shuffle(all_examples)
    
    print(f"\nTotal examples: {len(all_examples)}")
    
    # Split into shards (32 files)
    n_shards = 32
    shard_size = len(all_examples) // n_shards
    
    print(f"\nWriting {n_shards} shard files...")
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = start + shard_size if shard_idx < n_shards - 1 else len(all_examples)
        shard_examples = all_examples[start:end]
        
        shard_path = output_dir / f"shard_{shard_idx:02d}.json"
        with open(shard_path, "w") as f:
            json.dump(shard_examples, f, indent=2)
    
    # Write metadata
    metadata = {
        "name": "GSM-SBS",
        "description": "GSM8K Step-By-Step with recursive multiplication decomposition",
        "total_examples": len(all_examples),
        "gsm8k_examples": len(gsm_examples),
        "synthetic_examples": len(synth_examples),
        "n_shards": n_shards,
        "created_from": "openai/gsm8k train split",
        "decomposition": "recursive until single-digit × single-digit base cases",
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Write combined file
    with open(output_dir / "all_examples.json", "w") as f:
        json.dump(all_examples, f, indent=2)
    
    print(f"\nFiles written to: {output_dir}")
    
    # Show sample synthetic examples
    print("\n" + "=" * 60)
    print("Sample Synthetic Examples")
    print("=" * 60)
    
    for ex in synth_examples[:3]:
        print(f"\n--- {ex['id']} ---")
        print(f"Q: {ex['question']}")
        print(f"A:\n{ex['sbs_answer']}")
    
    print("\n" + "=" * 60)
    print("GSM-SBS Dataset Generation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    main()
