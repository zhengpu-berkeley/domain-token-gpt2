#!/usr/bin/env python3
"""
Prepare text data for pretraining.

Generates a tiny synthetic corpus with multiplication facts for smoke testing.
For production, this would download/process real datasets.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.inject_mul import create_injector


def generate_multiplication_drills(num_samples: int = 1000, seed: int = 42) -> List[str]:
    """
    Generate synthetic multiplication drill sentences.
    
    These contain multiplication facts in various formats that the injector can detect.
    """
    random.seed(seed)
    
    templates = [
        "What is {a}*{b}? The answer is {c}.",
        "{a} times {b} equals {c}.",
        "Calculate {a}*{b}={c}.",
        "The product of {a} and {b} is {c}.",
        "{a} × {b} = {c}",
        "Multiply {a} by {b} to get {c}.",
        "If you multiply {a} and {b}, you get {c}.",
        "{a}*{b} is equal to {c}.",
        "The result of {a} x {b} is {c}.",
        "{a} multiplied by {b} gives {c}.",
    ]
    
    samples = []
    for _ in range(num_samples):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        c = a * b
        template = random.choice(templates)
        samples.append(template.format(a=a, b=b, c=c))
    
    return samples


def generate_word_problems(num_samples: int = 500, seed: int = 42) -> List[str]:
    """
    Generate simple word problems involving multiplication.
    """
    random.seed(seed + 1)
    
    templates = [
        "Sarah has {a} baskets. Each basket has {b} apples. How many apples does she have in total? Sarah has {a}*{b}={c} apples.",
        "A farmer has {a} rows of trees. Each row has {b} trees. The total number of trees is {a}*{b}={c}.",
        "There are {a} boxes. Each box contains {b} pencils. The total is {c} pencils because {a}*{b}={c}.",
        "If a book costs ${b} and you buy {a} books, you pay ${c} total since {a}*{b}={c}.",
        "A rectangle has length {a} and width {b}. Its area is {a}*{b}={c}.",
    ]
    
    samples = []
    for _ in range(num_samples):
        a = random.randint(2, 9)
        b = random.randint(2, 9)
        c = a * b
        template = random.choice(templates)
        samples.append(template.format(a=a, b=b, c=c))
    
    return samples


def generate_multiplication_tables(include_all: bool = True) -> List[str]:
    """
    Generate multiplication table in various formats.
    """
    samples = []
    
    # Standard table format
    samples.append("Multiplication Table (1-9):")
    for a in range(1, 10):
        row = " | ".join(f"{a}*{b}={a*b}" for b in range(1, 10))
        samples.append(row)
    
    # Natural language format
    samples.append("\nMultiplication facts in words:")
    for a in range(1, 10):
        for b in range(a, 10):  # Only upper triangle to avoid duplicates
            c = a * b
            samples.append(f"{a} times {b} is {c}")
    
    return samples


def generate_filler_text(num_samples: int = 200, seed: int = 42) -> List[str]:
    """
    Generate filler text without multiplication (to ensure model doesn't overfit).
    """
    random.seed(seed + 2)
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process information.",
        "Python is a popular programming language for data science.",
        "The sun rises in the east and sets in the west.",
        "Reading books helps expand your knowledge and vocabulary.",
        "Exercise is important for maintaining good health.",
        "Music can evoke powerful emotions and memories.",
        "The ocean covers about 71% of the Earth's surface.",
        "Cooking at home is often healthier than eating out.",
        "Learning a new skill requires patience and practice.",
        "The human brain contains approximately 86 billion neurons.",
        "Trees absorb carbon dioxide and produce oxygen.",
        "Communication is key to successful relationships.",
        "History teaches us valuable lessons about the past.",
        "Art can express ideas that words cannot capture.",
    ]
    
    samples = []
    for _ in range(num_samples):
        # Pick 1-3 random sentences
        num_sents = random.randint(1, 3)
        selected = random.sample(sentences, min(num_sents, len(sentences)))
        samples.append(" ".join(selected))
    
    return samples


def prepare_corpus(
    output_path: Path,
    condition: str = "baseline",
    num_drills: int = 1000,
    num_problems: int = 500,
    num_filler: int = 200,
    seed: int = 42,
) -> dict:
    """
    Prepare the full corpus for a given condition.
    
    Args:
        output_path: Where to write the output text file
        condition: "baseline" or "mul_tokens"
        num_drills: Number of multiplication drill sentences
        num_problems: Number of word problems
        num_filler: Number of filler sentences
        seed: Random seed
        
    Returns:
        Dict with statistics about the generated corpus
    """
    print(f"Preparing corpus for condition: {condition}")
    
    # Generate all samples
    all_samples = []
    
    print(f"  Generating {num_drills} multiplication drills...")
    drills = generate_multiplication_drills(num_drills, seed)
    all_samples.extend(drills)
    
    print(f"  Generating {num_problems} word problems...")
    problems = generate_word_problems(num_problems, seed)
    all_samples.extend(problems)
    
    print("  Generating multiplication tables...")
    tables = generate_multiplication_tables()
    all_samples.extend(tables)
    
    print(f"  Generating {num_filler} filler sentences...")
    filler = generate_filler_text(num_filler, seed)
    all_samples.extend(filler)
    
    # Shuffle
    random.seed(seed + 100)
    random.shuffle(all_samples)
    
    # Optionally inject mul tokens
    if condition == "mul_tokens":
        print("  Injecting mul-fact tokens...")
        injector = create_injector("weak")
        all_samples = injector.inject_batch(all_samples)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(all_samples)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    # Count mul tokens in output (for mul_tokens condition)
    mul_token_count = text.count("<MUL_")
    
    stats = {
        "condition": condition,
        "num_drills": num_drills,
        "num_problems": num_problems,
        "num_tables": len(tables),
        "num_filler": num_filler,
        "total_samples": len(all_samples),
        "total_chars": len(text),
        "mul_token_count": mul_token_count,
        "output_path": str(output_path),
    }
    
    print(f"  Wrote {len(all_samples)} samples ({len(text):,} chars) to {output_path}")
    print(f"  Mul-token occurrences: {mul_token_count}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare text data for pretraining")
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        default="baseline",
        help="Experimental condition: baseline or mul_tokens",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "raw",
        help="Output directory for text files",
    )
    parser.add_argument(
        "--num-drills",
        type=int,
        default=1000,
        help="Number of multiplication drill sentences",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=500,
        help="Number of word problems",
    )
    parser.add_argument(
        "--num-filler",
        type=int,
        default=200,
        help="Number of filler sentences",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    output_path = args.output_dir / f"train_{args.condition}.txt"
    
    stats = prepare_corpus(
        output_path=output_path,
        condition=args.condition,
        num_drills=args.num_drills,
        num_problems=args.num_problems,
        num_filler=args.num_filler,
        seed=args.seed,
    )
    
    # Save stats
    stats_path = args.output_dir / f"train_{args.condition}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats to {stats_path}")


if __name__ == "__main__":
    main()

