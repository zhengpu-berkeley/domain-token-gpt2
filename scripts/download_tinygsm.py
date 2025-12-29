#!/usr/bin/env python3
"""
Download TinyGSM dataset and prepare for conversion.

TinyGSM contains 12.3M grade school math problems with Python solutions.
We sample a subset for efficient processing.

Usage:
    python scripts/download_tinygsm.py --output data/tinygsm --sample-size 100000
"""

import argparse
import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re


def extract_question_and_code(example: dict) -> dict:
    """
    Extract question from docstring and code from function body.
    
    TinyGSM format:
    ```python
    def solution():
        \"\"\"Question text here.\"\"\"
        var1 = ...
        ...
        return result
    ```
    """
    code = example.get("code", "")
    
    # Extract question from docstring
    docstring_match = re.search(r'"""(.+?)"""', code, re.DOTALL)
    if docstring_match:
        question = docstring_match.group(1).strip()
    else:
        # Try single quotes
        docstring_match = re.search(r"'''(.+?)'''", code, re.DOTALL)
        if docstring_match:
            question = docstring_match.group(1).strip()
        else:
            question = ""
    
    # Extract code body (everything after docstring, before return)
    # Remove the def solution(): and docstring
    code_body = code
    
    # Remove function definition line
    code_body = re.sub(r'^def solution\(\):\s*\n', '', code_body)
    
    # Remove docstring
    code_body = re.sub(r'^\s*""".*?"""\s*\n', '', code_body, flags=re.DOTALL)
    code_body = re.sub(r"^\s*'''.*?'''\s*\n", '', code_body, flags=re.DOTALL)
    
    # Clean up indentation
    lines = code_body.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove leading 4 spaces (function body indent)
        if line.startswith('    '):
            cleaned_lines.append(line[4:])
        else:
            cleaned_lines.append(line)
    code_body = '\n'.join(cleaned_lines).strip()
    
    # Extract the answer by finding the return statement
    answer_match = re.search(r'return\s+(\w+)', code_body)
    if answer_match:
        var_name = answer_match.group(1)
        # Try to find the variable assignment
        var_match = re.search(rf'{var_name}\s*=\s*(.+)', code_body)
        if var_match:
            # The answer is usually the result of the last calculation
            pass
    
    # Get the answer from the dataset if available
    answer = example.get("answer", "")
    
    return {
        "question": question,
        "code": code_body,
        "full_code": code,
        "answer": str(answer) if answer else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Download and sample TinyGSM dataset")
    parser.add_argument("--output", type=Path, default=Path("data/tinygsm"),
                        help="Output directory")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="Number of examples to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Downloading TinyGSM dataset...")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset("TinyGSM/TinyGSM", split="train")
    print(f"  Total examples: {len(dataset):,}")
    
    # Sample
    random.seed(args.seed)
    if args.sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.sample_size)
        sampled = dataset.select(indices)
        print(f"  Sampled: {len(sampled):,} examples")
    else:
        sampled = dataset
        print(f"  Using all {len(sampled):,} examples")
    
    # Process and save
    output_file = args.output / f"sample_{args.sample_size // 1000}k.jsonl"
    print(f"\nProcessing and saving to {output_file}...")
    
    processed = []
    errors = 0
    
    for example in tqdm(sampled, desc="Processing"):
        try:
            processed_ex = extract_question_and_code(example)
            if processed_ex["question"] and processed_ex["code"]:
                processed.append(processed_ex)
            else:
                errors += 1
        except Exception as e:
            errors += 1
            continue
    
    # Save
    with open(output_file, "w") as f:
        for ex in processed:
            f.write(json.dumps(ex) + "\n")
    
    print(f"\n" + "=" * 60)
    print(f"Download complete!")
    print(f"=" * 60)
    print(f"  Processed: {len(processed):,} examples")
    print(f"  Errors: {errors:,}")
    print(f"  Saved to: {output_file}")
    
    # Show sample
    print(f"\n  Sample example:")
    if processed:
        sample = processed[0]
        print(f"  Question: {sample['question'][:100]}...")
        print(f"  Code: {sample['code'][:100]}...")
        print(f"  Answer: {sample['answer']}")
    
    # Save metadata
    metadata = {
        "total_original": len(dataset),
        "sample_size": args.sample_size,
        "processed_count": len(processed),
        "error_count": errors,
        "seed": args.seed,
    }
    
    with open(args.output / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

