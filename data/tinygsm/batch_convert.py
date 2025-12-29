#!/usr/bin/env python3
"""
Batch convert TinyGSM Python solutions to natural language CoT.

Uses async with semaphore for concurrency control and tqdm for progress.
"""

import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEW_SHOT_EXAMPLES
from converter import inject_mul_tokens

# Load API key from .env
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")


async def convert_single(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    question: str,
    code: str,
    idx: int,
) -> dict:
    """Convert a single example with semaphore for concurrency control."""
    async with semaphore:
        try:
            user_prompt = FEW_SHOT_EXAMPLES + "\n\nNow convert:\n" + USER_PROMPT_TEMPLATE.format(
                question=question, code=code
            )
            
            response = await client.responses.create(
                model="gpt-5-nano",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "step_by_step_answer",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "steps": {"type": "array", "items": {"type": "string"}},
                                "final_answer": {"type": "string"}
                            },
                            "required": ["steps", "final_answer"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                reasoning={"effort": "low"},
                store=True,
            )
            
            # Parse output
            result = json.loads(response.output_text)
            steps = result.get("steps", [])
            final_answer = result.get("final_answer", "")
            
            # Build answer text
            sbs_answer = "\n".join(steps) + f"\n#### {final_answer}"
            sbs_with_mul = inject_mul_tokens(sbs_answer)
            
            return {
                "idx": idx,
                "success": True,
                "question": question,
                "answer": sbs_answer,
                "answer_with_mul_tokens": sbs_with_mul,
                "final_answer": final_answer,
                "mul_token_count": sbs_with_mul.count("<MUL_"),
            }
            
        except Exception as e:
            return {
                "idx": idx,
                "success": False,
                "question": question,
                "error": str(e)[:200],
            }


async def batch_convert(
    input_file: Path,
    output_file: Path,
    max_samples: int,
    max_concurrent: int = 512,
):
    """Convert a batch of examples with concurrency control."""
    print("=" * 70)
    print(f"BATCH CONVERT: TinyGSM → Natural Language CoT")
    print(f"Max samples: {max_samples}")
    print(f"Max concurrent: {max_concurrent}")
    print("=" * 70)
    
    # Load examples
    examples = []
    with open(input_file) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples from {input_file}")
    
    # Create client and semaphore
    client = AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks
    tasks = [
        convert_single(client, semaphore, ex["question"], ex["code"], i)
        for i, ex in enumerate(examples)
    ]
    
    # Run with progress bar
    print(f"\nConverting with {max_concurrent} concurrent requests...")
    start_time = datetime.now()
    
    results = await tqdm_asyncio.gather(*tasks, desc="Converting")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n{'=' * 70}")
    print(f"CONVERSION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Time: {elapsed:.1f}s ({len(examples)/elapsed:.1f} examples/sec)")
    print(f"Success: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_mul = sum(r["mul_token_count"] for r in successful)
        avg_mul = total_mul / len(successful)
        has_mul = sum(1 for r in successful if r["mul_token_count"] > 0)
        print(f"\nMul-token stats:")
        print(f"  Total: {total_mul}")
        print(f"  Avg per example: {avg_mul:.2f}")
        print(f"  Examples with ≥1: {has_mul}/{len(successful)} ({100*has_mul/len(successful):.0f}%)")
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save baseline (without mul-tokens)
    baseline_file = output_file.parent / f"{output_file.stem}_baseline.jsonl"
    with open(baseline_file, "w") as f:
        for r in successful:
            out = {
                "question": r["question"],
                "answer": r["answer"],
                "final_answer": r["final_answer"],
            }
            f.write(json.dumps(out) + "\n")
    print(f"\nSaved baseline: {baseline_file}")
    
    # Save mul_tokens variant
    mul_file = output_file.parent / f"{output_file.stem}_mul_tokens.jsonl"
    with open(mul_file, "w") as f:
        for r in successful:
            out = {
                "question": r["question"],
                "answer": r["answer_with_mul_tokens"],
                "final_answer": r["final_answer"],
            }
            f.write(json.dumps(out) + "\n")
    print(f"Saved mul_tokens: {mul_file}")
    
    # Save full results with metadata
    full_file = output_file.parent / f"{output_file.stem}_full.json"
    with open(full_file, "w") as f:
        json.dump({
            "metadata": {
                "input_file": str(input_file),
                "max_samples": max_samples,
                "max_concurrent": max_concurrent,
                "elapsed_seconds": elapsed,
                "success_count": len(successful),
                "failed_count": len(failed),
                "total_mul_tokens": total_mul if successful else 0,
            },
            "results": results,
        }, f, indent=2)
    print(f"Saved full results: {full_file}")
    
    # Log failed examples if any
    if failed:
        print(f"\nFailed examples:")
        for r in failed[:5]:
            print(f"  - {r['idx']}: {r.get('error', 'Unknown')[:80]}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description="Batch convert TinyGSM to CoT")
    parser.add_argument("--input", type=Path, default=Path("sample_100k.jsonl"),
                        help="Input JSONL file")
    parser.add_argument("--output", type=Path, default=Path("converted/shard_10k"),
                        help="Output path prefix")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max samples to convert")
    parser.add_argument("--max-concurrent", type=int, default=512,
                        help="Max concurrent API requests")
    
    args = parser.parse_args()
    
    asyncio.run(batch_convert(
        input_file=args.input,
        output_file=args.output,
        max_samples=args.max_samples,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()

