#!/usr/bin/env python3
"""
Convert all 10 shards of TinyGSM (100K total).

Shard 1 (0-10K) already exists.
This script converts shards 2-10 (10K-100K).
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


async def convert_shard(
    client: AsyncOpenAI,
    examples: list,
    shard_num: int,
    output_dir: Path,
    max_concurrent: int = 512,
):
    """Convert a single shard."""
    print(f"\n{'='*70}")
    print(f"SHARD {shard_num}: Converting {len(examples)} examples")
    print(f"{'='*70}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        convert_single(client, semaphore, ex["question"], ex["code"], i)
        for i, ex in enumerate(examples)
    ]
    
    start_time = datetime.now()
    results = await tqdm_asyncio.gather(*tasks, desc=f"Shard {shard_num}")
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Analyze
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"Time: {elapsed:.1f}s ({len(examples)/elapsed:.1f} ex/sec)")
    print(f"Success: {len(successful)}/{len(results)}")
    
    # Save shard files
    shard_prefix = f"shard_{shard_num}"
    
    baseline_file = output_dir / f"{shard_prefix}_baseline.jsonl"
    with open(baseline_file, "w") as f:
        for r in successful:
            out = {"question": r["question"], "answer": r["answer"], "final_answer": r["final_answer"]}
            f.write(json.dumps(out) + "\n")
    
    mul_file = output_dir / f"{shard_prefix}_mul_tokens.jsonl"
    with open(mul_file, "w") as f:
        for r in successful:
            out = {"question": r["question"], "answer": r["answer_with_mul_tokens"], "final_answer": r["final_answer"]}
            f.write(json.dumps(out) + "\n")
    
    print(f"Saved: {baseline_file}, {mul_file}")
    
    return successful, failed


async def main():
    parser = argparse.ArgumentParser(description="Convert all TinyGSM shards")
    parser.add_argument("--input", type=Path, default=Path("sample_100k.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("converted"))
    parser.add_argument("--shard-size", type=int, default=10000)
    parser.add_argument("--start-shard", type=int, default=2, help="Start from shard N (1-indexed)")
    parser.add_argument("--end-shard", type=int, default=10, help="End at shard N (inclusive)")
    parser.add_argument("--max-concurrent", type=int, default=512)
    
    args = parser.parse_args()
    
    # Load all examples
    print(f"Loading examples from {args.input}...")
    all_examples = []
    with open(args.input) as f:
        for line in f:
            all_examples.append(json.loads(line))
    print(f"Loaded {len(all_examples)} examples")
    
    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create client (reuse for all shards)
    client = AsyncOpenAI(api_key=API_KEY)
    
    total_success = 0
    total_failed = 0
    
    # Convert each shard
    for shard_num in range(args.start_shard, args.end_shard + 1):
        start_idx = (shard_num - 1) * args.shard_size
        end_idx = shard_num * args.shard_size
        
        if start_idx >= len(all_examples):
            print(f"Shard {shard_num}: No data (start_idx={start_idx} >= {len(all_examples)})")
            break
        
        shard_examples = all_examples[start_idx:end_idx]
        successful, failed = await convert_shard(
            client, shard_examples, shard_num, args.output_dir, args.max_concurrent
        )
        
        total_success += len(successful)
        total_failed += len(failed)
    
    print(f"\n{'='*70}")
    print(f"ALL SHARDS COMPLETE")
    print(f"Total success: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"{'='*70}")
    
    # Now concatenate all shards into combined files
    print("\nConcatenating all shards...")
    
    for variant in ["baseline", "mul_tokens"]:
        combined_file = args.output_dir / f"combined_100k_{variant}.jsonl"
        with open(combined_file, "w") as outf:
            for shard_num in range(1, 11):
                shard_file = args.output_dir / f"shard_{shard_num}_{variant}.jsonl"
                if shard_file.exists():
                    with open(shard_file) as inf:
                        outf.write(inf.read())
                else:
                    # Handle shard 1 special case (old naming)
                    old_file = args.output_dir / f"shard_10k_{variant}.jsonl"
                    if old_file.exists() and shard_num == 1:
                        with open(old_file) as inf:
                            outf.write(inf.read())
        
        line_count = sum(1 for _ in open(combined_file))
        print(f"Created {combined_file} ({line_count} lines)")


if __name__ == "__main__":
    asyncio.run(main())

