#!/usr/bin/env python3
"""
Async smoke test for TinyGSM conversion pipeline.

Tests GPT-5-nano conversion on examples and SAVES results for inspection.
Iterates on prompt quality until mul-token usage is satisfactory.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEW_SHOT_EXAMPLES
from converter import inject_mul_tokens


# API key (load from environment / .env)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in environment or in .env.")

# Output file for inspection
OUTPUT_DIR = Path(__file__).parent / "smoke_test_results"


def clean_json_output(text: str) -> dict:
    """Clean and parse JSON from potentially malformed output."""
    # Remove any trailing garbage after the JSON
    text = text.strip()
    
    # Find the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    
    if start == -1 or end == -1:
        return None
        
    json_str = text[start:end+1]
    
    try:
        return json.loads(json_str)
    except:
        return None


async def convert_single(
    client: AsyncOpenAI,
    question: str,
    code: str,
    idx: int,
    use_few_shot: bool = True,
) -> dict:
    """Convert a single example asynchronously."""
    try:
        user_prompt = USER_PROMPT_TEMPLATE.format(question=question, code=code)
        
        # Add few-shot examples if enabled
        if use_few_shot:
            full_prompt = FEW_SHOT_EXAMPLES + "\n\nNow convert this:\n" + user_prompt
        else:
            full_prompt = user_prompt
        
        response = await client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
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
        output_text = response.output_text
        result = clean_json_output(output_text)
        
        if result is None:
            # Try direct JSON parse as fallback
            try:
                result = json.loads(output_text)
            except:
                return {
                    "idx": idx,
                    "success": False,
                    "question": question,
                    "code": code,
                    "error": f"Failed to parse JSON: {output_text[:200]}",
                }
        
        steps = result.get("steps", [])
        final_answer = result.get("final_answer", "")
        
        # Build answer text (clean version)
        sbs_answer = "\n".join(steps) + f"\n#### {final_answer}"
        sbs_with_mul = inject_mul_tokens(sbs_answer)
        
        # Count quality metrics
        mul_token_count = sbs_with_mul.count("<MUL_")
        
        # Count explicit multiplications in format "a × b = c" or "a * b = c"
        mult_pattern = r'\b(\d)\s*[×\*]\s*(\d)\s*=\s*(\d+)'
        explicit_mults = len(re.findall(mult_pattern, sbs_answer))
        
        return {
            "idx": idx,
            "success": True,
            "question": question,
            "code": code[:300],  # Truncate for readability
            "steps": steps,
            "final_answer": final_answer,
            "sbs_answer": sbs_answer,
            "sbs_with_mul_tokens": sbs_with_mul,
            "mul_token_count": mul_token_count,
            "explicit_multiplications": explicit_mults,
        }
        
    except Exception as e:
        return {
            "idx": idx,
            "success": False,
            "question": question,
            "code": code[:300],
            "error": str(e),
        }


async def run_smoke_test(n_examples: int = 10, use_few_shot: bool = True):
    """Run async smoke test on n examples."""
    print("=" * 70)
    print("ASYNC SMOKE TEST - TinyGSM Conversion (v2: refined prompt)")
    print(f"Few-shot examples: {'enabled' if use_few_shot else 'disabled'}")
    print("=" * 70)
    
    # Load examples
    data_file = Path(__file__).parent / "sample_100k.jsonl"
    examples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= n_examples:
                break
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} examples")
    
    # Create async client
    client = AsyncOpenAI(api_key=API_KEY)
    
    # Run all conversions in parallel
    print(f"Converting {len(examples)} examples in parallel...")
    start_time = datetime.now()
    
    tasks = [
        convert_single(client, ex["question"], ex["code"], i, use_few_shot)
        for i, ex in enumerate(examples)
    ]
    
    results = await asyncio.gather(*tasks)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Completed in {elapsed:.1f} seconds")
    
    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"smoke_test_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Also save latest for easy access
    latest_file = OUTPUT_DIR / "latest.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Latest saved to: {latest_file}")
    
    # Analyze results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nErrors:")
        for r in failed:
            print(f"  - Example {r['idx']}: {r.get('error', 'Unknown')[:100]}")
    
    if successful:
        total_mul = sum(r["mul_token_count"] for r in successful)
        avg_mul = total_mul / len(successful)
        total_explicit = sum(r.get("explicit_multiplications", 0) for r in successful)
        avg_explicit = total_explicit / len(successful)
        
        print(f"\n📊 QUALITY METRICS:")
        print(f"  Total mul-tokens: {total_mul}")
        print(f"  Avg mul-tokens per example: {avg_mul:.2f}")
        print(f"  Total explicit multiplications: {total_explicit}")
        print(f"  Avg explicit mults per example: {avg_explicit:.2f}")
        
        has_mul = sum(1 for r in successful if r["mul_token_count"] > 0)
        print(f"  Examples with ≥1 mul-token: {has_mul}/{len(successful)} ({100*has_mul/len(successful):.0f}%)")
        
        # Quality threshold check
        if avg_mul < 2.0:
            print(f"\n⚠️  LOW MUL-TOKEN USAGE ({avg_mul:.2f} < 2.0)")
            print("   Prompt needs refinement!")
        else:
            print(f"\n✅ MUL-TOKEN USAGE ACCEPTABLE ({avg_mul:.2f})")
        
        # Show detailed results
        print("\n" + "=" * 70)
        print("SAMPLE OUTPUTS (for intuition check)")
        print("=" * 70)
        
        for r in successful[:5]:  # Show first 5
            print(f"\n{'─' * 60}")
            print(f"EXAMPLE {r['idx'] + 1}")
            print(f"{'─' * 60}")
            print(f"Q: {r['question'][:120]}...")
            print(f"\n📝 STEPS:")
            for i, step in enumerate(r['steps'][:6], 1):  # Show first 6 steps
                print(f"   {i}. {step}")
            if len(r['steps']) > 6:
                print(f"   ... ({len(r['steps']) - 6} more steps)")
            print(f"\n🎯 Answer: {r['final_answer']}")
            print(f"🔢 Mul-tokens: {r['mul_token_count']}")
            
            if r['mul_token_count'] > 0:
                # Show where mul-tokens appear
                print(f"\n   WITH MUL-TOKENS:")
                for line in r['sbs_with_mul_tokens'].split('\n'):
                    if '<MUL_' in line:
                        print(f"   ► {line}")
    
    return results


def main():
    """Entry point."""
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    results = asyncio.run(run_smoke_test(n, use_few_shot=True))
    return results


if __name__ == "__main__":
    main()
