#!/usr/bin/env python3
"""
Convert TinyGSM Python solutions to step-by-step natural language answers.

Uses GPT-5-nano with structured output for efficient conversion.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI

from prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, FEW_SHOT_EXAMPLES


@dataclass
class ConversionResult:
    """Result of converting a single example."""
    question: str
    code: str
    steps: list[str]
    final_answer: str
    sbs_answer: str  # Full step-by-step answer as text
    sbs_answer_with_mul_tokens: str  # With mul-tokens injected
    success: bool
    error: Optional[str] = None


def inject_mul_tokens(text: str) -> str:
    """
    Inject mul-tokens into single-digit multiplication expressions.
    
    Examples:
        "7 × 8 = 56" -> "<MUL_7_8_56>"
        "6 × 9 = 54" -> "<MUL_6_9_54>"
        "3 * 5 = 15" -> "<MUL_3_5_15>"
        "(3 × 6)" -> "<MUL_3_6_18>" (when we can infer the result)
    """
    result_text = text
    
    # Pattern 1: a × b = c (explicit result)
    # Matches: "7 × 8 = 56", "7×8=56", "7 * 8 = 15", etc.
    pattern1 = r'(\d)\s*[×x\*]\s*(\d)\s*=\s*(\d+)'
    
    def replace_with_result(match):
        a, b, result = match.groups()
        a, b = int(a), int(b)
        # Skip trivial cases: 0 or 1
        if a == 0 or b == 0 or a == 1 or b == 1:
            return match.group(0)
        # Canonicalize: smaller number first
        if a > b:
            a, b = b, a
        expected = a * b
        # Only replace if the result matches
        if int(result) == expected:
            return f"<MUL_{a}_{b}_{expected}>"
        else:
            return match.group(0)
    
    result_text = re.sub(pattern1, replace_with_result, result_text)
    
    # Pattern 2: (a × b) in parentheses - infer the result
    # Matches: "(3 × 6)", "(4 × 5)", etc.
    pattern2 = r'\((\d)\s*[×x\*]\s*(\d)\)'
    
    def replace_infer_result(match):
        a, b = int(match.group(1)), int(match.group(2))
        # Skip trivial cases: 0 or 1
        if a == 0 or b == 0 or a == 1 or b == 1:
            return match.group(0)
        # Canonicalize: smaller number first
        if a > b:
            a, b = b, a
        result = a * b
        return f"<MUL_{a}_{b}_{result}>"
    
    result_text = re.sub(pattern2, replace_infer_result, result_text)
    
    return result_text


def convert_example(
    client: OpenAI,
    question: str,
    code: str,
    model: str = "gpt-5-nano",
) -> ConversionResult:
    """
    Convert a single Python solution to step-by-step answer.
    """
    try:
        # Format the user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question,
            code=code
        )
        
        # Call the API with structured output
        response = client.responses.create(
            model=model,
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
                            "steps": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "final_answer": {
                                "type": "string"
                            }
                        },
                        "required": ["steps", "final_answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            reasoning={
                "effort": "low"
            },
            store=True,
        )
        
        # Parse the response
        output_text = response.output_text
        result = json.loads(output_text)
        
        steps = result.get("steps", [])
        final_answer = result.get("final_answer", "")
        
        # Build the step-by-step answer text
        sbs_answer = "\n".join(steps) + f"\n#### {final_answer}"
        
        # Inject mul-tokens
        sbs_answer_with_mul_tokens = inject_mul_tokens(sbs_answer)
        
        return ConversionResult(
            question=question,
            code=code,
            steps=steps,
            final_answer=final_answer,
            sbs_answer=sbs_answer,
            sbs_answer_with_mul_tokens=sbs_answer_with_mul_tokens,
            success=True,
        )
        
    except Exception as e:
        return ConversionResult(
            question=question,
            code=code,
            steps=[],
            final_answer="",
            sbs_answer="",
            sbs_answer_with_mul_tokens="",
            success=False,
            error=str(e),
        )


def convert_batch(
    input_file: Path,
    output_file: Path,
    api_key: str,
    model: str = "gpt-5-nano",
    max_samples: Optional[int] = None,
    skip_existing: bool = True,
):
    """
    Convert a batch of examples from JSONL file.
    """
    client = OpenAI(api_key=api_key)
    
    # Load input
    examples = []
    with open(input_file) as f:
        for line in f:
            examples.append(json.loads(line))
    
    if max_samples:
        examples = examples[:max_samples]
    
    print(f"Converting {len(examples)} examples with {model}...")
    
    # Load existing results if resuming
    existing = set()
    if skip_existing and output_file.exists():
        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                existing.add(data.get("question", "")[:100])
        print(f"  Skipping {len(existing)} already converted examples")
    
    # Convert
    results = []
    errors = 0
    
    with open(output_file, "a") as f_out:
        for i, ex in enumerate(examples):
            question = ex.get("question", "")
            code = ex.get("code", "")
            
            # Skip if already done
            if question[:100] in existing:
                continue
            
            result = convert_example(client, question, code, model)
            
            if result.success:
                output = {
                    "question": result.question,
                    "answer": result.sbs_answer,
                    "answer_with_mul_tokens": result.sbs_answer_with_mul_tokens,
                    "final_answer": result.final_answer,
                    "steps": result.steps,
                }
                f_out.write(json.dumps(output) + "\n")
                f_out.flush()
                results.append(result)
            else:
                errors += 1
                print(f"  Error on example {i}: {result.error}")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(examples)} ({errors} errors)")
    
    print(f"\nConversion complete: {len(results)} success, {errors} errors")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert TinyGSM to SBS answers")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-5-nano", help="Model to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to convert")
    
    args = parser.parse_args()
    
    convert_batch(
        input_file=args.input,
        output_file=args.output,
        api_key=args.api_key,
        model=args.model,
        max_samples=args.max_samples,
    )

