#!/usr/bin/env python3
"""
Intuition check: sample from a list of checkpoints to quickly spot:
- EOS / non-stopping / repetition issues
- arithmetic format drift (e.g., `####` loops)
- whether mul-tokens appear (in mul_tokens condition)

This is intentionally lightweight (no metrics), and will skip checkpoints that don't exist.
"""

import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Test questions - simple to complex
TEST_QUESTIONS = [
    # Simple arithmetic
    "What is 7 times 8?",
    
    # Two-step arithmetic  
    "What is 6 times 9, then add 10?",
    
    # Simple word problem
    "If I have 5 apples and buy 3 more, how many apples do I have?",
    
    # GSM8K-style problem
    "Janet has 10 eggs. She uses 3 for breakfast and 2 for baking. How many eggs does she have left?",
]

# Checkpoints to test (in rough order of progression).
# Keep this list short; add/remove as experiments evolve.
CHECKPOINTS = [
    # 10B-pretrained HF exports (if present locally)
    ("outputs/hf_baseline_10b", "baseline", "Pretrained (10B)"),
    ("outputs/hf_mul_tokens_10b", "mul_tokens", "Pretrained (10B)"),

    # Instruction SFT
    ("outputs/sft_tulu_baseline", "baseline", "Tulu SFT"),
    ("outputs/sft_tulu_mul_tokens", "mul_tokens", "Tulu SFT"),

    # Transition bridge (instruction → TinyGSM style)
    ("outputs/transition_baseline", "baseline", "Transition SFT"),
    ("outputs/transition_mul_tokens", "mul_tokens", "Transition SFT"),

    # TinyGSM 100K (latest pipeline)
    ("outputs/tinygsm_100k_baseline_v3", "baseline", "TinyGSM 100K"),
    ("outputs/tinygsm_100k_mul_tokens_v3", "mul_tokens", "TinyGSM 100K"),
]


def generate_response(model, tokenizer, question, device="cuda", max_new_tokens=150):
    """Generate a single response."""
    prompt = f"User: {question}\nAssistant:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for consistency
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    # Truncate at EOS or User: marker
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0]
    if "\nUser:" in response:
        response = response.split("\nUser:")[0]
    
    return response.strip()


def test_checkpoint(path, condition, stage_name, questions, device="cuda"):
    """Test a single checkpoint with all questions."""
    path = Path(path)
    
    if not path.exists():
        print(f"  [SKIP] {path} does not exist")
        return None
    
    # Load model
    try:
        model = GPT2LMHeadModel.from_pretrained(path, torch_dtype=torch.bfloat16)
        tokenizer = GPT2TokenizerFast.from_pretrained(path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"  [ERROR] Failed to load: {e}")
        return None
    
    results = []
    for q in questions:
        try:
            response = generate_response(model, tokenizer, q, device)
            results.append((q, response))
        except Exception as e:
            results.append((q, f"[ERROR: {e}]"))
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    all_results = {}
    
    for path, condition, stage_name in CHECKPOINTS:
        header = f"{stage_name} [{condition}]"
        print("=" * 70)
        print(f"Testing: {header}")
        print(f"Path: {path}")
        print("=" * 70)
        
        results = test_checkpoint(path, condition, stage_name, TEST_QUESTIONS, device)
        
        if results is None:
            continue
        
        all_results[(path, condition, stage_name)] = results
        
        for q, response in results:
            print(f"\nQ: {q}")
            # Show first 200 chars of response
            display = response[:200] + "..." if len(response) > 200 else response
            print(f"A: {display}")
        
        print()
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Response Quality Check")
    print("=" * 70)
    print("\nLegend:")
    print("  ✓ = Reasonable response")
    print("  ✗ = Repetitive/broken")
    print("  ? = Partially correct")
    print("  - = Not tested")
    print()
    
    # Check for issues
    print("Key observations:")
    for (path, condition, stage_name), results in all_results.items():
        issues = []
        for q, response in results:
            # Check for repetition
            if response.count("####") > 2:
                issues.append("repetitive ####")
            if len(set(response.split())) < 10 and len(response) > 100:
                issues.append("low variety")
            if not response or response.startswith("[ERROR"):
                issues.append("error/empty")
        
        status = "OK" if not issues else ", ".join(set(issues))
        print(f"  {stage_name:30} [{condition:10}]: {status}")


if __name__ == "__main__":
    main()

