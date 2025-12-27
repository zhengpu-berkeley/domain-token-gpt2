#!/usr/bin/env python3
"""
Text completion script for sampling from trained models.

Supports multiple sampling strategies and can load any HuggingFace model
(pretrained, SFT, or RL fine-tuned).

Usage:
    uv run python sample/text_completion.py \
        --model-path outputs/hf_baseline_10b \
        --prompt "A cat is a small animal that likes to sleep. It sleeps 15 hours a day. So in a month, it will" \
        --num-samples 5 \
        --temperature 0.8

    # Or use the default sentence from the file:
    uv run python sample/text_completion.py \
        --model-path outputs/sft_mul_tokens_10b \
        --num-samples 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Default sentence from the file
DEFAULT_SENTENCE = """
A cat is a small animal that likes to sleep. It sleeps 15 hours a day. So in a month, it will
"""


def load_model_and_tokenizer(model_path: Path, device: str = "cuda") -> tuple:
    """
    Load model and tokenizer from a HuggingFace model directory.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    model.to(device)
    model.eval()
    
    print(f"  Device: {device}")
    print(f"  Parameters: {model.num_parameters():,}")
    print(f"  Vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> str:
    """
    Generate a single completion from the model.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2TokenizerFast instance
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (keep only top k tokens)
        top_p: Nucleus sampling (keep tokens with cumulative probability <= top_p)
        do_sample: Whether to use sampling (False = greedy)
        device: Device to run on
        seed: Random seed for reproducibility
        
    Returns:
        Generated text completion
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated, skip_special_tokens=True)
    
    return completion


def generate_multiple_completions(
    model,
    tokenizer,
    prompt: str,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    device: str = "cuda",
    base_seed: Optional[int] = None,
) -> List[str]:
    """
    Generate multiple completions with different random seeds.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2TokenizerFast instance
        prompt: Input text prompt
        num_samples: Number of completions to generate
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        do_sample: Whether to use sampling
        device: Device to run on
        base_seed: Base random seed (each sample uses base_seed + sample_idx)
        
    Returns:
        List of generated completions
    """
    completions = []
    
    for i in range(num_samples):
        seed = (base_seed + i) if base_seed is not None else None
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            seed=seed,
        )
        completions.append(completion)
    
    return completions


def print_completions(
    prompt: str,
    completions: List[str],
    show_prompt: bool = True,
    separator: str = "─" * 60,
):
    """
    Pretty print prompt and completions.
    
    Args:
        prompt: Original prompt
        completions: List of completions
        show_prompt: Whether to show the prompt
        separator: Separator string between samples
    """
    if show_prompt:
        print("\n" + "=" * 60)
        print("PROMPT:")
        print("=" * 60)
        print(prompt.strip())
        print()
    
    print("=" * 60)
    print(f"COMPLETIONS ({len(completions)} samples):")
    print("=" * 60)
    
    for i, completion in enumerate(completions, 1):
        print(f"\n[Sample {i}]")
        print(separator)
        print(completion.strip())
        if i < len(completions):
            print()
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text completions from trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 samples with default sentence:
  uv run python sample/text_completion.py --model-path outputs/hf_baseline_10b --num-samples 5

  # Generate with custom prompt and temperature:
  uv run python sample/text_completion.py \\
      --model-path outputs/sft_mul_tokens_10b \\
      --prompt "What is 6 times 9?" \\
      --num-samples 10 \\
      --temperature 0.8

  # Greedy decoding (deterministic):
  uv run python sample/text_completion.py \\
      --model-path outputs/hf_baseline_10b \\
      --num-samples 1 \\
      --greedy

  # Top-k sampling:
  uv run python sample/text_completion.py \\
      --model-path outputs/sft_mul_tokens_10b \\
      --num-samples 5 \\
      --top-k 50 \\
      --temperature 0.9
        """,
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to HuggingFace model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt (default: uses sentence from file)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of completions to generate (default: 5)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling: keep only top k tokens (default: None)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling: cumulative probability threshold (default: None)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (deterministic, ignores temperature/top-k/top-p)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None, random)",
    )
    parser.add_argument(
        "--hide-prompt",
        action="store_true",
        help="Hide the prompt in output (only show completions)",
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Get prompt
    if args.prompt is None:
        prompt = DEFAULT_SENTENCE.strip()
    else:
        prompt = args.prompt.strip()
    
    # Determine sampling mode
    do_sample = not args.greedy
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Generate completions
    print(f"\nGenerating {args.num_samples} completion(s)...")
    print(f"  Sampling: {'Greedy' if not do_sample else 'Stochastic'}")
    if do_sample:
        print(f"  Temperature: {args.temperature}")
        if args.top_k:
            print(f"  Top-k: {args.top_k}")
        if args.top_p:
            print(f"  Top-p: {args.top_p}")
    print()
    
    try:
        completions = generate_multiple_completions(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=do_sample,
            device=args.device,
            base_seed=args.seed,
        )
        
        # Print results
        print_completions(
            prompt=prompt,
            completions=completions,
            show_prompt=not args.hide_prompt,
        )
        
    except Exception as e:
        print(f"Error generating completions: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
