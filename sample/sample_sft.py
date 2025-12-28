#!/usr/bin/env python3
"""
Interactive sampling script for SFT-trained models.

Uses the User:/Assistant: format that matches our SFT training (Tulu-3 + GSM8K).
Supports both single-turn Q&A and multi-turn conversation.

Usage:
    # Interactive mode (type questions, get answers):
    uv run python sample/sample_sft.py \
        --model-path outputs/sft_tulu_baseline \
        --interactive

    # Single question:
    uv run python sample/sample_sft.py \
        --model-path outputs/sft_tulu_baseline \
        --question "What is 6 times 9?"

    # Math problem with multiple samples:
    uv run python sample/sample_sft.py \
        --model-path outputs/sft_gsm8k_baseline \
        --question "If a train travels at 60 mph for 2.5 hours, how far does it go?" \
        --num-samples 5 \
        --temperature 0.7

    # Greedy decoding for deterministic answers:
    uv run python sample/sample_sft.py \
        --model-path outputs/sft_tulu_baseline \
        --question "Explain photosynthesis briefly." \
        --greedy
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


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
    
    # Check for mul-tokens
    mul_token_count = sum(1 for i in range(50304, 50349) 
                          if tokenizer.convert_ids_to_tokens(i, skip_special_tokens=False).startswith("<MUL_"))
    if mul_token_count > 0:
        print(f"  Mul-tokens available: {mul_token_count}")
    
    return model, tokenizer


def format_prompt(question: str, system_prompt: Optional[str] = None) -> str:
    """
    Format a question into our SFT prompt format.
    
    Format: User: {question}\nAssistant:
    
    This matches the training format from sft_tulu.py and sft_gsm8k.py.
    
    Args:
        question: The user's question
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt.strip()}")
    
    parts.append(f"User: {question.strip()}")
    parts.append("Assistant:")
    
    return "\n".join(parts)


def generate_response(
    model,
    tokenizer,
    question: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    device: str = "cuda",
    seed: Optional[int] = None,
    stop_at_newline_user: bool = True,
) -> str:
    """
    Generate a response to a question using the SFT format.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2TokenizerFast instance
        question: User's question
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        do_sample: Whether to use sampling (False = greedy)
        device: Device to run on
        seed: Random seed
        stop_at_newline_user: Stop generation when "User:" is seen
        
    Returns:
        Generated response text
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Format the prompt
    prompt = format_prompt(question, system_prompt)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    
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
    generated_ids = outputs[0][prompt_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Stop at "User:" to prevent the model from continuing the conversation
    if stop_at_newline_user:
        # Look for common conversation markers that indicate a new turn
        stop_markers = ["\nUser:", "\nuser:", "\n\nUser:", "User:"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0]
                break
    
    return response.strip()


def generate_multiple_responses(
    model,
    tokenizer,
    question: str,
    num_samples: int = 5,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
    device: str = "cuda",
    base_seed: Optional[int] = None,
) -> List[str]:
    """
    Generate multiple responses with different random seeds.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2TokenizerFast instance
        question: User's question
        num_samples: Number of responses to generate
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        do_sample: Whether to use sampling
        device: Device to run on
        base_seed: Base random seed
        
    Returns:
        List of generated responses
    """
    responses = []
    
    for i in range(num_samples):
        seed = (base_seed + i) if base_seed is not None else None
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            question=question,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            device=device,
            seed=seed,
        )
        responses.append(response)
    
    return responses


def print_response(question: str, response: str, sample_idx: Optional[int] = None):
    """Pretty print a single response."""
    print()
    print("─" * 60)
    if sample_idx is not None:
        print(f"[Sample {sample_idx}]")
    print(f"User: {question}")
    print()
    # Clean up response - remove leading newlines and normalize
    clean_response = response.lstrip('\n')
    # If model already prefixed with "A:" or "Assistant:", don't add another
    if not clean_response.startswith(("A:", "Assistant:")):
        print(f"Assistant: {clean_response}")
    else:
        print(clean_response)
    print("─" * 60)


def print_responses(question: str, responses: List[str]):
    """Pretty print multiple responses."""
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    
    for i, response in enumerate(responses, 1):
        print(f"\n[Sample {i}]")
        print("─" * 60)
        # Clean up response - remove leading newlines and normalize
        clean_response = response.lstrip('\n')
        # If model already prefixed with "A:" or "Assistant:", don't add another
        if not clean_response.startswith(("A:", "Assistant:")):
            print(f"Assistant: {clean_response}")
        else:
            print(clean_response)
    
    print("\n" + "=" * 60)


def interactive_mode(
    model,
    tokenizer,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    device: str = "cuda",
):
    """
    Run an interactive Q&A session.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2TokenizerFast instance
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
    """
    print("\n" + "=" * 60)
    print("Interactive SFT Model Sampling")
    print("=" * 60)
    print("Type your questions and press Enter to get responses.")
    print("Commands:")
    print("  /quit, /exit, /q  - Exit the session")
    print("  /temp <value>     - Change temperature (current: {:.1f})".format(temperature))
    print("  /greedy           - Toggle greedy mode (current: off)")
    print("  /system <prompt>  - Set system prompt")
    print("=" * 60)
    
    greedy_mode = False
    
    while True:
        try:
            print()
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("Goodbye!")
                break
            
            if user_input.startswith("/temp "):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Usage: /temp <value> (e.g., /temp 0.8)")
                continue
            
            if user_input.lower() == "/greedy":
                greedy_mode = not greedy_mode
                print(f"Greedy mode: {'ON' if greedy_mode else 'OFF'}")
                continue
            
            if user_input.startswith("/system "):
                system_prompt = user_input[8:].strip()
                print(f"System prompt set to: {system_prompt}")
                continue
            
            # Generate response
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                question=user_input,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if not greedy_mode else None,
                top_p=top_p if not greedy_mode else None,
                do_sample=not greedy_mode,
                device=device,
            )
            
            print(f"\nAssistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample from SFT-trained models using User:/Assistant: format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive Q&A session:
  uv run python sample/sample_sft.py \\
      --model-path outputs/sft_tulu_baseline \\
      --interactive

  # Single math question:
  uv run python sample/sample_sft.py \\
      --model-path outputs/sft_gsm8k_baseline \\
      --question "What is 7 times 8?"

  # Multiple samples with temperature:
  uv run python sample/sample_sft.py \\
      --model-path outputs/sft_tulu_baseline \\
      --question "Explain machine learning in simple terms." \\
      --num-samples 3 \\
      --temperature 0.8

  # Greedy (deterministic) decoding:
  uv run python sample/sample_sft.py \\
      --model-path outputs/sft_tulu_baseline \\
      --question "What is the capital of France?" \\
      --greedy

  # With system prompt:
  uv run python sample/sample_sft.py \\
      --model-path outputs/sft_tulu_baseline \\
      --question "Solve 15 + 27" \\
      --system-prompt "You are a helpful math tutor. Show your work step by step."
        """,
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to SFT model directory",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask the model (required unless --interactive)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ignore --question)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to guide responses",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of responses to generate (default: 1)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (deterministic)",
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
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and args.question is None:
        parser.error("Either --question or --interactive is required")
    
    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
    else:
        # Determine sampling mode
        do_sample = not args.greedy
        
        print(f"\nGenerating {args.num_samples} response(s)...")
        print(f"  Sampling: {'Greedy' if not do_sample else 'Stochastic'}")
        if do_sample:
            print(f"  Temperature: {args.temperature}")
            print(f"  Top-k: {args.top_k}")
            print(f"  Top-p: {args.top_p}")
        
        try:
            if args.num_samples == 1:
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    question=args.question,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    do_sample=do_sample,
                    device=args.device,
                    seed=args.seed,
                )
                print_response(args.question, response)
            else:
                responses = generate_multiple_responses(
                    model=model,
                    tokenizer=tokenizer,
                    question=args.question,
                    num_samples=args.num_samples,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    do_sample=do_sample,
                    device=args.device,
                    base_seed=args.seed,
                )
                print_responses(args.question, responses)
                
        except Exception as e:
            print(f"Error generating response: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()

