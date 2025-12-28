#!/usr/bin/env python3
"""
Download model checkpoints from Hugging Face Hub to local outputs folder.

This script syncs trained models across different compute environments by
downloading them from HuggingFace Hub to the standard outputs directory structure.

Usage:
    # Download all known models:
    uv run python scripts/load_from_hf.py --all

    # Download specific models:
    uv run python scripts/load_from_hf.py --models baseline mul_tokens sft_tulu_baseline

    # Download to custom directory:
    uv run python scripts/load_from_hf.py --all --output-root /path/to/outputs

    # List available models without downloading:
    uv run python scripts/load_from_hf.py --list
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import snapshot_download, HfApi


# Registry of known models and their HuggingFace repos
# Format: {local_name: repo_id}
MODEL_REGISTRY: Dict[str, str] = {
    # Pretrained models (10B tokens)
    "hf_baseline_10b": "zhengpu-berkeley/domain-token-gpt2-baseline-10b",
    "hf_mul_tokens_10b": "zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b",
    
    # Tulu-3 SFT models
    "sft_tulu_baseline": "zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline",
    "sft_tulu_mul_tokens": "zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens",
    
    # GSM8K SFT models (after Tulu)
    "sft_gsm8k_baseline": "zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-baseline",
    "sft_gsm8k_mul_tokens": "zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-mul-tokens",
}

# Model groups for convenience
MODEL_GROUPS = {
    "pretrained": ["hf_baseline_10b", "hf_mul_tokens_10b"],
    "sft_tulu": ["sft_tulu_baseline", "sft_tulu_mul_tokens"],
    "sft_gsm8k": ["sft_gsm8k_baseline", "sft_gsm8k_mul_tokens"],
    "baseline": ["hf_baseline_10b", "sft_tulu_baseline", "sft_gsm8k_baseline"],
    "mul_tokens": ["hf_mul_tokens_10b", "sft_tulu_mul_tokens", "sft_gsm8k_mul_tokens"],
}


def check_repo_exists(repo_id: str) -> bool:
    """Check if a HuggingFace repo exists and is accessible."""
    try:
        api = HfApi()
        api.repo_info(repo_id=repo_id, repo_type="model")
        return True
    except Exception:
        return False


def list_available_models() -> Dict[str, bool]:
    """List all registered models and check which exist on HuggingFace."""
    print("Checking available models on HuggingFace Hub...\n")
    
    results = {}
    for local_name, repo_id in MODEL_REGISTRY.items():
        exists = check_repo_exists(repo_id)
        results[local_name] = exists
        status = "✅" if exists else "❌"
        print(f"  {status} {local_name}")
        print(f"     → {repo_id}")
    
    available = sum(1 for v in results.values() if v)
    print(f"\n{available}/{len(results)} models available on HuggingFace Hub")
    
    return results


def download_model(
    local_name: str,
    output_root: Path,
    force: bool = False,
    token: Optional[str] = None,
) -> Optional[Path]:
    """
    Download a single model from HuggingFace Hub.
    
    Args:
        local_name: Local model name (key in MODEL_REGISTRY)
        output_root: Root directory for outputs (e.g., outputs/)
        force: Re-download even if local directory exists
        token: HuggingFace token for private repos
        
    Returns:
        Path to downloaded model, or None if failed
    """
    if local_name not in MODEL_REGISTRY:
        print(f"  ❌ Unknown model: {local_name}")
        print(f"     Available: {', '.join(MODEL_REGISTRY.keys())}")
        return None
    
    repo_id = MODEL_REGISTRY[local_name]
    local_path = output_root / local_name
    
    # Check if already exists
    if local_path.exists() and not force:
        # Check if it has model files
        has_model = (local_path / "config.json").exists()
        if has_model:
            print(f"  ⏭️  {local_name} already exists (use --force to re-download)")
            return local_path
    
    # Check if repo exists
    if not check_repo_exists(repo_id):
        print(f"  ❌ {local_name}: repo not found on HuggingFace ({repo_id})")
        return None
    
    print(f"  📥 Downloading {local_name}...")
    print(f"     From: {repo_id}")
    print(f"     To:   {local_path}")
    
    try:
        # Download the entire repo
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"  ✅ {local_name} downloaded successfully")
        return local_path
        
    except Exception as e:
        print(f"  ❌ Failed to download {local_name}: {e}")
        return None


def download_models(
    model_names: List[str],
    output_root: Path,
    force: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Optional[Path]]:
    """
    Download multiple models from HuggingFace Hub.
    
    Args:
        model_names: List of local model names to download
        output_root: Root directory for outputs
        force: Re-download even if local directories exist
        token: HuggingFace token for private repos
        
    Returns:
        Dict mapping model names to download paths (None if failed)
    """
    results = {}
    
    print(f"\nDownloading {len(model_names)} model(s) to {output_root}/\n")
    
    for i, name in enumerate(model_names, 1):
        print(f"[{i}/{len(model_names)}] {name}")
        results[name] = download_model(
            local_name=name,
            output_root=output_root,
            force=force,
            token=token,
        )
        print()
    
    # Summary
    success = sum(1 for v in results.values() if v is not None)
    print("=" * 60)
    print(f"Download Summary: {success}/{len(model_names)} successful")
    print("=" * 60)
    
    for name, path in results.items():
        status = "✅" if path else "❌"
        print(f"  {status} {name}")
    
    return results


def expand_model_names(names: List[str]) -> List[str]:
    """Expand model group names into individual model names."""
    expanded = []
    for name in names:
        if name in MODEL_GROUPS:
            expanded.extend(MODEL_GROUPS[name])
        elif name in MODEL_REGISTRY:
            expanded.append(name)
        else:
            # Try partial matching
            matches = [k for k in MODEL_REGISTRY.keys() if name in k]
            if matches:
                expanded.extend(matches)
            else:
                print(f"Warning: Unknown model '{name}', skipping")
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            result.append(name)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download models from HuggingFace Hub to local outputs folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models:
  uv run python scripts/load_from_hf.py --list

  # Download all available models:
  uv run python scripts/load_from_hf.py --all

  # Download specific models:
  uv run python scripts/load_from_hf.py --models hf_baseline_10b sft_tulu_baseline

  # Download model groups:
  uv run python scripts/load_from_hf.py --models pretrained  # Both pretrained models
  uv run python scripts/load_from_hf.py --models baseline    # All baseline condition models

  # Force re-download:
  uv run python scripts/load_from_hf.py --models hf_baseline_10b --force

Model Groups:
  pretrained  - hf_baseline_10b, hf_mul_tokens_10b
  sft_tulu    - sft_tulu_baseline, sft_tulu_mul_tokens
  sft_gsm8k   - sft_gsm8k_baseline, sft_gsm8k_mul_tokens
  baseline    - All baseline condition models
  mul_tokens  - All mul_tokens condition models
        """,
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models without downloading",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Model names or groups to download",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for outputs (default: outputs/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local directory exists",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for private repos",
    )
    
    args = parser.parse_args()
    
    # List mode
    if args.list:
        list_available_models()
        return
    
    # Validate arguments
    if not args.all and not args.models:
        parser.error("Either --all or --models is required (use --list to see available models)")
    
    # Determine which models to download
    if args.all:
        # Get all available models
        available = list_available_models()
        model_names = [name for name, exists in available.items() if exists]
        if not model_names:
            print("\nNo models available to download.")
            sys.exit(1)
    else:
        model_names = expand_model_names(args.models)
        if not model_names:
            print("No valid model names provided.")
            sys.exit(1)
    
    # Create output directory
    args.output_root.mkdir(parents=True, exist_ok=True)
    
    # Download
    results = download_models(
        model_names=model_names,
        output_root=args.output_root,
        force=args.force,
        token=args.token,
    )
    
    # Exit with error if any downloads failed
    if any(v is None for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

