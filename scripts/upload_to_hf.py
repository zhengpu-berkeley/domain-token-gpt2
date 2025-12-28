#!/usr/bin/env python3
"""
Upload model checkpoints to Hugging Face Hub.

This script handles uploading trained models (pretrained, SFT, or RL) to HF Hub
for storage and sharing across different compute environments.

Usage:
    # Upload a single model:
    uv run python scripts/upload_to_hf.py \
        --model-path outputs/hf_baseline_10b \
        --repo-name zhengpu-berkeley/domain-token-gpt2-baseline-10b

    # Upload with custom commit message:
    uv run python scripts/upload_to_hf.py \
        --model-path outputs/sft_tulu_baseline \
        --repo-name zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline \
        --commit-message "Upload Tulu-3 SFT model (baseline condition)"

    # Upload multiple models at once:
    uv run python scripts/upload_to_hf.py \
        --model-path outputs/hf_baseline_10b outputs/hf_mul_tokens_10b \
        --repo-name zhengpu-berkeley/domain-token-gpt2-baseline-10b zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi, create_repo, login


def upload_model_to_hub(
    model_path: Path,
    repo_name: str,
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Upload a model directory to Hugging Face Hub.
    
    Args:
        model_path: Path to the model directory
        repo_name: Full repo name (e.g., "username/model-name")
        commit_message: Commit message for the upload
        private: Whether to make the repo private
        token: HF API token (optional if already logged in)
        
    Returns:
        URL of the uploaded model
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    if not model_path.is_dir():
        raise ValueError(f"Model path must be a directory: {model_path}")
    
    # Check for required files
    required_files = ["config.json"]
    optional_model_files = ["model.safetensors", "pytorch_model.bin"]
    
    for f in required_files:
        if not (model_path / f).exists():
            raise FileNotFoundError(f"Missing required file: {model_path / f}")
    
    has_model = any((model_path / f).exists() for f in optional_model_files)
    if not has_model:
        print(f"  Warning: No model weights found (expected {optional_model_files})")
    
    # Initialize API
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    print(f"Creating/accessing repo: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token,
        )
    except Exception as e:
        print(f"  Note: {e}")
    
    # Generate commit message if not provided
    if commit_message is None:
        commit_message = f"Upload model from {model_path.name}"
    
    # Upload folder
    print(f"Uploading {model_path} to {repo_name}...")
    
    url = api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message,
        token=token,
    )
    
    print(f"  ✅ Upload complete: {url}")
    
    return url


def upload_multiple_models(
    model_paths: List[Path],
    repo_names: List[str],
    commit_message: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> List[str]:
    """
    Upload multiple models to Hugging Face Hub.
    
    Args:
        model_paths: List of paths to model directories
        repo_names: List of repo names (must match length of model_paths)
        commit_message: Commit message (applied to all uploads)
        private: Whether to make repos private
        token: HF API token
        
    Returns:
        List of URLs
    """
    if len(model_paths) != len(repo_names):
        raise ValueError(
            f"Number of model paths ({len(model_paths)}) must match "
            f"number of repo names ({len(repo_names)})"
        )
    
    urls = []
    for i, (model_path, repo_name) in enumerate(zip(model_paths, repo_names), 1):
        print(f"\n[{i}/{len(model_paths)}] Uploading {model_path.name}...")
        try:
            url = upload_model_to_hub(
                model_path=model_path,
                repo_name=repo_name,
                commit_message=commit_message,
                private=private,
                token=token,
            )
            urls.append(url)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            urls.append(None)
    
    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Upload models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload pretrained baseline model:
  uv run python scripts/upload_to_hf.py \\
      --model-path outputs/hf_baseline_10b \\
      --repo-name zhengpu-berkeley/domain-token-gpt2-baseline-10b

  # Upload SFT model:
  uv run python scripts/upload_to_hf.py \\
      --model-path outputs/sft_tulu_baseline \\
      --repo-name zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline

  # Upload multiple models:
  uv run python scripts/upload_to_hf.py \\
      --model-path outputs/hf_baseline_10b outputs/hf_mul_tokens_10b \\
      --repo-name zhengpu-berkeley/domain-token-gpt2-baseline-10b \\
                  zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b

  # Login first (if not already logged in):
  huggingface-cli login --token YOUR_TOKEN
        """,
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to model directory(ies) to upload",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        nargs="+",
        required=True,
        help="HuggingFace repo name(s) (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Commit message for the upload",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repo(s) private",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (optional if already logged in via CLI)",
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Prompt for HuggingFace login before uploading",
    )
    
    args = parser.parse_args()
    
    # Handle login if requested
    if args.login:
        print("Please enter your HuggingFace token:")
        login()
    
    # Validate arguments
    if len(args.model_path) != len(args.repo_name):
        print(
            f"Error: Number of model paths ({len(args.model_path)}) must match "
            f"number of repo names ({len(args.repo_name)})",
            file=sys.stderr,
        )
        sys.exit(1)
    
    # Upload
    try:
        if len(args.model_path) == 1:
            url = upload_model_to_hub(
                model_path=args.model_path[0],
                repo_name=args.repo_name[0],
                commit_message=args.commit_message,
                private=args.private,
                token=args.token,
            )
            print(f"\n✅ Model uploaded successfully!")
            print(f"   URL: {url}")
        else:
            urls = upload_multiple_models(
                model_paths=args.model_path,
                repo_names=args.repo_name,
                commit_message=args.commit_message,
                private=args.private,
                token=args.token,
            )
            
            print(f"\n{'='*60}")
            print("Upload Summary:")
            print(f"{'='*60}")
            for model_path, repo_name, url in zip(args.model_path, args.repo_name, urls):
                status = "✅" if url else "❌"
                print(f"  {status} {model_path.name} → {repo_name}")
            
            success_count = sum(1 for u in urls if u)
            print(f"\nUploaded {success_count}/{len(urls)} models successfully.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

