#!/usr/bin/env python3
"""
Export nanoGPT checkpoint to HuggingFace format.

Converts a checkpoint from train_nanogpt.py to a HuggingFace GPT2LMHeadModel
that can be used with Transformers/TRL for SFT and GRPO.

Usage:
    python export_hf.py \
        --checkpoint outputs/pretrain_baseline_pilot/model_00762.pt \
        --output-dir outputs/hf_baseline_pilot \
        --condition baseline
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_to_hf(
    checkpoint_path: Path,
    output_dir: Path,
    condition: str,
) -> None:
    """
    Export nanoGPT checkpoint to HuggingFace format.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_dir: Directory to save HF model
        condition: "baseline" or "mul_tokens"
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Extract config
    config_dict = checkpoint.get("config", {})
    vocab_size = config_dict.get("vocab_size", 50349)
    n_layer = config_dict.get("n_layer", 12)
    n_head = config_dict.get("n_head", 12)
    n_embd = config_dict.get("n_embd", 768)
    block_size = config_dict.get("block_size", 1024)
    
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_head: {n_head}")
    print(f"  n_embd: {n_embd}")
    print(f"  block_size: {block_size}")
    
    # Create HuggingFace config
    hf_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4 * n_embd,  # Default GPT-2 uses 4x
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
    )
    
    # Create HF model
    print("Creating HuggingFace GPT2LMHeadModel...")
    hf_model = GPT2LMHeadModel(hf_config)
    
    # Map nanoGPT weights to HF format
    nanogpt_sd = checkpoint["model"]
    hf_sd = hf_model.state_dict()
    
    # Weight mapping from nanoGPT -> HuggingFace
    # nanoGPT uses slightly different naming
    mapping = {
        "transformer.wte.weight": "transformer.wte.weight",
        "transformer.wpe.weight": "transformer.wpe.weight",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    
    # Add per-layer mappings
    for i in range(n_layer):
        # Attention
        mapping[f"transformer.h.{i}.ln_1.weight"] = f"transformer.h.{i}.ln_1.weight"
        mapping[f"transformer.h.{i}.ln_1.bias"] = f"transformer.h.{i}.ln_1.bias"
        mapping[f"transformer.h.{i}.attn.c_attn.weight"] = f"transformer.h.{i}.attn.c_attn.weight"
        mapping[f"transformer.h.{i}.attn.c_attn.bias"] = f"transformer.h.{i}.attn.c_attn.bias"
        mapping[f"transformer.h.{i}.attn.c_proj.weight"] = f"transformer.h.{i}.attn.c_proj.weight"
        mapping[f"transformer.h.{i}.attn.c_proj.bias"] = f"transformer.h.{i}.attn.c_proj.bias"
        
        # MLP
        mapping[f"transformer.h.{i}.ln_2.weight"] = f"transformer.h.{i}.ln_2.weight"
        mapping[f"transformer.h.{i}.ln_2.bias"] = f"transformer.h.{i}.ln_2.bias"
        mapping[f"transformer.h.{i}.mlp.c_fc.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
        mapping[f"transformer.h.{i}.mlp.c_fc.bias"] = f"transformer.h.{i}.mlp.c_fc.bias"
        mapping[f"transformer.h.{i}.mlp.c_proj.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"
        mapping[f"transformer.h.{i}.mlp.c_proj.bias"] = f"transformer.h.{i}.mlp.c_proj.bias"
    
    # Copy weights
    print("Copying weights...")
    copied = 0
    skipped = []
    
    for nanogpt_key, hf_key in mapping.items():
        if nanogpt_key in nanogpt_sd and hf_key in hf_sd:
            nanogpt_tensor = nanogpt_sd[nanogpt_key]
            hf_tensor = hf_sd[hf_key]
            
            # HF GPT-2 uses Conv1D for attention/MLP projections, which have transposed weights
            # But our nanoGPT uses Linear, so shapes should match directly
            # However, HF GPT2 actually uses Conv1D which stores [out_features, in_features]
            # while our Linear stores [out_features, in_features] - they should match
            
            if nanogpt_tensor.shape != hf_tensor.shape:
                # Try transpose for weight matrices
                if len(nanogpt_tensor.shape) == 2 and nanogpt_tensor.shape == hf_tensor.shape[::-1]:
                    nanogpt_tensor = nanogpt_tensor.t()
                else:
                    print(f"  Shape mismatch: {nanogpt_key} {nanogpt_tensor.shape} vs {hf_key} {hf_tensor.shape}")
                    skipped.append(nanogpt_key)
                    continue
            
            hf_sd[hf_key] = nanogpt_tensor
            copied += 1
        else:
            if nanogpt_key not in nanogpt_sd:
                skipped.append(f"missing in nanogpt: {nanogpt_key}")
            if hf_key not in hf_sd:
                skipped.append(f"missing in hf: {hf_key}")
    
    print(f"  Copied {copied} tensors")
    if skipped:
        print(f"  Skipped: {skipped[:5]}...")
    
    # Load state dict
    hf_model.load_state_dict(hf_sd)
    
    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {output_dir}...")
    hf_model.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "source_checkpoint": str(checkpoint_path),
        "condition": condition,
        "step": checkpoint.get("step"),
        "val_loss": checkpoint.get("val_loss"),
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "block_size": block_size,
    }
    
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("Done!")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export nanoGPT checkpoint to HuggingFace format",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to nanoGPT checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for HuggingFace model",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition",
    )
    
    args = parser.parse_args()
    
    export_to_hf(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        condition=args.condition,
    )


if __name__ == "__main__":
    main()

