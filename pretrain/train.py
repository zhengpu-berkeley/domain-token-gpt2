#!/usr/bin/env python3
"""
Pretraining script for domain-token GPT experiments.

This is a simplified training loop inspired by nanoGPT, adapted for our experiments.
Key differences from vanilla nanoGPT:
- Reads vocab_size from tokenizer metadata
- Uses YAML config files
- Supports both baseline and mul_tokens conditions
- Simplified for local CPU/MPS smoke tests
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml


# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class GPTConfig:
    """GPT model configuration."""
    block_size: int = 256
    vocab_size: int = 50349  # GPT-2 + mul tokens
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
    
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} > block_size {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str):
        """Create AdamW optimizer with weight decay."""
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DataLoader:
    """Simple data loader for tokenized data."""
    
    def __init__(self, data_dir: Path, split: str, B: int, T: int):
        self.B = B
        self.T = T
        
        # Find shard files
        pattern = f"{split}_*.npy"
        shards = sorted(data_dir.glob(pattern))
        if not shards:
            raise ValueError(f"No shards found for split '{split}' in {data_dir}")
        
        self.shards = shards
        self.current_shard = 0
        self.tokens = self._load_shard(0)
        self.current_position = 0
    
    def _load_shard(self, idx: int) -> torch.Tensor:
        """Load a shard and convert to tensor."""
        npt = np.load(self.shards[idx])
        npt = npt.astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)
    
    def reset(self):
        """Reset to beginning."""
        self.current_shard = 0
        self.tokens = self._load_shard(0)
        self.current_position = 0
    
    def next_batch(self):
        """Get next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        
        self.current_position += B * T
        
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_shard(self.current_shard)
            self.current_position = 0
        
        return x, y


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    config_path: Path,
    condition: str,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    seed: int = 42,
):
    """Main training function."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "processed" / condition
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs" / f"pretrain_{condition}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer metadata for vocab_size
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        vocab_size = meta["vocab_size"]
    else:
        print("Warning: No meta.json found, using default vocab_size")
        vocab_size = 50349
    
    # Device setup
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
    
    # Extract config values
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["eval"]
    log_cfg = config["logging"]
    ckpt_cfg = config["checkpoint"]
    
    B = train_cfg["batch_size"]
    T = model_cfg["block_size"]
    
    # Create model
    gpt_config = GPTConfig(
        block_size=T,
        vocab_size=vocab_size,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
    )
    
    print(f"\nModel configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_layer: {gpt_config.n_layer}")
    print(f"  n_head: {gpt_config.n_head}")
    print(f"  n_embd: {gpt_config.n_embd}")
    print(f"  block_size: {gpt_config.block_size}")
    
    model = GPT(gpt_config)
    model.to(device)
    
    num_params = model.count_parameters()
    print(f"  parameters: {num_params:,}")
    
    # Create data loaders
    train_loader = DataLoader(data_dir, "train", B, T)
    val_loader = DataLoader(data_dir, "val", B, T)
    
    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=train_cfg["weight_decay"],
        learning_rate=train_cfg["max_lr"],
        device_type=device_type,
    )
    
    # Calculate gradient accumulation steps
    total_batch_size = train_cfg["total_batch_size"]
    grad_accum_steps = max(1, total_batch_size // (B * T))
    print(f"\nTraining configuration:")
    print(f"  batch_size: {B}")
    print(f"  block_size: {T}")
    print(f"  total_batch_size: {total_batch_size}")
    print(f"  grad_accum_steps: {grad_accum_steps}")
    print(f"  max_steps: {train_cfg['max_steps']}")
    
    # Training loop
    log_file = output_dir / "log.txt"
    with open(log_file, "w") as f:
        f.write(f"condition: {condition}\n")
        f.write(f"vocab_size: {vocab_size}\n")
        f.write(f"parameters: {num_params}\n")
    
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for step in range(train_cfg["max_steps"]):
        t0 = time.time()
        last_step = (step == train_cfg["max_steps"] - 1)
        
        # Evaluate periodically
        if step % eval_cfg["interval"] == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(eval_cfg["val_steps"]):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss_accum += loss.item()
                val_loss = val_loss_accum / eval_cfg["val_steps"]
            
            val_losses.append({"step": step, "loss": val_loss})
            print(f"step {step:5d} | val loss: {val_loss:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss:.4f}\n")
        
        # Save checkpoint
        if (step > 0 and step % ckpt_cfg["interval"] == 0) or (last_step and ckpt_cfg["save_final"]):
            ckpt_path = output_dir / f"model_{step:05d}.pt"
            torch.save({
                "model": model.state_dict(),
                "config": gpt_config,
                "step": step,
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.item()
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        
        # Update learning rate
        lr = get_lr(step, train_cfg["warmup_steps"], train_cfg["max_steps"],
                   train_cfg["max_lr"], train_cfg["min_lr"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        optimizer.step()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (B * T * grad_accum_steps) / dt
        
        train_losses.append({"step": step, "loss": loss_accum})
        
        if step % log_cfg["interval"] == 0 or last_step:
            print(f"step {step:5d} | loss: {loss_accum:.4f} | lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tokens_per_sec:.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum:.4f}\n")
    
    # Save final results
    results = {
        "condition": condition,
        "vocab_size": vocab_size,
        "num_params": num_params,
        "config": config,
        "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
        "final_val_loss": val_losses[-1]["loss"] if val_losses else None,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Final train loss: {results['final_train_loss']:.4f}")
    print(f"  Final val loss: {results['final_val_loss']:.4f}")
    print(f"  Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Pretrain GPT model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "tiny.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        default="baseline",
        help="Experimental condition",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: data/processed/{condition})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/pretrain_{condition})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        condition=args.condition,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

