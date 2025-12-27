#!/usr/bin/env python3
"""
Efficient GPT-2 pretraining script based on build-nanogpt.

This is an adapted version of Karpathy's train_gpt2.py with:
- YAML config file support
- CLI arguments for data-root, output-dir, condition
- vocab_size=50349 for mul-token experiments
- Automatic bf16/fp16 selection
- Disabled HellaSwag eval by default (can enable via flag)
- Clean output structure with results.json

Supports DDP for multi-GPU training:
    torchrun --standalone --nproc_per_node=8 train_nanogpt.py --config ...

Single GPU:
    python train_nanogpt.py --config pretrain/configs/gpt2_124m_pilot.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import yaml

# -----------------------------------------------------------------------------
# Model Definition (from build-nanogpt)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
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
    def __init__(self, config):
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
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50349  # GPT-2 + padding + mul tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
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
        assert T <= self.config.block_size, f"Sequence {T} > block_size {self.config.block_size}"
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Data Loading

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Get shard filenames
        shards = [s for s in os.listdir(data_root) if split in s and s.endswith('.npy')]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {data_root}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y


# -----------------------------------------------------------------------------
# Training

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train(
    config_path: Path,
    data_root: Path,
    output_dir: Path,
    condition: str,
    seed: int = 42,
    resume_from: Optional[Path] = None,
):
    """Main training function."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config.get("eval", {})
    log_cfg = config.get("logging", {})
    ckpt_cfg = config.get("checkpoint", {})
    
    # DDP setup
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        from torch.distributed import init_process_group, destroy_process_group
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.distributed as dist
        
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    if master_process:
        print(f"Using device: {device}")
        print(f"DDP: {ddp}, world_size: {ddp_world_size}")
    
    # Set seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Determine dtype
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
        dtype_name = "bfloat16"
    elif device_type == "cuda":
        autocast_dtype = torch.float16
        dtype_name = "float16"
    else:
        autocast_dtype = torch.float32
        dtype_name = "float32"
    
    if master_process:
        print(f"Using autocast dtype: {dtype_name}")
    
    # Training params
    B = train_cfg["batch_size"]
    T = model_cfg["block_size"]
    total_batch_size = train_cfg["total_batch_size"]
    max_steps = train_cfg["max_steps"]
    warmup_steps = train_cfg["warmup_steps"]
    max_lr = train_cfg["max_lr"]
    min_lr = train_cfg["min_lr"]
    weight_decay = train_cfg["weight_decay"]
    grad_clip = train_cfg["grad_clip"]
    
    assert total_batch_size % (B * T * ddp_world_size) == 0
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
    
    # Load vocab_size from data metadata if available
    meta_path = data_root / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        vocab_size = meta.get("vocab_size", 50349)
    else:
        vocab_size = 50349
    
    # Create model
    gpt_config = GPTConfig(
        block_size=T,
        vocab_size=vocab_size,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
    )
    
    if master_process:
        print(f"\nModel configuration:")
        print(f"  vocab_size: {vocab_size}")
        print(f"  n_layer: {gpt_config.n_layer}")
        print(f"  n_head: {gpt_config.n_head}")
        print(f"  n_embd: {gpt_config.n_embd}")
        print(f"  block_size: {gpt_config.block_size}")
    
    model = GPT(gpt_config)
    model.to(device)
    
    if master_process:
        num_params = model.count_parameters()
        print(f"  parameters: {num_params:,}")
    
    torch.set_float32_matmul_precision('high')
    
    # Optionally compile model
    use_compile = train_cfg.get("compile", False)
    if use_compile and hasattr(torch, 'compile'):
        if master_process:
            print("Compiling model...")
        model = torch.compile(model)
    
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model
    
    # Create data loaders
    train_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="train", data_root=str(data_root), master_process=master_process
    )
    val_loader = DataLoaderLite(
        B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,
        split="val", data_root=str(data_root), master_process=master_process
    )
    
    # Create optimizer
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        device_type=device_type,
        master_process=master_process,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from is not None and resume_from.exists():
        if master_process:
            print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        raw_model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = checkpoint.get('step', 0) + 1
    
    # Setup output directory
    output_dir = Path(output_dir)
    if master_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "log.txt"
        with open(log_file, "w") as f:
            f.write(f"condition: {condition}\n")
            f.write(f"config: {config_path}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"vocab_size: {vocab_size}\n")
            f.write(f"dtype: {dtype_name}\n")
    
    # Eval settings
    eval_interval = eval_cfg.get("interval", 250)
    val_loss_steps = eval_cfg.get("val_steps", 20)
    
    # Checkpoint settings
    ckpt_interval = ckpt_cfg.get("interval", 5000)
    save_final = ckpt_cfg.get("save_final", True)
    
    # Log settings
    log_interval = log_cfg.get("interval", 10)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    if master_process:
        print(f"\nStarting training from step {start_step}...")
    
    for step in range(start_step, max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)
        
        # Validation
        if step % eval_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp:
                import torch.distributed as dist
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                val_loss = val_loss_accum.item()
                val_losses.append({"step": step, "loss": val_loss})
                print(f"step {step:5d} | val loss: {val_loss:.4f}")
                with open(output_dir / "log.txt", "a") as f:
                    f.write(f"{step} val {val_loss:.4f}\n")
        
        # Checkpointing
        if master_process and step > 0 and (step % ckpt_interval == 0 or (last_step and save_final)):
            checkpoint_path = output_dir / f"model_{step:05d}.pt"
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': asdict(raw_model.config),
                'step': step,
                'val_loss': val_losses[-1]["loss"] if val_losses else None,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            import torch.distributed as dist
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        train_losses.append({"step": step, "loss": loss_accum.item()})
        
        if master_process and (step % log_interval == 0 or last_step):
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(output_dir / "log.txt", "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")
    
    # Save final results
    if master_process:
        results = {
            "condition": condition,
            "seed": seed,
            "vocab_size": vocab_size,
            "num_params": raw_model.count_parameters(),
            "config": config,
            "gpt_config": asdict(raw_model.config),
            "final_train_loss": train_losses[-1]["loss"] if train_losses else None,
            "final_val_loss": val_losses[-1]["loss"] if val_losses else None,
            "total_steps": max_steps,
            "total_tokens": max_steps * total_batch_size,
            "train_losses": train_losses[-100:],  # Last 100 for space
            "val_losses": val_losses,
        }
        
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"  Final train loss: {results['final_train_loss']:.6f}")
        print(f"  Final val loss: {results['final_val_loss']:.4f}")
        print(f"  Total tokens: {results['total_tokens']:,}")
        print(f"  Results saved to: {results_path}")
    
    if ddp:
        from torch.distributed import destroy_process_group
        destroy_process_group()
    
    return results if master_process else None


def main():
    parser = argparse.ArgumentParser(
        description="Efficient GPT-2 pretraining based on build-nanogpt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing tokenized shards (*.npy files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["baseline", "mul_tokens"],
        required=True,
        help="Experimental condition (for logging)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        condition=args.condition,
        seed=args.seed,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()

