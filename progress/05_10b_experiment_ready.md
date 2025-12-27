# Phase 3: 10B Token Full Experiment — Execution Status

**Date:** December 27, 2024  
**Last Updated:** December 27, 2025  
**Status:** ✅ Pretraining complete; ✅ checkpoint eval complete; ⚠️ post-training/evals need rerun after export fix  
**Hardware:** 4× NVIDIA H200 GPUs (143GB VRAM each)

---

## Update (Dec 27, 2025)

- Fixed nanoGPT → HuggingFace export bug (`pretrain/export_hf.py`). Prior HF/SFT/GSM8K eval artifacts derived from broken exports should be treated as **invalid**.
- Validated learning signal via checkpoint HellaSwag evals in `outputs/checkpoint_evals/{baseline,mul_tokens}/`.
- Latest checkpoint (step **19072**, ~10B tokens): baseline acc_norm **0.3034**, mul_tokens acc_norm **0.3077**.

## Current Status

### ✅ Baseline Condition — COMPLETE

| Stage | Status | Details |
|-------|--------|---------|
| Data Preparation | ✅ Complete | 12 shards, ~10B tokens |
| Pretraining | ✅ Complete | 19,072 steps, final loss: 3.01 (train) / 3.12 (val) |
| Checkpoint HellaSwag Eval | ✅ Complete | Saved to `outputs/checkpoint_evals/baseline/` |
| HF Export + Post-training (SFT/RL) | ⚠️ Needs rerun | Prior exports were broken; rerun with fixed exporter |
| Task Eval (GSM8K / probes) | ⚠️ Needs rerun | Prior task metrics were derived from broken exports |

**Baseline (checkpoint HellaSwag, acc_norm):**
- Step 2000: 0.2604 → Step 19072: **0.3034**

### ✅ Mul_Tokens Condition — COMPLETE

| Stage | Status | Details |
|-------|--------|---------|
| Data Preparation | ✅ Complete | 12 shards, ~10B tokens with mul-token injection |
| Pretraining | ✅ Complete | 19,072 steps, final loss: 3.01 (train) / 3.12 (val) |
| Checkpoint HellaSwag Eval | ✅ Complete | Saved to `outputs/checkpoint_evals/mul_tokens/` |
| HF Export + Post-training (SFT/RL) | ⚠️ Needs rerun | Prior exports were broken; rerun with fixed exporter |
| Task Eval (GSM8K / probes) | ⚠️ Needs rerun | Prior task metrics were derived from broken exports |

**Mul_tokens (checkpoint HellaSwag, acc_norm):**
- Step 2000: 0.2618 → Step 19072: **0.3077**

---

## Overview

All infrastructure for the 10B token compute-matched experiment is in place. This document summarizes the setup, execution status, and provides instructions for (re)running post-training + task eval with the **fixed** HF exporter.

---

## Pretraining + Checkpoint Eval Summary (Post-fix)

### Pretraining (both conditions)
- **Total Steps:** 19,072 / 19,073 (final saved at step 19072)
- **Total Tokens:** 9,999,745,024 (~10B)
- **Final losses:** ~3.01 (train) / ~3.12 (val)
- **Checkpoints saved:** steps 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 19072

### Checkpoint HellaSwag (acc_norm)
- Baseline: 0.2604 → **0.3034** (step 2000 → 19072)
- Mul_tokens: 0.2618 → **0.3077** (step 2000 → 19072)

See:
- `outputs/checkpoint_evals/baseline/`
- `outputs/checkpoint_evals/mul_tokens/`

### Post-training + task eval (GSM8K / probes)
Post-training must be re-run using the fixed exporter. Any previously reported GSM8K/probe numbers derived from pre-fix HF exports are **not trusted**.

---

## Files Created

| File | Purpose |
|------|---------|
| `pretrain/configs/gpt2_124m_10b.yaml` | 10B token pretraining config for 4× H200 DDP |
| `sft/configs/tulu.yaml` | Tulu-3 SFT configuration (instruction-tuning, sampled subset) |
| `sft/configs/gsm8k.yaml` | GSM8K SFT configuration (math specialization) |
| `scripts/run_10b.sh` | Complete automation script for both conditions |

---

## Configuration Summary

### Pretraining (gpt2_124m_10b.yaml)

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 124M (12 layers, 12 heads, 768 dim) |
| Total tokens | 10B |
| Batch size per GPU | 32 |
| Total batch size | 524,288 tokens/step |
| Optimizer steps | 19,073 |
| Warmup steps | 715 (~3.75%) |
| Learning rate | 6e-4 → 6e-5 (cosine) |
| Precision | bfloat16 |
| torch.compile | Enabled |

### SFT (Two-stage: Tulu → GSM8K)

| Parameter | Value |
|-----------|-------|
| Stage 1 | Tulu-3 mixture (subset), 1 epoch, max_len=1024 |
| Stage 2 | GSM8K (full train), 3 epochs, max_len=512 |
| Prompt format | `User:` / `Assistant:` (consistent across SFT + eval) |
| Mul-tokens condition | Inject mul-tokens in Tulu assistant replies + optionally in GSM8K answers (`--inject-mul-tokens`) |

---

## Execution

### Resume Mul_Tokens Condition

Since baseline is complete, run only the mul_tokens condition:

```bash
cd /workspace/domain-token-gpt2
bash scripts/run_10b.sh --skip-baseline
```

This will:
1. Prepare FineWeb-Edu data with mul-token injection (~2 hours)
2. Pretrain GPT-2 124M on mul_tokens data (~4-6 hours)
3. Export to HuggingFace format (~1 minute)
4. SFT (Tulu → GSM8K) (time depends on subset size)
5. Evaluate on GSM8K and arithmetic probes (~30 minutes)

### Options

```bash
# Skip data preparation (if shards already exist)
bash scripts/run_10b.sh --skip-data --skip-baseline

# Run only baseline condition (already complete)
bash scripts/run_10b.sh --skip-mul-tokens

# Run both conditions from scratch
bash scripts/run_10b.sh
```

### Manual Execution (Individual Steps)

```bash
cd /workspace/domain-token-gpt2

# 1. Data preparation (baseline)
uv run python data/prepare_fineweb_pilot.py \
    --condition baseline \
    --out-dir data/fineweb_10b/baseline \
    --target-tokens 10000000000 \
    --shard-size 100000000

# 2. Pretraining with DDP
torchrun --standalone --nproc_per_node=4 pretrain/train_nanogpt.py \
    --config pretrain/configs/gpt2_124m_10b.yaml \
    --data-root data/fineweb_10b/baseline \
    --output-dir outputs/pretrain_baseline_10b \
    --condition baseline

# 3. Export to HuggingFace
uv run python pretrain/export_hf.py \
    --checkpoint outputs/pretrain_baseline_10b/model_19072.pt \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline

uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline

# 4a. SFT Stage 1: Tulu-3 mixture (instruction-tuning)
uv run python sft/sft_tulu.py \
    --model-path outputs/hf_baseline_10b \
    --output-dir outputs/sft_tulu_baseline \
    --condition baseline \
    --config sft/configs/tulu.yaml

# 4b. SFT Stage 2: GSM8K (math specialization)
uv run python sft/sft_gsm8k.py \
    --model-path outputs/sft_tulu_baseline \
    --output-dir outputs/sft_gsm8k_baseline \
    --condition baseline \
    --config sft/configs/gsm8k.yaml

# 5. Evaluation
uv run python eval/run_gsm8k.py \
    --model-path outputs/sft_gsm8k_baseline \
    --output-dir outputs/eval_baseline_10b \
    --condition baseline \
    --max-samples 1319

uv run python eval/run_arithmetic_probes.py \
    --model-path outputs/sft_gsm8k_baseline \
    --output-dir outputs/eval_baseline_10b \
    --condition baseline
```

---

## Timeline

### Actual (Baseline)
| Stage | Time | Notes |
|-------|------|-------|
| Data prep | ~1 hour | 12 shards, ~10B tokens |
| Pretraining | ~2.5 hours | 19,072 steps on 4× H200 |
| HF Export | ~1 minute | |
| SFT (Tulu → GSM8K) | varies | GSM8K is fast (~minutes); Tulu stage depends on subset size (see `sft/configs/tulu.yaml`) |
| Evaluation | ~23 minutes | 1,319 GSM8K + 281 arithmetic probes |
| **Total** | **~3.5 hours** | |

### Estimated (Mul_Tokens - Remaining)
| Stage | Estimated Time |
|-------|----------------|
| Data prep | ~2 hours |
| Pretraining | ~4-6 hours |
| HF Export | ~1 minute |
| SFT (Tulu → GSM8K) | varies | depends on Tulu subset size; GSM8K typically ~minutes |
| Evaluation | ~30 minutes |
| **Total** | **~7-9 hours** |

---

## Disk Space Requirements

| Item | Size |
|------|------|
| FineWeb-Edu shards (10B × 2 bytes × 2 conditions) | ~40 GB |
| Pretrain checkpoints (~5 per condition) | ~10 GB |
| HuggingFace models | ~1 GB |
| SFT outputs | ~2 GB |
| **Total** | **~55 GB** |

Verify space before running:
```bash
df -h /workspace
```

---

## Expected Outputs

```
outputs/
├── pretrain_baseline_10b/
│   ├── model_02000.pt
│   ├── model_04000.pt
│   ├── ...
│   ├── model_19072.pt      # Final checkpoint
│   ├── results.json
│   └── log.txt
├── pretrain_mul_tokens_10b/
│   └── ...
├── checkpoint_evals/
│   ├── baseline/
│   │   ├── checkpoint_results.json
│   │   ├── hellaswag_learning_curve.png
│   │   └── step_*_eval/hellaswag_results.json
│   └── mul_tokens/
│       ├── checkpoint_results.json
│       ├── hellaswag_learning_curve.png
│       └── step_*_eval/hellaswag_results.json
├── (optional) hf_baseline_10b/     # regenerated when re-running post-training
├── (optional) hf_mul_tokens_10b/   # regenerated when re-running post-training
├── (optional) sft_baseline_10b/    # regenerated when re-running post-training
├── (optional) sft_mul_tokens_10b/  # regenerated when re-running post-training
├── (optional) eval_baseline_10b/   # regenerated when re-running task evals
├── (optional) eval_mul_tokens_10b/ # regenerated when re-running task evals
└── (legacy) comparison_report_10b.json
```

---

## Success Criteria

Based on pilot results (6.17% mul-table accuracy vs 0% for baseline at 50M tokens):

1. **GSM8K accuracy**: Both conditions should show measurable (non-zero) accuracy at 10B tokens
2. **Multiplication table probes**: Mul_tokens condition should maintain or extend advantage
3. **Loss curves**: Comparable or lower perplexity for mul_tokens on math-heavy text
4. **Token efficiency**: Mul_tokens may generate fewer tokens for arithmetic expressions

---

## Monitoring During Training

### Watch training progress
```bash
tail -f outputs/pretrain_baseline_10b/log.txt
```

### Check GPU utilization
```bash
watch -n 1 nvidia-smi
```

### Monitor disk usage
```bash
watch -n 60 'df -h /workspace'
```

---

## Troubleshooting

### Out of Memory
Reduce `batch_size` in config from 32 to 16 or 8.

### Data streaming timeout
Check network connection. Consider using `--no-streaming` flag for local download (requires more disk).

### Checkpoint corruption
Resume from previous checkpoint:
```bash
torchrun ... --resume outputs/pretrain_baseline_10b/model_16000.pt
```

---

## Next Steps After Experiment

### ✅ Pretraining + checkpoint eval complete

Both baseline and mul_tokens **pretraining** runs are complete, and checkpoint HellaSwag evals are saved in:
- `outputs/checkpoint_evals/baseline/`
- `outputs/checkpoint_evals/mul_tokens/`

### ⚠️ Post-training metrics need rerun

HF export + SFT + GSM8K/probe eval should be re-run using the fixed exporter. The prior `outputs/comparison_report_10b.json` is kept as a small legacy artifact, but its task metrics were derived from broken exports and are **not trusted**.

### Recommended Next Steps

1. **Re-run post-training + eval** (low effort): re-export final checkpoints with fixed `pretrain/export_hf.py`, then rerun SFT + GSM8K/probes.
2. **Then decide on intervention tweaks**: e.g. SFT injection of mul-tokens (A1/A2 in `progress/07_bug_fix_route.md`).
3. **If pursuing rigor**: add seeds and/or scale model size.

### Key Questions to Address

- What happens to GSM8K/probes after re-running post-training with the fixed exporter?
- Does injecting mul-tokens into SFT targets increase mul-token usage and arithmetic accuracy?
- Is model size (124M) the limiting factor for arithmetic gains?

---

*✅ Pretraining + checkpoint eval are complete and validated. Post-training/task eval needs rerun with the fixed exporter before drawing conclusions about mul-tokens on GSM8K.*

