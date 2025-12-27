# Phase 3: 10B Token Full Experiment — Execution Status

**Date:** December 27, 2024  
**Status:** Baseline Complete, Mul_Tokens Pending  
**Hardware:** 4× NVIDIA H200 GPUs (143GB VRAM each)

---

## Current Status

### ✅ Baseline Condition — COMPLETE

| Stage | Status | Details |
|-------|--------|---------|
| Data Preparation | ✅ Complete | 12 shards, ~10B tokens |
| Pretraining | ✅ Complete | 19,072 steps, final loss: 3.01 (train) / 3.12 (val) |
| HF Export | ✅ Complete | Saved to `outputs/hf_baseline_10b/` |
| SFT | ✅ Complete | Train loss: 6.47, Eval loss: 5.37 |
| Evaluation | ✅ Complete | See results below |

**Baseline Results:**
- **GSM8K Accuracy:** 0.23% (3/1319)
- **Arithmetic Probes:**
  - Overall: 0.71% (2/281)
  - Mul-table: 2.47% (2/81)
  - Mul-multidigit: 0.00% (0/100)
  - Addition: 0.00% (0/100)

### ⏸️ Mul_Tokens Condition — NOT STARTED

Data preparation was initiated but stopped for refinements. No mul_tokens data or models exist yet.

---

## Overview

All infrastructure for the 10B token compute-matched experiment is in place. This document summarizes the setup, execution status, and provides instructions for completing the mul_tokens condition.

---

## Baseline Results Summary

### Pretraining Metrics
- **Total Steps:** 19,072 / 19,073 (99.9% - final step completed)
- **Total Tokens:** 9,999,745,024 (~10B)
- **Final Train Loss:** 3.01
- **Final Val Loss:** 3.12
- **Training Time:** ~2.5 hours (4× H200 DDP)
- **Checkpoints Saved:** 5 (steps 2000, 4000, 8000, 10000, 14000, 18000, 19072)

### SFT Metrics
- **Train Loss:** 6.47
- **Eval Loss:** 5.37
- **Training Time:** ~2 minutes

### Evaluation Results
- **GSM8K Test Set:** 0.23% accuracy (3/1319 correct)
- **Arithmetic Probes:** 0.71% overall accuracy
  - Mul-table: 2.47% (2/81) — both correct answers were "3" (1×3 and 3×1)
  - Mul-multidigit: 0.00% (0/100)
  - Addition: 0.00% (0/100)

**Key Observations:**
- Model shows minimal arithmetic capability at 124M parameters
- Mul-table performance (2.47%) is slightly above random chance for single-digit multiplication
- Model struggles with multi-digit operations and addition
- GSM8K performance is very low, as expected for a 124M model

---

## Files Created

| File | Purpose |
|------|---------|
| `pretrain/configs/gpt2_124m_10b.yaml` | 10B token pretraining config for 4× H200 DDP |
| `sft/configs/sft_full.yaml` | Full experiment SFT configuration |
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

### SFT (sft_full.yaml)

| Parameter | Value |
|-----------|-------|
| Dataset | GSM8K (full training set) |
| Epochs | 3 |
| Batch size | 8 per GPU |
| Gradient accumulation | 2 |
| Learning rate | 2e-5 |
| Max length | 512 tokens |

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
4. SFT on GSM8K (~30-60 minutes)
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
    --checkpoint outputs/pretrain_baseline_10b/model_19073.pt \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline

uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline

# 4. SFT
uv run python sft/sft_train.py \
    --model-path outputs/hf_baseline_10b \
    --output-dir outputs/sft_baseline_10b \
    --condition baseline \
    --config sft/configs/sft_full.yaml

# 5. Evaluation
uv run python eval/run_gsm8k.py \
    --model-path outputs/sft_baseline_10b \
    --output-dir outputs/eval_baseline_10b \
    --condition baseline \
    --max-samples 1319

uv run python eval/run_arithmetic_probes.py \
    --model-path outputs/sft_baseline_10b \
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
| SFT | ~2 minutes | 3 epochs, 7,473 samples |
| Evaluation | ~23 minutes | 1,319 GSM8K + 281 arithmetic probes |
| **Total** | **~3.5 hours** | |

### Estimated (Mul_Tokens - Remaining)
| Stage | Estimated Time |
|-------|----------------|
| Data prep | ~2 hours |
| Pretraining | ~4-6 hours |
| HF Export | ~1 minute |
| SFT | ~30-60 minutes |
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
│   ├── model_19073.pt      # Final checkpoint
│   ├── results.json
│   └── log.txt
├── pretrain_mul_tokens_10b/
│   └── ...
├── hf_baseline_10b/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── ...
├── hf_mul_tokens_10b/
│   └── ...
├── sft_baseline_10b/
│   ├── pytorch_model.bin
│   └── sft_results.json
├── sft_mul_tokens_10b/
│   └── ...
├── eval_baseline_10b/
│   ├── gsm8k_results.json
│   └── arithmetic_results.json
├── eval_mul_tokens_10b/
│   └── ...
└── comparison_report_10b.json
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

1. **Analyze results**: Run `scripts/compare_runs.py` to generate comparison
2. **Statistical significance**: Consider running additional seeds (2 more for N=3)
3. **GRPO RL**: If SFT results are promising, add GRPO stage
4. **Write-up**: Document findings in research report

---

## Next Steps

1. **Complete mul_tokens condition:** Run `bash scripts/run_10b.sh --skip-baseline`
2. **Compare results:** After mul_tokens completes, run `scripts/compare_runs.py`
3. **Analyze findings:** Compare baseline vs mul_tokens performance on arithmetic probes

---

*Baseline condition complete. Mul_tokens condition pending. Run `bash scripts/run_10b.sh --skip-baseline` to continue.*

