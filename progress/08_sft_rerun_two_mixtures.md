# Phase 4 (Step 8): SFT Rerun — Two Data Mixtures (Tulu-3 → GSM8K)

**Date:** December 27, 2025  
**Last Updated:** December 28, 2025  
**Status:** 🟢 Tulu SFT (baseline) complete; GSM8K SFT pending  
**Goal:** Re-run post-training SFT using the **fixed HF export** artifacts and a **two-stage SFT pipeline** to produce trustworthy GSM8K/probe results for baseline vs mul_tokens.

---

## Current Progress

| Stage | Baseline | Mul_Tokens |
|-------|----------|------------|
| Pretrained (10B) | ✅ Complete | ✅ Complete |
| Tulu-3 SFT | ✅ Complete | ⏳ Pending |
| GSM8K SFT | ⏳ Pending | ⏳ Pending |
| GSM8K Eval | ⏳ Pending | ⏳ Pending |
| Probes Eval | ⏳ Pending | ⏳ Pending |

### Baseline Tulu SFT Results
- **Train loss:** 1.6944
- **Eval loss:** 1.5728
- **Runtime:** ~70 minutes on RTX 4090
- **Samples:** 200k train, 1k val

---

## HuggingFace Hub Sync

All trained models are uploaded to HuggingFace Hub for cross-machine sync.

### Models on Hub

| Local Path | HuggingFace Repo | Status |
|------------|------------------|--------|
| `outputs/hf_baseline_10b` | [zhengpu-berkeley/domain-token-gpt2-baseline-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-baseline-10b) | ✅ |
| `outputs/hf_mul_tokens_10b` | [zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b) | ✅ |
| `outputs/sft_tulu_baseline` | [zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline) | ✅ |
| `outputs/sft_tulu_mul_tokens` | zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens | ⏳ Pending |
| `outputs/sft_gsm8k_baseline` | zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-baseline | ⏳ Pending |
| `outputs/sft_gsm8k_mul_tokens` | zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-mul-tokens | ⏳ Pending |

### Upload Models

```bash
# Upload a single model:
uv run python scripts/upload_to_hf.py \
    --model-path outputs/sft_tulu_baseline \
    --repo-name zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline

# Upload multiple models:
uv run python scripts/upload_to_hf.py \
    --model-path outputs/hf_baseline_10b outputs/hf_mul_tokens_10b \
    --repo-name zhengpu-berkeley/domain-token-gpt2-baseline-10b \
                zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b
```

### Download Models (On New Machine)

```bash
# List available models:
uv run python scripts/load_from_hf.py --list

# Download all available models:
uv run python scripts/load_from_hf.py --all

# Download specific models:
uv run python scripts/load_from_hf.py --models hf_baseline_10b sft_tulu_baseline

# Download by group:
uv run python scripts/load_from_hf.py --models pretrained  # Both pretrained models
uv run python scripts/load_from_hf.py --models baseline    # All baseline condition models
```

---

## Why This Rerun

Prior GSM8K/probe metrics were derived from **broken** nanoGPT → HuggingFace exports (see `progress/06_experiment_debug.md`). With the exporter fixed and HF checkpoints re-exported, we are re-running SFT so the eval numbers are **trusted**.

---

## What Changed Since Prior Attempts

- **Unified prompt format across SFT + eval:** all prompts now use:
  - `User: ...`
  - `Assistant: ...`
- **Two-stage SFT** (instruction-tuning then math specialization):
  1. **Tulu-3 mixture SFT** (general instruction-following)
  2. **GSM8K SFT** (math specialization)
- **Mul-token injection policy**
  - In `mul_tokens` condition:
    - Tulu stage: inject mul-tokens into **assistant** messages when applicable
    - GSM8K stage: inject mul-tokens into **answers** using `--inject-mul-tokens`
- **Config cleanup**:
  - `sft/configs/tulu.yaml` (Tulu stage)
  - `sft/configs/gsm8k.yaml` (GSM8K stage)

---

## The Two Mixtures (Conditions)

### Condition A — Baseline
- **Starting model:** `outputs/hf_baseline_10b/`
- **Tulu output:** `outputs/sft_tulu_baseline/`
- **GSM8K output:** `outputs/sft_gsm8k_baseline/`

### Condition B — Mul_Tokens
- **Starting model:** `outputs/hf_mul_tokens_10b/`
- **Tulu output:** `outputs/sft_tulu_mul_tokens/`
- **GSM8K output:** `outputs/sft_gsm8k_mul_tokens/`
- **Extra:** `--inject-mul-tokens` during GSM8K SFT

---

## Commands Used (4 Total)

### Baseline
```bash
uv run python sft/sft_tulu.py \
  --model-path outputs/hf_baseline_10b \
  --output-dir outputs/sft_tulu_baseline \
  --condition baseline \
  --config sft/configs/tulu.yaml

uv run python sft/sft_gsm8k.py \
  --model-path outputs/sft_tulu_baseline \
  --output-dir outputs/sft_gsm8k_baseline \
  --condition baseline \
  --config sft/configs/gsm8k.yaml
```

### Mul_Tokens
```bash
uv run python sft/sft_tulu.py \
  --model-path outputs/hf_mul_tokens_10b \
  --output-dir outputs/sft_tulu_mul_tokens \
  --condition mul_tokens \
  --config sft/configs/tulu.yaml

uv run python sft/sft_gsm8k.py \
  --model-path outputs/sft_tulu_mul_tokens \
  --output-dir outputs/sft_gsm8k_mul_tokens \
  --condition mul_tokens \
  --inject-mul-tokens \
  --config sft/configs/gsm8k.yaml
```

---

## Quick Qualitative Sampling

Use the SFT sampling script to test model outputs:

```bash
# Interactive mode:
uv run python sample/sample_sft.py \
    --model-path outputs/sft_tulu_baseline \
    --interactive

# Single question:
uv run python sample/sample_sft.py \
    --model-path outputs/sft_tulu_baseline \
    --question "What is 7 times 8?"

# Multiple samples with temperature:
uv run python sample/sample_sft.py \
    --model-path outputs/sft_tulu_baseline \
    --question "Explain photosynthesis." \
    --num-samples 3 \
    --temperature 0.7
```

---

## Expected Runtime (RTX 4090)

Observed on a 200k-sample Tulu run: ~70 minutes.

Estimates:
- **Tulu (200k samples)**: ~70 minutes per condition
- **GSM8K (7.5k samples, 3 epochs)**: ~10 minutes per condition
- **Total**: ~160 minutes for both conditions

---

## Expected Artifacts

### After Tulu stage finishes
Each directory should contain:
- `tulu_sft_results.json`
- HF model files (e.g. `config.json`, `model.safetensors`, tokenizer files)

### After GSM8K stage finishes
Each directory should contain:
- `sft_results.json`
- HF model files

**Note:** Intermediate checkpoints (e.g., `checkpoint-6248/`) can be deleted after training completes to save ~2.8GB per model.

---

## Next Steps

1. **Run Tulu SFT for mul_tokens condition**
   ```bash
   uv run python sft/sft_tulu.py \
     --model-path outputs/hf_mul_tokens_10b \
     --output-dir outputs/sft_tulu_mul_tokens \
     --condition mul_tokens \
     --config sft/configs/tulu.yaml
   ```

2. **Run GSM8K SFT for both conditions**
   ```bash
   # Baseline
   uv run python sft/sft_gsm8k.py \
     --model-path outputs/sft_tulu_baseline \
     --output-dir outputs/sft_gsm8k_baseline \
     --condition baseline \
     --config sft/configs/gsm8k.yaml

   # Mul_tokens
   uv run python sft/sft_gsm8k.py \
     --model-path outputs/sft_tulu_mul_tokens \
     --output-dir outputs/sft_gsm8k_mul_tokens \
     --condition mul_tokens \
     --inject-mul-tokens \
     --config sft/configs/gsm8k.yaml
   ```

3. **Upload completed models to HuggingFace**

4. **Run evaluations**
   - `eval/run_gsm8k.py` on both final GSM8K SFT outputs
   - `eval/run_arithmetic_probes.py` on both final GSM8K SFT outputs

5. **Compare results**
   - Use `scripts/compare_runs.py` to summarize deltas

---

## Known Warnings (Non-blocking)

- HF Trainer warns that passing `tokenizer=` is deprecated (future Transformers change).
- `huggingface/tokenizers` may warn about forked processes and parallelism; this is noisy but not a failure.
- `huggingface_hub` warns about deprecated `local_dir_use_symlinks` parameter; this is harmless.

