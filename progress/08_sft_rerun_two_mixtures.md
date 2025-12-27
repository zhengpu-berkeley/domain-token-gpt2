# Phase 4 (Step 8): SFT Rerun — Two Data Mixtures (Tulu-3 → GSM8K)

**Date:** December 27, 2025  
**Status:** 🟡 Running (kicked off on RTX 4090)  
**Goal:** Re-run post-training SFT using the **fixed HF export** artifacts and a **two-stage SFT pipeline** to produce trustworthy GSM8K/probe results for baseline vs mul_tokens.

---

## Why This Rerun

Prior GSM8K/probe metrics were derived from **broken** nanoGPT → HuggingFace exports (see `progress/06_experiment_debug.md`). With the exporter fixed and HF checkpoints re-exported, we are re-running SFT so tomorrow’s eval numbers are **trusted**.

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

## Expected Runtime (RTX 4090)

Observed on a 10k-sample Tulu run (same hyperparams, same model): ~3.5 minutes.

Extrapolated estimates:
- **Tulu (200k samples)**: ~70 minutes per condition
- **GSM8K (7.5k samples, 3 epochs)**: ~10 minutes per condition
- **Total**: ~160 minutes for both conditions

---

## Expected Artifacts (What to Check Tomorrow)

### After Tulu stage finishes
Each directory should contain:
- `tulu_sft_results.json`
- HF model files (e.g. `config.json`, `model.safetensors`, tokenizer files)

### After GSM8K stage finishes
Each directory should contain:
- `sft_results.json`
- HF model files

---

## Next Steps (Tomorrow)

1. **Sanity-check checkpoints exist**
   - `outputs/sft_tulu_{baseline,mul_tokens}/tulu_sft_results.json`
   - `outputs/sft_gsm8k_{baseline,mul_tokens}/sft_results.json`
2. **Run evaluations**
   - `eval/run_gsm8k.py` on both final GSM8K SFT outputs
   - `eval/run_arithmetic_probes.py` on both final GSM8K SFT outputs
3. **Quick qualitative sampling**
   - Use `sample/text_completion.py` on both final models
4. **Compare results**
   - Use `scripts/compare_runs.py` to summarize deltas

---

## Known Warnings (Non-blocking)

- HF Trainer warns that passing `tokenizer=` is deprecated (future Transformers change).
- `huggingface/tokenizers` may warn about forked processes and parallelism; this is noisy but not a failure.


