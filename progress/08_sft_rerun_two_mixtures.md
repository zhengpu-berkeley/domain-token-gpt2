# Phase 4 (Step 8): SFT Rerun — Two Data Mixtures (Tulu-3 → GSM8K)

**Date:** December 27, 2025  
**Last Updated:** December 28, 2025  
**Status:** ✅ All SFT runs complete (baseline + mul_tokens); ✅ evals complete (subset); ⚠️ mul-token visibility issue in decoding (see below)  
**Goal:** Re-run post-training SFT using the **fixed HF export** artifacts and a **two-stage SFT pipeline** to produce trustworthy GSM8K/probe results for baseline vs mul_tokens.

---

## Current Progress

| Stage | Baseline | Mul_Tokens |
|-------|----------|------------|
| Pretrained (10B) | ✅ Complete | ✅ Complete |
| Tulu-3 SFT | ✅ Complete | ✅ Complete |
| GSM8K SFT | ✅ Complete | ✅ Complete |
| HellaSwag Eval | ✅ Complete (500 val samples) | ✅ Complete (500 val samples) |
| GSM8K Eval | ✅ Complete (200 test samples) | ✅ Complete (200 test samples) |
| Probes Eval | ✅ Complete (281 probes) | ✅ Complete (281 probes) |

### SFT Results (Loss)

| Stage | Condition | Train loss | Eval loss |
|------|-----------|-----------:|----------:|
| Tulu-3 SFT | baseline | 1.6944 | 1.5728 |
| Tulu-3 SFT | mul_tokens | 1.6923 | 1.5721 |
| GSM8K SFT (3 epochs) | baseline | 1.5432 | 1.5652 |
| GSM8K SFT (3 epochs) | mul_tokens | 1.5667 | 1.5894 |

---

## HuggingFace Hub Sync

All trained models are uploaded to HuggingFace Hub for cross-machine sync.

### Models on Hub

| Local Path | HuggingFace Repo | Status |
|------------|------------------|--------|
| `outputs/hf_baseline_10b` | [zhengpu-berkeley/domain-token-gpt2-baseline-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-baseline-10b) | ✅ |
| `outputs/hf_mul_tokens_10b` | [zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b) | ✅ |
| `outputs/sft_tulu_baseline` | [zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline) | ✅ |
| `outputs/sft_gsm8k_baseline` | [zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-baseline](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-baseline) | ✅ |
| `outputs/sft_tulu_mul_tokens` | [zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens) | ✅ |
| `outputs/sft_gsm8k_mul_tokens` | [zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-mul-tokens](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-mul-tokens) | ✅ |

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
uv run python scripts/load_from_hf.py --models mul_tokens  # All mul_tokens condition models

# NOTE: by default we do NOT download Trainer checkpoint-* folders (multi-GB).
# If you need them, opt in:
uv run python scripts/load_from_hf.py --models mul_tokens --include-checkpoints
```

---

## Evaluation Results (Subset Runs)

**Important:** These are *subset* evaluations for quick iteration:
- HellaSwag: **500** validation examples
- GSM8K: **200** test examples
- Probes: **281** synthetic probes (full suite)

| condition | stage | hellaswag_acc_norm | gsm8k_acc | probes_overall | probes_mul_table | probes_mul_multidigit | probes_addition |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | pretrained | 0.3520 | 0.0200 | 0.0427 | 0.1481 | 0.0000 | 0.0000 |
| baseline | tulu_sft | 0.3320 | 0.0150 | 0.0249 | 0.0864 | 0.0000 | 0.0000 |
| baseline | gsm8k_sft | 0.3420 | 0.0250 | 0.2206 | 0.6790 | 0.0500 | 0.0200 |
| mul_tokens | pretrained | 0.3600 | 0.0200 | 0.0356 | 0.1235 | 0.0000 | 0.0000 |
| mul_tokens | tulu_sft | 0.3760 | 0.0150 | 0.0463 | 0.1605 | 0.0000 | 0.0000 |
| mul_tokens | gsm8k_sft | 0.3500 | 0.0400 | 0.1566 | 0.3704 | 0.1200 | 0.0200 |

Artifacts:
- Baseline eval outputs:
  - `outputs/eval_hf_baseline_10b/`
  - `outputs/eval_sft_tulu_baseline/`
  - `outputs/eval_sft_gsm8k_baseline/`
- Mul-tokens eval outputs:
  - `outputs/eval_hf_mul_tokens_10b/`
  - `outputs/eval_sft_tulu_mul_tokens/`
  - `outputs/eval_sft_gsm8k_mul_tokens/`

---

## Mul-Token “Internal Usage” Check (Tokenization vs Emission)

### What we verified
- **Tokenization:** The tokenizer encodes the literal string `<MUL_6_9_54>` as **one token** (ID 50342) in *all* models.
- **No automatic conversion:** Raw text like `6*9=54` does **not** tokenize into `<MUL_...>`; it stays normal GPT-2 tokens. The **injector** must rewrite text.
- **Emission (generation IDs):** On several direct prompts (e.g., “What is 6 times 9?”) we saw **0 generated token IDs** in the mul-token ID range (50304–50348) for both baseline and mul_tokens models.

### ⚠️ Critical gotcha: mul_tokens tokenizer marks `<MUL_...>` as *special tokens*
In the exported HF tokenizer for **mul_tokens** models, the 45 mul tokens are stored under `additional_special_tokens` (see `outputs/sft_gsm8k_mul_tokens/special_tokens_map.json`). That means:
- `tokenizer.decode(..., skip_special_tokens=True)` will **drop** them.
- Our eval scripts currently decode with `skip_special_tokens=True`, so **mul-token string visibility and mul-token counting-from-decoded-text are unreliable**.

Root cause in `tokenizer/hf_gpt2_with_mul.py`:
- baseline: `tokenizer.add_tokens(mul_token_strs)`
- mul_tokens: `tokenizer.add_special_tokens({'additional_special_tokens': mul_token_strs})`

**Tomorrow fix idea (recommended):** treat mul tokens as *regular tokens* in both conditions (or, alternatively, change eval scripts to decode with `skip_special_tokens=False` and count mul-token IDs directly from generated IDs).

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

## Tomorrow Handoff Checklist (Dec 29, 2025)

1. **Fix mul-token visibility + counting**
   - Decide one of:
     - **Preferred:** modify `tokenizer/hf_gpt2_with_mul.py` so mul tokens are added via `add_tokens()` for both conditions; then re-save tokenizer files into the model dirs and re-upload tokenizer-only changes to HF.
     - **Alternative:** modify eval scripts to decode with `skip_special_tokens=False` and count mul-token IDs from generated token IDs (not from decoded text).

2. **Rerun evals with corrected mul-token visibility**
   - At least on the **mul_tokens** models to measure whether they *ever* emit `<MUL_...>` once decoding doesn’t hide them.

3. **Run full GSM8K eval**
   - Replace `--max-samples 200` with `--max-samples 1319` for both GSM8K-SFT models (baseline + mul_tokens).

4. **Generate a proper comparison report**
   - Run `scripts/compare_runs.py` (and/or add a small report script) to emit a single markdown table for the writeup.

5. **Optional: inference-time injector experiment**
   - Apply `MulExpressionInjector` to evaluation prompts for mul_tokens condition (and keep baseline untouched) to test “tooling + tokens” vs “tokens only”.

---

## Known Warnings (Non-blocking)

- HF Trainer warns that passing `tokenizer=` is deprecated (future Transformers change).
- `huggingface/tokenizers` may warn about forked processes and parallelism; this is noisy but not a failure.
- `huggingface_hub` warns about deprecated `local_dir_use_symlinks` parameter; this is harmless.

