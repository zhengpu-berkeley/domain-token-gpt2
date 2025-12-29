# Phase 4 (Step 8): SFT Rerun — Two Data Mixtures (Tulu-3 → GSM8K)

**Date:** December 27, 2025  
**Last Updated:** December 29, 2025  
**Status:** ✅ SFT complete; ✅ Mul-token bug fixed; ✅ Full eval done; ✅ GRPO with shaped rewards: mul_tokens 3.67% accuracy  
**Goal:** Re-run post-training SFT using the **fixed HF export** artifacts and a **two-stage SFT pipeline** to produce trustworthy GSM8K/probe results for baseline vs mul_tokens.

**Scope note:** This doc is about the **10B FineWeb pretrain → (Tulu → GSM8K) SFT** pipeline. TinyGSM scaling/distillation is tracked separately in `progress/11_tinygsm_distillation.md`.

---

## Current Progress

| Stage | Baseline | Mul_Tokens |
|-------|----------|------------|
| Pretrained (10B) | ✅ Complete | ✅ Complete |
| Tulu-3 SFT | ✅ Complete | ✅ Complete |
| GSM8K SFT | ✅ Complete | ✅ Complete |
| HellaSwag Eval | ✅ Complete (500 val samples) | ✅ Complete (500 val samples) |
| GSM8K Eval (subset) | ✅ Complete (200 test samples) | ✅ Complete (200 test samples) |
| GSM8K Eval (full) | ✅ Complete (1319 test samples) | ✅ Complete (1319 test samples) |
| Probes Eval | ✅ Complete (281 probes) | ✅ Complete (281 probes) |
| Mul-token visibility fix | ✅ Applied | ✅ Applied |

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

## Evaluation Results

### Subset Runs (Quick Iteration)

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

### Full GSM8K Evaluation (1319 Test Samples) — Post Bug-Fix

**Date:** December 28, 2025  
**Bug fix applied:** Mul-tokens now visible in decoded output; counting from raw token IDs.

| Metric | Baseline | Mul_Tokens |
|--------|----------|------------|
| **Accuracy** | **2.50%** (33/1319) | **2.05%** (27/1319) |
| Avg response tokens | 256.0 | 256.0 |
| Total mul-tokens used | 0 | **1,123** |
| Mul-tokens per response | 0 | **0.85** |
| Eval time (seconds) | 1,960 | 1,998 |

**Key Observations:**

1. **Bug fix verified:** Mul-tokens now appear in decoded output (e.g., `<MUL_2_3_6>=<<<MUL_2_3_6>>>6`) and are counted correctly from raw generated token IDs.

2. **Both conditions have ~2% GSM8K accuracy.** This is a training/model quality issue, not a measurement bug.

3. **All responses hit max_new_tokens=256.** Models enter repetitive loops and don't emit EOS tokens naturally.

4. **Mul-token usage is sparse:** Only 0.85 mul-tokens per response on average. Some responses use many (28-34), but most use none.

5. **Output quality issues:**
   - Repetitive calculations (same operation repeated)
   - Corrupted calculator annotations (<<expr=result>> format breaks down)
   - Circular reasoning without convergence to final answer

**Example mul-token usage (mul_tokens condition):**
```
He drives for <MUL_2_3_6>=<<<MUL_2_3_6>>>6 hours at 60 mph
He drives for <MUL_3_8_24>=<<<MUL_3_8_24>>>24 hours at 80 mph
```

### Artifacts

- Subset eval outputs:
  - `outputs/eval_hf_baseline_10b/`
  - `outputs/eval_sft_tulu_baseline/`
  - `outputs/eval_sft_gsm8k_baseline/`
  - `outputs/eval_hf_mul_tokens_10b/`
  - `outputs/eval_sft_tulu_mul_tokens/`
  - `outputs/eval_sft_gsm8k_mul_tokens/`
- Full GSM8K eval outputs (post bug-fix):
  - `outputs/eval_sft_gsm8k_baseline_full/`
  - `outputs/eval_sft_gsm8k_mul_tokens_full/`

---

## Mul-Token Visibility Bug — RESOLVED ✅

### Original Issue (Dec 27)
The mul_tokens condition tokenizer marked `<MUL_...>` as *special tokens* via `add_special_tokens()`. This caused:
- `tokenizer.decode(..., skip_special_tokens=True)` to **drop** them
- Eval scripts couldn't see or count mul-tokens in decoded output

### Fix Applied (Dec 28)

**Belt-and-suspenders approach:**

1. **Tokenizer builder fix** (`tokenizer/hf_gpt2_with_mul.py`):
   - Changed mul-tokens from `add_special_tokens()` to `add_tokens()` for both conditions
   - Mul-tokens are now **regular tokens**, not special tokens

2. **Eval scripts fix** (`eval/run_gsm8k.py`, `eval/run_arithmetic_probes.py`):
   - Changed `skip_special_tokens=True` → `skip_special_tokens=False`
   - Count mul-tokens directly from raw generated token IDs (not re-encoded text)

3. **Existing model fix** (`scripts/fix_tokenizers.py`):
   - Created script to regenerate tokenizer files in existing model directories
   - Applied to: `hf_mul_tokens_10b`, `sft_tulu_mul_tokens`, `sft_gsm8k_mul_tokens`

4. **HuggingFace Hub sync**:
   - Re-uploaded corrected tokenizers to all 3 affected repos

### Verification
After fix, mul-tokens appear correctly in decoded output:
```
He drives for <MUL_2_3_6>=<<<MUL_2_3_6>>>6 hours at 60 mph
```
And counting works: 1,123 total mul-tokens across 1,319 responses (0.85 per response).

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

## Completed Tasks (Dec 28, 2025)

| Task | Status |
|------|--------|
| Fix mul-token visibility (tokenizer builder) | ✅ Done |
| Fix mul-token visibility (eval scripts) | ✅ Done |
| Create `scripts/fix_tokenizers.py` for existing models | ✅ Done |
| Apply fix to 3 mul_tokens model directories | ✅ Done |
| Re-upload corrected tokenizers to HuggingFace | ✅ Done |
| Run full GSM8K eval (1319 samples, both conditions) | ✅ Done |

---

## Next Steps

### Immediate Analysis Questions

1. **Why is GSM8K accuracy so low (~2%)?**
   - Both conditions degenerate into repetitive loops
   - All responses hit max_new_tokens=256 (no natural EOS)
   - SFT training may need hyperparameter tuning (lr, epochs, batch size)
   - The base model (124M params) may be too small for GSM8K without RL

2. **Is the mul-token condition learning to use the tokens?**
   - Yes, but sparsely: 0.85 mul-tokens per response on average
   - Some responses use many (28-34 tokens), but most use none
   - The model may need stronger supervision to prefer mul-tokens

### Potential Next Steps

1. **RL Stage (GRPO)**
   - Use reward signal to correct looping behavior
   - Encourage EOS emission and correct final answers
   - May help mul_tokens condition learn to prefer mul-tokens for arithmetic

2. **Ablation: Increase max_new_tokens**
   - Try 512 or 1024 to see if models eventually terminate
   - Useful for debugging but won't fix root cause

3. **Training diagnostics**
   - Check SFT training loss curves for signs of underfitting
   - Consider longer training or different learning rate

4. **Inference-time injection experiment**
   - Apply `MulExpressionInjector` to prompts at eval time
   - Test "tooling + tokens" vs "tokens only" hypothesis

---

---

## GRPO Reinforcement Learning Experiment (Dec 28, 2025)

### Motivation

With ~2% GSM8K accuracy from SFT, we hypothesized that RL (GRPO) might help the model:
1. Learn to emit EOS tokens instead of looping
2. Discover how to leverage mul-tokens for step-by-step reasoning
3. Improve accuracy through reward-based learning

### Pipeline

**Stage 1: GSM-SBS Warm-Start SFT**
- Created `data/gsm_sbs/` dataset: 640 examples teaching step-by-step multiplication decomposition
- Format: `12 × 12 = (10 + 2) × 12 = 10×12 + 2×12 = 120 + 24 = 144`
- Recursive decomposition until single-digit × single-digit base cases
- SFT for 2 epochs on both conditions (~20 seconds each)

**Stage 2: GRPO Training**
- Config: `rl/configs/grpo_optimized.yaml`
- 1000 samples, batch_size=8, num_generations=8
- Binary reward: 1.0 for exact answer match, 0.0 otherwise
- KL penalty (beta=0.1)
- ~14 minutes per condition on RTX 4090

### GRPO Results

| Metric | Baseline GRPO | Mul_tokens GRPO | Pre-GRPO SFT |
|--------|--------------|-----------------|--------------|
| **Accuracy** | **2.2%** (11/500) | **1.2%** (6/500) | ~2.05% |
| Avg Response Tokens | 256 (max) | 256 (max) | ~256 |
| Mul Token Count | 0 | 128 | ~1,123 |
| Mul Tokens/Response | 0.0 | 0.256 | 0.85 |

### Diagnosis: Mode Collapse

**The GRPO training caused model degradation rather than improvement.**

1. **Mode Collapse**: Both models hit max tokens (256) on every response, generating repetitive loops:
   ```
   She makes 48 eggs because 48 x 12 = <<48×48=48=48>>48
   #### 48×48=48
   She will eat 48 eggs because 48 x 12 = <<48×48=48=48>>48
   ...
   ```

2. **Sparse Reward Problem**: With ~2% base accuracy, 98% of completions receive zero reward. The model has no useful gradient signal for most samples.

3. **Mul-tokens Partially Working**: The mul_tokens model generates some mul-tokens (128 total), but uses them incorrectly:
   ```
   Brandon's iPhone is <MUL_2_4_8> = <<<MUL_2_4_8>>>8 years old.
   The age difference is 8 - 2 = <<8-2=6>>6 years.
   #### 6 years old  # Wrong answer (should be 8)
   ```
   - The model outputs the token but doesn't integrate the value into reasoning

4. **KL Penalty Insufficient**: The model drifted too far from the SFT distribution despite β=0.1.

### Artifacts

- GSM-SBS dataset: `data/gsm_sbs/`
- GSM-SBS SFT models: `outputs/sft_sbs_baseline/`, `outputs/sft_sbs_mul_tokens/`
- GRPO models: `outputs/grpo_baseline/`, `outputs/grpo_mul_tokens/`
- GRPO configs: `rl/configs/grpo_optimized.yaml`, `rl/configs/grpo_pilot.yaml`
- Eval results: `outputs/eval_grpo_baseline/`, `outputs/eval_grpo_mul_tokens/`

### Lessons Learned

1. **Sparse rewards don't work for RL** when base accuracy is <5%
2. **Need intermediate rewards** for step-by-step reasoning
3. **KL regularization** needs to be stronger (β > 0.1) for small models
4. **More SFT warm-start** may be needed before RL can help

### Next Steps

1. ~~**More GSM-SBS SFT**: Train more epochs on step-by-step dataset~~ ✅ Tried, didn't help
2. ~~**Shaped Rewards**: Add partial credit for correct intermediate steps~~ ✅ Implemented and tested
3. ~~**Stronger KL**: Try β=0.5 or higher to prevent mode collapse~~ ✅ Used β=0.5
4. **Rejection Sampling**: Alternative to GRPO that may work better with sparse rewards

---

## GRPO with Shaped Rewards (Dec 28, 2025)

Following the mode collapse with binary rewards, we implemented shaped rewards with stronger KL.

### Configuration (`rl/configs/grpo_shaped.yaml`)

- **KL penalty**: β=0.5 (5x stronger than original 0.1)
- **Shaped rewards**:
  - Correct answer: +1.0
  - Partial answer (within 10%): +0.3
  - Shows work (has = or ####): +0.1
  - Uses mul-tokens: +0.2
  - Short response (<150 words): +0.1
  - Repetition penalty: -0.3

### Results

| Condition | Reward Type | Accuracy | Mul Tokens/Response |
|-----------|-------------|----------|---------------------|
| Baseline | Binary | 2.2% | 0.0 |
| Mul_tokens | Binary | 1.2% | 0.26 |
| Baseline | Shaped | 2.33% | 0.0 |
| **Mul_tokens** | **Shaped** | **3.67%** | **1.03** |

**Key Findings:**

1. **Shaped rewards dramatically improved mul_tokens**: 3.67% vs 1.2% (3x improvement!)
2. **Mul-token usage increased 4x**: 1.03 vs 0.26 per response
3. **Baseline essentially unchanged**: Shaped rewards don't help without the token bonus
4. **The mul-token bonus provides useful learning signal**

### Interpretation

The mul-token condition benefits from shaped rewards because:
- The `uses_mul_tokens: +0.2` bonus provides gradient signal even when answers are wrong
- This encourages the model to learn when/how to use mul-tokens
- The increased mul-token usage correlates with improved accuracy

This suggests **domain-specific tokens can be leveraged through RL** when proper reward shaping is used.

### Artifacts

- Shaped GRPO config: `rl/configs/grpo_shaped.yaml`
- Models: `outputs/grpo_shaped_baseline/`, `outputs/grpo_shaped_mul_tokens/`
- Eval results: `outputs/eval_grpo_shaped_baseline/`, `outputs/eval_grpo_shaped_mul_tokens/`

---

## Known Warnings (Non-blocking)

- HF Trainer warns that passing `tokenizer=` is deprecated (future Transformers change).
- `huggingface/tokenizers` may warn about forked processes and parallelism; this is noisy but not a failure.
- `huggingface_hub` warns about deprecated `local_dir_use_symlinks` parameter; this is harmless.

