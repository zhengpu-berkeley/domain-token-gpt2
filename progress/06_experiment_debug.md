# Experiment Debugging & HellaSwag Validation

**Status:** ✅ Resolved — HF export bug fixed; checkpoint HellaSwag improves over training

---

## Executive Summary

We hit a scary mismatch: **training/val loss fell** (≈11 → ≈3), but **HellaSwag accuracy over exported checkpoints** stayed ~25% (near random). This was a *false alarm* caused by a bug in our nanoGPT → HuggingFace export path.

**Root cause:** HuggingFace GPT-2 uses `Conv1D` for projections. Its weight layout differs from `nn.Linear`, and our exporter failed to transpose some weights when the matrix is square (notably `attn.c_proj.weight`). That produced incorrect HF models, which made HellaSwag look flat/random.

After fixing `pretrain/export_hf.py` and re-running checkpoint evaluation, HellaSwag **improves with training** for both conditions.

---

## Current Source-of-Truth Artifacts

- **Pretrain checkpoints (raw)**:
  - `outputs/pretrain_baseline_10b/`
  - `outputs/pretrain_mul_tokens_10b/`
- **Checkpoint HellaSwag evals (fixed export)**:
  - `outputs/checkpoint_evals/baseline/`
  - `outputs/checkpoint_evals/mul_tokens/`
  - Includes per-step JSONs (`step_<N>_eval/hellaswag_results.json`) and per-condition PNGs (`hellaswag_learning_curve.png`)

## What Was Wrong (HF Export)

HuggingFace GPT-2 stores projection weights as `Conv1D` matrices shaped **(in_features, out_features)**, while our training model uses `nn.Linear` weights shaped **(out_features, in_features)**.

Our exporter previously transposed only when shapes mismatched. That misses the case where shapes match but a transpose is still required (e.g., square 768×768). This corrupted HF exports and all HF-based downstream evals.

## Validation / Sanity Checks

✅ **Evaluator correctness**: `eval/run_hellaswag.py` matches Karpathy’s reference on OpenAI `gpt2` (≈0.2955 acc_norm).  
✅ **Export correctness**: After the fix, logits from raw checkpoints match logits from the exported HF model (numerical noise only).  
✅ **Learning curve**: HellaSwag accuracy increases over training steps for both conditions.

### Checkpoint HellaSwag (acc_norm)

From `outputs/checkpoint_evals/{baseline,mul_tokens}/checkpoint_results.json`:

| Step | Tokens (B) | Baseline | Mul_tokens |
|------|------------|----------|------------|
| 2000 | 1.05 | 0.2604 | 0.2618 |
| 10000 | 5.24 | 0.2929 | 0.2932 |
| 19072 | 10.00 | 0.3034 | 0.3077 |

## Implication for Prior GSM8K / SFT Results

Any GSM8K/SFT/“final model” metrics produced **before** the export fix should be treated as **invalid**, because they were derived from incorrect HF exports. Those large HF/SFT artifacts were deleted during disk cleanup and can be regenerated if/when we re-run post-training.

---

## Next Steps (When You Continue)

1. If you want **post-training + GSM8K** numbers, re-export final checkpoints using the fixed exporter and re-run SFT/evals.
2. If you want to iterate on mul-tokens, now you can do so without worrying the pretraining/eval pipeline is broken.

