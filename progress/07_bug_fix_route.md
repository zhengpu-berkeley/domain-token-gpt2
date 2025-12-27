# Next Iteration Research Options

**Date:** December 27, 2024  
**Last Updated:** December 27, 2024  
**Context:** 10B-token pretraining complete; post-training (SFT/GSM8K) needs rerun after HF-export bug fix

---

## ✅ RESOLVED: HellaSwag Checkpoint Evaluation Issue

**Status:** ✅ Resolved

**What happened:** We initially saw training/val loss improve, but checkpoint HellaSwag appeared flat (~25% acc_norm).  
**Root cause:** A bug in nanoGPT → HuggingFace export (`pretrain/export_hf.py`): GPT‑2 `Conv1D` weights must be transposed vs `nn.Linear`, including the square `attn.c_proj.weight`. The exporter missed that, corrupting exported checkpoints and HF-based evals/SFT.

**Fix:** Patched the exporter and re-ran checkpoint evaluation. HellaSwag now improves over training.

**Current checkpoint HellaSwag (acc_norm, step 19072 / ~10B tokens):**
- Baseline: **0.3034**
- Mul_tokens: **0.3077**

Artifacts live at `outputs/checkpoint_evals/{baseline,mul_tokens}/`.

---

---

## Summary of Current State

| Metric | Baseline | Mul_Tokens | Note |
|--------|----------|------------|------|
| **Pretrain HellaSwag acc_norm (step 19072)** | **0.3034** | **0.3077** | From `outputs/checkpoint_evals/*` (post-fix) |
| GSM8K / arithmetic probes | — | — | Must be re-run after export fix (old numbers are not trusted) |
| Mul-token usage in generations | — | — | Must be re-run after post-training rerun |

---

## Option A: Fix the Signal Problem (SFT Data Augmentation)

**Hypothesis:** The model never learned to *output* mul-tokens because SFT data (GSM8K) contains no mul-tokens.

### A1. Mul-Token Injected SFT Data

**What:** Preprocess GSM8K training solutions to inject mul-tokens into the CoT (e.g., `6 * 9 = 54` → `<MUL_6_9_54>`).

**Effort:** Low (1-2 days)

**Implementation:**
```python
# In sft_gsm8k.py preprocessing:
def inject_mul_tokens_in_solution(solution: str) -> str:
    # Use existing MulExpressionInjector on solution text
    return injector.inject(solution)
```

**Expected outcome:** Model sees mul-tokens in target output during SFT, learns to produce them.

**Risk:** May not help if model embedding for mul-tokens wasn't learned well during pretrain.

---

### A2. Synthetic Mul-Token Drills in SFT

**What:** Add synthetic arithmetic problems to SFT that explicitly use mul-tokens.

**Example training sample:**
```
Q: What is 6 times 9?
A: <MUL_6_9_54>. The answer is 54.
```

**Effort:** Low-Medium (2-3 days)

**Rationale:** Forces model to associate arithmetic questions with mul-token outputs.

**Mix:** 50% GSM8K + 50% synthetic drills, or interleave.

---

### A3. Two-Stage SFT

**What:**
1. Stage 1: Train on synthetic mul-token drills only
2. Stage 2: Fine-tune on GSM8K (with or without mul-token injection)

**Effort:** Medium (3-4 days)

**Rationale:** Ensures mul-token associations are established before GSM8K adaptation.

---

## Option B: Increase Mul-Token Signal in Pretraining

**Hypothesis:** Mul-tokens were too rare during pretraining (~0.002% of tokens). The model never built strong representations.

### B1. 10x Mul-Token Injection Rate

**What:** Increase synthetic math augmentation from current level to ensure mul-tokens appear at least 1-2% of training tokens.

**Effort:** Medium (requires re-running pretraining ~4-6 hours on H200 cluster)

**Implementation:**
- Modify `data/prepare_fineweb_pilot.py` to inject more synthetic math drills
- Target: ~100M mul-token occurrences in 10B token run (1%)

**Risk:** May hurt general language modeling if math is too dominant.

---

### B2. Math-Heavy Pretraining Corpus

**What:** Replace FineWeb-Edu with a math-focused corpus:
- OpenWebMath (14B tokens of math web pages)
- DeepMind Mathematics dataset
- Synthetic arithmetic expressions

**Effort:** High (3-5 days for data prep + retraining)

**Rationale:** FineWeb-Edu has very little arithmetic; even with injection, mul-tokens are drowned out.

---

## Option C: Reinforce Mul-Token Usage with RL

**Hypothesis:** SFT alone is insufficient. RL can discover and reward mul-token usage.

### C1. GRPO with Mul-Token Bonus

**What:** Add a shaped reward that gives bonus points when mul-tokens appear in correct contexts:
```python
def reward(response: str, target: int) -> float:
    base_reward = 1.0 if extract_answer(response) == target else 0.0
    mul_token_bonus = 0.1 * count_mul_tokens(response)  # Up to +0.5 bonus
    return base_reward + mul_token_bonus
```

**Effort:** Medium (2-3 days)

**Risk:** May lead to mul-token spam if not balanced correctly.

---

### C2. Contrastive RL: Mul-Token vs Standard Responses

**What:** Generate paired rollouts (with/without mul-token prompting) and reward solutions that correctly use mul-tokens.

**Effort:** High (1 week)

**Rationale:** Explicitly teaches the model *when* mul-tokens are useful.

---

## Option D: Model Architecture / Size Changes

**Hypothesis:** 124M parameters is too small to benefit from specialized tokens.

### D1. Scale to GPT-2 Medium (350M)

**What:** Repeat experiment with 3x larger model.

**Effort:** Medium-High (3x compute cost, ~$150-200 for both conditions)

**Rationale:** Larger models may have capacity to learn token-task associations.

---

### D2. Specialized Embeddings for Mul-Tokens

**What:** Instead of random initialization, initialize mul-token embeddings as:
- Composition of operand embeddings: `emb(<MUL_6_9_54>) = f(emb(6), emb(9), emb(54))`
- Use a small MLP to combine operand embeddings

**Effort:** Medium (3-4 days)

**Rationale:** Gives mul-tokens meaningful starting representations instead of random noise.

---

## Option E: Alternative Tokenization Strategies

**Hypothesis:** The `<MUL_a_b_c>` format may be suboptimal.

### E1. Numeral Tokenization (Digit-Level)

**What:** Instead of mul-fact tokens, tokenize numbers digit-by-digit with positional markers:
- `54` → `[5:tens][4:ones]`
- Smaller vocab expansion (just 10 digit tokens × position markers)

**Effort:** Medium (3-4 days)

**References:**
- "Teaching Arithmetic to Small Transformers" (Lee et al., 2023)
- "Impact of Tokenization on LLM Arithmetic" (Singh et al., 2024)

---

### E2. Hybrid: Mul-Facts + Digit Tokens

**What:** Combine mul-fact tokens with digit-level tokenization for non-table multiplications.

**Effort:** High (1 week)

---

## Option F: Evaluation / Analysis First

**Hypothesis:** We need to understand *why* mul-tokens weren't used before changing the approach.

### F0. Fix HellaSwag Evaluation (URGENT)

**What:** Verify that our HellaSwag evaluation matches Karpathy's implementation exactly.

**Status:** ✅ Done — evaluator verified (matches Karpathy on OpenAI `gpt2`) and export bug fixed; checkpoints re-evaluated and show improving HellaSwag over training.

**Effort:** Low (2-4 hours)

**Priority:** **Completed**

---

### F1. Probe Mul-Token Embeddings

**What:** Analyze learned embeddings:
1. Check if mul-token embeddings cluster meaningfully (by operand, by result)
2. Compare with random baseline embeddings
3. Check attention patterns when mul-tokens appear in context

**Effort:** Low (1-2 days)

**Output:** Diagnostic report on whether mul-tokens were learned at all.

---

### F2. Generation Analysis with Forced Mul-Token Prefix

**What:** Prompt model with partial mul-token and see if it can complete:
```
Q: What is 6 × 9?
A: <MUL_6_9_
```

**Effort:** Low (1 day)

**Rationale:** Tests if model has learned mul-token semantics but fails to retrieve them.

---

### F3. Compare to OpenAI GPT-2 Baseline

**What:** Evaluate pretrained OpenAI GPT-2-124M on same benchmarks.

**Effort:** Very Low (1 hour)

**Rationale:** Establishes whether our 10B training is on par with original GPT-2 (which used ~40B tokens).

---

## Option G: Pivot the Research Question

**Hypothesis:** Mul-tokens for 1-9 table is too narrow. Broader domain tokenization may show clearer signal.

### G1. Formula Tokens for Common Equations

**What:** Extend beyond multiplication to include:
- `<SQUARE_n_n2>`: 5² = 25
- `<ADD_a_b_c>`: addition facts
- `<DIV_a_b_c>`: division facts
- Common formula templates: `<AREA_CIRCLE_r_A>`

**Effort:** High (1-2 weeks)

---

### G2. Code/Programming Domain Tokens

**What:** Test domain tokenization on code (e.g., common function signatures, API calls).

**Effort:** High (pivot to different evaluation)

---

### G3. Chinese Math Curriculum Tokens

**What:** Since the original inspiration was 九九乘法表, test with Chinese math instruction data where multiplication table is culturally embedded.

**Effort:** Medium (need Chinese math datasets)

---

## Recommended Priority Order

Based on effort/impact tradeoff:

| Priority | Option | Effort | Expected Impact | Rationale |
|----------|--------|--------|-----------------|-----------|
| **0** | **Re-run post-training + eval (SFT/GSM8K)** | Low | **CRITICAL** | Prior post-training metrics were derived from broken HF exports |
| 1 | **A1** | Low | Medium | Simplest fix — inject mul-tokens into SFT targets |
| 2 | **A2** | Low-Medium | Medium-High | Explicit training signal for mul-token usage |
| 3 | **C1** | Medium | High | RL can discover mul-token utility |
| 4 | **F1 + F2** | Low | Diagnostic | Understand mul-token (non-)usage mechanisms |
| 5 | **B1** | Medium | Medium | More pretrain signal (expensive) |
| 6 | **D1** | High | Unknown | Scale may help, but 3x compute cost |

**Note:** F0 is resolved; it is now safe to proceed with options A–D without retraining pretrain from scratch.

---

## Quick Wins to Try First

1. **Re-run post-training + eval** (low effort):
   - Re-export final checkpoints with fixed `pretrain/export_hf.py`
   - Re-run SFT + GSM8K/arithmetic eval to get trustworthy task metrics

2. **Run F3** (1 hour): Benchmark OpenAI GPT-2-124M on GSM8K and arithmetic probes to calibrate expectations

3. **Run F2** (1 day): Test if mul-tokens can be generated with forced prefix

4. **Implement A1** (1-2 days): Inject mul-tokens into GSM8K SFT data and re-run SFT + eval (only after F0 is resolved)

---

## Decision Matrix

| Constraint | Recommended Path |
|------------|------------------|
| **First Priority (ALL)** | Re-run post-training + eval (trustworthy GSM8K/probes) |
| **Minimal compute budget** | A1 → F2 → (optional) C1 |
| **Want publishable result fast** | A2 + C1 (synthetic drills + RL bonus) |
| **Have compute, want rigor** | B1 + D1 (more tokens + larger model) |
| **Exploratory research** | G1 (expand to formula tokens) |

---

## Files Created for Debugging

- `scripts/eval_checkpoints.py` — Evaluates all checkpoints on HellaSwag
- `scripts/plot_checkpoint_curve.py` — Visualizes HellaSwag accuracy over training (per condition)
- `scripts/visualize_pretrain_losses.py` — Visualizes train/val loss curves
- `eval/run_hellaswag.py` — Karpathy-compatible HellaSwag evaluation using tiktoken

---

*✅ F0 is resolved. The pretraining + checkpoint evaluation pipeline is now validated; remaining work is to re-run post-training/evals with the fixed exporter and then iterate on mul-token usage interventions.*
