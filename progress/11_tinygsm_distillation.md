# Phase 7: TinyGSM Distillation - Natural Language CoT Conversion

**Date:** December 28, 2025  
**Status:** 🚧 In progress — 100K pipeline completed (baseline + mul_tokens); next step is scaling beyond 100K  
**Goal:** Leverage TinyGSM's 12.3M synthetic problems to dramatically improve GSM8K accuracy (and test mul-token lift under enough math data)

**Last Updated:** 2025-12-29

---

## Background & Motivation

### The TinyGSM Breakthrough

[TinyGSM (Liu et al., 2023)](https://arxiv.org/pdf/2312.09241) demonstrated:

| Model Size | TinyGSM Training | GSM8K Accuracy |
|------------|------------------|----------------|
| 125M | 12.3M problems | **63.1%** |
| 350M | 12.3M problems | **65.9%** |
| 125M + verifier | Same + verifier | **68.9%** |

**Key insight:** Data scale and quality >> model size for small models.

### Our Current State

| Model | Training Data | GSM8K Accuracy |
|-------|---------------|----------------|
| GPT-2 124M (ours) | ~25K examples | **2-4%** |
| Phi-1.5 125M (TinyGSM) | 12.3M examples | **63%** |

**Gap:** We have 500x less training data. TinyGSM proves that a 125M model CAN do math reasoning with enough high-quality data.

---

## The Plan

### Step 1: Download TinyGSM Dataset

```python
from datasets import load_dataset
dataset = load_dataset("TinyGSM/TinyGSM")
```

TinyGSM format:
```python
def solution():
    """Sarah is buying supplies for a party. She buys 10 trays of food 
    at $50 each and 8 cases of beverages at $20 each. How much does 
    she spend in total?"""
    trays = 10
    tray_cost = 50
    cases = 8
    case_cost = 20
    tray_total = trays * tray_cost
    case_total = cases * case_cost
    total_cost = tray_total + case_total
    result = total_cost
    return result
```

### Step 2: Convert Python → Natural Language CoT

**Why convert?** Our models are trained on natural language, not Python. We need:

**Before (Python):**
```python
trays = 10
tray_cost = 50
tray_total = trays * tray_cost  # 500
cases = 8
case_cost = 20
case_total = cases * case_cost  # 160
total_cost = tray_total + case_total  # 660
return total_cost
```

**After (Natural Language CoT):**
```
Sarah buys 10 trays at $50 each, so the trays cost 10 × 50 = $500.
She buys 8 cases of beverages at $20 each, so the beverages cost 8 × 20 = $160.
The total cost is 500 + 160 = $660.
#### 660
```

**With mul-tokens (for mul_tokens condition):**
```
Sarah buys 10 trays at $50 each. Since 10 is a two-digit number, let's break this down:
10 × 50 = 10 × 5 × 10 = 50 × 10 = $500.
She buys 8 cases at $20 each, so the beverages cost <MUL_8_2_16> × 10 = $160.
The total cost is 500 + 160 = $660.
#### 660
```

### Step 3: LLM-Based Conversion Pipeline

Rule-based conversion won't capture reasoning narrative well. We'll use an LLM:

**Option A: GPT-4o-mini** (~$0.15/1M input, $0.60/1M output)
- 100K examples ≈ $10-20
- Fast, reliable

**Option B: Claude Haiku** (~$0.25/1M input, $1.25/1M output)
- 100K examples ≈ $15-30
- Also good quality

**Conversion prompt:**
```
Convert this Python math solution into natural language step-by-step reasoning.

Python code:
{code}

Question: {question}

Output a natural language explanation that:
1. Explains each calculation step in words
2. Shows the arithmetic (e.g., "8 × 20 = 160")
3. Ends with "#### {final_answer}"

Natural language solution:
```

---

## Implementation Plan

### Phase A: Data Pipeline (2-3 hours)

1. **Download TinyGSM**
   ```bash
   uv run python scripts/download_tinygsm.py --output data/tinygsm/
   ```

2. **Sample 100K examples**
   - Random sample from 12.3M
   - Stratify by difficulty if possible

3. **Convert to natural language CoT**
   ```bash
   uv run python scripts/convert_tinygsm_to_cot.py \
       --input data/tinygsm/sample_100k.jsonl \
       --output data/tinygsm/cot_100k.jsonl \
       --model gpt-4o-mini \
       --batch-size 100
   ```

4. **Create mul-token variant**
   - Inject mul-tokens into single-digit multiplications
   - Save as `cot_100k_mul_tokens.jsonl`

### Phase B: Training (4-6 hours)

1. **Fine-tune from Tulu-SFT models**
   ```bash
   uv run python sft/sft_tinygsm.py \
       --model-path outputs/sft_tulu_baseline \
       --data-path data/tinygsm/cot_100k.jsonl \
       --output-dir outputs/tinygsm_baseline \
       --epochs 3
   ```

2. **Train both conditions**
   - Baseline: natural language CoT
   - Mul_tokens: CoT with mul-token injection

### Phase C: Evaluation (1 hour)

1. Evaluate on GSM8K test set (full 1.3K)
2. Compare to pre-TinyGSM models
3. Analyze mul-token usage patterns

---

## Expected Outcomes

Based on TinyGSM paper:

| Scenario | Expected GSM8K Accuracy |
|----------|-------------------------|
| Current (25K data) | 2-4% |
| +100K TinyGSM | 15-25% (conservative) |
| +500K TinyGSM | 30-40% |
| +1M TinyGSM | 40-50% |
| Full 12.3M | 60%+ (matches paper) |

**Note:** Our conversion to natural language may lose some precision vs Python, but should still dramatically improve over current state.

---

## Mul-Token Integration Strategy

For the mul_tokens condition, we inject tokens during conversion:

| Expression | Baseline | Mul_tokens |
|------------|----------|------------|
| 7 × 8 = 56 | 7 × 8 = 56 | <MUL_7_8_56> |
| 9 × 6 = 54 | 9 × 6 = 54 | <MUL_6_9_54> |
| 12 × 5 = 60 | 12 × 5 = 60 | 12 × 5 = 60 (no token) |

This tests whether mul-tokens provide benefit when trained on sufficient data.

---

## Cost Estimate

| Item | Cost |
|------|------|
| GPT-4o-mini for 100K conversions | ~$15-20 |
| Compute (training 2 conditions) | ~$10 (GPU hours) |
| **Total** | **~$25-30** |

Very reasonable for potentially 10-20x accuracy improvement.

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/download_tinygsm.py` | Download and sample TinyGSM |
| `scripts/convert_tinygsm_to_cot.py` | Python→NL conversion with LLM |
| `data/tinygsm/cot_100k.jsonl` | Converted training data (baseline) |
| `data/tinygsm/cot_100k_mul_tokens.jsonl` | With mul-token injection |
| `sft/sft_tinygsm.py` | Training script for TinyGSM data |

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| GSM8K Accuracy (Baseline) | 2.4% | 15% | 25% |
| GSM8K Accuracy (Mul_tokens) | 4.0% | 18% | 30% |
| Mul-tokens benefit | +1.6% | +3% | +5% |

If we achieve 15%+ accuracy, we have validated that:
1. Data quantity is the primary bottleneck
2. 124M models CAN do math reasoning
3. Mul-tokens may provide meaningful benefit at scale

---

## Next Steps After TinyGSM

If successful (>15% accuracy):
1. **Scale to 500K-1M examples**
2. **Add verifier model** (TinyGSM's key innovation)
3. **GRPO on stronger base** (now RL has signal to optimize)

If unsuccessful (<10% accuracy):
1. Model architecture may be limiting factor
2. Consider switching to Phi-1.5 base model
3. Investigate Python-based solutions instead of NL

---

## Implementation Checklist

- [x] Download TinyGSM dataset ✅
- [x] Sample 100K examples ✅  
- [x] Set up OpenAI API for conversion ✅
- [x] Smoke test conversion pipeline (20 examples) ✅
- [ ] Convert Python → Natural Language CoT (100K)
- [ ] Inject mul-tokens for mul_tokens condition
- [ ] Train baseline model on TinyGSM CoT
- [ ] Train mul_tokens model on TinyGSM CoT
- [ ] Evaluate both on full GSM8K test set
- [ ] Compare results and analyze

---

## Smoke Test Results (December 28, 2025)

**Status:** ✅ Pipeline Working

### Async Conversion Pipeline
- **Model:** GPT-5-nano (low reasoning, structured output)
- **Success Rate:** 100% (20/20 examples)
- **Speed:** ~1.5 sec/example (async parallel)

### Mul-Token Quality

| Metric | Value |
|--------|-------|
| Examples with ≥1 mul-token | 45% |
| Avg mul-tokens (when present) | 1.8 |
| Max mul-tokens | 3 |

**Distribution:**
- 55% (0 tokens): Problems with fractions/divisions
- 30% (1 token): Single multiplication
- 15% (2-3 tokens): Multi-step with breakdown

### Example Quality

**Good Example (bakery - 3 mul-tokens):**
```
Q: A bakery produces x=4 loaves per day for a year...
Steps:
- Total yearly = 365 × 6
- Break 365 × 6: 3 × 6 = 18, 6 × 6 = 36, 5 × 6 = 30
- Total = 1800 + 360 + 30 = 2190

With mul-tokens:
<MUL_3_6_18>
<MUL_6_6_36>
<MUL_5_6_30>
```

### Key Findings

1. **When problems HAVE mul-token opportunities, the model breaks them down correctly**
2. Problems without multiplication (fractions, divisions) correctly get 0 tokens
3. Output is clean JSON, no garbling
4. Trivial cases (1×, 0×) are excluded

### Files Created

| File | Purpose |
|------|---------|
| `data/tinygsm/prompt.py` | Optimized prompt templates |
| `data/tinygsm/converter.py` | Conversion + mul-token injection |
| `data/tinygsm/smoke_test.py` | Async smoke test runner |
| `data/tinygsm/smoke_test_results/` | Saved results for inspection |

---

## 10K Shard Experiment Results (December 29, 2025)

**Status:** ✅ Complete — Critical Bug Found & Fixed

### Critical Bug: EOS label was getting masked during TinyGSM SFT

Root cause (TinyGSM path): we were using `DataCollatorForLanguageModeling` while **`pad_token_id == eos_token_id`**. That collator overwrites labels and masks every `pad_token_id` occurrence — which unintentionally masks the *real* EOS in the middle of sequences (the EOS we want the model to learn).

**Fix:** preserve explicit labels (mask padding via `attention_mask`, not `pad_token_id`) and use `default_data_collator` (not LM collator) for TinyGSM SFT.

**Primary fix file:** `sft/sft_tinygsm.py`

### 10K Shard Results (with EOS fix)

| Pipeline | Baseline | Mul_tokens |
|----------|----------|------------|
| Tulu → Curriculum → TinyGSM (10K) | 3.0% | 1.5% |
| **Tulu → TinyGSM (10K) direct** | **4.5%** | **3.5%** |

**Key Findings:**
1. **Curriculum hurts!** Direct path (Tulu → TinyGSM) outperforms curriculum path
2. **EOS learning improved** - Avg tokens dropped from 256 (max) to ~150
3. **Still low accuracy** - 10K examples insufficient (paper used 12.3M)

### Curriculum Analysis (Intuition Check)

| Checkpoint | Behavior |
|------------|----------|
| Pretrained | Echoes question, no math |
| Tulu SFT | Stops correctly, wrong answers |
| Curriculum: Arithmetic | ✅ `7×8=56` correct, but starts `####` repetition |
| Curriculum: Simple Word | ❌ Uses wrong operations (5+3 → 5×3) |
| Curriculum: GSM8K Mixed | ❌ Broken CoT, wrong decomposition |
| TinyGSM 10K | ❌ Catastrophic forgetting (7×8=28) |

**Conclusion:** Curriculum stages introduce bad patterns. Skip directly to TinyGSM.

---

## Scaling to 100K (December 29, 2025)

**Status:** ✅ Complete — 10 shards → combined 100K; trained + evaluated both conditions

### What we built / fixed

- **10-shard data layout**: `data/tinygsm/converted/shard_{1..10}_{baseline|mul_tokens}.jsonl`
- **Combined training files**:
  - `data/tinygsm/converted/combined_100k_baseline.jsonl`
  - `data/tinygsm/converted/combined_100k_mul_tokens.jsonl`
- **Transition stage (bridge)**: `sft/sft_transition.py` + `sft/configs/transition.yaml`
  - Added to reduce style shock from instruction SFT → TinyGSM format
  - Also fixed mul-token injector call (`injector.inject(...)` vs calling the object)
- **TinyGSM SFT fix**: `sft/sft_tinygsm.py` uses `default_data_collator` so EOS labels are not masked.

### 100K SFT + Eval Results (GSM8K 500-sample quick eval)

**Baseline** (`outputs/tinygsm_100k_baseline_v3`):
- GSM8K: **1.8% (9/500)**, avg response tokens **96.9**
- Probes: **6.41% (18/281)** overall; mul_table **22.22% (18/81)**

**Mul_tokens** (`outputs/tinygsm_100k_mul_tokens_v3`):
- GSM8K: **2.4% (12/500)**, avg response tokens **103.4**
- Mul-tokens used (GSM8K): **83 total** (**0.17 / response**)
- Probes: **6.05% (17/281)** overall; mul_table **20.99% (17/81)**
- Mul-tokens used (probes): **182 total** (**0.65 / probe**)

### Key takeaways

1. **Repetition / non-stopping behavior is fixed** for this pipeline (avg tokens ~100 instead of always hitting `max_new_tokens`).
2. At 100K scale, **mul_tokens shows a small GSM8K lift** (+0.6% absolute on 500-sample quick eval) and **actually uses mul-tokens** in generation.
3. Absolute accuracy is still low at 100K — consistent with TinyGSM paper’s claim that scale (millions) matters.

---

*Plan created: December 28, 2025*  
*Smoke test completed: December 28, 2025*  
*EOS/label bug fixed: December 29, 2025*  
*100K baseline + mul_tokens completed: December 29, 2025*
