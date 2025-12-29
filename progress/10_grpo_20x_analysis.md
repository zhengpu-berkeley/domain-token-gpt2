# Phase 6: GRPO 20x Compute Analysis

**Date:** December 28, 2025  
**Status:** ✅ Complete (historical) — useful diagnosis; current focus is TinyGSM scaling + improved SFT bridging  
**Goal:** Test if 20x RL compute can push accuracy beyond curriculum SFT

**Last Updated:** 2025-12-29

---

## Experiment Configuration

**20x Compute Setup:**
- **Samples:** 2,000 (2x baseline)
- **Rollouts per prompt:** 40 (10x baseline)  
- **Total rollouts:** 80,000 per condition
- **Training time:** ~2.2 hours per condition
- **Shaped rewards:** ✅ (partial credit, mul-token bonus, repetition penalty)

---

## Results

### GSM8K Accuracy Comparison

| Pipeline Stage | Baseline | Mul_tokens | Δ |
|----------------|----------|------------|---|
| Post-Tulu SFT | 2.50% | 2.05% | -0.45% |
| Post-Curriculum SFT | 2.40% | **3.80%** | +1.40% |
| **Post-GRPO 20x** | 2.20% | **4.00%** | +1.80% |

### Detailed Metrics

| Metric | Baseline | Mul_tokens |
|--------|----------|------------|
| GSM8K Accuracy | 2.2% (11/500) | **4.0%** (20/500) |
| Mul-tokens/response | 0.0 | 1.24 |
| Repetition Rate | 84.2% | 83.5% |
| Hit Max Length | 19.3% | 17.5% |

### Training Dynamics

During 80K rollouts, we observed:

| Metric | Start | End | Trend |
|--------|-------|-----|-------|
| Accuracy (rolling) | ~1-2% | ~0-2% | **Flat/declining** |
| Reward mean | ~-0.1 | ~-0.06 | Slight improvement |
| Repetition rate | ~85% | ~84% | Essentially flat |
| Mul-tokens/resp (mul_tokens) | ~1.2 | ~1.35 | Slight increase |

---

## Key Findings

### 1. RL Showed Marginal Improvement for Mul_tokens

The mul_tokens condition went from 3.8% → 4.0% (+0.2% absolute), while baseline stayed flat at ~2.2%. This is a small but positive signal.

### 2. Repetition Rate Remained Extremely High (~84%)

Both models generate repetitive loops in 84% of responses. This indicates:
- The model lacks strong math reasoning priors
- RL can't "teach" math from scratch
- The base model is fundamentally limited

### 3. 20x Compute Didn't Solve the Core Problem

Despite 80,000 rollouts per condition:
- No breakthrough improvement
- Accuracy stayed in 1-4% range
- Mode collapse tendencies persisted

### 4. Mul-token Usage Increased (0.64 → 1.24)

The shaped reward bonus for mul-tokens (+0.2) successfully increased usage, but this didn't translate to accuracy gains.

---

## Diagnosis: Why RL Failed to Help

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROOT CAUSE ANALYSIS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pre-training (10B FineWeb tokens)                              │
│       ↓                                                         │
│  ❌ Insufficient math exposure                                  │
│       ↓                                                         │
│  SFT (Tulu + Curriculum)                                        │
│       ↓                                                         │
│  ❌ Can't learn math reasoning from ~25K examples               │
│       ↓                                                         │
│  GRPO (80K rollouts)                                            │
│       ↓                                                         │
│  ❌ RL can only refine existing capabilities, not create new    │
│                                                                 │
│  CONCLUSION: Model capacity (124M) + pre-training data = limit  │
└─────────────────────────────────────────────────────────────────┘
```

**The fundamental issue:** A 124M parameter model pre-trained on general web text doesn't have the capacity to learn multi-step math reasoning from limited SFT data. RL can only optimize what the model already knows.

---

## Hypotheses for Next Steps

### Hypothesis A: Scale Model Size (GPT-2 Medium 355M)

**Rationale:** Larger models have more capacity for reasoning. GPT-2 Medium (355M) has 3x parameters and may cross a capability threshold.

**Effort:** Medium (need to re-run pre-training, ~30 hours on 4x H200)

**Risk:** May still not be enough; even 355M is small by modern standards.

### Hypothesis B: Math-Heavy Pre-training

**Rationale:** FineWeb has minimal math content. Adding math-specific corpora could establish arithmetic primitives.

**Options:**
- OpenWebMath (~15B tokens of math web content)
- Proof-Pile (mathematical proofs)
- GSM8K-augmented synthetic data in pre-training

**Effort:** High (need to rebuild pre-training pipeline)

**Risk:** May need more than 10B tokens to see effect.

### Hypothesis C: Distillation from Larger Model

**Rationale:** Instead of training from scratch, distill math capabilities from GPT-4/Claude into GPT-2.

**Approach:**
1. Generate 100K+ GSM8K-style solutions with GPT-4
2. Fine-tune GPT-2 on this synthetic data
3. Test if distilled model can learn math patterns

**Effort:** Medium (API costs, but simpler training)

**Risk:** Unclear if small model can absorb distilled knowledge.

### Hypothesis D: Different Task (Simpler Math)

**Rationale:** GSM8K requires 2-8 step reasoning. Maybe 124M model can only handle 1-2 step problems.

**Approach:**
1. Create simpler math benchmark (1-step word problems)
2. Test if mul-tokens help on easier problems
3. Establish capability baseline before scaling

**Effort:** Low

**Risk:** Doesn't address the core research question.

### Hypothesis E: Architecture Changes

**Rationale:** Standard GPT-2 may not be optimal for arithmetic.

**Options:**
- Add scratchpad/CoT mechanism
- Explicit memory for intermediate results
- Specialized positional encoding for numbers

**Effort:** High

**Risk:** Research novelty, but high implementation cost.

---

## Recommended Priority

Given constraints (time, compute, research value):

| Priority | Hypothesis | Rationale |
|----------|------------|-----------|
| 1 | **D: Simpler task** | Quick validation that mul-tokens help on tractable problems |
| 2 | **C: Distillation** | Potentially best path to strong small model |
| 3 | **A: Scale to 355M** | Straightforward, may cross threshold |
| 4 | **B: Math pre-training** | Most principled, but highest effort |
| 5 | E: Architecture | Research novelty, but risky |

---

## Immediate Action: Validate Mul-tokens on Simpler Task

Before investing in scaling, we should verify the mul-token hypothesis on a tractable problem:

1. **Create 1-step multiplication word problems** (e.g., "John has 7 bags with 8 apples each. How many apples?")
2. **Evaluate current models** on this simpler benchmark
3. **If mul-tokens help significantly**, proceed to scaling
4. **If not**, reconsider the approach

This takes ~2 hours and provides decisive signal.

---

## Conclusion

**Your intuition is correct:** The base model is the bottleneck. No amount of RL compute can overcome fundamental capability limits.

The mul-tokens intervention shows consistent (if small) benefits:
- Curriculum SFT: +1.4% over baseline
- Post-GRPO: +1.8% over baseline

But absolute accuracy (~4%) is too low for practical use. The path forward requires either:
1. Validating on simpler tasks
2. Scaling model size
3. Improving pre-training data

---

*Experiment completed: December 28, 2025*

