# Research Objective

Test whether **domain-specific tokenization** that injects “multiplication-fact tokens” (inspired by 九九乘法表, e.g., collapsing `6*9=54` into a single token) yields **measurable, reproducible performance gains** on math reasoning benchmarks (primary: **GSM8K**) under a **strict compute-matched** training and post-training regime.

Concretely: run two end-to-end training pipelines that are identical in every way **except** for tokenizer/vocabulary (baseline vs mul-fact augmented). Compare benchmark performance with identical pretraining compute and identical post-training compute (SFT + RL), and evaluate whether the tokenization change alone produces statistically meaningful improvements.

---

# Hypothesis

**H1 (main):** With the same training compute, a model trained with a tokenizer augmented with multiplication-fact tokens will achieve higher accuracy on GSM8K (and related arithmetic benchmarks) than a model with a standard tokenizer.

**Mechanism intuition:** The augmented tokenizer can represent certain arithmetic substructures more “atomically,” potentially reducing sequence length and learning burden for basic arithmetic facts; RL may also discover how to exploit these tokens to increase reward.

**H0 (null):** No meaningful performance difference after controlling for compute, data, and pipeline.

---

# Scope and Constraints

## What this study is (and is not)
- ✅ **Is:** a controlled A/B experiment on tokenizer intervention (vocabulary + merges + preprocessing) with compute-matched training.
- ✅ **Is:** end-to-end: **pretrain → SFT → RL (GRPO) → benchmark eval**.
- ❌ **Not:** “prove novelty” in the philosophical sense. We will document related work and position our contribution as “expression-as-token” (mul-fact tokens) as a specific instantiation of domain-specific tokenization.
- ❌ **Not:** maximizing absolute GSM8K SOTA. We care about **differential lift** under constraints.

## Target budget / scale
- Model size target: **~124M parameters** (GPT-2 small class) unless constrained by hardware.
- Compute budget: optimized for **$200–$1,000** on rented GPUs; design supports smoke-tests at smaller scales.

---

# Experimental Design Overview

We will run two conditions:

## Condition A — Baseline
- Tokenizer: standard subword tokenizer (e.g., GPT-2 BPE / tiktoken-compatible, or a SentencePiece/BPE tokenizer trained on the same pretraining text without special injection).
- All other settings fixed.

## Condition B — Mul-Fact Tokenizer
- Tokenizer: same as baseline **plus** injected tokens representing multiplication facts.
- Pretraining and post-training data are **preprocessed** to insert these tokens (so they’re actually seen and their embeddings are trained).
- Everything else fixed (architecture, optimizer, schedules, random seeds, compute budgets, SFT/RL procedures, eval).

---

# “Same Compute” Definition (Critical)

Because tokenizers change token counts, “same dataset” ≠ “same compute.” Our goal is compute-matched experiments.

## Primary compute-matching rule
We define “same compute” as:
- Same model architecture
- Same global batch size (tokens per step)
- Same sequence length (tokens)
- Same number of optimizer steps
- Same precision settings
- Same optimizer and schedule

This implies **approximately same FLOPs** for pretraining and post-training.

**Resulting implication:** the mul-fact tokenizer may consume **more or less raw text** per token budget (since tokenization compresses/expands). That is acceptable: we are testing *tokenization as an inductive bias + compression mechanism* under a compute budget.

## Secondary checks
We will also report:
- Raw bytes / characters processed
- Number of examples processed (where meaningful)
- Distribution of sequence lengths and truncation rates

---

# Resources to Leverage (Cursor agent can browse + pull code)

We will build on battle-tested public stacks, but keep the pipeline simple:

## Pretraining base
Choose one of these two approaches (prefer #1 for simplicity and integration with post-training):

1) **PyTorch-based minimal GPT training** (Karpathy-style)
   - `karpathy/nanoGPT` as the lightweight GPT training reference.
   - Pros: simple, readable, easy to control; can export to HF easily; stable.
   - Cons: you’ll need to wire tokenizer and data pipeline cleanly.

2) **llm.c / llama2.c-style pretraining**
   - Extremely efficient, but post-training (SFT/RL) integration is more friction unless exporting weights into HF format.

**Recommendation:** Start with **nanoGPT-style pretraining**, then run SFT/RL using HF tooling.

## SFT + RL (GRPO)
- Use Hugging Face ecosystem for post-training:
  - Transformers model wrapper for your GPT-2-like model
  - PEFT/LoRA for cost-efficient SFT and RL (optional; can do full fine-tune if budget allows)
  - TRL (or equivalent) for GRPO

## Evaluation harness
- Standard evaluation via `lm-evaluation-harness` or a clean benchmark runner with consistent decoding settings (temperature=0, max_new_tokens, etc.).
- Primary: GSM8K. Secondary: simple arithmetic probes + related datasets.

---

# Benchmarks

## Primary
- **GSM8K** (test set), accuracy on final answer.
  - Decide evaluation format:
    - “Answer-only” (e.g., `#### 54`)
    - or “CoT + final answer” (but careful: reasoning tokens may dominate effect size; we want tokenization signal).

**Recommendation:** Evaluate both:
1) **Direct answer** format (minimize confounds)
2) **CoT** format (realistic but noisier)

## Secondary (to isolate mechanisms)
Add at least one “pure arithmetic” suite, because GSM8K is not only multiplication:
- Synthetic multiplication table probes (1–9, 1–20, multi-digit)
- Simple word arithmetic problems (SVAMP / MultiArith / ASDiv style tasks)
- Optional: MATH subset (harder; may be too noisy at 124M)

---

# Tokenizer Intervention Spec

## Baseline tokenizer
- Start from a standard tokenizer:
  - GPT-2 BPE compatible (common baseline)
  - or a SentencePiece Unigram/BPE tokenizer trained on the same corpus

**Constraints:**
- Same vocab size (or nearly same) across conditions, unless we explicitly test vocab-size sensitivity.
- Same special tokens and formatting.

## Mul-Fact tokens (domain-specific augmentation)

### Token set
We will add a controlled set of tokens that represent multiplication facts.

**Option 1 (literal string tokens):**
- Each token is literally the ASCII sequence `a*b=c` for small integers.
- Example tokens: `6*9=54`, `7*8=56`, etc.

**Option 2 (structured special tokens):**
- Tokens like `<MUL_6_9_54>` to avoid collisions and simplify preprocessing.

**Recommendation:** Use **structured tokens** to reduce accidental merges and simplify canonicalization.

### Coverage
Start with a minimal set:
- 81 tokens for 1–9 multiplication table (九九乘法表 equivalent)
Optionally expand:
- include commutative duplicates or canonicalize `a<=b`
- extend to 1–20 if compute allows

### Data injection requirement
These tokens must appear during training or they won’t be meaningfully learned.

We will implement a deterministic preprocessor that:
1) Identifies multiplication expressions in text (`a*b`, `a × b`, `a times b`)
2) Optionally identifies known results (`=c`) if present
3) Rewrites canonical patterns into the special token form

**Two injection modes:**
- **Mode B1 (strict):** only replace when full `a*b=c` occurs.
- **Mode B2 (weak supervision):** replace `a*b` occurrences into `<MUL_a_b>` and let the model learn `=c` patterns; or add synthetic statements containing `a*b=c`.

**Recommendation:** Use **B2** + a synthetic math corpus to ensure enough signal.

---

# Datasets

## Pretraining data (must include math signal)
We need enough coverage for multiplication patterns to matter.

### Base corpus
Pick one:
- A general web-text subset (OpenWebText-like) small enough for budget
- A curated “math-heavy” corpus
- A mixture: general text + synthetic math + math explanations

### Synthetic augmentation (strongly recommended)
Generate a synthetic dataset that includes:
- Multiplication tables in multiple textual forms:
  - `6*9=54`
  - `6 × 9 = 54`
  - `six times nine is fifty-four`
  - Chinese forms (optional): `六九五十四` / `6乘9等于54` etc.
- Short arithmetic drills and micro word problems that require multiplication
- Ensure the preprocessor can rewrite these into mul-fact tokens in Condition B

**Goal:** Make the augmented tokens frequent enough to learn embeddings and usage.

## SFT data
Use a consistent math instruction dataset, e.g.:
- GSM8K training split (careful about leakage: use train only)
- Additional math instruction data if desired, but keep identical across conditions

Format: either CoT or answer-only; include a strict separator for “final answer.”

## RL data (GRPO)
- Prompts: GSM8K train prompts (or a held-out subset of training prompts)
- Reward: exact match on final numeric answer (parsed robustly)
- Optional shaped rewards:
  - partial credit for correct intermediate steps (risky; might overshadow tokenizer effect)
  - length penalty or format penalty (to stabilize)

**Recommendation:** Start with **binary exact-match** reward on final answer.

---

# Training Pipeline Spec

## Architecture
- GPT-2-small class transformer (≈124M):
  - n_layer / n_head / n_embd aligned with GPT-2 small if feasible
  - tied input/output embeddings
- Same architecture in both conditions.

## Pretraining
- Objective: causal LM
- Fixed:
  - global batch size (tokens/step)
  - seq length
  - optimizer (AdamW)
  - LR schedule (cosine or linear warmup + decay)
  - weight decay, grad clipping
  - precision (bf16/fp16)
  - number of steps

**Deliverable:** checkpoint `pretrain_final` for each condition.

## Post-training Stage 1: SFT
- Load `pretrain_final`
- Fine-tune on identical SFT dataset
- Same:
  - steps (or epochs converted into steps)
  - max seq length
  - batch size
  - LR schedule
- Consider LoRA adapters for cost control:
  - if using LoRA, keep rank and target modules fixed across both conditions

**Deliverable:** checkpoint `sft_final`.

## Post-training Stage 2: RL with GRPO
- Initialize policy from `sft_final`
- Reward on GSM8K train prompts
- Same:
  - number of RL updates
  - number of rollouts per update
  - decoding settings (temperature, top_p, max_new_tokens)
  - KL regularization strategy (if used)
  - reference model handling (if used)
- Use identical seeds and evaluation intervals.

**Deliverable:** checkpoint `grpo_final`.

---

# Evaluation Protocol

## Decoding settings (must be identical)
- temperature = 0 (greedy) for benchmark scoring
- max_new_tokens fixed (e.g., 256 or 512)
- stop tokens / stop sequences standardized
- For CoT mode: enforce a consistent “final answer” marker.

## Metrics
Primary:
- GSM8K accuracy (exact match on final answer)

Secondary:
- arithmetic probe accuracy (multiplication tables, multi-digit multiplication, etc.)
- average tokens generated per solution
- error taxonomy:
  - arithmetic mistake vs parsing vs reasoning mistake
- token usage stats:
  - how often mul-fact tokens appear in generations (Condition B)
  - whether RL increases their usage over time

## Statistical rigor
- Run at least **N=3 seeds** per condition if budget allows.
- Report mean ± std and paired comparisons (same seed mapping).
- If budget constrained: N=1 initially for feasibility, then expand.

---

# Implementation and Repo Layout

## Directory structure (suggested)

project/
README.md
spec.md                       # this document
configs/
base.yaml
exp_baseline.yaml
exp_mul_tokens.yaml
tokenizer/
train_tokenizer.py
inject_mul_tokens.py
preprocess_math.py
data/
raw/                         # downloaded datasets
processed/
baseline/
mul_tokens/
pretrain/
train.py                     # nanoGPT-style or HF trainer
sft/
sft_gsm8k.py
rl/
grpo_train.py
rewards.py
eval/
run_gsm8k.py
run_arithmetic_probes.py
scripts/
setup_env.sh
download_data.sh
run_all.sh
outputs/
exp_baseline/
exp_mul_tokens/

## Reproducibility requirements
- One `base.yaml` that defines:
  - model size, seq length, precision
  - step budgets for pretrain/SFT/RL
  - evaluation suite + decoding settings
- Two experiment config overlays:
  - `exp_baseline.yaml`: baseline tokenizer + baseline preprocessing
  - `exp_mul_tokens.yaml`: tokenizer augmentation + preprocessing rewrite enabled

Everything else identical.

---

# Engineering / Ops Notes for Rampart + SSH

## Environment setup
- Use a single reproducible environment:
  - conda/venv or uv
  - pinned versions for torch, transformers, trl, datasets, tokenizers/sentencepiece
- Ensure deterministic flags where possible:
  - fixed seeds
  - deterministic ops (note: full determinism may reduce speed)

## Smoke tests (mandatory before full run)
1) Tokenizer unit tests:
   - encode/decode roundtrip
   - ensure mul-fact tokens are recognized
2) Tiny model pretrain:
   - 10–30M params
   - 1k–5k steps
   - verify loss decreases
3) Tiny SFT:
   - 500–2k examples
   - verify format + evaluation scripts
4) Tiny GRPO:
   - 50–200 prompts
   - verify reward improves and training is stable

Only then run the full 124M pipeline.

---

# Risks and Mitigations

## Risk: No signal because mul-fact tokens are too rare
Mitigation:
- synthetic augmentation
- explicit injection preprocessor
- monitoring token frequency

## Risk: Confound from sequence length differences
Mitigation:
- primary compute matching is tokens/step × steps (FLOPs matched)
- report raw text processed as a secondary diagnostic

## Risk: RL instability obscures effect
Mitigation:
- start with SFT-only comparisons
- add RL after SFT pipeline is stable
- keep reward simple (exact-match), conservative KL

## Risk: Overfitting / leakage on GSM8K
Mitigation:
- strict train/test split
- no test contamination in RL prompts
- document data provenance

---

# Success Criteria

Minimum viable success:
- Condition B shows **consistent improvement** over Condition A on GSM8K across seeds (or at least a strong effect in a pilot), under compute-matched training.

Stronger success:
- Gains also appear on arithmetic probes (mechanism support)
- RL measurably increases usage of mul-fact tokens in Condition B
- Token count or generation length decreases without hurting accuracy (efficiency win)

---

# Run Commands (Automation)

## Current runner (recommended)
```bash
The repo currently uses `scripts/run_10b.sh` as the primary “end-to-end” runner for the 10B-token experiment (data prep → pretrain → export → SFT → eval). The earlier idea of a single `run_all.sh` with config overlays is aspirational and not the current source-of-truth.

Example:

```bash
cd /workspace/domain-token-gpt2
bash scripts/run_10b.sh
```

Where `scripts/run_10b.sh` performs, in order:
	1.	build/download datasets
	2.	preprocess datasets according to config (baseline vs mul token injection)
	3.	pretrain for fixed steps
	4.	SFT for fixed steps
	5.	GRPO RL for fixed steps/rollouts
	6.	run evaluation suite and write a single results.json + results.md

⸻

What Cursor Agent Should Do (Task Checklist)
	1.	Repo bootstrap
	•	Create directory structure
	•	Write config system and run scripts
	2.	Tokenizer pipeline
	•	Implement baseline tokenizer training/loading
	•	Implement mul-token augmentation
	•	Implement canonicalization + injection preprocessor
	3.	Training pipeline
	•	Pretrain script (nanoGPT-style) that saves HF-compatible checkpoint (or provide conversion)
	•	SFT script (Transformers Trainer or TRL SFTTrainer)
	•	GRPO script (TRL GRPOTrainer or chosen framework)
	4.	Evaluation pipeline
	•	GSM8K evaluator with robust answer extraction
	•	Arithmetic probe suite generator + evaluator
	5.	Logging and artifacts
	•	Save configs, seeds, git commit hash
	•	Save metrics per stage and final aggregated report

⸻

Deliverables
	•	outputs/exp_baseline/ and outputs/exp_mul_tokens/ each containing:
	•	final checkpoints: pretrain_final, sft_final, grpo_final
	•	eval outputs: gsm8k_results.json, arithmetic_results.json
	•	training logs + plots
	•	compare.py that reads both outputs and prints:
	•	GSM8K delta
	•	probe deltas
	•	token usage stats
	•	summary table for the report
