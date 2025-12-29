# Status & Handoff Document

**Project:** Domain-Token GPT-2 Experiment  
**Status:** Active — pretraining complete (10B tokens, both conditions); post-training + evaluation in progress; TinyGSM distillation scaling underway  
**Last Updated:** 2025-12-29  

---

## Executive Summary

We have an end-to-end, compute-matched A/B pipeline (baseline vs `mul_tokens`) for a GPT‑2‑class model with a fixed vocab (`50349`), plus multiple post-training routes:

- **10B-token pretraining** on FineWeb‑Edu (both conditions) is complete, with checkpoint evaluation working after fixing the nanoGPT → HF export path.
- **Instruction SFT** (Tulu) and **math SFT** (GSM8K) have been rerun post-export-fix (see `progress/08_sft_rerun_two_mixtures.md`).
- **TinyGSM distillation** tooling exists and is being scaled (conversion + 10‑shard / 100K workflow), including a transition stage to bridge instruction style → TinyGSM style (see `progress/11_tinygsm_distillation.md`).

### What’s Done (high-confidence / current)
- ✅ **Tokenizer + injection**: `<MUL_a_b_c>` tokens + `MulExpressionInjector` for `mul_tokens` condition; **same vocab size across conditions**.
- ✅ **10B pretraining (both conditions)**: raw checkpoints + checkpoint HellaSwag evals are source-of-truth.
- ✅ **HF export**: exporter fixed; downstream HF-based evaluations are now meaningful when derived from post-fix exports.
- ✅ **SFT + eval harness**: Tulu SFT, GSM8K SFT, GSM8K eval, arithmetic probes.
- ✅ **TinyGSM data pipeline**: sharding + conversion pipeline + SFT script; transition-stage script for bridging instruction → TinyGSM.

### What’s Next (practical)
- ⏳ **TinyGSM scaling**: push beyond 100K examples (toward 500K–1M) to match TinyGSM paper regime.
- ⏳ **Post-training rigor**: paired seeds (≥3) for the 10B pipeline, plus better stopping/repetition controls during eval.
- ⏳ **RL (GRPO)**: only after the base model has sufficient math competence (avoid sparse-reward failure mode); prefer shaped rewards + strong KL.

---

## Repository Structure (high level; see repo for details)

```
domain-token-gpt2/
│
├── progress/                     # Experiment notes (this series of docs)
├── README.md                     # Quick start guide
├── pyproject.toml                # Dependencies (use `uv sync`)
│
├── tokenizer/                    # Tokenization module
│   ├── __init__.py
│   ├── mul_facts.py              # MulFactTokens class (45 tokens, IDs 50304-50348)
│   ├── gpt2_tiktoken.py          # GPT2TokenizerWithMulFacts wrapper
│   └── inject_mul.py             # MulExpressionInjector (text preprocessing)
│
├── data/                         # Data preparation
│   ├── __init__.py
│   ├── prepare_text.py           # Generate synthetic corpus
│   ├── tokenize_to_bin.py        # Text → uint16 .npy files
│   ├── raw/                      # Generated text files
│   │   ├── train_baseline.txt
│   │   ├── train_mul_tokens.txt
│   │   └── *_stats.json
│   └── processed/                # Tokenized binary files
│       ├── baseline/
│       │   ├── train_00000.npy
│       │   ├── val_00000.npy
│       │   └── meta.json
│       └── mul_tokens/
│           ├── train_00000.npy
│           ├── val_00000.npy
│           └── meta.json
│
├── pretrain/                     # Pretraining scripts
│   ├── __init__.py
│   ├── train.py                  # Main training loop (nanoGPT-inspired)
│   └── configs/
│       └── tiny.yaml             # Tiny model config (7M params, smoke test)
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_tokenizer.py         # 22 tokenizer tests
│   └── test_inject_mul.py        # 21 injector tests
│
├── scripts/                      # Utilities (export, upload, eval helpers, etc.)
│   ├── run_10b.sh                # Full 10B-token run automation
│   ├── eval_checkpoints.py       # Checkpoint HellaSwag eval runner
│   └── plot_checkpoint_curve.py  # Plot checkpoint learning curves
│
├── third_party/                  # Vendored dependencies
│   └── build-nanogpt/            # Karpathy's nanoGPT (MIT license)
│       ├── train_gpt2.py         # Reference implementation
│       ├── fineweb.py            # FineWeb data prep (use for full runs)
│       ├── hellaswag.py          # HellaSwag eval
│       ├── README_UPSTREAM.md
│       └── UPSTREAM.md           # Provenance: commit 6104ab1b
│
└── outputs/                      # Training outputs / artifacts
    ├── pretrain_baseline/
    │   ├── model_*.pt
    │   ├── log.txt
    │   └── results.json
    └── pretrain_mul_tokens/
        ├── model_*.pt
        ├── log.txt
        └── results.json
```

---

## Key Technical Details

### Tokenizer Design

| Property | Value |
|----------|-------|
| Base tokenizer | GPT-2 BPE via `tiktoken` |
| Base vocab size | 50,304 (Karpathy's padded GPT-2) |
| Mul-fact tokens | 45 (canonicalized: `a ≤ b` for commutativity) |
| Total vocab size | **50,349** (same for both conditions) |
| Mul-token ID range | 50,304 – 50,348 |
| Token format | `<MUL_a_b_c>` (e.g., `<MUL_6_9_54>`) |

**Critical design choice:** Both conditions use `vocab_size=50349`. In baseline, the 45 reserved token IDs are never seen in training data; in mul_tokens, they are used. This ensures **identical model architecture** for fair comparison.

### Injection Patterns Recognized

The injector (`tokenizer/inject_mul.py`) detects and rewrites:

| Pattern | Example | Result |
|---------|---------|--------|
| Asterisk | `6*9` | `<MUL_6_9_54>` |
| Unicode × | `6 × 9` | `<MUL_6_9_54>` |
| Lowercase x | `6x9` | `<MUL_6_9_54>` |
| With result | `6*9=54` | `<MUL_6_9_54>` |
| Commutative | `9*6` | `<MUL_6_9_54>` (canonicalized) |

**Modes:**
- `weak` (default): Injects bare expressions like `6*9`
- `strict`: Only injects when `=c` is present and correct

### Smoke Test Results (Local)

| Metric | Baseline | Mul-Tokens | Delta |
|--------|----------|------------|-------|
| Total tokens | 26,881 | 23,986 | -10.8% |
| Mul-fact tokens | 0 | 1,102 (4.6%) | — |
| Final train loss | 5.31 | 5.45 | +0.14 |
| Final val loss | 5.38 | 5.54 | +0.15 |

**Note:** This is a tiny 7M parameter model with 100 steps on synthetic data. Results are not meaningful for the hypothesis — they only verify the pipeline works.

---

## Cluster Operationalization Checklist

### Prerequisites

```bash
# Clone and setup
git clone <repo-url>
cd domain-token-gpt2
uv sync  # or: pip install -e .

# Verify tests pass
uv run pytest tests/ -v
```

### Step 1: Prepare Real Data

The current synthetic data is for smoke testing. For real experiments:

**Option A: Use FineWeb (recommended for compute-matched experiments)**
```bash
# Use vendored Karpathy script to download FineWeb
cd third_party/build-nanogpt
uv run python fineweb.py  # Downloads ~10B tokens to edu_fineweb10B/

# Then preprocess with our injector for mul_tokens condition
cd ../..
# TODO: Create data/prepare_fineweb.py that applies injection
```

**Option B: Custom math-heavy corpus**
- Combine: OpenWebText subset + synthetic math + GSM8K-style drills
- Ensure enough multiplication patterns for mul-tokens to matter
- Use `data/prepare_text.py` as template

### Step 2: Create Production Config

Create `pretrain/configs/gpt2_124m.yaml`:

```yaml
# GPT-2 124M configuration
model:
  n_layer: 12
  n_head: 12
  n_embd: 768
  block_size: 1024

training:
  batch_size: 64          # micro batch size (adjust for GPU memory)
  total_batch_size: 524288  # ~0.5M tokens per step (matches Karpathy)
  
  max_steps: 19073        # ~1 epoch over 10B tokens
  warmup_steps: 715
  
  max_lr: 6.0e-4
  min_lr: 6.0e-5
  
  weight_decay: 0.1
  grad_clip: 1.0
  
  dtype: bfloat16         # Use bf16 on A100/H100

eval:
  interval: 250
  val_steps: 20

logging:
  interval: 10

checkpoint:
  interval: 5000
  save_final: true
```

### Step 3: Run Pretraining

```bash
# Baseline condition
uv run python pretrain/train.py \
  --config pretrain/configs/gpt2_124m.yaml \
  --condition baseline \
  --data-dir /path/to/processed/baseline \
  --output-dir outputs/pretrain_baseline_124m

# Mul-tokens condition
uv run python pretrain/train.py \
  --config pretrain/configs/gpt2_124m.yaml \
  --condition mul_tokens \
  --data-dir /path/to/processed/mul_tokens \
  --output-dir outputs/pretrain_mul_tokens_124m
```

**For DDP multi-GPU:**
```bash
torchrun --standalone --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/gpt2_124m.yaml \
  --condition baseline
```

### Step 4: Post-Training (SFT + GRPO)

✅ **Implemented (SFT).** Current two-stage SFT flow:

1. `sft/sft_tulu.py` — Instruction-tune on Tulu-3 mixture (subset sampling)
2. `sft/sft_gsm8k.py` — Fine-tune on GSM8K train (math specialization)
3. Configs:
   - `sft/configs/tulu.yaml`
   - `sft/configs/gsm8k.yaml`

⚠️ **GRPO**: `rl/grpo_train.py` exists but should be treated as a separate stage to run after SFT is validated on GSM8K/probes.

### Step 5: Evaluation

✅ **Implemented.**

1. `eval/run_gsm8k.py` — Accuracy on GSM8K test (exact match on `#### answer`)
2. `eval/run_arithmetic_probes.py` — Synthetic arithmetic benchmarks

---

## Files to Extend for Full Experiments

| File | Status | Notes |
|------|--------|-------|
| `pretrain/train.py` | ✅ Ready | Add DDP support, wandb logging |
| `pretrain/configs/gpt2_124m.yaml` | ⏳ TODO | Production config |
| `data/prepare_fineweb.py` | ⏳ TODO | Apply injection to FineWeb |
| `sft/sft_tulu.py` | ✅ Ready | HF Trainer for Tulu-3 mixture SFT (subset sampling) |
| `sft/sft_gsm8k.py` | ✅ Ready | HF Trainer for GSM8K SFT |
| `rl/grpo_train.py` | ⏳ TODO | TRL GRPOTrainer |
| `eval/run_gsm8k.py` | ✅ Ready | Task evaluation (uses `User:` / `Assistant:` prompts) |
| `eval/run_arithmetic_probes.py` | ✅ Ready | Synthetic probes (uses `User:` / `Assistant:` prompts) |

---

## Compute Budget Guidance

From `research_spec.md`:

| Stage | Tokens | Est. Time (8×A100) | Est. Cost |
|-------|--------|---------------------|-----------|
| Pretrain (124M) | 10B | ~1 hour | ~$30 |
| SFT | 1M | ~5 min | ~$2 |
| GRPO | 10M rollouts | ~30 min | ~$15 |
| **Total per condition** | — | ~2 hours | ~$50 |
| **Both conditions + seeds** | — | ~12 hours | ~$300 |

Budget for 3 seeds per condition: **~$300–$500**

---

## Critical Invariants to Maintain

1. **Same vocab_size:** Both conditions must use `vocab_size=50349`
2. **Same architecture:** `n_layer`, `n_head`, `n_embd`, `block_size` identical
3. **Same compute:** Same `total_batch_size × max_steps` = same total tokens processed
4. **Same optimizer:** AdamW with identical hyperparameters
5. **Same seeds:** Use paired seeds (0, 1, 2) for statistical comparison
6. **No test leakage:** GSM8K test set must NOT appear in pretraining or SFT data

---

## Contact & References

- **Research spec:** `research_spec.md` (full experimental design)
- **Karpathy nanoGPT:** [github.com/karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt)
- **Vendored commit:** `6104ab1b53920f6e2159749676073ff7d815c1fa`

---

## Quick Commands Reference

```bash
# Install deps
uv sync

# Run tests
uv run pytest tests/ -v

# Prepare data for a condition
uv run python data/prepare_text.py --condition mul_tokens
uv run python data/tokenize_to_bin.py --condition mul_tokens

# Train (tiny local)
uv run python pretrain/train.py --condition mul_tokens
```

---

**Phase 1 Complete. Ready for cluster deployment.**

