# Architecture and Design Decisions

This document describes the codebase structure and key design choices.

---

## Repository Structure

```
domain-token-gpt2/
├── docs/                    # Documentation (this folder)
│   ├── RESEARCH.md          # Research findings and results
│   ├── REPLICATION.md       # Step-by-step reproduction guide
│   └── ARCHITECTURE.md      # This file
│
├── tokenizer/               # Core tokenization module
│   ├── mul_facts.py         # MulFactTokens: 45 tokens for 1-9 × 1-9
│   ├── gpt2_tiktoken.py     # GPT2TokenizerWithMulFacts wrapper
│   ├── hf_gpt2_with_mul.py  # Build HuggingFace tokenizer
│   └── inject_mul.py        # MulExpressionInjector for preprocessing
│
├── data/                    # Data preparation
│   ├── prepare_fineweb_pilot.py  # FineWeb-Edu streaming + injection
│   ├── tokenize_to_bin.py        # Text → .npy tokenized files
│   └── tinygsm/                  # TinyGSM conversion pipeline
│       ├── converter.py          # Python → Natural language CoT
│       ├── batch_convert.py      # Parallel conversion with OpenAI
│       └── prompt.py             # LLM prompts for conversion
│
├── pretrain/                # Pretraining scripts
│   ├── train_nanogpt.py     # DDP-capable GPT-2 training
│   ├── export_hf.py         # nanoGPT → HuggingFace export
│   └── configs/             # YAML configs for different scales
│
├── sft/                     # Supervised fine-tuning
│   ├── sft_tulu.py          # Tulu-3 instruction tuning
│   ├── sft_tinygsm.py       # TinyGSM math training
│   ├── sft_gsm8k.py         # GSM8K fine-tuning
│   └── configs/             # SFT hyperparameters
│
├── rl/                      # Reinforcement learning (experimental)
│   ├── grpo_train.py        # GRPO with TRL
│   ├── rewards.py           # Answer extraction and reward computation
│   └── configs/             # GRPO configs
│
├── eval/                    # Evaluation scripts
│   ├── run_gsm8k.py         # GSM8K test set evaluation
│   ├── run_arithmetic_probes.py  # Synthetic arithmetic benchmarks
│   └── run_hellaswag.py     # HellaSwag language modeling eval
│
├── scripts/                 # Utilities
│   ├── upload_to_hf.py      # Push models to HuggingFace Hub
│   ├── load_from_hf.py      # Download models from Hub
│   ├── download_tinygsm.py  # Get TinyGSM dataset
│   └── eval_checkpoints.py  # Evaluate pretrain checkpoints
│
├── tests/                   # Unit tests
│   ├── test_tokenizer.py    # Tokenizer tests (22 tests)
│   ├── test_inject_mul.py   # Injector tests (21 tests)
│   ├── test_hf_export.py    # HF export tests
│   ├── test_sft_dataset.py  # Dataset formatting tests
│   ├── test_eval_answer.py  # Answer extraction tests
│   └── conftest.py          # Shared fixtures
│
├── third_party/             # Vendored dependencies
│   └── build-nanogpt/       # Karpathy's nanoGPT (MIT license)
│
└── sample/                  # Inference utilities
    ├── sample_sft.py        # Interactive model sampling
    └── text_completion.py   # Basic text completion
```

---

## Key Design Decisions

### 1. Same Vocab Size for Both Conditions

**Decision:** Both baseline and mul_tokens conditions use `vocab_size=50349`.

**Rationale:** This ensures identical model architectures for fair A/B comparison. In baseline, token IDs 50304-50348 are reserved but never appear in training data.

```python
# tokenizer/mul_facts.py
BASE_VOCAB_SIZE = 50304  # Karpathy's padded GPT-2
NUM_MUL_TOKENS = 45      # 1-9 × 1-9 canonicalized
TOTAL_VOCAB_SIZE = 50349  # Used by BOTH conditions
```

### 2. Canonicalized Multiplication Tokens

**Decision:** Use `<MUL_a_b_c>` format where `a ≤ b` (canonicalized for commutativity).

**Rationale:** `6*9` and `9*6` produce the same token `<MUL_6_9_54>`, reducing vocabulary size from 81 to 45 tokens.

```python
# tokenizer/mul_facts.py
def canonicalize(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)
```

### 3. Tiktoken for Training, HuggingFace for Inference

**Decision:** Use tiktoken during pretraining (fast), convert to HuggingFace format for SFT/eval.

**Rationale:** tiktoken is 5-10x faster for batch tokenization. HuggingFace is required for Transformers/TRL integration.

**Critical:** The export must transpose weights (see RESEARCH.md for the bug).

### 4. User:/Assistant: Prompt Format

**Decision:** Use simple `User:` / `Assistant:` format throughout.

```
User: What is 7 times 8?
Assistant: 7 × 8 = 56. The answer is 56.
#### 56<|endoftext|>
```

**Rationale:** Consistent format across Tulu SFT, GSM8K, TinyGSM, and evaluation prevents format mismatch issues.

### 5. Labels Computed from Attention Mask

**Decision:** Mask padding in labels using `attention_mask`, not `pad_token_id`.

```python
# Correct approach in sft_tinygsm.py
labels = [-100 if mask == 0 else token 
          for token, mask in zip(input_ids, attention_mask)]
```

**Rationale:** GPT-2 uses `pad_token_id == eos_token_id == 50256`. Token-based masking would kill EOS learning.

---

## Module Dependencies

```
tokenizer/
    ↓
data/ ──────────────────┐
    ↓                   │
pretrain/               │
    ↓                   │
    └──→ export_hf.py ──┘
              ↓
         sft/ ──→ eval/
              ↓
             rl/
```

---

## Configuration System

All scripts use YAML configs with CLI overrides:

```yaml
# sft/configs/tinygsm.yaml
training:
  epochs: 2
  learning_rate: 2.0e-5
  batch_size: 8
  max_length: 512
  
data:
  question_key: "question"
  answer_key: "solution"
```

```bash
# CLI override example
uv run python sft/sft_tinygsm.py \
    --config sft/configs/tinygsm.yaml \
    --learning-rate 1e-5  # Override
```

---

## Testing Strategy

### Test Categories

1. **Unit tests** (fast, no GPU): `tests/test_*.py`
2. **Integration tests** (require model files): Mark with `@pytest.mark.integration`
3. **Smoke tests** (full pipeline, minimal data): See REPLICATION.md

### Running Tests

```bash
# All tests (should complete in <30 seconds)
uv run pytest tests/ -v

# Specific test file
uv run pytest tests/test_tokenizer.py -v

# With coverage
uv run pytest tests/ --cov=tokenizer --cov=rl
```

---

## HuggingFace Hub Integration

Models are synced via HuggingFace Hub for cross-machine reproducibility:

```python
# scripts/upload_to_hf.py
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path=model_path, repo_id=repo_name, ...)

# scripts/load_from_hf.py
from huggingface_hub import snapshot_download
snapshot_download(repo_id=repo_name, local_dir=local_path, ...)
```

---

## Key Invariants

1. **Vocab size:** Always 50349 for both conditions
2. **Prompt format:** Always `User:` / `Assistant:`
3. **Answer format:** Always `#### {number}` for GSM8K-style problems
4. **EOS token:** Always appended to training examples
5. **Mul-token range:** IDs 50304-50348 (45 tokens)

---

## Extending the Codebase

### Adding a New Dataset

1. Create loader in `sft/datasets/` (or `data/`)
2. Add config in `sft/configs/`
3. Use existing SFT script with `--data-path` override

### Adding New Eval Benchmark

1. Create `eval/run_<benchmark>.py`
2. Use `batch_generate()` pattern from `run_gsm8k.py`
3. Add result parsing specific to benchmark format

### Modifying Tokenizer

1. Update `tokenizer/mul_facts.py` for new token types
2. Update `tokenizer/inject_mul.py` for new patterns
3. Regenerate HF tokenizers with `hf_gpt2_with_mul.py`
4. **Critical:** Update tests in `tests/test_tokenizer.py`

