# Domain-Token GPT-2 Experiment

Testing whether domain-specific tokenization (multiplication-fact tokens) yields measurable performance gains on math reasoning benchmarks under compute-matched training.

## Quick Start

```bash
# Install uv
curl -sSf https://astral.sh/uv/install.sh | sh
```

```bash
# Install dependencies with uv
uv sync

# Run tokenizer tests
uv run pytest tests/ -v

# Prepare tiny data (baseline condition)
uv run python data/prepare_text.py --condition baseline

# Prepare tiny data (mul-token condition)
uv run python data/prepare_text.py --condition mul_tokens

# Tokenize to .bin
uv run python data/tokenize_to_bin.py --condition baseline
uv run python data/tokenize_to_bin.py --condition mul_tokens

# Smoke pretrain (tiny model, CPU/MPS)
uv run python pretrain/train.py --config pretrain/configs/tiny.yaml --condition baseline
uv run python pretrain/train.py --config pretrain/configs/tiny.yaml --condition mul_tokens
```

## Project Structure

```
domain-token-gpt2/
├── third_party/
│   └── build-nanogpt/       # Vendored Karpathy nanoGPT (MIT license)
├── tokenizer/
│   ├── mul_facts.py         # Mul-token generation + ID mapping
│   ├── gpt2_tiktoken.py     # GPT-2 BPE wrapper with special tokens
│   └── inject_mul.py        # Regex-based injector
├── data/
│   ├── raw/                 # Downloaded/generated raw text
│   ├── processed/           # Tokenized .bin files
│   ├── prepare_text.py      # Tiny corpus builder
│   └── tokenize_to_bin.py   # Text → tokens → .bin
├── pretrain/
│   ├── train.py             # Training wrapper
│   └── configs/
│       └── tiny.yaml        # Tiny smoke-test config
├── tests/
│   ├── test_tokenizer.py
│   └── test_inject_mul.py
└── scripts/
    ├── run_10b.sh                 # Full 10B-token end-to-end run (data→pretrain→SFT→eval)
    ├── eval_checkpoints.py        # Evaluate pretrain checkpoints on HellaSwag
    ├── plot_checkpoint_curve.py   # Plot per-condition HellaSwag learning curve
    └── visualize_pretrain_losses.py  # Plot pretrain loss curves
```

## Experimental Conditions

- **Condition A (baseline)**: Standard GPT-2 BPE tokenizer, injection disabled
- **Condition B (mul_tokens)**: GPT-2 BPE + 81 multiplication-fact tokens, injection enabled

Both conditions share the same vocab_size (reserved token IDs) to ensure identical architecture.

## References

- [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) - Training base
- [karpathy/llm.c](https://github.com/karpathy/llm.c) - Future optimization target

## License

MIT (see vendored code licenses in `third_party/`)

