# Domain-Token GPT-2 Experiment

Testing whether domain-specific tokenization (multiplication-fact tokens like `<MUL_6_9_54>`) yields measurable performance gains on math reasoning benchmarks under compute-matched training.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/RESEARCH.md](docs/RESEARCH.md) | Research findings, results, and lessons learned |
| [docs/REPLICATION.md](docs/REPLICATION.md) | Step-by-step commands to reproduce experiments |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Code structure and design decisions |

## Quick Start

```bash
# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run tests
uv run pytest tests/ -v

# Download pretrained models from HuggingFace Hub
uv run python scripts/load_from_hf.py --all
```

## Key Results

| Pipeline Stage | Baseline GSM8K | Mul_tokens GSM8K |
|----------------|----------------|------------------|
| After Tulu SFT | 1.5% | 1.5% |
| After Curriculum SFT | 2.4% | **3.8%** (+1.4%) |
| After TinyGSM 100K | 1.8% | **2.4%** (+0.6%) |

See [docs/RESEARCH.md](docs/RESEARCH.md) for full analysis.

## Project Structure

```
domain-token-gpt2/
├── docs/                    # Documentation
├── tokenizer/               # Core tokenization (mul-tokens, injection)
├── data/                    # Data preparation (FineWeb, TinyGSM)
├── pretrain/                # GPT-2 pretraining (nanoGPT-style)
├── sft/                     # Supervised fine-tuning (Tulu, TinyGSM, GSM8K)
├── rl/                      # GRPO reinforcement learning (experimental)
├── eval/                    # Evaluation (GSM8K, arithmetic probes, HellaSwag)
├── scripts/                 # Utilities (HuggingFace upload/download)
├── tests/                   # Unit tests (88 tests, runs in <3 seconds)
└── third_party/             # Vendored nanoGPT (MIT license)
```

## Experimental Conditions

- **Baseline**: Standard GPT-2 BPE tokenizer (vocab_size=50349)
- **Mul_tokens**: Same tokenizer + 45 multiplication-fact tokens (`<MUL_a_b_c>` for 1-9 x 1-9)

Both conditions use identical architecture (vocab_size reserved) for fair A/B comparison.

## Models on HuggingFace Hub

| Model | Repository |
|-------|------------|
| Pretrained (10B, baseline) | [zhengpu-berkeley/domain-token-gpt2-baseline-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-baseline-10b) |
| Pretrained (10B, mul_tokens) | [zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b) |
| Tulu SFT (baseline) | [zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline) |
| Tulu SFT (mul_tokens) | [zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens](https://huggingface.co/zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens) |

## License

MIT (see vendored code licenses in `third_party/`)
