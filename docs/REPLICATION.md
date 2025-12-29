# Replication Guide

This guide provides step-by-step commands to reproduce the domain-token experiments.

---

## Prerequisites

### Local Setup

```bash
# Clone repository
git clone https://github.com/zhengpu-berkeley/domain-token-gpt2.git
cd domain-token-gpt2

# Install dependencies (requires Python 3.11+)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Run tests to verify setup
uv run pytest tests/ -v
```

### GPU Cluster Setup (RunPod)

```bash
# SSH into cluster
ssh runpod-domain-token  # Configure in ~/.ssh/config

# On cluster:
cd /workspace/domain-token-gpt2
git pull
uv sync

# Verify GPUs
nvidia-smi
uv run python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

---

## Download Pretrained Models

All models are available on HuggingFace Hub:

```bash
# List available models
uv run python scripts/load_from_hf.py --list

# Download all models
uv run python scripts/load_from_hf.py --all

# Download specific models
uv run python scripts/load_from_hf.py --models hf_baseline_10b hf_mul_tokens_10b
```

### Available Models

| Model | HuggingFace Repo |
|-------|------------------|
| Pretrained (baseline) | `zhengpu-berkeley/domain-token-gpt2-baseline-10b` |
| Pretrained (mul_tokens) | `zhengpu-berkeley/domain-token-gpt2-mul-tokens-10b` |
| Tulu SFT (baseline) | `zhengpu-berkeley/domain-token-gpt2-sft-tulu-baseline` |
| Tulu SFT (mul_tokens) | `zhengpu-berkeley/domain-token-gpt2-sft-tulu-mul-tokens` |
| GSM8K SFT (baseline) | `zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-baseline` |
| GSM8K SFT (mul_tokens) | `zhengpu-berkeley/domain-token-gpt2-sft-gsm8k-mul-tokens` |

---

## Full Pipeline: From Scratch

### Step 1: Prepare FineWeb-Edu Data (10B tokens)

```bash
# Baseline condition
uv run python data/prepare_fineweb_pilot.py \
    --condition baseline \
    --out-dir data/fineweb_10b/baseline \
    --target-tokens 10000000000 \
    --shard-size 100000000

# Mul_tokens condition (with injection)
uv run python data/prepare_fineweb_pilot.py \
    --condition mul_tokens \
    --out-dir data/fineweb_10b/mul_tokens \
    --target-tokens 10000000000 \
    --shard-size 100000000
```

**Time:** ~2 hours per condition (streaming from HuggingFace)

### Step 2: Pretrain GPT-2 124M

```bash
# Single GPU
uv run python pretrain/train_nanogpt.py \
    --config pretrain/configs/gpt2_124m_10b.yaml \
    --data-root data/fineweb_10b/baseline \
    --output-dir outputs/pretrain_baseline_10b \
    --condition baseline

# Multi-GPU (4x H200)
torchrun --standalone --nproc_per_node=4 pretrain/train_nanogpt.py \
    --config pretrain/configs/gpt2_124m_10b.yaml \
    --data-root data/fineweb_10b/baseline \
    --output-dir outputs/pretrain_baseline_10b \
    --condition baseline
```

**Time:** ~3 hours on 4× H200 for 10B tokens

### Step 3: Export to HuggingFace Format

```bash
uv run python pretrain/export_hf.py \
    --checkpoint outputs/pretrain_baseline_10b/model_19072.pt \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline

uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir outputs/hf_baseline_10b \
    --condition baseline
```

### Step 4: Tulu SFT (Instruction-Tuning)

```bash
uv run python sft/sft_tulu.py \
    --model-path outputs/hf_baseline_10b \
    --output-dir outputs/sft_tulu_baseline \
    --condition baseline \
    --config sft/configs/tulu.yaml
```

**Time:** ~70 minutes on RTX 4090

### Step 5: TinyGSM SFT (Math Specialization)

```bash
# Download TinyGSM dataset
uv run python scripts/download_tinygsm.py

# Convert to natural language CoT (requires OpenAI API key)
export OPENAI_API_KEY="your-key"
uv run python data/tinygsm/batch_convert.py \
    --num-shards 10 \
    --output-dir data/tinygsm/converted

# Train on converted data
uv run python sft/sft_tinygsm.py \
    --model-path outputs/sft_tulu_baseline \
    --data-path data/tinygsm/converted/combined_100k_baseline.jsonl \
    --output-dir outputs/tinygsm_100k_baseline \
    --condition baseline \
    --config sft/configs/tinygsm.yaml
```

### Step 6: Evaluate

```bash
# GSM8K evaluation
uv run python eval/run_gsm8k.py \
    --model-path outputs/tinygsm_100k_baseline \
    --output-dir outputs/eval_tinygsm_baseline \
    --condition baseline \
    --max-samples 500  # Use 1319 for full test set

# Arithmetic probes
uv run python eval/run_arithmetic_probes.py \
    --model-path outputs/tinygsm_100k_baseline \
    --output-dir outputs/eval_tinygsm_baseline \
    --condition baseline

# HellaSwag (general language ability)
uv run python eval/run_hellaswag.py \
    --model-path outputs/tinygsm_100k_baseline \
    --output-dir outputs/eval_tinygsm_baseline \
    --max-samples 500
```

---

## Quick Smoke Test (Local, No GPU)

```bash
# Tiny model, 100 steps, synthetic data
uv run python data/prepare_text.py --condition baseline
uv run python data/tokenize_to_bin.py --condition baseline

uv run python pretrain/train_nanogpt.py \
    --config pretrain/configs/gpt2_124m_pilot.yaml \
    --data-root data/processed/baseline \
    --output-dir outputs/smoke_baseline \
    --condition baseline \
    --max-steps 100
```

---

## Mul_tokens Condition

Replace `baseline` with `mul_tokens` in all commands. The key differences:

1. **Data prep:** Applies `MulExpressionInjector` to inject `<MUL_a_b_c>` tokens
2. **Tokenizer:** Uses the same vocab but recognizes mul-tokens as single tokens
3. **SFT:** Use `--inject-mul-tokens` flag for GSM8K/TinyGSM SFT

```bash
# Example: Mul_tokens Tulu SFT
uv run python sft/sft_tulu.py \
    --model-path outputs/hf_mul_tokens_10b \
    --output-dir outputs/sft_tulu_mul_tokens \
    --condition mul_tokens \
    --config sft/configs/tulu.yaml
```

---

## Expected Results

| Stage | Baseline GSM8K | Mul_tokens GSM8K |
|-------|----------------|------------------|
| Pretrained (10B) | ~0% | ~0% |
| After Tulu SFT | ~1.5% | ~1.5% |
| After TinyGSM 100K | ~1.8% | ~2.4% |
| Target (TinyGSM 12M) | ~60% | ~60%+ |

---

## Troubleshooting

### Out of Memory
Reduce `batch_size` in config or use gradient accumulation.

### Tokenizer Mismatch
Regenerate tokenizer files:
```bash
uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir <model-path> \
    --condition <baseline|mul_tokens>
```

### Model produces repetitive output
This indicates insufficient training or mode collapse. Try:
- More SFT epochs
- Lower learning rate
- More training data

---

## Upload Models to HuggingFace

```bash
uv run python scripts/upload_to_hf.py \
    --model-path outputs/sft_tulu_baseline \
    --repo-name your-username/model-name
```

