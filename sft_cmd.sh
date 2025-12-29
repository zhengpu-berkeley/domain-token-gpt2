#!/bin/bash
# Convenience runner for the TinyGSM 100K pipeline (transition → TinyGSM → eval).
#
# Notes:
# - This is a *helper* script; outputs are large and are gitignored.
# - Requires OPENAI_API_KEY in environment/.env only if you're regenerating TinyGSM shards.

set -euo pipefail

cd /workspace/domain-token-gpt2

echo "=== Step 1/6: Transition SFT (baseline) ==="
TOKENIZERS_PARALLELISM=false uv run python sft/sft_transition.py \
  --model-path outputs/sft_tulu_baseline \
  --output-dir outputs/transition_baseline \
  --condition baseline \
  --config sft/configs/transition.yaml

echo "=== Step 2/6: Transition SFT (mul_tokens) ==="
TOKENIZERS_PARALLELISM=false uv run python sft/sft_transition.py \
  --model-path outputs/sft_tulu_mul_tokens \
  --output-dir outputs/transition_mul_tokens \
  --condition mul_tokens \
  --config sft/configs/transition.yaml

echo "=== Step 3/6: TinyGSM 100K SFT (baseline) ==="
TOKENIZERS_PARALLELISM=false uv run python sft/sft_tinygsm.py \
  --model-path outputs/transition_baseline \
  --output-dir outputs/tinygsm_100k_baseline_v3 \
  --condition baseline \
  --config sft/configs/tinygsm.yaml

echo "=== Step 4/6: TinyGSM 100K SFT (mul_tokens) ==="
TOKENIZERS_PARALLELISM=false uv run python sft/sft_tinygsm.py \
  --model-path outputs/transition_mul_tokens \
  --output-dir outputs/tinygsm_100k_mul_tokens_v3 \
  --condition mul_tokens \
  --config sft/configs/tinygsm.yaml

echo "=== Step 5/6: GSM8K eval (baseline, 500) ==="
uv run python eval/run_gsm8k.py \
  --model-path outputs/tinygsm_100k_baseline_v3 \
  --output-dir outputs/eval_tinygsm_100k_baseline_v3 \
  --condition baseline \
  --max-samples 500

echo "=== Step 6/6: GSM8K eval (mul_tokens, 500) ==="
uv run python eval/run_gsm8k.py \
  --model-path outputs/tinygsm_100k_mul_tokens_v3 \
  --output-dir outputs/eval_tinygsm_100k_mul_tokens_v3 \
  --condition mul_tokens \
  --max-samples 500

echo "=== DONE ==="

