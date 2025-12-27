#!/bin/bash
# Run full pilot experiment: baseline vs mul_tokens
#
# This script runs the complete pipeline:
# 1. Prepare FineWeb-Edu data (streaming, ~200M tokens)
# 2. Pretrain GPT-2 124M on each condition
# 3. Export to HuggingFace format
# 4. SFT on GSM8K
# 5. GRPO RL fine-tuning
# 6. Evaluate on GSM8K + arithmetic probes
# 7. Generate comparison report
#
# Usage:
#   bash scripts/run_pilot.sh [--skip-data] [--skip-pretrain] [--small]
#
# Options:
#   --skip-data     Skip data preparation (use existing shards)
#   --skip-pretrain Skip pretraining (use existing checkpoints)
#   --small         Use smaller token budget for faster testing

set -e
set -o pipefail

cd "$(dirname "$0")/.."

# Parse arguments
SKIP_DATA=false
SKIP_PRETRAIN=false
SMALL_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            ;;
        --skip-pretrain)
            SKIP_PRETRAIN=true
            ;;
        --small)
            SMALL_MODE=true
            ;;
    esac
done

# Configuration
SEED=42
if [ "$SMALL_MODE" = true ]; then
    TARGET_TOKENS=50000000   # 50M tokens for quick test
    SHARD_SIZE=5000000       # 5M per shard
else
    TARGET_TOKENS=200000000  # 200M tokens for pilot
    SHARD_SIZE=10000000      # 10M per shard
fi

echo "=============================================="
echo "Domain-Token GPT-2 Pilot Experiment"
echo "=============================================="
echo "Target tokens: $TARGET_TOKENS"
echo "Small mode: $SMALL_MODE"
echo "Seed: $SEED"
echo ""

# Step 1: Prepare data
if [ "$SKIP_DATA" = false ]; then
    echo "=============================================="
    echo "Step 1: Preparing FineWeb-Edu data"
    echo "=============================================="
    
    echo ""
    echo "--- Baseline condition ---"
    uv run python data/prepare_fineweb_pilot.py \
        --condition baseline \
        --out-dir data/fineweb_pilot/baseline \
        --target-tokens $TARGET_TOKENS \
        --shard-size $SHARD_SIZE \
        --seed $SEED
    
    echo ""
    echo "--- Mul-tokens condition ---"
    uv run python data/prepare_fineweb_pilot.py \
        --condition mul_tokens \
        --out-dir data/fineweb_pilot/mul_tokens \
        --target-tokens $TARGET_TOKENS \
        --shard-size $SHARD_SIZE \
        --seed $SEED
fi

# Step 2: Pretrain
if [ "$SKIP_PRETRAIN" = false ]; then
    echo ""
    echo "=============================================="
    echo "Step 2: Pretraining GPT-2 124M"
    echo "=============================================="
    
    echo ""
    echo "--- Baseline condition ---"
    uv run python pretrain/train_nanogpt.py \
        --config pretrain/configs/gpt2_124m_pilot.yaml \
        --data-root data/fineweb_pilot/baseline \
        --output-dir outputs/pretrain_baseline_pilot \
        --condition baseline \
        --seed $SEED
    
    echo ""
    echo "--- Mul-tokens condition ---"
    uv run python pretrain/train_nanogpt.py \
        --config pretrain/configs/gpt2_124m_pilot.yaml \
        --data-root data/fineweb_pilot/mul_tokens \
        --output-dir outputs/pretrain_mul_tokens_pilot \
        --condition mul_tokens \
        --seed $SEED
fi

# Step 3: Export to HuggingFace
echo ""
echo "=============================================="
echo "Step 3: Exporting to HuggingFace format"
echo "=============================================="

# Find latest checkpoint
BASELINE_CKPT=$(ls -t outputs/pretrain_baseline_pilot/model_*.pt 2>/dev/null | head -1)
MUL_CKPT=$(ls -t outputs/pretrain_mul_tokens_pilot/model_*.pt 2>/dev/null | head -1)

if [ -z "$BASELINE_CKPT" ]; then
    echo "ERROR: No baseline checkpoint found"
    exit 1
fi
if [ -z "$MUL_CKPT" ]; then
    echo "ERROR: No mul_tokens checkpoint found"
    exit 1
fi

echo "Baseline checkpoint: $BASELINE_CKPT"
echo "Mul-tokens checkpoint: $MUL_CKPT"

echo ""
echo "--- Exporting baseline ---"
uv run python pretrain/export_hf.py \
    --checkpoint "$BASELINE_CKPT" \
    --output-dir outputs/hf_baseline_pilot \
    --condition baseline

uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir outputs/hf_baseline_pilot \
    --condition baseline

echo ""
echo "--- Exporting mul-tokens ---"
uv run python pretrain/export_hf.py \
    --checkpoint "$MUL_CKPT" \
    --output-dir outputs/hf_mul_tokens_pilot \
    --condition mul_tokens

uv run python tokenizer/hf_gpt2_with_mul.py \
    --output-dir outputs/hf_mul_tokens_pilot \
    --condition mul_tokens

# Step 4: SFT
echo ""
echo "=============================================="
echo "Step 4: Supervised Fine-Tuning on GSM8K"
echo "=============================================="

echo ""
echo "--- Baseline SFT ---"
uv run python sft/sft_train.py \
    --model-path outputs/hf_baseline_pilot \
    --output-dir outputs/sft_baseline_pilot \
    --condition baseline \
    --config sft/configs/sft_pilot.yaml \
    --seed $SEED

echo ""
echo "--- Mul-tokens SFT ---"
uv run python sft/sft_train.py \
    --model-path outputs/hf_mul_tokens_pilot \
    --output-dir outputs/sft_mul_tokens_pilot \
    --condition mul_tokens \
    --config sft/configs/sft_pilot.yaml \
    --seed $SEED

# Step 5: GRPO (optional - can be slow)
echo ""
echo "=============================================="
echo "Step 5: GRPO RL Fine-Tuning"
echo "=============================================="

echo ""
echo "--- Baseline GRPO ---"
uv run python rl/grpo_train.py \
    --model-path outputs/sft_baseline_pilot \
    --output-dir outputs/grpo_baseline_pilot \
    --condition baseline \
    --config rl/configs/grpo_pilot.yaml \
    --seed $SEED

echo ""
echo "--- Mul-tokens GRPO ---"
uv run python rl/grpo_train.py \
    --model-path outputs/sft_mul_tokens_pilot \
    --output-dir outputs/grpo_mul_tokens_pilot \
    --condition mul_tokens \
    --config rl/configs/grpo_pilot.yaml \
    --seed $SEED

# Step 6: Evaluation
echo ""
echo "=============================================="
echo "Step 6: Evaluation"
echo "=============================================="

# Evaluate final GRPO models
BASELINE_MODEL=outputs/grpo_baseline_pilot
MUL_MODEL=outputs/grpo_mul_tokens_pilot

# If GRPO failed, fall back to SFT models
if [ ! -d "$BASELINE_MODEL" ]; then
    BASELINE_MODEL=outputs/sft_baseline_pilot
fi
if [ ! -d "$MUL_MODEL" ]; then
    MUL_MODEL=outputs/sft_mul_tokens_pilot
fi

echo ""
echo "--- GSM8K evaluation (baseline) ---"
uv run python eval/run_gsm8k.py \
    --model-path $BASELINE_MODEL \
    --output-dir outputs/eval_baseline_pilot \
    --condition baseline \
    --max-samples 200

echo ""
echo "--- GSM8K evaluation (mul-tokens) ---"
uv run python eval/run_gsm8k.py \
    --model-path $MUL_MODEL \
    --output-dir outputs/eval_mul_tokens_pilot \
    --condition mul_tokens \
    --max-samples 200

echo ""
echo "--- Arithmetic probes (baseline) ---"
uv run python eval/run_arithmetic_probes.py \
    --model-path $BASELINE_MODEL \
    --output-dir outputs/eval_baseline_pilot \
    --condition baseline \
    --seed $SEED

echo ""
echo "--- Arithmetic probes (mul-tokens) ---"
uv run python eval/run_arithmetic_probes.py \
    --model-path $MUL_MODEL \
    --output-dir outputs/eval_mul_tokens_pilot \
    --condition mul_tokens \
    --seed $SEED

# Step 7: Comparison report
echo ""
echo "=============================================="
echo "Step 7: Generating Comparison Report"
echo "=============================================="

uv run python scripts/compare_runs.py \
    --baseline-dir outputs/eval_baseline_pilot \
    --mul-tokens-dir outputs/eval_mul_tokens_pilot \
    --output-path outputs/comparison_report.json

echo ""
echo "=============================================="
echo "Pilot Experiment Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - outputs/comparison_report.json"
echo "  - outputs/eval_baseline_pilot/"
echo "  - outputs/eval_mul_tokens_pilot/"

