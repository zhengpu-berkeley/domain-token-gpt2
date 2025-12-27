#!/bin/bash
# =============================================================================
# 10B Token Full Experiment
# =============================================================================
#
# Runs the complete domain-token GPT-2 experiment with 10B tokens:
# - Baseline condition (standard tokenizer)
# - Mul_tokens condition (multiplication-fact tokens)
#
# Hardware requirements: 4x H200 GPUs (or equivalent)
#
# Estimated runtime: 12-16 hours total
#
# Usage:
#   bash scripts/run_10b.sh
#   bash scripts/run_10b.sh --skip-data        # Skip data prep (use existing)
#   bash scripts/run_10b.sh --skip-baseline    # Skip baseline (run mul_tokens only)
#   bash scripts/run_10b.sh --skip-mul-tokens  # Skip mul_tokens (run baseline only)
#
# =============================================================================

set -e
set -o pipefail

cd "$(dirname "$0")/.."

# Parse arguments
SKIP_DATA=false
SKIP_BASELINE=false
SKIP_MUL_TOKENS=false

for arg in "$@"; do
    case $arg in
        --skip-data)
            SKIP_DATA=true
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            ;;
        --skip-mul-tokens)
            SKIP_MUL_TOKENS=true
            ;;
        --help)
            echo "Usage: bash scripts/run_10b.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-data        Skip data preparation (use existing shards)"
            echo "  --skip-baseline    Skip baseline condition"
            echo "  --skip-mul-tokens  Skip mul_tokens condition"
            echo "  --help             Show this help message"
            exit 0
            ;;
    esac
done

# Configuration
SEED=42
TARGET_TOKENS=10000000000  # 10B tokens
SHARD_SIZE=100000000       # 100M per shard (100 shards total)
NUM_GPUS=4

echo "=============================================="
echo "Domain-Token GPT-2: 10B Token Full Experiment"
echo "=============================================="
echo "Target tokens: $(printf "%'d" $TARGET_TOKENS)"
echo "GPUs: $NUM_GPUS"
echo "Seed: $SEED"
echo ""

# Check disk space
echo "Checking disk space..."
AVAILABLE_GB=$(df -BG /workspace | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -lt 60 ]; then
    echo "WARNING: Less than 60GB available. Recommend at least 60GB for 10B experiment."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborting."
        exit 1
    fi
fi

# Verify GPU setup
echo ""
echo "Verifying GPU setup..."
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    echo "WARNING: Expected $NUM_GPUS GPUs but found $GPU_COUNT"
    echo "Adjusting NUM_GPUS to $GPU_COUNT"
    NUM_GPUS=$GPU_COUNT
fi

# =============================================================================
# BASELINE CONDITION
# =============================================================================

if [ "$SKIP_BASELINE" = false ]; then
    echo ""
    echo "=============================================="
    echo "BASELINE CONDITION"
    echo "=============================================="
    
    # Step 1: Data preparation
    if [ "$SKIP_DATA" = false ]; then
        echo ""
        echo "--- Step 1: Preparing FineWeb-Edu data (baseline) ---"
        echo "This will stream 10B tokens. Estimated time: ~2 hours"
        
        uv run python data/prepare_fineweb_pilot.py \
            --condition baseline \
            --out-dir data/fineweb_10b/baseline \
            --target-tokens $TARGET_TOKENS \
            --shard-size $SHARD_SIZE \
            --seed $SEED
    else
        echo ""
        echo "--- Skipping data preparation (--skip-data) ---"
    fi
    
    # Step 2: Pretraining
    echo ""
    echo "--- Step 2: Pretraining GPT-2 124M (baseline) ---"
    echo "Using $NUM_GPUS GPUs with DDP. Estimated time: ~4-6 hours"
    
    torchrun --standalone --nproc_per_node=$NUM_GPUS pretrain/train_nanogpt.py \
        --config pretrain/configs/gpt2_124m_10b.yaml \
        --data-root data/fineweb_10b/baseline \
        --output-dir outputs/pretrain_baseline_10b \
        --condition baseline \
        --seed $SEED
    
    # Step 3: Export to HuggingFace format
    echo ""
    echo "--- Step 3: Exporting to HuggingFace format (baseline) ---"
    
    BASELINE_CKPT=$(ls -t outputs/pretrain_baseline_10b/model_*.pt 2>/dev/null | head -1)
    if [ -z "$BASELINE_CKPT" ]; then
        echo "ERROR: No baseline checkpoint found"
        exit 1
    fi
    echo "Using checkpoint: $BASELINE_CKPT"
    
    uv run python pretrain/export_hf.py \
        --checkpoint "$BASELINE_CKPT" \
        --output-dir outputs/hf_baseline_10b \
        --condition baseline
    
    uv run python tokenizer/hf_gpt2_with_mul.py \
        --output-dir outputs/hf_baseline_10b \
        --condition baseline
    
    # Step 4: SFT on GSM8K
    echo ""
    echo "--- Step 4: Supervised Fine-Tuning on GSM8K (baseline) ---"
    
    uv run python sft/sft_train.py \
        --model-path outputs/hf_baseline_10b \
        --output-dir outputs/sft_baseline_10b \
        --condition baseline \
        --config sft/configs/sft_full.yaml \
        --seed $SEED
    
    # Step 5: Evaluation
    echo ""
    echo "--- Step 5: Evaluation (baseline) ---"
    
    uv run python eval/run_gsm8k.py \
        --model-path outputs/sft_baseline_10b \
        --output-dir outputs/eval_baseline_10b \
        --condition baseline \
        --max-samples 1319  # Full GSM8K test set
    
    uv run python eval/run_arithmetic_probes.py \
        --model-path outputs/sft_baseline_10b \
        --output-dir outputs/eval_baseline_10b \
        --condition baseline \
        --seed $SEED
    
    echo ""
    echo "=============================================="
    echo "BASELINE CONDITION COMPLETE"
    echo "=============================================="
fi

# =============================================================================
# MUL_TOKENS CONDITION
# =============================================================================

if [ "$SKIP_MUL_TOKENS" = false ]; then
    echo ""
    echo "=============================================="
    echo "MUL_TOKENS CONDITION"
    echo "=============================================="
    
    # Step 1: Data preparation
    if [ "$SKIP_DATA" = false ]; then
        echo ""
        echo "--- Step 1: Preparing FineWeb-Edu data (mul_tokens) ---"
        echo "This will stream 10B tokens with injection. Estimated time: ~2 hours"
        
        uv run python data/prepare_fineweb_pilot.py \
            --condition mul_tokens \
            --out-dir data/fineweb_10b/mul_tokens \
            --target-tokens $TARGET_TOKENS \
            --shard-size $SHARD_SIZE \
            --seed $SEED
    else
        echo ""
        echo "--- Skipping data preparation (--skip-data) ---"
    fi
    
    # Step 2: Pretraining
    echo ""
    echo "--- Step 2: Pretraining GPT-2 124M (mul_tokens) ---"
    echo "Using $NUM_GPUS GPUs with DDP. Estimated time: ~4-6 hours"
    
    torchrun --standalone --nproc_per_node=$NUM_GPUS pretrain/train_nanogpt.py \
        --config pretrain/configs/gpt2_124m_10b.yaml \
        --data-root data/fineweb_10b/mul_tokens \
        --output-dir outputs/pretrain_mul_tokens_10b \
        --condition mul_tokens \
        --seed $SEED
    
    # Step 3: Export to HuggingFace format
    echo ""
    echo "--- Step 3: Exporting to HuggingFace format (mul_tokens) ---"
    
    MUL_CKPT=$(ls -t outputs/pretrain_mul_tokens_10b/model_*.pt 2>/dev/null | head -1)
    if [ -z "$MUL_CKPT" ]; then
        echo "ERROR: No mul_tokens checkpoint found"
        exit 1
    fi
    echo "Using checkpoint: $MUL_CKPT"
    
    uv run python pretrain/export_hf.py \
        --checkpoint "$MUL_CKPT" \
        --output-dir outputs/hf_mul_tokens_10b \
        --condition mul_tokens
    
    uv run python tokenizer/hf_gpt2_with_mul.py \
        --output-dir outputs/hf_mul_tokens_10b \
        --condition mul_tokens
    
    # Step 4: SFT on GSM8K
    echo ""
    echo "--- Step 4: Supervised Fine-Tuning on GSM8K (mul_tokens) ---"
    
    uv run python sft/sft_train.py \
        --model-path outputs/hf_mul_tokens_10b \
        --output-dir outputs/sft_mul_tokens_10b \
        --condition mul_tokens \
        --config sft/configs/sft_full.yaml \
        --seed $SEED
    
    # Step 5: Evaluation
    echo ""
    echo "--- Step 5: Evaluation (mul_tokens) ---"
    
    uv run python eval/run_gsm8k.py \
        --model-path outputs/sft_mul_tokens_10b \
        --output-dir outputs/eval_mul_tokens_10b \
        --condition mul_tokens \
        --max-samples 1319  # Full GSM8K test set
    
    uv run python eval/run_arithmetic_probes.py \
        --model-path outputs/sft_mul_tokens_10b \
        --output-dir outputs/eval_mul_tokens_10b \
        --condition mul_tokens \
        --seed $SEED
    
    echo ""
    echo "=============================================="
    echo "MUL_TOKENS CONDITION COMPLETE"
    echo "=============================================="
fi

# =============================================================================
# COMPARISON REPORT
# =============================================================================

echo ""
echo "=============================================="
echo "Generating Comparison Report"
echo "=============================================="

uv run python scripts/compare_runs.py \
    --baseline-dir outputs/eval_baseline_10b \
    --mul-tokens-dir outputs/eval_mul_tokens_10b \
    --output-path outputs/comparison_report_10b.json

echo ""
echo "=============================================="
echo "10B TOKEN EXPERIMENT COMPLETE"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - outputs/comparison_report_10b.json"
echo "  - outputs/eval_baseline_10b/"
echo "  - outputs/eval_mul_tokens_10b/"
echo ""
echo "Checkpoints:"
echo "  - outputs/pretrain_baseline_10b/"
echo "  - outputs/pretrain_mul_tokens_10b/"
echo "  - outputs/sft_baseline_10b/"
echo "  - outputs/sft_mul_tokens_10b/"
echo ""

# Print summary
echo "--- Quick Summary ---"
if [ -f outputs/comparison_report_10b.json ]; then
    uv run python -c "
import json
with open('outputs/comparison_report_10b.json') as f:
    data = json.load(f)
print(json.dumps(data, indent=2))
"
fi

