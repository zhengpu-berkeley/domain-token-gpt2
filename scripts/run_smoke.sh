#!/bin/bash
# Run smoke test for both baseline and mul_tokens conditions
# This prepares data, tokenizes, and runs a tiny pretrain for each condition

set -e  # Exit on error

cd "$(dirname "$0")/.."

echo "============================================"
echo "Domain-Token GPT-2 Smoke Test"
echo "============================================"

# Parse arguments
CONDITION="${1:-both}"

run_condition() {
    local cond=$1
    echo ""
    echo "============================================"
    echo "Running condition: $cond"
    echo "============================================"
    
    echo ""
    echo "Step 1: Preparing text data..."
    uv run python data/prepare_text.py --condition "$cond"
    
    echo ""
    echo "Step 2: Tokenizing to binary..."
    uv run python data/tokenize_to_bin.py --condition "$cond"
    
    echo ""
    echo "Step 3: Running pretrain..."
    uv run python pretrain/train.py --condition "$cond" --config pretrain/configs/tiny.yaml
}

if [ "$CONDITION" = "both" ]; then
    run_condition "baseline"
    run_condition "mul_tokens"
    
    echo ""
    echo "============================================"
    echo "Comparing results..."
    echo "============================================"
    uv run python scripts/compare_results.py
else
    run_condition "$CONDITION"
fi

echo ""
echo "============================================"
echo "Smoke test complete!"
echo "============================================"

