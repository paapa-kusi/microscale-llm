#!/usr/bin/env bash
set -e

# Activate the environment
conda activate microscale-llm

# Define pruning ratios
PRUNE_RATIOS=(0.1 0.5 0.9)

# Run combined pruning + quantization experiments
for RATIO in "${PRUNE_RATIOS[@]}"; do
    echo "Running combined pruning (ratio $RATIO) and INT8 quantization..."
    python src/combine/prune_then_quant.py --model gpt2-medium --prune_ratio $RATIO --quantization int8

    echo "Running combined pruning (ratio $RATIO) and INT4 quantization..."
    python src/combine/prune_then_quant.py --model gpt2-medium --prune_ratio $RATIO --quantization int4
done

echo "Combined pruning + quantization experiments completed."