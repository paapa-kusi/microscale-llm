#!/usr/bin/env bash
set -e

conda activate microscale-llm

PRUNE_RATIOS=(0.1 0.5 0.9)

for RATIO in "${PRUNE_RATIOS[@]}"; do
    echo "Running structured pruning with ratio $RATIO..."
    python src/prune/prune_heads_gpt2.py --model gpt2-medium --prune_ratio $RATIO

    echo "Running unstructured pruning with ratio $RATIO..."
    python src/prune/prune_unstructured.py --model gpt2-medium --prune_ratio $RATIO
done

echo "Pruning experiments completed."