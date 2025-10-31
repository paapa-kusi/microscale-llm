#!/usr/bin/env bash
set -e

conda activate microscale-llm

PRUNE_RATIOS=(0.1 0.5 0.9)

for RATIO in "${PRUNE_RATIOS[@]}"; do
    echo "Running combined pruning (ratio $RATIO) and INT8 quantization..."
    python src/combine/prune_then_quant.py --model gpt2-medium --prune_ratio $RATIO --quantization int8

    echo "Running combined pruning (ratio $RATIO) and INT4 quantization..."
    python src/combine/prune_then_quant.py --model gpt2-medium --prune_ratio $RATIO --quantization int4
done

echo "Combined pruning + quantization experiments completed."