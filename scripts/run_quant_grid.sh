#!/usr/bin/env bash
set -e

# Activate the environment
conda activate microscale-llm

# Run quantization experiments
echo "Running INT8 quantization..."
python src/quant/quant_bnb_8bit_4bit.py --model gpt2-medium --quantization int8

echo "Running INT4 quantization..."
python src/quant/quant_bnb_8bit_4bit.py --model gpt2-medium --quantization int4

echo "Quantization experiments completed."