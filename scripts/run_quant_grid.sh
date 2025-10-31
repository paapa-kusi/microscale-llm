#!/usr/bin/env bash
set -e

conda activate microscale-llm

echo "Running INT8 quantization..."
python src/quant/quant_bnb_8bit_4bit.py --model gpt2-medium --quantization int8

echo "Running INT4 quantization..."
python src/quant/quant_bnb_8bit_4bit.py --model gpt2-medium --quantization int4

echo "Quantization experiments completed."