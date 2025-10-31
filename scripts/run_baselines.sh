#!/usr/bin/env bash
set -e

source /apps/conda/25.7.0/etc/profile.d/conda.sh
conda activate microscale-llm

echo "Running baseline evaluation for GPT-2 Medium..."
python src/baselines/eval_baseline_models.py --model gpt2-medium --task perplexity

echo "Running baseline evaluation for LLaMA-2-7B..."
python src/baselines/eval_baseline_models.py --model llama-2-7b --task perplexity

echo "Baseline evaluations completed."