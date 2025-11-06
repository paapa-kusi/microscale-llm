#!/usr/bin/env bash
set -e

# Run baselines via scripts/run_experiment.py and record results

# Activate environment if available (ignore errors in CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

# Default models; model-specific settings come from src/model_configs.py
MODELS_ARRAY=(${MODELS:-gpt2-medium gpt2-large mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-v0.1-int4 microsoft/Phi-3-mini-4k-instruct-int4})

for MODEL in "${MODELS_ARRAY[@]}"; do
    echo "========================================"
    echo "Running baseline evaluation for ${MODEL}"
    echo "========================================"
    python scripts/run_experiment.py --model "${MODEL}"
    echo
done

echo "Baseline completed for all models. Results in results/"
