#!/usr/bin/env bash
set -e

# Run quantization experiments and save results

# Activate environment if available (ignore errors on CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

# Focus on models that fit common GPUs by default
MODELS_ARRAY=(${MODELS:-gpt2-medium gpt2-large})

for MODEL in "${MODELS_ARRAY[@]}"; do
    echo "========================================"
    echo "Quantization experiments for ${MODEL}"
    echo "========================================"
    
    echo "Running INT8 quantization on $MODEL ..."
    python scripts/run_experiment.py --model "${MODEL}" --compression quantization --quant_level INT8
    
    echo "Running INT4 quantization on $MODEL ..."
    python scripts/run_experiment.py --model "${MODEL}" --compression quantization --quant_level INT4
    echo
done

echo "Quantization experiments completed for all models. Results in results/"
