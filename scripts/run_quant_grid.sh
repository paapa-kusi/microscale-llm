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
SEED=${SEED:-42}

for MODEL in "${MODELS_ARRAY[@]}"; do
    echo "========================================"
    echo "Quantization experiments for ${MODEL}"
    echo "========================================"
    
  # Fast defaults to shorten wall time while keeping results comparable
  FAST_ARGS=(--quick --perplexity-samples 50 --seq-length 256 --batch-size 4 --precision bf16 --use-flash-attn)

  echo "Running INT8 quantization on $MODEL ..."
  python scripts/run_experiment.py --model "${MODEL}" --compression quantization --quant_level INT8 --seed "${SEED}" "${FAST_ARGS[@]}"
    
    echo "Running INT4 quantization on $MODEL ..."
  python scripts/run_experiment.py --model "${MODEL}" --compression quantization --quant_level INT4 --seed "${SEED}" "${FAST_ARGS[@]}"
    echo
done

echo "Quantization experiments completed for all models. Results in results/"
