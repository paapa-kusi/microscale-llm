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
# Only test models that are also used in pruning/quantization experiments
MODELS_ARRAY=(${MODELS:-gpt2-medium gpt2-large})
SEED=${SEED:-42}
TRIALS=${TRIALS:-1}

for MODEL in "${MODELS_ARRAY[@]}"; do
  echo "========================================"
  echo "Running baseline evaluation for ${MODEL}"
  echo "========================================"
  for (( t=0; t<TRIALS; t++ )); do
    SEED_EFF=$((SEED + t))
    echo "Trial $((t+1))/${TRIALS} (seed=${SEED_EFF})"
    python scripts/run_experiment.py --model "${MODEL}" --seed "${SEED_EFF}"
    echo
  done
done

echo "Baseline completed for all models. Results in results/"
