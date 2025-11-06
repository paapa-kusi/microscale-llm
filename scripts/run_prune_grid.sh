#!/usr/bin/env bash
set -e

# Run pruning experiments and save results

# Activate environment if available (ignore errors on CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

PRUNE_RATIOS=(0.1 0.5 0.9)
# Focus on models that fit common GPUs by default
MODELS_ARRAY=(${MODELS:-gpt2 gpt2-medium gpt2-large})
SEED=${SEED:-42}

for MODEL in "${MODELS_ARRAY[@]}"; do
    echo "========================================"
    echo "Pruning experiments for ${MODEL}"
    echo "========================================"
    for RATIO in "${PRUNE_RATIOS[@]}"; do
        echo "Running pruning with ratio $RATIO on $MODEL ..."
        python scripts/run_experiment.py \
          --model "${MODEL}" \
          --compression pruning \
          --prune_ratio "${RATIO}" \
          --seed "${SEED}"
        echo
        echo "---"
        echo
    done
done

echo "Pruning experiments completed for all models. Results in results/"
