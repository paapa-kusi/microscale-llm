#!/usr/bin/env bash
set -e

# Run combined pruning + quantization experiments and save results

# Activate environment if available (ignore errors on CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

# Grid settings (overridable via env)
PRUNE_RATIOS=(${PRUNE_RATIOS:-0.1 0.5 0.9})
# Focus on models that fit common GPUs by default
MODELS_ARRAY=(${MODELS:-gpt2-medium gpt2-large})
QUANT_LEVELS=(${QUANT_LEVELS:-INT8 INT4})
SEED=${SEED:-42}

# Determine parallel workers (env WORKERS wins; else use number of GPUs or CPU cores)
if [[ -n "${WORKERS}" ]]; then
  N_WORKERS=${WORKERS}
else
  # Try to detect GPUs via python; fallback to nproc
  GPU_COUNT=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
  )
  if [[ "${GPU_COUNT}" -gt 0 ]]; then
    N_WORKERS=${GPU_COUNT}
  else
    if command -v nproc >/dev/null 2>&1; then
      N_WORKERS=$(nproc)
    else
      N_WORKERS=2
    fi
  fi
fi

# Fast args to reduce wall-clock while keeping comparability
FAST_ARGS=(--quick --perplexity-samples 50 --seq-length 256 --batch-size 4 --precision bf16 --use-flash-attn)

echo "========================================"
echo "Running combined pruning+quantization grid via run_grid.py"
echo "Models: ${MODELS_ARRAY[*]} | Prune: ${PRUNE_RATIOS[*]} | Quant: ${QUANT_LEVELS[*]} | Workers: ${N_WORKERS} | Seed: ${SEED}"
echo "========================================"

python scripts/run_grid.py \
  --models "${MODELS_ARRAY[@]}" \
  --modes combined \
  --prune-ratios "${PRUNE_RATIOS[@]}" \
  --quant-levels "${QUANT_LEVELS[@]}" \
  --workers "${N_WORKERS}" \
  --trials "${TRIALS:-1}" \
  --seed "${SEED}" \
  --resume \
  "${FAST_ARGS[@]}"

echo "Combined pruning + quantization experiments completed for selected models. Results in results/"
