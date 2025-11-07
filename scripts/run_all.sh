#!/usr/bin/env bash
set -e

# Full pipeline: baselines -> pruning -> quantization -> combined -> aggregate -> plot

# Activate environment if available (ignore errors on CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

# Parse simple CLI flags (e.g., --trials 3, --seed 123)
TRIALS_ENV=${TRIALS:-}
SEED_ENV=${SEED:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trials)
      TRIALS_ENV=$2; shift 2 ;;
    --seed)
      SEED_ENV=$2; shift 2 ;;
    *)
      # ignore unknown args for now
      shift ;;
  esac
done

# Baseline models to run (tuned to fit common GPUs)
# Focus on models that are also tested in pruning/quantization grids
BASELINE_MODELS=${MODELS:-"gpt2 gpt2-medium gpt2-large"}
SEED=${SEED_ENV:-42}
TRIALS=${TRIALS_ENV:-1}

mkdir -p results
mkdir -p logs

{
  echo "================================================================================"
  echo "MICROSCALE LLM - FULL PIPELINE"
  echo "Baseline Models: ${BASELINE_MODELS}"
  echo "Pruning/Quant (grids): gpt2, gpt2-medium, gpt2-large"
  echo "Seed: ${SEED}"
  echo "Trials: ${TRIALS}"
  echo "================================================================================"
  echo

  echo "[1/6] Baseline runs"
  MODELS="${BASELINE_MODELS}" SEED="${SEED}" TRIALS="${TRIALS}" bash scripts/run_baselines.sh
  
  echo
  echo "[2/6] Pruning grid"
  SEED="${SEED}" TRIALS="${TRIALS}" bash scripts/run_prune_grid.sh

  echo
  echo "[3/6] Quantization grid"
  SEED="${SEED}" TRIALS="${TRIALS}" bash scripts/run_quant_grid.sh

  echo
  echo "[4/6] Combined grid"
  SEED="${SEED}" TRIALS="${TRIALS}" bash scripts/run_combine_grid.sh
  echo
  # Removed aggregated metrics step
  # echo "[5/6] Aggregate results"
  # python scripts/aggregate_results.py --min-trials 1

  echo
  echo "[5/5] Generate summary stats"
  python scripts/plot_results.py

  echo
  echo "================================================================================"
  echo "ALL DONE. Outputs in results/"
  echo "  - results/*.json (per-run data)"
  echo "  - results/metrics.csv (raw results)"
  echo "  - results/aggregated_metrics.csv (aggregated stats)"
  # echo "  - results/aggregated_metrics.csv (aggregated stats)"
  # echo "  - results/*.png (plots)"
  echo "  - results/summary_stats.txt (summary)"
  echo "================================================================================"
} 2>&1 | tee logs/run_all.log
