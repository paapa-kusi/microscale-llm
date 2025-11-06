#!/usr/bin/env bash
set -e

# Full pipeline: baselines -> pruning -> quantization -> combined -> aggregate -> plot

# Activate environment if available (ignore errors on CI)
if command -v conda &> /dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate microscale-llm || true
fi

# Baseline models to run (tuned to fit common GPUs)
BASELINE_MODELS=${MODELS:-"gpt2-medium gpt2-large mistralai/Mistral-7B-v0.1 mistralai/Mistral-7B-v0.1-int4"}

mkdir -p results
mkdir -p logs

{
  echo "================================================================================"
  echo "MICROSCALE LLM - FULL PIPELINE"
  echo "Baseline Models: ${BASELINE_MODELS}"
  echo "Pruning/Quant: gpt2-medium, gpt2-large"
  echo "================================================================================"
  echo

  echo "[1/6] Baseline runs"
  MODELS="${BASELINE_MODELS}" bash scripts/run_baselines.sh

  echo
  echo "[2/6] Pruning grid"
  bash scripts/run_prune_grid.sh

  echo
  echo "[3/6] Quantization grid"
  bash scripts/run_quant_grid.sh

  echo
  echo "[4/6] Combined grid"
  bash scripts/run_combine_grid.sh

  echo
  echo "[5/6] Aggregate results"
  python scripts/aggregate_results.py --min-trials 1

  echo
  echo "[6/6] Generate plots"
  python scripts/plot_results.py

  echo
  echo "================================================================================"
  echo "ALL DONE. Outputs in results/"
  echo "  - results/*.json (per-run data)"
  echo "  - results/metrics.csv (raw results)"
  echo "  - results/aggregated_metrics.csv (aggregated stats)"
  echo "  - results/*.png (plots)"
  echo "  - results/summary_stats.txt (text summary)"
  echo "================================================================================"
} 2>&1 | tee logs/run_all.log
