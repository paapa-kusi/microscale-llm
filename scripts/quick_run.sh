#!/bin/bash
# Quick experiment launcher for microscale LLM compression
# Usage: ./scripts/quick_run.sh [baseline|prune|quant|combined|full]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

MODEL="${MODEL:-gpt2-medium}"
WORKERS="${WORKERS:-4}"

case "${1:-help}" in
  baseline)
    echo "=== Running Baseline (No Compression) ==="
    python scripts/run_experiment.py --model "$MODEL"
    ;;
  
  prune)
    echo "=== Running Pruning Experiments ==="
    for ratio in 0.1 0.3 0.5 0.7 0.9; do
      echo "Pruning ratio: $ratio"
      python scripts/run_experiment.py --model "$MODEL" --compression pruning --prune_ratio "$ratio"
    done
    ;;
  
  quant)
    echo "=== Running Quantization Experiments ==="
    for level in INT8 INT4; do
      echo "Quantization level: $level"
      python scripts/run_experiment.py --model "$MODEL" --compression quantization --quant_level "$level"
    done
    ;;
  
  combined)
    echo "=== Running Combined Compression Experiments ==="
    python scripts/run_experiment.py --model "$MODEL" --compression combined --prune_ratio 0.3 --quant_level INT4
    python scripts/run_experiment.py --model "$MODEL" --compression combined --prune_ratio 0.5 --quant_level INT4
    python scripts/run_experiment.py --model "$MODEL" --compression combined --prune_ratio 0.7 --quant_level INT4
    ;;
  
  full)
    echo "=== Running Full Grid Search ==="
    python scripts/run_grid.py --models "$MODEL" --workers "$WORKERS"
    ;;
  
  plot)
    echo "=== Generating Plots ==="
    python scripts/plot_results.py
    echo "Plots saved to results/"
    ;;
  
  edge-sim)
    echo "=== Running Edge Device Simulations ==="
    
    echo "1. Mobile Phone (6GB RAM, 4 cores)"
    docker run --memory=6g --cpus=4 --gpus '"device=0"' \
      -v "$(pwd)":/work microscale-llm \
      python scripts/run_experiment.py --model "$MODEL" --compression combined --prune_ratio 0.5 --quant_level INT4
    
    echo "2. Tablet (4GB RAM, 2 cores)"
    docker run --memory=4g --cpus=2 --gpus '"device=0"' \
      -v "$(pwd)":/work microscale-llm \
      python scripts/run_experiment.py --model "$MODEL" --compression quantization --quant_level INT4
    
    echo "3. IoT Device (2GB RAM, 1 core, CPU only)"
    docker run --memory=2g --cpus=1 \
      -v "$(pwd)":/work microscale-llm \
      python scripts/run_experiment.py --model "$MODEL" --compression quantization --quant_level INT8
    ;;
  
  extrinsic)
    echo "=== Running Extrinsic Evaluation (Sentiment) ==="
    echo "Baseline:"
    python scripts/run_experiment.py --model "$MODEL" --extrinsic
    echo ""
    echo "INT4 Quantized:"
    python scripts/run_experiment.py --model "$MODEL" --compression quantization --quant_level INT4 --extrinsic
    echo ""
    echo "Combined (0.5 prune + INT4):"
    python scripts/run_experiment.py --model "$MODEL" --compression combined --prune_ratio 0.5 --quant_level INT4 --extrinsic
    ;;
  
  smoke)
    echo "=== Running Smoke Test ==="
    python scripts/smoke_eval.py
    ;;
  
  clean)
    echo "=== Cleaning Results ==="
    read -p "Delete all results? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      rm -rf results/*.json results/*.csv results/*.png results/*.txt
      echo "Results cleaned."
    else
      echo "Cancelled."
    fi
    ;;
  
  help|*)
    cat <<EOF
Microscale LLM Compression - Quick Run Script

Usage: ./scripts/quick_run.sh [command]

Commands:
  baseline    - Run baseline (no compression) experiment
  prune       - Run pruning experiments (0.1, 0.3, 0.5, 0.7, 0.9)
  quant       - Run quantization experiments (INT8, INT4)
  combined    - Run combined compression experiments
  full        - Run full grid search (all combinations)
  plot        - Generate visualization plots
  edge-sim    - Run edge device simulations (requires Docker)
  extrinsic   - Run extrinsic task evaluation (sentiment)
  smoke       - Run quick smoke test
  clean       - Clean results directory
  help        - Show this help message

Environment Variables:
  MODEL       - Model to use (default: gpt2-medium)
  WORKERS     - Number of parallel workers (default: 4)

Examples:
  # Run baseline
  ./scripts/quick_run.sh baseline

  # Run full grid with 8 workers
  WORKERS=8 ./scripts/quick_run.sh full

  # Run pruning experiments on different model
  MODEL=gpt2-large ./scripts/quick_run.sh prune

  # Generate plots
  ./scripts/quick_run.sh plot

  # Run edge simulations
  ./scripts/quick_run.sh edge-sim

For more details, see IMPLEMENTATION.md
EOF
    ;;
esac

echo ""
echo "=== Complete ==="
