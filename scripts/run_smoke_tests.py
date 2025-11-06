#!/usr/bin/env python3
"""
Minimal reproducible tests for the microscale-llm pipeline.
Runs 3 quick experiments (baseline, pruning=0.1, quant=INT8), aggregates,
plots, and asserts that outputs look sane. Intended to run on CPU or GPU.
"""
import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RUN_EXPERIMENT = str(ROOT / "scripts" / "run_experiment.py")
AGG = [PY, str(ROOT / "scripts" / "aggregate_results.py"), "--min-trials", "1"]
PLOT = [PY, str(ROOT / "scripts" / "plot_results.py")]


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run smoke tests for microscale LLM pipeline")
    parser.add_argument("--model", type=str, default=None, help="Model to test (default: from env SMOKE_MODEL or 'gpt2')")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick mode (default: True)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Keep model tiny to run fast and avoid OOMs on CPU
    model = args.model or os.environ.get("SMOKE_MODEL", "gpt2")
    seed = args.seed
    
    print("=" * 80)
    print("MICROSCALE LLM - SMOKE TEST")
    print(f"Model: {model}")
    print(f"Seed: {seed}")
    print("=" * 80)
    
    common = [PY, RUN_EXPERIMENT, "--model", model, "--seed", str(seed)]
    if args.quick:
        common.append("--quick")

    # 1) Baseline
    print("\n[1/3] Baseline")
    run(common[:])
    # 2) Pruning 0.1
    print("\n[2/3] Pruning (0.1)")
    run(common + ["--compression", "pruning", "--prune_ratio", "0.1"]) 
    # 3) INT8 (CPU dynamic) quantization
    print("\n[3/3] Quantization (INT8)")
    run(common + ["--compression", "quantization", "--quant_level", "INT8"]) 

    # Aggregate and plot
    print("\n[4/5] Aggregating results")
    run(AGG)
    print("\n[5/5] Generating plots")
    run(PLOT)

    # Basic sanity checks
    metrics_csv = RESULTS / "metrics.csv"
    assert metrics_csv.exists(), "metrics.csv not created"
    rows = sum(1 for _ in metrics_csv.open()) - 1
    assert rows >= 3, f"Expected at least 3 rows in metrics.csv, found {rows}"

    agg_csv = RESULTS / "aggregated_metrics.csv"
    assert agg_csv.exists(), "aggregated_metrics.csv not created"

    # Check last JSON exists and contains core keys
    json_files = sorted(RESULTS.glob("*.json"))
    assert json_files, "No per-run JSON files written"
    with json_files[-1].open() as f:
        data = json.load(f)
    for k in ("model", "compression", "metrics", "system"):
        assert k in data, f"Missing '{k}' in run JSON"

    print("\n" + "=" * 80)
    print("SMOKE TESTS PASSED")
    print("=" * 80)
    print(f"Outputs in {RESULTS}/")


if __name__ == "__main__":
    main()
