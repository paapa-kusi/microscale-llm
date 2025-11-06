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
    # Keep model tiny to run fast and avoid OOMs on CPU
    model = os.environ.get("SMOKE_MODEL", "gpt2")
    common = [PY, RUN_EXPERIMENT, "--model", model, "--quick"]

    # 1) Baseline
    run(common[:])
    # 2) Pruning 0.1
    run(common + ["--compression", "pruning", "--prune_ratio", "0.1"]) 
    # 3) INT8 (CPU dynamic) quantization
    run(common + ["--compression", "quantization", "--quant_level", "INT8"]) 

    # Aggregate and plot
    run(AGG)
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

    print("Smoke tests PASSED. Outputs in results/")


if __name__ == "__main__":
    main()
