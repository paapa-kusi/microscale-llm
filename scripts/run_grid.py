#!/usr/bin/env python3
"""
Run a grid of experiments (pruning, quantization, combined) by calling
`scripts/run_experiment.py` for each configuration. Assigns GPUs in round-robin
and supports simple parallelism.

Usage examples:
  # default grid (gpt2-medium)
  python scripts/run_grid.py

  # limit to CPU-only runs
  CUDA_VISIBLE_DEVICES= python scripts/run_grid.py --use-gpu false

  # run with 4 parallel workers
  python scripts/run_grid.py --workers 4

  # dry run (print commands only)
  python scripts/run_grid.py --dry-run
"""
import os
import sys
import subprocess
import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

RUN_SCRIPT = os.path.join(ROOT, "scripts", "run_experiment.py")


def build_commands(models, prune_ratios, quant_levels, modes, quick=False, trials=1):
    cmds = []
    modes = set(modes)
    for model in models:
        # baseline (no compression)
        if "baseline" in modes:
            for trial in range(trials):
                cmds.append((model, None, None, trial))
        # pruning only
        if "pruning" in modes:
            for p in prune_ratios:
                for trial in range(trials):
                    cmds.append((model, "pruning", str(p), trial))
        # quant only
        if "quantization" in modes:
            for q in quant_levels:
                for trial in range(trials):
                    cmds.append((model, "quantization", q, trial))
        # combined
        if "combined" in modes:
            for p, q in itertools.product(prune_ratios, quant_levels):
                for trial in range(trials):
                    # pass both prune and quant via --compression combined
                    cmds.append((model, "combined", f"{p}:{q}", trial))
    return cmds


def make_subprocess_command(model, compression, config, trial=0, quick=False,
                            perplexity_dataset=None, perplexity_split=None, perplexity_samples=None,
                            extrinsic=False, downstream_dataset=None, downstream_split=None, extrinsic_samples=None,
                            run_id=None, skip_if_exists=False,
                            seq_length=None, batch_size=None, precision=None, use_flash_attn=False,
                            base_seed: int = 42):
    base = [sys.executable, RUN_SCRIPT, "--model", model]
    if compression is None:
        pass
    elif compression == "pruning":
        base += ["--compression", "pruning", "--prune_ratio", str(config)]
    elif compression == "quantization":
        base += ["--compression", "quantization", "--quant_level", str(config)]
    elif compression == "combined":
        p, q = config.split(":")
        base += ["--compression", "combined", "--prune_ratio", str(p), "--quant_level", str(q)]
    if quick:
        base += ["--quick"]
    if run_id:
        base += ["--run-id", run_id]
    if skip_if_exists:
        base += ["--skip-if-exists"]
    # Pass-through model/eval configuration
    if isinstance(seq_length, int):
        base += ["--seq-length", str(seq_length)]
    if isinstance(batch_size, int):
        base += ["--batch-size", str(batch_size)]
    if isinstance(precision, str) and precision:
        base += ["--precision", str(precision)]
    if use_flash_attn:
        base += ["--use-flash-attn"]
    # Dataset/eval args
    if perplexity_dataset:
        base += ["--perplexity-dataset", str(perplexity_dataset)]
    if perplexity_split:
        base += ["--perplexity-split", str(perplexity_split)]
    if isinstance(perplexity_samples, int) and perplexity_samples > 0:
        base += ["--perplexity-samples", str(perplexity_samples)]
    if extrinsic:
        base += ["--extrinsic"]
        if downstream_dataset:
            base += ["--downstream-dataset", str(downstream_dataset)]
        if downstream_split:
            base += ["--downstream-split", str(downstream_split)]
        if isinstance(extrinsic_samples, int) and extrinsic_samples > 0:
            base += ["--extrinsic-samples", str(extrinsic_samples)]
    # Add seed for reproducibility (different per trial)
    base += ["--seed", str(base_seed + trial)]
    return base


def worker(cmd, gpu_id, use_gpu, dry_run=False):
    env = os.environ.copy()
    if use_gpu and gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        # force CPU
        env["CUDA_VISIBLE_DEVICES"] = ""
    if dry_run:
        print("DRY:", "CUDA_VISIBLE_DEVICES=" + env.get("CUDA_VISIBLE_DEVICES", ""), " ", " ".join(cmd))
        return 0, "dry-run"
    print(f"Running: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
        return proc.returncode, proc.stdout + proc.stderr
    except Exception as e:
        return 1, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
    default=["gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/gpt-neo-2.7B"],
        help="models to test (add/remove as needed depending on availability)",
    )
    parser.add_argument("--prune-ratios", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    parser.add_argument("--quant-levels", nargs="*", default=["INT8", "INT4"]) 
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per configuration for statistical robustness")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; per-trial seeds are seed+trial")
    parser.add_argument("--use-gpu", type=lambda s: s.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--modes", nargs="*", default=["baseline", "pruning", "quantization", "combined"],
                        help="Which experiment types to run: baseline, pruning, quantization, combined")
    parser.add_argument("--resume", action="store_true", help="Skip scheduling configs whose result JSON already exists")
    # Model/eval configuration passthrough
    parser.add_argument("--seq-length", type=int, default=None, help="Max sequence length to pass to run_experiment.py")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size to pass to run_experiment.py")
    parser.add_argument("--precision", type=str, default=None, help="Precision (fp32, fp16, bf16) to pass to run_experiment.py")
    parser.add_argument("--use-flash-attn", action="store_true", help="Enable Flash Attention 2 if available")
    # Dataset/evaluation configuration
    parser.add_argument("--perplexity-dataset", type=str, default="wikitext-2",
                        help="Dataset to use for perplexity eval (e.g., wikitext-2, wikitext-2-raw-v1)")
    parser.add_argument("--perplexity-split", type=str, default="test",
                        help="Split for perplexity dataset (e.g., validation, test)")
    parser.add_argument("--perplexity-samples", type=int, default=200,
                        help="Max samples for perplexity eval (reduced when --quick)")
    parser.add_argument("--extrinsic", action="store_true", help="Enable extrinsic evaluation (sentiment)")
    parser.add_argument("--downstream-dataset", type=str, default="sst2",
                        help="Downstream dataset for sentiment (sst2 or imdb)")
    parser.add_argument("--downstream-split", type=str, default="validation",
                        help="Split for downstream dataset (validation or test)")
    parser.add_argument("--extrinsic-samples", type=int, default=100,
                        help="Number of samples for extrinsic eval (reduced when --quick)")
    args = parser.parse_args()

    # detect available GPUs
    try:
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0
    if not args.use_gpu:
        n_gpus = 0

    commands = build_commands(args.models, args.prune_ratios, args.quant_levels, args.modes, quick=args.quick, trials=args.trials)
    tasks = []
    for model, comp, cfg, trial in commands:
        # Adjust sample sizes when --quick is enabled
        perp_samples = max(20, min(args.perplexity_samples, 50)) if args.quick else args.perplexity_samples
        extr_samples = max(10, min(args.extrinsic_samples, 30)) if args.quick else args.extrinsic_samples
        # Build deterministic run-id for resume/skip
        if comp is None:
            run_id = f"{model}__baseline__seed={args.seed+trial}"
        elif comp == "pruning":
            run_id = f"{model}__pruning__prune={cfg}__seed={args.seed+trial}"
        elif comp == "quantization":
            run_id = f"{model}__quantization__quant={cfg}__seed={args.seed+trial}"
        elif comp == "combined":
            p, q = str(cfg).split(":")
            run_id = f"{model}__combined__prune={p}__quant={q}__seed={args.seed+trial}"
        else:
            run_id = None
        safe_run_id = run_id.replace("/", "-") if run_id else None
        # Resume check: if JSON exists, skip scheduling
        if args.resume and safe_run_id:
            out_json = os.path.join(ROOT, "results", f"{safe_run_id}.json")
            if os.path.exists(out_json):
                print(f"Resume: skipping existing result {out_json}")
                continue
        cmd = make_subprocess_command(
            model, comp, cfg, trial=trial, quick=args.quick,
            perplexity_dataset=args.perplexity_dataset,
            perplexity_split=args.perplexity_split,
            perplexity_samples=perp_samples,
            extrinsic=args.extrinsic,
            downstream_dataset=args.downstream_dataset,
            downstream_split=args.downstream_split,
            extrinsic_samples=extr_samples,
            run_id=safe_run_id,
            skip_if_exists=args.resume,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            precision=args.precision,
            use_flash_attn=args.use_flash_attn,
            base_seed=args.seed,
        )
        tasks.append(cmd)

    print(f"Prepared {len(tasks)} tasks ({args.trials} trials per config). GPUs available: {n_gpus}. Workers: {args.workers}. dry_run={args.dry_run}")

    # submit tasks with round-robin GPU assignment
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i, cmd in enumerate(tasks):
            if n_gpus > 0:
                gpu_id = i % n_gpus
            else:
                gpu_id = None
            futures.append(ex.submit(worker, cmd, gpu_id, use_gpu=(n_gpus>0), dry_run=args.dry_run))
        for f in as_completed(futures):
            rc, out = f.result()
            results.append((rc, out))

    failed = [r for r in results if r[0] != 0]
    if failed:
        print(f"{len(failed)} tasks failed")
        sys.exit(2)
    print("All tasks completed successfully")

if __name__ == "__main__":
    main()
