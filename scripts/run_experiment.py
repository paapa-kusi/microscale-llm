#!/usr/bin/env python3
"""
Run a single compression experiment and save metrics.
Saves JSON to results/<run_id>.json and appends a row to results/metrics.csv

Usage example:
  python scripts/run_experiment.py --model gpt2-medium --compression pruning --prune_ratio 0.5
  python scripts/run_experiment.py --model gpt2-medium --compression quantization --quant_level INT8
"""
import argparse
import json
import os
import time
import csv
import uuid
import socket
import platform
import shutil

from datetime import datetime

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in os.sys.path:
    os.sys.path.insert(0, ROOT)

from src.baselines.eval_baseline_models import (
    load_compressed_model,
    evaluate_model,
    device as module_device,
)
# Extrinsic evaluation removed from the minimal pipeline for simplicity

RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "metrics.csv")


def write_json(result, run_id):
    path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def ensure_csv_header(header):
    """Ensure metrics.csv has the expected header; rewrite if mismatched."""
    if not os.path.exists(CSV_PATH):
        return
    try:
        with open(CSV_PATH, "r", newline="") as f:
            first_line = f.readline().strip()
        current = [h.strip() for h in first_line.split(",")] if first_line else []
        expected = header
        if current == expected:
            return
        # Migrate: read all rows and rewrite with the expected header
        print("Rewriting metrics.csv to match expected header (dropping unused columns)...")
        import tempfile
        rows = []
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
        with tempfile.NamedTemporaryFile(delete=False, mode="w", newline="") as tmp:
            writer = csv.DictWriter(tmp, fieldnames=expected)
            writer.writeheader()
            for r in rows:
                # Remove unexpected keys (including None from malformed rows)
                r = {k: v for k, v in r.items() if k in expected}
                # backfill missing fields
                for k in expected:
                    if k not in r:
                        r[k] = ""
                writer.writerow(r)
            tmp_path = tmp.name
        try:
            os.replace(tmp_path, CSV_PATH)
        except Exception:
            # Fallback for cross-device moves
            shutil.move(tmp_path, CSV_PATH)
    except Exception as e:
        print(f"Warning: Failed to upgrade metrics.csv header: {e}")


def append_csv(row):
    header = [
        "run_id",
        "timestamp",
        "model",
        "compression",
        "prune_ratio",
        "quant_level",
        "seed",
        "perplexity_dataset",
        "perplexity_split",
        "perplexity",
        "inference_speed",
        "memory_footprint_mb",
        "total_params",
        "nonzero_params",
        "sparsity",
        "model_size_mb",
        "pruning_unstable",
        "eval_device",
        "torch_version",
        "torch_cuda",
        "hostname",
    ]
    # Ensure header is up to date; migrate if needed
    ensure_csv_header(header)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--compression", choices=[None, "pruning", "quantization", "combined"], default=None)
    parser.add_argument("--prune_ratio", type=float, default=None)
    parser.add_argument("--quant_level", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument("--run-id", type=str, default=None, help="Deterministic run id used for result filenames and resume")
    parser.add_argument("--skip-if-exists", action="store_true", help="Skip execution if result JSON for run-id already exists")
    parser.add_argument("--quick", action="store_true", help="run a quick short evaluation")
    parser.add_argument("--seq-length", type=int, default=None, 
                        help="Max sequence length (default: use model config or 512)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for evaluation (default: use model config or 1)")
    parser.add_argument("--precision", type=str, default=None,
                        help="Model precision: fp32, fp16, bf16 (default: use model config)")
    parser.add_argument("--use-flash-attn", action="store_true",
                        help="Enable Flash Attention 2 if available")
    parser.add_argument("--perplexity-dataset", type=str, default="wikitext-2",
                        help="Dataset for perplexity evaluation (e.g., wikitext-2, wikitext-2-raw-v1)")
    parser.add_argument("--perplexity-split", type=str, default="test",
                        help="Split for perplexity dataset (validation/test)")
    parser.add_argument("--perplexity-samples", type=int, default=200,
                        help="Max samples for perplexity eval; reduced when --quick")
    # Extrinsic task flags removed to keep focus on core LM metrics
    args = parser.parse_args()
    
    # Set seed at the start for reproducibility
    from src.baselines.eval_baseline_models import set_seed
    set_seed(args.seed)
    
    # Load model configuration if available
    from src.model_configs import get_model_config
    model_config = get_model_config(args.model)
    
    # Use CLI args if provided, otherwise fall back to model config
    seq_length = args.seq_length if args.seq_length is not None else model_config.get('seq_length', 512)
    batch_size = args.batch_size if args.batch_size is not None else model_config.get('batch_size', 1)
    precision = args.precision if args.precision is not None else model_config.get('precision', 'fp32')
    use_flash_attn = args.use_flash_attn or model_config.get('use_flash_attn', False)
    
    print(f"Model config: seq_length={seq_length}, batch_size={batch_size}, precision={precision}, flash_attn={use_flash_attn}")
    
    # Additional performance optimizations on CUDA
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")  # PyTorch 2.x
        except Exception:
            pass

    # build compression_config
    compression_config = None
    if args.compression == "pruning":
        compression_config = args.prune_ratio if args.prune_ratio is not None else 0.5
    elif args.compression == "quantization":
        compression_config = args.quant_level if args.quant_level is not None else "INT8"
    elif args.compression == "combined":
        compression_config = {"pruning": args.prune_ratio or 0.5, "quantization": args.quant_level or "INT8"}

    # Tokenizer and dataset loading
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model.split('/')[-1], use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    # Helper: build LM evaluation dataset from HF datasets with batching support
    def build_lm_dataset(tokenizer, dataset_name: str, split: str, max_samples: int, 
                        max_length: int = 256, batch_size: int = 1):
        from datasets import load_dataset
        # Map friendly names
        ds = None
        cfg = None
        name = dataset_name.lower()
        if name in ("wikitext-2", "wikitext-2-raw-v1", "wikitext2"):
            ds_name = "wikitext"
            cfg = "wikitext-2-raw-v1"
            ds = load_dataset(ds_name, cfg, split=split)
            text_field = "text"
        elif name in ("ptb", "penn-treebank"):
            ds_name = "ptb_text_only"
            ds = load_dataset(ds_name, split=split)
            text_field = "sentence"
        else:
            # Try to load directly; assume field 'text'
            ds = load_dataset(dataset_name, split=split)
            text_field = "text"

        if max_samples and len(ds) > max_samples:
            import numpy as np
            idx = np.random.choice(len(ds), max_samples, replace=False)
            ds = ds.select(list(idx))

        # Collect texts and batch them
        texts = []
        for ex in ds:
            text = ex.get(text_field, None)
            if text and isinstance(text, str) and len(text.strip()) > 0:
                texts.append(text)
        
        # Create batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=max_length)
            batches.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", None),
                "labels": enc["input_ids"].clone(),
            })
        return batches

    # Build perplexity dataset with proper batching, with simple on-disk caching to avoid repeated tokenization
    perp_samples = max(20, min(args.perplexity_samples, 50)) if args.quick else args.perplexity_samples
    cache_dir = os.path.join(RESULTS_DIR, ".cache", "datasets")
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"{args.perplexity_dataset}__{args.perplexity_split}__n{perp_samples}__L{seq_length}__bs{batch_size}__tok={getattr(tokenizer, 'name_or_path', 'unknown')}.pt".replace("/", "-")
    cache_path = os.path.join(cache_dir, cache_key)
    dataset = None
    used_perplexity_dataset = args.perplexity_dataset
    used_perplexity_split = args.perplexity_split
    if os.path.exists(cache_path):
        try:
            dataset = torch.load(cache_path)
            print(f"Loaded tokenized dataset cache: {cache_path}")
        except Exception:
            dataset = None
    if dataset is None:
        try:
            dataset = build_lm_dataset(tokenizer, args.perplexity_dataset, args.perplexity_split, 
                                       max_samples=perp_samples, max_length=seq_length, 
                                       batch_size=batch_size)
            try:
                torch.save(dataset, cache_path)
                print(f"Saved tokenized dataset cache: {cache_path}")
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to load perplexity dataset '{args.perplexity_dataset}': {e}. Falling back to toy sentences.")
            sentences = [
                "Hello world!", "The quick brown fox jumps over the lazy dog.", "Test sentence for evaluation."
            ] * (1 if args.quick else 4)
            enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=seq_length)
            # Batch the toy sentences
            dataset = [{
                "input_ids": enc["input_ids"],
                "attention_mask": enc.get("attention_mask", None),
                "labels": enc["input_ids"].clone(),
            }]
            used_perplexity_dataset = "toy"
            used_perplexity_split = "n/a"

    # load model with compression
    model_type = args.model
    compression_type = args.compression
    
    # Check if model config specifies a base model (for pre-quantized models)
    if 'base_model' in model_config:
        model_type = model_config['base_model']
    
    try:
        model = load_compressed_model(model_type, compression_type, compression_config,
                                     precision=precision, use_flash_attn=use_flash_attn)
    except Exception as e:
        print("Failed to load compressed model:", e)
        raise

    # run evaluation
    metrics = evaluate_model(model, dataset, metrics={}, batch_size=batch_size)
    
    # Extrinsic evaluation removed

    # Build run id (deterministic if provided)
    if args.run_id:
        run_id = args.run_id
    else:
        # Construct a stable key if possible, else random
        parts = [
            str(args.model),
            str(compression_type or "baseline"),
            f"prune={compression_config.get('pruning') if isinstance(compression_config, dict) else (compression_config if compression_type=='pruning' else '')}",
            f"quant={compression_config.get('quantization') if isinstance(compression_config, dict) else (compression_config if compression_type=='quantization' else '')}",
            f"seed={args.seed}",
        ]
        stable = "__".join([p for p in parts if p is not None])
        # shorten for filename safety
        run_id = stable.replace("/", "-")

    # Early exit if skipping and file exists
    existing_json_path = os.path.join(RESULTS_DIR, f"{run_id}.json")
    if args.skip_if_exists and os.path.exists(existing_json_path):
        print(f"Skip: results already exist at {existing_json_path}")
        return

    result = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "compression": compression_type,
        "compression_config": compression_config,
        "metrics": metrics,
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
    }

    json_path = write_json(result, run_id)
    csv_row = {
        "run_id": run_id,
        "timestamp": result["timestamp"],
        "model": args.model,
        "compression": compression_type,
        "prune_ratio": compression_config.get("pruning") if isinstance(compression_config, dict) else (compression_config if compression_type == "pruning" else None),
        "quant_level": compression_config.get("quantization") if isinstance(compression_config, dict) else (compression_config if compression_type == "quantization" else None),
        "seed": args.seed,
        "perplexity_dataset": used_perplexity_dataset,
        "perplexity_split": used_perplexity_split,
        "perplexity": metrics.get("perplexity"),
        "inference_speed": metrics.get("inference_speed"),
        "memory_footprint_mb": metrics.get("memory_footprint"),
        "total_params": metrics.get("total_params"),
        "nonzero_params": metrics.get("nonzero_params"),
        "sparsity": metrics.get("sparsity"),
        "model_size_mb": metrics.get("model_size_mb"),
        "pruning_unstable": metrics.get("pruning_unstable"),
        "eval_device": metrics.get("eval_device"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "hostname": socket.gethostname(),
    }
    append_csv(csv_row)

    print(f"Saved results to {json_path} and appended CSV row to {CSV_PATH}")


if __name__ == "__main__":
    main()
