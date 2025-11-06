#!/usr/bin/env python3
"""
Quick smoke test for the microscale-llm evaluation pipeline.
Runs a short evaluation on a small GPT-2 model and exercises pruning and INT8 quantization paths.

Usage:
  python scripts/smoke_eval.py --device auto --model gpt2 --quick

This script is intentionally conservative (uses `gpt2` by default) and prints
process RSS and GPU memory (if CUDA available). It avoids heavy 4-bit flows by default
but will attempt INT4 via bitsandbytes if the environment supports it.
"""

import os
import sys
import time
import gc
import argparse
import traceback

import psutil
import torch

# make repo importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoTokenizer, GPT2LMHeadModel

# import our pipeline helpers
from src.baselines.eval_baseline_models import (
    evaluate_model,
    apply_pruning,
    apply_quantization,
    move_batch_to_device,
    device as module_device,
    load_compressed_model,
)


def print_mem(prefix=""):
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    print(f"{prefix} RSS: {rss_mb:.1f} MB")
    if torch.cuda.is_available():
        try:
            print(f"{prefix} CUDA allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
            print(f"{prefix} CUDA reserved:  {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")
        except Exception:
            pass


def make_dataset(tokenizer, n_repeat=2, max_length=64):
    sentences = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Test sentence for evaluation.",
    ] * n_repeat
    enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    dataset = []
    for i in range(enc["input_ids"].size(0)):
        dataset.append({
            "input_ids": enc["input_ids"][i].unsqueeze(0),
            "attention_mask": enc["attention_mask"][i].unsqueeze(0),
            "labels": enc["input_ids"][i].unsqueeze(0),
        })
    return dataset


def set_device(arg_device: str):
    if arg_device == "auto":
        return module_device
    if arg_device == "cpu":
        return torch.device("cpu")
    if arg_device == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("GPU requested but not available, falling back to CPU")
            return torch.device("cpu")
    raise ValueError("device must be one of auto|cpu|gpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--quick", action="store_true", help="run a shorter/faster smoke test")
    args = parser.parse_args()

    dev = set_device(args.device)
    print(f"Effective device: {dev}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    dataset = make_dataset(tokenizer, n_repeat=1 if args.quick else 2)

    # Load a small model to keep this lightweight
    print("Loading model (small) ...")
    try:
        model = GPT2LMHeadModel.from_pretrained(args.model)
        try:
            model.to(dev)
        except Exception:
            pass
    except Exception as e:
        print("Failed to load small model:", e)
        traceback.print_exc()
        return

    print_mem("Before baseline:")
    print("Running baseline evaluation (perplexity)...")
    try:
        metrics = evaluate_model(model, dataset, metrics={})
        print("Baseline metrics:", metrics)
    except Exception as e:
        print("Baseline evaluation failed:", e)
        traceback.print_exc()

    # Pruning smoke
    print("\nApplying pruning (ratio=0.1) ...")
    try:
        apply_pruning(model, 0.1)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print_mem("After pruning:")
        metrics_p = evaluate_model(model, dataset, metrics={})
        print("Post-pruning metrics:", metrics_p)
    except Exception as e:
        print("Pruning failed:", e)
        traceback.print_exc()

    # INT8 quantization smoke
    print("\nApplying INT8 quantization (dynamic) ...")
    try:
        # quantize in-place (function returns new model object)
        model_q = apply_quantization(model, "INT8", model_name=args.model)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print_mem("After INT8 quant:")
        metrics_q = evaluate_model(model_q, dataset, metrics={})
        print("Post-INT8 metrics:", metrics_q)
    except Exception as e:
        print("INT8 quantization failed:", e)
        traceback.print_exc()

    # INT4 attempt (optional, may fallback)
    print("\nAttempting INT4 quantization (bitsandbytes) - may fail if bnb not properly installed")
    try:
        model_4 = apply_quantization(model, "INT4", model_name=args.model)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print_mem("After INT4 attempt:")
        metrics_4 = evaluate_model(model_4, dataset, metrics={})
        print("Post-INT4 metrics:", metrics_4)
    except Exception as e:
        print("INT4 path failed:", e)
        traceback.print_exc()

    print('\nSmoke test completed.')


if __name__ == "__main__":
    main()
