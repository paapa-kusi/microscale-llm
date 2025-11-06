#!/usr/bin/env python3
"""
Baseline model loading, compression (pruning/quantization), and evaluation utilities.
Used by scripts/run_experiment.py and scripts/smoke_eval.py
"""
from __future__ import annotations

import os
import time
import math
import psutil
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from transformers import AutoModelForCausalLM

# Default device preference
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def move_batch_to_device(batch: Dict[str, torch.Tensor], dev: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            try:
                out[k] = v.to(dev)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


def _linear_modules(model: nn.Module) -> List[nn.Module]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def count_params(model: nn.Module) -> Tuple[int, int]:
    total = 0
    nonzero = 0
    for p in model.parameters():
        total += p.numel()
        try:
            nonzero += (p.detach().to("cpu") != 0).sum().item()
        except Exception:
            nonzero += p.numel()  # fallback if counting fails
    return total, nonzero


def measure_model_size_mb(model: nn.Module) -> float:
    # Approximate in-memory size from parameter sizes
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2)


def apply_pruning(model: nn.Module, prune_ratio: float) -> nn.Module:
    """
    Global unstructured L1 pruning across Linear layers.
    Modifies the model in-place and removes reparametrization so zeros are baked in.
    """
    parameters_to_prune = [(m, 'weight') for m in _linear_modules(model)]
    if not parameters_to_prune:
        return model
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=float(prune_ratio),
    )
    # Remove reparametrization to finalize zeros
    for m, name in parameters_to_prune:
        try:
            prune.remove(m, name)
        except Exception:
            pass
    return model


def apply_quantization(model: nn.Module, level: str, model_name: Optional[str] = None) -> nn.Module:
    level = (level or "").upper()
    if level in ("INT8", "8", "QDYN", "DYNAMIC"):
        try:
            qmodel = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            return qmodel
        except Exception as e:
            raise RuntimeError(f"INT8 quantization failed: {e}")
    elif level in ("INT4", "4"):
        # Try bitsandbytes path by reloading the model in 4-bit
        try:
            from transformers import BitsAndBytesConfig  # type: ignore
        except Exception as e:
            raise RuntimeError("bitsandbytes/transformers 4-bit support not available: " + str(e))
        if not model_name:
            raise RuntimeError("model_name is required to reload in 4-bit quantization")
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16,
            )
            qmodel = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
            return qmodel
        except Exception as e:
            raise RuntimeError(f"INT4 (bitsandbytes) quantization failed: {e}")
    else:
        raise ValueError(f"Unknown quantization level: {level}")


def load_compressed_model(model_type: str, compression: Optional[str], compression_config, 
                          precision: str = "fp32", use_flash_attn: bool = False) -> nn.Module:
    """
    Load a causal LM and optionally apply pruning/quantization or both.
    compression can be None, 'pruning', 'quantization', or 'combined'.
    For combined, compression_config should be {'pruning': <ratio>, 'quantization': <level>}.
    
    Args:
        model_type: HuggingFace model identifier
        compression: Type of compression (None, 'pruning', 'quantization', 'combined')
        compression_config: Configuration for compression
        precision: Model precision ('fp32', 'fp16', 'bf16', 'int4', 'int8')
        use_flash_attn: Whether to use Flash Attention (for supported models)
    """
    load_kwargs = {}
    
    # Handle precision and attention
    if precision == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    
    if use_flash_attn:
        # Enable Flash Attention 2 only if the package is available; otherwise, fall back silently
        try:
            import flash_attn  # type: ignore  # noqa: F401
            load_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            # Package not available; do not set attn_implementation to avoid ImportError in transformers
            print("Note: flash-attn not installed; falling back to default attention.")
    
    # Always start from a fresh model with specified precision
    try:
        model = AutoModelForCausalLM.from_pretrained(model_type, **load_kwargs)
    except ImportError as e:
        # If Flash Attention was requested and triggers an ImportError inside transformers, retry without it
        if use_flash_attn and load_kwargs.get("attn_implementation") == "flash_attention_2":
            print("Warning: Flash Attention requested but unavailable. Retrying without it...")
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(model_type, **load_kwargs)
        else:
            raise

    if not compression:
        return model

    ctype = compression.lower()
    if ctype == 'pruning':
        ratio = float(compression_config)
        model = apply_pruning(model, ratio)
        return model
    elif ctype == 'quantization':
        level = str(compression_config).upper()
        model = apply_quantization(model, level, model_name=model_type)
        return model
    elif ctype == 'combined':
        ratio = float(compression_config.get('pruning', 0.5))
        level = str(compression_config.get('quantization', 'INT8')).upper()
        # Prune first, then quantize
        model = apply_pruning(model, ratio)
        model = apply_quantization(model, level, model_name=model_type)
        return model
    else:
        raise ValueError(f"Unsupported compression type: {compression}")


def evaluate_model(model: nn.Module, dataset: List[Dict[str, torch.Tensor]], metrics: Dict, 
                   batch_size: int = 1) -> Dict:
    """
    Evaluate LM loss/perplexity on a batched dataset list and collect system metrics.
    
    Args:
        model: The model to evaluate
        dataset: List of batched inputs (each is a dict with input_ids, attention_mask, labels)
        metrics: Dictionary to update with evaluation metrics
        batch_size: Batch size for evaluation (used for throughput calculation)

    Returns metrics dict containing (keys are read downstream):
      - perplexity
      - inference_speed (samples/sec)
      - tokens_per_sec
      - memory_footprint_mb
      - total_params, nonzero_params, sparsity
      - model_size_mb
      - eval_device ('cpu'|'cuda')
      - pruning_unstable (bool)
    """
    proc = psutil.Process(os.getpid())

    # Decide device; try to honor CUDA but some quantized models are CPU-only
    target_dev = device
    try:
        model.to(target_dev)
        eval_device = str(target_dev)
    except Exception:
        eval_device = 'cpu'

    model.eval()

    total_loss = 0.0
    n_batches = 0
    samples = 0
    total_tokens = 0
    start = time.time()
    unstable = False

    with torch.no_grad():
        for batch in dataset:
            n_batches += 1
            batch_samples = batch['input_ids'].size(0) if 'input_ids' in batch else 1
            samples += batch_samples
            
            if 'input_ids' in batch:
                total_tokens += batch['input_ids'].numel()
            
            b = move_batch_to_device(batch, torch.device(eval_device))
            try:
                out = model(**{k: v for k, v in b.items() if k in ("input_ids", "attention_mask", "labels")})
                loss = out.loss if hasattr(out, 'loss') and out.loss is not None else None
                if loss is None:
                    # If model doesn't compute loss, compute manually
                    logits = out.logits  # [B, T, V]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = b['labels'][..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if torch.isnan(loss) or torch.isinf(loss):
                    unstable = True
                total_loss += float(loss.detach().cpu().item())
            except Exception as e:
                # If forward fails (e.g., device mismatch), try CPU
                try:
                    out = model.cpu()(**{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
                    loss = out.loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        unstable = True
                    total_loss += float(loss.detach().cpu().item())
                    eval_device = 'cpu'
                except Exception:
                    # Skip this sample
                    n_batches -= 1
                    samples -= batch_samples
                    continue

    elapsed = max(1e-6, time.time() - start)
    avg_loss = total_loss / max(1, n_batches)
    perp = math.exp(avg_loss) if avg_loss < 50 else float('inf')

    mem_mb = proc.memory_info().rss / (1024 ** 2)
    total_params, nonzero_params = count_params(model)
    sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0
    size_mb = measure_model_size_mb(model)

    metrics.update({
        'perplexity': perp,
        'inference_speed': samples / elapsed,  # samples/sec
        'tokens_per_sec': total_tokens / elapsed if total_tokens > 0 else 0.0,
        'memory_footprint_mb': mem_mb,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'model_size_mb': size_mb,
        'eval_device': eval_device,
        'pruning_unstable': unstable,
    })
    return metrics
