#!/usr/bin/env python3
"""
Baseline model loading, compression (pruning/quantization), and evaluation utilities.
Used by scripts/run_experiment.py and scripts/smoke_eval.py
"""
from __future__ import annotations

import os
import time
import math
import random
import psutil
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from transformers import AutoModelForCausalLM

# Default device preference
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_batch_to_device(batch: Dict[str, torch.Tensor], dev: Optional[torch.device]) -> Dict[str, torch.Tensor]:
    """Move tensors in batch to a device. If dev is None, return batch unchanged."""
    if dev is None:
        return batch
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


def apply_quantization(model: nn.Module, level: str, model_name: Optional[str] = None, 
                       preserve_pruning: bool = False) -> nn.Module:
    """Apply quantization.

    Levels:
      - INT8/8: Prefer GPU 8-bit (bitsandbytes) when CUDA + bnb available; fallback to CPU dynamic int8
      - QDYN/DYNAMIC: Force CPU dynamic int8
      - INT4/4: bitsandbytes 4-bit reload
    
    Args:
        model: Model to quantize
        level: Quantization level
        model_name: HuggingFace model name (required for bitsandbytes reload paths)
        preserve_pruning: If True, force CPU dynamic quantization to preserve existing pruning.
                         When False, may reload model (discarding pruning) for better GPU quantization.
    """
    level = (level or "").upper()
    if level in ("QDYN", "DYNAMIC"):
        # Force CPU dynamic INT8
        try:
            qmodel = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            return qmodel
        except Exception as e:
            raise RuntimeError(f"Dynamic INT8 quantization failed: {e}")
    if level in ("INT8", "8"):
        # If preserve_pruning=True, use save+reload to keep pruned weights with GPU INT8
        if preserve_pruning and torch.cuda.is_available():
            if not model_name:
                raise RuntimeError("model_name is required for INT8 quantization with preserved pruning")
            
            import tempfile
            import shutil
            from transformers import AutoTokenizer
            
            # Create temp directory to save pruned model
            temp_dir = tempfile.mkdtemp(prefix="pruned_model_int8_")
            try:
                print(f"Saving pruned model to {temp_dir} for GPU INT8 reload...")
                
                # Save pruned model
                model.save_pretrained(temp_dir)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.save_pretrained(temp_dir)
                except Exception:
                    pass
                
                # Count params before reload
                _, nonzero_before = count_params(model)
                
                # Reload in 8-bit on GPU
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    qmodel = AutoModelForCausalLM.from_pretrained(
                        temp_dir,
                        quantization_config=quant_config,
                        device_map="auto",
                    )
                    
                    # Verify pruning preserved
                    _, nonzero_after = count_params(qmodel)
                    sparsity_change = abs(nonzero_before - nonzero_after) / max(1, nonzero_before)
                    if sparsity_change > 0.05:
                        print(f"Warning: Nonzero params changed after INT8 reload: "
                              f"{nonzero_before} -> {nonzero_after} ({sparsity_change*100:.1f}% change)")
                    
                    print(f"Successfully reloaded pruned model in GPU INT8 (nonzero: {nonzero_before} -> {nonzero_after})")
                    return qmodel
                except Exception as e:
                    print(f"Warning: GPU INT8 reload failed ({e}); falling back to CPU dynamic INT8")
                    # Fall through to CPU dynamic below
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        
        # CPU dynamic INT8 (for preserve_pruning without GPU, or as fallback)
        if preserve_pruning:
            try:
                qmodel = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                    model.cpu(), {nn.Linear}, dtype=torch.qint8
                )
                return qmodel
            except Exception as e:
                raise RuntimeError(f"INT8 dynamic quantization failed: {e}")
        
        # Prefer GPU 8-bit with bitsandbytes if available (but this reloads model)
        if torch.cuda.is_available():
            if not model_name:
                raise RuntimeError("model_name is required to reload in 8-bit quantization")
            try:
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                    qmodel = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quant_config,
                        device_map="auto",
                    )
                except ImportError:
                    # Older transformers may not have BitsAndBytesConfig; try legacy kwarg
                    qmodel = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        load_in_8bit=True,  # type: ignore[call-arg]
                        device_map="auto",
                    )
                return qmodel
            except Exception as e:
                print(f"Warning: GPU 8-bit path failed ({e}); falling back to dynamic INT8 on CPU")
                # fall through to CPU dynamic
        # CPU dynamic int8 fallback
        try:
            qmodel = torch.quantization.quantize_dynamic(  # type: ignore[attr-defined]
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            return qmodel
        except Exception as e:
            raise RuntimeError(f"INT8 quantization failed: {e}")
    elif level in ("INT4", "4"):
        # INT4 with preserve_pruning: save pruned model, then reload in 4-bit
        if preserve_pruning:
            if not model_name:
                raise RuntimeError("model_name is required for INT4 quantization with preserved pruning")
            
            import tempfile
            import shutil
            from transformers import AutoTokenizer
            
            # Create temp directory to save pruned model
            temp_dir = tempfile.mkdtemp(prefix="pruned_model_")
            try:
                print(f"Saving pruned model to {temp_dir} for INT4 reload...")
                
                # Save pruned model and tokenizer
                model.save_pretrained(temp_dir)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    tokenizer.save_pretrained(temp_dir)
                except Exception:
                    pass  # tokenizer not critical for model reload
                
                # Count params before reload to verify preservation
                _, nonzero_before = count_params(model)
                
                # Reload in 4-bit
                try:
                    from transformers import BitsAndBytesConfig  # type: ignore
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if hasattr(torch, 'bfloat16') else torch.float16,
                    )
                    qmodel = AutoModelForCausalLM.from_pretrained(
                        temp_dir,
                        quantization_config=quant_config,
                        device_map="auto",
                    )
                    
                    # Verify pruning was preserved (approximate check)
                    _, nonzero_after = count_params(qmodel)
                    sparsity_change = abs(nonzero_before - nonzero_after) / max(1, nonzero_before)
                    if sparsity_change > 0.05:  # Allow 5% tolerance for quantization effects
                        print(f"Warning: Nonzero params changed significantly after INT4 reload: "
                              f"{nonzero_before} -> {nonzero_after} ({sparsity_change*100:.1f}% change)")
                    
                    print(f"Successfully reloaded pruned model in INT4 (nonzero params: {nonzero_before} -> {nonzero_after})")
                    return qmodel
                except Exception as e:
                    raise RuntimeError(f"INT4 reload of pruned model failed: {e}")
            finally:
                # Clean up temp directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
        
        # Standard INT4 path (no pruning preservation)
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
        
        # Prune first, then quantize with preserve_pruning=True
        # (INT4 will use save+reload path to preserve pruning)
        model = apply_pruning(model, ratio)
        model = apply_quantization(model, level, model_name=model_type, preserve_pruning=True)
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

    # Decide device; handle device-mapped (accelerate/bitsandbytes) models specially
    target_dev = device
    dev_for_batch: Optional[torch.device] = target_dev
    
    # Check if model is quantized (CPU-only)
    is_quantized = any(hasattr(m, '_packed_params') for m in model.modules())
    
    if hasattr(model, 'hf_device_map'):
        # For accelerate device-mapped models, avoid moving the model; keep inputs as-is
        dev_for_batch = None
        # Heuristically set eval_device for logging
        try:
            devs = list(getattr(model, 'hf_device_map', {}).values())
            eval_device = 'cuda' if any('cuda' in str(d) for d in devs) else 'cpu'
        except Exception:
            eval_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif is_quantized:
        # Quantized models must stay on CPU
        eval_device = 'cpu'
        dev_for_batch = torch.device('cpu')
        try:
            model.to('cpu')
        except Exception:
            pass  # Already on CPU
    else:
        try:
            model.to(target_dev)
            eval_device = str(target_dev)
        except Exception:
            eval_device = 'cpu'
            dev_for_batch = torch.device('cpu')

    model.eval()

    total_loss = 0.0
    n_batches = 0
    samples = 0
    total_tokens = 0
    start = time.time()
    unstable = False

    # inference_mode can be measurably faster than no_grad for eval-only loops
    with torch.inference_mode():
        for batch in dataset:
            n_batches += 1
            batch_samples = batch['input_ids'].size(0) if 'input_ids' in batch else 1
            samples += batch_samples
            
            if 'input_ids' in batch:
                total_tokens += batch['input_ids'].numel()
            
            b = move_batch_to_device(batch, dev_for_batch if dev_for_batch is not None else None)
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
