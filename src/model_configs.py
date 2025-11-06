#!/usr/bin/env python3
"""
Model configuration registry for microscale LLM experiments.

Defines recommended sequence lengths, batch sizes, and precision settings
for each model based on memory constraints and performance characteristics.
"""

MODEL_CONFIGS = {
    # GPT-2 Medium: 355M params, moderate memory, FP16
    "gpt2-medium": {
        "seq_length": 1024,
        "batch_size": 4,
        "precision": "fp16",
        "description": "GPT-2 Medium (355M params)",
    },
    
    # GPT-2 Large: 774M params, higher memory, FP16
    "gpt2-large": {
        "seq_length": 1024,
        "batch_size": 2,  # Conservative batch size for larger model
        "precision": "fp16",
        "description": "GPT-2 Large (774M params)",
    },
    
    # Mistral-7B: 7B params, requires Flash Attention for efficiency
    "mistralai/Mistral-7B-v0.1": {
        "seq_length": 1024,
        "batch_size": 2,
        "precision": "fp16",
        "use_flash_attn": True,
        "description": "Mistral-7B FP16 (7B params)",
    },
    
    # Mistral-7B INT4: quantized version for memory efficiency
    "mistralai/Mistral-7B-v0.1-int4": {
        "base_model": "mistralai/Mistral-7B-v0.1",
        "seq_length": 1024,
        "batch_size": 2,
        "precision": "int4",
        "quantization": "INT4",
        "description": "Mistral-7B INT4 quantized (7B params)",
    },
    
    # Llama-3-8B INT4: Meta's Llama 3 model, quantized
    "meta-llama/Meta-Llama-3-8B-int4": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "seq_length": 2048,
        "batch_size": 4,
        "precision": "int4",
        "quantization": "INT4",
        "description": "Llama-3-8B INT4 quantized (8B params)",
    },
    
    # Phi-3-mini: Microsoft's efficient small model, INT4
    "microsoft/Phi-3-mini-4k-instruct-int4": {
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "seq_length": 3072,  
        "batch_size": 4,    
        "precision": "int4",
        "quantization": "INT4",
        "description": "Phi-3-mini INT4 (3.8B params)",
    },
}


def get_model_config(model_name: str) -> dict:
    """
    Get configuration for a model. Returns default config if model not found.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        dict with keys: seq_length, batch_size, precision, etc.
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
    
    # Default configuration for unknown models
    return {
        "seq_length": 512,
        "batch_size": 4,
        "precision": "fp32",
        "description": f"Unknown model: {model_name}",
    }


def list_available_models():
    """Print all configured models with their settings."""
    print("Available model configurations:")
    print("=" * 80)
    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Seq Length:  {config['seq_length']}")
        print(f"  Batch Size:  {config['batch_size']}")
        print(f"  Precision:   {config['precision']}")
        if config.get('quantization'):
            print(f"  Quantization: {config['quantization']}")
        if config.get('use_flash_attn'):
            print(f"  Flash Attention: {config['use_flash_attn']}")


if __name__ == "__main__":
    list_available_models()
