import time
import torch
import psutil
import os
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM

def evaluate_model(model, dataset, metrics):
    """
    Evaluate a given model on the dataset using specified metrics.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataset:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataset)
    metrics["perplexity"] = torch.exp(torch.tensor(avg_loss)).item()

    # Add system metrics tracking (e.g., inference speed, memory usage)
    system_metrics = {
        "inference_speed": measure_inference_speed(model, dataset),
        "memory_footprint": measure_memory_footprint(model),
    }
    metrics.update(system_metrics)
    return metrics

def measure_inference_speed(model, dataset):
    """
    Measure the inference speed of the model on the dataset.
    """
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in dataset:
            model(**batch)
    end_time = time.time()
    total_time = end_time - start_time
    num_samples = len(dataset)
    return num_samples / total_time  # Samples per second

def measure_memory_footprint(model):
    """
    Measure the memory footprint of the model.
    """
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # Resident Set Size in bytes
    return memory_usage / (1024 ** 2)  # Convert to MB

def load_compressed_model(model_type, compression_type, compression_config):
    """
    Load a compressed model based on the specified type, compression method, and configuration.
    """
    if model_type == "GPT-2 Medium":
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    elif model_type == "LLaMA-2-7B":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if compression_type == "pruning":
        apply_pruning(model, compression_config)
    elif compression_type == "quantization":
        apply_quantization(model, compression_config)
    elif compression_type == "combined":
        apply_pruning(model, compression_config["pruning"])
        apply_quantization(model, compression_config["quantization"])
    else:
        raise ValueError(f"Unsupported compression type: {compression_type}")

    return model

def apply_pruning(model, pruning_ratio):
    """
    Apply pruning to the model with the specified pruning ratio.
    """
    # Placeholder for pruning logic
    print(f"Applying pruning with ratio {pruning_ratio}...")
    # Example: Use torch.nn.utils.prune for structured/unstructured pruning
    pass

def apply_quantization(model, quantization_level):
    """
    Apply quantization to the model with the specified quantization level.
    """
    # Placeholder for quantization logic
    print(f"Applying {quantization_level} quantization...")
    # Example: Use torch.quantization for post-training quantization
    pass

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    
    # Ensure the tokenizer has a padding token
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

    sentences = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Test sentence for evaluation.",
    ] * 4  # repeat to get a few samples

    # Tokenize the sentences with padding and truncation
    enc = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    dataset = []
    for i in range(enc["input_ids"].size(0)):
        dataset.append({
            "input_ids": enc["input_ids"][i].unsqueeze(0),
            "attention_mask": enc["attention_mask"][i].unsqueeze(0),
            "labels": enc["input_ids"][i].unsqueeze(0),
        })

    # Evaluate baseline models
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    baseline_metrics = evaluate_model(model, dataset, metrics={})
    print(f"Baseline GPT-2 Medium: {baseline_metrics}")

    # Evaluate compressed models
    compression_configs = [
        {"type": "pruning", "ratios": [0.1, 0.5, 0.9]},
        {"type": "quantization", "levels": ["INT8", "INT4"]},
        {"type": "combined", "configs": [{"pruning": 0.5, "quantization": "INT8"}]},
    ]
    for config in compression_configs:
        if config["type"] == "pruning":
            for ratio in config["ratios"]:
                model = load_compressed_model("GPT-2 Medium", "pruning", ratio)
                metrics = evaluate_model(model, dataset, metrics={})
                print(f"Pruning {ratio}: {metrics}")
        elif config["type"] == "quantization":
            for level in config["levels"]:
                model = load_compressed_model("GPT-2 Medium", "quantization", level)
                metrics = evaluate_model(model, dataset, metrics={})
                print(f"Quantization {level}: {metrics}")
        elif config["type"] == "combined":
            for combined_config in config["configs"]:
                model = load_compressed_model(
                    "GPT-2 Medium", "combined", combined_config
                )
                metrics = evaluate_model(model, dataset, metrics={})
                print(f"Combined {combined_config}: {metrics}")