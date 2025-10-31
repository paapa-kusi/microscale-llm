import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def benchmark_inference(model_name: str, input_text: str):
    """
    Benchmark the inference time, memory usage, and latency of a model.

    Args:
        model_name (str): The name of the pre-trained model to load.
        input_text (str): The input text for inference.

    Returns:
        dict: A dictionary containing latency and memory usage metrics.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(input_text, return_tensors="pt")

    # Measure latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    latency = time.time() - start_time

    # Measure memory usage
    memory_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    return {
        "latency": latency,
        "memory_allocated": memory_allocated
    }

if __name__ == "__main__":
    # Example usage
    input_text = "The quick brown fox jumps over the lazy dog."
    model_name = "gpt2-medium"
    metrics = benchmark_inference(model_name, input_text)
    print(f"Latency: {metrics['latency']} seconds")
    print(f"Memory Allocated: {metrics['memory_allocated']} bytes")