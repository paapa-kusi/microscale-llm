import os, time, json, psutil, torch
from contextlib import contextmanager
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def mem_gb():
    vmem = psutil.virtual_memory()
    return dict(total_gb=vmem.total/1e9, used_gb=vmem.used/1e9, avail_gb=vmem.available/1e9)

@contextmanager
def timer(name):
    t0=time.time(); yield; dt=time.time()-t0
    print(json.dumps({"timer_s":{name:round(dt,3)}}))

def evaluate_perplexity(model_name: str, dataset: list):
    """
    Evaluate the perplexity of a language model on a given dataset.

    Args:
        model_name (str): The name of the pre-trained model to load.
        dataset (list): A list of text samples to evaluate.

    Returns:
        float: The average perplexity over the dataset.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    total_loss = 0
    total_words = 0

    for text in dataset:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_words += inputs["input_ids"].size(1)

    return torch.exp(torch.tensor(total_loss / total_words))

if __name__ == "__main__":
    # Example usage
    dataset = ["The quick brown fox jumps over the lazy dog.", "Language models are fascinating."]
    model_name = "gpt2-medium"
    perplexity = evaluate_perplexity(model_name, dataset)
    print(f"Average Perplexity: {perplexity}")
