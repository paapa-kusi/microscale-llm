import torch
import torch.nn.utils.prune as prune
from transformers import GPT2Model

def prune_weights_unstructured(model: GPT2Model, amount: float):
    """
    Apply unstructured weight pruning to the GPT-2 model.

    Args:
        model (GPT2Model): The GPT-2 model to prune.
        amount (float): The proportion of weights to prune (e.g., 0.5 for 50%).
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

if __name__ == "__main__":
    model = GPT2Model.from_pretrained("gpt2-medium")
    prune_weights_unstructured(model, amount=0.5)  # Prune 50% of weights
    print("Unstructured pruning applied successfully.")