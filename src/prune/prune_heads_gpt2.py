import torch
from transformers import GPT2Model

def prune_heads(model: GPT2Model, heads_to_prune: dict):
    """
    Prune heads of the GPT-2 model.

    Args:
        model (GPT2Model): The GPT-2 model to prune.
        heads_to_prune (dict): A dictionary where keys are layer indices and values are lists of head indices to prune.
    """
    model.prune_heads(heads_to_prune)

if __name__ == "__main__":
    # Example usage
    model = GPT2Model.from_pretrained("gpt2-medium")
    heads_to_prune = {0: [0, 1], 1: [2, 3]}  # Example: prune heads 0, 1 in layer 0 and heads 2, 3 in layer 1
    prune_heads(model, heads_to_prune)
    print("Heads pruned successfully.")