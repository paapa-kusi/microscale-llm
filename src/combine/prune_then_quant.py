from transformers import GPT2Model
from src.prune.prune_heads_gpt2 import prune_heads
from src.quant.quant_bnb_8bit_4bit import quantize_model

def prune_and_quantize(model_name: str, heads_to_prune: dict, quantization_type: str):
    """
    Apply pruning and quantization to a model.

    Args:
        model_name (str): The name of the pre-trained model to load.
        heads_to_prune (dict): A dictionary where keys are layer indices and values are lists of head indices to prune.
        quantization_type (str): The type of quantization ('int8' or 'int4').

    Returns:
        GPT2Model: The pruned and quantized model.
    """
    # Load the model
    model = GPT2Model.from_pretrained(model_name)

    # Apply pruning
    prune_heads(model, heads_to_prune)

    # Apply quantization
    quantize_model(model, quantization_type)

    return model

if __name__ == "__main__":
    # Example usage
    model_name = "gpt2-medium"
    heads_to_prune = {0: [0, 1], 1: [2, 3]}  # Example: prune heads 0, 1 in layer 0 and heads 2, 3 in layer 1
    quantization_type = "int8"

    model = prune_and_quantize(model_name, heads_to_prune, quantization_type)
    print("Model pruned and quantized successfully.")