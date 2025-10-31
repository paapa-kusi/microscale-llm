import torch
from transformers import GPT2Model
from bitsandbytes.nn import Int8Params, Int4Params

def quantize_model(model: GPT2Model, quantization_type: str):
    """
    Apply quantization to the GPT-2 model.

    Args:
        model (GPT2Model): The GPT-2 model to quantize.
        quantization_type (str): The type of quantization ('int8' or 'int4').
    """
    if quantization_type == 'int8':
        for name, param in model.named_parameters():
            if param.requires_grad:
                model._parameters[name] = Int8Params(param.data)
    elif quantization_type == 'int4':
        for name, param in model.named_parameters():
            if param.requires_grad:
                model._parameters[name] = Int4Params(param.data)
    else:
        raise ValueError("Unsupported quantization type. Use 'int8' or 'int4'.")

if __name__ == "__main__":
    # Example usage
    model = GPT2Model.from_pretrained("gpt2-medium")
    quantize_model(model, quantization_type='int8')  # Apply INT8 quantization
    print("Model quantized successfully.")