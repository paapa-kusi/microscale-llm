import torch
import onnx
from transformers import GPT2Model
import tensorflow as tf
import coremltools as ct

def export_to_coreml(model: GPT2Model, output_path: str):
    """
    Export a PyTorch model to CoreML format.

    Args:
        model (GPT2Model): The PyTorch model to export.
        output_path (str): The path to save the CoreML model.
    """
    model.eval()
    example_input = torch.rand(1, 128)  # Example input for tracing
    traced_model = torch.jit.trace(model, example_input)
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(shape=example_input.shape)],
    )
    coreml_model.save(output_path)
    print(f"Model exported to CoreML format at {output_path}")

def export_to_tflite(model: GPT2Model, output_path: str):
    """
    Export a PyTorch model to TensorFlow Lite format.

    Args:
        model (GPT2Model): The PyTorch model to export.
        output_path (str): The path to save the TensorFlow Lite model.
    """
    model.eval()
    example_input = torch.rand(1, 128)  # Example input for tracing
    traced_model = torch.jit.trace(model, example_input)

    # Convert to ONNX
    onnx_path = "temp_model.onnx"
    torch.onnx.export(traced_model, example_input, onnx_path, input_names=['input'], output_names=['output'])

    # Convert ONNX to TensorFlow
    from onnx_tf.backend import prepare
    tf_rep = prepare(onnx.load(onnx_path))
    tf_rep.export_graph("temp_model.pb")

    # Convert TensorFlow to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_model.pb")
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model exported to TensorFlow Lite format at {output_path}")

if __name__ == "__main__":
    model = GPT2Model.from_pretrained("gpt2-medium")
    export_to_coreml(model, "gpt2_medium.mlmodel")
    export_to_tflite(model, "gpt2_medium.tflite")