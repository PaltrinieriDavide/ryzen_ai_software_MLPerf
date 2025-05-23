import torch
import torchvision.models as models
import os

# --- Configuration ---
MODEL_NAME = "resnet50"
ONNX_FP32_MODEL_PATH = f"{MODEL_NAME}_fp32.onnx"
OPSET_VERSION = 13 # Ryzen AI typically supports Opset 11, 13 or higher. Check latest docs.

# --- Main Script ---
def export_model():
    print(f"Loading pre-trained {MODEL_NAME} model...")
    # Load pre-trained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor with the expected dimensions
    # ResNet50 typically takes (batch_size, channels, height, width)
    # ImageNet images are usually 224x224
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"Exporting {MODEL_NAME} to ONNX format at {ONNX_FP32_MODEL_PATH}...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_FP32_MODEL_PATH,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=['input'],   # Name for the input node
        output_names=['output'], # Name for the output node
        dynamic_axes={'input': {0: 'batch_size'},  # Allow variable batch size
                      'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {ONNX_FP32_MODEL_PATH}")

if __name__ == "__main__":
    if os.path.exists(ONNX_FP32_MODEL_PATH):
        print(f"{ONNX_FP32_MODEL_PATH} already exists. Skipping export.")
    else:
        export_model()