import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms # Keep this for consistency in preprocessing
import os
import time
import requests # For downloading sample image

# --- Configuration ---
QUANTIZED_MODEL_PATH = "resnet50_fp32.onnx" # Model from Step 3
IMAGE_PATH = "sample_image.jpg"  # Replace with your test image or let it download
LABELS_PATH = "imagenet_classes.txt" # Make sure this file is in your project dir
IMAGE_SIZE = 224
NUM_TOP_PREDICTIONS = 5

# --- Load Labels ---
def load_labels(labels_path):
    if not os.path.exists(labels_path):
        print(f"Error: Labels file '{labels_path}' not found.")
        return None
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

# --- Preprocessing ---
# MUST BE IDENTICAL to the preprocessing used for calibration (and FP32 model export)
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str): # if it's a path
            if not os.path.exists(image_path_or_pil):
                print(f"Error: Image path '{image_path_or_pil}' not found.")
                return None
            img = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image): # if it's already a PIL image
             img = image_path_or_pil.convert('RGB')
        else:
            print("Invalid input to preprocess_image. Expecting path string or PIL Image.")
            return None

        img_tensor = preprocess_transform(img)
        # Add batch dimension and convert to numpy float32 (ONNX Runtime expects this)
        img_numpy = img_tensor.unsqueeze(0).numpy().astype(np.float32)
        return img_numpy
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# --- Postprocessing ---
def postprocess_output(output_data, labels):
    # Output is (batch_size, num_classes). We have batch_size=1.
    # Apply softmax to get probabilities
    probabilities = np.exp(output_data[0]) / np.sum(np.exp(output_data[0]))
    # Get top N predictions
    top_indices = np.argsort(probabilities)[-NUM_TOP_PREDICTIONS:][::-1]
    results = [(labels[i], float(probabilities[i])) for i in top_indices] # Ensure score is float
    return results

# --- Main Inference ---
def run_inference():
    if not os.path.exists(QUANTIZED_MODEL_PATH):
        print(f"Error: Quantized model '{QUANTIZED_MODEL_PATH}' not found.")
        print("Please run the quantization step (Step 3) first.")
        return

    labels = load_labels(LABELS_PATH)
    if labels is None: return

    # Download a sample image if it doesn't exist
    if not os.path.exists(IMAGE_PATH):
        print(f"'{IMAGE_PATH}' not found. Attempting to download a sample cat image...")
        try:
            url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
            # It's good practice to set a user-agent
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, stream=True, headers=headers, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes
            pil_image = Image.open(response.raw)
            pil_image.save(IMAGE_PATH) # Save it for future runs
            print(f"Sample image downloaded and saved as {IMAGE_PATH}")
            input_data = preprocess_image(pil_image) # Preprocess the downloaded PIL image
        except Exception as e:
            print(f"Could not download or process sample image: {e}")
            print(f"Please place an image at '{IMAGE_PATH}' manually.")
            return
    else:
        input_data = preprocess_image(IMAGE_PATH) # Preprocess existing image

    if input_data is None:
        print("Failed to preprocess image. Exiting.")
        return

    print(f"Loading ONNX Runtime session for {QUANTIZED_MODEL_PATH}...")
    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available ONNX Runtime providers: {onnxruntime.get_available_providers()}")

    # For Ryzen AI NPU with onnxruntime-amd, 'CPUExecutionProvider' is typically used.
    # The onnxruntime-amd library handles the offload to NPU if available and model is compatible.
    # You might see VitisAIExecutionProvider listed if the full Vitis AI stack is involved,
    # but for Ryzen AI client, CPUExecutionProvider with onnxruntime-amd is common.
    providers = ['VitisAIExecutionProvider']
    # providers = onnxruntime.get_available_providers() # Or let ORT decide (might be safer)
    # To be specific for Ryzen AI NPU, you might need to check documentation for provider options,
    # but often just using CPUExecutionProvider with onnxruntime-amd is sufficient.
    # Forcing VitisAIExecutionProvider might be needed if CPUExecutionProvider isn't offloading.
    # if 'VitisAIExecutionProvider' in onnxruntime.get_available_providers():
    #    providers = [('VitisAIExecutionProvider', {'config_file': 'vaip_config.json'}), 'CPUExecutionProvider'] # vaip_config.json might be needed
    # else:
    #    providers = ['CPUExecutionProvider']

    try:
        session_options = onnxruntime.SessionOptions()
        # session_options.log_severity_level = 0 # More verbose logging
        session = onnxruntime.InferenceSession(QUANTIZED_MODEL_PATH, sess_options=session_options, providers=providers)
        print(f"Using execution provider: {session.get_providers()}")
    except Exception as e:
        print(f"Error creating ONNX Runtime session: {e}")
        print("Ensure 'onnxruntime-amd' is correctly installed and your model is valid.")
        print("Also check if any NPU drivers or Ryzen AI SDK components are missing/corrupted.")
        return

    input_name = session.get_inputs()[0].name
    print(f"Model Input Name: {input_name}")

    print(f"\nRunning inference on '{IMAGE_PATH}'...")
    # Warm-up run (optional, but good for timing)
    try:
        _ = session.run(None, {input_name: input_data})
    except Exception as e:
        print(f"Error during warm-up inference run: {e}")
        return

    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: input_data}) # None for output_names gets all outputs
    end_time = time.perf_counter()

    print(f"Inference took: {(end_time - start_time) * 1000:.2f} ms")

    output_data = outputs[0] # ResNet50 has one output
    results = postprocess_output(output_data, labels)

    print("\nTop Predictions:")
    for label, score in results:
        print(f"- {label}: {score:.4f}")

    # Optional: Benchmark
    num_runs = 20
    print(f"\nRunning benchmark ({num_runs} runs)...")
    timings = []
    for i in range(num_runs):
        iter_start_time = time.perf_counter()
        session.run(None, {input_name: input_data})
        iter_end_time = time.perf_counter()
        timings.append(iter_end_time - iter_start_time)
        # print(f"Run {i+1}/{num_runs}: {(iter_end_time - iter_start_time)*1000:.2f} ms")
    
    avg_time_ms = (sum(timings) / num_runs) * 1000
    print(f"Average inference time over {num_runs} runs: {avg_time_ms:.2f} ms")
    # Calculate FPS
    if avg_time_ms > 0:
        fps = 1000 / avg_time_ms
        print(f"Average FPS: {fps:.2f}")


if __name__ == "__main__":
    run_inference()
    print("Inference script finished.")