import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms # Keep this for consistency in preprocessing
import os
import time
import json # For loading the class index JSON

# --- Configuration ---
QUANTIZED_MODEL_PATH = "resnet50_quark_int8.onnx"
IMAGE_DATASET_DIR = "imagenet_val_dataset\ILSVRC2012_img_val" # Path to your validation dataset
# USER_CLASS_INDEX_JSON_PATH will be used as the sole source for class names
USER_CLASS_INDEX_JSON_PATH = "imagenet_val_dataset\ILSVRC2012_GT.json" # Your JSON mapping indices "0"-"999" to names
USER_GT_INDICES_PATH = "imagenet_val_dataset\ILSRVC2012_GT.txt" # Your TXT file with GT indices (assumed 0-999)

IMAGE_SIZE = 224
NUM_TOP_PREDICTIONS = 5
VALID_IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# --- Load Ground Truth and Class Mappings ---
def load_gt_indices_from_txt(filepath):
    """Loads ground truth indices from the user's TXT file."""
    if not os.path.exists(filepath):
        print(f"Error: Ground truth indices file '{filepath}' not found.")
        return None
    try:
        with open(filepath, "r") as f:
            # These are assumed to be 0-indexed keys for USER_CLASS_INDEX_JSON_PATH
            indices = [int(line.strip()) for line in f if line.strip()]
        return indices
    except ValueError:
        print(f"Error: Non-integer value found in ground truth indices file '{filepath}'. Ensure it contains one integer per line.")
        return None
    except Exception as e:
        print(f"Error loading ground truth indices file '{filepath}': {e}")
        return None

def load_class_map_from_json(json_filepath):
    """Loads the JSON mapping from class index (string key "0"-"999") to class name.
       This map will be used for both model output interpretation and ground truth.
    """
    if not os.path.exists(json_filepath):
        print(f"Error: Class index JSON file '{json_filepath}' not found.")
        return None
    try:
        with open(json_filepath, "r") as f:
            data = json.load(f)
        # Convert string keys from JSON (e.g., "0", "1") to integer keys (0, 1)
        # This map will convert an integer index (0-999) to a class name.
        class_map = {int(k): v for k, v in data.items()}
        if not all(isinstance(k, int) and k >= 0 for k in class_map.keys()):
            print(f"Warning: Some keys in '{json_filepath}' are not non-negative integers after conversion.")
        if not len(class_map) > 0 : # Check if map is not empty
             print(f"Warning: The loaded class map from '{json_filepath}' is empty.")
             return None
        # Check if it has 1000 classes, common for ImageNet
        # if len(class_map) != 1000:
        # print(f"Warning: Class map from '{json_filepath}' has {len(class_map)} entries, not the typical 1000 for ImageNet.")
        return class_map
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{json_filepath}'.")
        return None
    except ValueError:
        print(f"Error: Keys in '{json_filepath}' could not be converted to integers.")
        return None
    except Exception as e:
        print(f"Error loading class map from '{json_filepath}': {e}")
        return None

# --- Preprocessing ---
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str):
            if not os.path.exists(image_path_or_pil):
                print(f"Error: Image path '{image_path_or_pil}' not found.")
                return None
            img = Image.open(image_path_or_pil).convert('RGB')
        elif isinstance(image_path_or_pil, Image.Image):
             img = image_path_or_pil.convert('RGB')
        else:
            print("Invalid input to preprocess_image. Expecting path string or PIL Image.")
            return None
        img_tensor = preprocess_transform(img)
        img_numpy = img_tensor.unsqueeze(0).numpy().astype(np.float32)
        return img_numpy
    except Exception as e:
        image_id = image_path_or_pil if isinstance(image_path_or_pil, str) else 'PIL image'
        print(f"Error processing image '{image_id}': {e}")
        return None

# --- Postprocessing ---
def postprocess_output(output_data, class_label_map):
    """
    Postprocesses the model output.
    Args:
        output_data: Raw output from the ONNX model (batch_size, num_classes).
        class_label_map: The map {int_index: "class_name"} used for model's classes.
    Returns:
        A list of (class_name, score) tuples for top N predictions,
        and a list of the top N predicted model indices.
    """
    probabilities = np.exp(output_data[0]) / np.sum(np.exp(output_data[0]))
    # predicted_model_indices are indices from 0 to num_classes-1
    predicted_model_indices = np.argsort(probabilities)[-NUM_TOP_PREDICTIONS:][::-1]
    
    results_named = []
    for i in predicted_model_indices:
        if i in class_label_map:
            results_named.append((class_label_map[i], float(probabilities[i])))
        else:
            # This case should ideally not happen if class_label_map is correct for the model
            results_named.append((f"Unknown_Index_{i}", float(probabilities[i])))
            print(f"Warning: Model predicted index {i} which is not in the provided class_label_map.")
            
    return results_named, predicted_model_indices

# --- Main Inference and Evaluation ---
def run_inference_and_evaluation():
    # 1. Load Model
    if not os.path.exists(QUANTIZED_MODEL_PATH):
        print(f"Error: Quantized model '{QUANTIZED_MODEL_PATH}' not found.")
        return

    # 2. Load Class Mappings and Ground Truth
    # This class_map is assumed to be {model_output_index: "class_name"}
    # AND also {gt_index_from_txt_file: "class_name"}
    class_map = load_class_map_from_json(USER_CLASS_INDEX_JSON_PATH)
    if class_map is None:
        print("Failed to load the class map. Exiting.")
        return

    ground_truth_indices = load_gt_indices_from_txt(USER_GT_INDICES_PATH)
    if ground_truth_indices is None:
        print("Failed to load ground truth indices. Exiting.")
        return

    # 3. Load Image Files
    if not os.path.isdir(IMAGE_DATASET_DIR):
        print(f"Error: Image dataset directory '{IMAGE_DATASET_DIR}' not found.")
        return

    image_files = []
    for root, _, files in os.walk(IMAGE_DATASET_DIR):
        for file in files:
            if file.lower().endswith(VALID_IMAGE_EXTENSIONS):
                image_files.append(os.path.join(root, file))
    
    image_files.sort() # CRITICAL: Sort image files to match ground truth order

    if not image_files:
        print(f"No image files found in '{IMAGE_DATASET_DIR}'.")
        return
    
    print(f"Found {len(image_files)} images in '{IMAGE_DATASET_DIR}'.")
    print(f"Loaded {len(ground_truth_indices)} ground truth indices.")
    print(f"Loaded {len(class_map)} mappings in class JSON.")

    if len(image_files) != len(ground_truth_indices):
        print(f"Warning: Number of image files ({len(image_files)}) does not match "
              f"number of ground truth labels ({len(ground_truth_indices)}).")
        print("Evaluation will proceed, but ensure this is expected. Accuracy will be based on the smaller count.")
    
    num_images_to_evaluate = min(len(image_files), len(ground_truth_indices))
    if num_images_to_evaluate == 0:
        print("No images or ground truth available to evaluate.")
        return
    print(f"Will evaluate {num_images_to_evaluate} images for accuracy.")

    # 4. Initialize ONNX Runtime Session
    print(f"Loading ONNX Runtime session for {QUANTIZED_MODEL_PATH}...")
    providers = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
    try:
        session_options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(QUANTIZED_MODEL_PATH, sess_options=session_options, providers=providers)
        print(f"Using execution provider(s): {session.get_providers()}")
    except Exception as e:
        print(f"Error creating ONNX Runtime session: {e}")
        print("Try with providers=['CPUExecutionProvider'] if VitisAI fails.")
        return
    input_name = session.get_inputs()[0].name
    output_node_shape = session.get_outputs()[0].shape # e.g., [None, 1000]
    
    # Check if model output classes match class_map size (e.g. 1000)
    # This is a heuristic, as shape might be (None, num_classes)
    model_num_classes = None
    if isinstance(output_node_shape, list) and len(output_node_shape) > 1 and isinstance(output_node_shape[1], int):
        model_num_classes = output_node_shape[1]
        if model_num_classes != len(class_map):
            print(f"Warning: Model expects to output {model_num_classes} classes, but class_map has {len(class_map)} entries.")
            print("This might lead to incorrect class name lookups for model predictions.")
    else:
        print(f"Warning: Could not determine number of output classes from model shape: {output_node_shape}. Assuming class_map is correct.")


    # --- Accuracy and Timing Variables ---
    total_inference_time_ms = 0
    evaluated_image_count = 0 # Images for which accuracy was computed
    top1_correct_count = 0
    top5_correct_count = 0
    first_valid_input_data_for_benchmark = None

    # 5. Warm-up (Optional but recommended for stable benchmark)
    warmup_done = False
    for img_idx_warmup in range(num_images_to_evaluate):
        current_image_path_warmup = image_files[img_idx_warmup]
        temp_input_data = preprocess_image(current_image_path_warmup)
        if temp_input_data is not None:
            try:
                print(f"Performing warm-up run on {os.path.basename(current_image_path_warmup)}...")
                _ = session.run(None, {input_name: temp_input_data})
                first_valid_input_data_for_benchmark = temp_input_data # Save for benchmark
                warmup_done = True
                print("Warm-up successful.")
            except Exception as e_warmup:
                print(f"Error during warm-up inference run on {os.path.basename(current_image_path_warmup)}: {e_warmup}")
            break # Only need one successful warm-up image
    if not warmup_done:
         print("Warm-up could not be performed. Timings might be less stable.")


    # 6. Iterate, Predict, and Evaluate
    print(f"\n--- Running inference and evaluation on {num_images_to_evaluate} images ---")
    for img_idx in range(num_images_to_evaluate):
        image_path = image_files[img_idx]
        base_image_name = os.path.basename(image_path)
        print(f"\nProcessing image {img_idx + 1}/{num_images_to_evaluate}: {base_image_name}")

        # Get Ground Truth
        gt_class_idx = ground_truth_indices[img_idx] # This is an integer index, e.g., 490
        
        if gt_class_idx not in class_map:
            print(f"  Warning: Ground truth index {gt_class_idx} for image {base_image_name} not found in class_map. Skipping accuracy for this image.")
            # Attempt to process for timing if possible
            temp_input_data = preprocess_image(image_path)
            if temp_input_data is not None:
                try:
                    start_time = time.perf_counter()
                    session.run(None, {input_name: temp_input_data}) # Run inference
                    end_time = time.perf_counter()
                    total_inference_time_ms += (end_time - start_time) * 1000
                    # Don't increment evaluated_image_count
                except: pass # Ignore errors here, already warned
            continue

        ground_truth_name = class_map[gt_class_idx]
        print(f"  Ground Truth: Index {gt_class_idx} -> Name '{ground_truth_name}'")

        # Preprocess Image
        input_data = preprocess_image(image_path)
        if input_data is None:
            print(f"  Skipping {base_image_name} due to preprocessing error (cannot evaluate accuracy).")
            continue
        
        if first_valid_input_data_for_benchmark is None: # If warm-up failed but this image is good
            first_valid_input_data_for_benchmark = input_data

        # Inference
        try:
            start_time = time.perf_counter()
            outputs = session.run(None, {input_name: input_data})
            end_time = time.perf_counter()
            inference_time_ms = (end_time - start_time) * 1000
            
            # Postprocess
            # class_map is used here to map model's output indices to names
            predicted_results_named, predicted_top_model_indices = postprocess_output(outputs[0], class_map)

            # Accuracy Calculation
            top1_predicted_name = predicted_results_named[0][0] if predicted_results_named else "N/A"
            top1_match = False
            if top1_predicted_name == ground_truth_name:
                top1_correct_count += 1
                top1_match = True

            top5_predicted_names = [res[0] for res in predicted_results_named]
            top5_match = False
            if ground_truth_name in top5_predicted_names:
                top5_correct_count += 1
                top5_match = True
            
            # Update totals for successfully evaluated images
            total_inference_time_ms += inference_time_ms
            evaluated_image_count += 1

            print(f"  Inference took: {inference_time_ms:.2f} ms")
            print("  Top Predictions:")
            for i, (label, score) in enumerate(predicted_results_named):
                highlight = ""
                if label == ground_truth_name: # Check if this prediction matches GT
                    if i == 0: # Top-1 match
                        highlight = " [* TOP-1 MATCH *]"
                    else: # Top-5 match (but not Top-1)
                        highlight = " [* TOP-5 MATCH *]"
                print(f"  - {label}: {score:.4f}{highlight}")
            if not top5_match and ground_truth_name != "N/A": # If GT was valid but not in top 5
                 print(f"  (Ground truth '{ground_truth_name}' not in top {NUM_TOP_PREDICTIONS})")

        except Exception as e:
            print(f"  Error during inference or postprocessing for {image_path}: {e}")
            continue

    # --- Summary and Benchmark ---
    print("\n--- Dataset Inference and Accuracy Summary ---")
    if evaluated_image_count > 0:
        avg_dataset_inference_time_ms = total_inference_time_ms / evaluated_image_count
        top1_accuracy = (top1_correct_count / evaluated_image_count) * 100
        top5_accuracy = (top5_correct_count / evaluated_image_count) * 100

        print(f"Successfully processed and evaluated {evaluated_image_count}/{num_images_to_evaluate} images with valid ground truth.")
        print(f"Total inference time for these images: {total_inference_time_ms:.2f} ms")
        print(f"Average inference time per image: {avg_dataset_inference_time_ms:.2f} ms")
        if avg_dataset_inference_time_ms > 0:
            avg_fps_dataset = 1000 / avg_dataset_inference_time_ms
            print(f"Average FPS (dataset, single inference per image): {avg_fps_dataset:.2f}")
        
        print(f"\nTop-1 Accuracy: {top1_correct_count}/{evaluated_image_count} = {top1_accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_correct_count}/{evaluated_image_count} = {top5_accuracy:.2f}%")
    else:
        print("No images were successfully processed and evaluated for accuracy.")

    # Benchmark (using first_valid_input_data_for_benchmark)
    if first_valid_input_data_for_benchmark is not None:
        num_runs_benchmark = 20
        print(f"\n--- Running benchmark ({num_runs_benchmark} runs on one preprocessed image) ---")
        timings_benchmark = []
        try:
            # Additional warm-up specifically for the benchmark input data
            for _ in range(min(5, num_runs_benchmark // 2)):
                 session.run(None, {input_name: first_valid_input_data_for_benchmark})

            for _ in range(num_runs_benchmark):
                iter_start_time = time.perf_counter()
                session.run(None, {input_name: first_valid_input_data_for_benchmark})
                iter_end_time = time.perf_counter()
                timings_benchmark.append(iter_end_time - iter_start_time)
            
            if timings_benchmark: # Ensure list is not empty
                avg_time_ms_bench = (sum(timings_benchmark) / len(timings_benchmark)) * 1000
                print(f"Average inference time (single image benchmark): {avg_time_ms_bench:.2f} ms")
                if avg_time_ms_bench > 0:
                    fps_bench = 1000 / avg_time_ms_bench
                    print(f"Average FPS (single image benchmark): {fps_bench:.2f}")
            else:
                print("No benchmark timings recorded.")
        except Exception as e_bench:
            print(f"Error during benchmark runs: {e_bench}")
    else:
        print("\nSkipping benchmark as no image was successfully preprocessed for it.")

if __name__ == "__main__":
    run_inference_and_evaluation()
    print("\nInference and evaluation script finished.")