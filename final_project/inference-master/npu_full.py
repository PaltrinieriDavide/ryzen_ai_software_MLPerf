import os
import json
import time
import random
import numpy as np
import statistics
from pathlib import Path
from PIL import Image
import onnxruntime as ort
# import torch # Not strictly needed if torchvision.transforms is used directly
# from torch.utils.data import Dataset # Not strictly needed for ImagenetDataset structure
from torchvision import transforms

from mlperf_loadgen import (
    TestSettings, TestScenario, TestMode,
    QuerySample, QuerySampleResponse,
    StartTest, QuerySamplesComplete,
    ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
    LogSettings, LoggingMode
)

# === Configuration ===
RUN_ACCURACY = True
RUN_PERFORMANCE = True
# BATCH_SIZE is less relevant for inference step now, but MLPerf might still group query_samples
# based on QSL behavior or internal batching. The core inference will be 1-by-1.
BATCH_SIZE = 1 # Kept for consistency, but session.run will be on single images
NUM_IMAGES = 10000  # Smaller subset for quicker testing, set to None for all
MIN_DURATION_MS = 1  # 5 seconds
MIN_QUERY_COUNT = 1 # Ensure enough samples for meaningful performance

# === Paths (Ensure these are correct for your system) ===
# Absolute paths are generally safer for scripts like this
base_dir = Path("C:\\Users\\User02\\Desktop\\paltrinieri\\inference-master")
image_dir = Path("C:\\Users\\User02\\Desktop\\paltrinieri\\imagenet_val_dataset\\ILSVRC2012_img_val")
map_file = Path("C:\\Users\\User02\\Desktop\\paltrinieri\\imagenet_val_dataset\\val_map.txt")
results_dir_base = Path("C:\\Users\\User02\\Desktop\\paltrinieri\\inference-master\\results_mlperf_mod") # New results dir
onnx_model_path = Path("C:\\Users\\User02\\Desktop\\paltrinieri\\resnet50_quark_int8.onnx")

# Create a unique results directory for this run
timestamp = time.strftime("%Y%m%d-%H%M%S")
results_dir = results_dir_base / f"run_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

original_working_dir = os.getcwd()
os.chdir(results_dir)
print(f"Changed working directory to: {os.getcwd()}")

# === Load val_map.txt and select subset ===
if not map_file.exists():
    print(f"Error: Map file not found at {map_file}")
    exit()
with open(map_file) as f:
    entries = [line.strip().split() for line in f if line.strip()]

full_dataset_size = len(entries)

if NUM_IMAGES is not None and NUM_IMAGES > 0 and NUM_IMAGES < len(entries):
    print(f"Using subset of {NUM_IMAGES} images (from {len(entries)} total)")
    random.seed(42) # for reproducibility
    entries = random.sample(entries, NUM_IMAGES)
elif NUM_IMAGES is None or NUM_IMAGES == 0 :
    print(f"Using all {len(entries)} images.")
else: # NUM_IMAGES >= len(entries)
    print(f"NUM_IMAGES ({NUM_IMAGES}) is >= dataset size ({len(entries)}). Using all {len(entries)} images.")


if not entries:
    print("Error: No entries loaded from map file or subset is empty.")
    exit()

image_paths = [image_dir / e[0] for e in entries]
ground_truth = [int(e[1]) for e in entries] # These are 0-999 indices
sample_indices = list(range(len(image_paths))) # Indices for MLPerf QSL

print(f"Dataset size: {len(image_paths)} images")

# === Preprocessing (Consistent with Code 1) ===
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Dataset Class ===
class ImagenetDataset:
    def __init__(self, image_paths_list, labels_list, transform_func):
        self.image_paths = image_paths_list
        self.labels = labels_list
        self.transform = transform_func
        self.cache = {} # To store preprocessed images

    def __len__(self):
        return len(self.image_paths)

    def load_samples(self, indices_to_load):
        # print(f"Loading {len(indices_to_load)} samples into RAM cache...")
        for idx in indices_to_load:
            if idx not in self.cache:
                try:
                    img_path = self.image_paths[idx]
                    if not os.path.exists(img_path):
                        print(f"Warning: Image path {img_path} for index {idx} not found. Skipping.")
                        self.cache[idx] = (None, -1) # Mark as problematic
                        continue
                    img = Image.open(img_path).convert("RGB")
                    tensor = self.transform(img).unsqueeze(0).numpy().astype(np.float32) # Add batch dim
                    self.cache[idx] = (tensor, self.labels[idx])
                except Exception as e:
                    print(f"Error loading/processing image {self.image_paths[idx]} (index {idx}): {e}")
                    self.cache[idx] = (None, -1) # Mark as problematic
        # print(f"Sample loading complete. Cache size: {len(self.cache)}")

    def unload_samples(self, indices_to_unload):
        for idx in indices_to_unload:
            self.cache.pop(idx, None)

    def get_sample(self, idx):
        if idx not in self.cache:
            # This should ideally not happen if load_samples was called by MLPerf
            print(f"Warning: Sample {idx} not in cache. Attempting to load it now.")
            self.load_samples([idx])
        
        sample_data = self.cache.get(idx)
        if sample_data is None or sample_data[0] is None:
            print(f"Error: Could not retrieve valid data for sample {idx}. Returning dummy data.")
            # Return a dummy tensor and label to prevent crashes, though this sample will be wrong
            dummy_tensor = np.zeros((1, 3, 224, 224), dtype=np.float32)
            dummy_label = -1
            return dummy_tensor, dummy_label
        return sample_data


dataset = ImagenetDataset(image_paths, ground_truth, preprocess_transform)

# === ONNX Runtime ===
available_providers = ort.get_available_providers()
print(f"Available ONNX providers: {available_providers}")

# Choose provider: VitisAIExecutionProvider or DmlExecutionProvider or CPUExecutionProvider
# providers_list = ['VitisAIExecutionProvider', 'CPUExecutionProvider']
providers_list = ['VitisAIExecutionProvider'] # As in Code 1
# providers_list = ['DmlExecutionProvider', 'CPUExecutionProvider'] # For GPU on Windows
# providers_list = ['CPUExecutionProvider'] # For CPU

print(f"Loading ONNX model: {onnx_model_path}")
if not onnx_model_path.exists():
    print(f"Error: ONNX model not found at {onnx_model_path}")
    exit()

try:
    session_options = ort.SessionOptions()
    # Example of Vitis AI specific config if needed (refer to Vitis AI EP documentation)
    # session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # session_options.intra_op_num_threads = 1
    # session_options.inter_op_num_threads = 1
    # provider_options = [{"config_file": "vaip_config.json"}] # If you have a vaip_config.json

    session = ort.InferenceSession(
        str(onnx_model_path),
        sess_options=session_options,
        providers=providers_list
        # provider_options=provider_options # if using Vitis AI specific options
    )
    print(f"Using ONNX execution provider(s): {session.get_providers()}")
except Exception as e:
    print(f"Error creating ONNX Runtime session: {e}")
    print("Ensure the chosen Execution Provider is available and configured correctly.")
    exit()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"Model loaded with input name: '{input_name}', output name: '{output_name}'")

# === Performance Metrics Collection & Inference Logic ===
predictions = {} # Stores {sample_idx: {"top1": pred_idx, "top5": [pred_idx, ...]}}
latencies = []   # Stores individual inference times (ms) for each sample
total_samples_processed_by_sut = 0
sut_start_time = None
sut_end_time = None

def issue_queries(query_samples):
    global total_samples_processed_by_sut, sut_start_time, sut_end_time

    if sut_start_time is None: # Mark start time of SUT processing
        sut_start_time = time.perf_counter()

    for qs in query_samples: # Process one QuerySample at a time
        sample_idx = qs.index
        
        input_tensor, _ = dataset.get_sample(sample_idx)

        if input_tensor is None:
            print(f"  Skipping sample {sample_idx} due to preprocessing error.")
            # We still need to respond to MLPerf for this query
            # Create a dummy response or handle error appropriately
            response = QuerySampleResponse(qs.id, 0, 0) # 0 for error or skip
            QuerySamplesComplete([response])
            continue

        # Perform inference for the single image
        try:
            individual_infer_start_time = time.perf_counter()
            # input_tensor from dataset.get_sample is already (1, C, H, W)
            model_outputs = session.run([output_name], {input_name: input_tensor})
            individual_infer_end_time = time.perf_counter()
            
            inference_time_ms = (individual_infer_end_time - individual_infer_start_time) * 1000
            latencies.append(inference_time_ms)

            # Postprocess output for this single image
            # model_outputs[0] is the output tensor, shape (1, num_classes) for a single image input
            output_scores = model_outputs[0][0] # Scores for the single image in the batch of 1

            # Get top-1 and top-5 predicted indices
            # No explicit softmax needed if only indices are required, as argmax/argsort order is preserved
            top1_pred_idx = int(np.argmax(output_scores))
            top5_pred_indices = [int(i) for i in np.argsort(output_scores)[-5:][::-1]]

            predictions[sample_idx] = {
                "top1": top1_pred_idx,
                "top5": top5_pred_indices
            }
            
            response_data_size = output_scores.nbytes # Example, can be 0 if not returning data
            response = QuerySampleResponse(qs.id, 0, response_data_size) # Using 0 for uintptr_t data

        except Exception as e:
            print(f"  Error during inference or postprocessing for sample {sample_idx} (image: {os.path.basename(image_paths[sample_idx])}): {e}")
            # Mark as error, perhaps store a specific error code or empty prediction
            predictions[sample_idx] = {"top1": -1, "top5": []} # Indicate error
            response = QuerySampleResponse(qs.id, 0, 0) # Error response

        QuerySamplesComplete([response])
        total_samples_processed_by_sut += 1

        if total_samples_processed_by_sut % 100 == 0:
             print(f"SUT processed {total_samples_processed_by_sut} samples...", end="\r")

    # Mark end time if all samples expected by QSL are processed by SUT
    # This is a bit tricky as issue_queries can be called multiple times.
    # We'll use MLPerf's own timing primarily for official reports.
    # sut_end_time can be set in flush_queries or after StartTest.
    # For now, this provides a rough SUT processing duration.
    if total_samples_processed_by_sut >= len(sample_indices): # Assuming all samples are issued
        if sut_end_time is None:
            sut_end_time = time.perf_counter()


def flush_queries():
    global sut_end_time
    # This function is called by MLPerf to signal that no more queries will be issued.
    # It's a good place to finalize any SUT-specific timing.
    if sut_end_time is None and sut_start_time is not None:
        sut_end_time = time.perf_counter()
    print("SUT: All queries flushed.")

def load_samples_to_ram_wrapper(indices_to_load):
    dataset.load_samples(indices_to_load)

def unload_samples_from_ram_wrapper(indices_to_unload):
    dataset.unload_samples(indices_to_unload)

def run_mlperf_test(mode_is_accuracy):
    """Runs MLPerf test in either AccuracyOnly or PerformanceOnly mode."""
    global predictions, latencies, total_samples_processed_by_sut
    global sut_start_time, sut_end_time

    # Reset metrics for each run
    predictions = {}
    latencies = []
    total_samples_processed_by_sut = 0
    sut_start_time = None
    sut_end_time = None
    
    test_mode_str = "ACCURACY" if mode_is_accuracy else "PERFORMANCE"
    print(f"\n--- Running MLPerf {test_mode_str} test ---")

    log_settings = LogSettings()
    log_settings.log_output.outdir = str(results_dir) # MLPerf logs to current dir if not set like this
    log_settings.log_output.prefix = "mlperf_log_"
    log_settings.log_output.copy_summary_to_stdout = True
    # log_settings.log_output.copy_detail_to_stdout = False # Can be verbose
    
    # Check for ASYNC_WRITE_BACK, otherwise fallback
    if hasattr(LoggingMode, 'ASYNC_WRITE_BACK'):
        log_settings.log_mode = LoggingMode.ASYNC_WRITE_BACK
    elif hasattr(LoggingMode, 'AsyncPoll'):
        print("Warning: LoggingMode.ASYNC_WRITE_BACK not found, using LoggingMode.AsyncPoll.")
        log_settings.log_mode = LoggingMode.AsyncPoll
    else:
        print("Warning: LoggingMode.ASYNC_WRITE_BACK and AsyncPoll not found, using default log mode (could be SYNC).")
        # If neither is available, it will use the default or you might pick another available one.
        # For example, if SYNC is an option: log_settings.log_mode = LoggingMode.SYNC

    settings = TestSettings()
    settings.scenario = TestScenario.Offline
    settings.mode = TestMode.AccuracyOnly if mode_is_accuracy else TestMode.PerformanceOnly
    
    num_qsl_samples = len(sample_indices)

    if mode_is_accuracy:
        settings.min_query_count = num_qsl_samples 
        settings.min_duration_ms = 0 
    else: # Performance
        settings.min_query_count = MIN_QUERY_COUNT
        settings.min_duration_ms = MIN_DURATION_MS
    
    settings.offline_expected_qps = 100000 

    sut = ConstructSUT(issue_queries, flush_queries)
    qsl = ConstructQSL(
        count=num_qsl_samples,
        performance_sample_count=min(num_qsl_samples, 50000),
        load_fn=load_samples_to_ram_wrapper,
        unload_fn=unload_samples_from_ram_wrapper
    )
    
    print(f"Starting MLPerf test (Mode: {settings.mode}, Scenario: {settings.scenario})...")
    print(f"QSL configured with {num_qsl_samples} samples.")
    if not mode_is_accuracy:
        print(f"Performance test constraints: min_query_count={settings.min_query_count}, min_duration_ms={settings.min_duration_ms}")

    StartTest(sut, qsl, settings, log_settings)
    print(f"MLPerf {test_mode_str} test finished.")

    if sut_end_time is None and sut_start_time is not None:
        sut_end_time = time.perf_counter()

    results_summary = {}
    if mode_is_accuracy:
        results_summary = calculate_accuracy_metrics()
    else:
        results_summary = calculate_performance_metrics()

    DestroyQSL(qsl)
    DestroySUT(sut)
    return results_summary

def calculate_accuracy_metrics():
    if not predictions:
        print("No predictions collected for accuracy calculation.")
        return {"top1_accuracy": 0, "top5_accuracy": 0, "samples_evaluated": 0}
        
    top1_correct = 0
    top5_correct = 0
    num_evaluated = 0

    for i, pred_data in predictions.items():
        if i >= len(ground_truth): 
            print(f"Warning: Prediction index {i} out of bounds for ground_truth.")
            continue
        
        gt_idx = ground_truth[i]
        if pred_data.get("top1", -1) == -1 : 
            # print(f"Skipping accuracy for sample {i} due to prediction error.") # Can be verbose
            continue 

        num_evaluated +=1
        if pred_data["top1"] == gt_idx:
            top1_correct += 1
        if gt_idx in pred_data["top5"]:
            top5_correct += 1
    
    top1_acc = (top1_correct / num_evaluated * 100) if num_evaluated > 0 else 0
    top5_acc = (top5_correct / num_evaluated * 100) if num_evaluated > 0 else 0

    print("\n--- Accuracy Summary ---")
    print(f"Total samples with predictions: {len(predictions)}")
    print(f"Samples evaluated for accuracy (no errors): {num_evaluated}")
    print(f"Top-1 Correct: {top1_correct}, Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Correct: {top5_correct}, Top-5 Accuracy: {top5_acc:.2f}%")

    with open(results_dir / "mlperf_log_accuracy.json", "w") as f:
        json.dump({"top1_accuracy_percent": top1_acc, "top5_accuracy_percent": top5_acc, "samples_evaluated": num_evaluated}, f, indent=4)
    
    print(f"Accuracy summary saved to: {results_dir / 'mlperf_log_accuracy.json'}")
    return {"top1_accuracy": top1_acc, "top5_accuracy": top5_acc, "samples_evaluated": num_evaluated}


def calculate_performance_metrics():
    print("\n--- Performance Summary (SUT Internal) ---")
    if not latencies:
        print("No SUT latency data collected.")
        return {}

    sut_duration_sec = (sut_end_time - sut_start_time) if sut_start_time and sut_end_time else 0
    
    avg_latency_ms = statistics.mean(latencies) if latencies else 0
    median_latency_ms = statistics.median(latencies) if latencies else 0
    p90_latency_ms = np.percentile(latencies, 90) if latencies else 0
    min_latency_ms = min(latencies) if latencies else 0
    max_latency_ms = max(latencies) if latencies else 0
    
    sut_throughput_qps = total_samples_processed_by_sut / sut_duration_sec if sut_duration_sec > 0 else 0

    print(f"Total samples processed by SUT: {total_samples_processed_by_sut}")
    print(f"SUT processing duration: {sut_duration_sec:.2f} seconds")
    print(f"SUT estimated throughput: {sut_throughput_qps:.2f} QPS (samples/sec)")
    print(f"Latency (based on individual session.run calls):")
    print(f"  Average: {avg_latency_ms:.2f} ms")
    print(f"  Median: {median_latency_ms:.2f} ms")
    print(f"  Min: {min_latency_ms:.2f} ms")
    print(f"  Max: {max_latency_ms:.2f} ms")
    print(f"  90th Percentile: {p90_latency_ms:.2f} ms")
    
    sut_perf_summary = {
        "total_samples_processed_by_sut": total_samples_processed_by_sut,
        "sut_processing_duration_sec": sut_duration_sec,
        "sut_throughput_qps": sut_throughput_qps,
        "avg_individual_latency_ms": avg_latency_ms,
        "median_individual_latency_ms": median_latency_ms,
        "p90_individual_latency_ms": p90_latency_ms,
        "min_individual_latency_ms": min_latency_ms,
        "max_individual_latency_ms": max_latency_ms,
        "num_latency_samples": len(latencies)
    }
    with open(results_dir / "sut_performance_summary.json", "w") as f:
        json.dump(sut_perf_summary, f, indent=4)
    print(f"SUT internal performance summary saved to: {results_dir / 'sut_performance_summary.json'}")
    return sut_perf_summary


# === Main Execution ===
if __name__ == "__main__":
    overall_results = {
        "run_timestamp": timestamp,
        "model_path": str(onnx_model_path),
        "dataset_image_dir": str(image_dir),
        "dataset_map_file": str(map_file),
        "num_total_images_in_dataset": full_dataset_size,
        "num_images_used_for_test": len(sample_indices),
        "mlperf_CONFIG": {
            "NUM_IMAGES": NUM_IMAGES,
            "MIN_DURATION_MS": MIN_DURATION_MS,
            "MIN_QUERY_COUNT": MIN_QUERY_COUNT,
            "BATCH_SIZE_config": BATCH_SIZE 
        }
    }

    if len(sample_indices) > 0 :
        print("\n--- Performing SUT Warm-up ---")
        warmup_samples_count = min(10, len(sample_indices)) 
        warmup_indices = sample_indices[:warmup_samples_count]
        dataset.load_samples(warmup_indices) 
        
        dummy_query_samples_warmup = []
        for i in range(warmup_samples_count):
            sample_id_for_warmup = i 
            actual_sample_index = warmup_indices[i] 
            dummy_query_samples_warmup.append(QuerySample(sample_id_for_warmup, actual_sample_index)) # CORRECTED CONSTRUCTOR

        print(f"Warm-up with {len(dummy_query_samples_warmup)} samples...")
        
        _predictions_backup = predictions.copy()
        _latencies_backup = latencies.copy()
        _total_backup = total_samples_processed_by_sut
        _sut_start_backup, _sut_end_backup = sut_start_time, sut_end_time

        predictions, latencies, total_samples_processed_by_sut = {}, [], 0
        sut_start_time, sut_end_time = None, None
        
        issue_queries(dummy_query_samples_warmup) 

        predictions = _predictions_backup
        latencies = _latencies_backup
        total_samples_processed_by_sut = _total_backup
        sut_start_time, sut_end_time = _sut_start_backup, _sut_end_backup
        
        dataset.unload_samples(warmup_indices) 
        print("Warm-up complete.")
    else:
        print("Skipping warm-up as no samples are available.")


    if RUN_ACCURACY:
        accuracy_results = run_mlperf_test(mode_is_accuracy=True)
        overall_results["accuracy_summary"] = accuracy_results
        
    if RUN_PERFORMANCE:
        performance_results = run_mlperf_test(mode_is_accuracy=False)
        overall_results["performance_summary_sut_internal"] = performance_results 

    print("\n--- MLPerf Inference Test Suite Finished ---")
    print(f"All results and logs saved in: {results_dir.resolve()}")

    mlperf_summary_file = results_dir / "mlperf_log_summary.txt"
    if mlperf_summary_file.exists():
        print(f"\nOfficial MLPerf summary log: {mlperf_summary_file}")
    else:
        print(f"\nWarning: MLPerf summary log (mlperf_log_summary.txt) not found in {results_dir}")
        print("This might indicate an issue with the MLPerf LoadGen run or logging setup.")

    with open(results_dir / "overall_run_summary.json", "w") as f:
        json.dump(overall_results, f, indent=4)
    print(f"Overall run configuration and SUT summary saved to: {results_dir / 'overall_run_summary.json'}")

    os.chdir(original_working_dir)
    print(f"Restored working directory to: {os.getcwd()}")