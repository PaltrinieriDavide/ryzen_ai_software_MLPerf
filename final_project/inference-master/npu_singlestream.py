import os
import json
import time
import random
import numpy as np
import statistics
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import torch
from torch.utils.data import Dataset
from torchvision import transforms


from mlperf_loadgen import (
    TestSettings, TestScenario, TestMode,
    QuerySample, QuerySampleResponse,
    StartTest, QuerySamplesComplete,
    ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
    LogSettings, LoggingMode
)
"""
inside the loadgen directory to export the modules
pip install .
"""

RUN_PERFORMANCE = True
BATCH_SIZE = 1
NUM_IMAGES = 10000  # None = use all available images

image_dir = Path("imagenet_val_dataset\\ILSVRC2012_img_val")
map_file = Path("imagenet_val_dataset\\val_map.txt")
results_dir = Path("results_single_stream")
onnx_model_path = Path("quantized_models\\resnet50_quark_int8.onnx")

os.makedirs(results_dir, exist_ok=True)

original_working_dir = os.getcwd()
os.chdir(results_dir)

# === Load val_map.txt and select subset if configured ===
with open(map_file) as f:
    entries = [line.strip().split() for line in f]

full_dataset_size = len(entries)

if NUM_IMAGES is not None and NUM_IMAGES < len(entries):
    print(f" Using subset of {NUM_IMAGES} images (from {len(entries)} total)")
    random.seed(42)
    entries = random.sample(entries, NUM_IMAGES)

image_paths = [image_dir / e[0] for e in entries]
ground_truth = [int(e[1]) for e in entries]
sample_indices = list(range(len(image_paths)))

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Dataset ===
class ImagenetDataset:
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels
        self.cache = {}

    def __len__(self):
        return len(self.image_paths)

    def load_samples(self, indices):
        for idx in indices:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            tensor = transform(img).unsqueeze(0).numpy()
            self.cache[idx] = (tensor, self.labels[idx])

    def unload_samples(self, indices):
        for idx in indices:
            self.cache.pop(idx, None)

    def get_sample(self, idx):
        return self.cache[idx]

dataset = ImagenetDataset(image_paths, ground_truth)

# "CPUExecutionProvider", "DmlExecutionProvider", VitisAIExecutionProvider
session = ort.InferenceSession(
    onnx_model_path,
    providers=["VitisAIExecutionProvider"])


input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === Performance Metrics Collection ===
predictions = {}
issued_sample_indices = set()
latencies = []
batch_times = []
total_samples_processed = 0
start_time = None
end_time = None

def issue_queries(query_samples):
    global total_samples_processed, start_time, end_time

    if start_time is None:
        start_time = time.time()

    for query in query_samples:
        sample_idx = query.index
        tensor, _ = dataset.get_sample(sample_idx)  # Assuming (data, label)

        batch_start = time.time()
        output = session.run([output_name], {input_name: tensor})[0]
        batch_end = time.time()

        batch_latency = batch_end - batch_start
        batch_times.append(batch_latency)
        latencies.append(batch_latency)
        
        top1 = int(np.argmax(output[0]))
        top5 = [int(x) for x in np.argsort(output[0])[-5:][::-1]]

        issued_sample_indices.add(sample_idx)
        predictions[sample_idx] = {
            "top1": top1,
            "top5": top5
        }

        response = QuerySampleResponse(query.id, 0, 0)
        QuerySamplesComplete([response])
        total_samples_processed += 1

        if total_samples_processed % 500 == 0:
            print(f"Processed {total_samples_processed} samples...", end="\r")

    end_time = time.time()

def flush_queries():
    pass

def load_samples_to_ram(sample_indices):
    dataset.load_samples(sample_indices)

def unload_samples_from_ram(sample_indices):
    dataset.unload_samples(sample_indices)

def run_performance_test():
    """Run MLPerf performance test"""
    global latencies, batch_times, total_samples_processed, start_time, end_time
    global predictions, issued_sample_indices
    
    predictions = {}
    issued_sample_indices = set()
    latencies = []
    batch_times = []
    total_samples_processed = 0
    start_time = None
    end_time = None
    
    print("\n Running PERFORMANCE test...")
    
    # Configure logging
    log_settings = LogSettings()
    log_settings.log_output.outdir = "."  # Use current directory (which is now results_dir)
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.log_output.copy_detail_to_stdout = True
    
    
    settings = TestSettings()
    settings.scenario = TestScenario.SingleStream
    settings.mode = TestMode.PerformanceOnly
    settings.min_query_count = 2048
    settings.min_duration_ms = 10000
    
    performance_count = len(sample_indices) # Limit the performance queries to the same number of samples
    
    sut = ConstructSUT(issue_queries, flush_queries)
    qsl = ConstructQSL(
        performance_count, 
        min(2048, performance_count),
        load_samples_to_ram,
        unload_samples_from_ram
    )
    
    start_time = time.time()
    
    StartTest(sut, qsl, settings)

    if end_time is None:
        end_time = time.time()
    
    perf_stats = save_performance_stats()
    
    # Cleanup
    DestroyQSL(qsl)
    DestroySUT(sut)
    
    return perf_stats

def save_performance_stats():
    """Save detailed performance statistics"""
    if not latencies:
        print(" No performance data collected")
        return {}
    
    test_duration = end_time - start_time if start_time and end_time else 0
    throughput = total_samples_processed / test_duration if test_duration > 0 else 0
    
    if len(latencies) == 0:
        print(" No latency data collected, using estimates")
        latencies.append(test_duration / total_samples_processed if total_samples_processed > 0 else 0)
    
    if len(batch_times) == 0:
        print(" No batch time data collected, using estimates")
        batch_times.append(test_duration / (total_samples_processed / BATCH_SIZE) if total_samples_processed > 0 else 0)
    
    stats = {
        "total_samples": total_samples_processed,
        "test_duration_seconds": test_duration,
        "throughput_samples_per_second": throughput,
        "latency_stats": {
            "mean": statistics.mean(latencies) * 1000,  # ms
            "median": statistics.median(latencies) * 1000,  # ms
            "min": min(latencies) * 1000,  # ms
            "max": max(latencies) * 1000,  # ms
            "p90": np.percentile(latencies, 90) * 1000,  # ms
            "p95": np.percentile(latencies, 95) * 1000,  # ms
            "p99": np.percentile(latencies, 99) * 1000,  # ms
        },
        "batch_stats": {
            "mean": statistics.mean(batch_times) * 1000,  # ms
            "median": statistics.median(batch_times) * 1000,  # ms
            "min": min(batch_times) * 1000,  # ms
            "max": max(batch_times) * 1000,  # ms
        }
    }
    
    # Explicitly print key performance metrics
    print(f"\n PERFORMANCE SUMMARY:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Duration: {stats['test_duration_seconds']:.2f} seconds")
    print(f"Throughput: {stats['throughput_samples_per_second']:.2f} samples/second")
    print(f"Average latency: {stats['latency_stats']['mean']:.2f} ms/sample")
    #print(f"P90 latency: {stats['latency_stats']['p90']:.2f} ms")
    
    # Save performance results
    perf_txt = Path("performance.txt")
    with open(perf_txt, "w") as f:
        f.write(f"===== PERFORMANCE RESULTS =====\n\n")
        f.write(f"Scenario: Offline\n")
        f.write(f"Total samples processed: {stats['total_samples']}\n")
        f.write(f"Test duration: {stats['test_duration_seconds']:.2f} seconds\n")
        f.write(f"Throughput: {stats['throughput_samples_per_second']:.2f} samples/second\n\n")
        
        f.write(f"===== LATENCY (ms) =====\n")
        f.write(f"Mean: {stats['latency_stats']['mean']:.2f}\n")
        f.write(f"Median: {stats['latency_stats']['median']:.2f}\n")
        f.write(f"Min: {stats['latency_stats']['min']:.2f}\n")
        f.write(f"Max: {stats['latency_stats']['max']:.2f}\n")
        f.write(f"90th percentile: {stats['latency_stats']['p90']:.2f}\n")
        f.write(f"95th percentile: {stats['latency_stats']['p95']:.2f}\n")
        f.write(f"99th percentile: {stats['latency_stats']['p99']:.2f}\n\n")
        
        f.write(f"===== BATCH PROCESSING TIME (ms) =====\n")
        f.write(f"Mean: {stats['batch_stats']['mean']:.2f}\n")
        f.write(f"Median: {stats['batch_stats']['median']:.2f}\n")
        f.write(f"Min: {stats['batch_stats']['min']:.2f}\n")
        f.write(f"Max: {stats['batch_stats']['max']:.2f}\n")
    
    print(f"\n Performance statistics written to {perf_txt}")
    
    perf_json = Path("performance_stats.json")
    with open(perf_json, "w") as f:
        json.dump(stats, f, indent=2)
    return stats

# === Run the tests ===
results = {
    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "size": len(image_paths),
        "full_dataset_size": full_dataset_size,
        "batch_size": BATCH_SIZE
    }
}

print(f"\n MLPerf Inference ResNet50 - SingleStream Scenario")
print(f"Dataset size: {len(image_paths)} images, Batch size: {BATCH_SIZE}")

if RUN_PERFORMANCE:
    performance_stats = run_performance_test()
    results["performance"] = performance_stats

print(f"Results saved to: {os.getcwd()}")

# Check if the required MLPerf log files were generated
summary_file = Path("mlperf_log_summary.txt")
if not summary_file.exists():
    print(f" MLPerf log summary file not found at: {summary_file}")
    
if RUN_PERFORMANCE and "performance" in results:
    perf = results["performance"]
    if perf and "throughput_samples_per_second" in perf:
        print(f"\nThroughput: {perf['throughput_samples_per_second']:.2f} samples/second")
        #print(f"Average latency: {perf['latency_stats']['mean']:.2f} ms/sample")
        #print(f"P90 latency: {perf['latency_stats']['p90']:.2f} ms")

os.chdir(original_working_dir)