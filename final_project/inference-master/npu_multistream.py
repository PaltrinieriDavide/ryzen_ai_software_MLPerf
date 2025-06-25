import os
import json
import time
import random
import numpy as np
import statistics
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from torchvision import transforms
import threading
import queue

from mlperf_loadgen import (
    TestSettings, TestScenario, TestMode,
    QuerySample, QuerySampleResponse,
    StartTest, QuerySamplesComplete,
    ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
    LogSettings, LoggingMode
)

RUN_PERFORMANCE = True
BATCH_SIZE = 4
NUM_IMAGES = 10000

image_dir = Path("imagenet_val_dataset\\ILSVRC2012_img_val")
map_file = Path("imagenet_val_dataset\\val_map.txt")
results_dir = Path("results_multistream")
onnx_model_path = Path("uantized_models\\resnet50_quark_int8.onnx")
vaip_config = Path("vaip_config.json")

os.makedirs(results_dir, exist_ok=True)

original_working_dir = os.getcwd()
os.chdir(results_dir)
print(f"Changed working directory to: {os.getcwd()}")

# === Load val_map.txt and select subset if configured ===
with open(map_file) as f:
    entries = [line.strip().split() for line in f]

# Apply NUM_IMAGES limitation if set
if NUM_IMAGES is not None and NUM_IMAGES < len(entries):
    print(f"Using subset of {NUM_IMAGES} images (from {len(entries)} total)")
    random.seed(42)
    entries = random.sample(entries, NUM_IMAGES)

image_paths = [image_dir / e[0] for e in entries]
ground_truth = [int(e[1]) for e in entries]

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
        print(f"Loading {len(indices)} samples into RAM...")
        for idx in indices:
            if idx not in self.cache:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                tensor = transform(img).numpy()
                self.cache[idx] = (tensor, self.labels[idx])
        print("Sample loading complete")

    def unload_samples(self, indices):
        for idx in indices:
            self.cache.pop(idx, None)

    def get_sample(self, idx):
        return self.cache[idx]

dataset = ImagenetDataset(image_paths, ground_truth)


# === System Under Test (SUT) class for MultiStream ===
class SUT:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.work_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker_loop)
        self.is_done = False
        provider_options = [{'config_file': 'vaip_config.json'}]
        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            onnx_model_path,
            provider_options = provider_options,
            providers=["VitisAIExecutionProvider"]) # "CPUExecutionProvider", "DmlExecutionProvider"

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Model loaded with input name: {self.input_name}, output name: {self.output_name}")

        self.worker_thread.start()

    def worker_loop(self):
        """The background thread that performs inference."""
        while not self.is_done:
            try:
                # Use a timeout to periodically check the is_done flag
                # and to form batches even if they are not full.
                work_items = [self.work_queue.get(timeout=0.01)]

                # Build a batch
                while len(work_items) < self.batch_size:
                    try:
                        work_items.append(self.work_queue.get_nowait())
                    except queue.Empty:
                        break # Queue is empty, process the current batch

                # Prepare batch for inference
                batch_indices = [q.index for q in work_items]
                tensors = [dataset.get_sample(idx)[0] for idx in batch_indices]
                batch_tensor = np.stack(tensors, axis=0)

                # Run inference
                outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]

                # Process results and send responses
                for i, query in enumerate(work_items):
                    output = outputs[i]
                    # The response payload is not used in PerformanceOnly mode, so 0 is fine
                    response = QuerySampleResponse(query.id, 0, 0)
                    QuerySamplesComplete([response])

            except queue.Empty:
                # This is expected when the queue is empty, just continue
                continue

    def issue_queries(self, query_samples):
        """Called by the loadgen to send new queries."""
        for query in query_samples:
            self.work_queue.put(query)

    def flush_queries(self):
        """Called by the loadgen at the end of the test to process remaining queries."""
        while not self.work_queue.empty():
            time.sleep(0.01)

    def stop(self):
        """Stops the worker thread."""
        self.is_done = True
        self.worker_thread.join()
        print("SUT worker stopped.")


def run_performance_test(sut):
    """Run MLPerf performance test"""
    print("\nRunning PERFORMANCE test (MultiStream)...")

    log_settings = LogSettings()
    log_settings.log_output.outdir = "."
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.log_mode = LoggingMode.AsyncPoll

    settings = TestSettings()
    settings.scenario = TestScenario.MultiStream
    settings.mode = TestMode.PerformanceOnly

    # Min duration and query count for a valid run
    settings.min_duration_ms = 60000 # Use 60 seconds for official runs
    settings.min_query_count = 2048

    performance_count = len(image_paths)

    qsl = ConstructQSL(
        performance_count,
        min(2048, performance_count),
        dataset.load_samples,
        dataset.unload_samples
    )

    lg_sut = ConstructSUT(sut.issue_queries, sut.flush_queries)

    print("Starting MLPerf test...")
    StartTest(lg_sut, qsl, settings)

    print("MLPerf test finished.")

    # Cleanup
    DestroyQSL(qsl)
    DestroySUT(lg_sut)


# === Main Execution Logic ===
if __name__ == "__main__":
    print(f"\nMLPerf Inference ResNet50 - MultiStream Scenario")
    print(f"Dataset size: {len(image_paths)} images, SUT Batch size: {BATCH_SIZE}")

    sut = None
    try:
        sut = SUT(batch_size=BATCH_SIZE)
        if RUN_PERFORMANCE:
            run_performance_test(sut)

        print("\n--- TEST COMPLETE ---")
        print(f"Official results are in: {results_dir / 'mlperf_log_summary.txt'}")
        print("Look for '90th percentile latency' and 'Queries/second' under the 'Result is VALID' line.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if sut:
            sut.stop()
        os.chdir(original_working_dir)