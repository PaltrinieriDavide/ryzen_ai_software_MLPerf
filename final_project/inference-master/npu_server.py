import os
import time
import random
import numpy as np
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

# === Configuration ===
NUM_IMAGES = 1000
BATCH_SIZE = 1

image_dir = Path("imagenet_val_dataset\\ILSVRC2012_img_val")
map_file = Path("imagenet_val_dataset\\val_map.txt")
results_dir = Path("results_server")
onnx_model_path = Path("quantized_models\\resnet50_quark_int8.onnx")

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)
original_working_dir = os.getcwd()
os.chdir(results_dir)
print(f"Changed working directory to: {os.getcwd()}")

# === Dataset Loading ===
with open(map_file) as f:
    entries = [line.strip().split() for line in f]

if NUM_IMAGES is not None and NUM_IMAGES < len(entries):
    print(f"Using subset of {NUM_IMAGES} images (from {len(entries)} total)")
    random.seed(42)
    entries = random.sample(entries, NUM_IMAGES)

image_paths = [image_dir / e[0] for e in entries]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class ImagenetDataset:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.cache = {}

    def load_samples(self, indices):
        print(f"Loading {len(indices)} samples into RAM...")
        for idx in indices:
            if idx not in self.cache:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                tensor = transform(img).unsqueeze(0).numpy()
                self.cache[idx] = tensor
        print("Sample loading complete")

    def unload_samples(self, indices):
        for idx in indices:
            self.cache.pop(idx, None)

    def get_sample(self, idx):
        return self.cache[idx]

dataset = ImagenetDataset(image_paths)


# === System Under Test (SUT) for Server/MultiStream - WITH WORKER THREAD & POISON PILL ===
class SUT_Server:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.work_queue = queue.Queue()
        self.poison_pill = object()
        self.worker_thread = threading.Thread(target=self.worker_loop)

        print("Loading ONNX model...")
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=["VitisAIExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Model loaded with input name: {self.input_name}, output name: {self.output_name}")

        self.worker_thread.start()

    def worker_loop(self):
        while True:
            work_items = [self.work_queue.get()]

            if work_items[0] is self.poison_pill:
                break

            while len(work_items) < self.batch_size:
                try:
                    next_item = self.work_queue.get_nowait()
                    if next_item is self.poison_pill:
                        self.work_queue.put(self.poison_pill)
                        break
                    work_items.append(next_item)
                except queue.Empty:
                    break

            if not work_items:
                continue

            tensors = [dataset.get_sample(q.index) for q in work_items]
            batch_tensor = np.concatenate(tensors, axis=0)

            self.session.run([self.output_name], {self.input_name: batch_tensor})

            for query in work_items:
                response = QuerySampleResponse(query.id, 0, 0)
                QuerySamplesComplete([response])

    def issue_queries(self, query_samples):
        for query in query_samples:
            self.work_queue.put(query)

    def flush_queries(self):
        pass

    def stop(self):
        self.work_queue.put(self.poison_pill)
        self.worker_thread.join()
        print("SUT worker stopped.")

# === Main Test Logic ===
def run_performance_test(sut):
    print("\nRunning QUICK PERFORMANCE test (Server)...")

    log_settings = LogSettings()
    log_settings.log_output.outdir = "."
    log_settings.log_output.copy_summary_to_stdout = True
    log_settings.log_mode = LoggingMode.AsyncPoll

    settings = TestSettings()
    settings.scenario = TestScenario.Server
    settings.mode = TestMode.PerformanceOnly
    
    settings.server_target_qps = 20
    
    # ===============================================
    # === QUICK TEST SETTINGS                     ===
    # ===============================================
    print("Using settings for a quick test run (~1-2 seconds)...")
    settings.min_duration_ms = 1000
    settings.min_query_count = 1
    # ===============================================

    performance_count = len(image_paths)

    qsl = ConstructQSL(
        performance_count,
        performance_count,
        dataset.load_samples,
        dataset.unload_samples
    )

    lg_sut = ConstructSUT(sut.issue_queries, sut.flush_queries)

    print("Starting MLPerf test (Server)...")
    StartTest(lg_sut, qsl, settings)
    print("MLPerf test finished.")

    DestroyQSL(qsl)
    DestroySUT(lg_sut)

# === Main Execution Logic ===
if __name__ == "__main__":
    print(f"\nMLPerf Inference ResNet50 - Server Scenario")
    print(f"BATCH_SIZE: {BATCH_SIZE}")

    sut = SUT_Server(batch_size=BATCH_SIZE)
    try:
        run_performance_test(sut)
    except Exception as e:
        print(f"An error occurred during the test: {e}")
    finally:
        print("Test run finished or failed. Stopping SUT...")
        sut.stop()
    
    print("\n--- QUICK TEST COMPLETE ---")
    print("If the script exited without hanging, the deadlock is fixed.")
    print("You can now increase min_duration_ms and min_query_count for a full run.")
    
    os.chdir(original_working_dir)