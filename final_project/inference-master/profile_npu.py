# profile_npu.py
import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Attempt to import required modules and provide a helpful error message on failure.
try:
    import numpy as np
    import onnxruntime as ort
    from mlperf_loadgen import (ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
                                LogSettings, LoggingMode, QuerySampleResponse,
                                QuerySamplesComplete, TestMode, TestScenario,
                                TestSettings)
    from PIL import Image
    from torchvision import transforms
except ImportError as e:
    print(f"Import Error: {e}")
    print("\nPlease ensure all required packages are installed:")
    print("  pip install numpy pillow onnxruntime-vitisai torch torchvision")
    print("  To install mlperf_loadgen, navigate to the 'loadgen' directory and run: pip install .")
    sys.exit(1)


class ImagenetDataset:
    """Handles loading and preprocessing of the ImageNet dataset samples."""
    def __init__(self, image_paths: list, labels: list, preprocessor) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.cache = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def load_samples(self, indices: list) -> None:
        for idx in indices:
            if idx not in self.cache:
                img = Image.open(self.image_paths[idx]).convert("RGB")
                tensor = self.preprocessor(img).unsqueeze(0).numpy()
                self.cache[idx] = (tensor, self.labels[idx])

    def unload_samples(self, indices: list) -> None:
        for idx in indices:
            self.cache.pop(idx, None)

    def get_sample(self, idx: int) -> tuple:
        return self.cache[idx]


class MLPerfRunner:
    """Encapsulates the logic for running MLPerf accuracy and performance tests."""
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.image_paths = []
        self.ground_truth = []
        self.sample_indices = []
        self.dataset = None
        self.session = None

        # Test state variables
        self.predictions = {}
        self.latencies = []
        self.total_samples_processed = 0
        self.start_time = None
        self.end_time = None

        self.results_dir = self.config.results_dir.resolve()
        self.original_working_dir = Path.cwd()
        self.results_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.results_dir)
        print(f"Changed working directory to: {self.results_dir}")

    def setup(self) -> None:
        """Initializes the dataset and ONNX model."""
        print("\n--- Initializing Inference Worker ---")
        self._load_dataset()
        self._load_onnx_model()

    def _load_dataset(self) -> None:
        print("Loading dataset...")
        map_file = self.config.dataset_dir / "val_map.txt"
        image_dir = self.config.dataset_dir / "val"

        if not map_file.is_file() or not image_dir.is_dir():
            print(f"Error: Dataset directory or val_map.txt not found.")
            print(f"  - Searched in: {self.config.dataset_dir}")
            print(f"  - Please ensure 'val_map.txt' and a 'val' subdirectory exist.")
            sys.exit(1)

        with open(map_file) as f:
            entries = [line.strip().split() for line in f]

        num_images = self.config.num_images if self.config.num_images > 0 else len(entries)
        if num_images < len(entries):
            print(f"Using a subset of {num_images} images (out of {len(entries)} total).")
            random.seed(42)  # For reproducibility
            entries = random.sample(entries, num_images)

        self.image_paths = [image_dir / e[0] for e in entries]
        self.ground_truth = [int(e[1]) for e in entries]
        self.sample_indices = list(range(len(self.image_paths)))
        print(f"Dataset loaded: {len(self.image_paths)} images.")

        preprocessor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = ImagenetDataset(self.image_paths, self.ground_truth, preprocessor)

    def _load_onnx_model(self) -> None:
        print(f"Loading ONNX model: {self.config.model_path}")
        if not self.config.model_path.is_file():
            print(f"Error: ONNX model file not found at '{self.config.model_path}'")
            sys.exit(1)

        providers = ["VitisAIExecutionProvider"]
        print(f"Available ONNX providers: {ort.get_available_providers()}")
        print(f"Selected provider: {providers}")

        self.session = ort.InferenceSession(str(self.config.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"Model loaded. Input: '{self.input_name}', Output: '{self.output_name}'.")

    def _reset_state(self) -> None:
        """Resets metrics between test runs."""
        self.predictions.clear()
        self.latencies.clear()
        self.total_samples_processed = 0
        self.start_time = self.end_time = None

    def run_all_tests(self) -> None:
        """Runs the configured tests."""
        if self.config.run_accuracy:
            self._run_test(TestMode.AccuracyOnly, "Accuracy")
            self._calculate_and_save_accuracy()
        if self.config.run_performance:
            self._run_test(TestMode.PerformanceOnly, "Performance")
            self._calculate_and_save_performance()

    # --- MLPerf LoadGen Callbacks ---
    def issue_queries(self, query_samples: list) -> None:
        if self.start_time is None: self.start_time = time.perf_counter()

        indices = [qs.index for qs in query_samples]
        tensors = [self.dataset.get_sample(i)[0] for i in indices]

        for i in range(0, len(tensors), self.config.batch_size):
            batch_start_time = time.perf_counter()
            
            mini_batch_slice = slice(i, i + self.config.batch_size)
            mini_batch = np.vstack(tensors[mini_batch_slice])
            
            outputs = self.session.run([self.output_name], {self.input_name: mini_batch})[0]
            
            batch_end_time = time.perf_counter()
            
            batch_indices = indices[mini_batch_slice]
            per_sample_latency = (batch_end_time - batch_start_time) / len(batch_indices)
            self.latencies.extend([per_sample_latency] * len(batch_indices))

            responses = []
            for j, sample_idx in enumerate(batch_indices):
                top1_pred = np.argmax(outputs[j])
                self.predictions[sample_idx] = {"top1": int(top1_pred)}
                
                response_id = query_samples[i + j].id
                responses.append(QuerySampleResponse(response_id, 0, 0))
            
            QuerySamplesComplete(responses)
            self.total_samples_processed += len(batch_indices)
        
        self.end_time = time.perf_counter()

    def flush_queries(self) -> None: pass
    def load_samples_to_ram(self, indices: list) -> None: self.dataset.load_samples(indices)
    def unload_samples_from_ram(self, indices: list) -> None: self.dataset.unload_samples(indices)

    def _run_test(self, mode: TestMode, test_name: str) -> None:
        """Helper function to run a generic MLPerf test."""
        self._reset_state()
        print(f"\n--- Running MLPerf Test: {test_name} ---")

        log_settings = LogSettings()
        log_settings.log_output.outdir = str(self.results_dir)
        log_settings.log_output.copy_summary_to_stdout = True
        log_settings.log_mode = LoggingMode.ASYNC_POLL

        settings = TestSettings()
        settings.scenario = TestScenario.Offline
        settings.mode = mode
        settings.min_query_count = len(self.sample_indices)
        settings.min_duration_ms = 1  # Let query count dictate test length

        qsl = ConstructQSL(len(self.sample_indices), min(1024, len(self.sample_indices)), self.load_samples_to_ram, self.unload_samples_from_ram)
        sut = ConstructSUT(self.issue_queries, self.flush_queries)

        print("Starting MLPerf LoadGen...")
        StartTest(sut, qsl, settings)
        print("MLPerf LoadGen test complete.")

        DestroyQSL(qsl)
        DestroySUT(sut)

    def _calculate_and_save_accuracy(self) -> None:
        print("\n--- Calculating Accuracy ---")
        if not self.predictions:
            print("No predictions were collected. Skipping accuracy calculation.")
            return

        top1_correct = sum(1 for i, pred in self.predictions.items() if pred.get("top1") == self.ground_truth[i])
        top1_acc = (top1_correct / len(self.predictions)) * 100 if self.predictions else 0
        
        print(f"Top-1 Accuracy: {top1_acc:.2f}% ({len(self.predictions)} samples)")
        
        with open(self.results_dir / "accuracy_summary.txt", "w") as f:
            f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
            f.write(f"Total samples: {len(self.predictions)}\n")

    def _calculate_and_save_performance(self) -> None:
        print("\n--- Saving Performance Statistics ---")
        if not self.latencies or self.total_samples_processed == 0:
            print("No performance data was collected.")
            return

        duration = self.end_time - self.start_time
        throughput = self.total_samples_processed / duration
        
        stats = {
            "config": {k: str(v) for k, v in vars(self.config).items() if not k == "run_inference_worker"},
            "results": {
                "total_samples": self.total_samples_processed,
                "duration_sec": round(duration, 3),
                "throughput_fps": round(throughput, 2),
                "avg_latency_ms": round(statistics.mean(self.latencies) * 1000, 3),
                "p90_latency_ms": round(np.percentile(self.latencies, 90) * 1000, 3),
            }
        }
        
        print(f"Throughput: {stats['results']['throughput_fps']:.2f} FPS")
        print(f"Average Latency: {stats['results']['avg_latency_ms']:.3f} ms")
        print(f"90th Percentile Latency: {stats['results']['p90_latency_ms']:.3f} ms")

        with open(self.results_dir / "performance_summary.json", "w") as f:
            json.dump(stats, f, indent=4)

    def cleanup(self) -> None:
        """Restores the original working directory."""
        os.chdir(self.original_working_dir)
        print(f"\nRestored working directory to: {self.original_working_dir}")


def check_system() -> bool:
    """Verifies that required command-line tools are available."""
    print("--- System Check ---")
    if not shutil.which("vaitrace"):
        print("Error: 'vaitrace' not found in system PATH.")
        print("Please ensure the Ryzen AI SDK is installed correctly, as 'vaitrace' is the required profiling tool.")
        return False
    print("Found 'ryzen-ai-analyzer'.")
    return True


def main() -> None:
    """Main entry point for orchestrating and running the benchmark."""
    parser = argparse.ArgumentParser(
        description="Unified script for ONNX model benchmarking with energy analysis on AMD Ryzen AI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Group arguments for better --help output
    config_group = parser.add_argument_group('Benchmark Configuration')
    config_group.add_argument("--model-path", type=Path, required=True, help="Path to the .onnx model file.")
    config_group.add_argument("--dataset-dir", type=Path, required=True, help="Path to ImageNet dataset root (containing 'val' and 'val_map.txt').")
    config_group.add_argument("--results-dir", type=Path, default=Path("./inference_results"), help="Directory to save inference performance/accuracy results.")
    config_group.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    config_group.add_argument("--num-images", type=int, default=1000, help="Number of images to use for the test (0 for all).")

    mode_group = parser.add_argument_group('Execution Mode')
    mode_group.add_argument("--run-accuracy", action="store_true", help="Run the accuracy test.")
    mode_group.add_argument("--run-performance", action="store_true", help="Run the performance test. (Default if no mode is selected)")

    orch_group = parser.add_argument_group('Orchestrator Control')
    orch_group.add_argument("--no-analyzer", action="store_true", help="Run inference test directly without the energy profiler.")
    orch_group.add_argument("--analyzer-output-dir", type=Path, default=Path("./energy_analysis"), help="Directory for ryzen-ai-analyzer reports.")
    
    # Internal flag for self-invocation, hidden from help
    parser.add_argument("--run-inference-worker", action="store_true", help=argparse.SUPPRESS)
    
    args = parser.parse_args()

    # Default to performance mode if no mode is specified
    if not args.run_accuracy and not args.run_performance:
        args.run_performance = True

    if args.run_inference_worker:
        # --- WORKER MODE ---
        # This block is executed by the subprocess launched by the orchestrator.
        runner = None
        try:
            runner = MLPerfRunner(args)
            runner.setup()
            runner.run_all_tests()
        except Exception as e:
            print(f"\nFatal error in inference worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if runner:
                runner.cleanup()
    else:
        # --- ORCHESTRATOR MODE ---
        # This block launches the script again under ryzen-ai-analyzer.
        print("--- Starting in Orchestrator Mode ---")
        
        if args.no_analyzer:
            print("\n--no-analyzer specified. Running inference worker directly.")
            args.run_inference_worker = True
            # Re-call main to enter the worker block
            main()
            return

        if not check_system():
            sys.exit(1)

        # Build the command to re-invoke this script as a worker
        command = [
    "vaitrace",
    "-p",
    "-o", str(args.analyzer_output_dir.resolve()),
    sys.executable,
    __file__,
    "--run-inference-worker"
]
        # Propagate all other arguments to the worker process
        for arg, value in vars(args).items():
            if arg in ["run_inference_worker", "analyzer_output_dir", "no_analyzer"]:
                continue
            if isinstance(value, bool) and value:
                command.append(f"--{arg.replace('_', '-')}")
            elif not isinstance(value, bool):
                command.append(f"--{arg.replace('_', '-')}")
                command.append(str(value))
        
        print("\n--- Launching Analysis ---")
        print(f"Inference results will be saved to: {args.results_dir.resolve()}")
        print(f"Energy analysis will be saved to: {args.analyzer_output_dir.resolve()}")
        print(f"Executing command: {' '.join(command)}\n")

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            
            process.wait()
            if process.returncode == 0:
                print("\nAnalysis complete.")
            else:
                print(f"\nAnalysis failed with exit code: {process.returncode}")
        
        except FileNotFoundError:
            print("Error: 'ryzen-ai-analyzer' could not be found. Please check your installation.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()