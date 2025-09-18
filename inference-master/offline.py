"""
MLPerf Inference Benchmark for ResNet50 using ONNX Runtime - Offline Scenario.

This script runs the MLPerf Offline scenario to measure the maximum possible
throughput of the system by providing all samples to the SUT at once.

Example usage:
python inference-master/offline.py `
    --image_dir "dataset/ILSVRC2012_img_val" `
    --map_file "dataset/val_map.txt" `
    --onnx_model_path "pipeline_scripts\quantized_models\resnet50_quant_int8.onnx" `
    --results_dir "inference-master/results/offline" `
    --num_images 10 `
    --run_performance `
    --npu

Note: The execution provider flag is required. 
Replace --cpu with --gpu or --npu to target different hardware.
"""
import os
import sys
import argparse
import logging
import random
import time
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

import utils

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

from mlperf_loadgen import (
    TestSettings, TestScenario, TestMode,
    QuerySample, QuerySampleResponse,
    StartTest, QuerySamplesComplete,
    ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
    LogSettings
)

class SUT:
    """
    Defines the System Under Test (SUT) for the MLPerf Offline scenario.
    It processes all available queries by iterating through them one by one,
    as required by execution providers that do not support large batches.
    """
    def __init__(self, onnx_model_path: Path, dataset: utils.ImagenetDataset, provider: str):
        log.info("Initializing SUT for Offline scenario...")
        self.dataset = dataset
        self.ground_truth = {i: label for i, label in enumerate(dataset.labels)}
        
        # State and metrics are now instance variables
        self.predictions: Dict[int, Dict[str, Any]] = {}
        self.total_samples_processed = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        log.info(f"Loading ONNX model from: {onnx_model_path}")
        self.session = ort.InferenceSession(str(onnx_model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
        log.info("SUT Initialized.")

    def issue_queries(self, query_samples: List[QuerySample]) -> None:
        if not query_samples:
            log.warning("Received an empty list of queries.")
            return

        log.info(f"Received {len(query_samples)} queries")
        self.start_time = time.time()

        responses = []
        for query in query_samples:
            tensor, _ = self.dataset.get_sample(query.index)
            output = self.session.run([self.output_name], {self.input_name: tensor})[0]
            top1_pred = np.argmax(output, axis=1)[0]
            
            sample_idx = query.index
            self.predictions[sample_idx] = {"top1": int(top1_pred)}
            responses.append(QuerySampleResponse(query.id, 0, 0))

        self.end_time = time.time()
        
        QuerySamplesComplete(responses)
        self.total_samples_processed = len(query_samples)
        log.info(f"Completed processing {self.total_samples_processed} samples.")
        
    def flush_queries(self) -> None:
        """No-op for the Offline scenario as all queries are handled in issue_queries."""
        pass
        
    def reset(self) -> None:
        """Resets the state for a new test run (e.g., from accuracy to performance)."""
        self.predictions = {}
        self.total_samples_processed = 0
        self.start_time = 0.0
        self.end_time = 0.0
        log.info("SUT state has been reset.")

    def get_accuracy(self) -> Dict[str, Any]:
        """Calculates and returns a summary of accuracy metrics."""
        if not self.predictions:
            log.warning("No predictions available for accuracy calculation.")
            return {"top1_accuracy": 0, "samples": 0}

        top1_correct = 0
        for idx, pred_data in self.predictions.items():
            if self.ground_truth[idx] == pred_data.get("top1", -1):
                top1_correct += 1
        
        top1_acc = top1_correct / len(self.predictions) * 100 if self.predictions else 0
        
        accuracy_results = {"top1_accuracy": top1_acc, "samples": len(self.predictions)}
        
        with open("accuracy.txt", "w") as f:
            f.write(f"Top-1 Accuracy: {top1_acc:.2f}%\n")
            f.write(f"Total samples: {len(self.predictions)}\n")
        log.info(f"Accuracy results saved to accuracy.txt: {top1_acc:.2f}%")
        
        return accuracy_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculates and returns a summary of performance metrics."""
        if self.total_samples_processed == 0:
            log.warning("No performance data collected.")
            return {}

        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0
        throughput = self.total_samples_processed / duration if duration > 0 else 0
        avg_latency_ms = (duration / self.total_samples_processed) * 1000 if self.total_samples_processed > 0 else 0
        
        stats = {
            "total_samples": self.total_samples_processed,
            "test_duration_seconds": duration,
            "throughput_samples_per_second": throughput,
            "mean_latency_ms": avg_latency_ms
        }
        
        with open("performance_stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        log.info(f"Performance stats saved to performance_stats.json. Throughput: {throughput:.2f} samples/sec")

        return stats

def setup_logging(log_dir: Path) -> None:
    """Configures a logger to write to a file and the console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / "benchmark.log").resolve()

    logging.basicConfig(level=logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)-5.5s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(console_handler)
    log.propagate = False
    log.info(f"Logging configured. Log file at: {log_file}")

def run_test(sut_instance: SUT, dataset: utils.ImagenetDataset, settings: TestSettings):
    """Orchestrates a single MLPerf test run (either accuracy or performance)."""
    sut_instance.reset()
    
    qsl = ConstructQSL(len(dataset), min(len(dataset), 1024), dataset.load_samples, dataset.unload_samples)
    sut = ConstructSUT(sut_instance.issue_queries, sut_instance.flush_queries)

    log.info(f"Starting MLPerf test (Mode: {settings.mode.name})...")
    StartTest(sut, qsl, settings)
    log.info("MLPerf test finished.")
    
    DestroySUT(sut)
    DestroyQSL(qsl)

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate the benchmark run."""
    original_wd = Path.cwd()
    os.chdir(args.results_dir)
    log.info(f"Temporarily changed working directory to: {Path.cwd()}")

    try:
        for arg, value in vars(args).items():
            log.info(f"  {arg}: {value}")
        
        selected_provider = ""
        if args.npu:
            selected_provider = "VitisAIExecutionProvider"
        elif args.gpu:
            selected_provider = "DmlExecutionProvider"
        elif args.cpu:
            selected_provider = "CPUExecutionProvider"
        
        if not selected_provider:
            log.error("Execution provider not specified. Please use --cpu, --gpu, or --npu.")
            sys.exit(1)

        log.info("--- MLPerf Inference Benchmark - Offline ---")
        log.info("Configuration (using absolute paths for clarity):")
        log.info(f"  image_dir: {args.image_dir.resolve()}")
        log.info(f"  map_file: {args.map_file.resolve()}")
        log.info(f"  onnx_model_path: {args.onnx_model_path.resolve()}")
        log.info(f"  num_images: {args.num_images}")

        with open(args.map_file) as f:
            entries = [line.strip().split() for line in f]

        if args.num_images and args.num_images < len(entries):
            log.info(f"Using a random subset of {args.num_images} images.")
            random.seed(42)
            entries = random.sample(entries, args.num_images)

        image_paths = [args.image_dir / e[0] for e in entries]
        ground_truth = [int(e[1]) for e in entries]

        preprocessor = transforms.Compose([
            transforms.Resize(utils.IMAGE_RESIZE),
            transforms.CenterCrop(utils.IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=utils.IMAGE_NET_MEAN, std=utils.IMAGE_NET_STD)
        ])
        dataset = utils.ImagenetDataset(image_paths, ground_truth, preprocessor)
        sut_instance = SUT(args.onnx_model_path, dataset, provider=selected_provider)

        log_settings = LogSettings()
        log_settings.log_output.outdir = "."
        log_settings.log_output.copy_summary_to_stdout = True

        if args.run_accuracy:
            settings = TestSettings()
            settings.scenario = TestScenario.Offline
            settings.mode = TestMode.AccuracyOnly
            run_test(sut_instance, dataset, settings)
            sut_instance.get_accuracy()

        if args.run_performance:
            settings = TestSettings()
            settings.scenario = TestScenario.Offline
            settings.mode = TestMode.PerformanceOnly
            settings.offline_expected_qps = 2000
            run_test(sut_instance, dataset, settings)
            sut_instance.get_performance_stats()

        log.info(f"Benchmark complete. Results are in: {args.results_dir.resolve()}")

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        log.info(f"Restoring original working directory: {original_wd}")
        os.chdir(original_wd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark for ResNet50 - Offline.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to ImageNet validation images.")
    parser.add_argument("--map_file", type=Path, required=True, help="Path to 'val_map.txt' file.")
    parser.add_argument("--onnx_model_path", type=Path, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--results_dir", type=Path, default=Path("results_offline"), help="Directory to save logs and results.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to use. Default is all.")
    parser.add_argument("--run_accuracy", action="store_true", help="Run the accuracy test.")
    parser.add_argument("--run_performance", action="store_true", help="Run the performance test.")
    
    provider_group = parser.add_mutually_exclusive_group(required=True)
    provider_group.add_argument("--cpu", action="store_true", help="Use CPUExecutionProvider.")
    provider_group.add_argument("--gpu", action="store_true", help="Use DmlExecutionProvider")
    provider_group.add_argument("--npu", action="store_true", help="Use VitisAIExecutionProvider")
    
    args = parser.parse_args()

    args.image_dir = args.image_dir.resolve()
    args.map_file = args.map_file.resolve()
    args.onnx_model_path = args.onnx_model_path.resolve()
    args.results_dir = args.results_dir.resolve()

    if not (args.run_accuracy or args.run_performance):
        sys.exit("ERROR: You must specify at least one test to run (--run_accuracy or --run_performance).")
    
    # Validate inputs
    for path_arg in [args.image_dir, args.map_file, args.onnx_model_path]:
        if not path_arg.exists():
            sys.exit(f"ERROR: File or directory not found: {path_arg.resolve()}")
    
    log = utils.setup_logging(args.results_dir)
    
    main(args)