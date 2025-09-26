"""
MLPerf Inference Benchmark for ResNet50 using ONNX Runtime - Offline Scenario.

This script runs the MLPerf Offline scenario to measure the maximum possible
throughput of the system by providing all samples to the SUT at once.

Example usage:
python inference-master/offline.py `
    --image_dir "dataset/ILSVRC2012_img_val" `
    --onnx_model_path "pipeline_scripts\quantized_models\resnet50_quant_int8.onnx" `
    --results_dir "inference-master/results/offline" `
    --num_images 10000 `
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
from torchvision import transforms # type: ignore

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
            tensor = self.dataset.get_sample(query.index)
            _ = self.session.run([self.output_name], {self.input_name: tensor})
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

def run_test(sut_instance: SUT, dataset: utils.ImagenetDataset, settings: TestSettings):
    """Hand a single MLPerf test run (either accuracy or performance)."""
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
        else:
            log.error("Execution provider not specified. Please use --cpu, --gpu, or --npu.")
            sys.exit(1)

        log.info("--- MLPerf Inference Benchmark - Offline ---")
        log.info("Configuration (using absolute paths for clarity):")
        log.info(f"  image_dir: {args.image_dir.resolve()}")
        log.info(f"  onnx_model_path: {args.onnx_model_path.resolve()}")
        log.info(f"  num_images: {args.num_images}")

        image_paths = sorted(list(args.image_dir.glob("*.JPEG")))
        if not image_paths:
             image_paths = sorted(list(args.image_dir.glob("*.[jJ][pP][gG]")) + list(args.image_dir.glob("*.[jJ][pP][eE][gG]")) + list(args.image_dir.glob("*.[pP][nN][gG]")))

        if not image_paths:
            log.error(f"No images found in {args.image_dir}. Check the path and file extensions.")
            sys.exit(1)
        log.info(f"Found {len(image_paths)} images.")

        if args.num_images and args.num_images < len(image_paths):
            log.info(f"Using a random subset of {args.num_images} images.")
            random.seed(42)
            image_paths = random.sample(image_paths, args.num_images)

        preprocessor = transforms.Compose([
            transforms.Resize(utils.IMAGE_RESIZE),
            transforms.CenterCrop(utils.IMAGE_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=utils.IMAGE_NET_MEAN, std=utils.IMAGE_NET_STD)
        ])
        dataset = utils.ImagenetDataset(image_paths, preprocessor)
        sut_instance = SUT(args.onnx_model_path, dataset, provider=selected_provider)

        # MLPerf LoadGen Configuration
        log_settings = LogSettings()
        log_settings.log_output.outdir = "."
        log_settings.log_output.copy_summary_to_stdout = True

        settings = TestSettings()
        settings.scenario = TestScenario.Offline
        settings.mode = TestMode.PerformanceOnly
        settings.offline_expected_qps = 5000
        settings.min_query_count = args.num_images if args.num_images else 100 
        settings.min_duration_ms = 100
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
    parser.add_argument("--onnx_model_path", type=Path, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--results_dir", type=Path, default=Path("results_offline"), help="Directory to save logs and results.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to use. Default is all.")
    
    provider_group = parser.add_mutually_exclusive_group(required=True)
    provider_group.add_argument("--cpu", action="store_true", help="Use CPUExecutionProvider.")
    provider_group.add_argument("--gpu", action="store_true", help="Use DmlExecutionProvider")
    provider_group.add_argument("--npu", action="store_true", help="Use VitisAIExecutionProvider")
    
    args = parser.parse_args()

    args.image_dir = args.image_dir.resolve()
    args.onnx_model_path = args.onnx_model_path.resolve()
    args.results_dir = args.results_dir.resolve()
    
    # Input validation
    for path_arg in [args.image_dir, args.onnx_model_path]:
        if not path_arg.exists():
            sys.exit(f"ERROR: File or directory not found: {path_arg.resolve()}")
    
    log = utils.setup_logging(args.results_dir)
    
    main(args)