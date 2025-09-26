"""
MLPerf Inference Benchmark for ResNet50 using ONNX Runtime

This script runs the MLPerf SingleStream scenario,
which is designed to measure the latency of a single inference request.

python inference-master/singlestream.py `
    --image_dir "dataset/ILSVRC2012_img_val" `
    --onnx_model_path "pipeline_scripts\quantized_models\resnet50_quant_int8.onnx" `
    --results_dir "inference-master/results/singleStream" `
    --num_images 100 `
    --npu

Note: The execution provider flag is required. 
Replace --cpu with --gpu or --npu to target different hardware.
"""
import os
import sys
import json
import time
import random
import argparse
import logging
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms

import utils

from mlperf_loadgen import (
    TestSettings, TestScenario, TestMode,
    QuerySample, QuerySampleResponse,
    StartTest, QuerySamplesComplete,
    ConstructQSL, ConstructSUT, DestroyQSL, DestroySUT,
    LogSettings
)

class SUT:
    """Defines the System Under Test (SUT) for MLPerf."""
    def __init__(self, onnx_model_path: Path, dataset: utils.ImagenetDataset, provider: str):
        log.info("Initializing SUT...")
        self.dataset = dataset
        self.latencies: List[float] = []
        self.total_samples_processed = 0
        self.start_time: float = 0.0
        self.end_time: float = 0.0

        log.info(f"Loading ONNX model from: {onnx_model_path}")
        self.session = ort.InferenceSession(str(onnx_model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info("SUT Initialized.")

    def issue_queries(self, query_samples: List[QuerySample]) -> None:
        """Processes queries issued by the MLPerf load generator."""
        if not self.start_time:
            self.start_time = time.time()

        for query in query_samples:
            tensor = self.dataset.get_sample(query.index)
            inference_start = time.time()
            _ = self.session.run([self.output_name], {self.input_name: tensor})[0]
            self.latencies.append(time.time() - inference_start)
            
            response = QuerySampleResponse(query.id, 0, 0)
            QuerySamplesComplete([response])
            self.total_samples_processed += 1
            if self.total_samples_processed % 500 == 0:
                log.info(f"Processed {self.total_samples_processed} samples...")

        self.end_time = time.time()

    def flush_queries(self) -> None:
        """Flushes any pending queries. No-op for SingleStream."""
        pass

    def get_performance_stats(self) -> Dict[str, Any]:
        """Calculates and returns a summary of performance metrics."""
        if not self.latencies:
            log.warning("No performance data collected.")
            return {}
        
        duration = self.end_time - self.start_time
        throughput = self.total_samples_processed / duration if duration > 0 else 0

        return {
            "total_samples": self.total_samples_processed,
            "test_duration_seconds": duration,
            "throughput_samples_per_second": throughput,
            "latency_stats_ms": {
                "mean": statistics.mean(self.latencies) * 1000,
                "median": statistics.median(self.latencies) * 1000,
                "min": min(self.latencies) * 1000,
                "max": max(self.latencies) * 1000,
                "p90": np.percentile(self.latencies, 90) * 1000,
                "p95": np.percentile(self.latencies, 95) * 1000,
                "p99": np.percentile(self.latencies, 99) * 1000,
            }
        }

def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate the benchmark run."""

    original_wd = Path.cwd()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.results_dir)
    log.info(f"Temporarily changed working directory to: {Path.cwd()}")

    sut_instance = None
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

        log.info("--- MLPerf Inference Benchmark - singlestream ---")
        log.info("Configuration:")
        log.info(f"  image_dir: {args.image_dir.resolve()}")
        log.info(f"  onnx_model_path: {args.onnx_model_path.resolve()}")
        log.info(f"  num_images: {args.num_images}")

        log.info("Loading dataset...")
        # Scansiona la directory delle immagini invece di leggere il map_file
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
        sut_instance = SUT(args.onnx_model_path, dataset, selected_provider)
        
        log_settings = LogSettings()
        log_settings.log_output.outdir = "."
        log_settings.log_output.copy_summary_to_stdout = True
        log_settings.log_output.copy_detail_to_stdout = True

        settings = TestSettings()
        settings.scenario = TestScenario.SingleStream
        settings.mode = TestMode.PerformanceOnly
        settings.min_query_count = 2048
        settings.min_duration_ms = 10000

        qsl = ConstructQSL(len(dataset), min(2048, len(dataset)), dataset.load_samples, dataset.unload_samples)
        sut = ConstructSUT(sut_instance.issue_queries, sut_instance.flush_queries)

        log.info("Starting MLPerf performance test...")
        StartTest(sut, qsl, settings)
        log.info("MLPerf test finished.")
        DestroySUT(sut)
        DestroyQSL(qsl)

        log.info("Processing and saving performance results...")
        perf_stats = sut_instance.get_performance_stats()
        with open("performance_stats.json", 'w') as f:
            json.dump(perf_stats, f, indent=4)
        log.info(f"Results successfully saved to: {Path.cwd().resolve()}")

    finally:
        log.info(f"Restoring original working directory: {original_wd}")
        os.chdir(original_wd)
        log.info("Benchmark complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark for ResNet50.")

    parser.add_argument("--image_dir", type=Path, required=True, help="Path to ImageNet validation images.")
    parser.add_argument("--onnx_model_path", type=Path, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Directory to save logs and results.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to use. Default is all.")
    
    provider_group = parser.add_mutually_exclusive_group(required=True)
    provider_group.add_argument("--cpu", action="store_true", help="Use CPUExecutionProvider.")
    provider_group.add_argument("--gpu", action="store_true", help="Use DmlExecutionProvider")
    provider_group.add_argument("--npu", action="store_true", help="Use VitisAIExecutionProvider")

    args = parser.parse_args()

    args.image_dir = args.image_dir.resolve()
    args.onnx_model_path = args.onnx_model_path.resolve()
    args.results_dir = args.results_dir.resolve()

    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: Image directory not found: {args.image_dir.resolve()}")
    if not args.onnx_model_path.is_file():
        sys.exit(f"ERROR: ONNX model not found: {args.onnx_model_path.resolve()}")
    
    log = utils.setup_logging(args.results_dir)

    main(args)