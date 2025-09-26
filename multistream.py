"""
MLPerf Inference Benchmark for ResNet50 using ONNX Runtime

This script runs the MLPerf MultiStream scenario to measure the throughput
of an ONNX model, simulating a workload with multiple, simultaneous nference streams.

python inference-master/multistream.py `
    --image_dir "dataset/ILSVRC2012_img_val" `
    --onnx_model_path "pipeline_scripts\quantized_models\resnet50_quant_int8.onnx" `
    --results_dir "inference-master/results/multiStream" `
    --num_images 10 `
    --npu

Note: The execution provider flag is required. 
Replace --cpu with --gpu or --npu to target different hardware.
"""
import os
import sys
import argparse
import logging
import random
import threading
import queue
import time
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
    LogSettings, LoggingMode
)

class SUT:
    """
    Defines the System Under Test (SUT) for the MLPerf MultiStream scenario.
    It uses a background worker thread and a queue to process inference requests
    asynchronously, one by one.
    """
    def __init__(self, onnx_model_path: Path, dataset: utils.ImagenetDataset, provider: str):
        log.info("Initializing SUT for MultiStream...")
        self.dataset = dataset
        self.work_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.is_done = False

        log.info(f"Loading ONNX model from: {onnx_model_path}")
        self.session = ort.InferenceSession(str(onnx_model_path), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")

        self.worker_thread.start()
        log.info("SUT Initialized and worker thread started.")

    def _worker_loop(self):
        """The background thread that performs inference on one sample at a time."""
        log.info("Worker thread loop started.")
        while not self.is_done:
            try:
                # Get a single query from the queue with a timeout
                query = self.work_queue.get(timeout=0.01)

                # Retrieve the pre-processed tensor (already has batch dimension of 1)
                tensor = self.dataset.get_sample(query.index)

                # Run inference on the single sample
                _ = self.session.run([self.output_name], {self.input_name: tensor})

                # Create and send the response for the completed query
                response = QuerySampleResponse(query.id, 0, 0)
                QuerySamplesComplete([response])

            except queue.Empty:
                # This is expected when the queue is idle; allows checking `is_done`
                continue
        log.info("Worker thread loop finished.")

    def issue_queries(self, query_samples: List[QuerySample]) -> None:
        """Called by the loadgen to send new queries to the SUT."""
        for query in query_samples:
            self.work_queue.put(query)

    def flush_queries(self) -> None:
        """Called by the loadgen at the end of the test to process remaining queries."""
        log.info("Flushing queries...")
        while not self.work_queue.empty():
            time.sleep(0.01) # Give worker time to process
        log.info("Query flush complete.")

    def stop(self) -> None:
        """Stops the worker thread."""
        log.info("Stopping SUT worker thread...")
        self.is_done = True
        self.worker_thread.join(timeout=5)
        if self.worker_thread.is_alive():
            log.warning("Worker thread did not stop gracefully.")
        else:
            log.info("SUT worker thread stopped.")


def main(args: argparse.Namespace) -> None:
    """Main function to orchestrate the benchmark run."""

    original_wd = Path.cwd()
    log.info(f"Original working directory: {original_wd}")

    # MLPerf loadgen requires the CWD to be the results directory
    args.results_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.results_dir)
    log.info(f"Temporarily changed working directory to: {Path.cwd()}")

    sut_instance = None
    try:
        log.info("--- MLPerf Inference Benchmark - MultiStream ---")
        log.info("Configuration:")
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

        log.info(f"Searching for images in: {args.image_dir}")
        image_paths = sorted(list(args.image_dir.glob("*.JPEG")))
        if not image_paths:
             image_paths = sorted(list(args.image_dir.glob("*.[jJ][pP][gG]")) + list(args.image_dir.glob("*.[jJ][pP][eE][gG]")) + list(args.image_dir.glob("*.[pP][nN][gG]")))

        if not image_paths:
            log.error(f"No images found in {args.image_dir}. Check the path and file extensions.")
            sys.exit(1)
        log.info(f"Found {len(image_paths)} images.")

        # Applica il campionamento se richiesto
        if args.num_images and args.num_images < len(image_paths):
            log.info(f"Using a random subset of {args.num_images} images from {len(image_paths)} total.")
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

        log_settings = LogSettings()
        log_settings.log_output.outdir = "."
        log_settings.log_output.copy_summary_to_stdout = True
        log_settings.log_mode = LoggingMode.AsyncPoll

        settings = TestSettings()
        settings.scenario = TestScenario.MultiStream
        settings.mode = TestMode.PerformanceOnly
        settings.min_duration_ms = 1000
        settings.min_query_count = 300

        qsl = ConstructQSL(len(dataset), min(2048, len(dataset)), dataset.load_samples, dataset.unload_samples)
        sut = ConstructSUT(sut_instance.issue_queries, sut_instance.flush_queries)

        log.info("Starting MLPerf performance test...")
        StartTest(sut, qsl, settings)
        log.info("MLPerf test finished.")

        DestroySUT(sut)
        DestroyQSL(qsl)

        log.info(f"Benchmark complete. Results are in: {args.results_dir.resolve()}")

    except Exception as e:
        log.error(f"An unexpected error occurred: {e}", exc_info=True)

    finally:
        if sut_instance:
            sut_instance.stop()
        log.info(f"Restoring original working directory: {original_wd}")
        os.chdir(original_wd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark for ResNet50 - MultiStream.")

    parser.add_argument("--image_dir", type=Path, required=True, help="Path to ImageNet validation images.")
    parser.add_argument("--onnx_model_path", type=Path, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--results_dir", type=Path, default=Path("results_multistream"), help="Directory to save logs and results.")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images to use from the dataset. Default is all.")

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