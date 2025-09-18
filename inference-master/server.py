"""
MLPerf Inference Benchmark for ResNet50 - Server Scenario

This script runs the MLPerf Server scenario, designed to measure the
performance of a system under a load of requests arriving according to a
Poisson distribution, simulating a typical online service.

The implementation uses a design based on a single worker thread and a queue
to process inference requests asynchronously.

python inference-master/server.py `
>>     --image_dir "dataset/ILSVRC2012_img_val" `
>>     --map_file "dataset/val_map.txt" `
>>     --onnx_model_path "pipeline_scripts\quantized_models\resnet50_quant_int8.onnx" `
>>     --results_dir "inference-master/results/server" `
>>     --num_images 10000 `
>>     --target_qps 100 `
>>     --npu 
"""
import os
import sys
import time
import random
import argparse
import logging
import threading
import queue
from pathlib import Path
from typing import List

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

# --- System Under Test (SUT) Definition ---

class SUT_Server:
    """
    Implements the System Under Test (SUT) for the Server scenario.
    It uses a dedicated worker thread that fetches requests from a queue
    to perform inferences decoupled from the arrival of queries.
    """
    def __init__(self, model_path: Path, dataset: utils.ImagenetDataset, provider: str):
        self.dataset = dataset
        self.work_queue = queue.Queue()
        self.poison_pill = object()
        self.worker_thread = threading.Thread(target=self._worker_loop)

        self.session = ort.InferenceSession(str(model_path), providers=[provider])
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        log.info(f"Model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
        
        self.worker_thread.start()
        log.info("SUT initialized and worker thread started.")

    def _worker_loop(self):
        """
        Main loop of the worker thread.
        Waits for requests on the queue, performs inference, and completes the query.
        """
        while True:
            work_items = [self.work_queue.get()]

            if work_items[0] is self.poison_pill:
                break

            if not work_items:
                continue

            #tensors = [self.dataset.get_sample(q.index) for q in work_items]
            tensors = [self.dataset.get_sample(q.index)[0] for q in work_items]
            batch_tensor = np.concatenate(tensors, axis=0)

            self.session.run([self.output_name], {self.input_name: batch_tensor})

            for query in work_items:
                response = QuerySampleResponse(query.id, 0, 0)
                QuerySamplesComplete([response])

    def issue_queries(self, query_samples: List[QuerySample]):
        """
        Method called by MLPerf LoadGen to submit new queries.
        Adds the queries to the work queue.
        """
        for query in query_samples:
            self.work_queue.put(query)

    def flush_queries(self):
        """Function required by MLPerf, not necessary in this design."""
        pass

    def stop(self):
        """Sends the stop signal to the worker and waits for its termination."""
        log.info("Sending stop signal to worker thread...")
        self.work_queue.put(self.poison_pill)
        self.worker_thread.join()
        log.info("Worker thread terminated successfully.")


def main(args: argparse.Namespace):
    original_wd = Path.cwd()

    os.chdir(args.results_dir)

    sut_instance = None
    try:
        log.info("--- MLPerf Inference Benchmark - Server ---")
        log.info("Configuration:")
        for arg, value in vars(args).items():
            log.info(f"  - {arg}: {value}")
        
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

        with open(args.map_file) as f:
            entries = [line.strip().split() for line in f]

        if args.num_images and args.num_images < len(entries):
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

        # --- SUT Initialization ---
        sut_instance = SUT_Server(args.onnx_model_path, dataset, provider=selected_provider)

        # --- MLPerf LoadGen Configuration ---
        log_settings = LogSettings()
        log_settings.log_output.outdir = "."
        log_settings.log_output.copy_summary_to_stdout = True

        settings = TestSettings()
        settings.scenario = TestScenario.Server
        settings.mode = TestMode.PerformanceOnly
        settings.server_target_qps = args.target_qps
        
        settings.min_duration_ms = 10000
        settings.min_query_count = 300


        log.info(f"Test settings: Target QPS={settings.server_target_qps}, "
                 f"min_duration_ms={settings.min_duration_ms}ms, "
                 f"min_query_count={settings.min_query_count}")

        qsl = ConstructQSL(
            len(dataset),
            len(dataset),
            dataset.load_samples,
            dataset.unload_samples
        )

        sut = ConstructSUT(sut_instance.issue_queries, sut_instance.flush_queries)

        log.info("Starting MLPerf Test - Server")
        StartTest(sut, qsl, settings)
        log.info("MLPerf Test finished - Server")

        # --- MLPerf Resource Cleanup ---
        DestroyQSL(qsl)
        DestroySUT(sut)

    except Exception as e:
        log.error(f"An error occurred during the test: {e}", exc_info=True)
    finally:
        # Ensure the SUT is stopped and the directory is restored
        if sut_instance:
            log.info("Stopping the SUT...")
            sut_instance.stop()
        
        log.info(f"Restoring original working directory: {original_wd}")
        os.chdir(original_wd)
        log.info("Benchmark complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLPerf Inference Benchmark for ResNet50 - Server Scenario.")
    
    parser.add_argument("--image_dir", type=Path, required=True, help="Path to the ImageNet validation images directory.")
    parser.add_argument("--map_file", type=Path, required=True, help="Path to the ImageNet 'val_map.txt' file.")
    parser.add_argument("--onnx_model_path", type=Path, required=True, help="Path to the ONNX model file.")
    parser.add_argument("--results_dir", type=Path, default=Path("results/server"), help="Directory to save logs and results.")
    parser.add_argument("--num_images", type=int, default=1000, help="Number of images to use for the test. Default: 1000.")
    parser.add_argument("--target_qps", type=int, default=100, help="Target Queries Per Second for the Server scenario.")

    provider_group = parser.add_mutually_exclusive_group(required=True)
    provider_group.add_argument("--cpu", action="store_true", help="Use CPUExecutionProvider.")
    provider_group.add_argument("--gpu", action="store_true", help="Use DmlExecutionProvider")
    provider_group.add_argument("--npu", action="store_true", help="Use VitisAIExecutionProvider")

    args = parser.parse_args()

    args.image_dir = args.image_dir.resolve()
    args.map_file = args.map_file.resolve()
    args.onnx_model_path = args.onnx_model_path.resolve()
    args.results_dir = args.results_dir.resolve()

    if not args.image_dir.is_dir():
        sys.exit(f"ERROR: Image directory not found: {args.image_dir}")
    if not args.map_file.is_file():
        sys.exit(f"ERROR: Map file not found: {args.map_file}")
    if not args.onnx_model_path.is_file():
        sys.exit(f"ERROR: ONNX model not found: {args.onnx_model_path}")
    
    log = utils.setup_logging(args.results_dir)

    main(args)