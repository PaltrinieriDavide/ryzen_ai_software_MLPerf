# MLPerf Inference & Ryzen AI: Execution Guide

This guide provides step-by-step instructions on how to use the provided scripts to quantize the ResNet50 model and run the MLPerf Inference benchmarks across various scenarios.

---

### **1. Model Quantization Pipeline**

The first step is to prepare an optimized model for the NPU. The `run_pipeline.py` script automates the entire process of converting a standard FP32 model into a quantized INT8 ONNX model.

The pipeline executes three main scripts in sequence:

1.  **`export_fp32_resnet50.py`**: Exports a pre-trained ResNet50 model from the PyTorch library into the standard FP32 ONNX format. This is the baseline model for our pipeline.

2.  **`prepare_calibration_data.py`**: Prepares the dataset required for quantization. It takes a small subset of raw JPEG images, applies the same preprocessing used during inference (resizing, cropping, normalization), and saves them as individual `.npy` files.

3.  **`model_quantization.py`**: Performs the actual post-training quantization. It uses the FP32 ONNX model and the preprocessed calibration data to determine the optimal scaling factors for converting model weights and activations to INT8 precision.

#### **How to Run the Pipeline**

Place a small subset of ImageNet images (e.g., 300-500 images) in a dedicated directory for calibration. Then, execute the main pipeline script, pointing it to that directory.

**Command:**
```bash
python pipeline_scripts/run_pipeline.py --image_dir <path_to_your_calibration_images>
```
*Example:*
```bash
python pipeline_scripts/run_pipeline.py --image_dir dataset/imagenet_calib_subset
```
The final quantized model will be created at: `pipeline_scripts/quantized_models/resnet50_quant_int8.onnx`.

---

### **2. MLPerf Benchmark Scenarios**

Once the model is ready, you can run the MLPerf benchmarks. Each scenario is designed to measure a different aspect of system performance and has its own script.

#### **Offline Scenario**

*   **What it does**: Measures the maximum possible throughput of the system. The benchmark provides all data samples to the System Under Test (SUT) at once, simulating a batch processing workload. This script can also be used to test the model's accuracy.

*   **Command**:
    ```bash
    python inference-master/offline.py ^
        --image_dir "dataset/ILSVRC2012_img_val" ^
        --map_file "dataset/val_map.txt" ^
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
        --results_dir "inference-master/results/offline" ^
        --run_performance ^
        --run_accuracy ^
        --npu
    ```

#### **Single-Stream Scenario**

*   **What it does**: Measures the latency of a single inference. The benchmark sends one sample at a time and waits for the response before sending the next, simulating applications where immediate response time is critical.

*   **Command**:
    ```bash
    python inference-master/singlestream.py ^
        --image_dir "dataset/ILSVRC2012_img_val" ^
        --map_file "dataset/val_map.txt" ^
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
        --results_dir "inference-master/results/singleStream" ^
        --npu
    ```

#### **Multi-Stream Scenario**

*   **What it does**: Measures the system's ability to handle multiple inference streams simultaneously. It aims to find the maximum throughput the system can sustain while ensuring all streams are processed.

*   **Command**:
    ```bash
    python inference-master/multistream.py ^
        --image_dir "dataset/ILSVRC2012_img_val" ^
        --map_file "dataset/val_map.txt" ^
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
        --results_dir "inference-master/results/multiStream" ^
        --npu
    ```

#### **Server Scenario**

*   **What it does**: Simulates a real-world online service where inference requests arrive randomly according to a Poisson distribution. The goal is to measure the Queries Per Second (QPS) the system can handle while keeping latency below a specific threshold.

*   **Command**:
    ```bash
    python inference-master/server.py ^
        --image_dir "dataset/ILSVRC2012_img_val" ^
        --map_file "dataset/val_map.txt" ^
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
        --results_dir "inference-master/results/server" ^
        --target_qps 100 ^
        --npu
    ```

---

### **3. NPU Monitoring**

The `monitor_npu.py` script allows you to observe the NPU's status and utilization in real-time while a benchmark is running. It uses the `xrt-smi.exe` command-line tool, which is part of the Ryzen AI software stack.

#### **Configuration (Required)**

Before running the script, you **must** edit it to provide the correct path to `xrt-smi.exe` on your system.

1.  Open the file `monitor_scripts/monitor_npu.py`.
2.  Find the `XRT_SMI_PATH` variable and replace the placeholder path with your actual path.

#### **How to Run the Monitor**

Open a new, separate terminal and run the following command. The monitor will display updated NPU statistics on the screen and save a detailed log to a file.

*   **Command**:
    ```bash
    python monitor_scripts/monitor_npu.py --interval 5 --log-file npu_monitor.log
    ```
    *   `--interval 5`: Refreshes the NPU status every 5 seconds.
    *   `--log-file`: Saves all historical data to `npu_monitor.log`.