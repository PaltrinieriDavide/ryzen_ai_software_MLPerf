# MLPerf Inference & Ryzen AI: Execution Guide

This guide provides step-by-step instructions on how to use the provided scripts to quantize the ResNet50 model and run the MLPerf Inference benchmarks across various scenarios.

---

### **1. Dataset**

For this project we use the ILSVRC2012 (ImageNet) validation dataset, which is the standard benchmark dataset for ResNet50 in MLPerf.
ILSVRC2012 can be downloaded from the [ImageNet web page](https://www.image-net.org/download.php)

Since the focus of this project is performance only, not accuracy, ground-truth labels are not required. The model predictions will be ignored during benchmarking, and the evaluation will only consider throughput and latency metrics.

If you want to skip the quantization step (Step 2), we are providing the ready quantized model that we used in "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx"
Otherwise, in order to proceed with the quantization, we need a calibration dataset, which can be created by directly selecting a small subset (300–500 images) from ILSVRC2012. You can freely create the calibration dataset in the folder dataset/imagent_calib_subset.

### **2. Model Quantization Pipeline**

The first step is to prepare an optimized model for the NPU. The `run_pipeline.py` script automates the entire process of converting a standard FP32 model into a quantized INT8 ONNX model. 

The pipeline executes three main scripts in sequence:

1.  **`export_fp32_resnet50.py`**: Exports a pre-trained ResNet50 model from the PyTorch library into the standard FP32 ONNX format. This is the baseline model for our pipeline.

2.  **`prepare_calibration_data.py`**: Prepares the dataset required for quantization. It takes a small subset of raw JPEG images, applies the same preprocessing used during inference (resizing, cropping, normalization), and saves them as individual `.npy` files.

3.  **`model_quantization.py`**: Performs the actual post-training quantization. It uses the FP32 ONNX model and the preprocessed calibration data to determine the optimal scaling factors for converting model weights and activations to INT8 precision.

#### **How to Run the Pipeline**

Execute the main pipeline script, pointing it to the calibration dataset directory, created in the previous step.

**Command:**
```bash
python pipeline_scripts/run_pipeline.py --image_dir "<path_to_your_calibration_images>"
```
*Example:*
```bash
python pipeline_scripts/run_pipeline.py --image_dir "dataset/imagenet_calib_subset"
```
The final quantized model will be created at: `pipeline_scripts/quantized_models/resnet50_quant_int8_new.onnx`.

---

### **3. MLPerf Benchmark Scenarios**

Once the model is ready, you can run the MLPerf benchmarks. Each scenario is designed to measure a different aspect of system performance and has its own script. The MLPerf generated result file will be called mlperf_log_summary.txt and placed in the results_dir folder.

#### **Offline Scenario**

*   **What it does**: Measures the maximum possible throughput of the system. The benchmark provides all data samples to the System Under Test (SUT) at once, simulating a batch processing workload.

*   **Command**:
    ```bash
    python inference-master/offline.py `
        --image_dir "<path_to_your_dataset>" `
        --onnx_model_path "<path_to_your_quantized_model>" `
        --results_dir "<path_where_to_print_inference_results>" `
        #select the number of images to be evaluated, default is all
        [--num_images 10000] `
        # Choose one of the following: --npu, --gpu, or --cpu
        --npu
    ```
    *Example:*

    ```bash
    python inference-master/offline.py `
        --image_dir "dataset/ILSVRC2012_img_val" `
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" `
        --results_dir "inference-master/results/offline" `
        --num_images 10000 `
        --npu
    ```
#### **Single-Stream Scenario**

*   **What it does**: Measures the latency of a single inference. The benchmark sends one sample at a time and waits for the response before sending the next, simulating applications where immediate response time is critical.

*   **Command**:
    ```bash
    python inference-master/singlestream.py `
        --image_dir "dataset/ILSVRC2012_img_val" `
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" `
        --results_dir "inference-master/results/singleStream" `
        --num_images 10000 `
        --npu
    ```

#### **Multi-Stream Scenario**

*   **What it does**: Measures the system's ability to handle multiple inference streams simultaneously. It aims to find the maximum throughput the system can sustain while ensuring all streams are processed.

*   **Command**:
    ```bash
    python inference-master/multistream.py `
        --image_dir "dataset/ILSVRC2012_img_val" `
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" `
        --results_dir "inference-master/results/multiStream" `
        --num_images 10000 `
        --npu
    ```

#### **Server Scenario**

*   **What it does**: Simulates a real-world online service where inference requests arrive randomly according to a Poisson distribution. The goal is to measure the Queries Per Second (QPS) the system can handle while keeping latency below a specific threshold.

*   **Command**:
    ```bash
    python inference-master/server.py `
        --image_dir "dataset/ILSVRC2012_img_val" `
        --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" `
        --results_dir "inference-master/results/server" `
        --num_images 10000 `
        --target_qps 100 `
        --npu
    ```

---

### **4. NPU Monitoring**

The `monitor_npu.py` script allows you to observe the NPU's status and utilization in real-time while a benchmark is running. It uses the `xrt-smi.exe` command-line tool, which is part of the Ryzen AI software stack.

#### **Configuration (Required)**

Before running the script, you **must** add the absolute path of the folder containing `xrt-smi.exe` to the PATH environment variable:

```powershell
$env:PATH = "<PATH-TO-RYZEN-AI-INSTALL-DIR>;" + $env:PATH
```
*Example:*

```powershell
$env:PATH = "C:\Users\aiene\Downloads\NPU_RAI1.5_280_WHQL\npu_mcdm_stack_prod;" + $env:PATH
```

If you prefer, you can also directly insert the `xrt-smi.exe` absolute path in the XRT_SMI_PATH variable in the code of monitor_scripts/monitor_npu.py at line 13.

#### **How to Run the Monitor**

Run the following command in a terminal. The monitor will display updated NPU statistics on the screen and save a detailed log to a file.

*   **Command**:
```bash
python monitor_scripts/monitor_npu.py --interval 5 --log-file npu_monitor.log
```
*   `--interval 5`: Refreshes the NPU status every 5 seconds.
*   `--log-file`: Saves all historical data to `npu_monitor.log`.

### **5. Power profiling**

We've used AMD uProf (https://www.amd.com/en/developer/uprof.html) to measure socket and cores power.

#### **NPU power mode selection (Optional)**

To change the NPU power operating mode, you can use xrt-smi, first adding to the PATH environment variable the absolute path of the folder containing `xrt-smi.exe` (Ryzen AI installation directory):

```powershell
$env:PATH = "<PATH-TO-RYZEN-AI-INSTALL-DIR>;" + $env:PATH
```
*Example:*

```powershell
$env:PATH = "C:\Users\aiene\Downloads\NPU_RAI1.5_280_WHQL\npu_mcdm_stack_prod;" + $env:PATH
```
And then executing the following command, selecting one of the power modes:

```bash
xrt-smi configure --pmode <default | powersaver | balanced | performance | turbo>
```
*Example:*

```bash
xrt-smi configure --pmode performance
```

#### **How to Run the live profiler**

First you need to add the AMDuProfCLI exe file path to the PATH environment variable:

```powershell
$env:PATH = "<PATH-TO-AMDuProf-bin-FOLDER>;" + $env:PATH
```

*Example:*

```powershell
$env:PATH = "C:\Program Files\AMD\AMDuProf\bin;" + $env:PATH
```
Then execute the command 

```bash
AMDuProfCLI.exe timechart --event power --interval 100 --duration 10 -o "power_measurements/power_profiling"
```

Which prints the detailed power outputs, measured every 100 ms (interval), for 10 seconds (duration), in the file "timechart.csv" in the "power_measurements/power_profiling" directory
To measure the frequency, just substitute "power" in the previous command with "frequency"

