# MLPerf Inference & Ryzen AI: Quick Start Guide

### **1. Model Quantization (INT8)**

Generate the quantized INT8 ONNX ResNet50 model required for the benchmarks.

**Command:**
```bash
python pipeline_scripts/run_pipeline.py --image_dir <path_to_calibration_images>
```
The quantized model will be saved to: `pipeline_scripts/quantized_models/resnet50_quant_int8.onnx`.

---

### **2. Running MLPerf Benchmarks**

#### **Offline**
Measures the maximum system throughput using batch processing.

**Command:**
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

#### **Single-Stream**
Measures the latency of a single inference.

**Command:**
```bash
python inference-master/singlestream.py ^
    --image_dir "dataset/ILSVRC2012_img_val" ^
    --map_file "dataset/val_map.txt" ^
    --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
    --results_dir "inference-master/results/singleStream" ^
    --npu
```

#### **Multi-Stream**
Measures throughput with multiple concurrent inference streams.

**Command:**
```bash
python inference-master/multistream.py ^
    --image_dir "dataset/ILSVRC2012_img_val" ^
    --map_file "dataset/val_map.txt" ^
    --onnx_model_path "pipeline_scripts/quantized_models/resnet50_quant_int8.onnx" ^
    --results_dir "inference-master/results/multiStream" ^
    --npu
```

#### **Server**
Simulates an online server and measures queries per second (QPS).

**Command:**
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

Monitor NPU utilization in real-time while the benchmarks are running.

**Configuration (Required):**
Edit the `monitor_scripts/monitor_npu.py` script and set the correct path to `xrt-smi.exe` in the `XRT_SMI_PATH` variable.

**Command (in a separate terminal):**
```bash
python monitor_scripts/monitor_npu.py --interval 5 --log-file npu_monitor.log
```

---

### **4. Configuring NPU Performance Mode**

For optimal and consistent results, set the NPU to its highest performance mode before running any benchmarks.

**Command:**
```bash
"<path_to_xrt-smi>\\xrt-smi.exe" configure --pmode performance
```
*Available modes include: `performance`, `turbo`, `balanced`, `default`.*
