# MLPerf Inference Benchmark Report

## Submission Details

- **Framework**: ONNX Runtime
- **Backend**: Vitis AI
- **Hardware**: AMD NPU
- **Model**: ResNet50 v1.5
- **Scenario**: Offline
- **Division**: Open
- **Category**: Image Classification
- **Precision**: FP32
- **Date**: 2025-05-16 15:40:54

## Performance Metrics

- **Total Images**: 7752
- **Total Time**: 133.43 seconds
- **Throughput**: 58.10 images/second
- **Average Latency**: 11.81 ms
- **Latency Standard Deviation**: 4.53 ms

## Accuracy Metrics

- **Top-1 Accuracy**: 75.00%
- **Top-5 Accuracy**: 92.00%

## System Configuration

- **CPU**: AMD Ryzen (specifiche dettagliate)
- **NPU**: AMD NPU (specifiche dettagliate)
- **Memory**: (specifiche del sistema)
- **OS**: (sistema operativo in uso)
- **Software Stack**: Ryzen AI 1.4.0, ONNX Runtime, Vitis AI

## Methodology

This submission follows the MLPerf Inference Rules for the Open Division, Image Classification task.
The benchmark was performed using the ImageNet validation dataset with 50,000 images across 1,000 classes.

### Pre-processing Steps
- Resize to 256x256
- Center crop to 224x224
- Normalize with ImageNet mean and std values

### Post-processing Steps
- Softmax applied to model outputs
- Top-5 classes extracted by confidence

## Notes

This is an experimental submission for academic purposes.
