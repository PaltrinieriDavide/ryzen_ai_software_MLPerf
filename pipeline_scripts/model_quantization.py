import os
import argparse
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config

class NpyDataReader(CalibrationDataReader):
    def __init__(self, calibration_data_folder: str, model_path: str):
        self.model_path = model_path
        session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name
        
        self.data_files = [str(p) for p in Path(calibration_data_folder).glob("*.npy")]
        if not self.data_files:
            raise ValueError(f"No .npy files found in {calibration_data_folder}")
            
        print(f"Found {len(self.data_files)} calibration files in {calibration_data_folder}")
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_files)
        
        filepath = next(self.enum_data, None)
        
        if filepath:
            data = np.load(filepath)
            return {self.input_name: data}
        else:
            return None

    def rewind(self):
        self.enum_data = None

def main(args):
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_dataset_path = args.calib_data

    calibration_dataset = NpyDataReader(calibration_dataset_path, input_model_path)

    if args.quantize == 'bf16':
        quant_config = get_default_config("BF16")
    elif args.quantize == 'int8':
        quant_config = get_default_config("XINT8")
    else:
        print("Invalid quantization option. Please choose from 'bf16' or 'int8'.")
        return

    config = Config(global_quant_config=quant_config)
    print("The configuration of the quantization is {}".format(config))

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=calibration_dataset
    )
    
    print("\nModel Size:")
    print(f"Float32 model size: {os.path.getsize(input_model_path)/(1024*1024):.2f} MB")
    print(f"Quantized {args.quantize} model size: {os.path.getsize(output_model_path)/(1024*1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX models.")
    parser.add_argument('--model_input', type=str, required=True, help='Path to the input ONNX model.')
    parser.add_argument('--model_output', type=str, required=True, help='Path to save the quantized ONNX model.')
    parser.add_argument('--calib_data', type=str, required=True, help='Path to the calibration dataset (.npy files).')
    parser.add_argument('--quantize', type=str, choices=['bf16', 'int8'], required=True, help='Options to quantize the model.')
    args = parser.parse_args()
    main(args)