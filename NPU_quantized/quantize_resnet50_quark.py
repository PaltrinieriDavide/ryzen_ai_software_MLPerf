import os
import argparse
import onnx
import time
import numpy as np
from pathlib import Path # For cache directory in benchmarking

# --- Quark Imports (based on the example you found) ---
from quark.onnx import ModelQuantizer
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx.calibrate import CalibrationDataReader # Assuming this is where Quark expects it

# --- Configuration (Adjust these paths) ---
DEFAULT_INPUT_MODEL_PATH = "resnet50_fp32.onnx"
DEFAULT_OUTPUT_MODEL_PATH_INT8 = "resnet50_quark_int8.onnx" # Specific name for INT8
# This should point to your directory with preprocessed .npy files
DEFAULT_CALIB_DATA_DIR = "preprocessed_imagenet_calib_data"

# --- Helper Function to Get Model Input Name (from previous attempt) ---
def get_onnx_model_input_name(model_path):
    model = onnx.load(model_path)
    if not model.graph.input:
        raise ValueError("Model graph has no inputs!")
    return model.graph.input[0].name

# --- Custom NPY Calibration Data Reader ---
class NpyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_data_dir: str, model_input_name: str):
        super().__init__() # Initialize the base class
        self.model_input_name = model_input_name
        self.calibration_files = [
            os.path.join(calibration_data_dir, f)
            for f in os.listdir(calibration_data_dir)
            if f.endswith(".npy")
        ]
        self.calibration_files.sort()
        self.file_iterator = iter(self.calibration_files)

        if not self.calibration_files:
            raise ValueError(f"No .npy files found in {calibration_data_dir}")
        print(f"[NpyCalibrationDataReader] Found {len(self.calibration_files)} calibration files in {calibration_data_dir}")

    def get_next(self) -> dict:
        try:
            next_file = next(self.file_iterator)
            data_array = np.load(next_file)
            if data_array.dtype != np.float32:
                data_array = data_array.astype(np.float32)
            return {self.model_input_name: data_array}
        except StopIteration:
            return None

    def rewind(self):
        self.file_iterator = iter(self.calibration_files)

# --- Main Quantization Function ---
def quantize_model_main(args):
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_dataset_path = args.calib_data
    quant_type = args.quant_type.upper() # Ensure uppercase for "XINT8" or "BF16"

    if not os.path.exists(input_model_path):
        print(f"Error: Input model '{input_model_path}' not found.")
        return
    if not os.path.exists(calibration_dataset_path):
        print(f"Error: Calibration data directory '{calibration_dataset_path}' not found.")
        print(f"Please run 'prepare_calibration_data.py' and ensure it creates this directory containing .npy files.")
        return

    print(f"--- Starting ONNX Model Quantization with AMD Quark ({quant_type}) ---")
    print(f"Input FP32 model: {input_model_path}")
    print(f"Output quantized model: {output_model_path}")
    print(f"Calibration data directory: {calibration_dataset_path}")

    # 1. Get Model Input Name
    try:
        model_input_name = get_onnx_model_input_name(input_model_path)
        print(f"Determined model input name: '{model_input_name}'")
    except Exception as e:
        print(f"Error getting model input name: {e}")
        return

    # 2. Define the calibration data reader
    try:
        calibration_reader = NpyCalibrationDataReader(calibration_dataset_path, model_input_name)
    except Exception as e:
        print(f"Error creating calibration data reader: {e}")
        return

    # 3. Get quantization configuration
    if quant_type == 'INT8':
        # For Ryzen AI NPU, "XINT8" is typical. "DefaultINT8" might also work.
        # Check Quark/Ryzen AI docs for the best profile name.
        quant_profile_name = "XINT8" # Or "DefaultINT8" or other Vitis/Ryzen specific
        global_quant_config = get_default_config(quant_profile_name)
        # You can customize global_quant_config here if needed
        # e.g., global_quant_config.set_option("ActivationSymmetric", True)
        #       global_quant_config.set_option("WeightSymmetric", True)
        #       global_quant_config.set_option("OpTypesToQuantize", ["Conv", "MatMul", "Add", ...]) # If not default
        print(f"Using quantization profile: {quant_profile_name}")
    elif quant_type == 'BF16':
        global_quant_config = get_default_config("BF16")
        # The example had: global_quant_config.extra_options["BF16QDQToCast"] = True
        # This might require a different way to set extra_options or might be default.
        # For Quark, check how to set specific options if needed.
        # global_quant_config.set_option("BF16QDQToCast", True) # Hypothetical
        print(f"Using quantization profile: BF16")
    else:
        print(f"Error: Invalid quantization type '{args.quant_type}'. Choose from 'int8' or 'bf16'.")
        return

    # Wrap the global config in the main Config object
    config = Config(global_quant_config=global_quant_config)
    # print(f"Quantization Config: {config}") # Quark's Config object might not have a nice __str__

    # 4. Create an ONNX Quantizer and quantize
    try:
        quantizer = ModelQuantizer(config)
        print("Quantizing model...")
        start_time = time.time()
        quantizer.quantize_model(model_input=input_model_path,
                                 model_output=output_model_path,
                                 calibration_data_reader=calibration_reader)
        end_time = time.time()
        print(f"Quantization completed in {end_time - start_time:.2f} seconds.")
        print(f"Quantized model saved to: {output_model_path}")

        print("\nModel Sizes:")
        print(f"  FP32 model ('{input_model_path}'): {os.path.getsize(input_model_path)/(1024*1024):.2f} MB")
        print(f"  {quant_type} quantized model ('{output_model_path}'): {os.path.getsize(output_model_path)/(1024*1024):.2f} MB")

    except Exception as e:
        print(f"Error during model quantization: {e}")
        import traceback
        traceback.print_exc()

# --- Benchmarking (Optional, adapted from the example) ---
# You'll need onnxruntime and potentially onnxruntime-amd for VitisAIExecutionProvider
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("Warning: onnxruntime not found. Benchmarking will be skipped.")

def benchmark_model_on_provider(model_path, provider_name, runs=100):
    if not ORT_AVAILABLE: return
    if not os.path.exists(model_path):
        print(f"Benchmark: Model {model_path} not found. Skipping.")
        return

    print(f"\n--- Benchmarking '{model_path}' on {provider_name} ---")
    try:
        sess_options = ort.SessionOptions()
        providers = [provider_name]
        provider_options_list = None

        if provider_name == 'VitisAIExecutionProvider':
            # VitisAIExecutionProvider often requires specific provider options
            # like a config file, cache directory etc.
            # The example used:
            cache_dir = Path(__file__).parent.resolve() / "onnx_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # You might need a 'vaip_config.json' or similar in your project directory
            # or specify its path correctly.
            # For Ryzen AI, the 'config_file' might point to something specific
            # installed by Ryze AI Studio, or it might not be needed if onnxruntime-amd handles it.
            provider_options_list = [{'config_file': 'vaip_config.json', # CHECK IF THIS FILE IS NEEDED/EXISTS
                                     'cacheDir': str(cache_dir),
                                     'cacheKey': Path(model_path).stem + '_cache'}]
            print(f"Using VitisAIExecutionProvider options: {provider_options_list}")

            # For Ryzen AI, it might also require registering custom ops from onnxruntime-amd
            # from quark.onnx import get_library_path
            # custom_op_lib_path = get_library_path() # device='ipu' or 'npu' might be needed
            # if custom_op_lib_path and os.path.exists(custom_op_lib_path):
            #     sess_options.register_custom_ops_library(custom_op_lib_path)
            #     print(f"Registered custom ops library: {custom_op_lib_path}")
            # else:
            #     print(f"Warning: Custom OP library not found at {custom_op_lib_path}")


        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers, provider_options=provider_options_list)
        print(f"Session created with provider: {session.get_providers()}")

        input_desc = session.get_inputs()[0]
        input_name = input_desc.name
        input_shape = tuple(1 if isinstance(dim, str) or dim is None else dim for dim in input_desc.shape) # Handle dynamic axes
        input_data = np.random.rand(*input_shape).astype(np.float32)

        # Warm-up
        _ = session.run(None, {input_name: input_data})

        start_time_total = time.perf_counter()
        for _ in range(runs):
            session.run(None, {input_name: input_data})
        end_time_total = time.perf_counter()

        avg_time_ms = ((end_time_total - start_time_total) / runs) * 1000
        print(f"Average inference time over {runs} runs: {avg_time_ms:.2f} ms")
        print(f"FPS: {1000 / avg_time_ms:.2f}")

    except Exception as e:
        print(f"Error during benchmarking on {provider_name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize ONNX ResNet50 model using AMD Quark.")
    parser.add_argument('--model_input', type=str, default=DEFAULT_INPUT_MODEL_PATH,
                        help='Path to the input FP32 ONNX model.')
    parser.add_argument('--model_output', type=str, default=None, # Set dynamically
                        help='Path to save the quantized ONNX model.')
    parser.add_argument('--calib_data', type=str, default=DEFAULT_CALIB_DATA_DIR,
                        help='Path to the directory containing preprocessed .npy calibration data.')
    parser.add_argument('--quant_type', type=str, choices=['int8', 'bf16'], default='int8',
                        help='Type of quantization to perform (int8 or bf16). Default: int8.')
    parser.add_argument('--benchmark', action='store_true',
                        help='Flag to benchmark the FP32 and quantized models.')

    args = parser.parse_args()

    # Set default output model path based on quant_type if not provided
    if args.model_output is None:
        if args.quant_type.lower() == 'int8':
            args.model_output = DEFAULT_OUTPUT_MODEL_PATH_INT8
        elif args.quant_type.lower() == 'bf16':
            # Define a default for BF16 if you plan to use it
            args.model_output = "resnet50_quark_bf16.onnx"
        else:
            args.model_output = "resnet50_quark_quantized.onnx" # Generic fallback

    # --- Run Quantization ---
    quantize_model_main(args)

    # --- Run Benchmarking (if requested and quantized model was created) ---
    if args.benchmark and os.path.exists(args.model_output):
        # Benchmark FP32 on CPU
        benchmark_model_on_provider(args.model_input, 'CPUExecutionProvider')

        # Benchmark Quantized on CPU
        benchmark_model_on_provider(args.model_output, 'CPUExecutionProvider')

        # Benchmark Quantized on VitisAIExecutionProvider (NPU)
        # Ensure onnxruntime-amd is installed and VitisAIExecutionProvider is available
        if ORT_AVAILABLE and 'VitisAIExecutionProvider' in ort.get_available_providers():
            benchmark_model_on_provider(args.model_output, 'VitisAIExecutionProvider')
        elif ORT_AVAILABLE:
            print("\nVitisAIExecutionProvider not found in available ONNX Runtime providers.")
            print(f"Available providers: {ort.get_available_providers()}")
            print("Skipping NPU benchmark. Ensure 'onnxruntime-amd' is installed for Ryzen AI NPU support.")
        else:
            print("\nONNX Runtime not available. Skipping all benchmarks.")