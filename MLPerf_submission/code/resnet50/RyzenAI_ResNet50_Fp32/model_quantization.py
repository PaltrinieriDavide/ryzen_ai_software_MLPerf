import os
import argparse
import onnx
import time
import numpy as np
from PIL import Image
from pathlib import Path
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx import ModelQuantizer, PowerOfTwoMethod
from quark.onnx.quantization.config import Config, get_default_config

def _preprocess_images(images_folder: str,
                       height: int,
                       width: int,
                       size_limit=0,
                       batch_size=100):
    image_path = os.listdir(images_folder)
    image_names = []
    for file_name in os.listdir(images_folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            image_names.append(os.path.join(images_folder, file_name))
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    batch_data = []
    for index, image_name in enumerate(batch_filenames):
        image_filepath =  image_name
        pillow_img = Image.new("RGB", (width, height))
        pillow_img.paste(Image.open(image_filepath).resize((width, height)))
        image_array = np.array(pillow_img) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        image_array = (image_array - mean)
        std = np.array([0.229, 0.224, 0.225])
        nchw_data = image_array / std
        nchw_data = nchw_data.transpose((2, 0, 1))
        nchw_data = np.expand_dims(nchw_data, axis=0)
        nchw_data = nchw_data.astype(np.float32)
        unconcatenated_batch_data.append(nchw_data)

        if (index + 1) % batch_size == 0:
            one_batch_data = np.concatenate(unconcatenated_batch_data,
                                               axis=0)
            unconcatenated_batch_data.clear()
            batch_data.append(one_batch_data)

    return batch_data

class ImageDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int, batch_size: int):
        self.enum_data = None
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        (_, _, height, width) = session.get_inputs()[0].shape
        self.nhwc_data_list = _preprocess_images(calibration_image_folder,
                                                 height, width, data_size, batch_size)
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{
                self.input_name: nhwc_data
            } for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None

def benchmark_model(session, runs=100):
    input_shape = session.get_inputs()[0].shape
    input_shape = tuple(1 if isinstance(dim, str) else dim for dim in input_shape)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    start_time = time.time()
    for _ in range(runs):
        outputs = session.run(None, {session.get_inputs()[0].name: input_data})
    end_time = time.time()
    avg_time = (end_time - start_time) / runs
    print('Average inference time over {} runs: {} ms'.format(runs, avg_time * 1000))

def main(args):
    input_model_path = args.model_input
    output_model_path = args.model_output
    calibration_dataset_path = args.calib_data

    from quark.onnx import ModelQuantizer, VitisQuantFormat, VitisQuantType
    from quark.onnx.quantization.config import QuantizationConfig
    num_calib_data = 100
    calibration_dataset = ImageDataReader(calibration_dataset_path, input_model_path, data_size=num_calib_data, batch_size=1)

    if args.quantize == 'bf16':
        quant_config = get_default_config("BF16")
        config = Config(global_quant_config=quant_config)
        print("The configuration of the quantization is {}".format(config))
    elif args.quantize == 'int8':
        quant_config = get_default_config("XINT8")
        config = Config(global_quant_config=quant_config)
        print("The configuration of the quantization is {}".format(config))
    else:
        print("Invalid quantization option. Please choose from 'BF16' or 'INT8.")

    if args.quantize:
        quantizer = ModelQuantizer(config)
        quant_model = quantizer.quantize_model(model_input=input_model_path,
                                               model_output=output_model_path,
                                               calibration_data_reader=calibration_dataset)
        print("Model Size:")
        print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))
        print("Quantized {} quantized model size: {:.2f} MB".format(args.quantize, os.path.getsize(output_model_path)/(1024 * 1024)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize and evaluate ONNX models.")
    parser.add_argument('--model_input', type=str, default='models\\resnet50_fp32.onnx', help='Path to the input ONNX model.')
    parser.add_argument('--model_output', type=str, default='quantized_models\\resnet50_quant_bf16.onnx', help='Path to save the quantized ONNX model.')
    parser.add_argument('--calib_data', type=str, default='calib_data_imagenet', help='Path to the calibration dataset.')
    parser.add_argument('--quantize', type=str, choices=['bf16', 'int8'], required=False, help='options to quantize the model.')

    args = parser.parse_args()
    main(args)