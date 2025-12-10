import argparse
import numpy as np
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader
from quark.onnx.quantization.config import Config, get_default_config
from quark.onnx import ModelQuantizer

# Define calibration data reader for static quantization
class RandomCalibDataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.data_iter = None
        self.inputs = {}
        session = onnxruntime.InferenceSession(model_path)
        for input in session.get_inputs():
            print(f"Input Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
            shape = [args.batch_size if isinstance(s, str) else s for s in input.shape]
            print(shape)
            self.inputs[input.name] = np.random.rand(*shape).astype(np.float32)

    def get_next(self):
        if self.data_iter is None:
            self.data_iter = iter([self.inputs])
        return next(self.data_iter, None)

def main(args):
    # Instantiate the calibration data reader
    calib_data_reader = RandomCalibDataReader(args.input)

    # Set up quantization with a specified configuration
    # For example, use "XINT8" for Ryzen AI INT8 quantization
    xint8_config = get_default_config("XINT8")
    quantization_config = Config(global_quant_config=xint8_config )
    quantizer = ModelQuantizer(quantization_config)

    # Quantize the ONNX model and save to specified path
    quantizer.quantize_model(args.input, args.output, calib_data_reader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()

    main(args)