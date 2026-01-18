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


class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, input_name: str):
        self.enum_data = None

        self.input_name = input_name

        self.data_list = self._preprocess_images(
                calibration_image_folder)

    def _preprocess_images(self, image_folder: str):
        data_list = []
        img_names = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
        for name in img_names:
            input_image = cv2.imread(os.path.join(image_folder, name))
            # Resize the input image. Because the size of Resnet50 is 224.
            input_image = cv2.resize(input_image, (224, 224))
            input_data = np.array(input_image).astype(np.float32)
            # Custom Pre-Process
            input_data = input_data.transpose(2, 0, 1)
            input_size = input_data.shape
            if input_size[1] > input_size[2]:
                input_data = input_data.transpose(0, 2, 1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data / 255.0
            data_list.append(input_data)
        print(f" Number of preprocessed images: {len(data_list)}")
        return data_list

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.data_list])
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None


def main(args):
    # Instantiate the calibration data reader
    calib_data_reader = RandomCalibDataReader(args.input) #random data
    #calib_data = "./data"
    #calib_data_reader = ImageDataReader(calib_data_path, arg.input)

    # Set up quantization with a specified configuration
    # For example, use "XINT8" for Ryzen AI INT8 quantization
    xint8_config = get_default_config("XINT8")
    quantization_config = Config(global_quant_config=xint8_config )
    quantizer = ModelQuantizer(quantization_config)

    # Quantize the ONNX model and save to specified path
    quantizer.quantize_model(args.input, args.output, calib_data_reader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=str, help ="input_file")
    parser.add_argument('--output', type=str, help ="output_file")
    args = parser.parse_args()

    main(args)