import os
import argparse
import onnx
from onnxsim import simplify

def main(args):
    os.system(f"ssc4onnx --input_onnx_file_path {args.input}")

    model = onnx.load(args.input)
    model = onnx.shape_inference.infer_shapes(model)
    model, _ = simplify(model)
    model, _ = simplify(model)
    model, _ = simplify(model)
    onnx.save(model, args.output)

    os.system(f"ssc4onnx --input_onnx_file_path {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify onnx model')
    parser.add_argument('--input', type=str, help ="input_file")
    parser.add_argument('--output',type=str, help ="output_file")
    args = parser.parse_args()

    main(args)
