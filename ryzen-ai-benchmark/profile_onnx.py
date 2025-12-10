
import argparse
import numpy as np
import onnxruntime
import onnx_tool
from onnx_tool.utils import NODE_REGISTRY
from onnx_tool.node import List, Tensor, PWNode


@NODE_REGISTRY.register()
class DequantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[1].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [0, 0]

@NODE_REGISTRY.register()
class QuantizeLinearNode(PWNode):
    def __init__(self, node_proto):
        super().__init__(node_proto)

    def shape_infer(self, intensors: List[Tensor], outtensors: List[Tensor]):
        outtensors[0].update_shape(intensors[0].get_shape())
        outtensors[0].update_dtype(intensors[2].dtype)

    def profile(self, intensors: List[Tensor], outtensors: List[Tensor]):
        return [0, 0]


def main(args):
    session = onnxruntime.InferenceSession(args.input)

    inputs = {}
    for input in session.get_inputs():
        print(f"Input Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
        shape = [args.batch_size if isinstance(s, str) else s for s in input.shape]
        print(shape)
        inputs[input.name] = np.random.rand(*shape).astype(np.float32)
    
    m = onnx_tool.Model(args.input)
    m.graph.graph_reorder_nodes()
    m.graph.shape_infer(inputs)  # update tensor shapes with new input tensor
    m.graph.profile()
    m.graph.print_node_map(metric="FLOPs")
    m.graph.print_node_map(args.input + ".txt", metric="FLOPs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input')
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    main(args)