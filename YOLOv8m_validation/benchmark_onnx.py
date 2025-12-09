from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue
from re import X
from threading import Thread

import argparse
import numpy as np
import onnxruntime
from tqdm import tqdm

def create_session(args, num_of_dpu_runners=4, enable_analyzer=False):
    print(f"Load {args.onnx} with {args.onnx_ep} EP")

    sess_options = onnxruntime.SessionOptions()

    if args.cpu_threads > 0:
        sess_options.intra_op_num_threads = args.cpu_threads

    if args.onnx_ep == "cpu":
        return onnxruntime.InferenceSession(args.onnx, sess_options=sess_options)

    elif args.onnx_ep == 'dml':
        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['DmlExecutionProvider'],
        )

    elif args.onnx_ep == 'vai':
        cache_dir = os.path.join(os.getcwd(),  r'cache')

        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['VitisAIExecutionProvider'],
            provider_options = [{
                #'config_file': f"{os.environ['VAIP_CONFIG_HOME']}/vaip_config.json",
                #'num_of_dpu_runners': num_of_dpu_runners, # it is not relevant in RAI 1.5
                'cacheDir': cache_dir,
                'cacheKey': os.path.basename(args.onnx),
                'ai_analyzer_visualization': enable_analyzer,
                'ai_analyzer_profiling': enable_analyzer,
            }]
        )

    else:
        raise ValueError(f"Invalid onnxruntime execution provider : {args.onnx_ep}")


def main(args):
    session = create_session(args)

    inputs = {}
    for input in session.get_inputs():
        print(f"Input Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
        shape = [args.batch_size if isinstance(s, str) else s for s in input.shape]
        print(shape)
        inputs[input.name] = np.random.rand(*shape).astype(np.float32)

    outputs = []
    for output in session.get_outputs():
        print(f"Output Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
        outputs.append(output.name)

    # Warm up
    session.run(outputs, inputs)

    if args.parallel < 2:
        for _ in tqdm(range(args.test_num), desc="Processing batches"):
            session.run(outputs, inputs)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            queue = Queue(maxsize=args.parallel-1)

            def get_results():
                for _ in tqdm(range(args.test_num), desc="Processing batches"):
                    future = queue.get()
                    future.result()

            thread = Thread(target=get_results)
            thread.start()

            for _ in range(args.test_num):
                queue.put(executor.submit(session.run, outputs, inputs))

            thread.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YoloV8m on COCO dataset')
    parser.add_argument('onnx', type=str)
    parser.add_argument('--cpu_threads', type=int, default=0)
    parser.add_argument('--onnx_ep', type=str, default='cpu')
    parser.add_argument('--test_num', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--input_shape', type=int, nargs="+", default=[1, 3, 640, 640])
    parser.add_argument('--parallel', type=int, default=1)
    args = parser.parse_args()

    assert args.test_num > 0

    main(args)
