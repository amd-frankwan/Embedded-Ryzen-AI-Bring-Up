from concurrent.futures import ThreadPoolExecutor
import os
from queue import Queue
#from re import X
from threading import Thread

import argparse
import numpy as np
import onnxruntime
from tqdm import tqdm
from pathlib import Path

def create_session(args, num_of_dpu_runners=4, enable_analyzer=False):
    input(f"Load {args.onnx} in {args.config} with {args.ep} EP; Press enter to continue...")

    sess_options = onnxruntime.SessionOptions()

    if args.cpu_threads > 0:
        sess_options.intra_op_num_threads = args.cpu_threads

    if args.ep == "CPU":
        return onnxruntime.InferenceSession(args.onnx, sess_options=sess_options)

    elif args.ep == 'GPU':
        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['DmlExecutionProvider'],
        )

    elif args.ep == 'NPU':
        cache_dir = str(Path(__file__).parent.resolve())
        if args.config == 'XINT8':
            provider_options_dict = {
                'cache_dir': cache_dir,
                'cache_key': 'modelcachekey',
                'enable_cache_file_io_in_mem':'0',
                'target': 'X1' 
                #'enable_preemption': '0',
                #'enable_txn_elf': '0'       
            }
        elif args.config == 'BF16':
            provider_options_dict = {
                "config_file": 'vaiml_config_ryzen.json',
                "cache_dir":   cache_dir,
                "cache_key":   'modelcachekey',
                "target": "VAIML",
            }
            #provider_options_dict = {
            #    'config_file': 'vaiml_config.json',
            #    'cache_dir': cache_dir,
            #    'cache_key': 'modelcachekey',
            #    'enable_cache_file_io_in_mem':'0' 
            #}
        else:
            raise ValueError(f"Invalid onnxruntime config : {args.config}")


        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['VitisAIExecutionProvider'],
            provider_options = [provider_options_dict]
        )

    else:
        raise ValueError(f"Invalid onnxruntime execution provider : {args.ep}")


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
    parser = argparse.ArgumentParser(description='Evaluate random model on random dataset')
    parser.add_argument('onnx', type=str)
    parser.add_argument('--cpu_threads', type=int, default=0)
    parser.add_argument('--ep', type=str, default='CPU')
    parser.add_argument("--config", type=str, default="XINT8")
    parser.add_argument('--test_num', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--input_shape', type=int, nargs="+", default=[1, 3, 640, 640])
    parser.add_argument('--parallel', type=int, default=1)
    args = parser.parse_args()

    assert args.test_num > 0

    main(args)
