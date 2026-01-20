from argparse import ArgumentParser, Namespace

import os
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread

import numpy as np
import onnxruntime

from tqdm import tqdm


xclbin_file = '{}/voe-4.0-linux_x86_64/xclbins/strix/AMD_AIE2P_4x4_Overlay.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])

os.environ["DEBUG_LOG_LEVEL"] = "info"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "2"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1"

def create_session(args, num_of_dpu_runners=4, enable_analyzer=False):
    available_providers = onnxruntime.get_available_providers()
    print(f"Available execution providers: {available_providers}")

    print (xclbin_file)

    input(f"Load {args.onnx} in {args.config} with {args.ep} EP; Press enter to continue...")

    sess_options = onnxruntime.SessionOptions()

    if args.cpu_threads > 0:
        sess_options.intra_op_num_threads = args.cpu_threads

    if args.ep == "CPU":
        return onnxruntime.InferenceSession(args.onnx, sess_options=sess_options)

    elif args.ep == 'ROCM':
        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['ROCMExecutionProvider'],
        )

    elif args.ep == 'MIGRAPHX':
        provider_options_dict = {}
        if args.config == 'FP16':
            provider_options_dict = {
                'migraphx_fp16_enable': 'True'
            }
        else:
            provider_options_dict = {
                'migraphx_fp16_enable': 'False'
            }

        return onnxruntime.InferenceSession(
            args.onnx,
            providers = ['MIGraphXExecutionProvider'],
            provider_options = [provider_options_dict]
        )

    elif args.ep == 'NPU':
        if args.config == 'INT8':
            cache_dir = str(Path(__file__).parent.resolve())+'/int8_cache'
            provider_options_dict = {
                'cache_dir': cache_dir,
                'cache_key': 'modelcachekey',
                'xclbin': xclbin_file, #'xlnx_target_name': 'AMD_AIE2P_4x4_Overlay',
                'enable_preemption': '0',
                'enable_txn_elf': '0'
            }

        elif args.config == 'BF16':
            cache_dir = str(Path(__file__).parent.resolve())+'/bf16_cache'
            provider_options_dict = {
                "config_file": 'vaiml_config.json',
                "cache_dir":   cache_dir,
                "cache_key":   'modelcachekey' #"target": "VAIML"
            }
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

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument('onnx', type=str)
    parser.add_argument('--cpu_threads', type=int, default=0)
    parser.add_argument('--ep', type=str, default='CPU')
    parser.add_argument("--config", type=str, default="INT8")
    parser.add_argument('--test_num', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    #parser.add_argument('--input_shape', type=int, nargs="+", default=[1, 3, 640, 640])
    parser.add_argument('--parallel', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert args.test_num > 0
    main(args)
