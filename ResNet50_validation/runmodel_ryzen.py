import numpy as np
import onnxruntime as ort
import time
import os
import argparse

quantized_int8_model='./resnet50_quantized_int8.onnx'

provider_options_dict = {
    "config_file": 'vitisai_config_ryzen.json',
    "cache_dir":   'ryzen_cache_dir',
    "cache_key":   'resnet50_quantized_int8',
    "ai_analyzer_visualization": True,
    "ai_analyzer_profiling": True,
    "target": "VAIML"
}

# NPU session
npu_session = ort.InferenceSession(
    quantized_int8_model,
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict]
)

input_folder="./input"
def benchmark_model(session, output_dir="./output_ryzen"):
    total_time=0
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
    input_name = npu_session.get_inputs()[0].name
    runs=0
    for i,f in enumerate(files):
        runs+=1
        fp = os.path.join(input_folder, f)
        inputs = np.load(fp)
        start_time = time.time()
        outputs = npu_session.run(None, {input_name:inputs})
        end_time = time.time()
        total_time += end_time - start_time
        # Create outpu directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        for idx, out in enumerate(outputs):
            np.save(f"{output_dir}/output_{i}_{idx}.npy", out)
    avg_time = total_time / runs
    print('Average inference time over {} runs: {} ms'.format(runs, avg_time * 1000))
    print("Inference done")

benchmark_model(npu_session)
