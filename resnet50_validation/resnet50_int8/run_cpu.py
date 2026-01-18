import numpy as np
import onnxruntime as ort
import time
import os
import argparse
import cv2

quantized_int8_model='./resnet50_quantized_int8.onnx'
# CPU session
cpu_session = ort.InferenceSession(
    quantized_int8_model,
	providers=["CPUExecutionProvider"]
)

input_folder="./input"
files = sorted([f for f in os.listdir(input_folder) if f.endswith(".npy")])
input_name = cpu_session.get_inputs()[0].name
total_time=0
runs=0
for i,f in enumerate(files):
    runs+=1
    fp = os.path.join(input_folder, f)
    inputs = np.load(fp)
    start_time = time.time()
    outputs = cpu_session.run(None, {input_name:inputs})
    end_time = time.time()
    total_time += end_time - start_time
    # Create outpu directory if it doesn't exist
    os.makedirs("./output_cpu", exist_ok=True)
    for idx, out in enumerate(outputs):
        np.save(f"./output_cpu/output_{i}_{idx}.npy", out)
avg_time = total_time / runs
print('Average inference time over {} runs: {} ms'.format(runs, avg_time * 1000))
print("CPU Inference done")