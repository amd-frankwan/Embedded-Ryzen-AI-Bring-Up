import os
import sys
import time
import cv2
import onnxruntime
import torch
import torchvision
import numpy as np

import queue
import threading

from lib.yolov8 import (
    onnx_model_path,
    setup_npu_env,
    post_process,
    non_max_suppression,
    output_to_target,
    load_coco_names,
    Annotator,
    colors,
)

setup_npu_env()

num_instance = 4
#os.environ['NUM_OF_DPU_RUNNERS'] = str(num_instance)

enable_analyzer = False

npu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers = ['VitisAIExecutionProvider'],
    provider_options = [{
        'config_file': f"{os.environ['VAIP_CONFIG_HOME']}/vaip_config.json",
        'num_of_dpu_runners': num_instance,
        'cacheDir': os.path.join(os.getcwd(),  r'cache'),
		'cacheKey': 'yolov8m',
        'ai_analyzer_visualization': enable_analyzer,
        'ai_analyzer_profiling': enable_analyzer,
    }]
)

names = load_coco_names()

def preprocess_func(img_org):
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    # Letterbox
    src_size = img_rgb.shape[0:2]
    dst_size = [640, 640]
    assert src_size[1] == dst_size[1]
    letterbox = [range((d-s)//2, s + (d-s)//2) for s,d in zip(src_size, dst_size)]

    img = np.zeros((1, *dst_size, 3), dtype=np.float32)
    img[0, letterbox[0], :, :] = img_rgb / 255

    return (img_org, letterbox, img)

def infer_func(session, img_org, letterbox, img):
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    return (img_org, letterbox, outputs)

def postprocess_func(img_org, letterbox, outputs):
    outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
    preds = post_process(outputs)
    preds = non_max_suppression(
        preds, 0.25, 0.7, agnostic=False, max_det=300, classes=None
    )
    # [batch_id], [class_id], [x, y, w, h, conf]
    preds = output_to_target(preds, max_det=30)

    for i, (batch, cls, bbox) in enumerate(zip(*preds)):
        cls = int(cls)
        conf = int(bbox[4] * 100)

        label = f"{names[cls]} :{conf:3}%"

        xy = bbox[0:2].astype(np.int32)
        wh = bbox[2:4].astype(np.int32)
        xy -= wh // 2 # center to top-left
        xy[1] -= letterbox[0].start
        xy2 = xy + wh

        annotator = Annotator(img_org)
        annotator.box_label([*xy, *xy2], label, colors(cls))
    
    return (img_org, )

def run_task(task_func, input_queue, output_queue, args=()):
    while True:
        try:
            inputs = input_queue.get(timeout=3) # 3 sec
            outputs = task_func(*args, *inputs)
            output_queue.put(outputs)
        except queue.Empty:
            break

def start_threads(task_func, input_queue, output_queue, args=(), num_threads=1):
    threads = []
    for _ in range(num_threads):
        thr = threading.Thread(target=run_task, args=(task_func, input_queue, output_queue, args))
        thr.start()
        threads.append(thr)
    return threads

pre_q     = queue.Queue()
infer_q   = queue.Queue()
post_q    = queue.Queue()
display_q = queue.Queue()

start_threads(preprocess_func, pre_q, infer_q, num_threads=1)
start_threads(infer_func, infer_q, post_q, (npu_session,), num_threads=4)
start_threads(postprocess_func, post_q, display_q, num_threads=1)

img_org = cv2.imread('sample/furniture_store.jpg')

num_test = 20
start = time.perf_counter()

for _ in range(num_test):
    pre_q.put((img_org, ))
for i in range(num_test):
    print(i)
    img_org, = display_q.get()

end = time.perf_counter()

print(end - start)
print(num_test / (end - start), "fps")

cv2.imwrite("output.jpg", img_org)
