import os
import sys
import time
import collections
import queue
import threading

import cv2
import onnxruntime
import torch
import torchvision
import numpy as np

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

def create_cpu_session(onnx_model_path):
    cpu_options = onnxruntime.SessionOptions()

    # Create Inference Session to run the quantized model on the CPU
    cpu_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers = ['CPUExecutionProvider'],
        sess_options=cpu_options,
    )

    return cpu_session

def create_npu_session(
    onnx_model_path,
    num_instance = 4,
    enable_analyzer = False,
):
    setup_npu_env()

    #os.environ['NUM_OF_DPU_RUNNERS'] = str(num_instance)

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

    return npu_session

names = load_coco_names()

def preprocess_func(img_org, frame_id):
    # BGR to RGB
    img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    # Scale
    org_size = img_rgb.shape[0:2]
    scale = 640 / org_size[1]
    new_size = (640, int(org_size[0] * scale))
    img_rgb = cv2.resize(img_rgb, new_size)

    # Letterbox
    src_size = img_rgb.shape[0:2]
    dst_size = [640, 640]
    assert src_size[1] == dst_size[1]
    letterbox = [range((d-s)//2, s + (d-s)//2) for s,d in zip(src_size, dst_size)]

    img = np.zeros((1, *dst_size, 3), dtype=np.float32)
    img[0, letterbox[0], :, :] = img_rgb / 255

    return (img_org, frame_id, scale, letterbox, img)

def infer_func(img_org, frame_id, scale, letterbox, img, session):
    outputs = session.run(None, {session.get_inputs()[0].name: img})
    return (img_org, frame_id, scale, letterbox, outputs)

class NoOutputException(Exception):
    pass

def sort_frame_func(img_org, frame_id, scale, letterbox, outputs):
    if not hasattr(sort_frame_func, 'init'):
        sort_frame_func.buffer = []
        sort_frame_func.next_frame = 0
        sort_frame_func.init = True

    sort_frame_func.buffer.append((img_org, frame_id, scale, letterbox, outputs))

    for i, item in enumerate(sort_frame_func.buffer):
        if item[1] == sort_frame_func.next_frame:
            sort_frame_func.next_frame += 1
            del sort_frame_func.buffer[i]
            return item
    
    raise NoOutputException()

def postprocess_func(img_org, frame_id, scale, letterbox, outputs, line_width):
    outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
    preds = post_process(outputs)
    preds = non_max_suppression(
        preds, conf_thres=0.25, iou_thres=0.45, agnostic=False, max_det=300, classes=None
    )
    # [batch_id], [class_id], [x, y, w, h, conf]
    preds = output_to_target(preds, max_det=300)

    # Draw detection result
    annotator = Annotator(img_org, line_width=line_width)
    for i, (batch, cls, bbox) in enumerate(zip(*preds)):
        cls = int(cls)
        conf = int(bbox[4] * 100)

        label = f"{names[cls]} :{conf:3}%"

        xy = bbox[0:2]
        wh = bbox[2:4]
        xy -= wh // 2 # center to top-left
        xy[1] -= letterbox[0].start
        xy2 = xy + wh

        xy  /= scale
        xy2 /= scale

        annotator.box_label([*xy, *xy2], label, colors(cls))
    
    return (img_org, )

def run_task(task_func, input_queue, output_queue, args=()):
    while True:
        try:
            inputs = input_queue.get(timeout=3) # 3 sec
            outputs = task_func(*inputs, *args)
            output_queue.put(outputs)
        except NoOutputException:
            continue
        except queue.Empty:
            print("finish task", task_func)
            break

def start_threads(task_func, input_queue, output_queue, args=(), num_threads=1):
    threads = []
    for _ in range(num_threads):
        thr = threading.Thread(target=run_task, args=(task_func, input_queue, output_queue, args))
        thr.start()
        threads.append(thr)
    return threads

# https://stackoverflow.com/a/54539292
class FPS:
    def __init__(self, average_of=60):
        self.frametimestamps = collections.deque(maxlen=average_of)

    def __call__(self):
        self.frametimestamps.append(time.time())
        return self.get()
    
    def get(self):
        if len(self.frametimestamps) > 1:
            elapsed_time = self.frametimestamps[-1] - self.frametimestamps[0]
            if elapsed_time > 0:
                return (len(self.frametimestamps) - 1) / elapsed_time
            else:
                return 0.0
        else:
            return 0.0
