import os
import sys
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

setup_npu_env()

enable_analyzer = False

npu_session = onnxruntime.InferenceSession(
    onnx_model_path,
    providers = ['VitisAIExecutionProvider'],
    provider_options = [{
        'config_file': f"{os.environ['VAIP_CONFIG_HOME']}/vaip_config.json",
        'cacheDir': os.path.join(os.getcwd(),  r'cache'),
		'cacheKey': 'yolov8m',
        'ai_analyzer_visualization': enable_analyzer,
        'ai_analyzer_profiling': enable_analyzer,
    }]
)

names = load_coco_names()

# HWC
img_org = cv2.imread('sample/furniture_store.jpg')
img_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

# Letterbox
src_size = img_rgb.shape[0:2]
dst_size = [640, 640]
assert src_size[1] == dst_size[1]
letterbox = [range((d-s)//2, s + (d-s)//2) for s,d in zip(src_size, dst_size)]

# Torch tensor
#img = torch.empty((1, 3, *dst_size), dtype=torch.float)
#img[0, :, letterbox[0], :] = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255
#img = torch.empty((1, *dst_size, 3), dtype=torch.float)
#img[0, letterbox[0], :, :] = torch.from_numpy(img_rgb).float() / 255
img = np.zeros((1, *dst_size, 3), dtype=np.float32)
img[0, letterbox[0], :, :] = img_rgb / 255

# NHWC
#outputs = npu_session.run(None, {npu_session.get_inputs()[0].name: img.permute(0, 2, 3, 1).cpu().numpy()})
#outputs = npu_session.run(None, {npu_session.get_inputs()[0].name: img.cpu().numpy()})
outputs = npu_session.run(None, {npu_session.get_inputs()[0].name: img})

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

#torchvision.utils.save_image(img[0], "output.jpg")
cv2.imwrite("output.jpg", img_org)
