# Dependencies

import time
import os
import platform
import sys
import subprocess
import torch
import torch.nn as nn
import onnxruntime
import numpy as np

from huggingface_hub import hf_hub_download
from lib.yolov8_utils import *


current_dir = get_directories()

# Download Yolov8 model from Ryzen AI model zoo. Registration is required before download.
#hf_hub_download(repo_id="amd/yolov8m", filename="yolov8m.onnx", local_dir=str(current_dir))

def preprocess(img):
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    return img


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(
            b, 4, a
        )


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


anchors_ = None
strides_ = None

def post_process(x):
    global anchors_
    global strides_
    dfl = DFL(16)
    if anchors_ is None:
        print("Load anchors.npy")
        anchors_ = torch.tensor(
            np.load(
                f"{current_dir}/anchors.npy",
                allow_pickle=True,
            )
        )
    if strides_ is None:
        print("Load strides.npy")
        strides_ = torch.tensor(
            np.load(
                f"{current_dir}/strides.npy",
                allow_pickle=True,
            )
        )
    box, cls = torch.cat([xi.view(x[0].shape[0], 144, -1) for xi in x], 2).split(
        (16 * 4, 80), 1
    )
    dbox = dist2bbox(dfl(box), anchors_.unsqueeze(0), xywh=True, dim=1) * strides_
    y = torch.cat((dbox, cls.sigmoid()), 1)
    return y, x


def load_coco_names():
    # Load labels of coco dataaset
    with open(f'{current_dir}/coco.names', 'r') as f:
        names = f.read().splitlines()
    return names


def setup_npu_env():
    # NPU inference

    is_linux = platform.system() == 'Linux'

    # Before running, we need to set the ENV variable for the specific NPU we have
    if is_linux:
        command = '/opt/xilinx/xrt/bin/xrt-smi examine | grep -o "npu[0-9]"'
        npu_code = subprocess.run(command, shell=True, capture_output=True, text=True, check=True).stdout.strip()
        if npu_code == "npu1":
            apu_type = 'PHX/HPT'
        else:
            apu_type = 'STX'
    else:
        # Run pnputil as a subprocess to enumerate PCI devices
        command = r'pnputil /enum-devices /bus PCI /deviceids '
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        # Check for supported Hardware IDs
        apu_type = ''
        if 'PCI\\VEN_1022&DEV_1502&REV_00' in stdout.decode(): apu_type = 'PHX/HPT'
        if 'PCI\\VEN_1022&DEV_17F0&REV_00' in stdout.decode(): apu_type = 'STX'
        if 'PCI\\VEN_1022&DEV_17F0&REV_10' in stdout.decode(): apu_type = 'STX'
        if 'PCI\\VEN_1022&DEV_17F0&REV_11' in stdout.decode(): apu_type = 'STX'

    print(f"APU Type: {apu_type}")

    install_dir = os.environ['RYZEN_AI_INSTALLATION_PATH']
    match apu_type:
        case 'PHX/HPT':
            print("Setting environment for PHX/HPT")
            if is_linux:
                os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, '1x4.xclbin')
            else:
                os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'phoenix', '1x4.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case 'STX':
            print("Setting environment for STX")
            if is_linux:
                os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'AMD_AIE2P_Nx4_Overlay.xclbin')
            else:
                os.environ['XLNX_VART_FIRMWARE']= os.path.join(install_dir, 'voe-4.0-win_amd64', 'xclbins', 'strix', 'AMD_AIE2P_Nx4_Overlay.xclbin')
            os.environ['NUM_OF_DPU_RUNNERS']='1'
            os.environ['XLNX_TARGET_NAME']='AMD_AIE2_Nx4_Overlay'
        case _:
            print("Unrecognized APU type. Exiting.")
            exit()
    print('XLNX_VART_FIRMWARE=', os.environ['XLNX_VART_FIRMWARE'])
    print('NUM_OF_DPU_RUNNERS=', os.environ['NUM_OF_DPU_RUNNERS'])
    print('XLNX_TARGET_NAME=', os.environ['XLNX_TARGET_NAME'])


# Specify the path to the quantized ONNX Model
onnx_model_path = f"{current_dir}/yolov8m.onnx"


if __name__ == '__main__':

    names = load_coco_names()

    imgsz = [640, 640]

    # Set input image
    image_path = f"{current_dir}/sample_yolov8.jpg"
    image = Image.open(image_path)
    image = image.resize((640,640), Image.BICUBIC)

    # Load image
    dataset = LoadImages(
        image_path, imgsz=imgsz, stride=32, auto=False, transforms=None, vid_stride=1
    )

    #CPU inference


    # Output file
    output_path = f"{current_dir}/cpu_result.jpg"

    cpu_options = onnxruntime.SessionOptions()

    # Create Inference Session to run the quantized model on the CPU
    cpu_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers = ['CPUExecutionProvider'],
        sess_options=cpu_options,
    )

    for batch in dataset:
        path, im, im0s, vid_cap, s = batch
        im = preprocess(im)
        if len(im.shape) == 3:
            im = im[None]
        start = time.time()
        outputs = cpu_session.run(None, {cpu_session.get_inputs()[0].name: im.permute(0, 2, 3, 1).cpu().numpy()})
        end = time.time()

        inference_time = np.round((end - start) * 1000, 2)

        # Postprocessing
        outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
        preds = post_process(outputs)
        preds = non_max_suppression(
            preds, 0.25, 0.7, agnostic=False, max_det=300, classes=None
        )
        
        print('----------------------------------------')
        print('Inference time: ' + str(inference_time) + " ms")
        print('----------------------------------------')



    # iGPU inference

    # Output file
    output_path = f"{current_dir}/dml_result.jpg"

    dml_options = onnxruntime.SessionOptions()

    # Create Inference Session to run the quantized model on the iGPU
    dml_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers = ['DmlExecutionProvider'],
        provider_options = [{"device_id": "0"}]
    )

    for batch in dataset:
        path, im, im0s, vid_cap, s = batch
        im = preprocess(im)
        if len(im.shape) == 3:
            im = im[None]
        start = time.time()
        outputs = dml_session.run(None, {dml_session.get_inputs()[0].name: im.permute(0, 2, 3, 1).cpu().numpy()})
        end = time.time()

        inference_time = np.round((end - start) * 1000, 2)

        # Postprocessing
        outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
        preds = post_process(outputs)
        preds = non_max_suppression(
            preds, 0.25, 0.7, agnostic=False, max_det=300, classes=None
        )
        
        print('----------------------------------------')
        print('Inference time: ' + str(inference_time) + " ms")
        print('----------------------------------------')

    # NPU inference
    setup_npu_env()

    # Output file
    output_path = f"{current_dir}/npu_result.jpg"

    # Point to the config file path used for the VitisAI Execution Provider
    config_file_home = os.environ['VAIP_CONFIG_HOME']
    config_file_path = f"{config_file_home}/vaip_config.json"

    provider_options = [{
                'config_file': config_file_path,
                'ai_analyzer_visualization': True,
                'ai_analyzer_profiling': True,
            }]

    npu_session = onnxruntime.InferenceSession(
        onnx_model_path,
        providers = ['VitisAIExecutionProvider'],
        provider_options = provider_options
    )

    for batch in dataset:
        path, im, im0s, vid_cap, s = batch
        im = preprocess(im)
        if len(im.shape) == 3:
            im = im[None]
        start = time.time()
        outputs = npu_session.run(None, {npu_session.get_inputs()[0].name: im.permute(0, 2, 3, 1).cpu().numpy()})
        end = time.time()

        inference_time = np.round((end - start) * 1000, 2)

        # Postprocessing
        outputs = [torch.tensor(item).permute(0, 3, 1, 2) for item in outputs]
        preds = post_process(outputs)
        preds = non_max_suppression(
            preds, 0.25, 0.7, agnostic=False, max_det=300, classes=None
        )
        
        print('----------------------------------------')
        print('Inference time: ' + str(inference_time) + " ms")
        print('----------------------------------------')