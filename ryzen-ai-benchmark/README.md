## Setup

```
conda activate ryzen-ai-1.3.1
pip install -r requirements.txt
```

## ResNet50

```
# CPU, PyTorch
python benchmark_torchvision.py resnet50 --test_num 400 --cpu_threads 10 --export_onnx resnet50.onnx

# CPU, ORT
python simplify_onnx.py resnet50.onnx resnet50_sim.onnx
python benchmark_onnx.py resnet50_sim.onnx --onnx_ep cpu --test_num 800 --cpu_threads 10

# GPU, batch=1
python benchmark_onnx.py resnet50_sim.onnx --onnx_ep dml --test_num 800

# GPU, batch=8
python benchmark_onnx.py resnet50_sim.onnx --onnx_ep dml --test_num 200 --batch_size 8

# NPU
curl -O https://github.com/amd/RyzenAI-SW/raw/refs/heads/main/onnx-benchmark/models/resnet50/resnet50_fp32_qdq.onnx
python benchmark_onnx.py resnet50_fp32_qdq.onnx --onnx_ep vai --test_num 3200 --parallel 4
```

## YOLOv8m

```
# CPU, PyTorch
yolo benchmark model=yolov8m.pt data='coco8.yaml' imgsz=640 format=-

# CPU, OpenVINO
yolo benchmark model=yolov8m.pt data='coco8.yaml' imgsz=640 format=openvino

# CPU, ORT
yolo export model=yolov8m.pt format=onnx
REN yolov8m.onnx yolov8m_b1.onnx
python benchmark_onnx.py yolov8m_b1.onnx --onnx_ep cpu --test_num 80 --cpu_threads 10

# GPU, batch=1
python benchmark_onnx.py yolov8m_b1.onnx --onnx_ep dml --test_num 100

# GPU, batch=8
yolo export model=yolov8m.pt format=onnx batch=8
REN yolov8m.onnx yolov8m_b8.onnx
python benchmark_onnx.py yolov8m_b8.onnx --onnx_ep dml --test_num 20

# NPU
huggingface-cli download amd/yolov8m yolov8m.onnx --local-dir .
REN yolov8m.onnx yolov8m_int8.onnx
python benchmark_onnx.py yolov8m_int8.onnx --onnx_ep vai --test_num 600 --parallel 4
```

## Trouble shooting

```
# Specified provider 'VitisAIExecutionProvider' is not in available provider names.
# Available providers: 'DmlExecutionProvider, CPUExecutionProvider'

pip install --force-reinstall "C:\Program Files\RyzenAI\1.3.1\onnxruntime_vitisai-1.19.0-cp310-cp310-win_amd64.whl" "numpy<2" "onnxruntime<1.20.0,>=1.17.0"

cd "C:\Program Files\RyzenAI\1.3.1"
pip install --force-reinstall aianalyzer-1.3.1-py3-none-any.whl quark-0.6.0-py3-none-any.whl vai_q_onnx-1.19.0-py2.py3-none-win_amd64.whl flexmlrt-1.3.1-py3-none-any.whl ryzen_ai_rt-1.3.1-py3-none-any.whl vaitrace-1.3.1-py3-none-any.whl microkernel_tiling-1.3.1-py310-none-win_amd64.whl torch-2.3.1+cpu-cp310-cp310-win_amd64.whl voe-1.3.1-cp310-cp310-win_amd64.whl onnxruntime_vitisai-1.19.0-cp310-cp310-win_amd64.whl torchvision-0.18.1+cpu-cp310-cp310-win_amd64.whl "numpy<2" "onnxruntime<1.20.0,>=1.17.0"
```

