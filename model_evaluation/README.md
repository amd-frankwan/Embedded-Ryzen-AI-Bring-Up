

python benchmark_onnx_1.5.py pretrained_deep_ocr_detection.onnx --ep NPU --test_num 600 --parallel 4 --config BF16
python quantize_onnx_rai.py --input pretrained_deep_ocr_detection.onnx --output pretrained_deep_ocr_detection_int8.onnx --config XINT8
python benchmark_onnx_1.5.py pretrained_deep_ocr_detection_int8.onnx --ep NPU --test_num 600 --parallel 4 --config INT8

"aie_single_core_compiler": "peano"


### ONNXRuntime Docker launch:
```
docker run -it --rm -v ~/:/tmp rocm/onnxruntime:rocm7.1.1_ub24.04_ort1.23_torch2.8.0 "bash"
```
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installryz/native_linux/install-ryzen.html
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-migraphx.html
https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-onnx.html


export HSA_OVERRIDE_GFX_VERSION=11.0.0

```
amd@GMKtec:~$ /opt/rocm-7.0.2/bin/migraphx-driver perf --test
Running [ MIGraphX Version: 2.13.0.20250912-42-9-g526105976 ]: /opt/rocm-7.0.2/bin/migraphx-driver perf --test
[2026-01-20 21:10:22]
Compiling ... 
module: "main"
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}
main:#output_0 = @param:main:#output_0 -> float_type, {4, 3}, {3, 1}
b = @param:b -> float_type, {5, 3}, {3, 1}
a = @param:a -> float_type, {4, 5}, {5, 1}
@4 = gpu::code_object[code_object=4608,symbol_name=mlir_dot,global=256,local=256,output_arg=2,](a,b,main:#output_0) -> float_type, {4, 3}, {3, 1}


Allocating params ... 
Running performance report ... 
@0 = check_context::migraphx::gpu::context -> float_type, {}, {}: 0.00028794ms, 3%
main:#output_0 = @param:main:#output_0 -> float_type, {4, 3}, {3, 1}: 0.00023ms, 3%
b = @param:b -> float_type, {5, 3}, {3, 1}: 0.00021218ms, 3%
a = @param:a -> float_type, {4, 5}, {5, 1}: 0.00020716ms, 2%
@4 = gpu::code_object[code_object=4608,symbol_name=mlir_dot,global=256,local=256,output_arg=2,](a,b,main:#output_0) -> float_type, {4, 3}, {3, 1}: 0.00948216ms, 92%

Summary:
gpu::code_object::mlir_dot: 0.00948216ms / 1 = 0.00948216ms, 92%
@param: 0.00064934ms / 3 = 0.000216447ms, 7%
check_context::migraphx::gpu::context: 0.00028794ms / 1 = 0.00028794ms, 3%

Batch size: 1
Rate: 78498.4 inferences/sec
Total time: 0.0127391ms (Min: 0.009838ms, Max: 0.030393ms, Mean: 0.0128411ms, Median: 0.013082ms)
Percentiles (90%, 95%, 99%): (0.014356ms, 0.014547ms, 0.025104ms)
Total instructions time: 0.0104194ms
Overhead time: 0.00065204ms, 0.00231968ms
Overhead: 5%, 18%
[2026-01-20 21:10:23]
[ MIGraphX Version: 2.13.0.20250912-42-9-g526105976 ] Complete(0.67124s): /opt/rocm-7.0.2/bin/migraphx-driver perf --test
```

initial_dl_anomaly_medium.onnx       65FPS => 136FPS =>INT8 NPU (54FPS)

Input Name: image, Shape: [1, 3, 480, 480], Type: tensor(float)
[1, 3, 480, 480]
Output Name: output, Shape: [1, 64, 120, 120], Type: tensor(float)
Processing batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:10<00:00, 57.79it/s]


pretrained_deep_ocr_detection.onnx   19FPS => 32FPS

[Vitis AI EP] No. of Operators : VAIML   140 
[Vitis AI EP] No. of Subgraphs : VAIML     1 
Input Name: image, Shape: [1, 3, 1024, 1024], Type: tensor(float)
[1, 3, 1024, 1024]
Output Name: score_maps, Shape: [1, 4, 512, 512], Type: tensor(float)
Processing batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [02:52<00:00,  3.47it/s]


pretrained_deep_ocr_recognition.onnx 553FPS => 1294 FPS

pretrained_dl_3d_gripping_point.onnx 8.52FPS => 14.8 FPS => FP16 NPU Falls to CPU (8.43FPS)/INT8 NPU (11FPS)

[Vitis AI EP] No. of Operators :   CPU   143 
Input Name: image, Shape: [1, 3, 480, 640], Type: tensor(float)
[1, 3, 480, 640]
Output Name: affordance_confidence, Shape: [1, 4, 480, 640], Type: tensor(float)
Output Name: affordance_argmax, Shape: [1, 1, 480, 640], Type: tensor(float)
Processing batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [01:11<00:00,  8.43it/s]

pretrained_dl_3d_gripping_point.onnx 26.63FPS => 42.88FPS => INT8 NPU (most in CPU, 3.59FPS)

[Vitis AI EP] No. of Operators :   CPU   140 
Input Name: image, Shape: [32, 3, 224, 224], Type: tensor(float)
[32, 3, 224, 224]
Output Name: output, Shape: [32, 10, 1, 1], Type: tensor(float)
Processing batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 600/600 [00:22<00:00, 26.78it/s]

https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.2/install/quick-start.html
https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.3/install/quick-start.html

wget https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/noble/amdgpu-install_6.4.60404-1_all.deb  
sudo apt install ./amdgpu-install_6.4.60404-1_all.deb  
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
sudo apt install amdgpu-dkms

https://repo.radeon.com/rocm/manylinux/

https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.4.4/docs/install/installrad/native_linux/install-onnx.html

sudo apt autoremove rocm-*

sudo amdgpu-install --uninstall --rocmrelease=all

