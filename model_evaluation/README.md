python quantize_onnx.py --input initial_dl_anomaly_medium.onnx --output anomaly_medium_int8.onnx --config XINT8
python benchmark_onnx.py anomaly_medium_int8.onnx --ep NPU --test_num 600 --parallel 4 --config XINT8
python benchmark_onnx.py anomaly_medium_BF16.onnx --ep NPU --test_num 600 --parallel 4 --config BF16

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

initial_dl_anomaly_medium.onnx       65FPS => 136FPS
pretrained_deep_ocr_detection.onnx   19FPS => 32FPS
pretrained_deep_ocr_recognition.onnx 553FPS => 1294 FPS
pretrained_dl_3d_gripping_point.onnx 8.52FPS => 14.8 FPS
pretrained_dl_3d_gripping_point.onnx 26.63FPS => 42.88FPS

https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.0.0/install/quick-start.html

wget https://repo.radeon.com/amdgpu-install/7.0/ubuntu/noble/amdgpu-install_7.0.70000-1_all.deb
sudo apt install ./amdgpu-install_7.0.70000-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
sudo apt install rocm
sudo apt install amdgpu-dkms