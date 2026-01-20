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