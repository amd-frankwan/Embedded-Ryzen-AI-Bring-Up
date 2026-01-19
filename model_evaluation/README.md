python quantize_onnx.py --input initial_dl_anomaly_medium.onnx --output anomaly_medium_int8.onnx --config XINT8
python benchmark_onnx.py anomaly_medium_int8.onnx --ep NPU --test_num 600 --parallel 4 --config XINT8
python benchmark_onnx.py anomaly_medium_BF16.onnx --ep NPU --test_num 600 --parallel 4 --config BF16