#Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

import onnxruntime

#quantized_int8_model='../models/resnet50-v1-12_int8.onnx'
fp32_model = '../models/resnet50-v1-12.onnx'

#Compile for Ryzen
provider_options_dict = {
    "config_file": 'vitisai_config_ryzen.json',
    "cache_dir":   'ryzen_cache_dir',
    "cache_key":   'resnet50',
    "ai_analyzer_visualization": True,
    "ai_analyzer_profiling": True,
    "target": "VAIML",
}
   
session = onnxruntime.InferenceSession(
    fp32_model, #quantized_int8_model,
    providers=["VitisAIExecutionProvider"],
    provider_options=[provider_options_dict]
)   
