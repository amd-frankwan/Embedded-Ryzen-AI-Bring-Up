## Preparation
```
sudo apt update
sudo apt install build-essential dkms git git-lfs vim libboost-filesystem1.74.0 -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-venv python3.10-dev -y
```

## Download [RyzenAI-SW](https://github.com/amd/RyzenAI-SW) propely
```
git lfs install
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW
git lfs pull
```

### Check older package and remove if necessary
```
dpkg -l | grep xrt
sudo apt purge xrt-base xrt-npu
sudo apt --fix-broken install
```

### Install XRT package and NPU Driver package
```
sudo apt reinstall --fix-broken -y xxxxx.deb
```

### Setup environment
```
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh
```

### Launch virtualenv once installed (e.g. installed in `~/rai-1.6.1-venv`):
```
source ~/rai-1.6.1-venv/bin/activate
```
Once getting into the virtualenv, `echo $RYZEN_AI_INSTALLATION_PATH` should print `$HOME/rai-1.6.1-venv`


### Some known bugs:

1. `lib/libstdc++.so.6: version 'GLIBCXX_3.4.32' not found (required by /usr/lib/python3/dist-packages/apt_pkg.cpython-312-x86_64-linux-gnu.so)`


```
cd $RYZEN_AI_INSTALLATION_PATH/lib/python3.10/site-packages/flexml/flexml_extras/lib
mv libstdc++.so.6 libstdc++.so.6_backup
ln -s /lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```

2. Installation of RAI 1.5 virtualenv (don't use sudo)
```
./install_ryzen_ai.sh -a yes -n rai-1.5.0-venv -p $HOME/rai-1.5.0-venv ../ -c ../ryzen_ai-1.5.0/
```
For RAI1.6.1 will be
```
./install_ryzen_ai.sh -a yes -p ~/rai-1.6.1-venv/

3. ResNet50/Yolov8m tutorial Ubuntu run issue
In `xxx_util.py`, switch the `get_npu_info()` and `get_xclbin()` to

```
def get_npu_info():
    # Run pnputil as a subprocess to enumerate PCI devices
    command = r'lspci'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Check for supported Hardware IDs
    npu_type = ''
    if 'Device 17f0' in stdout.decode(): npu_type = 'KRK'
    elif 'Device 14ec' in stdout.decode(): npu_type = 'PHX/HPT'
    else: npu_type = 'STX'
    return npu_type

def get_xclbin(npu_device):
    xclbin_file = ''
    if npu_device == 'STX' or npu_device=='KRK':
        xclbin_file = '{}/voe-4.0-linux_x86_64/xclbins/strix/AMD_AIE2P_4x4_Overlay.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    if npu_device == 'PHX/HPT':
        xclbin_file = '{}/voe-4.0-linux_x86_64/xclbins/phoenix/4x4.xclbin'.format(os.environ["RYZEN_AI_INSTALLATION_PATH"])
    return xclbin_file

```

4. OpenCV version issue (numpy conflict)
Install the following version:
```
opencv-python==4.11.0.86
pycocotools==2.0.10
wget==3.2
ultralytics==8.3.155
timm==1.0.22
```

5. Yolov8m INT8 "DPU Timeout" on Ubunut on StrixHalo device:
Add following flags in provider_options dictionary in `run_inference.py` to disable preemption
```
provider_options = [{
    'cache_dir': str(Path(__file__).parent.resolve())+'/STX-INT8',
    'cache_key': 'modelcachekey',
    'enable_cache_file_io_in_mem':'0',
    'enable_preemption': '0',
    'enable_txn_elf': '0'
}]
```

6. Yolov8m BF16 "DPU Timeout" on Ubunut on StrixHalo device:
Add following flag in vaiml_config in `vaiml_config.json` to change CG engine:
```
"vaiml_config": {
    "optimize_level": 2,
    "logging_level": "info",
    "aie_single_core_compiler": "peano"
}
```

7. Add more printout in R8000 debug log:
Add following flags in python script:
```
os.environ["DEBUG_LOG_LEVEL"] = "info"
os.environ["XLNX_ONNX_EP_VERBOSE"] = "2"
os.environ["XLNX_ENABLE_DUMP_XIR_MODEL"] = "1"
os.environ["VAIP_COMPILE_RESERVE_CONST_DATA"] = "1"
```

8. R8000 "DPU Timeout" Issue
Add following flags in provider_options to explicit define these variables:
```
'xlnx_target_name': 'AMD_AIE2_4x4_Overlay',
'xclbin': get_xclbin(npu_device), # it must be absolute path
```

9. `--exclude-subgraphs` not found:
Missing `\` after `--config XINT8` in the command:
```
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_XINT8.onnx \
                         --config XINT8 \
                         --exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]"
```

### ONNXRuntime Docker launch:
```
docker run -it --rm -v ~/Downloads:/tmp rocm/onnxruntime:rocm7.1.1_ub24.04_ort1.23_torch2.8.0 "bash"
```