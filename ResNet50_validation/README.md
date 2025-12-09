## Inference on Ryzen

Set up the inference environment on the Linux-based Ryzen machine:

```bash
source <ryzen_ai_install_path>/lnx64/bin/activate
source /opt/xilinx/xrt/setup.sh
```

Run Inference on Ryzen:

```bash
python runmodel_ryzen.py
```

Example output:

```
Average inference time over 3 runs: 12.86458969116211 ms
Inference done
```

## Inference on CPU

To run CPU inference

```bash
python run_cpu.py
```

Example output:
```
Average inference time over 3 runs: 63.75988324483235 ms
CPU Inference done
```