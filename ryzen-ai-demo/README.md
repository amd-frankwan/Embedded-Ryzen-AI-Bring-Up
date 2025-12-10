# Ryzen AI YOLOv8 demo

About this demo, see https://amd.atlassian.net/wiki/spaces/~juna/pages/507880485.

## Setup on Windows

This demo is tested on Phoenix and Hawk Point.

Install Ryzen AI following [the official documentation](https://ryzenai.docs.amd.com/en/latest/inst.html) and follow the instruction below.

```
# Setup conda environment
conda activate ryzen-ai-1.3.1
pip install -r ./requirements.txt
```

## Setup on Linux

This demo is tested on Lilac CRB.

Install Ryzen AI 1.4 following the README on the lounge. Since Ryzen AI offers only an ONNX Runtime installer for Python 3.10, install Python 3.10 using pyenv.

```
curl -fsSL https://pyenv.run | bash
```

Append the following code to your `~/.bashrc` and reopen the shell.

```
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
```

Install dependencies and Python 3.10. This will change your default Python version to 3.10.

```
sudo apt install -y build-essential libssl-dev libbz2-dev libncurses-dev libffi-dev libreadline-dev libsqlite3-dev
pyenv install 3.10
pyenv global 3.10
```

Create a venv environtment and install the ONNX Runtime and requirements.

```
python -m venv venv
source ./venv/bin/activate
pip install /usr/share/amdxdna/onnx_rt/*.whl
pip install -r ./requirements.txt
```

Setup your environment. Do this every time you open a new shell.

```
source ./venv/bin/activate
source ./setup.sh
```

## Download the model

Download the YOLOv8m model to the local directory.

```
huggingface-cli download amd/yolov8m yolov8m.onnx --local-dir lib
```

## Video inference demo

Before running the app, please modify the video_path variable in the source code so that it points to your input video.

```
python app-video.py
```

Press 'q' to stop the app.

## Runtime tutorial (Japanese)

Open [demo-jp.ipynb](demo-jp.ipynb) in Jupyter or VS Code.
