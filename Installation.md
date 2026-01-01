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
`echo $RYZEN_AI_INSTALLATION_PATH` should print `$HOME/rai-1.6.1-venv`


### Some known bugs:

1. `lib/libstdc++.so.6: version 'GLIBCXX_3.4.32' not found (required by /usr/lib/python3/dist-packages/apt_pkg.cpython-312-x86_64-linux-gnu.so)`


```
cd $RYZEN_AI_INSTALLATION_PATH/lib/python3.10/site-packages/flexml/flexml_extras/lib
mv libstdc++.so.6 libstdc++.so.6_backup
ln -s /lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```


