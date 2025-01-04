# Quante

quante 是处理一维自旋量子系统的 python 工具集合，包含了一维自旋量子系统的精确对角化、张量网络以及数据处理保存相关的工具。同时集成了 QuSpin 中一维费米和玻色子相关的精确对角化功能。

## Installation

- 建立虚拟环境：`conda create -n py311 python=3.11`.

- 激活虚拟环境：`conda activate py311`.

- 安装依赖：`conda install numba=0.60 numpy=1 scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler`

- 进入 setup.py 所在的文件夹：`cd path/to/setup.py`

- 使用开发者模式安装：`pip install -e .`（如何需要安装到本地，可以使用`pip install .`）

- （可选）torch tenpy 的安装，确认 gpu 版本：`nvcc --version`

- （可选）通过 https://pytorch.org/ 说明安装对应版本 pytorch，cpu 安装命令为：`conda install pytorch torchvision torchaudio cpuonly -c pytorch`

- （可选）安装 quspin-extension: `pip install quspin-extension`
