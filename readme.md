# Quante

quante 是处理一维自旋量子系统的 python 工具集合，包含了一维自旋量子系统的精确对角化、张量网络以及数据处理保存相关的工具。同时集成了 QuSpin 中一维费米和玻色子相关的精确对角化功能。

## Installation

- 建立虚拟环境：`conda create -n py312 python=3.12`.

- 激活虚拟环境：`conda activate py312`.

- 安装依赖：`conda install -c conda-forge numba numpy scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler`

- 安装 torch: 确认 gpu 版本，`nvcc --version`, 通过 https://pytorch.org/ 说明安装对应版本 pytorch，cpu 安装命令为：`pip3 install torch torchvision torchaudio`

- 安装 quspin-extensions: `pip install quspin-extensions`

- 进入 setup.py 所在的文件夹：`cd path/to/setup.py`

- 使用开发者模式安装：`pip install -e .`（如果需要安装到本地，可以使用`pip install .`）


