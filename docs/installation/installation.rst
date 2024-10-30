.. _installation-label:

Installation
============

quante 需要 `numpy <https://numpy.org/doc/stable/reference/index.html#reference>`_, `scipy <https://docs.scipy.org/doc/scipy/reference/index.html#scipy-api>`_, `numba <https://numba.readthedocs.io/en/stable/index.html>`_, `h5py <https://docs.h5py.org/en/stable/index.html>`_, `matplotlib <https://matplotlib.org/stable/users/index.html>`_, `torch <https://pytorch.org/docs/stable/index.html>`_ (optional), `line_profiler <https://kernprof.readthedocs.io/en/latest/>`_ (optional), `cupy <https://docs.cupy.dev/en/stable/user_guide/index.html>`_ (optional), `tenpy <https://tenpy.readthedocs.io/en/latest/>`_ (optional) 等库的支持，安装下面步骤配置环境。

配置环境
++++++++++++++++++++++++

安装 conda 环境
#######################

Anaconda 是 python 的环境管理工具；miniconda 是 Anaconda 的轻量版本，可以用于快速安装和配置环境。

它们的下载地址分别是：

  - anaconda：https://www.anaconda.com/download/success

  - miniconda：https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

windows 系统直接从上述网址下载安装即可

Linux 系统（xxx 为版本号，可以在上述网址找到）:

.. prompt:: bash $

    wget wget https://repo.anaconda.com/archive/Anaconda-xxx.sh
    chmod +x Anaconda3-xxx.sh
    ./Anaconda3-xxx.sh

配置 quante 环境
#######################

新建 `conda` 环境，这里名为 `py311`，可根据需要改变：

.. prompt:: bash $

    conda create -n py311 python=3.11 

进入 `py311` 环境：

.. prompt:: bash $

    conda activate py311

安装依赖包（ `numba`, `numpy`, `scipy` 是必需的，其他的根据需要安装）：

.. prompt:: bash $

    conda install numba=0.60 numpy=1. scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler

安装 `quante` 库。首先要进入 `quante` 库所在目录：

.. prompt:: bash $

    cd /path/to/quante

然后安装（这种安装方式会在` site-packages` 目录下建立到 `quante` 库的软链接，如果需要直接安装到 `site-packages` 目录下，可以用 `pip install .`）：

.. prompt:: bash $

    pip install -e .

到此，quante 环境配置完成。下面是安装其他的库：

安装 `torch`，参考 `torch 官方文档 <https://pytorch.org/get-started/locally/>`_ 获得安装命令。如果需要 `gpu` 版本，可以通过：

.. prompt:: bash $

    nvcc --version

查看 `cuda` 版本，然后安装对应版本的 `torch`， `cpu` 版本的命令为：

.. prompt:: bash $

    conda install pytorch torchvision torchaudio cpuonly -c pytorch

`cupy` 的安装(`cupy 官方文档 <https://docs.cupy.dev/en/stable/install.html>`_)：

.. prompt:: bash $

    conda install --channel=conda-forge cupy

`tenpy` 的安装(`tenpy 官方文档 <https://tenpy.readthedocs.io/en/latest/INSTALL.html>`_)：

.. prompt:: bash $

    conda install --channel=conda-forge physics-tenpy

`qutip` 的安装

.. prompt:: bash $

    conda install qutip

兼容 quspin 的配置方法
#######################

由于 `quspin 1.0.0` 要求 `numpy>=2.0.0`，而同时 `torch` 不能很好的兼容 `numpy 2.0.0`，因此不能安装在同一环境，下面时同时安装 `quante` 和 `quspin` 的方法。

创建环境：

.. prompt:: bash $

    conda create -n py312 python=3.12

激活该环境：

.. prompt:: bash $

    conda activate py312

安装依赖包：

.. prompt:: bash $

    conda install --channel=conda-forge numba=0.60 numpy=2 scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler

使用 `pip` 安装` quspin`，同时保持 `numpy` 版本（ `numba=0.60` 最高支持到 `numpy=2.0.1`）：

.. prompt:: bash $

    pip install quspin numpy==2.0.1

进入 `quante` 库所在目录：

.. prompt:: bash $

    cd /path/to/quante

安装 `quante` 库：

.. prompt:: bash $
    
    pip install -e .

到此， `quspin` 环境配置完成。这个环境中安装如 `torch`, `tenpy` 都会失败

conda 中的常用命令
++++++++++++++++++++++++

* 查看已安装环境列表：

    .. prompt:: bash $

	    conda env list

* 删除环境

    .. prompt:: bash $

	    conda remove -n envname --all

* 删除环境

    .. prompt:: bash $

	    conda create -n envnew --clone envold

* 查看已安装的包：
  
    .. prompt:: bash $

	    conda list

* 删除没有用的包：
  
    .. prompt:: bash $

	    conda clean -p

* 查看安装历史：
  
    .. prompt:: bash $

	    conda list --revisions

* 恢复安装历史：
  
    .. prompt:: bash $

	    conda install --revision N

安装 slepc4py
++++++++++++++++++++++++

