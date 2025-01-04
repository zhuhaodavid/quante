# quante

## Anaconda

### Anaconda3/miniconda3:

#### Linux: 

- Download `wget https://repo.anaconda.com/archive/Anaconda-xxx.sh`

  - anaconda 下载官网：https://www.anaconda.com/download/success

  - miniconda 下载官网：https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html

- 赋权 `chmod +x Anaconda3-xxx.sh`

- Install: `./Anaconda3-xxx.sh`

### conda basic operations:

- 查看环境：conda env list

- 删除环境：conda remove -n envname --all

- 复制环境：conda create -n envnew --clone envold

- 查看已安装的包：conda list

- 删除没有用的包：conda clean -p

- 查看安装历史：conda list --revisions

- 恢复安装历史：conda install --revision N

- 查看 channel: conda config --show channels

- 移除 channel: conda config --remove channels channel_name

## package dependence (python=3.11) new version

（下一次更新提醒：numba 更新 0.61?，conda 更新 scipy 1.14？）

- `conda create -n envname python=3.11`, create new virtual python envrionment.

- `conda activate envname`, login that virtual environment.

- `conda install numba=0.60 numpy=1 scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler` 

- `cd xxx` 进入 setup.py 所在的文件夹

- `pip install -e .` 到此已经完成安装，可以 `import quante as qt`; 如果想要安装到本地，使用 `pip install .`; 如果卸载 使用 `pip uninstall quante`

接下来是 torch tenpy 的安装（不需要安装下面的包也能运行 quante）

- `nvcc --version`, check version `release xxx`. https://pytorch.org/ `conda install pytorch ...`, according to CUDA version. For cpu,  `conda install pytorch torchvision torchaudio cpuonly -c pytorch`

- `conda install -c conda-forge cupy` cupy

- `conda install --channel=conda-forge physics-tenpy` tenpy

- `conda install qutip` qutip

## package dependence (python=3.12) for quspin new version

quspin 1.0.0 要求 numpy>=2.0.0

但 torch 要求 numpy <= 2.0.0，因此不能同时安装，下面时同时安装 quante 和 quspin 的方法。

- `conda create -n envname python=3.12`, create new virtual python envrionment.

- `conda activate envname`, login that virtual environment.

- `conda install --channel=conda-forge numba=0.60 numpy=2 scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler` 支持 numpy=2.0 的 scipy=1.14 暂时只能通过 conda-forge 获取

- `pip install quspin numpy==2.0.1` 在保持 numpy 版本的情况下安装 quspin， numpy=2.0.1 是因为 numba=0.60 最高支持到 2.0.1

- `cd xxx` 进入 setup.py 所在的文件夹

- `pip install -e .` 到此已经完成安装，可以 `import quante as qt`; 如果想要安装到本地，使用 `pip install .`; 如果卸载 使用 `pip uninstall quante`

安装完 quspin 其他的包，如 torch, tenpy 都会安装失败

## Git

- 创建新分支 `git checkout -b branchname`

- 查看某一分支 `git checkout branchname`

- 创建分支 `git branch branchname`

- 合并分支 `git merge branchname`

- 上传到 github `git push origin branchname`

- 从 github 上下载 `git pull origin branchname`

- 查看历史记录：`git log`

- 回到某个历史版本：`git reset --hard <commit_hash>`

- 查看历史版本：`git checkout <commit_hash>`

## install slepc4py

slepc is suggested to be used on linux or wsl on windows

### wsl

If linux is already available, this can be skipped.

- first check the Hyper-V on on windows 10/11. This can be done by first search `Turn Windows features on or off`/`启用或关闭windows功能`, check the `Hyper-V` and `Windows Subsystem for Linux`/`适用于Linux的Windows子系统`. Restart the computer.

- `wsl --install` in cmd, and then follow the instruction to complete installation (the default is Ubuntu. for CentOS, refer to <https://github.com/mishamosher/CentOS-WSL/releases>).

- move the wsl destination (optional). first input `wsl --shutdown` in cmd. the change the file and the path at `Computer\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Lxss`

### ensure gcc

ensure is installed and `gcc --verion` is greater than (suggest v11.4.0). if not it should be installed by one of following method

#### with root

##### For CentOS

- `yum -y update`

- `yum remove gcc yum remove gdb` and ensure `gcc -v` `g++ -v` `gdb` will all raise error.

- `sudo yum install scl-utils`

- `sudo yum install centos-release-scl`

- `yum list all --enablerepo='centos-sclo-rh' | grep "devtoolset"` to check the install

- `sudo yum install -y devtoolset-11-toolchain`

- `sudo scl enable devtoolset-11 bash` (for local user, this should be added to its own .bashrc file, or alternatively, the root user can append `PATH=$PATH::/opt/rh/devtoolset-11/root/usr/bin` `export PATH`  `sudo scl enable devtoolset-11 bash` to `/etc/profile`)

##### For Ubuntu

`gcc --version` should equal and larger than v11.4.0. Otherwise:

- `sudo apt-get update`

- `sudo apt-get install build-essential gdb` (check the success by `whereis g++`)

- `sudo apt install cmake`

- `sudo apt-get install libblas-dev liblapack-dev`

#### without root

This part is mainly from <https://blog.csdn.net/weixin_38505222/article/details/120967948> and this process may take a lot time

- download gcc source file from <https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/>, `wget https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-11.4.0/`

- `tar -zxvf gcc-11.4.0.tar.gz`

- `cd gcc-11.4.0`

- `./contrib/download_prerequisites`

- `cd ..`

- `mkdir gcc-11.4.0-build`

- `cd gcc-11.4.0-build`

- `../gcc-11.4.0/configure --enable-checking=release --enable-languages=c,c++ --disable-multilib --prefix=/home/yourname/myinstall/gcc-11.4.0`, if there is problem after, run `../gcc-11.4.0/configure --enable-checking=release --enable-languages=c,c++ --disable-multilib --prefix=/home/yourname/myinstall/gcc-11.4.0 --disable-libsanitizer` instead

- `make -j4`

- `make install -j4`

- `vim ~/.bashrc`

- add:
  
  ```bash
  export PATH=/home/yourname/myinstall/gcc-8.5.0/bin:/home/yourname/myinstall/gcc-8.5.0/lib64$PATH  
  export LD_LIBRARY_PATH=/home/yourname/myinstall/gcc-8.5.0/lib64:/home/yourname/myinstall/gcc-8.5.0/lib$LD_LIBRARY_PATH
  ```

- `source ~/.bashrc`

- `gcc --version` check version

### ensure cmake

suggest v3.22 or higher, check with `cmake --version`

#### For CentOS (with root)

- `yum remove cmake -y`

- `yum install openssl`

- `yum install openssl-devel`

- <https://cmake.org/files>, `wget https://cmake.org/files/v3.27/cmake-3.27.4.tar.gz`

- `tar xf cmake-3.27.4.tar.gz`

- `cd cmake-3.27.4`...

- `./bootstrap`

- `gmake`

- `gmake install`

- `ln -s /usr/local/bin/cmake /usr/bin/cmake`

#### For Ubuntu (with root)

- `sudo apt install cmake`

### install openmpi

quimb suggest v1.10.x for the most robust for spawning processes

- `wget https://download.open-mpi.org/release/open-mpi/v1.10/openmpi-1.10.7.tar.gz` from site <https://www.open-mpi.org/software/ompi/v1.10/>

- `tar -xzvf openmpi-1.10.7.tar.gz`

- `mkdir openmpi-1.10.7-build`

- `cd openmpi-1.10.7-build`

- `../openmpi-1.10.7/configure --prefix=/home/yourname/myinstall/openmpi-1.10.7 --enable-mpi-cxx`

- `make -j4`

- `make install -j4`

- append to `.bashrc`

  ```bash
  export OPENMPI=/home/yourname/myinstall/openmpi-1.10.7
  export PATH=$OPENMPI/bin:$PATH
  export LD_LIBRARY_PATH=$OPENMPI/lib:$LD_LIBRARY_PATH
  export INCLUDE=$OPENMPI/include/:$INCLUDE
  export CPATH=$OPENMPI/include/:$CPATH
  export MANPATH=$OPENMPI/share/man:$MANPATH
  ```

- `source .bashrc`

- `which mpirun` to check install

### install mpi4py

v2.1.0+

- `wget https://github.com/mpi4py/mpi4py/releases/download/3.1.5/mpi4py-3.1.5.tar.gz` from <https://github.com/mpi4py/mpi4py/releases/>

- `tar -zxvf mpi4py-3.1.5.tar.gz`

- `cd mpi4py-3.1.5`

- `vim mpi.cfg`

- change openmpi part:

  ```bash
    # Open MPI example
    # ----------------
    [openmpi]
    mpi_dir              = /home/yourname/myinstall/openmpi-1.10.7
    mpicc                = %(mpi_dir)s/bin/mpicc
    mpicxx               = %(mpi_dir)s/bin/mpicxx
    include_dirs         = %(mpi_dir)s/include
    libraries            = mpi
    library_dirs         = %(mpi_dir)s/lib
    runtime_library_dirs = %(library_dirs)s
  ```

- `conda install -c conda-forge ld_impl_linux-64`

- `python setup.py build --mpi=openmpi`

- `python setup.py install --prefix=/home/yourname/myinstall/miniconda3/envs/envname`

### install PETSC

- `wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.20.0.tar.gz` from <https://petsc.org/release/install/download/>

- `tar xf petsc-3.20.0.tar.gz`

- `mkdir /home/yourname/myinstall/petsc-3.20.0`

- `mv petsc-3.20.0 /home/yourname/myinstall/petsc-3.20.0/complex-double`

- `cd /home/yourname/myinstall/petsc-3.20.0/complex-double`

- `export PETSC_ARCH=arch-linux-c-debug-complex-double`

- install PETSC by `./configure --download-scalapack --download-mumps --download-metis --download-parmetis --with-precision=double --with-scalar-type=complex`. the `--with-fftw=1 --download-fftw --download-scalapack --download-mumps` are optional. the scalar type can also set to be real, but complex seems to be required by quimb. `--with-debugging=0` can be added to close debugging but not checked.

- `make ... all` by the instruction to follow the installation

### install petsc4py

- `cd /home/yourname/myinstall/petsc-3.20.0/complex-double`. 

- `export PETSC_DIR=/home/yourname/myinstall/petsc-3.20.0/complex-double`

- `PETSC_ARCH=arch-linux-c-debug-complex-double python -m pip install src/binding/petsc4py` (wait a bit long ...)

### install SLEPC

should be the same version as petsc

- `wget https://slepc.upv.es/download/distrib/slepc-3.20.0.tar.gz` from <https://petsc.org/release/install/download/>

- `tar -zxvf slepc-3.20.0.tar.gz`

- `mkdir /home/yourname/myinstall/slepc-3.20.0`

- `mv slepc-3.20.0 /home/yourname/myinstall/slepc-3.20.0/complex-double`

- `cd /home/yourname/myinstall/slepc-3.20.0/complex-double`

- `export PETSC_ARCH=arch-linux-c-debug-complex-double`

- `export PETSC_DIR=/home/yourname/myinstall/petsc-3.20.0/complex-double`

- `export SLEPC_DIR=/home/yourname/myinstall/slepc-3.20.0/complex-double`

- `./configure`

- `make ... all`

### install slepc4py

- `cd /home/yourname/myinstall/slepc-3.20.0/complex-double/src/binding/slepc4py`. 

- `python setup.py install`

## install quimb

- `pip install quimb`

## 关于 vscode 颜色的一些设置：

### self 等关键字颜色

其中 scope 通过 >developer: Inspect Editor Tokens and Scopes 获得
```
"editor.tokenColorCustomizations": {
    "textMateRules": [
        {
        "scope": "variable.parameter.function.language.special.self.python",
        "settings": {"foreground": "#FF0000"}
        },
        {
        "scope": "variable.parameter",
        "settings": {"foreground": "#FF0000"}
        }
    ]
},
```

### 关于安装显卡驱动

要升级NVIDIA CUDA编译器驱动程序（即`nvcc`），你需要升级CUDA Toolkit和相应的NVIDIA显卡驱动程序。以下是详细步骤：

### 1. 升级NVIDIA显卡驱动程序
#### （1）卸载旧版驱动程序：
1. 打开“控制面板”->“程序”->“程序和功能”。
2. 找到与NVIDIA相关的驱动程序，卸载它们（例如`NVIDIA Graphics Driver`、`NVIDIA GeForce Experience`等）。
3. 重启计算机。

#### （2）下载并安装新版本驱动程序：
1. 前往[NVIDIA官方网站](https://www.nvidia.com/Download/index.aspx)。
2. 选择你的显卡型号和操作系统，下载最新的驱动程序。
3. 运行安装程序，按照提示完成安装。选择“自定义安装”，并勾选“执行干净安装”选项，以确保完全更新。
4. 安装完成后，重启计算机。

### 2. 升级CUDA Toolkit
#### （1）卸载旧版CUDA Toolkit：
1. 打开“控制面板”->“程序”->“程序和功能”。
2. 找到并卸载旧版的CUDA Toolkit（如`NVIDIA CUDA Toolkit 11.x`）。
3. 重启计算机。

#### （2）下载并安装新版CUDA Toolkit：
1. 前往[NVIDIA CUDA Toolkit下载页面](https://developer.nvidia.com/cuda-downloads)。
2. 选择你需要的CUDA Toolkit版本、操作系统、架构和安装类型（通常选择“exe (local)”）。
3. 运行下载的安装程序，并按照提示进行安装。
4. 安装过程中，可以选择是否安装CUDA Samples、CUDA Visual Studio Integration等组件。

### 3. 配置环境变量
如果你安装了新版CUDA Toolkit，并且希望使用新的CUDA编译器驱动程序，需要更新系统的环境变量：

#### （1）更新`Path`变量：
1. 打开“系统属性”->“高级系统设置”->“环境变量”。
2. 在“系统变量”下找到并选择`Path`变量，然后点击“编辑”。
3. 确保新的CUDA Toolkit路径在`Path`变量中，并且优先于旧版本的路径。例如，假设你安装了CUDA 12.x：
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
   - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp`
4. 删除旧版CUDA Toolkit路径，或者将其移到`Path`变量的下方。

### 4. 验证升级
打开命令提示符，输入以下命令来验证升级：
```bash
nvcc --version
```
你应该会看到新的CUDA编译器驱动程序的版本信息。

### 5. 更新cuDNN（可选）
如果你在使用cuDNN（CUDA Deep Neural Network library），请确保下载与新CUDA版本兼容的cuDNN版本，并按照与上文类似的步骤安装。

### 6. 验证PyTorch的GPU支持
如果你使用PyTorch，请确保你安装了对应CUDA版本的PyTorch包。你可以通过以下命令更新PyTorch：
```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu12x
```
将`cu12x`替换为对应的CUDA版本号。

### 7. 检查Python环境中的CUDA版本
最后，可以在Python中运行以下代码，检查是否可以成功调用新的CUDA版本：
```python
import torch
print(torch.version.cuda)  # 输出当前PyTorch支持的CUDA版本
print(torch.cuda.is_available())  # 检查CUDA是否可用
```

如果输出显示正确的CUDA版本，并且`torch.cuda.is_available()`返回`True`，那么你的NVIDIA CUDA编译器驱动程序升级成功。