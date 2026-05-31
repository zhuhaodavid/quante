# Quante

quante 是处理一维自旋量子系统的 python 工具集合，包含了一维自旋量子系统的精确对角化、张量网络以及数据处理保存相关的工具。同时集成了 QuSpin 中一维费米和玻色子相关的精确对角化功能。

## Installation

- 建立虚拟环境：`conda create -n py312 python=3.12`.

- 激活虚拟环境：`conda activate py312`.

- 安装依赖：`conda install -c defaults numba numpy scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler`

- 安装 torch: cpu 安装命令为：`pip3 install torch torchvision torchaudio`；gpu 安装时先确认 cuda 版本，`nvcc --version`, 再通过 [torch 主页](https://pytorch.org/) 查看相应的按照命令。

- 进入 setup.py 所在的文件夹 `cd path/to/setup.py` 后，使用开发者模式安装：`pip install -e .`（如果需要安装到本地，可以使用`pip install .`）

- 其他很好的包：`pip install quspin tenpy dowhen`

## Usage

通过 `import quante as qt` 导入 quante。更多使用方法请参考 `examples` 文件夹中的示例。

### show

通过 `show` 函数可以方便的打印变量的值。

提供类似 [**`icecream`**](https://github.com/gruns/icecream) [**`objprint`**](https://github.com/gaogaotiantian/objprint) 的功能。

```python
a, b = 1, [1, 2, 3]
show(a, b)
```

```
a: 1; b: [1, 2, 3]
```

`show` 通过 `logging` 模块输出，可以调整输出内容和位置

```python
qt.basicfun.set_logging(savelog=False, logtime=True)
```

### save and load

`quante` 提供了 `save_hdf5` 和 `load_hdf5` 函数来保存和加载数据。

```python
import numpy as np
qt.basicfun.save_hdf5(
    "data.h5", 
    data = {
        "a": np.random.randn(2,2),
        "b": [1,2,3],
        "c": {"d": 1, "e": 2}
    },
    group='test'
)
```

将建立一个 `data.h5` 文件，`test` 组中包含三个数据集。

```python
a, b, c = qt.basicfun.load_hdf5("data.h5", ['a', 'b', 'c'], group='test')
```

将会加载 `data.h5` 文件中 `test` 组下的 `a`, `b`, `c` 数据集。

### Generate Matrix

生成一维自旋系统的哈密顿量和精确对角化。

提供类似功能的库 [**`QuSpin`**](https://github.com/QuSpin/QuSpin), [**`quimb`**](https://github.com/jcmgray/quimb), [**`qutip`**](https://github.com/qutip/qutip)

自旋方法存放在 `quante.generate.operas` 中。

```python
op = qt.generate.operas.spin
```

以 Heisenberg 模型为例：

#### 方法 1

```python
L = 4
ham = op.heisenberg_operator(L)
ham.show()
```

```
|   x     x       coef. |   y     y       coef. |   z     z       coef. |
|-----------------------|-----------------------|-----------------------|
|   0     1       1.000 |   0     1       1.000 |   0     1       1.000 |
|   1     2       1.000 |   1     2       1.000 |   1     2       1.000 |
|   2     3       1.000 |   2     3       1.000 |   2     3       1.000 |
```

#### 方法 2
```python
L = 4
builder = op.builder()
for i in range(L-1):
    builder += 'xx', [i, i+1], 1.
    builder += 'yy', [i, i+1], 1.
    builder += 'zz', [i, i+1], 1.
ham = builder.build()
```

#### 方法 3
```python
L = 4
ham = op.sum(op.xx(i,i+1) + op.yy(i,i+1) + op.zz(i,i+1) for i in range(L-1))
```

通过 `basis` 获得基矢，进而获得哈密顿量的表示矩阵。

```python
basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
hammat = ham.to_matrix(basis=basis, pauli=False)
hammat
```

```
array([[ 0.25,  0.5 ,  0.  ,  0.  ,  0.  ,  0.  ],
       [ 0.5 , -0.75,  0.5 ,  0.5 ,  0.  ,  0.  ],
       [ 0.  ,  0.5 , -0.25,  0.  ,  0.5 ,  0.  ],
       [ 0.  ,  0.5 ,  0.  , -0.25,  0.5 ,  0.  ],
       [ 0.  ,  0.  ,  0.5 ,  0.5 , -0.75,  0.5 ],
       [ 0.  ,  0.  ,  0.  ,  0.  ,  0.5 ,  0.25]])
```

`Oper` 支持一些简单的算符运算，例如: 加、减、数乘、算符乘法、求和、取厄密、展开等。

可以利用 `sympy` 进行一些简单的化简，例如
$$
    H = c \vec{S}_i \cdot (\vec{S}_{m} \times \vec{S}_{n})
$$

```python
c = sy.Symbol('c', real=True)
i,m,n = 0,1,2
si = [op.x(i), op.y(i), op.z(i)]
smxsn = [
    op.y(m) * op.z(n) - op.z(m) * op.y(n),
    op.z(m) * op.x(n) - op.x(m) * op.z(n),
    op.x(m) * op.y(n) - op.y(m) * op.x(n),
]
ham = c * op.sum(si[p] * smxsn[p] for p in range(3))
ham = ham.expandxy(pauli=False)
print(ham)
```

得到：
```
pmZ, (0, 1, 2), I*c/4
mpZ, (0, 1, 2), -I*c/4
pZm, (0, 1, 2), -I*c/4
mZp, (0, 1, 2), I*c/4
Zpm, (0, 1, 2), I*c/4
Zmp, (0, 1, 2), -I*c/4
```

这里 `p,m,Z` 分别表示 $\sigma^-, \sigma^{ +}, \sigma^{z}$
`I` 表示虚数单位。

可以进一步利用 `ham.subs({c:1})` 将符号变量 `c` 替换为数值 `1`。


### Krylov Eigsolve

`quante` 提供了基于 Krylov 子空间方法的线性代数工具，主要用于求解大规模稀疏矩阵的特征值问题。

参考 [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) 的代码实现了 Lanczos 和 Arnoldi 算法。

同时支持 numpy, torch-cpu, torch-gpu 的数据格式。

```python
import quante as qt
L = 10
mat = qt.generate.matrix.ising_matrix(L, sparse=True)
print(qt.linalg.krylov.eigsolve(mat, which='SR')[0])
print(qt.generate.solvable.heisenberg.ising_ground_energy(L))
```

```
running Lanczos ...
[-5.28482978]
-5.2848297789078
```

### Evolve

基于 `scipy` 的 `expm_multiply`, 提供了一个可以利用 GPU 的高效的演化工具。

一个计算实例(薛定谔方程演化)：

```python
import quante as qt
import numpy as np
op = qt.generate.operas.spin

L = 10
basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)

J, γ = 1., 0.
builder = op.builder()
for l in range(L-1):
    builder += '+-', [l+1, l], 1/2 * (J + γ),
    builder += '+-', [l, l+1], 1/2 * (J - γ),
ham = builder.build()
hammat = ham.to_matrix(basis=basis, pauli=False, sparse=True)

tlist = np.linspace(0, 10, 200)
init_state = qt.generate.state.neel(L=L, down_first=True, Nup=L//2)
obsoper = [op.z(i).to_matrix(basis=basis, pauli=False, sparse=True) for i in range(L)]

qt.linalg.evolve_and_measure(
    hammat, init_state, tlist,
    measure=obsoper
)
```

```
100%|##########| 200/200 [00:00<00:00, 672.05it/s]
array([[-0.5       ,  0.5       , -0.5       , ...,  0.5       , -0.5       ,  0.5       ],
       [-0.49936897,  0.49873807, -0.4987382 , ...,  0.4987382 , -0.49873807,  0.49936897],
       [-0.49747906,  0.49496024, -0.49496236, ...,  0.49496236, -0.49496024,  0.49747906],
       ...,
       [ 0.14443265,  0.07763028,  0.10006989, ..., -0.10006989, -0.07763028, -0.14443265],
       [ 0.15040737,  0.07828669,  0.09211759, ..., -0.09211759, -0.07828669, -0.15040737],
       [ 0.15657966,  0.07840081,  0.08461557, ..., -0.08461557, -0.07840081, -0.15657966]])
```

如果需要手动调优，可以参考 `examples/evolve.ipynb` 中的例子。

主方程的演化：

```python
import quante as qt
import numpy as np

op = qt.generate.operas

L = 30
J = 1.
gamma_R = 1.0
gamma_L = 0.5

ham = op.builder()
for i in range(L-1):
    ham += "+-", [i+1, i], -J
    ham += "+-", [i, i+1], -J
ham = ham.build()

jump_R = [np.sqrt(gamma_R) * op.pm(i+1,i) for i in range(L-1)]
jump_L = [np.sqrt(gamma_L) * op.pm(i,i+1) for i in range(L-1)]
jump_ops = jump_R + jump_L
liou = op.Lindbladian(L, ham, jump_ops)

basis = qt.generate.basis.spin_basis(L=L, Nup=1)
liou_mat = liou.to_matrix(basis=basis, pauli=False)

state = qt.generate.state.product_state(['up']+['dn']*(L-1), Nup=1)
rhoinit = np.outer(state, state)

particle_number_mat = [
    op.n(i).to_matrix(basis=basis, pauli=False, sparse=True).T.reshape(-1) 
    for i in range(L)
]
measure = lambda t, rho: np.real_if_close([n @ rho.reshape(-1) for n in particle_number_mat])

res = qt.linalg.evolve_and_measure(
    liou_mat, rhoinit, [10, 20, 30, 40, 50], 
    measure=measure, matrix_role="generator"
)

res.shape
```

结果为：
```
100%|##########| 5/5 [00:00<00:00, 79.36it/s]
(5, 30)
```

### Tensor Network

`quante` 提供了一个基于 `torch` 的张量网络计算工具（没有支持对称性）。

类似的解决方案，如 [**`ITensors`**](https://github.com/ITensor/ITensors.jl), [**`tenpy`**](https://github.com/tenpy/tenpy), [**`quimb`**](https://github.com/jcmgray/quimb) 

```python
import quante.bridge.torch_utils as qtc
```

生成随机的 MPS:

```python
import quante.tensornetwork as tn
L = 12
𝜓 = tn.MPS.from_random(L, bond_dim=4)
𝜓.show()
```

```
MPS;  complex128;  norm: 8.760e+06;  maxbonddim: 4;
physdim:    2|    2|    2|    2|    2|    2|    2|    2|    2|    2|    2|    2|                 
         ----O-----O-----O-----O-----O-----O-----O-----O-----O-----O-----O-----O-----
bonddim:  1     4     4     4     4     4     4     4     4     4     4     4     1  
site:        0     1     2     3     4     5     6     7     8     9     10    11
```

计算局域观测量

```python
(𝜓 | ('x', 0) | 𝜓)
```

```
(-27749684652982.14+0.00054931640625j)
```

DMRG 

```python
L = 10
ham = op.heisenberg_operator(L)
H = ham.to_mpo(pauli=False)
eng, vec = H.dmrg(nsweep=10)
ham.gdenergy(pauli=False)
```

```
Sweep 1: 100%|##########| 18/18 [00:00<00:00, 529.41it/s, pE=-4.2577e+00, chi=8] 
Sweep 2: 100%|##########| 18/18 [00:00<00:00, 178.21it/s, pE=-4.2580e+00, chi=20]
Sweep 3: 100%|##########| 18/18 [00:00<00:00, 173.08it/s, pE=-4.2580e+00, chi=20]
Energy converged to -4.2580352068 after 3 sweeps.

-4.25803520728288
```

## todos

- Clifford circuits

- spin basis general/super 支持 Zn、稀疏投影

- spin 基矢的激发态表象，large int 表示

- 玻色基矢、复合基矢

- Krylov evolve and svd

- SU(2) 矩阵高效生成

- 对称性张量网络




## Git

### GitHub 同步流程

- 首先确定自己分支的改动全部提交!!!

- 拉取并提交当前分支的改动
    
    `git pull && git push`

- 添加版本号
    `git tag vx.x.x`
    `git push origin --tags`


### git cmt 风格

以下是一套推荐的 Git 提交信息风格指南：

```
<类型>:

- <模块>: <详细描述>
- <模块>: <详细描述>
- <模块>: <详细描述>

```

- **feat**: 新功能
- **fix**: 修复问题/BUG
- **docs**: 文档更新
- **perf**: 性能优化
- **test**: 增加测试
- **exam**: 修改实例代码
- **style**: 代码格式（不影响功能，例如空格、分号等格式修正）


### Git 常用命令

- 创建新分支 `git checkout -b branchname`

- 查看某一分支 `git checkout branchname`

- 创建分支 `git branch branchname`

- 合并分支 `git merge branchname`

- 上传到 github `git push origin branchname`

- 从 github 上下载 `git pull origin branchname`

- 查看历史记录：`git log`

- 回到某个历史版本：`git reset --hard <commit_hash>`

- 查看历史版本：`git checkout <commit_hash>`

- 合并 commit, 首先通过 git log --oneline 查看 commit 的 hash 值，然后 `git rebase -i <commit_hash>` 进入交互模式，将需要合并的 commit 前面的 pick 改为 squash（保留提交的第一个pick不动），然后保存退出，运行 `git rebase --continue` 完成合并。

- 修改刚才提交的 commit: `git commit --amend`

## Anaconda

### Anaconda3/miniconda3 (Linux):

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

- `sudo apt-get install gfortran`

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
- install PETSC by `./configure --download-metis --download-parmetis --with-precision=double --with-scalar-type=complex`. the `--with-fftw=1 --download-fftw --download-scalapack --download-mumps` are optional. the scalar type can also set to be real, but complex seems to be required by quimb. `--with-debugging=0` can be added to close debugging but not checked.

- `make ... all` by the instruction to follow the installation

### install petsc4py

- `cd /home/yourname/myinstall/petsc-3.20.0/complex-double`. 

- `export PETSC_DIR=/home/yourname/myinstall/petsc-3.20.0/complex-double`

- `PETSC_ARCH=arch-linux-c-debug-complex-double python -m pip install src/binding/petsc4py` (wait a bit long ...)

### install SLEPC

should be the same version as petsc

- `wget https://slepc.upv.es/download/distrib/slepc-3.20.0.tar.gz` from <https://petsc.org/release/install/download/>

- `tar xf slepc-3.20.0.tar.gz`

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


## 曾经的安装流程（最新的安装请参考 readme）

### package dependence (python=3.11) new version

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

### package dependence (python=3.12) for quspin new version

quspin 1.0.0 要求 numpy>=2.0.0

但 torch 要求 numpy <= 2.0.0，因此不能同时安装，下面时同时安装 quante 和 quspin 的方法。

- `conda create -n envname python=3.12`, create new virtual python envrionment.

- `conda activate envname`, login that virtual environment.

- `conda install --channel=conda-forge numba=0.60 numpy=2 scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler` 支持 numpy=2.0 的 scipy=1.14 暂时只能通过 conda-forge 获取

- `pip install quspin numpy==2.0.1` 在保持 numpy 版本的情况下安装 quspin， numpy=2.0.1 是因为 numba=0.60 最高支持到 2.0.1

- `cd xxx` 进入 setup.py 所在的文件夹

- `pip install -e .` 到此已经完成安装，可以 `import quante as qt`; 如果想要安装到本地，使用 `pip install .`; 如果卸载 使用 `pip uninstall quante`

安装完 quspin 其他的包，如 torch, tenpy 都会安装失败