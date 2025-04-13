# Quante

quante 是处理一维自旋量子系统的 python 工具集合，包含了一维自旋量子系统的精确对角化、张量网络以及数据处理保存相关的工具。同时集成了 QuSpin 中一维费米和玻色子相关的精确对角化功能。

## Installation

- 建立虚拟环境：`conda create -n py312 python=3.12`.

- 激活虚拟环境：`conda activate py312`.

- 安装依赖：`conda install -c defaults numba numpy scipy matplotlib h5py cytoolz psutil tqdm ipykernel ipympl line_profiler`

- 安装 torch: cpu 安装命令为：`pip3 install torch torchvision torchaudio`；gpu 安装时先确认 cuda 版本，`nvcc --version`, 再通过 [torch 主页](https://pytorch.org/) 查看相应的按照命令。

- 进入 setup.py 所在的文件夹 `cd path/to/setup.py` 后，使用开发者模式安装：`pip install -e .`（如果需要安装到本地，可以使用`pip install .`）

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

可以寻找上一个赋值语句 (10 行以内) 的左值并输出。

```python
a, b = 1, [1, 2, 3]
show()
```

```
a: 1; b: [1, 2, 3]
```

`show` 通过 `logging` 模块输出，可以调整输出内容和位置

```python
qt.basicfun.set_logging(savelog=False, logtime=True)
```

### save and load

`quante` 提供了 `isave` 和 `iload` 函数来保存和加载数据。

```python
a, b = [1,2,3], 1
qt.basicfun.isave("data.h5", a, b)
```

将建立一个 `data.h5` 文件，`/` 组中包含一个 `a` 数据集和一个 `b` 数据集。

如果不填写数据，那么会自动寻找上一个赋值语句 (10 行以内) 的左值并保存，如：

```python
a, b = [1,2,3], 1
qt.basicfun.isave("data.h5") # 与上面等价
```

如果需要自定义数据集名称，需要使用 keyword 参数 `dataset`：

```python
a, b = [1,2,3], 1
qt.basicfun.isave("data.h5", data={"data1": a, "data2": b})
```


通过 `iload` 函数可以加载数据。

```python
data = qt.basicfun.iload("data.h5")
```

将会 load 数据为一个字典

```python
b, = qt.basicfun.iload("data.h5")
```

则只会 load `b` 数据集。

如果需要自定义数据集名称，可以提供字符串名称：

```python
data1, data2 = qt.basicfun.iload("data.h5", ('a', 'b'))
```

### Exact Diagonalization

生成一维自旋系统的哈密顿量和精确对角化。

提供类似功能的库 [**`QuSpin`**](https://github.com/QuSpin/QuSpin), [**`quimb`**](https://github.com/jcmgray/quimb), [**`qutip`**](https://github.com/qutip/qutip)

自旋方法存放在 `quante.generate.operas` 中。

```python
op = qt.generate.operas
```

以 Heisenberg 模型为例：

#### 方法 1

```python
L = 4
ham = op.heisenberg_operator(L)
ham
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
ham = op.sum(op.xx(i,i+1) + op.yy(i,i+1) + op.zz(i,i+1) for i in range(L-1))
```

#### 方法 3
```python
L = 4
builder = op.SpinOperBuilder()
for i in range(L-1):
    builder += 1., 'x', i, 'x', i+1
    builder += 1., 'y', i, 'y', i+1
    builder += 1., 'z', i, 'z', i+1
ham = builder.build()
```

通过 `basis` 获得基矢，进而获得哈密顿量的表示矩阵。

```python
basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
hammat = ham.to_matrix(basis=basis)
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

### Evolve

基于 `scipy` 的 `expm_multiply`, 提供了一个可以利用 GPU 的高效的演化工具。

一个计算实例：

```python
L = 10
tlist = np.linspace(0, 10, 200)

# Model
J, γ = 1., 0.
builder = op.SpinOperBuilder()
for l in range(L-1):
    builder += 1/2 * (J + γ), 'p', l+1, 'm',   l
    builder += 1/2 * (J - γ), 'p',   l, 'm', l+1
ham = builder.build()

basis = qt.generate.basis.spin_basis(L=L, Nup=L//2)
obsoper = [op.z(i) for i in range(L)]
init_state = qt.generate.state.neel(L=L, down_first=True, basis=basis)

res = ham.evolve(init_state, tlist, obsoper, basis=basis)
res
```

```
array([[-0.5       , -0.49936897, -0.49747906, ...,  0.14443265,  0.15040737,  0.15657966],
       [ 0.5       ,  0.49873807,  0.49496024, ...,  0.07763028,  0.07828669,  0.07840081],
       [-0.5       , -0.4987382 , -0.49496236, ...,  0.10006989,  0.09211759,  0.08461557],
       ...,
       [ 0.5       ,  0.4987382 ,  0.49496236, ..., -0.10006989, -0.09211759, -0.08461557],
       [-0.5       , -0.49873807, -0.49496024, ..., -0.07763028, -0.07828669, -0.07840081],
       [ 0.5       ,  0.49936897,  0.49747906, ..., -0.14443265, -0.15040737, -0.15657966]])
```

会根据尺寸自动选择合适的方法，如果需要手动调优，可以参考 `examples/evolve.ipynb` 中的例子。

### Tensor Network

`quante` 提供了一个基于 `torch` 的张量网络计算工具（没有支持对称性）。

类似的解决方案，如 [**`ITensors`**](https://github.com/ITensor/ITensors.jl), [**`tenpy`**](https://github.com/tenpy/tenpy), [**`quimb`**](https://github.com/jcmgray/quimb) 

```python
import quante.torch_utils as qtc
```

生成随机的 MPS:

```python
L = 12
𝜓 = qtc.MPS.from_random(L, bond_dim=4, dtype=tc.float64, device='cpu')
𝜓.show()
```

```
MPS;  torch.float64;  norm: 1.096e+05;  maxbonddim: 4;  device: cpu;
physdim:    2|    2|    2|    2|    2|    2|    2|    2|    2|    2|    2|    2| 
         ----O-----O-----O-----O-----O-----O-----O-----O-----O-----O-----O-----O----
bonddim:  1     4     4     4     4     4     4     4     4     4     4     4     1
site:        0     1     2     3     4     5     6     7     8     9     10    11
```

计算局域观测量

```python
𝜓.measure('x',0)
```

```
tensor(-0.0180, dtype=torch.float64)
```

DMRG 

```python
L = 10
ham = op.heisenberg_operator(L)
H = ham.to_mpo(pauli=False)
eng, vec = H.dmrg(nsweep=10)
ham.gdenergy()
```

```
Sweep 1: 100%|##########| 18/18 [00:00<00:00, 304.74it/s, pE=-4.2575e+00, chi=8]
Sweep 2: 100%|##########| 18/18 [00:00<00:00, 148.58it/s, pE=-4.2580e+00, chi=20]
Sweep 3: 100%|##########| 18/18 [00:00<00:00, 162.75it/s, pE=-4.2580e+00, chi=20]
Sweep 4: 100%|##########| 18/18 [00:00<00:00, 155.70it/s, pE=-4.2580e+00, chi=20]
Energy converged to -4.2580352068 after 4 sweeps.
```

## Third-Party Licenses

This project uses the QuSpin library, which is licensed under the BSD 3-Clause License. 
The license file can be found at `quante/generate/basis/quspin/LICENSE.rst`.