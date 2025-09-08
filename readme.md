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

`quante` 提供了 `save_hdf5` 和 `load_hdf5` 函数来保存和加载数据。

```python
import numpy as np
qt.basicfun.save_hdf5(
    "data.h5", 
    {
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

同时提供了 `isave` 和 `iload` 函数可以更加方便的保存和加载数据。

通过 `isave` 和 `iload` 函数可以加载数据。

```python
a, b, c = qt.basicfun.isave("data.h5", group='test')
a, b, c = qt.basicfun.iload("data.h5", group='test')
```

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
ham.show_string_form(form='v')
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
mat = qt.generate.matrix.heisenberg_matrix(L=10, sparse=True)
x0 = qt.generate.state.random(mat.shape[0])
val, vec , _ = qt.linalg.krylov.eigsolve(
    mat, x0, howmany=1, which='SR', isherm=True
)
val
```

```
running Lanczos ...
[-4.25803521 -3.93067359]
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
hammat = ham.to_matrix(basis=basis, sparse=True)

tlist = np.linspace(0, 10, 200)
init_state = qt.generate.state.neel(L=L, down_first=True, Nup=L//2)
obsoper = [op.z(i).to_matrix(basis=basis, sparse=True) for i in range(L)]

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

op = qt.generate.operas.spin

L = 30
J = 1.
gamma_R = 1.0
gamma_L = 0.5

ham = op.builder()
for i in range(L-1):
    ham += "+-", [i+1, i], -J
    ham += "+-", [i, i+1], -J
ham = ham.build()
Lindblad_R = [np.sqrt(gamma_R) * op.pm(i+1,i) for i in range(L-1)]
Lindblad_L = [np.sqrt(gamma_L) * op.pm(i,i+1) for i in range(L-1)]

basis = qt.generate.basis.spin_basis(L=L, Nup=1)
lvn = qt.generate.superoper.make_Liouvillian(ham, Lindblad_R + Lindblad_L, basis)

state = qt.generate.state.product_state(['up']+['dn']*(L-1), Nup=1)
rhoinit = np.outer(state, state)
particle_number = [op.n(i).to_matrix(basis=basis, sparse=True) for i in range(L)]

res = qt.linalg.evolve_and_measure(
    lvn, rhoinit, [10, 20, 30, 40, 50], 
    measure=particle_number, 
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

Parts of this package (quante/linalg/krylov/) are based on the Julia package KrylovKit.jl,
which is licensed under the MIT License.
The original license is included in quante/linalg/krylov/LICENSE.KrylovKit
