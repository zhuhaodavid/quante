# Change Log

## 2024-10-09

- 调整了 quante 的整体结构

- 增加了 symmetry 中的对称性处理的方法

- 增加了 torch_util 中的张量网络处理方法

- 改进了 quante 的安装方法

注：目前版本相对稳定，基本不会改动


## 2024-08-21

hzhu

- .operas.py 重写了 Oper 类

## 2024-08-15

hzhu

- ./free_model 中分为 anderson_model, free_fermion 两个子文件加。相关的公式存入了 md 文件中。

## 2024-08-13

hzhu

- ./tensor/algorithm.py, ./tensor/mps.py 中引用 torch 改为延迟引用：

  具体来说：

  类型标注改为字符串，

  利用 'TYPE_CHECKING' 判断是否应用 torch,

  函数内部引用 torch


## 2024-08-07

hzhu

- ./basicfun.py 中 save_hdf5 增加对 idataclass 的支持

  注：现在 save_hdf5 支持自动提取变量名，如 save_hdf5("file.h5", a, b)，将自动生成：{"a": a, "b",b}，进行储存

- ./basicfun.py 中 log 函数改名为 println

  原因：log名字不好

## 2024-08-06

hzhu

- 增加 ./init.pyi, ./basicfun.pyi, ./opers.pyi

  原因：为代码提示提供更好的行为，这些 pyi 文件都是 mypy 生成，如 `stubgen __init__.py`，这样与 numpy 行为一致, `stubgen __init__.py`

- ./init.py 中将其他没有关系的包注释掉

  原因：这样和 numpy 的 __init__.py 是同样的写法

  注：如果需要一次性引用全部，可以用 `from bbb import *`，其中 `bbb.py` 在 ../bbb.py 中

  也可以用 aaa, ccc，按a,b,c的顺序，需要的时间递减，aaa 包含了 torch, matplotlib, 而 bbb 包含了 torch，ccc 只提供最基本的功能

- ./init.py 中增加 import linalg, myplot, matrix

  原因：这样和 numpy 的 __init__.py 是同样的写法

- 将 ./myplot/dataplot.py 中的所有 plt import 全部放在函数内部，这样初始化时不需要引用 plt


## 2024-07-22

hzhu

- ./tctensor/tcclass.py 中 MPO 增加了 trace 方法：

- ./tctensor/tcclass.py 中 MPO 增加了 exp 方法：

  用泰勒展开计算 exp

  注：因为可调参数太多，这只是一个示范代码

- ./tctensor/tcclass.py 中 MPS, MPO 改正了 applied_by 方法的问题

  原因：原先的版本中对 lognm 的处理有问题

- ./basicfun.py 中增加 getsizeof 方法

  目的：获得变量内存占用

## 2024-07-21

hzhu

- ./tctensor/tcclass.py 中 MPS, MPO 类增加 applied_by 方法

  目的：实现 MPO @ MPS, MPO @ MPO

- ./tctensor/tcclass.py 中 AbstractMPS 类的 norm 方法进行了改正

- ./tctensor/tcfuncs.py 中增加了 tc_eig_truncate 函数

  目的：方便的进行 eig 截断

- ./tctensor/tcclass.py 中 MPS, MPO 类增加方法：apply_gate_2b_EE_naive， apply_gate_2b_EE_eig, apply_gates

  目的：实现两体作用门时不同的收缩方式

  注：
  - apply_gate_2b_EE_naive 直接用除法，在不指定 eps 的情况下可能会出现数值误差
  - apply_gate_2b_EE_eig 使用 eigh 计算左正交矩阵
  - apply_gate_2b_EE 默认使用 svd 的右正交矩阵来计算左正交矩阵，避免除法
  - apply_gates 是一个演示函数，演示如何作用一个门列表

## 2024-07-19

hzhu

- ./oper.py 中 Oper 类中的 gate2_decomp 改名为 gate2_decomposition，并调整它的输出格式

  原因：与 suzuki_trotter_decomposition 格式一致

  注：输出格式改为 位置+算符 的 tuple[list[int],list[_np.ndarray]] 格式，通过 for i, gate in zip(*output) 来调用

- ./oper.py 中 Oper 类增加 suzuki_trotter_decomposition 方法

  目的：对于一个两体哈密顿量，生成它对应的 suzuki-trotter 分解后的两体门，便于做两体演化

  注：order 可以去 1,2,4,'4_opt'，生成的 gates 将演化总共 tau * N_step 的时间

## 2024-07-18

hzhu

- ./opers.py Oper 类增加 gate2_decomp(self, L:int, form="ladder", **kargs) -> list[tuple[int, _np.ndarray]] 方法

  原因：为了方便两体门的 tebd，生成对应的两体门（没有演化）

  注：有 “ladder” 和 “brick” 两种选择 (砖块/阶梯)

- ./opers.py Oper 类的 get_matrix 增加 typing

  原因：减少 vscode pylance 崩溃


## 2024-07-15

hzhu

- 新建 ./tctensor 文件夹，并将 ./tensor/tcfuncs.py 和 ./tensor/tcclass.py 放入其中

  原因：./tensor/__init__.py 中的引用，引起 mps.py, mpo.py 与 tcclass.py 函数名冲突，为了能使用 tcclass.py tcfuncs.py 将其暂时放在独立文件夹中

- 将 ./matrix 与 quspin 解耦，并且简化

  主要内容：
  - 将与量子态无关的函数移至 ./linalg/operations.py
  - 将 pauli_matrix, pauli_matrix 在 .opers.py 中提供接口（其他函数引用需要通过 ./linalg/operations.py 避免交叉引用）
  - 为了避免名称的混乱，./matrix 中的 `spin_basis_1d` 改为 `get_spin_basis`

  注：./matrix 将不在接受任何 quspin 的函数，也不返回任何 quspin 的对象

- 将 ./statistictools.py 移动到 ./linalg 文件夹中

  原因：statistictools.py 不依赖其他包，且与矩阵操作紧密相关

- 对 ./basicfun.py 中的函数做了整理，将不常用的函数移到了 ../PyLib_backup 中

  注：./basicfun.py 中的函数分成了 测试工具 系统层面的函数 hdf5 工具 日志工具 和 字典格式 五类

- 将 ./tick.py 移动到 ../PyLib_backup 中

  原因：里面主要记载一些函数的用法，但对库本身没有意义

- 将 ./accelerated_func.py 移动到 ./linalg 文件夹中

  原因：accelerated_func.py 中主要是 numba 的加速，用于矩阵处理，因此一起归入到 ./linalg 中

  注：不要主动 import accelerated_func.py 否则会极大增加运行时间！

- 将 ./usefulfunc.py 移动到 ./linalg, 并改名为 array_mani.py，原先的 ./linalg/array_mani.py 改名为 ./linalg/array_mani_numba.py

  原因：usefulfunc.py 中主要是矩阵的操作，以及对原先 ./linalg/array_mani.py 的封装，两者放在一起比较合适

- . 中所有的 PyLib 变量（含注释）替换为了 quante 变量

- ./plot/__init__.py 简化

  原因：保证与 ./tensor/__init__.py 的行为一致

- ./qsquspin/__init__.py 中删除了 `from ..importfile import *`

  原因：importfile.py 已删除，否则会报错

dzwang

- 将包的名字改为了 `quante`，引用起来好看，与 `quantum toolkit` 的连续发音相似

- ./__init__.py 的引用改了，清晰的知道函数在那个 py 文件下，便于引用和后续修改、润色、完善
  
  `from quante import *`, `basicfun.xxx`, `from quante.tensor import *, mpo.xxx`

- 后续还需要一起完善润色所有package的 `.` 出来的函数的内容格式等。


## 2024-07-14

hzhu

- ./tensor/tcclass.py 的 AbstractMPS 类中新增了 `save` 方法

  功能：将一个 AbstractMPS 用文件储存下来，数据中包含了重构的所有信息

  注：使用 torch.save 是因为这样可以保存其中所有的梯度数据！（hdf5文件做不到）

- ./tensor/tcclass.py 中增加了 `loadpth` 函数

  功能：下载并重构，AbstractMPS, MPO, MPS 三种函数

- ./__init__.py, ./tensor/__init__.py, ./plot/__init__.py 中删除 `import logging`, 在 ./usepl.py 中增加 `from logging import info, debug, error, warning`

  目的：log 输出时，可以直接使用 `pl.info`, `pl.debug`, `pl.error`, `pl.warning`

- ./tensor/tcclass.py 的 MPO 类中新增了 `diagonal_inner` 方法

  目的：同下，取一个 MPO 的对角元，计算其与另一个 MPS 的内积

- ./tensor/tcfuncs.py 中新增了 `diagonal_inner` 函数

  目的：取一个 MPO 的对角元，计算其与另一个 MPS 的内积

- ./__init__.py 中新增了 `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'`

  原因：在原来的版本中 scipy svd 需要在 import gpu torch 使用，否则程序 scipy svd 会报错。这个设定可以避免这个报错。

  注：这个设定下，如果先 import gpu torch 再进行 scipy svd 可能会出问题，因此在这种情况发生时，将会给出一个 warning 但不影响程序执行


- ./opers.py 中 `expand` 中的参数 `inplace` 默认改为 `False`

  原因：`H.expand()` 可以返回展开的结果，而不会改变 `H`，这样可以通过 `get_matrix().expand()` 来写，而不用拆成两行:
  ```
  H = pl.get_matrix()
  H.expand()
  ```

- ./tensor/automata.py 中 `automata_mpo` 默认不返回伪算符

  原因：伪算符只在 debug 的时候会用到，任何计算任务都不会用到伪算符，无需返回

- ./tensor/automata.py 中 `automata_mpo` 增加 `local_matrix_function` 的选项

  原因：支持自定义的局域算符的输入，如
  ```
  mat: pl.Oper = Op.heisenberg_operator(L=spin_num, j=1., h=0.0)
  f = lambda x: pl.pauli_matrix(x, S='1')
  mat.automata(spin_num, d=3, local_matrix_function=f)
  ```
  就可以实现自旋 `1` 的 automata

- ./tensor/tcclass.py 删除的 `AbstractMPS` 类中的 `full_contract` 的实现，改为 MPS 和 MPO 中具体实现

  原因：MPS,MPO full_contract 返回的结果格式不相同（向量和矩阵），放在同一个函数中不合适

## 2024-07-11

hzhu

- ./tensor/tcclass.py 中 `MPO` 类增加了 `apply_gate_2b_EE(self,pos:int,gate_2b_top:typing.Union[tc.Tensor, None] = None,gate_2b_bottom:typing.Union[tc.Tensor, None] = None,Dc: int = None,eps: float = None,normalize=False)` 方法

  目的：利用纠缠谱计算两体门作用

  注：
  - EE 表示纠缠谱
  - 使用这个函数时要保证是 is_canonical_form

- ./tensor/tcclass.py 中 `MPS` 类增加了 `apply_gate_2b_EE` 方法 中 `pos` 与 `gate_2b` 的参数位置做了调换

  目的：与 MPO 中的格式保持一致

- ./tensor/tcclass.py 中 `MPO` 类增加了 `apply_gate_2b(self,pos:int,gate_2b_top:typing.Union[tc.Tensor, None] = None,gate_2b_bottom:typing.Union[tc.Tensor, None] = None,dir: str = "right",Dc: int = None,eps: float = None,updateS=False,normalize=False,)` 方法

  目的：通过移动正交中心返回两体门的作用结果

  注：
  - pos 指两体门左侧的位置
  - 其中 gate_2b_top 从上面向下作用，gate_2b_bottom 从下面向上作用（不做转置）
  - dir 指的是作用后得到 ---▷---⬜--- (right) 还是 ---⬜---⨞---- (left)
  - updateS 表示是否保存过程的中的奇异谱

- ./tensor/tcclass.py 中增加 `MPO` 类

  目的：继承 `AbstractMPS` 的所有方法，并且能给实现针对 MPO 的方法

- ./tensor/tcfuncs.py 中增加了 `unitarize(gate_2b:tc.Tensor)->tc.Tensor`

  目的：利用 SVD 把一个矩阵变成幺正矩阵

- ./tensor/densitymatrix_method.py 末尾添加关于注释画图的快捷注释

  注：将改注释加入 vscode 代码用户片段可以方便输入制表符！

- ./tensor/automata.py 中删除了一些不用的函数：

  注：
  - 删除的函数包括，关于 hzhu 写的 Sparse to automata MPS and MPO 相关函数，这些函数包括生成 MPS, MPO, 伪矩阵，图示的功能。但这些函数与文件中剩余的函数功能重复，因而删除。
  - 删除了利用 MPO 生成系数矩阵的另一种方法，原因是与剩余的函数功能重复
  - 删除的函数备份在 ../quante_backup/automata 中


## 2024-07-10

hzhu:

- ./tensor/tcclass.py 中 `MPS` 类增加了 `apply_gate_2b_EE(self, gate_2b, pos, Dc: int = None, eps: float = None)` 方法

  目的：利用纠缠谱计算两体门作用

  注：
  - EE 表示纠缠谱
  - 使用这个函数时要保证是 is_canonical_form


- ./tensor/tcclass.py 中 `MPS` 类增加了 `apply_gate_2b(self,gate_2b,pos:int,dir: str = "right",Dc: int = None,eps: float = None,updateS=False)` 方法

  目的：通过移动正交中心返回两体门的作用结果

  注：
  - pos 指两体门左侧的位置
  - dir 指的是作用后得到 ---▷---⬜--- (right) 还是 ---⬜---⨞---- (left)
  - updateS 表示是否保存过程的中的奇异谱


- ./tensor/tcclass.py 中增加 `MPS` 类

  目的：继承 `AbstractMPS` 的所有方法，并且能给实现针对 MPS 的方法

- ./tensor/tcclass.py 中 `AbstractMPS` 类增加了 `orthogonalize(self, j: int, normalize=True)` 方法

  目的：为了方便的将正交中心移动到 `j` 处。

  注：该方法使用 QR 分解，不会产生也不会更新奇异谱

- ./tensor/tcclass.py 中 `AbstractMPS` 类增加了 `copy`, `full_contract`, `canonicalize`, `inner`, `norm` 方法

  注，分别利用 ./tensor/tcfuncs.py 中的 `clone_list`, `full_contract`, `canonicalize`, `tn_inner`, `tn_norm` 函数实现

- ./tensor/tcclass.py 中 `AbstractMPS` 类增加了 `is_orthogonal_form` 和 `is_canonical_form` 方法

  目的：更方便的判断当前数据是否具有 `orthogonal_form` 或 `canonical_form`

- ./tensor/tcclass.py 中 `AbstractMPS` 类增加了 `__rmul__`, `__sub__` 和 `__add__` 方法

  目的：使得能够用代码 `a * psi1 + b * psi2` 计算类似,${ a \left| \psi_1 \right> + b \left| \psi_2 \right> }$ 的表达式，其中 `a`, `b` 是数字，`psi1`, `psi2` 是 `AbstractMPS` 类的实例

  注：
  - 这个函数调用了 ./tensor/tcfuncs.py 中的 `add` 函数
  - `AbstractMPS` 相加会破坏原来两者的正则形式


- ./tensor/tcclass.py 中增加了 `AbstractMPS` 类

  内容：设置了MPS,MPO的一般结构，同时参考了 TenPy 和 ITensor 的写法，具体形式见 `__init__` 函数。

  注：
  - MPS, MPO 是继承 AbstractMPS 的子类
  - 这个类的方法都需要不依赖是 MPS/MPO


- ./tensor 文件夹中增加文件: tcclass.py

  目的：为了方便使用 torch 编写（带梯度的）张量网络程序，将关于 MPS/MPO 的功能集中到一个类中

  特点：
  
  - 此文件只调用 ./tensor/tcfuncs.py，以及 numpy 和 torch 中的函数，不依赖不调用 ./tensor 中任何其他的文件
  - 这个文件中的函数可以被 ./tensor/tcfuncs.py 之外的其他文件调用。
  - 此文件中的所有函数都应保证梯度链。

- ./tensor/tcfuncs.py 中删除了 `_add_mps` 和 `_add_mpo` 函数

  原因：修改了原先调用这两个函数的 `add`，使得 `add` 对 mps 和 mpo 可以使用同一套代码

- ./tensor/tcfuncs.py 中增加了 `_QR`, `_left2right_QR_step`, `_LU`, `_right2left_QR_step` 四个函数

  目的：方便对一个三/四阶张量的第一个或者最后一个指标做 QR 分解，以及方便移动正交中间，增加代码复用

- ./tensor/tcfuncs.py 中删除了 `canonical_form_mps` 和 `canonical_form_mpo` 函数

  原因：修改了原先调用这两个函数的 `canonicalize`，使得 `canonicalize` 对 mps 和 mpo 可以使用同一套代码


- ./tensor/tcfuncs.py 中修改了函数 `canonical_form` 改名为 `canonicalize`，并修改输出内容

  改名原因：itenosr 中通过 qr 移动正交中心的函数名称为 `orthogonalize` ，为了命名形式相同改为 `canonicalize`

  输出改动：由原先的 `Bs`, `Ss` 改为 `Bs`, `Ss`, `lognm`，其中 `Ss` 的长度从 L 改为 L + 1，其第一个和最后一个元素都是 1.

## 2024-07-09

hzhu:

- ./tensor/tcfuncs.py 中修改了函数 `convert_to_torch`

  内容：增加了 `dtype` `device` 两个参数，使得可以用户自定义这两个参数。如果不填，默认系统判断可用的设备，以及虚数类型。

- ./tensor/tcfuncs.py 中修改了函数 `tc_svd_truncate`

  内容：将 `tc.svd` 函数改为 `tc.linalg.svd`，并修改了调用参数，增加了 `tc.linalg.svd` 的注释。

  同时修改了使用到该函数的 `_right_to_left_SVD_step`，因为 `tc.linalg.svd` 返回的 v 矩阵不需要转置共轭

- ./tensor/tcfuncs.py 中新增函数 `tc_svd_truncate(mat:tc.Tensor, Dc:int=None, eps:float=None, compute_uv=True)`

  目的：使用 torch.svd 并加以截断，保证梯度链

  注：
  - 如果不需要 u,v 可以设置 `compute_uv=False` 加快计算

- ./tensor/tcfuncs.py 中新增函数 `canonical_form_mps(Ws:list[tc.Tensor], Dc:int=None, eps=None, qrnormalize=False) -> tuple[list[tc.Tensor], list[tc.Tensor]]`

  目的：使用 torch 完成 mps 的正则化

  注：
  - 返回 Bs, Ss，标准正则化形式为左正交，即由 Bs 组成的 MPS/MPO 正交中心在最左侧
  - 返回的 `Ss` 长度为链长、`Ss[0]` 记录 Ws 的模对数
  - 如果 Ws 不归一，遇到精度问题，可以设置 `qrnormalize=True` 在 QR 中逐步做归一处理

- ./tensor/tcfuncs.py 中新增函数 `canonical_form_mpo(Ws:list[tc.Tensor], Dc:int=None, eps=None, qrnormalize=False) -> tuple[list[tc.Tensor], list[tc.Tensor]]`

  目的：使用 torch 完成 mpo 的正则化

  注：调用和注意与 `canonical_form_mps` 相同

- ./tensor/tcfuncs.py 中新增函数 `open_grad(Ws:tc.Tensor) -> None:`

  目的：快速的打开 Ws 中所有变量的梯度

  注：不返回值

- ./tensor/tcfuncs.py 中新增函数 `add(W1s: list[tc.Tensor], W2s: list[tc.Tensor], alpha: float = 1.0, beta: float = 1.0) -> list[tc.Tensor]`

  目的：实现 MPS/MPO 的加法

  注：会根据输入的 `W1s`, `W2s` 判断调用 `_add_mps` 还是 `_add_mpo`

    

## 2024-07-08

hzhu:

- ./tensor 文件夹中增加文件: tcfuncs.py

  目的：为了方便使用 torch 编写（带梯度的）张量网络程序，将一些常用的函数集中到此文件夹中。

  特点：
  
  - 此文件只调用 numpy 和 torch 两个包，不依赖不调用 ./tensor 中任何其他的包。
  - 这个文件中的函数可以被 ./tensor 中任何其他文件调用。
  - 根据系统 cuda 是否可用，这个文件中的函数统一使用 device = cuda 或 cpu。
  - 此文件中的所有函数都应保证梯度链。
  
- ./tensor/tcfuncs.py 中新增函数 `convert_to_torch(list_of_arrays: list[np.ndarray]) -> list[tc.Tensor]`

  功能：将 `list[np.ndarray]` 格式的数据，如 MPO、MPS 中的 Ws，转换到系统选择的 device 中，格式为 `list[tc.Tensor]`

  注：如果输入变量不是 np.ndarray，那么直接尝试将其放入 device 中

- ./tensor/tcfuncs.py 中新增函数 `clone_list(ini_tensor: list) -> list`

  功能：复制 MPO、MPS 中的 Ws 中的数据。
  
  注：增加这个函数因为不确定 copy 函数对与 torch 变量，尤其是含梯度变量的行为。

- ./tensor/tcfuncs.py 中新增函数 `create_high_identity(dims, real=False) -> tc.Tensor`

  功能：利用 torch 生成高阶的单位矩阵，如：
  `T = create_high_identity([2,2,2])`
  将返回一个三阶张量，其只有形如 `T[i,i,i]` 的元素为1，其他元素都为零。

  默认返回复数合适，设置 `real=True` 可返回实数格式的结果。

- ./tensor/tcfuncs.py 中新增函数 `full_contract(Ws: list[tc.Tensor]) -> tc.Tensor`

  功能：利用 torch 将 MPS(MPO) 完全收缩成为 向量(矩阵)

  注：
  - 根据传入列表中每个张量的结束判断为 MPS(3阶张量) 还是 MPO(4阶张量)
  - 如果传入变量不是3或4阶张量，将返回 NotImplementedError 的报错信息
  - 周期性边界 MPO, MPS 可以返回正确的结果
  - 同样的功能在 ./tensor/densitymatrix_method.py 中的 dm_full_contract 函数也可以实现（使用numpy）
  - 类似的功能在 ./tensor/automata.py 中的 contract 函数也可以实现，但需要指定 type=mps/mpo，并且周期性 MPS/MPO会出现错误

- ./tensor/tcfuncs.py 中新增函数 `tn_inner(Ws1: list[tc.Tensor], Ws2: list[tc.Tensor], logscale=False, conj_at_1=False) -> tc.Tensor`

  功能：利用 torch 计算两个 MPS/MPO 的内积：${ \psi_1^{\dagger} \psi_2 }$ ，或 ${ \operatorname{tr}M_1^{\dagger} M_2 }$ 

  注：
  - 根据传入列表中每个张量的结束判断为 MPS(3阶张量) 还是 MPO(4阶张量)
  - 默认直接收缩，返回内积，如果遇到精度问题，可以设置`logscale=True`，在每一步收缩时提取张量的模避免精度损失，***这时返回的是内积的对数！***
  - 默认直积收缩，不去共轭，如果设置 `conj_at_1=True` 会在计算过程中对第一个参数 Ws1 去共轭再收缩。

- ./tensor/tcfuncs.py 中新增函数 `tn_norm(Ws: list[tc.Tensor], lognorm=False) -> tc.Tensor`

  功能：利用 torch，计算 MPS/MPO 的模

  注：
  - 利用 `tn_inner(Ws, Ws, logscale=lognorm, conj_at_1=True)` 函数实现的
  - 默认计算模。如果遇到精度问题，可以设置 `lognorm=True`，***这时返回的是模的对数！***
  - MPO 的模指的是：${ \sqrt{M^{\dagger} M} }$ 

---
(新的改动在最上方增加，方便阅读)