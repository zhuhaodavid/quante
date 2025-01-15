# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-25 22:20:58
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 01:06:24

# 梯度下降的工具

__all__ = ['open_grad', 'close_grad', 'tonp', 'totc', 'clone', 'AdaptiveLRScheduler']

import numpy as np
import torch as tc


if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

def promote_dtype(*datas):
    """将输入数据类型转换为相同的类型
    """
    if any(data.dtype.is_complex for data in datas):
        dtype = tc.complex128
    else:
        dtype = tc.float64
    return [data.to(dtype=dtype) for data in datas]

def open_grad(tensors: list | tc.Tensor) -> None:
    """打开梯度
    
    可以传入张量，或张量列表的嵌套；
    
    将递归的打开列表中所有张量的梯度

    Parameters
    ----------
    tensors : torch.Tensor or list
        要打开张量的列表或张量
        
    Raises
    ------
    TypeError
        只能识别列表或张量

    Examples
    --------
    >>> x = tc.tensor([1., 2.], dtype=tc.float64)
    >>> open_grad(x)
    >>> x.requires_grad
    True
    """
    if isinstance(tensors, list):
        for tensor in tensors:
            open_grad(tensor)
    elif isinstance(tensors, tc.Tensor):
        tensors.requires_grad = True
    else:
        raise TypeError("Expected a list or a torch.Tensor")


def close_grad(tensors: list | tc.Tensor) -> None:
    """关闭梯度
    
    可以传入张量，或张量列表的嵌套；
    
    将递归的关闭列表中所有张量的梯度
    
    Parameters
    ----------
    tensors : torch.Tensor or list
        要关闭张量的列表或张量
    
    Raises
    ------
    TypeError
        只能识别列表或张量
        
    Examples
    --------
    >>> x = tc.tensor([1., 2.], dtype=tc.float64, requires_grad=True)
    >>> close_grad(x)
    >>> x.requires_grad
    False
    """
    if isinstance(tensors, list):
        for tensor in tensors:
            close_grad(tensor)
    elif isinstance(tensors, tc.Tensor):
        tensors.requires_grad_(False)
    else:
        raise TypeError("Expected a list or a torch.Tensor")


def tonp(tensors: list | tc.Tensor | np.ndarray) -> list | np.ndarray:
    """将输入变量转换为 NumPy 数组
    
    接受（嵌套的） NumPy 数组、PyTorch 张量（带或不带梯度）的变量，并将其转换为 NumPy 数组。
    
    如果有梯度，梯度会丢掉。
    
    Parameters
    ----------
    input_var : numpy.ndarray or torch.Tensor
        可能是（嵌套的） NumPy 数组、PyTorch 张量（带或不带梯度）的变量。
        
    Returns
    -------
    numpy.ndarray
        转换后的 NumPy 数组。
        
    Raises
    ------
    TypeError
        输入类型不支持。
    
    Examples
    --------
    >>> x = [tc.randn(3), tc.randn(3)]
    >>> tonp(x)
    [array([-1.4201413 , -0.21939032, -0.31446534], dtype=float32),
     array([0.4318766, 0.021245 , 1.7427114], dtype=float32)]
    """
    if isinstance(tensors, np.ndarray):
        # 如果输入已经是 NumPy 数组，则直接返回
        return tensors
    elif isinstance(tensors, tc.Tensor):
        # 如果输入是 PyTorch 张量，则将其转换为 NumPy 数组
        if tensors.is_cuda:
            # 如果张量在 GPU 上，则先将其移动到 CPU
            tensors = tensors.cpu()
        if tensors.requires_grad:
            # 如果张量带有梯度，则先分离梯度
            tensors = tensors.detach()
        return tensors.numpy()
    elif isinstance(tensors, list):
        # 如果输入是列表，则递归地将每个元素转换为 NumPy 数组
        return [tonp(x) for x in tensors]
    else:
        raise TypeError(f"Unsupported input type: {type(tensors)}")


def totc(data: list | np.ndarray | tc.Tensor, dtype=None, device=None) -> list | tc.Tensor:
    """将输入数据转换为 PyTorch 张量
    
    接受（嵌套的） NumPy 数组、PyTorch 张量（带或不带梯度）的变量，并将其转换为 PyTorch 张量。

    Parameters
    ----------
    data : list | np.ndarray | tc.Tensor
        要转换的数据。
    dtype : torch.dtype, optional
        数据类型, by default None
    device : torch.device, optional
        设备, by default None

    Returns
    -------
    list | tc.Tensor
        转换后的 PyTorch 张量。

    Raises
    ------
    TypeError
        数据类型不支持。
        
    Examples
    --------
    >>> x = [np.array([1., 2.]), np.array([3., 4.])]
    >>> totc(x)
    [tensor([1., 2.], dtype=torch.float64), tensor([3., 4.], dtype=torch.float64)]
    """
    if isinstance(data, tc.Tensor):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, np.ndarray):
        return tc.from_numpy(data).to(dtype=dtype, device=device)
    elif isinstance(data, list):
        return [totc(x, dtype=dtype, device=device) for x in data]
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")


def clone(tensor):
    """与 deepcopy 完全相同"""
    from copy import deepcopy
    return deepcopy(tensor)


class AdaptiveLRScheduler:
    def __init__(self, optimizer, factor=0.1, patience=10, threshold=0.0001, cooldown=0, min_lr=1e-6, max_lr=0.1, increase_factor=1.5, increase_patience=10):
        """根据记录的 loss 动态调整学习率
        
        学习率会在指标没有改善的情况下减小，但会在一定次数的连续减小较小时增加，以防止学习率过大。

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            要调整学习率的优化器
        factor : float, optional
            下降学习率的比例, by default 0.1
        patience : int, optional
            忍耐期，在这期间如果指标没有改善则不会调整学习率, by default 10
        threshold : float, optional
            指标变化的最小阈值, by default 0.0001
        cooldown : int, optional
            冷却时间，在这段时间内不会调整学习率, by default 0
        min_lr : _type_, optional
            学习率的下限, by default 1e-6
        max_lr : float, optional
            学习率的上限, by default 0.1
        increase_factor : float, optional
            学习率增加的比例, by default 1.5
        increase_patience : int, optional
            连续减小较小的次数，在这些次数内损失减小较小时才增加学习率, by default 10
            
        Examples
        --------
        >>> optimizer = tc.optim.Adam(model.parameters(), lr=0.1)
        >>> scheduler = AdaptiveLRScheduler(optimizer, factor=0.1, patience=10, threshold=0.0001, cooldown=0, min_lr=1e-6, max_lr=0.1, increase_factor=1.5, increase_patience=10)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     scheduler.step(val_loss)
        """
        self.optimizer = optimizer  # 要调整学习率的优化器
        self.factor = factor  # 学习率减小的比例
        self.patience = patience  # 忍耐期，在这期间如果指标没有改善则不会调整学习率
        self.threshold = threshold  # 指标变化的最小阈值，只有超过这个阈值的变化才会被认为是改善
        self.cooldown = cooldown  # 冷却时间，在这段时间内不会调整学习率
        self.min_lr = min_lr  # 学习率的下限
        self.max_lr = max_lr  # 学习率的上限
        self.increase_factor = increase_factor  # 学习率增加的比例
        self.increase_patience = increase_patience  # 连续减小较小的次数，在这些次数内损失减小较小时才增加学习率
        self.cooldown_counter = 0  # 冷却计数器
        self.num_bad_epochs = 0  # 记录指标没有改善的epoch数
        self.loss_queue = []  # 记录最近指标变化的队列
        self.best = None  # 记录指标的最优值

    def step(self, metrics):
        current = float(metrics)  # 当前的指标值
        if self.best is None:
            self.best = current  # 初始化最优值为当前值
            self.loss_queue.append(current)
            return
        
        # 将当前损失添加到队列中
        self.loss_queue.append(current)
        if len(self.loss_queue) > self.increase_patience:
            self.loss_queue.pop(0)

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1  # 冷却期未结束时递减计数器
            self.num_bad_epochs = 0
            self.loss_queue = []
            return

        if current < self.best:
            self.best = current  # 更新最优值
            self.num_bad_epochs = 0
            self.num_small_improvement_epochs = 0
        else:
            self.num_bad_epochs += 1  # 增加没有改善的epoch计数

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()  # 如果没有改善的epoch计数达到忍耐期，减少学习率
            self.best = current
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
            
        # 检查队列中所有变化是否都小于阈值
        if len(self.loss_queue) == self.increase_patience:
            # all_small_improvements = (max(self.loss_queue) - min(self.loss_queue)) < self.threshold
            all_small_improvements = all(
                abs((self.loss_queue[i] - self.loss_queue[i - 1])/self.loss_queue[i]) < self.threshold
                for i in range(1, len(self.loss_queue))
            ) and all((self.loss_queue[i] - self.loss_queue[i - 1]) < 0 for i in range(1, len(self.loss_queue)))
            if all_small_improvements:
                self._increase_lr()
                # self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                self.loss_queue = []  # 重置队列


    def _reduce_lr(self):
        for param_group in self.optimizer.param_groups:
            plr = param_group['lr']
            new_lr = max(plr * self.factor, tc.tensor(self.min_lr, dtype=plr.dtype, device=plr.device))  # 按比例减少学习率但不低于下限
            param_group['lr'] = new_lr
            # print(f'Reduced learning rate to {new_lr}')

    def _increase_lr(self):
        for param_group in self.optimizer.param_groups:
            plr = param_group['lr']
            new_lr = min(plr * self.increase_factor, tc.tensor(self.max_lr, dtype=plr.dtype, device=plr.device))  # 按比例增加学习率但不高于上限
            param_group['lr'] = new_lr


def create_high_identity(dims:list, device:tc.device, dtype=tc.float64) -> tc.Tensor:
    """生成一个高阶的 delta 张量
    
    Parameters
    ----------
    dims : list
        张量的维度
    device : torch.device
        张量所在的设备
    dtype : torch.dtype, optional
        张量的数据类型, by default tc.float64

    Returns
    -------
    torch.Tensor
        高阶 delta 张量

    Examples
    --------
    可以生成一个三阶二维的 delta 张量
    
    >>> create_high_identity([2,2,2])
    """
    assert tc.all(
        tc.tensor(dims) == dims[0]
    ), f"elements in dims {dims} are supposed to be the same"

    delta_tensor = tc.zeros(*dims, dtype=dtype, device=device)
    for n in range(0, dims[0]):
        delta_tensor[n, n, n] = tc.tensor(1, dtype=dtype, device=device)

    return delta_tensor


