# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-25 22:20:58
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-24 22:39:05

# 梯度下降的工具

__all__ = ['AdaptiveLRScheduler', 'open_grad', 'close_grad', 'to_numpy_array', 'convert_to_torch', 'clone_list']
__all__ += ['save_h5', 'load_h5']

import numpy as np
import torch as tc


if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

dtype = tc.complex128



def open_grad(tensors):
    if isinstance(tensors, list):
        for tensor in tensors:
            open_grad(tensor)
    elif isinstance(tensors, tc.Tensor):
        tensors.requires_grad = True
    else:
        raise TypeError("Expected a list or a torch.Tensor")


def close_grad(tensors):
    if isinstance(tensors, list):
        for tensor in tensors:
            close_grad(tensor)
    elif isinstance(tensors, tc.Tensor):
        tensors.requires_grad_(False)
    else:
        raise TypeError("Expected a list or a torch.Tensor")


def to_numpy_array(input_var):
    """
    将输入变量转换为 NumPy 数组。

    参数:
    input_var: 可能是 NumPy 数组、PyTorch 张量（带或不带梯度）的变量。

    返回:
    NumPy 数组。
    """
    if isinstance(input_var, np.ndarray):
        # 如果输入已经是 NumPy 数组，则直接返回
        return input_var
    elif isinstance(input_var, tc.Tensor):
        # 如果输入是 PyTorch 张量，则将其转换为 NumPy 数组
        if input_var.is_cuda:
            # 如果张量在 GPU 上，则先将其移动到 CPU
            input_var = input_var.cpu()
        if input_var.requires_grad:
            # 如果张量带有梯度，则先分离梯度
            input_var = input_var.detach()
        return input_var.numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_var)}")


def convert_to_torch(list_of_arrays: list[np.ndarray], dtype=None, device=None) -> list[tc.Tensor]:
    converted_list = []
    for array in list_of_arrays:
        if isinstance(array, np.ndarray):
            tensor = tc.from_numpy(array).to(dtype=dtype, device=device)
            converted_list.append(tensor)
        else:
            converted_list.append(array.to(dtype=dtype, device=device))
    return converted_list


def clone_list(ini_tensor: list) -> list:
    len_list = len(ini_tensor)
    new_list = [None] * len_list
    for n in range(len_list):
        if ini_tensor[n] is not None:
            new_list[n] = ini_tensor[n] * 1.0
    return new_list
            # print(f'Increased learning rate to {new_lr}')



class AdaptiveLRScheduler:
    def __init__(self, optimizer, factor=0.1, patience=10, threshold=0.0001, cooldown=0, min_lr=1e-6, max_lr=0.1, increase_factor=1.5, increase_patience=10):
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


def create_high_identity(dims, device, dtype=tc.float64) -> tc.Tensor:
    """
    生成一个高阶的 delta 张量，如
    >>> create_high_identity([2,2,2])
    可以生成一个三阶二维的 delta 张量
    """
    assert tc.all(
        tc.tensor(dims) == dims[0]
    ), f"elements in dims {dims} are supposed to be the same"

    delta_tensor = tc.zeros(*dims, dtype=dtype, device=device)
    for n in range(0, dims[0]):
        delta_tensor[n, n, n] = tc.tensor(1, dtype=dtype, device=device)

    return delta_tensor

# 一个简化的 save_hdf5 和 load_hdf5

import os as _os
import inspect as _inspect
import logging as _logging
import h5py as _h5py
from typing import Dict, Any, Union
from ..basicfun import PrintLn, save_hdf5, _LOAD_FUNC, _load_dict, _default_load

def save_h5(filename:str, *data, group:Union[list[str],str] = [], mode:str='a') -> None:
    """
    简化的 save_hdf5

    Example:
    >>> import numpy as np
    >>> import quante.basicfun as bf
    >>> mat = np.random.randn(10,10)
    >>> bf.save_h5("data.h5", mat)
    """
    assert filename[-3:] == ".h5", "use h5 for consistance"
    if len(data) == 1 and isinstance(data[0], dict):
        data_dic = data[0]
    else:
        current_frame = _inspect.currentframe()
        assert current_frame is not None and current_frame.f_back is not None, "Can't get the caller's frame"
        paraname = PrintLn._get_paraname(current_frame.f_back)
        data_dic = dict()
        for i, arg in enumerate(data):
            if type(arg).__name__ == "type":
                data_dic[paraname[i+1]] = arg()
            else:
                data_dic[paraname[i+1]] = arg
    if isinstance(group, str):
        group = [group]
    assert isinstance(group, list) and "/" not in group
    group_name = "/".join(group)
    save_hdf5(filename, group_name, data_dic, mode=mode)


def load_h5(filename:str, *datanames, group=[]) -> Union[Dict[str, Any], list[Any]]:
    """
    简化的 load_hdf5

    Example:
    >>> import numpy as np
    >>> import quante.basicfun as bf
    >>> mat = np.random.randn(10,10)
    >>> bf.save_h5("data.h5", mat)
    >>> mat, = bf.saveh5("data.h5", "mat")
    """
    _logging.info("Loading from " + _os.path.abspath(filename) + " ... ")
    if isinstance(group, str):
        group = [group]
    assert isinstance(group, list) and "/" not in group
    group = "/".join(group)
    
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."
        group_location = f[group]  # 获取组对象
        if len(datanames) == 0:
            data: Union[Dict[str, Any], list[Any]] = _load_dict(group_location)
        else:
            data = []
            for dataname in datanames:
                group_location = f[group]  # 获取组对象
                data_location = group_location[dataname]
                data_type_str = data_location.attrs.get("object_type", None)
                if data_type_str is None and isinstance(data_location, _h5py.Group):
                    data_type_str = 'dict'
                load_func = _LOAD_FUNC.get(data_type_str, _default_load)
                data.append(load_func(data_location))
            if len(datanames) == 1:
                data = data[0]
    _logging.info("Load done")
    return data


