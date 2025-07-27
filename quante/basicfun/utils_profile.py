# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:13:13
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-27 17:52:01

import os as _os
import gc as _gc
import sys as _sys
import numpy as _np
import scipy as _sp
import time as _time
import traceback as _traceback
import builtins

from itertools import chain
from collections import deque
from typing import Callable, Any

_os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # 为了让 ctrl+c 中断程序可用
_os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 为了避免 scipy svd 与 gpu torch 冲突
_np.set_printoptions(linewidth=1000000, suppress=True) # 为了让 print 出的矩阵的宽度没有限制


__all__ = [
    "profile",
    "Timer",
    "print_memory_usage",
    "clear",
]

__profiles = []

def profile(
    on: bool=True, 
    save:bool=False, 
    output_unit:float|None=None
) -> Callable:
    """测试函数的运行时间
    
    在 notebook 中，建议使用 Timer 的上下文管理器。

    Parameters
    ----------
    on : bool, optional
        是否启用当前测试, by default True
    save : bool, optional
        是否保存测试结果, by default False
    output_unit : float, optional
        测试结果的输出单位, by default None

    Returns
    -------
    Callable
        返回装饰器
    
    Examples
    --------
    >>> @qt.basicfun.profile()
    >>> def test_func():
    >>>     a = 1
    >>>     b = 2
    >>>     c = a + b
    >>>     return c
    >>> 
    >>> @qt.basicfun.profile(False)
    >>> def outf():
    >>>     for i in range(1000):
    >>>         test_func()
    >>> 
    >>> outf()
    """
    if not on:
        return lambda func: func
    if not __profiles:
        from line_profiler import profile as lp
        # 设置环境变量和配置
        _os.environ["LINE_PROFILE"] = "1"
        lp.write_config.update({
            'lprof': False,
            'text': False,
            'timestamped_text': save,
            'stdout': not save,
        })
        lp.show_config.update({
            'output_unit': output_unit,
            'details': 1 if not save else None
        })
        __profiles.append(lp)
        return lp
    return __profiles[0]


class Timer:
    def __init__(self, *str_or_funcs, output_unit: float|None = None, save=False, level=1):
        """通过上下文管理器记录函数的执行时间.

        Parameters
        ----------
        functions : None | str | Callable
            要记录时间的函数, None 记录进出之间时间，str 记录时间并答应 str，Callable 记录时间并执行该函数。
        timer_unit : float | None, optional
            时间单位，默认根据实际运行时间自动调整, by default None
        save : bool, optional
            是否保存到文件中, by default False
        
        Examples
        --------
        >>> def test():
        >>>     time.sleep(1)
        >>>     return 1
        >>> 
        >>> def test2():
        >>>     time.sleep(1)
        >>>     return 1
        >>> 
        >>> with qt.basicfun.Timer(test,test2):
        >>>     a = test()
        >>>     b = test2()
        """
        if len(str_or_funcs) == 0:
            self.only_time = True
            self.string = "Time elapsed"
            self.level = level
            return
        elif len(str_or_funcs)==1 and isinstance(str_or_funcs[0], str):
            self.string = str_or_funcs[0]
            self.only_time = True
            self.level = level
            return
        else:
            self.only_time = False
        from line_profiler import LineProfiler
        self.profile = LineProfiler(*str_or_funcs)
        self.functions = str_or_funcs
        self.output_unit = output_unit
        self.save = save
    
    def logging(self, message: str) -> None:
        """记录日志信息."""
        from .utils_logging import logger
        if self.level == 1:
            logger.info(message)
        elif self.level == 2:
            logger.warning(message)
        elif self.level == 3:
            logger.error(message)
        elif self.level == 4:
            logger.critical(message)
        else:
            logger.debug(message)

    def __enter__(self):
        self.start_time = _time.perf_counter()  # 记录开始时间
        if not self.only_time:
            self.profile.enable()  # 开始分析时间
            return self.profile

    def __exit__(self, exc_type, exc_value, traceback_obj):
        elapsed_time = _time.perf_counter() - self.start_time  # 计算经过的时间
        if self.only_time:
            self.logging(f"{self.string}: {elapsed_time} seconds")
            if exc_type is not None:  # 检查是否发生错误
                _traceback.print_exc()  # 打印堆栈跟踪
            return elapsed_time
        self.profile.disable()  # 停止分析
        if self.output_unit is None:
            if elapsed_time < 0.0001:  # 如果小于 0.0001 秒，则认为是微秒级
                self.output_unit = None  # 微秒
            elif elapsed_time < 0.1:  # 如果小于 0.1 秒，则认为是毫秒级
                self.output_unit = 1e-3  # 毫秒
            else:  # 其他情况，使用秒
                self.output_unit = 1  # 秒
        if self.save:
            filename = "profile_" + _os.path.basename(_sys.argv[0])[:-3] + '.txt'  # 根据运行的脚本文件名生成日志文件名 
            header = "#"*60+'\n' + f"[{_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(_time.time()))}]\n" + f"Profiling result for {len(self.functions)} functions: {[i.__name__ for i in self.functions]}\n" + "#"*60+'\n\n'
            with open(filename, 'a') as f:
                # 写入当前时间
                f.write(header)
                # 写入性能分析结果
                original_stdout = _sys.stdout
                _sys.stdout = f
                self.profile.print_stats(output_unit=self.output_unit)  # 打印执行时间统计信息
                _sys.stdout = original_stdout
        else:
            self.profile.print_stats(_sys.stdout, output_unit=self.output_unit)  # 打印出性能分析结果
        
        if exc_type is not None:  # 检查是否发生错误
            _traceback.print_exc()  # 打印堆栈跟踪


def print_memory_usage(obj: Any) -> None:
    """ 打印对象的占用空间的大小.
    todo: 增加对一般类的支持
    
    Parameters
    ----------
    obj : Any
        需要测量内存占用的对象。
    
    Examples
    --------
    >>> import numpy as np
    >>> from quante.basicfun import test_memory
    >>> a = [np.random.randn(10,10) for i in range(100)]
    >>> print_memory_usage(a)
    79.02 KB
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    seen = set()  # 追踪已经处理过的对象ID，防止重复计算

    def sizeof(obj_inner: Any) -> int:
        """递归计算对象的大小（包括其子元素）。"""
        if id(obj_inner) in seen:  # 防止重复计算同一对象
            return 0
        seen.add(id(obj_inner))
        size = _eachsize(obj_inner)

        for typ, handler in all_handlers.items():
            if isinstance(obj_inner, typ):
                size += sum(map(sizeof, handler(obj_inner)))
                break
        return size

    size = float(sizeof(obj))
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            print(f"{size:.2f} {unit}")
            return None
        size /= 1024


def _eachsize(obj: Any) -> int:
    """计算对象的直接大小，不包括引用的子对象。"""
    if isinstance(obj, _np.ndarray):
        return obj.nbytes
    elif isinstance(obj, (_sp.sparse.csr_matrix, _sp.sparse.csr_array)):
        return obj.data.nbytes + obj.indptr.nbytes + obj.indices.nbytes
    elif isinstance(obj, (_sp.sparse.csc_matrix, _sp.sparse.csc_array)):
        return obj.data.nbytes + obj.indptr.nbytes + obj.indices.nbytes
    elif isinstance(obj, (_sp.sparse.coo_matrix, _sp.sparse.coo_array)):
        return obj.data.nbytes + obj.row.nbytes + obj.col.nbytes
    elif str(type(obj)) == "<class 'torch.Tensor'>":
        return obj.element_size() * obj.numel()
    else:
        return _sys.getsizeof(obj)

def clear(name: Any) -> None:
    """只有当非常在意内存的时候才需要使用这个函数，因为强制清除内存会降低性能"""
    del name
    _gc.collect()

