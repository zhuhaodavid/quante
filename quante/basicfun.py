# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-05-02 14:52:59
# @Last Modified by:   hzhu
# @Last Modified time: 2025-02-06 14:20:18

import gc as _gc
import os as _os
import ast as _ast
import sys as _sys
import numpy as _np
import scipy as _sp
import time as _time
import h5py as _h5py
import json as _json
import ctypes as _ctypes
import inspect as _inspect
import logging as _logging
import platform as _platform
import traceback as _traceback
import itertools as _itertools
import builtins

from dataclasses import is_dataclass, asdict
from itertools import chain
from collections import deque
from types import FunctionType
from typing import Callable, Any, Dict, Union

_os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # 为了让 ctrl+c 中断程序可用
_os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 为了避免 scipy svd 与 gpu torch 冲突
_np.set_printoptions(linewidth=1000000, suppress=True) # 为了让 print 出的矩阵的宽度没有限制

# =================
#     测试工具
# =================
__profiles = []

def profile(on: bool=True, save:bool=False, output_unit:float|None=None) -> Callable:
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
    def __init__(self, *str_or_funcs, output_unit: float|None = None, save=False):
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
            return
        elif len(str_or_funcs)==1 and isinstance(str_or_funcs[0], str):
            self.string = str_or_funcs[0]
            self.only_time = True
            return
        else:
            self.only_time = False
        from line_profiler import LineProfiler
        self.profile = LineProfiler(*str_or_funcs)
        self.functions = str_or_funcs
        self.outplut_unit = output_unit
        self.save = save

    def __enter__(self):
        self.start_time = _time.perf_counter()  # 记录开始时间
        if not self.only_time:
            self.profile.enable()  # 开始分析时间
            return self.profile

    def __exit__(self, exc_type, exc_value, traceback_obj):
        elapsed_time = _time.perf_counter() - self.start_time  # 计算经过的时间
        if self.only_time:
            print(f"{self.string}: {elapsed_time} seconds")
            if exc_type is not None:  # 检查是否发生错误
                _traceback.print_exc()  # 打印堆栈跟踪
            return elapsed_time
        self.profile.disable()  # 停止分析
        if self.outplut_unit is None:
            if elapsed_time < 0.0001:  # 如果小于 0.0001 秒，则认为是微秒级
                self.outplut_unit = None  # 微秒
            elif elapsed_time < 0.1:  # 如果小于 0.1 秒，则认为是毫秒级
                self.outplut_unit = 1e-3  # 毫秒
            else:  # 其他情况，使用秒
                self.outplut_unit = 1  # 秒
        if self.save:
            filename = "profile_" + _os.path.basename(_sys.argv[0])[:-3] + '.txt'  # 根据运行的脚本文件名生成日志文件名 
            header = "#"*60+'\n' + f"[{_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(_time.time()))}]\n" + f"Profiling result for {len(self.functions)} functions: {[i.__name__ for i in self.functions]}\n" + "#"*60+'\n\n'
            with open(filename, 'a') as f:
                # 写入当前时间
                f.write(header)
                # 写入性能分析结果
                original_stdout = _sys.stdout
                _sys.stdout = f
                self.profile.print_stats(output_unit=self.outplut_unit)  # 打印执行时间统计信息
                _sys.stdout = original_stdout
        else:
            self.profile.print_stats(_sys.stdout, output_unit=self.outplut_unit)  # 打印出性能分析结果
        
        if exc_type is not None:  # 检查是否发生错误
            _traceback.print_exc()  # 打印堆栈跟踪

builtins.Timer = Timer

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

# ===================================
#     系统层面的函数，如建立文件夹
# ===================================

def clear(name: Any) -> None:
    """只有当非常在意内存的时候才需要使用这个函数，因为强制清除内存会降低性能"""
    del name
    _gc.collect()


def get_free_space(folder: str) -> float:
    """获取磁盘剩余空间。
    
    Parameters
    ----------
    folder : str
        磁盘路径，例如 "D:\\"。
    
    Returns
    -------
    float
        剩余空间，单位为 GB。

    Examples
    --------
    >>> get_free_space("D:\\")
    """
    if _platform.system() == "Windows":
        free_bytes = _ctypes.c_ulonglong(0)
        _ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            _ctypes.c_wchar_p(folder), None, None, _ctypes.pointer(free_bytes)
        )
        return free_bytes.value / (1024 ** 3)
    else:
        import shutil as _shutil
        usage = _shutil.disk_usage(folder)
        return usage.free / (1024 ** 3)
    

def create_folder(path1:str, path2: Union[None, str]=None) -> str:
    """创建一个文件夹（包括路径中指定的所有父文件夹）。
    
    Parameters
    ----------
    path1 : str
        主路径，可以是类似 "xxx/xxx/..." 的格式。
    path2 : Union[None, str], optional
        子路径，必须为 None 或者类似 "xxx" 的字符串。
    
    Returns
    -------
    str
        创建的完整路径，以斜杠结尾。
    
    Examples
    --------
    >>> import quante as qt
    >>> path1 = "data/entanglement"
    >>> path2 = "XXZ"
    >>> create_folder(path1, path2)
    '.../data/entanglement/XXZ/'
    """
    whole_path = _os.path.abspath(path1).replace("\\", "/") + "/"  # 规范化主路径
    whole_path = _os.path.join(whole_path, path2).replace("\\", "/") if path2 else whole_path  # 如果提供了子路径，则加入到主路径中
    _os.makedirs(whole_path, exist_ok=True)  # 创建文件夹（包括所有父文件夹）
    return whole_path if whole_path.endswith("/") else whole_path+"/"   # 确保返回路径以斜杠结尾
 
#############################################################
#  日志工具
#############################################################

# 创建自定义的日志记录器
logger = _logging.getLogger('quante_logger')

def set_logging(level: int = 1, savelog: bool = False, filenameTime: bool = False, logtime: bool = False, showlevel=False):
    """配置日志记录功能.
    
    Parameters
    ----------
    level : int, optional
        日志记录的级别，可以填 -1, 1, 2, 3, 4，分别对应 debug, info, warning, error, critical，默认为 1。
    savelog : bool, optional
        是否将日志保存到文件中，默认为 `False`。
    filenameTime : bool, optional
        是否在日志文件名中包含时间戳，默认为 `False`。
    logtime : bool, optional
        是否在日志消息中包含时间戳，默认为 `False`。
    
    Returns
    -------
    None: 该函数无返回值。
    
    Examples
    --------
    >>> set_logging(savelog=True, logtime=True)
    """
    # 清除已有的处理器，防止重复添加
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    _format = ""
    if logtime:  # 根据 `logtime` 参数设置日志格式
        _format += "%(asctime)s"
    if showlevel:
        _format += " %(levelname)s"
    if _format:
        _format += ": %(message)s"
    else:
        _format += "%(message)s"

    if savelog:
        filename = "log/"
        create_folder("log/")  # 创建日志目录
        filename += _os.path.basename(_sys.argv[0])[:-3]  # 根据运行的脚本文件名生成日志文件名
        if filenameTime:  # 如果 `filenameTime` 为 `True`，在文件名中添加时间戳
            now = "_" + _time.strftime("%Y-%m-%d-%H_%M_%S", _time.localtime(_time.time()))
            filename += now
        filename += '.log'
        file_handler = _logging.FileHandler(filename, mode="w", encoding='utf-8')
        file_handler.setFormatter(_logging.Formatter(_format))
        logger.addHandler(file_handler)
        try:
            println.use_color = False
        except:
            pass
    else:
        console_handler = _logging.StreamHandler()
        console_handler.setFormatter(_logging.Formatter(_format))
        logger.addHandler(console_handler)
        try:
            println.use_color = True
        except:
            pass
    
    # 设置日志记录级别
    assert level in [-1, 1, 2, 3, 4], "Invalid log level, should be in [-1, 1, 2, 3, 4]"
    logger.setLevel({-1:_logging.DEBUG, 1:_logging.INFO, 2:_logging.WARNING, 3:_logging.ERROR, 4:_logging.CRITICAL}[level])
    logger.propagate = False  # 防止日志消息传播到root日志记录器

set_logging()  # 使用默认日志记录器

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # logger.debug("发生错误！这是固定的提示信息。")
    tb_str = "".join(_traceback.format_exception(exc_type, exc_value, exc_traceback))
    logger.error(tb_str)  # 输出格式化的traceback字符串

# 下面这句话可以让 logging 记录所有的报错信息
_sys.excepthook = custom_exception_handler

class COLOR:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    DEFAULT = '\033[39m'
    
class PrintLn:
    """
    打印输入变量的名称和值到日志中。
    
    任何以 f" 或 rf" 开头的字符串会被识别为 f-string，而不会被解析为变量名称。
    
    可以通过在 builtins.pyi 中添加：

    >>> def Timer(self, *str_or_funcs, output_unit: float|None = None, save=False) -> None: ...
    >>> def show(*values) -> None: ...

    使得 ide 可以识别 show 和 Timer 函数。

    Examples
    --------
    >>> from quante.basicfun import println as show
    >>> a = "this is a test"
    >>> show(a)
    a: this is a test
    >>> a;-show
    a: this is a test
    >>> L, a = 1, 2 ;show>>1
    L: 1; a: 2
    >>> _= L, a ;show>>1
    L: 1; a: 2
    
    Warning
    -------
    可能存在的问题：
    - 只能在单行中使用，否则报 SyntaxError。
    """
    def __init__(self, use_color=True):
        self.use_color = use_color

    def __rshift__(self, mode=1):
        if not mode:
            return None
        try:
            cf = _inspect.currentframe()  # 获取调用函数的栈帧
            if cf is None:
                raise ValueError("Can't get the caller's frame")
            paraname, values = PrintLn._get_paraname_value(cf.f_back)
            if paraname == []:
                return None
            out: str = self._constructArgumentOutput(paraname,values)
            if mode == 1:
                logger.info(out)
            elif mode == -1:
                logger.debug(out)
            elif mode == 2:
                logger.warning(out)
            elif mode == 3:
                logger.error(out)
            else:
                logger.critical(out)
        except SyntaxError as e:
            logger.warning("SyntaxError")
        
        logger.handlers[0].flush()  # 立即刷新日志

    @staticmethod
    def _get_paraname_value(callFrame):
        """利用 inspect 和 ast 模块获取变量名称"""
        # 获取调用函数的源代码
        frame_info = _inspect.getframeinfo(callFrame)
        if frame_info is None or frame_info.code_context is None:
            return [], []
        source_code = "".join(frame_info.code_context).strip()

        # 解析为 AST 并查找函数调用的节点
        tree = _ast.parse(source_code)

        node = next(_itertools.islice(_ast.walk(tree), 3, 4))

        local_vars = callFrame.f_locals
        global_vars = callFrame.f_globals

        if _ast.unparse(node) == "_":
            values = eval("_", global_vars, local_vars)
            node = next(_itertools.islice(_ast.walk(tree), 4, 5))
            if isinstance(node, _ast.Tuple):
                paraname = PrintLn.split_expression(_ast.unparse(node)[1:-1])
                return paraname, values
            else:
                paraname = PrintLn.split_expression(_ast.unparse(node))
                return paraname, (values, )

        if isinstance(node, _ast.Tuple):
            paraname = PrintLn.split_expression(_ast.unparse(node)[1:-1])
        else:
            paraname = PrintLn.split_expression(_ast.unparse(node))
        
        return paraname, [eval(arg, global_vars, local_vars) for arg in paraname]

    @staticmethod
    def split_expression(expression):
        # 初始化计数器和结果列表
        brackets_counter = 0
        quote_counter_1 = 0
        quote_counter_2 = 0
        
        result = []
        current_part = []
        
        # 从左向右循环处理字符串
        for char in expression:
            # 遇到逗号且所有计数器为零，将当前部分添加到结果列表
            if char == ',' and brackets_counter == 0 and quote_counter_1 == 0 and quote_counter_2 == 0:
                part = ''.join(current_part).strip()
                if part:
                    result.append(part)
                current_part = []
                continue
            
            current_part.append(char)
            
            # 更新计数器
            if char in '([{':
                brackets_counter += 1
            elif char in ')]}':
                brackets_counter -= 1
            if char == '"':
                quote_counter_1 = 1 - quote_counter_1
            if char == "'":
                quote_counter_2 = 1 - quote_counter_2
        
        # 添加最后一部分
        part = ''.join(current_part).strip()
        if part:
            result.append(part)
        
        return result

    
    def __call__(self, *inputargs, level=1):
        if level == 0:
            return None
        try:
            cf = _inspect.currentframe()  # 获取调用函数的栈帧
            if cf is None:
                raise ValueError("Can't get the caller's frame")
            paraname = PrintLn._get_paraname(cf.f_back)
            out: str = self._constructArgumentOutput(paraname, inputargs)
            if level == 1:
                logger.info(out)
            elif level == -1:
                logger.debug(out)
            elif level == 2:
                logger.warning(out)
            elif level == 3:
                logger.error(out)
            else:
                logger.critical(out)
        except SyntaxError as e:
            logger.warning("SyntaxError")
            logger.warning(inputargs)
        
        logger.handlers[0].flush()  # 立即刷新日志


    @staticmethod
    def _get_paraname(callFrame):
        """利用 inspect 和 ast 模块获取变量名称"""
        # 获取调用函数的源代码
        frame_info = _inspect.getframeinfo(callFrame)
        if frame_info is None or frame_info.code_context is None:
            return []
        source_code = "".join(frame_info.code_context).strip()
        
        # 解析为 AST 并查找函数调用的节点
        tree = _ast.parse(source_code)
        
        for node in _ast.walk(tree):
            
            if isinstance(node, _ast.Call):  # 查找函数调用节点 
                arg_values = []
                # 提取参数
                for arg in node.args:
                    # 使用 ast.unparse 来获取参数的源代码表示
                    arg_values.append(_ast.unparse(arg))
                return arg_values
        return []

    def _constructArgumentOutput(self, paraname, inputargs):
        if len(paraname) == 0:
            return ""
        if len(paraname) == 1 and self._isLiteral(paraname[0]):
            if isinstance(inputargs[0], str):
                return inputargs[0]
            return self._argumentToString(inputargs[0], use_color=self.use_color)[0]
        
        pairs = [(arg, *self._argumentToString(val, use_color=self.use_color)) for arg, val in zip(paraname, inputargs)]
        
        pairStrs = [val if self._isLiteral(arg) else PrintLn.set_color(f"{arg}: ", COLOR.RED, self.use_color) + val for arg, val, _ in pairs]
        allArgsOnOneLine = PrintLn.set_color(f"; ", COLOR.RED, self.use_color).join(pairStrs)
        
        multilineArgs = len(allArgsOnOneLine.splitlines()) > 1
        firstLineTooLong = len(allArgsOnOneLine.splitlines()[0]) > 70
        
        if multilineArgs or firstLineTooLong:
            lines = []
            for i, (arg, value, isobject) in enumerate(pairs):
                prefixTooLong =  (len(pairs[i][0]) > 8) or isobject
                lines.append(self._format_pair(arg, value, prefixTooLong))
        else:
            lines = [allArgsOnOneLine]
        
        return "\n".join(lines)

    @classmethod
    def _argumentToString(cls, obj, compact=False, indent=0, use_color=True) -> tuple[str, bool]:
        """如果是 ndarray，那么输出它的 __str__，否则用 pprint 得到字符串"""
        import pprint
        isobject = False
        if type(obj) == _np.ndarray:
            if compact:
                s = _np.array2string(obj, formatter={'float_kind': lambda x: "%.2f" % x}, max_line_width=None, threshold=4)
            else:
                s = obj.__str__()
        elif isinstance(obj, list):
            s = pprint.pformat(obj, compact=True)
        elif isinstance(obj, FunctionType):
            return f"<function {obj.__name__}>", False
        else:
            # object
            if (obj.__class__.__str__ is not object.__str__ or obj.__class__.__repr__ is not object.__repr__):
                s = pprint.pformat(obj)
            else:
                s = cls._get_custom_object_str(obj, use_color)
                isobject = True
        s = s.replace("\\n", "\n")  # Preserve string newlines in output.
        s = '\n'.join(v if i == 0 else " "*indent + v for i, v in enumerate(s.split('\n')))
        return s, isobject

    @staticmethod
    def set_color(s: str, color: str, use_color: bool = True) -> str:
        if use_color:
            return f"{color}{s}{COLOR.DEFAULT}"
        else:
            return s

    @classmethod
    def _get_custom_object_str(cls, obj: Any, use_color=True):
        import inspect
        # 首先拿到 header, footer
        obj_type = type(obj)
        header = PrintLn.set_color(f"<{obj_type.__name__} {hex(id(obj))}", COLOR.CYAN, use_color=use_color)
        footer = PrintLn.set_color(">", COLOR.CYAN, use_color=use_color)
        
        # 拿到所有的属性
        attrs = []
        attr_pattern: str = r"(?!_).*"
        import re
        for attr in dir(obj):
            if re.fullmatch(attr_pattern, attr):
                try:
                    attr_val = getattr(obj, attr)
                except AttributeError:
                    continue
                if inspect.ismethod(attr_val) or inspect.isbuiltin(attr_val):
                    continue
                else:
                    attrs.append(attr)
        
        # 将属性组装起来
        elems = ""
        for key in sorted(attrs):
            val = getattr(obj, key)
            indent = len(key) + 5
            elems += f" {PrintLn.set_color('.' + key, COLOR.GREEN, use_color=use_color)} = {cls._argumentToString(val,compact=True, indent=indent, use_color=use_color)[0]}\n"
        
        return f"{header}\n{elems}{footer}"
    
    @staticmethod
    def add_object_print(othercls):
        othercls.__str__ = lambda self: PrintLn._get_custom_object_str(self)
        return othercls

    def _isLiteral(self, s):
        if self._isFormatStr(s): return True
        try:
            import ast as _ast
            _ast.literal_eval(s)
        except Exception:
            return False
        return True

    def _isFormatStr(self, s):
        import re
        # 检查是否是 f-string
        pattern = r'^(f|rf|fr|Fr|fR|FR)([\'"])(.*?)\2'
        match = re.fullmatch(pattern, s)
        if bool(re.search(pattern, s)): return True
        
        # 检查是否是 % 格式化
        pattern = r'^([\'"])(.*?)%[sdxf](.*?)\1\s*%'
        if bool(re.search(pattern, s)): return True
        
        # 检查是否是 .format() 格式化
        pattern = r'^([\'"])(.+?)\1.format\((.*?)\)\s*$'
        match = re.fullmatch(pattern, s)
        if bool(match): return True
        
        return False

    def _format_pair(self, arg, value, prefixTooLong=False):
        arg_lines = self._indented_lines("", arg)
        value_prefix = arg_lines[-1] + ": "
        
        # str如果分行，那么整体退一格
        looksLikeAString = value[0] + value[-1] in ["''", '""']
        if looksLikeAString:  # Align the start of multiline strings.
            lines = value.splitlines(True)
            for i in range(1, len(lines)):
                lines[i] = " " + lines[i]
            value = "".join(lines)

        value_lines = self._indented_lines(value_prefix, value, prefixTooLong=prefixTooLong)
        lines = arg_lines[:-1] + value_lines
        return "\n".join(lines)

    def _indented_lines(self, prefix, string, prefixTooLong=False):
        lines = string.splitlines()
        prefixlen = len(prefix)
        
        if bool(prefix.strip()):
            prefix = PrintLn.set_color(prefix, COLOR.RED, self.use_color)
            
        if prefixTooLong:
            lth = 3
            return [prefix + '\n' + ' '*lth + lines[0]] + [" " * lth + line for line in lines[1:]]
        else:
            return [prefix + lines[0]] + [" " * prefixlen + line for line in lines[1:]]

println = PrintLn()  # 实例化 PrintLn 类
builtins.show = println  # 给内置函数 println 赋值

# =================
#    hdf5 工具
# =================

# -> save

def save_hdf5(filename:str, group:str, data: dict[str, Any], mode: str = "a") -> tuple[str, str]:
    """
    将数据保存到 HDF5 文件中。
    
    mode = 'a' 是指 append group，但
    **便是 append 模式下，相同 group 也会被覆盖，所以要注意。**
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    group : str
        HDF5 文件中的组路径，例如 "mygroup1/mygroup11"。
    data : Dict[str, Any]
        要保存的数据字典，其中键为数据集名称，值为数据。
    mode : str, optional
        文件打开模式。默认为 "a"（追加模式），可选为 "w"（覆盖模式）。
    
    Returns
    -------
    tuple[str, str]
        文件名和组路径。
    
    Examples
    --------
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "", {"mat": mat})
    """
    assert filename.endswith(".h5"), "Filename must to be `.h5` file."
    group = "/" + group.strip("/")  # make group to be "/xxx/xxx/..."
    logger.debug(f"Saving {list(data.keys())} to " + _os.path.abspath(filename) + " ... ")
    with _h5py.File(filename.encode("utf-8"), mode) as f:  # `f` is a type `h5py.File`
        # * Save data into group (group can be root "/" or not "/xxxx/xxx/...")
        g = f.require_group(group)
        _save_main(g, data) # g: _h5py.Group,  data: Dict[str,Any]
    logger.debug("Save done")
    return filename, group


def _save_main(h5group:_h5py.Group, data_dic: Dict[str, Any]) -> None:
    """"递归地将数据保存到 HDF5 组中。
    
    Parameters
    ----------
    h5group : _h5py.Group
        要保存数据的 HDF5 组。
    data_dic : Dict[str, Any]
        数据字典，其中键为数据集名称，值为数据。
    overwrite_dataset : bool
        是否覆盖已存在的数据集。
    """
    for key, value in data_dic.items():
        keystr = str(key)
        if isinstance(value, dict):  # dic[key] is a dict, save recursively
            subgroup = h5group.require_group(keystr)
            _save_main(subgroup, value)
        else:  # dic[key] is just a dataset
            if keystr in h5group:
                del h5group[keystr]
            value_type_name = type(value).__name__  # 拿到 value 的类型，如 "csr_array", "ndarray" 等
            save_func = _SAVE_FUNC.get(value_type_name, _default_save)  
            # _SAVE_DATA 是一个字典, save_func 是一个函数，根据类型 value_type_name 调用相应的保存函数
            save_func(h5group, keystr, value)

def _default_save(h5group:_h5py.Group, key:str, value) -> None:
    # 前置判断，是否是 dataclass, 或者是可以直接保存的类型
    value_dict = None
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        # namedtuple
        value_dict = value._asdict()
    elif is_dataclass(value):
        # dataclass
        value_dict = asdict(value)
    
    # 如果是可以直接保存的类型，如 dataclass，用 json 序列化
    if value_dict is not None:
        try:
            data = _json.dumps(value_dict, indent=4, ensure_ascii=False)
            dataset = h5group.create_dataset(key, data=data)
            dataset.attrs["object_type"] = "dataclass"
            dataset.attrs["dataset_name"] = type(value).__name__
        except TypeError as e:
            logger.debug(f"序列化 {key} 时发生错误: {e}\n将保存为字符串类型。")
            import pprint
            res = pprint.pformat(value_dict, indent=2, width=1)
            dataset = h5group.create_dataset(key, data=res)
    else:
        try:
            # 尝试直接保存
            h5group.create_dataset(key, data=value)
        except (ValueError, TypeError):
            # 如果失败，尝试序列化，但会失去可视化的能力
            import pickle as _pickle
                        
            # 将 value 序列化为字节流
            serialized_params = _pickle.dumps(value)
            dataset = h5group.create_dataset(key, data=_np.void(serialized_params))
            dataset.attrs["object_type"] = "serialized_bytes"
            

def _save_csr(h5group:_h5py.Group, key:str, csrdata) -> None:
    subgroup = h5group.create_group(key)
    subgroup.attrs["object_type"] = "csr"
    subgroup.attrs['shape'] = csrdata.shape
    subgroup.create_dataset('data', data=csrdata.data)
    subgroup.create_dataset('indices', data=csrdata.indices)
    subgroup.create_dataset('indptr', data=csrdata.indptr)

from typing import TYPE_CHECKING
if TYPE_CHECKING:  # 类型检查时，导入 torch
    import torch as _tc

def _save_torch(h5group:_h5py.Group, key:str, value:'_tc.Tensor') -> None:
    if value.grad is None:
        h5group.create_dataset(key, data=value.detach().cpu().numpy())
    else:
        subgroup = h5group.create_group(key)
        subgroup.attrs["dtype"] = f"{value.dtype}"
        subgroup.attrs["device"] = f"{value.device}"
        subgroup.attrs["requires_grad"] = value.requires_grad
        subgroup.create_dataset("data", data=value.detach().cpu().numpy())
        subgroup.create_dataset("gradient", data=value.grad.detach().cpu().numpy())

_SAVE_FUNC: Dict[str, Callable[[_h5py.Group, str, Any], None]] = {
    "csr_array": _save_csr,
    "csr_matrix": _save_csr,
    "Tensor": _save_torch,
}
# 也可以在外部文件中自定义其他方法，如：_SAVE_FUC.update({'coo_array':_save_coo})

# -> load

def load_hdf5(filename:str, group:str, dataname:str) -> Any:
    """从 HDF5 文件中加载数据。
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    group : str
        HDF5 文件中的组路径，例如 "/mygroup"。
    dataname : str
        要加载的数据名称。
        
    Returns
    -------
    Any
        加载的数据。

    Examples
    --------
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5, load_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "/", {"mat": mat})
    >>> mat = load_hdf5("data.h5", "/", "mat")
    """
    if not _os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."
        group_location = f[group]  # 获取组对象
        if isinstance(group_location, _h5py.Group):
            data_location = group_location[dataname]
        else:
            raise ValueError(f"Group {group} not found in {filename}.")
        data_type_str = data_location.attrs.get("object_type", None)
        if data_type_str is None and isinstance(data_location, _h5py.Group):
            data_type_str = 'dict'
        load_func = _LOAD_FUNC.get(data_type_str, _default_load)
        data = load_func(data_location)
    return data


def _default_load(data_location: _h5py.Group) -> Any:
    return data_location[()]

def _load_dict(h5group: _h5py.Group) -> Dict[str, Any]:
    dic = {}
    for key in h5group.keys():
        subgroup = h5group[key]
        if isinstance(subgroup, _h5py.Group):
            newdic = _load_dict(subgroup)  # 如果是字典，那么递归的下载
            dic[key] = newdic
        else:
            data_type_str = subgroup.attrs.get("object_type", None)
            load_func = _LOAD_FUNC.get(data_type_str, _default_load)
            dic[key] = load_func(subgroup)  # 否则用下载数据
    return dic


def _load_csr(data_location: _h5py.Group) -> _sp.sparse.csr_array:
    indptr: np.ndarray = data_location["indptr"][()] # type: ignore
    indices: np.ndarray = data_location["indices"][()] # type: ignore
    data: np.ndarray = data_location["data"][()] # type: ignore
    shape: tuple = data_location.attrs["shape"] # type: ignore
    return _sp.sparse.csr_array((data, indices, indptr), shape=shape, dtype=data.dtype)


def _load_dataclass(data_location: _h5py.Group) -> Any:
    data_str = data_location[()]
    data_name = data_location.attrs["dataset_name"]
    # print(data_str, data_name)
    # assert isinstance(data_str, str) and isinstance(data_name, str)
    data_dict = _json.loads(data_str)
    from collections import namedtuple
    Parameters = namedtuple(data_name, data_dict.keys())
    return Parameters(**data_dict)


def _load_serialized_bytes(data_location: _h5py.Group) -> Any:
    serialized_bytes = data_location[()]
    import pickle
    return pickle.loads(serialized_bytes) # type: ignore


_LOAD_FUNC: Dict[Union[str,None], Callable]  = {
    "dict": _load_dict,
    "csr": _load_csr,
    "dataclass": _load_dataclass,
    "serialized_bytes": _load_serialized_bytes
}


def view_hdf5(filename:str, group:str, depth=1):
    """显示 HDF5 文件中的目录结构.
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    group : str
        HDF5 文件中的组路径，例如 "/mygroup"。
    depth : int
        控制显示的层次深度，默认为 1。
    
    Returns
    -------
    None: 该函数无返回值，直接在控制台输出目录结构。

    Examples
    --------
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5, load_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "/", {"mat": mat})
    >>> view_hdf5("data.h5", "/")
    """
    def _print_attrs(name, obj):
        shift = name.count("/") * "    "
        namelist = name.split("/")
        item_name = namelist[-1]
        print(shift + item_name)
        if len(namelist) == depth:
            try:
                dic = dict(obj.attrs)
                dic["dtype"], dic["shape"] = obj.dtype, obj.shape
                for key, val in dic.items():
                    print(shift + "    " + f"{key}: {val}")
            except:
                pass
    
    with _h5py.File(filename.encode("utf-8"), "r") as f:
        gp = f[group]
        if isinstance(gp, _h5py.Group):
            gp.visititems(_print_attrs)
        else:
            raise ValueError(f"Group {group} not found in {filename}.")

# 下面两个是更高级的 save, load 用法
# 功能实现起来比较复杂，图方便的时候可以用

def _save_hdf5(filename:str, *data, group:Union[list[str],str, None] = None, mode:str='a') -> None:
    """将数据保存为 .h5 文件
    
    Parameters
    ----------
    filename : str
        保存的文件名，必须以.h5 结尾。
    *data : Any
        要保存的数据，可以是多个，也可以是字典。
    group : Union[list[str],str, None], optional
        保存到 HDF5 文件中的组路径，可以是字符串，也可以是列表。如果为 None，则保存到根目录。
    mode : str, optional
        文件打开模式，默认为 "a"（追加模式）。
    
    Returns
    -------
    None: 该函数无返回值。

    Examples
    --------
    >>> import numpy as np
    >>> import quante.basicfun as bf
    >>> mat = np.random.randn(10,10)
    >>> vec = np.random.randn(10)
    >>> bf.save_h5("data.h5", mat, vec)
    """
    assert filename[-3:] == ".h5", "use h5 for consistance"
    if len(data) == 1 and isinstance(data[0], dict):
        data_dic = data[0]
    else:
        current_frame = _inspect.currentframe()
        assert current_frame is not None and current_frame.f_back is not None, "Can't get the caller's frame"
        paraname = PrintLn._get_paraname(current_frame.f_back)
        if paraname is None:
            raise ValueError("Can't get the caller's parameter name")
        data_dic = dict()
        for i, arg in enumerate(data):
            if type(arg).__name__ == "type":
                data_dic[paraname[i+1]] = arg()
            else:
                data_dic[paraname[i+1]] = arg
    if group is None:
        group = []
    elif isinstance(group, str):
        group = [group]
    assert isinstance(group, list) and "/" not in group
    group_name = "/".join(group)
    save_hdf5(filename, group_name, data_dic, mode=mode)


def _load_hdf5(filename:str, *datanames, group=None) -> Union[Dict[str, Any], list[Any]]:
    """从 .h5 文件中加载数据.
    
    Parameters
    ----------
    filename : str
        保存的文件名，必须以.h5 结尾。
    *datanames : str
        要加载的数据名称，可以是多个。
    group : str, optional
        保存到 HDF5 文件中的组路径，可以是字符串。如果为 None，则从根目录开始查找。
    
    Returns
    -------
    Union[Dict[str, Any], list[Any]]
        加载的数据。

    Examples
    --------
    >>> import numpy as np
    >>> import quante.basicfun as bf
    >>> mat = np.random.randn(10,10)
    >>> bf.save_h5("data.h5", mat)
    >>> mat, = bf.saveh5("data.h5", "mat")
    """
    logger.debug("Loading from " + _os.path.abspath(filename) + " ... ")
    if group is None:
        group = []
    elif isinstance(group, str):
        group = [group]
    assert isinstance(group, list) and "/" not in group
    group = "/".join(group)
    
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."
        group_location = f[group]  # 获取组对象
        if not isinstance(group_location, _h5py.Group):
            raise ValueError(f"Group {group} not found in {filename}.")
        if len(datanames) == 0:
            data: Union[Dict[str, Any], list[Any]] = _load_dict(group_location)
        else:
            data = []
            for dataname in datanames:
                data_location = group_location[dataname]
                data_type_str = data_location.attrs.get("object_type", None)
                if data_type_str is None and isinstance(data_location, _h5py.Group):
                    data_type_str = 'dict'
                load_func = _LOAD_FUNC.get(data_type_str, _default_load)
                data.append(load_func(data_location))
            if len(datanames) == 1:
                data = data[0]
    logger.debug("Load done")
    return data

# =======
# 画图预设
# =======

def plt_style_use(stylename:str = "quante", svg: bool = True) -> None:
    """设置 pyplot 风格样式。
    
    Parameters
    ----------
    stylename: str, optional
        风格样式,常用的有 "quante", "default" 和 "science". 默认为 "quante"
    svg: bool, optional
        是否使用 SVG 格式. 默认为 True.
    
    References
    ----------
    https://matplotlib.org/stable/tutorials/introductory/customizing.html
    https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html#sphx-glr-gallery-style-sheets-style-sheets-reference-py
    """
    import matplotlib.pyplot as _plt
    
    try:
        if svg:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats("svg")
        else:
            from IPython.display import set_matplotlib_formats
            set_matplotlib_formats("png")
    except:
        pass
    
    if stylename == "quante":
        try:
            import matplotlib as _mpl
            _mpl.font_manager.findfont("Times New Roman", fallback_to_default=False) # type: ignore
            font = 'Times New Roman'
        except:
            font = 'sans-serif'
        defaultconfig = {
            "pdf.fonttype": 42,
            "figure.dpi": 100,
            "font.size": 12,
            "axes.labelsize": 14,
            "mathtext.fontset": "stix",
            "font.family": font,  # 'sans-serif', "Times New Roman"
            # 'dejavusans','dejavuserif', 'cm', 'stix','stixsans' or 'custom'
            "font.serif": ["SimSun"],
            # "figure.autolayout": True,
            "xtick.direction": "in",  # x tick 方向
            "ytick.direction": "in",  # y tick 方向
            # grid
            "axes.grid": "False",
            "grid.alpha": 0.4,  # 透明度
            "grid.linewidth": 1.0,  # 粗细
            # "svg.image_inline": True
            "legend.frameon":       False,
            "legend.fontsize":      13,
            "savefig.bbox" : "tight",
            "text.usetex" : False,
        }
        _plt.style.use("default")
        _plt.rcParams.update(defaultconfig)
    elif stylename == "science":
        # from https://github.com/garrettj403/SciencePlots/blob/master/scienceplots/styles/science.mplstyle
        scienceconfig = {
            # Set default figure size
            "figure.figsize" : (3.5, 2.625),
            
            # Set x axis
            "xtick.direction": "in",
            "xtick.major.size" : 3,
            "xtick.major.width" : 0.5,
            "xtick.minor.size" : 1.5,
            "xtick.minor.width" : 0.5,
            "xtick.minor.visible" : True,
            "xtick.top" : True,

            # Set y axis
            "ytick.direction" : "in",
            "ytick.major.size" : 3,
            "ytick.major.width" : 0.5,
            "ytick.minor.size" : 1.5,
            "ytick.minor.width" : 0.5,
            "ytick.minor.visible" : True,
            "ytick.right" : True,
            
            # Set line widths
            "axes.linewidth" : 0.5,
            "grid.linewidth" : 0.5,
            "lines.linewidth" : 1.,

            # Remove legend frame
            "legend.frameon" : False,

            # Always save as 'tight'
            "savefig.bbox" : "tight",
            "savefig.pad_inches" : 0.05,

            # Use serif fonts
            # font.serif : Times
            "font.family" : "serif",
            "mathtext.fontset" : "dejavuserif",

            # Use LaTeX for math formatting
            "text.usetex" : True,
            "text.latex.preamble" : "\\usepackage{amsmath} \\usepackage{amssymb}"
        }
        _plt.style.use("default")
        _plt.rcParams.update(scienceconfig)
    else:
        _plt.style.use(stylename)


def send_email(subject: str, body: str, to_email: str, from_email: str, smtp_server: str, smtp_port: int, login: str, password: str):
    r"""发送程序执行完成的邮件
    
    Parameters
    ----------
    subject : str
        邮件主题
    body : str
        邮件正文
    to_email : str
        收件人邮箱地址
    from_email : str
        发件人邮箱地址
    smtp_server : str
        SMTP 服务器地址（如'smtp.gmail.com'）
    smtp_port : int
        SMTP 端口号（通常 587 或 465）
    login : str
        SMTP 登录用户名（通常是发件人邮箱）
    password : str
        SMTP 登录密码或授权码
        
    Returns
    -------
    None

    Example
    -------
    >>> # 配置 SMTP 信息和邮件内容
    >>> smtp_server = "smtp.xxx.edu.cn"  # 例如 Gmail 的 SMTP 服务器
    >>> smtp_port = 25  # Gmail 使用的 TLS 端口
    >>> from_email = "xxxxxxxxxx@xxxx.edu.cn"
    >>> to_email = "xxxxxxxxxxx@outlook.com"
    >>> login = "xxxxxxxxxxx@xxx.edu.cn"
    >>> password = "xxxxxxxxxxxx"  # 应用专用密码或邮箱密码
    >>> 
    >>> # 自定义邮件内容
    >>> subject = "程序执行完成通知"
    >>> body = "您的程序已经成功运行完成！"
    >>> 
    >>> try:
    >>>     pass  # 程序运行代码
    >>>     body += "\n\n程序运行成功！"
    >>> except Exception as e:
    >>>     body += f"\n\n出现错误：\n{e}"
    >>> finally:
    >>>     send_email(subject, body, to_email, from_email, smtp_server, smtp_port, login, password)
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import traceback

    try:
        # 创建邮件
        message = MIMEMultipart()
        message["From"] = from_email
        message["To"] = to_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))  # 可以将 "plain" 替换为 "html" 发送 HTML 格式邮件

        # 连接 SMTP 服务器并发送邮件
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # 启用 TLS 加密
            server.login(login, password)
            server.sendmail(from_email, to_email, message.as_string())
            print(f"邮件已发送至 {to_email}")
    except Exception as e:
        print("发送邮件时发生错误：")
        traceback.print_exc()
