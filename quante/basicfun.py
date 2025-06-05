# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-05-02 14:52:59
# @Last Modified by:   hzhu
# @Last Modified time: 2025-05-23 11:11:11

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
import difflib
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
 

def check_file_exists(filename):
    """检查文件是否存在"""
    if not _os.path.exists(filename):
        directory = _os.path.dirname(filename)
        if not directory:
            directory = _os.getcwd()
        basename, ext = _os.path.splitext(_os.path.basename(filename))
        all_files_folders = _os.listdir(directory)
        all_files = [f for f in all_files_folders if _os.path.isfile(_os.path.join(directory, f))]
        similar_files = difflib.get_close_matches(basename, [f.split('.')[0] for f in all_files])
        import textwrap
        if similar_files:
            similar_files_with_ext = [f for f in all_files if f.split('.')[0] in similar_files]
            wrapped_filename = textwrap.fill("   ".join(similar_files_with_ext), width=80)
            raise FileNotFoundError(
                f"Did you mean: \n"
                f"{wrapped_filename}\n"
                f"desired file: {filename}"
                )
        else:
            wrapped_filename = textwrap.fill("   ".join(all_files), width=80)
            all_folder = [f for f in all_files_folders if _os.path.isdir(_os.path.join(directory, f))]
            if all_folder:
                wrapped_foldername = textwrap.fill("   ".join(all_folder), width=80)
                raise FileNotFoundError(
                    f"\navailable files here: \n"
                    f"    {wrapped_filename} \n"
                    f"folders here: \n"
                    f"    {wrapped_foldername}\n"
                    f"desired file: {filename}"
                    )
            else:
                raise FileNotFoundError(
                    f"\navailable files here: \n"
                    f"    {wrapped_filename}\n"
                    f"desired file: {filename}"
                    )
    
    
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

def set_color(s: str, color: str) -> str:
    return f"{color}{s}{COLOR.DEFAULT}"

def objstr(obj, color=True, compact=False, indent=0):
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
        if (obj.__class__.__str__ is not object.__str__ or obj.__class__.__repr__ is not object.__repr__):
            s = pprint.pformat(obj)
        else:
            isobject = True
            s = _get_custom_object_str(obj, color)
    s = s.replace("\\n", "\n")  # Preserve string newlines in output.
    s = '\n'.join(v if i == 0 else " "*indent + v for i, v in enumerate(s.split('\n')))
    return s, isobject

def _get_custom_object_str(obj, color=True, ):
    import inspect
    # 首先拿到 header, footer
    obj_type = type(obj)
    header = f"<{obj_type.__name__} {hex(id(obj))}"
    footer = ">"
    if color:
        header = set_color(header, COLOR.CYAN)
        footer = set_color(footer, COLOR.CYAN)
    
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
        coloredkey = '.' + key
        if color:
            coloredkey = set_color(coloredkey, COLOR.GREEN)
        elems += f" {coloredkey} = {objstr(val,compact=True, indent=indent, color=color)[0]}\n"
    
    return f"{header}\n{elems}{footer}"

def _isLiteral(s):
    if _isFormatStr(s): return True
    try:
        import ast as _ast
        _ast.literal_eval(s)
    except Exception:
        return False
    return True

def _isFormatStr(s):
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

def unparse_tgs(node):
    res = []
    for elt in node.elts:
        if isinstance(elt, _ast.Tuple):
            res.append(unparse_tgs(elt))
        elif isinstance(elt, _ast.Name):
            res.append(elt.id)
        else:
            res.append(_ast.unparse(elt))
    return tuple(res)

def get_last_lv(call_frame) -> tuple | str | None:
    from objprint.executing import Source
    # check if call_frame is None
    if call_frame is None:
        return None

    # get node
    node = Source.executing(call_frame).node
    if node is None:
        return None
    
    # initialize stack of statements
    last_stmt = None
    lineno = _inspect.getlineno(call_frame)
    statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
    if len(statement_node) > 1:
        last_stmt = sorted(statement_node, key=lambda x: x.col_offset)[-2]
    
    if last_stmt is None:
        # if no statement found, go back to the previous line
        for _ in range(10):
            lineno -= 1
            statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
            if len(statement_node) > 0:
                last_stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1]
                break

    if last_stmt is None:
        # return None if no statement found
        return None
    
    # find left values
    left_values = None
    if isinstance(last_stmt, _ast.Assign):
        tgs = last_stmt.targets[-1]
        if isinstance(tgs, _ast.Name):
            left_values = tgs.id
        elif isinstance(tgs, _ast.Tuple):
            left_values = unparse_tgs(tgs)
        else:
            left_values = _ast.unparse(tgs)

    return left_values

def get_lv(call_frame) -> tuple | str | None:
    from objprint.executing import Source
    # check if call_frame is None
    if call_frame is None:
        return None

    # get node
    node = Source.executing(call_frame).node
    if node is None:
        return None
    
    # initialize stack of statements
    lineno = _inspect.getlineno(call_frame)
    statement_node = Source.for_frame(call_frame).statements_at_line(lineno)
    stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1] # todo: 这里可能会有问题，可能不是最后一个
    
    # find left values
    left_values = None
    if isinstance(stmt, _ast.Assign):
        tgs = stmt.targets[-1]
        if isinstance(tgs, _ast.Name):
            left_values = tgs.id
        elif isinstance(tgs, _ast.Tuple):
            left_values = unparse_tgs(tgs)
        else:
            left_values = _ast.unparse(tgs)

    return left_values

def get_vals(last_lv, call_frame) -> list:
    """获取变量的值"""
    local_vars = call_frame.f_locals
    global_vars = call_frame.f_globals
    vals = [eval(lv, global_vars, local_vars) for lv in last_lv]
    return vals

def flatten_tuple(t):
    if isinstance(t, str):
        return (t, )
    result = []
    for item in t:
        result.extend(flatten_tuple(item))  # 递归处理嵌套元组
    return tuple(result)

def get_args(frame):
    import tokenize
    import io
    if frame is None:
        return None
    func_call_str = get_executing_function_call_str(frame)
    if func_call_str is None:
        func_call_str = get_executing_function_call_str2(frame)
    if func_call_str is None:
        return None
    func_call_io = io.StringIO(func_call_str)
    depth = 0
    args = []
    curr_arg = ""
    last_pos = (0, 0)
    for token in tokenize.generate_tokens(func_call_io.readline):
        if depth == 0 and token.string == "(":
            depth = 1
        elif depth == 1 and token.string == ")":
            args.append(curr_arg.strip())
            break
        elif depth == 1 and token.string == ",":
            args.append(curr_arg.strip())
            curr_arg = ""
        elif depth >= 1:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            if depth >= 1 and token.type != tokenize.NL:
                if token.start[0] != last_pos[0] or token.start[1] - last_pos[1] > 0:
                    curr_arg += f" {token.string}"
                else:
                    curr_arg += token.string
        last_pos = token.end
    return args

def get_executing_function_call_str(frame):
    from objprint.executing import Source
    node = Source.executing(frame).node
    if node is None:
        return None

    try:
        module = _inspect.getmodule(frame)
        if module is None:
            return None
        source = _inspect.getsource(module)
    except (OSError, TypeError):
        return None

    return _ast.get_source_segment(source, node)

def get_executing_function_call_str2(frame):
    from objprint.executing import Source
    node = Source.executing(frame).node
    if node is None:
        return None
    lineno = _inspect.getlineno(frame)
    statement_node = Source.for_frame(frame).statements_at_line(lineno)
    stmt = sorted(statement_node, key=lambda x: x.col_offset)[-1]
    return _ast.unparse(stmt)

class Show:
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
    >>> a, (b, c) = 1, (2, 3)
    >>> show()
    a: 1; b: 2; c: 3
    """
    def __init__(self, use_color=True):
        self.use_color = use_color
        self.arg_name = True
    
    def __call__(self, *ipt, level=1, end=None):
        if level == 0:
            return None
        
        if self.arg_name and end is None:
            call_frame = _inspect.currentframe()  # 获取调用函数的栈帧
            if call_frame is not None:
                call_frame = call_frame.f_back
            
            if len(ipt) == 0:
                last_lv = get_last_lv(call_frame)
                if last_lv is None:
                    return ""
                last_lv = flatten_tuple(last_lv)
                vals = get_vals(last_lv, call_frame)
                out: str = self._constructArgumentOutput(last_lv, vals)
            else:
                args = get_args(call_frame)
                if args is None:
                    args = ["Unknown Arg" for _ in range(len(ipt))]
                if len(args) != len(ipt):
                    out = "\n".join(map(str, ipt))
                else:
                    out: str = self._constructArgumentOutput(args, ipt)
        else:
            if end is None:
                end = "\n"
            out = end.join(map(str, ipt))

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
       
        logger.handlers[0].flush()  # 立即刷新日志

    def _constructArgumentOutput(self, paraname, inputargs):
        if len(paraname) == 1 and _isLiteral(paraname[0]):
            if isinstance(inputargs[0], str):
                return inputargs[0] # 这样不会显示引号
            return objstr(inputargs[0], color=self.use_color)[0]
        
        pairs = [(arg, *objstr(val, color=self.use_color)) 
                 for arg, val in zip(paraname, inputargs)]
        
        pairStrs = []
        for arg, val, isobject in pairs:
            if _isLiteral(arg):
                pairStrs.append(val)
            else:
                coloredarg = f"{arg}: "
                if self.use_color:
                    coloredarg = set_color(coloredarg, COLOR.RED)
                pairStrs.append(coloredarg + val)

        seperator = set_color("; ", COLOR.RED) if self.use_color else "; "
        allArgsOnOneLine = seperator.join(pairStrs)
        
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
        
        if bool(prefix.strip()) and self.use_color:
            prefix = set_color(prefix, COLOR.RED)
            
        if prefixTooLong:
            lth = 3
            return [prefix + '\n' + ' '*lth + lines[0]] + [" " * lth + line for line in lines[1:]]
        else:
            return [prefix + lines[0]] + [" " * prefixlen + line for line in lines[1:]]

builtins.show = show = println = Show()  # 实例化 PrintLn 类

def set_show(use_color=None, arg_name=None) -> None:
    if use_color is not None:
        println.use_color = use_color
    if arg_name is not None:
        println.arg_name = arg_name


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
            if isinstance(value, list):
                h5group[key].attrs["object_type"] = "pylist"
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
        h5group[key].attrs["object_type"] = "Tensor"
        h5group[key].attrs["device"] = f"{value.device}"
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

def load_hdf5(filename:str, group:str, dataname:str|list[str]) -> Any:
    """从 HDF5 文件中加载数据。
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    group : str
        HDF5 文件中的组路径，例如 "/mygroup"。
    dataname : str | list[str]
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
    check_file_exists(filename)
    group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."

    logger.debug("Loading from " + _os.path.abspath(filename) + " ... ")
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group_location = _get_data_location(f, group)
        data = _load_main(group_location, dataname)
    logger.debug("Load done")
    return data

def _load_main(group_location:_h5py.Group, lv: Union[str, list]) -> Any:
    """加载数据"""

    if isinstance(lv, str):
        data_location = _get_data_location(group_location, lv)
        data_type_str = data_location.attrs.get("object_type", None)
        if data_type_str is None and isinstance(data_location, _h5py.Group):
            data_type_str = 'dict'
        load_func = _LOAD_FUNC.get(data_type_str, _default_load)
        return load_func(data_location)
    
    res = []
    for dataname in lv:
        res.append(_load_main(group_location, dataname))
    return tuple(res)

def _get_data_location(f: _h5py.File | _h5py.Group, name: str) -> _h5py.Group:
    # 检查 f 中是否存在 name 的 group 或者 dataset
    try:
        return f[name]
    except:
        print(f, name)
        res = []
        names = name.split("/")
        eachname = name
        for eachname in names:
            try:
                f = f[eachname]
                res.append(eachname)
            except:
                break
        import textwrap
        available_names = list(f.keys())
        wrapped_names = textwrap.fill(str(available_names), width=80)
        tmp = "/".join(res)
        raise ValueError(
            f"Available names:\n{wrapped_names}.\n"
            f"Data '{eachname}' not found in '/{tmp}'."
        )

def _default_load(data_location: _h5py.Group) -> Any:
    return data_location[()]

def _load_pylist(data_location: _h5py.Group) -> list:
    data = _default_load(data_location)  # 先加载数据
    return data.tolist() if isinstance(data, _np.ndarray) else list(data)  # 如果是 ndarray，转为 list

def _load_tctensor(data_location: _h5py.Group) -> list:
    data = _default_load(data_location)  # 先加载数据
    import torch as _tc
    device = data_location.attrs["device"]
    return _tc.tensor(data, device=device)

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
    "pylist": _load_pylist,
    "dict": _load_dict,
    "csr": _load_csr,
    "Tensor": _load_tctensor,
    "dataclass": _load_dataclass,
    "serialized_bytes": _load_serialized_bytes
}


def view_hdf5(filename:str, group:str='/', depth=1):
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
    try:
        from anytree import Node, RenderTree
        useanytree = True
    except ImportError:
        useanytree = False
    
    if useanytree:
        with _h5py.File(filename.encode("utf-8"), "r") as f:
            # 定义一个递归函数来构建树状结构
            def build_tree(name, obj, parent=None, level=0):
                node = Node(name, parent=parent)
                if isinstance(obj, _h5py.Group) and level < depth:
                    for key in obj.keys():
                        build_tree(key, obj[key], parent=node, level=level+1)
                return node
            
            # 构建树状结构
            g = _get_data_location(f, group)
            root = build_tree(group, g)
    
        # 可视化树状结构
        for pre, fill, node in RenderTree(root):
            print(f"{pre}{node.name}")
        
    else:
        with _h5py.File(filename.encode("utf-8"), "r") as f:
            def print_tree(name, obj, level=0):
                indent = '  ' * level
                print(f"{indent}{name}")
                if isinstance(obj, _h5py.Group) and level < depth:
                    for key in obj.keys():
                        print_tree(key, obj[key], level + 1)
            
            # 使用递归函数遍历所有组和数据集，并只显示一个层级
            g = _get_data_location(f, group)
            print_tree(group, g)


def isave(filename:str, *dataargs, data:dict = None, group:Union[str, None] = '/', mode:str='a') -> None:
    """将数据保存为 .h5 文件
    
    Parameters
    ----------
    filename : str
        保存的文件名，必须以.h5 结尾。
    *dataargs : Any
        要保存的数据，将自动提取变量名。
    data : dict, optional
        要保存的数据集，字典形式。
    group : str, optional
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
    assert filename[-3:] == ".h5", "use .h5 file"

    if data is not None:
        assert len(dataargs) == 0, "data and datadic cannot be used at the same time."
        save_hdf5(filename, group, data, mode=mode)
        return None

    call_frame = _inspect.currentframe()
    if call_frame is not None:
        call_frame = call_frame.f_back
    
    if len(dataargs) == 0:
        args = get_last_lv(call_frame)
        args = flatten_tuple(args)
        vals = get_vals(args, call_frame)
    else:
        args = get_args(call_frame)
        if args is None:
            args = [f"Unknown Arg {i}" for i in range(len(dataargs)+1)]
        args = args[1:]
        vals = dataargs
    
    data_dic = dict()
    for i, eachdata in enumerate(vals):
        if type(eachdata).__name__ == "type":
            data_dic[args[i]] = eachdata()
        else:
            data_dic[args[i]] = eachdata
        
    save_hdf5(filename, group, data_dic, mode=mode)


def iload(filename:str, dataname:list[str]|str|None = None, *, group='/') -> Any:
    """从 .h5 文件中加载数据.
    
    Parameters
    ----------
    filename : str
        保存的文件名，必须以.h5 结尾。
    dataname : list[str] | str | None, optional
        要加载的数据名称，可以是多个。
    group : str, optional
        保存到 HDF5 文件中的组路径，可以是字符串。如果为 None，则从根目录开始查找。
    
    Returns
    -------
    Any:
        加载的数据。
        如果 datanames 为空，那么根据调用时的变量名来加载数据。
        如果返回为单个参数，则加载全部数据。

    Examples
    --------
    >>> import numpy as np
    >>> import quante as qt
    >>> mat = np.random.randn(10,10)
    >>> qt.basicfun.isave("data.h5")
    >>> mat, = qt.basicfun.iload("data.h5")
    """
    assert isinstance(group, str)
    
    # obtain left value names
    if dataname is None:
        call_frame = _inspect.currentframe()
        if call_frame is not None:
            call_frame = call_frame.f_back
        lv = get_lv(call_frame)
        if isinstance(lv, str) or lv is None:
            lv = group
    else:
        # dataset is list or tuple
        assert isinstance(dataname, (list, tuple, str)), "dataset must be list or tuple."
        lv = dataname

    return load_hdf5(filename, group, lv)

# =======
# 画图预设
# =======

def plt_style_use(stylename:str = "quante", svg: bool = True, svg_display_width=600) -> None:
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
            from IPython.display import set_matplotlib_formats, display
            set_matplotlib_formats("svg")
            display({
                'text/html': f'<style>svg {{width: {svg_display_width}px !important; height: auto;}}</style>'
            }, raw=True)
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
            # "figure.figsize" : (3.5, 2.625),
            "figure.figsize" : (4, 2.9),
            
            # Set x axis
            "xtick.direction": "in",
            # "xtick.major.size" : 3,
            # "xtick.major.width" : 0.5,
            # "xtick.minor.size" : 1.5,
            # "xtick.minor.width" : 0.5,
            # "xtick.minor.visible" : True,
            "xtick.top" : False,

            # Set y axis
            "ytick.direction" : "in",
            # "ytick.major.size" : 3,
            # "ytick.major.width" : 0.5,
            # "ytick.minor.size" : 1.5,
            # "ytick.minor.width" : 0.5,
            # "ytick.minor.visible" : True,
            "ytick.right" : False,
            
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


class DynamicPlot:
    def __init__(self, tlist, ax, *args, **kwargs):
        # save package
        import matplotlib.pyplot as plt
        self.pkg = plt

        # check ax
        if ax is None:
            fig, ax = plt.subplots()
        if not isinstance(ax, plt.Axes):
            raise TypeError("The 'ax' parameter must be a matplotlib Axes object or None.")
        self.ax = ax

        # check if in ipython
        in_ipython = False
        try:
            from IPython import get_ipython
            in_ipython = get_ipython() is not None
        except ImportError:
            in_ipython = False
        if in_ipython:
            from IPython.display import clear_output, display
            self.clear_output = clear_output
            self.display = display
        self.in_ipython = in_ipython
        
        # save data
        self.tlist = tlist
        self.data = None

        # save plot parameters
        self.ptype = None
        self.args = args
        self.kwargs = kwargs

        # initialize dp parameters
        self.i = 0
        self.xlim = None
        self.ylim = None
        self.clim = None
        self.legend = None
    
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return self.data.__repr__()
        
    def set(self, xlim=None, ylim=None, clim=None, legend=None, ptype=None):
        self.xlim = xlim
        self.ylim = ylim
        self.clim = clim
        self.legend = legend
        self.ptype = ptype
        return self

    def append(self, res_t):
        ax = self.ax
        plt = self.pkg
        i = self.i

        if self.data is None:
            res_t = self._init_plot(res_t)
            
        if self.ptype == "line":
            self.data[i] = res_t    
            self.plot.set_xdata(self.tlist[:i+1])
            self.plot.set_ydata(self.data[:i+1])
            if self.xlim is None:
                ax.set_xlim(min(self.tlist[:i+1]), max(self.tlist[:i+1]))
            if self.ylim is None:
                ax.set_ylim(min(self.data[:i+1]), max(self.data[:i+1]))
        elif self.ptype == "para":
            self.data[0, i] = res_t[0]
            self.data[1, i] = res_t[1]
            self.plot.set_xdata(self.data[0,:i+1])
            self.plot.set_ydata(self.data[1,:i+1])
            if self.xlim is None:
                ax.set_xlim(min(self.data[0,:i+1]), max(self.data[0,:i+1]))
            if self.ylim is None:
                ax.set_ylim(min(self.data[1,:i+1]), max(self.data[1,:i+1]))
        elif self.ptype == "dens":
            self.data[:, i] = _np.asarray(res_t)
            self.plot.set_data(self.data.T)
            if self.clim is None:
                valid = self.data[:, :i+1]
                vmin, vmax = _np.nanmin(valid), _np.nanmax(valid)
                if vmin != vmax:
                    self.plot.set_clim(vmin, vmax)           
        
        if self.in_ipython:
            self.clear_output(wait=True)
            self.display(plt.gcf())
        else:
            plt.pause(0.1)
        self.i += 1
        
        if i == len(self.tlist) - 1:
            if self.in_ipython:
                self.clear_output(wait=True)
            else:
                self.pkg.show()
        return self.data


    def _init_plot(self, res_t):
        ax = self.ax
        plt = self.pkg

        res_t = _np.asarray(res_t)

        if self.ptype is None:
            # determine plot type according to the data type
            if res_t.size == 1:
                self.ptype = "line"
            elif res_t.size == 2:
                self.ptype = "para"
            else:
                self.ptype = "dens"
        
        if self.ptype == "line":
            if self.xlim is None:
                self.xlim = (self.tlist[0], self.tlist[-1])
            self.data = _np.full(len(self.tlist), _np.nan, dtype=_np.float64)
            self.plot, = ax.plot(self.tlist, self.data, *self.args, **self.kwargs)
            if self.legend:
                ax.legend()
        elif self.ptype == "para":
            self.data = _np.full((2, len(self.tlist)), _np.nan, dtype=_np.float64)
            self.plot, = ax.plot(self.data[0,:], self.data[1,:], *self.args, **self.kwargs)
            if self.legend:
                ax.legend()
        elif self.ptype == "dens":
            n = len(res_t)
            self.data = _np.full((n, len(self.tlist)), _np.nan, dtype=_np.float64)
            self.plot = ax.imshow(self.data.T, *self.args, aspect='auto', origin='lower', **self.kwargs, extent=(0, n, self.tlist[0], self.tlist[-1]))
            if self.legend:
                plt.colorbar(self.plot, ax=ax)
            if self.clim is not None:
                self.plot.set_clim(*self.clim)           
        else:
            raise ValueError("Unknown plot type.")
        
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
            
        return res_t