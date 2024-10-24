# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-05-02 14:52:59
# @Last Modified by:   hzhu
# @Last Modified time: 2024-10-18 15:36:14
"""
功能：

- 测试工具：`test_time`, `test_memory`
- 系统层面的函数：`create_folder`, `save_hdf5`, `load_hdf5`
- 日志工具：`set_logging`, `println`
- 字典格式：`todict`, `idataclass`
- 画图预设：`set_matplotlib`
"""

#!! 这个文件不应该 import quante 中的任何其他文件！

import gc as _gc
import os as _os
import ast as _ast
import sys as _sys
import numpy as _np
import scipy as _sp
import time as _time
import h5py as _h5py
import ctypes as _ctypes
import inspect as _inspect
import logging as _logging
import platform as _platform

from itertools import chain
from collections import deque
from types import FunctionType
from typing import Callable, Any, Dict, Union, Optional


__all__ = ["test_time", "test_memory"]  # 测试工具

__all__ += ["clear", "get_free_space", "create_folder"]  # 系统层面的函数

__all__ += ["save_hdf5", "load_hdf5", "view_hdf5"]  # hdf5 工具

__all__ += ["set_logging", "println"]  # 日志工具

__all__ += ["idataclass", "NamedData"]  # 字典格式

__all__ += ["set_matplotlib"]  # 画图预设


import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'  # 为了让 ctrl+c 中断程序可用
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 为了避免 scipy svd 与 gpu torch 冲突

_np.set_printoptions(linewidth=1000000, suppress=True) # 为了让 print 出的矩阵的宽度没有限制

# =================
#     测试工具
# =================
def test_time(func: Callable, *args, inner_func_list:list[Callable] = [], timer_unit: float =1e-7, save: bool=True, **kwargs) -> None:
    """
    测试函数每一行代码的执行时间。
    
    另一个更高效的方法是，在文件最开始添加 os.environ["LINE_PROFILE"] = "1"
    
    然后在要测试的函数加上装饰器
    >>> from line_profiler import profile
    即可得到每一行代码的执行时间
    
    Example:
    >>> import numpy as np
    >>> from quante.basicfun import test_time
    >>> def func2(dim):
    >>>     return np.random.randn(dim,dim)
    >>> def my_function(dim):
    >>>     for _ in range(10):
    >>>         mat = func2(dim)
    >>>         val, vec = np.linalg.eig(mat)
    >>>     return val
    >>> test_time(my_function, 100, inner_func_list=[func2], timer_unit=0.01)
    
    参数:
    - func: 需要分析的目标函数。
    - args: 传递给目标函数的参数。
    - inner_func_list: 要分析的内部函数列表，默认为空。
    - timer_unit: 时间单位，默认为 1e-7 秒（100 纳秒）。

    返回:
    - None: 该函数不返回值，仅打印每行代码的执行时间。
    """
    from line_profiler import LineProfiler
    lp = LineProfiler()  # 初始化行分析器
    for i in inner_func_list:
        lp.add_function(i)  # 添加要分析的函数
    lp_wrapper = lp(func) # 包装函数
    lp_wrapper(*args, **kwargs)  # 传递参数
    if save:
        filename = "profile_" + _os.path.basename(_sys.argv[0])[:-3] + '.txt'  # 根据运行的脚本文件名生成日志文件名 
        with open(filename, 'a') as f:
            original_stdout = _sys.stdout
            _sys.stdout = f
            lp.print_stats(output_unit=timer_unit)  # 打印执行时间统计信息
            _sys.stdout = original_stdout
    else:
        lp.print_stats(output_unit=timer_unit)
    

def test_memory(obj: Any):
    """ 
    打印对象的占用空间的大小
    
    Example:
    >>> import numpy as np
    >>> from quante.basicfun import test_memory
    >>> a = [np.random.randn(10,10) for i in range(100)]
    >>> test_memory(a)
    
    参数:
    - obj: 需要测量内存占用的对象。
    
    返回:
    - None: 该函数不返回值，仅打印对象的占用空间大小。
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
    """只有当非常在意内存的时候才需要使用这个函数
    因为强制清除内存会降低性能"""
    del name
    _gc.collect()


def get_free_space(folder: str) -> float:
    """
    获取磁盘剩余空间。

    示例:
    >>> get_free_space("D:\\")
    
    参数:
    - folder (str): 磁盘路径，例如 "D:\\"。

    返回:
    - float: 剩余空间，单位为 GB。
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
        # st = _os.statvfs(folder)
        # return st.f_bavail * st.f_frsize / (1024 ** 3)
    

def create_folder(path1:str, path2: Union[None, str]=None) -> str:
    """
    创建一个文件夹（包括路径中指定的所有父文件夹）。
    
    Example:
    >>> import quante as qt
    >>> path1 = "data/entanglement"
    >>> path2 = "XXZ"
    >>> test_memory(a)

    参数:
    - path1 (str): 主路径，可以是类似 "xxx/xxx/..." 的格式。
    - path2 (str, 可选): 子路径，必须为 None 或者类似 "xxx" 的字符串。

    返回:
    - str: 创建的完整路径，以斜杠结尾。
    """
    whole_path = _os.path.abspath(path1).replace("\\", "/") + "/"  # 规范化主路径
    whole_path = _os.path.join(whole_path, path2).replace("\\", "/") if path2 else whole_path  # 如果提供了子路径，则加入到主路径中
    _os.makedirs(whole_path, exist_ok=True)  # 创建文件夹（包括所有父文件夹）
    return whole_path if whole_path.endswith("/") else whole_path+"/"   # 确保返回路径以斜杠结尾
    
# =================
#    hdf5 工具
# =================

# -> save

def save_hdf5(filename:str, group:str, data: Dict[str, Any], mode: str = "a") -> tuple[str, str]:
    """
    将数据保存到 HDF5 文件中。
    
    mode = 'a' 是指 append group，但
    **便是 append 模式下，相同group 也会被覆盖，所以要注意。**
    
    Example:
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "", {"mat": mat})
    
    参数:
        filename (str): HDF5 文件的路径。
        group (str): HDF5 文件中的组路径，例如 "mygroup1/mygroup11"。
        data (Dict[str, Any]): 要保存的数据字典，其中键为数据集名称，值为数据。
        mode (str, optional): 文件打开模式。默认为 "a"（追加模式），可选为 "w"（覆盖模式）。
    
    返回:
        tuple[str, str]: 文件名和组路径。
    """
    assert filename.endswith(".h5"), "Filename must to be `.h5` file."
    group = "/" + group.strip("/")  # make group to be "/xxx/xxx/..."
    _logging.info(f"Saving {list(data.keys())} to " + _os.path.abspath(filename) + " ... ")
    with _h5py.File(filename.encode("utf-8"), mode) as f:  # `f` is a type `h5py.File`
        # * Save data into group (group can be root "/" or not "/xxxx/xxx/...")
        g = f.require_group(group)
        _save_main(g, data) # g: _h5py.Group,  data: Dict[str,Any]
    _logging.info("Save done")
    return filename, group


def _save_main(h5group:_h5py.Group, data_dic: Dict[str, Any]) -> None:
    """"
    递归地将数据保存到 HDF5 组中。

    参数:
        h5group (_h5py.Group): 要保存数据的 HDF5 组。
        data_dic (Dict[str, Any]): 数据字典，其中键为数据集名称，值为数据。
        overwrite_dataset (bool): 是否覆盖已存在的数据集。
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
    h5group.create_dataset(key, data=value)

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
    """
    从 HDF5 文件中加载数据。

    Example:
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5, load_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "/", {"mat": mat})
    >>> mat = load_hdf5("data.h5", "/", "mat")
    
    参数:
        filename (str): HDF5 文件的路径。
        group (str): HDF5 文件中的组路径，例如 "/mygroup"。
        dataname (str): 要加载的数据名称。

    返回:
        Any: 加载的数据。
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."
        group_location: _h5py.Group = f[group]  # 获取组对象
        data_location = group_location[dataname]
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
    indptr = data_location["indptr"][()]
    indices = data_location["indices"][()]
    data = data_location["data"][()]
    shape = data_location.attrs["shape"]
    return _sp.sparse.csr_array((data, indices, indptr), shape=shape, dtype=data.dtype)


_LOAD_FUNC: Dict[str, Callable[[_h5py.Group], Any]] = {
    "dict": _load_dict,
    "csr": _load_csr,
}


def view_hdf5(filename:str, group:str, depth=1):
    """
    显示 HDF5 文件中的目录结构。


    Example:
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5, load_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", "/", {"mat": mat})
    >>> view_hdf5("data.h5", "/")
    
    参数:
        filename (str): HDF5 文件的路径。
        group (str): 要查看的组路径，例如 "/mygroup"。
        depth (int): 控制显示的层次深度，默认为 1。

    返回:
        None: 该函数无返回值，直接在控制台输出目录结构。
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
        f[group].visititems(_print_attrs)


#############################################################
#  日志工具
#############################################################

def set_logging(level: int = _logging.WARNING, savelog: bool = False, filenameTime: bool = False, logtime: bool = False):
    """
    配置日志记录功能。
    
    实例：
    >>> set_logging(savelog=True, logtime=True)

    参数:
        level (int): 设置日志记录的级别，默认为 `_logging.WARNING`。
        savelog (bool): 是否将日志保存到文件中，默认为 `False`。
        filenameTime (bool): 是否在日志文件名中包含时间戳，默认为 `False`。
        logtime (bool): 是否在日志消息中包含时间戳，默认为 `False`。

    返回:
        None: 该函数无返回值。
    """
    if savelog:
        filename = "log/"
        create_folder("log/")  # 创建日志目录
        filename += _os.path.basename(_sys.argv[0])[:-3]  # 根据运行的脚本文件名生成日志文件名
        if filenameTime:  # 如果 `filenameTime` 为 `True`，在文件名中添加时间戳
            now = "_" + _time.strftime("%Y-%m-%d-%H_%M_%S", _time.localtime(_time.time()))
            filename += now
        filename += '.log'
    else:
        filename = ""
    if logtime:  # 根据 `logtime` 参数设置日志格式
        _format = "%(asctime)s: %(message)s"
    else:
        _format = "%(message)s"
    _logging.basicConfig(level=level, format=_format, filename=filename, filemode="w", force=True, encoding='utf-8') # 配置日志记录器

set_logging()  # 使用默认日志记录器


import traceback
def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # _logging.warning("发生错误！这是固定的提示信息。")
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    _logging.error(tb_str)  # 输出格式化的traceback字符串

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
    
    todo: 整理这部分代码，实现多行输出，目前只能单行

    示例:
    >>> a = "this is a test"
    >>> println(a)
    """
    
    def __call__(self, *inputargs):
        try:
            paraname: list[str] = PrintLn._get_paraname(_inspect.currentframe().f_back)
            out: str = self._constructArgumentOutput(paraname, inputargs)
        except SyntaxError as e:
            _logging.warning("SyntaxError")
            out = inputargs
        
        _logging.warning(out)
        _logging.getLogger().handlers[0].flush()  # 立即刷新日志

    @staticmethod
    def _get_paraname(callFrame):
        """利用 inspect 和 ast 模块获取变量名称"""
        # 获取调用函数的源代码
        frame_info = _inspect.getframeinfo(callFrame)
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

    def _constructArgumentOutput(self, paraname, inputargs):
        if len(paraname) == 0:
            return ""
        if len(paraname) == 1 and self._isLiteral(paraname[0]):
            if isinstance(inputargs[0], str):
                return inputargs[0]
            return self._argumentToString(inputargs[0])[0]
        
        pairs = [(arg, *self._argumentToString(val)) for arg, val in zip(paraname, inputargs)]
        
        pairStrs = [val if self._isLiteral(arg) else "\033[31m%s:\033[0m " % arg + val for arg, val, _ in pairs]
        allArgsOnOneLine = ", ".join(pairStrs)
        
        multilineArgs = len(allArgsOnOneLine.splitlines()) > 1
        firstLineTooLong = len(allArgsOnOneLine.splitlines()[0]) > 70
        
        if multilineArgs or firstLineTooLong:
            lines = []
            for i, (arg, value, isobject) in enumerate(pairs):
                prefixTooLong =  (len(pairs[i][0]) > 8) or isinstance(inputargs[i], DictObj) or isobject
                lines.append(self._format_pair(arg, value, prefixTooLong))
        else:
            lines = [allArgsOnOneLine]
        
        return "\n".join(lines)

    @classmethod
    def _argumentToString(self, obj, compact=False, indent=0):
        """如果是 ndarray，那么输出它的 __str__，否则用 pprint 得到字符串"""
        import pprint
        isobject = False
        if type(obj) == _np.ndarray:
            if compact:
                s = _np.array2string(obj, formatter={'float_kind': lambda x: "%.2f" % x}, max_line_width=_np.inf, threshold=4)
            else:
                s = obj.__str__()
        elif isinstance(obj, list):
            s = pprint.pformat(obj, compact=True)
        elif isinstance(obj, FunctionType):
            return f"<function {obj.__name__}>"
        else:
            # object
            if (obj.__class__.__str__ is not object.__str__ or obj.__class__.__repr__ is not object.__repr__):
                s = pprint.pformat(obj)
            else:
                s = self._get_custom_object_str(obj)
                isobject = True
        s = s.replace("\\n", "\n")  # Preserve string newlines in output.
        s = '\n'.join(v if i == 0 else " "*indent + v for i, v in enumerate(s.split('\n')))
        return s, isobject

    @staticmethod
    def set_color(s: str, color: str) -> str:
        return f"{color}{s}{COLOR.DEFAULT}"

    @classmethod
    def _get_custom_object_str(self, obj: Any):
        import inspect
        # 首先拿到 header, footer
        obj_type = type(obj)
        header = PrintLn.set_color(f"<{obj_type.__name__} {hex(id(obj))}", COLOR.CYAN)
        footer = PrintLn.set_color(">", COLOR.CYAN)
        
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
            elems += f" {PrintLn.set_color('.' + key, COLOR.GREEN)} = {self._argumentToString(val,compact=True, indent=indent)[0]}\n"
        
        return f"{header}\n{elems}{footer}"
    
    @staticmethod
    def add_object_print(cls):
        cls.__str__ = lambda self: PrintLn._get_custom_object_str(self)
        return cls

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
            prefix = f"\033[31m{prefix}\033[0m"
            
        if prefixTooLong:
            lth = 3
            return [prefix + '\n' + ' '*lth + lines[0]] + [" " * lth + line for line in lines[1:]]
        else:
            return [prefix + lines[0]] + [" " * prefixlen + line for line in lines[1:]]

println = PrintLn()  # 实例化 PrintLn 类

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



# =======
# 字典格式
# =======

class DictObj(dict):
    """`dict` 类的子类。
    
    `DictObj` 类的实例拥有字典的所有属性。此外，该类的实例还可以使用点符号来访问和设置值。
    """
    def __getattr__(self, key:str) -> Any:
        return self[key]

    def __setattr__(self, key:str, value: Any) -> None:
        self[key] = value

    def __repr__(self) -> str:
        
        # 首先拿到 header, footer
        header = PrintLn.set_color(f"<{self.__objname__} {hex(id(self))}", COLOR.CYAN)
        footer = PrintLn.set_color(">", COLOR.CYAN)
        
        # 将属性组装起来
        elems = ""
        for key, value in self.items():
            elems += f" {PrintLn.set_color('.' + key, COLOR.GREEN)} = {PrintLn._argumentToString(value,compact=True)[0]}\n"
        
        return f"{header}\n{elems}{footer}"

def _todictobj(dictionary: Any) -> DictObj:
    """递归的将普通字典转换为 `DictObj` 类的实例。"""
    if not isinstance(dictionary, dict):
        return dictionary
    return DictObj({key: _todictobj(value) for key, value in dictionary.items()})

def idataclass(cls):
    """可以直接使用 todict
    """
    cls.__new__ = todict
    return cls

def todict(cls):
    """改进版本的 idataclass
    
    得到的就是一个字典，无需创建实例！！！
    
    Example:
    >>> import numpy as np
    >>> from quante.basicfun import idataclass, println, save_hdf5
    >>> @todict
    >>> class para:
    >>>     a = 1
    >>>     b = 2
    >>> save_hdf5("path.h5", "/", para)
    """
    newdict = dict()
    for key, value in vars(cls).items():
        if key.startswith('_'+cls.__name__+'__'):
            raise ValueError(f"Can't use '__' to start the key: {key[len('_'+cls.__name__):]}")
        if key not in ('__module__', '__qualname__', '__doc__', '__dict__', '__weakref__', '__new__'):
            newdict[key] = value
    dct = _todictobj(newdict)
    dct.__dict__["__objname__"] = cls.__name__
    return dct

# =======
# 画图预设
# =======
def set_matplotlib(config: Optional[dict] = None, reset: bool = False, svg: bool = True) -> None:
    """
    画图预设

    https://matplotlib.org/stable/tutorials/introductory/customizing.html

    参数:
    config (dict, optional): 自定义配置字典. 默认为 None.
    reset (bool, optional): 是否重置为默认配置. 默认为 False.
    svg (bool, optional): 是否使用 SVG 格式. 默认为 True.
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
    if config is None and reset is False:
        try:
            import matplotlib as _mpl
            _mpl.font_manager.findfont("Times New Roman", fallback_to_default=False)
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
            "legend.fontsize":      13
        }
        _plt.rcParams.update(defaultconfig)
    if reset:
        _mpl.rcParams.update(_mpl.rcParamsDefault)
    if config is not None:
        _plt.rcParams.update(config)


