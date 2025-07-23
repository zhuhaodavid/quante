# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-06-11 22:14:32
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-23 21:51:20

import os as _os
import numpy as _np
import scipy as _sp
import h5py as _h5py
import json as _json
import inspect as _inspect

from dataclasses import is_dataclass, asdict
from typing import Callable, Any, Dict, Union, Literal

from .utils_logging import (logger, check_file_exists, get_lv, 
                            flatten_tuple, get_vals,
                            get_args)

__all__ = [
    "save_hdf5",
    "load_hdf5",
    "view_hdf5",
    "exists_hdf5",
    "isave",
    "iload",
]

# -> save

def save_hdf5(
    filename:str, 
    data: dict[str, Any], 
    *,
    group:str='/', 
    mode: Literal["a", "w"] = "a",
) -> tuple[str, str]:
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
    >>> save_hdf5("data.h5", data={"mat": mat}, group="/mygroup")
    """
    assert filename.endswith(".h5"), "Filename must to be `.h5` file."
    assert isinstance(data, dict), "Data must be a dictionary."
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
    subgroup.attrs['nnz'] = csrdata.nnz
    subgroup.create_dataset('data', data=csrdata.data)
    subgroup.create_dataset('indices', data=csrdata.indices)
    subgroup.create_dataset('indptr', data=csrdata.indptr)

def _save_coo(h5group:_h5py.Group, key:str, coodata) -> None:
    subgroup = h5group.create_group(key)
    subgroup.attrs["object_type"] = "coo"
    subgroup.attrs['shape'] = coodata.shape
    subgroup.attrs['nnz'] = coodata.nnz
    subgroup.create_dataset('data', data=coodata.data)
    subgroup.create_dataset('row', data=coodata.row)
    subgroup.create_dataset('col', data=coodata.col)

def _save_csc(h5group:_h5py.Group, key:str, cscdata) -> None:
    subgroup = h5group.create_group(key)
    subgroup.attrs["object_type"] = "csc"
    subgroup.attrs['shape'] = cscdata.shape
    subgroup.attrs['nnz'] = cscdata.nnz
    subgroup.create_dataset('data', data=cscdata.data)
    subgroup.create_dataset('indices', data=cscdata.indices)
    subgroup.create_dataset('indptr', data=cscdata.indptr)
    
def _save_dia(h5group:_h5py.Group, key:str, diadata) -> None:
    subgroup = h5group.create_group(key)
    subgroup.attrs["object_type"] = "dia"
    subgroup.attrs['shape'] = diadata.shape
    subgroup.attrs['nnz'] = diadata.nnz
    subgroup.create_dataset('data', data=diadata.data)
    subgroup.create_dataset('offsets', data=diadata.offsets)

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
    "coo_array": _save_coo,
    "coo_matrix": _save_coo,
    "csc_array": _save_csc,
    "csc_matrix": _save_csc,
    "dia_array": _save_dia,
    "dia_matrix": _save_dia,
    "Tensor": _save_torch,
}
# 也可以在外部文件中自定义其他方法，如：_SAVE_FUC.update({'coo_array':_save_coo})

# -> load

def load_hdf5(filename:str, data:str|list[str]='/', *, group:str='/') -> Any:
    """从 HDF5 文件中加载数据。
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    data : str | list[str]
        要加载的数据名称。
    group : str
        HDF5 文件中的组路径，例如 "/mygroup"。
        
    Returns
    -------
    Any
        加载的数据。

    Examples
    --------
    >>> import numpy as np
    >>> from quante.basicfun import save_hdf5, load_hdf5
    >>> mat = np.random.randn(10,10)
    >>> save_hdf5("data.h5", group="/", data={"mat": mat})
    >>> mat = load_hdf5("data.h5", group="/", data="mat")
    """
    check_file_exists(filename)
    group = "/" + group.strip("/")  # # 规范化组路径 "/xxx/xxx/..."

    logger.debug("Loading from " + _os.path.abspath(filename) + " ... ")
    with _h5py.File(filename.encode("utf-8"), "r") as f:  # `f` is a type `h5py.File`
        group_location = _get_data_location(f, group)
        data = _load_main(group_location, data)
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
        res = []
        names = name.split("/")
        eachname = name
        for eachname in names:
            if eachname == "":
                continue
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
        if isinstance(subgroup, _h5py.Group) and subgroup.attrs.get("object_type", None) is None:
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

def _load_coo(data_location: _h5py.Group) -> _sp.sparse.coo_array:
    data: np.ndarray = data_location["data"][()] # type: ignore
    row: np.ndarray = data_location["row"][()] # type: ignore
    col: np.ndarray = data_location["col"][()] # type: ignore
    shape: tuple = data_location.attrs["shape"] # type: ignore
    return _sp.sparse.coo_array((data, (row, col)), shape=shape, dtype=data.dtype)

def _load_csc(data_location: _h5py.Group) -> _sp.sparse.csc_array:
    indptr: np.ndarray = data_location["indptr"][()] # type: ignore
    indices: np.ndarray = data_location["indices"][()] # type: ignore
    data: np.ndarray = data_location["data"][()] # type: ignore
    shape: tuple = data_location.attrs["shape"] # type: ignore
    return _sp.sparse.csc_array((data, indices, indptr), shape=shape, dtype=data.dtype)

def _load_dia(data_location: _h5py.Group) -> _sp.sparse.dia_array:
    data: np.ndarray = data_location["data"][()] # type: ignore
    offsets: np.ndarray = data_location["offsets"][()] # type: ignore
    shape: tuple = data_location.attrs["shape"] # type: ignore
    return _sp.sparse.dia_array((data, offsets), shape=shape, dtype=data.dtype)

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
    "coo": _load_coo,
    "csc": _load_csc,
    "dia": _load_dia,
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


def exists_hdf5(filename:str, data:Union[str, list[str]]=None, group:str='/') -> bool:
    """检查 HDF5 文件中是否存在指定的组或数据集。
    
    Parameters
    ----------
    filename : str
        HDF5 文件的路径。
    group : str
        HDF5 文件中的组路径，例如 "/mygroup"。
    data : str | list[str], optional
        要检查的数据集名称，可以是单个字符串或字符串列表。默认为 None，表示只检查组是否存在。
    
    Returns
    -------
    bool:
        如果指定的组和数据集存在，则返回 True，否则返回 False。

    Examples
    --------
    >>> exists_hdf5("data.h5", "/mygroup", "mydataset")
    True
    """
    if not _os.path.exists(filename):
        return False

    try:
        with _h5py.File(filename.encode("utf-8"), "r") as f:
            group_location = _get_data_location(f, group)
            if data is None:
                return True  # 只检查组是否存在
            
            if isinstance(data, str):
                return data in group_location
            
            for dataname in data:
                if dataname not in group_location:
                    return False
            return True
    
    except Exception as e:
        logger.error(f"Error checking HDF5 file: {e}")
        return False


def isave(
    filename:str, 
    *dataargs, 
    group:Union[str, None] = '/', 
    mode:str='a'
) -> None:
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
    >>> bf.isave("data.h5", mat, vec)
    """
    call_frame = _inspect.currentframe()
    if call_frame is not None:
        call_frame = call_frame.f_back
    
    if len(dataargs) == 0:
        args = get_lv(call_frame)
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
        
    save_hdf5(filename, data=data_dic, group=group, mode=mode)

    return vals


def iload(filename:str, data:list[str]|str|None = None, *, group='/') -> Any:
    """从 .h5 文件中加载数据.
    
    Parameters
    ----------
    filename : str
        保存的文件名，必须以.h5 结尾。
    data : list[str] | str | None, optional
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
    if data is None:
        call_frame = _inspect.currentframe()
        if call_frame is not None:
            call_frame = call_frame.f_back
        lv = get_lv(call_frame)
        if lv is None:
            # 如果没有找到 lv，那么使用 group 作为默认值
            lv = group
    else:
        # dataset is list or tuple
        assert isinstance(data, (list, tuple, str)), "dataset must be list or tuple."
        lv = data

    return load_hdf5(filename, group=group, data=lv)

