# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-04 10:42:20
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-05 21:51:11

import tenpy
import numpy as np
import tenpy.linalg.np_conserved as npc
from tenpy.models import CouplingModel, NearestNeighborModel, Chain
from tenpy.networks.site import SpinHalfSite
import logging
from ...basicfun import logger

#todo: how to convert a vector in u1 subspace into a tenpy MPS?

def to_tenpy_mps(vec):
    pass

def from_tenpy_mps(psi):
    res = psi.get_B(0, form='B', label_p='0')
    for i in range(1,psi.L):
        res = npc.tensordot(res, psi.get_B(i, form='B', label_p=f'{i}'), axes=('vR', 'vL'))
    return res.to_ndarray().reshape(-1) * psi.norm

#todo: how to convert a oper to a tenpy MPO?

def to_tenpy_mpo(oper):
    pass

def from_tenpy_mpo(oper):
    pass

def set_tenpy_logging(level: int = 1, savelog: bool = False, filenameTime: bool = False, logtime: bool = False, showlevel=False):
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
    from ...basicfun import println, create_folder, logger
    import os as _os
    import sys as _sys
    import time as _time
    import logging as _logging
    
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

    tenpy_logger = logging.getLogger("tenpy")
    tenpy_logger.handlers.clear()
    for h in logger.handlers:
        tenpy_logger.addHandler(h)
    tenpy_logger.setLevel(logger.level)
    tenpy_logger.propagate = False  # 防止重复输出到 root



