# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-07-04 10:42:20
# @Last Modified by:   hzhu
# @Last Modified time: 2025-07-06 13:22:45

import logging

from tenpy.models import CouplingMPOModel, NearestNeighborModel, Chain
from tenpy.networks import SpinHalfSite, OnsiteTerms, CouplingTerms
from ..operas.spin import SpinOper
from typing import Literal
from warnings import warn

from tenpy.algorithms import (
    get_full_wavefunction, get_numpy_Hamiltonian, get_scipy_sparse_Hamiltonian
)

__all__ = [
    "set_tenpy_logging",
    "tenpy_model_tebd", 
    "tenpy_model_mpo",
    "get_full_wavefunction",
    "get_numpy_Hamiltonian",
    "get_scipy_sparse_Hamiltonian"
]

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



class TenpyTEBDModel(CouplingMPOModel, NearestNeighborModel):
    """A TenPy model for the TEBD algorithm.

    This class is designed to work with the TEBD algorithm in TenPy.

    It initializes the model parameters, including the lattice, operator,
    boundary conditions, and conservation laws. It also sets up the onsite
    and coupling terms based on the provided operator.

    Parameters
    ----------
    model_params : dict
        A dictionary containing the model parameters. The following keys are expected:
        - 'L' : int
            The length of the chain.
        - 'oper' : OperSpin
            The operator defining the Hamiltonian.
        - 'bc_MPS' : str
            The boundary condition for the MPS, either 'finite' or 'periodic'.
        - 'conserve' : str or None
            The conservation law, can be 'parity', 'Sz', or None.
        - 'pauli' : bool
            Whether to use Pauli matrices for the operators.
        - 'explicit_plus_hc' : bool
            Whether to explicitly include the Hermitian conjugate terms in the Hamiltonian.
        - 'lattice' : class
            The lattice class to use, default is Chain.
        - 'random_seed' : int
            Random seed for reproducibility.
        - 'order' : str
            The order of the TEBD algorithm, default is 'default'.
        - 'sort_charge' : bool
            Whether to sort the charge of the sites.
        - 'bc_x' : str
            The boundary condition for the x-direction, either 'open' or 'periodic'.
        - 'helical' : None or str
            Whether to use helical boundary conditions, default is None.
        - 'irregular_remove' : None or str
            Whether to remove irregular sites, default is None.
        - 'sort_mpo_legs' : bool
            Whether to sort the legs of the MPO, default is False.
    """
    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'None', str)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):

        oper = model_params.get('oper', None, None)
        pauli = model_params.get('pauli', False, bool)
        oper._check_pauli(pauli)

        if any(o not in 'IpmZxyz' for opnm in oper.data for o in opnm):
            oper = oper.expandxy(pauli=pauli)
            warn(
                "Operator contains unsupported characters, "
                "expanding to Pauli operators."
            )
        
        Sx = 'Sigmax' if pauli else 'Sx'
        Sy = 'Sigmay' if pauli else 'Sy'
        Sz = 'Sigmaz' if pauli else 'Sz'
        name_map = {'I': 'Id', 'p': 'Sp', 'm': 'Sm', 'Z': 'Sigmaz',
                    'x': Sx, 'y': Sy, 'z': Sz}

        for opnm, pos_strength in oper.data.items():
            if len(opnm) == 1:
                # single-site operator
                tenpy_opname = name_map.get(opnm, None)
                assert tenpy_opname is not None, "Unknown operator name: {}".format(opnm)
                category = tenpy_opname
                ot = self.onsite_terms.setdefault(category, OnsiteTerms(self.lat.N_sites))
                for pos, strength in zip(*pos_strength):
                    ot.add_onsite_term(strength, pos[0], tenpy_opname)
            elif len(opnm) == 2:
                # two-site operator
                op1 = name_map.get(opnm[0], None)
                op2 = name_map.get(opnm[1], None)
                assert op1 is not None and op2 is not None, "Unknown operator name: {}".format(opnm)
                category = "{op1}_i {op2}_j".format(op1=op1, op2=op2)
                ct = self.coupling_terms.setdefault(category, CouplingTerms(self.lat.N_sites))
                for pos, strength in zip(*pos_strength):
                    i, j = pos
                    assert abs(i-j) == 1, "Only nearest-neighbor couplings are supported, got: {}".format(pos)
                    if i < j:
                        o1, o2 = op1, op2
                    else:
                        i, j = j, i
                        o1, o2 = op2, op1
                    ct.add_coupling_term(strength, i, j, o1, o2, 'Id')
            else:
                raise ValueError("Only single-site and two-site operators are supported, got: {}".format(opnm))


def tenpy_model_tebd(
    L: int,
    oper: SpinOper,  
    pauli: bool = False,
    conserve: Literal['Sz', 'parity', 'None'] = 'None',
    bc_MPS: Literal['finite', 'periodic'] = 'finite',
    **kwargs
):
    """Create a TenPy TEBD model.

    Parameters
    ----------
    L : int
        The length of the chain.
    oper : OperSpin
        The operator defining the Hamiltonian.
    pauli : bool, optional
        Whether to use Pauli matrices for the operators. Default is False.
    conserve : str, optional
        The conservation law, can be 'parity', 'Sz', or None. Default is None.
    bc_MPS : str, optional
        The boundary condition for the MPS, either 'finite' or 'periodic'. Default is 'finite'.
    **kwargs : dict, optional
        Additional parameters for the model, such as: 
        - explicit_plus_hc : bool, optional
            Whether to explicitly include the Hermitian conjugate terms in the Hamiltonian. Default is True.
        - lattice : type, optional
            The lattice class to use, default is Chain.
        - random_seed : int, optional
            Random seed for reproducibility. Default is None.
        - order : str, optional
            The order of the TEBD algorithm, default is 'default'.
        - sort_charge : bool, optional
            Whether to sort the charge of the sites. Default is True.
        - bc_x : str, optional
            The boundary condition for the x-direction, either 'open' or 'periodic'. Default is 'open'.
        - helical : str, optional
            Whether to use helical boundary conditions, default is None.
        - irregular_remove : str, optional
            Whether to remove irregular sites, default is None.
        - sort_mpo_legs : bool, optional
            Whether to sort the legs of the MPO, default is False.
    
    Returns
    -------
    TenpyTEBDModel
        An instance of the TenpyTEBDModel class, which is a CouplingMPOModel with SpinHalfSite as the site type.
    """
    model_params = {
        'L': L,
        'oper': oper,
        'pauli': pauli,
        'conserve': conserve,
        'bc_MPS': bc_MPS,
    }
    model_params.update(kwargs)
    return TenpyTEBDModel(model_params)


class TenpyMPOModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'None', str)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        oper = model_params.get('oper', None, None)
        pauli = model_params.get('pauli', False, bool)

        if any(o not in 'IpmZxyz' for opnm in oper.data for o in opnm):
            oper = oper.expandxy(pauli=pauli)
            warn(
            "Operator contains unsupported characters, "
            "expanding to Pauli operators."
            )
        
        oper._check_pauli(pauli)
        Sx = 'Sigmax' if pauli else 'Sx'
        Sy = 'Sigmay' if pauli else 'Sy'
        Sz = 'Sigmaz' if pauli else 'Sz'
        name_map = {'I': 'Id', 'p': 'Sp', 'm': 'Sm', 'Z': 'Sigmaz',
                    'x': Sx, 'y': Sy, 'z': Sz}
        for opnm, pos, strength in oper.each_term():
            term = [(name_map[o], [p, 0]) for o, p in zip(opnm, pos)] 
            self.add_local_term(strength, term, category=opnm)


def tenpy_model_mpo(
    L: int,
    oper: SpinOper,  
    pauli: bool = False,
    conserve: Literal['Sz', 'parity', 'None'] = 'None',
    bc_MPS: Literal['finite', 'periodic'] = 'finite',
    **kwargs
):
    """Create a TenPy MPO model.

    Parameters
    ----------
    L : int
        The length of the chain.
    oper : OperSpin
        The operator defining the Hamiltonian.
    pauli : bool, optional
        Whether to use Pauli matrices for the operators. Default is False.
    conserve : str, optional
        The conservation law, can be 'parity', 'Sz', or None. Default is None.
    bc_MPS : str, optional
        The boundary condition for the MPS, either 'finite' or 'periodic'. Default is 'finite'.
    **kwargs : dict, optional
        Additional parameters for the model, such as: 
        - explicit_plus_hc : bool, optional
            Whether to explicitly include the Hermitian conjugate terms in the Hamiltonian. Default is True.
        - lattice : type, optional
            The lattice class to use, default is Chain.
        - random_seed : int, optional
            Random seed for reproducibility. Default is None.
        - order : str, optional
            The order of the TEBD algorithm, default is 'default'.
        - sort_charge : bool, optional
            Whether to sort the charge of the sites. Default is True.
        - bc_x : str, optional
            The boundary condition for the x-direction, either 'open' or 'periodic'. Default is 'open'.
        - helical : str, optional
            Whether to use helical boundary conditions, default is None.
        - irregular_remove : str, optional
            Whether to remove irregular sites, default is None.
        - sort_mpo_legs : bool, optional
            Whether to sort the legs of the MPO, default is False.
    
    Returns
    -------
    TenpyMPOModel
        An instance of the TenpyMPOModel class, which is a CouplingMPOModel with SpinHalfSite as the site type.
    """
    model_params = {
        'L': L,
        'oper': oper,
        'pauli': pauli,
        'conserve': conserve,
        'bc_MPS': bc_MPS,
    }
    model_params.update(kwargs)
    return TenpyMPOModel(model_params)







