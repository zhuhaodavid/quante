# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-10-09 18:38:17
# @Last Modified by:   dzwang
# @Last Modified time: 2024-12-14 00:18:23
import numpy as np
from quante.linalg.svd_robust import TruncationError
import torch as tc


def truncate(S, chi_max=None, svd_min=None, trunc_cut=None):
    """ S 需要是降序排列，且全都是正数！ """
    good = tc.ones(len(S), dtype=tc.bool, device=S.device)
    if chi_max is not None:
        good2 = tc.zeros(len(S), dtype=tc.bool, device=S.device)
        good2[:chi_max] = True
        good = good & good2
    if svd_min is not None:
        good = good & (S > svd_min)
    if trunc_cut is not None:
        normS = S / tc.norm(S)
        revert_cumsum = tc.flip((tc.cumsum(tc.flip(normS**2, [0]), 0)), [0])
        good = good & (revert_cumsum > trunc_cut**2)
    eps = tc.square(S[~good]).sum()
    ov = 1. - 2. * eps
    return good, TruncationError(eps, ov)


def svd(tsr:tc.Tensor, *, lr_indx=None, trunc_para=(None, None, None), full_matrices:bool = False) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, TruncationError]:
    r"""张量 svd 分解，返回 A, S, B, error

    .. code-block:: text
    
        . ║          │     │
        --⬜--  ->  --▷--◇--⨞--
          ║          │     │

    Parameters
    ----------

    lrdims 左右指标

    `None` 从正中间的指标分开做 svd
    
    (`left_indx`, `right_indx`): left_indx 为左指标，right_indx 为右指标

    chi_max 保留的奇异值数目

    svd_min 最小奇异值

    trunc_cut 截断的比例

    normalize 是否归一化奇异值

    full_matrices 是否返回完整的矩阵


    Examples
    --------
    >>> tsr = tc.randn(3,7,5,2,dtype=tc.float64)
    >>> u,s,v,e = svd(tsr, lr_indx=[[1,2],[0,3]])
    >>> u.shape
    torch.Size([7, 5, 6])


    .. 警告:: 返回的张量 `U` 和 `V` 不是唯一的，也不是相对于 :attr:`A` 连续的。
                由于这种不唯一性，不同的硬件和软件可能会计算出不同的奇异向量。
                这种不唯一性是由于将任何一对奇异向量 :math:`u_k, v_k` 乘以 `-1`（在实数情况下）或
                乘以 :math:`e^{i \phi}, \phi \in \mathbb{R}`（在复数情况下）会产生另一对有效的奇异向量。
                因此，损失函数不应依赖于这种 :math:`e^{i \phi}` 量，因为它不是良定义的。
                在计算此函数的梯度时，会检查复数输入。因此，当输入是复数且在 CUDA 设备上时，
                计算此函数的梯度会将该设备与 CPU 同步。

    .. 警告:: 使用 `U` 或 `Vh` 计算的梯度仅在 :attr:`A` 没有重复的奇异值时才是有限的。
                如果 :attr:`A` 是矩形的，此外，零也不能是其奇异值之一。
                此外，如果任何两个奇异值之间的距离接近零，梯度将数值不稳定，因为它依赖于奇异值
                :math:`\sigma_i` 通过计算 
                :math:`\frac{1}{\min_{i \neq j} \sigma_i^2 - \sigma_j^2}`。
                在矩形情况下，当 :attr:`A` 有小奇异值时，梯度也将数值不稳定，因为它还依赖于计算
                :math:`\frac{1}{\sigma_i}`。

    """
    chi_max, svd_min, trunc_cut = trunc_para
    shp = tsr.shape

    if lr_indx is None:
        # 如果没给，那就从中间分开
        ac, fg = shp[:len(shp)//2], shp[len(shp)//2:]
        mat = tsr.reshape(np.prod(ac), -1)
    else:
        left_indx, right_indx = lr_indx
        ac = [shp[i] for i in left_indx]
        fg = [shp[i] for i in right_indx]
        mat = tsr.permute(*(left_indx + right_indx)).reshape(np.prod(ac), np.prod(fg))

    U, S, V = tc.linalg.svd(mat, full_matrices=full_matrices)
    good, trunc_err = truncate(S, chi_max, svd_min, trunc_cut)
    if not all(good):
        U = U[:, good]
        S = S[good]
        V = V[good, :]

    return U.reshape(*ac, -1), S, V.reshape(-1, *fg), trunc_err


def qr(tsr:tc.Tensor, *, lr_indx=None) -> tuple[tc.Tensor, tc.Tensor]:
    r"""

    .. code-block:: text
    
               |                           |
              (b)                         (b)
               |            QR             |
        --(a)--⬜--(c)--    ---->    --(a)--▷--(d)--⬜--(c)--
               W                           A       S

    Parameters
    ----------

    lrdims 左右指标

    `None` 从正中间的指标分开做 svd

    (`left_indx`, `right_indx`): left_indx 为左指标，right_indx 为右指标

    Examples
    --------
    >>> tsr = tc.randn(3,7,5,2,dtype=tc.float64)
    >>> q, r = qr(tsr, lr_indx=[[1,2],[0,3]])
    >>> print(q.shape, r.shape)


    .. 警告:: `R` 的对角线元素不一定为正。
            因此，返回的 QR 分解仅在 `R` 的对角线符号上是唯一的。
            因此，不同的平台（如 NumPy）或不同设备上的输入，
            可能会产生不同的有效分解。

    .. 警告:: QR 分解仅在每个矩阵的前 `k = min(m, n)` 列线性无关时才有定义。
            如果不满足此条件，不会抛出错误，但生成的 QR 可能不正确，
            其自动微分可能会失败或产生不正确的结果。
    """
    shp = tsr.shape

    if lr_indx is None:
        # 如果没给，用左右侧的指标分解
        ac = shp[:-1]
        fg = (shp[-1], )
        mat = tsr.reshape(-1, *fg)
    else:
        # 如果给了，需要排序
        left_indx, right_indx = lr_indx
        ac = [shp[i] for i in left_indx]
        fg = [shp[i] for i in right_indx]
        mat = tsr.permute(*(left_indx + right_indx)).reshape(np.prod(ac), np.prod(fg))

    Q, R = tc.linalg.qr(mat)
    return Q.reshape(*ac, -1), R.reshape(-1, *fg)


def rq(tsr:tc.Tensor, *, lr_indx=None) -> tuple[tc.Tensor, tc.Tensor]:
    """
    .. code-block:: text
    
                       |                       |         
                      (b)                     (b)        
                       |           QR          |         
        --(a)--⬜--(d)--⨞--(c)--   <---  --(a)--⬜--(c)--    
               S       A                       W

    Parameters
    ----------

    lrdims 左右指标

    `None` 从正中间的指标分开做 svd
    
    (`left_indx`, `right_indx`): left_indx 为左指标，right_indx 为右指标
    """
    shp = tsr.shape

    if lr_indx is None:
        # 如果没给，用左右侧的指标分解
        ac = (shp[0],)
        fg = shp[1:]
        mat = tsr.reshape(*ac, -1)
    else:
        # 如果给了，需要排序
        left_indx, right_indx = lr_indx
        ac = [shp[i] for i in left_indx]
        fg = [shp[i] for i in right_indx]
        mat = tsr.permute(*(left_indx + right_indx)).reshape(np.prod(ac), np.prod(fg))

    q, r = tc.linalg.qr(mat.T)
    L, U = r.T, q.T

    return L.reshape(*ac, -1), U.reshape(-1, *fg)


def eigh(tsr:tc.Tensor, *, lr_indx=None, direction=None, trunc_para=(None, None, None), pertube=None) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, TruncationError]:
    """
    注意！！
    虽然返回的是 U, S, V，但 U, V 不一定是半幺的！！
    如果 direction="right"，那么 U 是半幺的，V 不是；如果 direction="left"，那么 V 是半幺的，U 不是。
    但总之 tsr = U @ V

    本征分解的原理如图所示（direction="left"为例）：

    .. code-block:: text
    
        目标 - 利用本征分解实现（包含裁剪）：
                 ║                          |        |        
               (bc)                        (b)      (c)       
                 ║                          |        |        
         --(a)---⬜---(d)--     -->   --(a)--▷--(e)---⬜--(d)-- 
                 W                          U        A 


        本征分解是指（得到的上面要的 U，并且通过 S 裁剪）：
         --(a′)---⬜═════╗ 
                W |     ║                  |               |
                (b′)    ║                 (b)             (b′)
           rho    |     ║      eig         |               |
                       (cd)    -->  --(a)--▷--(e)--◇--(e)--⨞--(a′)--
                  |     ║                  U      S^2   U.conj()
                 (b)    ║                  
                W |     ║                     S 为 svd 得到的奇异值
         --(a)----⬜═════╝   


        那么为了得到 A 只需要对第一个式子两边同乘：
                U.conj()              U.conj()                       
            ╭----▷--(e)--        ╭-----▷--(e)--     
            |    |               |     |        |        
           (a)  (b)         ==  (a)   (b)      (c)        
            |    |               |     |        |        
            ╰----⬜══(cd)══       ╰-----▷--(e)---⬜--(d)-- 
                 W                     U        A        

                                       |      
                                      (e)     
                                       |      
                            ==  --(d)--⬜--(b)-- 
                                       A    
    """
    chi_max, svd_min, trunc_cut = trunc_para
    shp = tsr.shape

    if lr_indx is None:
        # 如果没给，那就从中间分开
        ab, cd = shp[:len(shp)//2], shp[len(shp)//2:]
        mat = tsr.reshape(np.prod(ab), -1)
    else:
        left_indx, right_indx = lr_indx
        ab = [shp[i] for i in left_indx]
        cd = [shp[i] for i in right_indx]
        mat = tsr.permute(*(left_indx + right_indx)).reshape(np.prod(ab), np.prod(cd))

    if direction is None:
        # 如果没有给 direction 优先选择小维数方向分解
        direction = "right" if mat.shape[0] < mat.shape[1] else "left"
    else:
        if chi_max is not None:
            chi_max = min(min(mat.shape), chi_max)  # 给定方向时，chi_max 不需要超过最小维数
        else:
            chi_max = min(mat.shape)

        if (direction == "right" and mat.shape[0] > mat.shape[1]) \
            or (direction == "left" and mat.shape[0] < mat.shape[1]):
                print("eig 方向指定，存在效率问题")
                # todo: 此时是否要用qr？

    if direction == "right":
        # mat 的指标为 (ab,cd)，现在可以实施本征分解：
        rho = mat @ mat.conj().T if pertube is None else mat @ mat.conj().T + pertube
        S, U = tc.linalg.eigh(rho)
        S = S.flip(0)
        U = U.flip(1)
        # U 指标是 (ab,e)
        tc.clamp_(S, min=0)  # 将 E 中的负数置为 0，避免根号错误
        tc.sqrt_(S)  # 开根号之后才得到奇异值
        good, trunc_err = truncate(S, chi_max, svd_min, trunc_cut)
        if not all(good):
            U = U[:, good]
            S = S[good]

        # (e,ab) @ (ab,cd) = (e,cd)
        V = U.conj().T @ mat  # 这实际上是 A
    elif direction == "left":
        rho = mat.conj().T @ mat if pertube is None else mat.conj().T @ mat + pertube
        # mat 的指标为 (ab,cd)，现在可以实施本征分解：
        S, V = tc.linalg.eigh(rho)
        S = S.flip(0)
        V = V.flip(1)
        # V 指标是 (cd,e)
        tc.clamp_(S, min=0)  # 将 E 中的负数置为 0，避免根号错误
        tc.sqrt_(S)  # 开根号之后才得到奇异值
        good, trunc_err = truncate(S, chi_max, svd_min, trunc_cut)
        if not all(good):
            V = V[:, good]
            S = S[good]

        # (ab,cd) @ (cd,e) = (ab,e)
        U = mat @ V  # 这实际上是 A
        V = V.T.conj()
    else:
        raise ValueError("direction must be 'left' or 'right'")

    return U.reshape(*ab, -1), S, V.reshape(-1, *cd), trunc_err, direction

def tt_decompose(tsr:tc.Tensor, phys_dim:int|list, trunc_para:tuple=(None,None,None)):
    """
    执行 tt 分解，是 full_contract 的逆过程
    
    输入 tsr 可以是一维或二维数组
    
    返回 tt, Ss, lognm
    
    Examples
    --------
    >>> tsr = tc.randn(2**10, dtype=tc.complex128)
    >>> tt, s, lognm = tensor_train_decompose(tsr, 2)
    >>> tc.dist(tn.full_contract(tt)*tc.exp(lognm), tsr)
    tensor(1.0391e-13, dtype=torch.float64)
    >>> tsr = tc.randn(2**10, 2**10, dtype=tc.complex128)
    >>> tt, s, lognm = tensor_train_decompose(tsr, 2)
    >>> tc.dist(tn.full_contract(tt)*tc.exp(lognm), tsr)
    tensor(7.2339e-12, dtype=torch.float64)
    """
    # 首先检查维数是否正确：
    if isinstance(phys_dim, int):
        tmp = np.log(tsr.shape[0]) / np.log(phys_dim)
        assert tmp.is_integer(), "The physical dimension is not compatible with the tensor shape."
        phys_dim = [phys_dim] * int(tmp)
    else:
        tmp = np.prod(phys_dim)
        assert tsr.shape[0] == tmp, "The physical dimension is not compatible with the tensor shape."
    
    # 然后进行TT分解：
    tt = [None]*len(phys_dim)
    Ss = [None]*(len(phys_dim)+1)
    Ss[-1] = tc.tensor([1.], dtype=tc.float64)
    lognm = 0.
    if tsr.ndim == 1:
        lstdim = 1
        for i in  range(1, len(phys_dim)+1):
            u, s, v = tc.linalg.svd(tsr.reshape(-1, phys_dim[-i]*lstdim), full_matrices=False)
            nms = tc.norm(s)
            s = s/nms
            lognm += tc.log(nms)
            good, _ = truncate(s, *trunc_para)
            u, s, v = u[:, good], s[good], v[good, :]
            tt[-i] = v.reshape(-1, phys_dim[-i], lstdim)
            Ss[-i-1] = s
            tsr = u * s.reshape(1,1,-1)
            lstdim = len(s)
    elif tsr.ndim == 2:
        for i in  range(1, len(phys_dim)+1):
            permute_indx = list(range(len(phys_dim)-1)) \
                + [i+len(phys_dim)-1 for i in range(1,len(phys_dim))] \
                + [len(phys_dim)-1, 2*len(phys_dim)-1, 2*len(phys_dim)]
            tsr = tsr.reshape(*phys_dim, *phys_dim, -1).permute(permute_indx) 
            
            lstdim = tsr.shape[-1]
            u, s, v = tc.linalg.svd(tsr.reshape(-1, phys_dim[-1]**2*lstdim), full_matrices=False)
            nms = tc.norm(s)
            s = s/nms
            lognm += tc.log(nms)
            good, _ = truncate(s, *trunc_para)
            u, s, v = u[:, good], s[good], v[good, :]
            
            tt[-i] = v.reshape(-1, phys_dim[-1], phys_dim[-1], lstdim)
            Ss[-i-1] = s
            tsr = u * s.reshape(1,1,-1)
            phys_dim = phys_dim[:-1]
    else:
        raise ValueError("The tensor must be 1- or 2-dimensional.")
    tt[0] *= tc.sign(u[0,0])
    lognm += tc.log(tc.abs(u[0,0]))
    Ss[0] = tc.tensor([1.], dtype=tc.float64)
    return tt, Ss, lognm

def kron(*mats):
    res = mats[0]
    for i in range(1,len(mats)):
        res = tc.kron(res, mats[i])
    return res