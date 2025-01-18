# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2024-07-08 13:53:40
# @Last Modified by:   hzhu
# @Last Modified time: 2025-01-18 16:16:15
# @Description:
#   目的：为了方便使用 torch 编写（带梯度的）张量网络程序，将一些常用的函数集中到此文件夹中。
#   注
#       - 此文件只调用 numpy 和 torch 两个包，不依赖不调用 ./tensor 中任何其他的包。
#       - 这个文件中的函数可以被 ./tensor 中任何其他文件调用。
#       - 根据系统 cuda 是否可用，这个文件中的函数统一使用 device = cuda 或 cpu。
#       - 此文件中的所有函数都应保证梯度链。

import torch as tc

from ..linalg.decomp import qr, svd, truncate, rq, log_or_not_update
from ..utils import clone
from ...linalg.svd_robust import TruncationError

__all__ = [
    "mpo_eye",
    "full_contract",
    "tn_inner",
    "tn_norm",
    "canonicalize",
    "orthogonalize",
    "add",
    "canonicalize_infinite",
    "periodic_trace",
    "batched_canonicalize"
]

if tc.cuda.is_available():
    device = tc.device("cuda")
else:
    device = tc.device("cpu")

dtype = tc.complex128

#######################################################################
# Tensor Train 的加法
#######################################################################

def add(
    Wss: list[list[tc.Tensor]], alphas: float | list[float] = 1.0) -> list[tc.Tensor]:
    """
    计算：α W1s + β W2s，默认 α = β = 1.0
    """
    if isinstance(alphas, float):
        alphas = [alphas] * len(Wss)
    L = len(Wss[0])
    for Ws in Wss:
        if len(Ws) != L:
            raise Exception(f"lenght: L={L} != {len(Ws)}")
    
    res = [None] * L
    res[0] = tc.cat(tuple(Ws[0] for Ws in Wss), dim=-1)

    for i in range(1, L - 1):
        for Ws in Wss:
            assert (Ws[i].shape[1:-1] == Wss[0][i].shape[1:-1]), f"The physical dims of the {i}th local tensor is different for Ws, which is {Ws[i].shape} and {Ws[0][0].shape}, respectively"
        res[i] = _add_each([Ws[i] for Ws in Wss])

    res[-1] = tc.cat(tuple(alphas[i] * Ws[-1] for i,Ws in enumerate(Wss)), dim=0)
    return res


def _add_each(tsrs: list[tc.Tensor]) -> tc.Tensor:
    """
    .. code-block:: text
    
        .      |                  |                         |
              (b)                (b)                       (b)
               |                  |                         |
        --(a)--W1--(d)-- + --(e)--W2--(f)--  --->  --(a+e)--W--(d+f)--

    对 MPS MPO 都适用
    W1, W2 只要有一个的追踪了梯度，那返回的结果就追踪梯度
    """
    # 获得两个张量的维数
    leftbond = [tsr.shape[0] for tsr in tsrs]
    sum_a = sum(leftbond)
    data = []
    a, *b, c = tsrs[0].shape
    rightpart = tc.zeros(sum_a - a, *b, c, device=tsrs[0].device, requires_grad=False)
    data.append(tc.cat((tsrs[0], rightpart), dim=0))
    for i in range(1, len(tsrs)-1):
        a, *b, c = tsrs[i].shape
        leftpart = tc.zeros(sum(leftbond[:i]), *b, c, device=tsrs[i].device, requires_grad=False)
        rightpart = tc.zeros(sum(leftbond[i+1:]), *b, c, device=tsrs[i].device, requires_grad=False)
        data.append(tc.cat((leftpart, tsrs[i], rightpart), dim=0))    
    a, *b, c = tsrs[-1].shape
    leftpart = tc.zeros(sum_a - a, *b, c, device=tsrs[-1].device, requires_grad=False)
    data.append(tc.cat((leftpart, tsrs[-1]), dim=0))
    return tc.cat(data, dim=-1)


def add_many(
    ψs: list[list[tc.Tensor]],
    αs: list[float] = None,
    trunc_para = (None,None,None),
) -> list[tc.Tensor]:
    """计算多个 MPS 相加，边相加边裁剪，但如果裁剪大于等于本身的应有维数，那么裁剪并没有意义
    这样得到的 MPS 是右正则形式，但不能获得奇异谱

    计算：
    .. code-block:: text
    
        |    |   |   |     |    |   |   |
        └-...┴---┴---┘  +  └-...┴---┴---┘

    计算过程：
    .. code-block:: text
    
        1). 最后一个格点：
                                              |
                  |                  |        ▽ V1
        ψ1[-1]  ╭-┘        ψ2[-1]  ╭-┘        |
        ψ1[-1]† ╰-┐  +     ψ2[-1]† ╰-┐   ->   ◇         记录：  --⨞--
                  |                  |        |                  V1
                                              △
                                              |

        2). 倒数第二个格点：

              ┌-╨-┐            ┌-╨-┐       ║
              |   △            |   △       ▽ V2
            ╭-┴---┘          ╭-┴---┘       |                    |  |
            ╰-┬---┐  +       ╰-┬---┐   ->  ◇           记录：  --⨞--⨞--
              |   ▽            |   ▽       |                   V2  V1
              └-╥-┘            └-╥-┘       △
                                           ║

        3). 倒数第三个格点：

          ┌--╨--┐          ┌--╨--┐
          |     △          |     △
          |   ┌-╨-┐        |   ┌-╨-┐       ║
          |   |   △        |   |   △       ▽ V3
        ╭-┴---┴---┘      ╭-┴---┴---┘       |                    |  |  |
        ╰-┬---┬---┐  +   ╰-┬---┬---┐   ->  ◇           记录：  --⨞--⨞--⨞--
          |   |   ▽        |   |   ▽       |                   V3  V2  V1
          |   └-╥-┘        |   └-╥-┘       △
          |     ▽          |     ▽         ║
          └--╥--┘          └--╥--┘

        ... 依次类推

        4). 最后一个格点： 

          |     |          |     |
          |     △          |     △
          |   ┌-╨-...      |   ┌-╨-...                       |     |  |  |
          |   |        +   |   |        ->   --◻--     记录： ◻-...-⨞--⨞--⨞--
          └---┴---...      └---┴---...                            V3  V2  V1

    Examples
    --------
    >>> N = 5
    >>> linkdims = [1] + [4] * (N-1) + [1]
    >>> ψ1 = [tc.randn(linkdims[i],2,linkdims[i+1], dtype=tc.complex128) for i in range(N)]
    >>> ψ2 = [tc.randn(linkdims[i],2,linkdims[i+1], dtype=tc.complex128) for i in range(N)]
    >>> ψ_out2 = add_many([ψ1, ψ2],[1., 1.])
    """
    chi_max, svd_min, trunc_cut = trunc_para
    
    N_mps = len(ψs)
    assert all([len(ψs[1]) == len(ψi) for ψi in ψs])

    if αs is None:
        αs = [1.0] * N_mps

    N = len(ψs[0])

    ψs_mps, phydims = [], []
    for i, ψ in enumerate(ψs):
        ψ_mps, lognm = _left2right_QR(ψ, N)
        ψ_mps[-1] = ψ_mps[-1] * tc.exp(lognm) * αs[i]
        ψs_mps.append(ψ_mps)
        phydim = [ψ_.shape[1:-1] for ψ_ in ψ_mps]
        phydims.append(phydim)

    assert all(phydims[0] == phydim for phydim in phydims)

    ρns = []
    for ψi in ψs_mps:
        a, *s1, b = ψi[-1].shape
        tmp = ψi[-1].reshape(a, -1, b).permute([1, 0, 2]).reshape(-1, a*b)
        ρns.append(tmp @ tmp.conj().T)

    ρn = sum(ρns)

    # Maximum theoretical link dimensions
    add_maxlinkdims = []
    for i in range(N - 1):
        add_maxlinkdims.append(sum([ψ[i].shape[-1] for ψ in ψs_mps]))

    ψ_out = [None] * N

    Cns = [ψ[-1] for ψ in ψs_mps]
    linkdim = 1
    for n in range(N, 1, -1):
        nm = tc.norm(ρn)
        ρn /= nm
        Dn, Vn = tc.linalg.eigh(ρn)

        Dn = Dn.flip(0)
        Vn = Vn.flip(1)
        # V 指标是 (cd,e)
        tc.clamp_(Dn, min=0)  # 将 E 中的负数置为 0，避免根号错误
        tc.sqrt_(Dn)  # 开根号之后才得到奇异值
        good, trunc_err = truncate(Dn, chi_max, svd_min, trunc_cut)
        if not all(good):
            Vn = Vn[:, good]
            Dn = Dn[good]
        
        maxdim = len(Dn)
        Vn = Vn[:, 0:maxdim]
        # Update the total state
        ψ_out[n - 1] = Vn.T.reshape(maxdim, -1, linkdim)
        linkdim = maxdim

        # Compute the new density matrix
        ρnm1 = 0
        for i in range(N_mps):
            # tmp_ = _np.einsum("abc,cde,df->abfe", ψs_mps[i][n - 2], Cns[i], Vn.conj(), optimize=True)

            c, *d, e = Cns[i].shape
            a, *b, c = ψs_mps[i][n - 2].shape
            d, f = Vn.shape

            # [(c,d,e) -> (c,e,d) -> (ce,d) @  (d,f)] = (ce,f) -> (c,ef)
            tmp = (Cns[i].reshape(c,-1,e).permute([0, 2, 1]).reshape(-1, d) @ Vn.conj()).reshape(c, -1)
            # [(a,b,c) -> (ab,c) @ (c,ef)] = (ab,ef) -> (a,b,e,f) -> (a,b,f,e)
            tmp = (
                (ψs_mps[i][n - 2].reshape(-1, c) @ tmp)
                .reshape(a, -1, e, f)
                .permute([0, 1, 3, 2])
            )

            s1, _, _, s2 = tmp.shape
            Cns[i] = tmp.reshape(s1, -1, s2)
            # Cnm1s.append(tmp.reshape(s1, -1, s2))
            tmp = tmp.permute([1, 2, 0, 3]).reshape(-1, s1 * s2)
            ρnm1 += tmp @ tmp.conj().T

        ρn = ρnm1

    ψ_out[0] = sum(Cns).reshape(1, -1, linkdim)

    for i, B in enumerate(ψ_out):
        linkdim1, linkdim2 = B.shape[0], B.shape[-1]
        ψ_out[i] = B.reshape(linkdim1, *phydims[0][i], linkdim2)

    return ψ_out



#######################################################################
# 完全收缩
#######################################################################


def _full_contract_right_mps(res: tc.Tensor, Wsi: tc.Tensor):
    """
    .. code-block:: text
    
        .       |         |                         |
               (b)       (b)          --->         (bd)
                |         |                         |
        --(a)--res--(c)--Wsi--(e)--         --(a)--res--(e)--
    
    >>> res = tc.einsum("abc,cde->abde", res, Ws[i])
    """
    a, _, c = res.shape
    c, _, e = Wsi.shape
    res = res.reshape(-1, c) @ Wsi.reshape(c, -1)
    return res.reshape(a, -1, e)

def _full_contract_right_mps2(res: tc.Tensor, Wsi: tc.Tensor):
    """
    .. code-block:: text
    
        _full_contract_right_mps2:
                |         |                         ║
               (b)       (d)          --->        (b)(d)
                |         |                         ║
        --(a)--res--(c)--Wsi--(e)--         --(a)--res--(e)--
    
    res = tc.einsum("abc,cde->abde", res, Ws[i])
    """
    a, b, c = res.shape
    c, d, e = Wsi.shape
    res = res.reshape(-1, c) @ Wsi.reshape(c, -1)
    return res.reshape(a, b, d, e)

def _full_contract_mps(Ws):
    result = Ws[0]
    for i in range(1, len(Ws)):
        result = _full_contract_right_mps(result, Ws[i])
    return sum(
        result[i, :, i] for i in range(result.shape[0])
    )  # for obc, return result is fine


def _full_contract_right_mpo(res: tc.Tensor, Wsi: tc.Tensor):
    """
    .. code-block:: text
    
        _full_contract_right_mpo:
                |         |                         |
               (b)       (e)                       (be)
                |         |                         |
        --(a)--res--(d)--Wsi--(g)--  --->   --(a)--res--(g)--
                |         |                         |
               (c)       (f)                       (cf)
                |         |                         |
    
    >>> res = tc.einsum("abcd,defg->abecfg", res, Ws[i])
    """
    a, b, c, d = res.shape
    d, e, f, g = Wsi.shape
    res = res.reshape(-1, d) @ Wsi.reshape(d, -1)
    # (a,b,c,e,f,g) -> (a,b,e,c,f,g) -> (a,be,cf,g)
    return (
        res.reshape(a, b, c, e, f, g).swapaxes(2,3)
        .reshape(a, b * e, c * f, g)
    )


def _full_contract_mpo(Ws):
    # mpo -> matrix
    result = Ws[0]
    for i in range(1, len(Ws)):
        result = _full_contract_right_mpo(result, Ws[i])
    return sum(
        result[i, :, :, i] for i in range(result.shape[0])
    )  # for obc, return result is fine


def full_contract(Ws: list[tc.Tensor]) -> tc.Tensor:
    """
    将 MPS(MPO) 完全收缩成为 向量(矩阵)
    根据传入列表中每个张量的结束判断为 MPS(3阶张量) 还是 MPO(4阶张量)
    """
    ndim = Ws[0].ndim
    if ndim == 3:
        return _full_contract_mps(Ws)
    elif ndim == 4:
        return _full_contract_mpo(Ws)
    else:
        # not defined
        raise NotImplementedError(
            f"ndim is {ndim}, which should be 3 for mps or 4 for mpo"
        )

#######################################################################
# 内积
#######################################################################


def _inner_init(Ws1i: tc.Tensor, Ws2i: tc.Tensor):
    """
    .. code-block:: text
    
        --(a)--Ws1i--(c)--         --(a)--┬--(c)--
                |                         |
               (b)          --->         Lenv
                |                         |
        --(d)--Ws2i--(e)--         --(d)--┴--(e)--
    
    >>> Lenv = tc.einsum("abc,dbe->adce", Ws1i, Ws2i)
    """
    a, *_, c = Ws1i.shape
    tsrMat1 = Ws1i.reshape(a, -1, c)

    # (a,b,c) -> (a,c,b) -> (ac,b)
    tsrMat1 = tsrMat1.swapaxes(1,2).reshape(a*c, -1)

    d, *_, e = Ws2i.shape
    tsrMat2 = Ws2i.reshape(d, -1, e)

    # (d,b,e) -> (b,d,e) -> (b,de)
    tsrMat2 = tsrMat2.swapaxes(0,1).reshape(-1, d*e)

    # (ac,b)@(b,de) -> (ac,de) -> (a,c,d,e) -> (a,d,c,e)
    return (tsrMat1 @ tsrMat2).reshape(a, c, d, e).swapaxes(1,2)


def _inner_step(Lenv: tc.Tensor, Ws1i: tc.Tensor, Ws2i: tc.Tensor):
    """
    .. code-block:: text
        
        --(a)--┬--(c)--Ws1i--(f)--         --(a)--┬--(c)--
               |        |                         |
             Lenv      (b)          --->         Lenv
               |        |                         |
        --(d)--┴--(e)--Ws2i--(g)--         --(d)--┴--(e)--

    >>> Lenv = tc.einsum("adce,cbf,ebg->adfg", Lenv, Ws1i, Ws2i)
    """
    e, *b, g = Ws2i.shape
    c, *b, f = Ws1i.shape
    a, d, c, e = Lenv.shape

    # (adc,e) @ (e,bg) -> (adc,bg)
    Lenv = Lenv.reshape(-1, e) @ Ws2i.reshape(e, -1)
    # (adc,bg) -> (ad,cb,g) -> (ad,g,cb) -> (adg,cb)
    Lenv = Lenv.reshape(a*d, -1, g).swapaxes(1,2).reshape(a*d*g, -1)
    # (adg,cb) @ (cb,f) -> (adg,f)
    Lenv = Lenv @ Ws1i.reshape(-1, f)
    # (adg,f) -> (a,d,g,f) -> (a,d,f,g)
    return Lenv.reshape(a, d, g, f).swapaxes(2,3)


def _trace_Lenv(Lenv):
    """
    .. code-block:: text
        
        ╭--(a)--┬--(c)--╮
        ╰       |       ╯
              Lenv            ---->  number
                |
        ╭--(d)--┴--(e)--╮
        ╰               ╯
    
    >>> Lenv = tc.einsum("adad->", Lenv)
    """
    a, d, _, _ = Lenv.shape
    return tc.trace(Lenv.reshape(a * d, a * d))


def tn_inner(
    Ws1: list[tc.Tensor], Ws2: list[tc.Tensor], logscale=False
):
    """计算两个 MPS(MPO) 的内积

    - 如果 logscale = True 那么就返回内积的对数值
    """
    Lenv = _inner_init(Ws1[0], Ws2[0])

    if logscale:
        lognm = 0.0
        for i in range(1, len(Ws2)):
            Lenv = _inner_step(Lenv, Ws1[i], Ws2[i])
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=True)
        return _trace_Lenv(Lenv), lognm
        
    for i in range(1, len(Ws2)):
        Lenv = _inner_step(Lenv, Ws1[i], Ws2[i])
    return _trace_Lenv(Lenv)
   

def tn_norm(Ws: list[tc.Tensor], lognorm=False) -> tc.Tensor:
    """计算 MPS(MPO) 的模（MPO 的模定义为： tr(M†M)）

    如果 lognorm = True 那么就返回模的对数值

    计算图：
    .. code-block:: text
    
        Ws1 ┌---┬--...┬---┐
        Ws2 └---┴--...┴---┘
    
    Example:
    --------
    >>> a = [tc.rand((2,2,2,2), dtype=complexdtype, device=device) for i in range(3)]
    >>> vec_a = tf.full_contract(a)
    >>> print(tc.trace(vec_a.conj().T @ vec_a)**0.5)
    >>> print(tf.tn_norm(a))
    """
    Lenv = _inner_init(Ws[0], Ws[0].conj())

    if lognorm:
        lognm = 0.0
        for i in range(1, len(Ws)):
            Lenv = _inner_step(Lenv, Ws[i], Ws[i].conj())
            Lenv, lognm = log_or_not_update(Lenv, lognm, use_log=True)
        return _trace_Lenv(Lenv).real**0.5, lognm/2
        
    for i in range(1, len(Ws)):
        Lenv = _inner_step(Lenv, Ws[i], Ws[i].conj())
    return _trace_Lenv(Lenv).real**0.5
 

#######################################################################
# 正交中心与正则化
#######################################################################

def _left2right_QR_step(W1:tc.Tensor, W2:tc.Tensor)->tuple[tc.Tensor,tc.Tensor]:
    """
    .. code-block:: text
        _left2right_QR_step
        
               |       |                         |       |
              (b)     (d)                       (b)     (d)
               |       |           QR            |       |
        --(a)--◻--(c)--◻--(e)--   ---->   --(a)--▷--(f)--◻--(e)-- 
               W1      W2                        W1p    W2p
    
    MPS MPO 都可以
    """
    W1p, S = qr(W1)
    c, *e = W2.shape
    W2p = S @ W2.reshape(c, -1)
    return W1p, W2p.reshape(-1, *e)


def _left2right_QR(Ws, L, qrnormalize=False)->tuple[tc.Tensor,tc.Tensor]:
    As, lognm = [None] * L, 0.0
    W1 = Ws[0]
    for i in range(L-1):
        # print(i)
        As[i], W1 = _left2right_QR_step(W1, Ws[i+1])
        W1, lognm = log_or_not_update(W1, lognm, use_log=qrnormalize)
    As[-1] = W1
    As[-1], lognm = log_or_not_update(As[-1], lognm, use_log=qrnormalize)
    return As, lognm


def _right2left_QR_step(W1:tc.Tensor, W2:tc.Tensor):
    """
    .. code-block:: text
        
        .
               |       |                        |       |        
              (b)     (d)                      (b)     (d)       
               |       |           QR           |       |        
        --(a)--◻--(f)--⨞--(e)--   <----  --(a)--◻--(c)--◻--(e)-- 
               W1p    W2p                       W1      W2       
    
    MPS MPO 都可以
    """
    S, W2p = rq(W2)
    *a, f = W1.shape
    W1p = W1.reshape(-1, f) @ S
    return W1p.reshape(*a, -1), W2p


def _SVD_constract_right(A, U, S):
    """
    .. code-block:: text
        
        .      |                                      |
              (b)                                    (b)
               |                                      |
        --(a)--A--(c)--U--(d)--S--(e)  ---->   --(a)--W--(e)--
    
    MPS MPO 都可以
    
    >>> tc.einsum("abc,cd,de->abe", A, U, S)
    """
    *a, c = A.shape
    W = (A.reshape(-1, c) @ U) * S
    return W.reshape(*a, -1)


def _right2left_SVD(As, L, trunc_para=(None,None,None)):
    """
    .. code-block:: text

        .                     |                        |
                              (b)                      (b)
                               |          SVD           |
        --(a)--▷--(d)--◇--(e)--⨞--(c)--  <----   --(a)--◻--(c)--
               U       S       B                        W
    """
    Ss, Bs = [None] * (L + 1), [None] * L
    trunc_err_sum = TruncationError(0.0, 1.0)
    lr_dims = [[0], list(range(1, As[0].ndim))]
    for i in range(L - 1, 0, -1):
        # print(i)
        U, Ss[i], Bs[i], trunc_err = svd(As[i], lr_indx=lr_dims, trunc_para=trunc_para)
        As[i - 1] = _SVD_constract_right(As[i - 1], U, Ss[i])
        trunc_err_sum += trunc_err
    Bs[0] = As[0]
    return Bs, Ss, trunc_err_sum


def canonicalize(
    Ws: list[tc.Tensor], trunc_para=(None,None,None), qrnormalize=False
) -> tuple[list[tc.Tensor], list[tc.Tensor]]:
    """
    将任意的 MPS/MPO Ws 变为标准正交的 MPS/MPO (Bs, Ss)
    """
    assert Ws[0].shape[0] == Ws[-1].shape[-1] == 1, "正则形式只对开边界mps有定义！"
    L = len(Ws)
    # print("QR")
    As, lognm = _left2right_QR(Ws, L, qrnormalize=qrnormalize)
    # print("SVD")
    Bs, Ss, trunc_err = _right2left_SVD(As, L, trunc_para=trunc_para)
    Ss[0] = Ss[-1] = tc.tensor([1.], dtype=Ss[1].dtype, device=Ss[1].device)
    return Bs, Ss, lognm, trunc_err


def orthogonalize(Ws:list[tc.Tensor], j: int)->list[tc.Tensor]:
    """
    .. code-block:: text
        
        .    |      |      |      |      |      |      |
        -----▷------▷------▷------◻------⨞------⨞------⨞-----
            Ws[0]  Ws[1]  Ws[2]  Ws[3]  Ws[4]  Ws[5]  Ws[6]
                                  ↑ 
                                 j=3
    """
    L = len(Ws)
    newWs = clone(Ws)
    for i in range(j-1):
        newWs[i], newWs[i+1] = _left2right_QR_step(newWs[i], newWs[i+1])
    for i in range(L-1,j,-1):
        # print(newWs[i-1].shape, newWs[i].shape)
        newWs[i-1], newWs[i] = _right2left_QR_step(newWs[i-1], newWs[i])
    return newWs


#######################################################################
# 作用门
#######################################################################

def _local_apply(res: tc.Tensor, Wsi: tc.Tensor):
    """
    .. code-block:: text
    
        plot:
               (d)
                |
                ◻
                |
               (b)
                |
        --(a)--res--(c)--
    """
    a, b, *c = res.shape
    d, b = Wsi.shape
    return (Wsi @ res.swapaxes(0,1).reshape(b,-1)).reshape(d, a, *c).swapaxes(0,1)

def _local_apply2(res: tc.Tensor, Ws1: tc.Tensor, Ws2: tc.Tensor):
    """
    .. code-block:: text
    
        _local_apply2:
                |
                ◻
                |
               (b)
                |
        --(a)--res--(c)--
                |
               (d)
                |
                ◻
                |
        
    """
    a, b, d, c = res.shape
    res = res.swapaxes(0,1).swapaxes(2,3)  # (b,a,c,d)
    res = Ws1 @ (res.reshape(-1,d) @ Ws2).reshape(b,-1)
    return res.reshape(b,a,c,d).swapaxes(0,1).swapaxes(2,3)


def _apply_2b_gate_mps(W1:tc.Tensor, W2:tc.Tensor, gate_2b:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      |         |
              (c)       (f)
               |         |
               ├-gate2_b-┤
               |         |                       |         
              (b)       (e)                     (cf)       
               |         |                       |         
        --(a)--◻---(d)---◻--(g)--  ----> --(a)---◻---(g)-- 
               W1        W2                             
    
    >>> tc.einsum("abd,deg,cfbe->acfg", W1, W2, gate_2b)
    """
    a, b, d = W1.shape
    d, e, g = W2.shape
    c, f, b, e = gate_2b.shape
    # (a,b,d) -> (ab,d) @ (d,e,g) -> (d,eg)
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    # (ab,eg) -> (a,b,e,g) -> (b,e,a,g) -> (be,ag)
    W = W.reshape(a,b,e,g).permute([1,2,0,3]).reshape(b*e, -1)
    # (cf,be) @ (be,ag)
    W = gate_2b.reshape(-1,b*e) @ W
    # (cf,ag) -> (c,f,a,g) -> (a,c,f,g)
    res = W.reshape(c,f,a,g).permute([2,0,1,3]).reshape(a,c,f,g)
    # res2 = tc.einsum("abd,deg,cfbe->acfg", W1, W2, gate_2b)
    # print(tc.norm(res-res2))
    return res


def _apply_2b_gate_mpo_from_top(W1:tc.Tensor, W2:tc.Tensor, gate_2b:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      |         | 
              (c)       (f)
               |         | 
               ├-gate2_b-┤ 
               |         |                        |        
              (b)       (e)                      (cf)       
               |         |                        |        
        --(a)--◻---(d)---◻--(g)--  ---->  --(a)---▷---(j)--
               |         |                        |        
              (h)       (i)                      (hi)       
               |         |                        |        
               W1        W2            
    """
    a, b, h, d = W1.shape
    d, e, i, g = W2.shape
    c, f, b, e = gate_2b.shape
    # (a,b,h,d) -> (abh,d) @ (d,e,i,g) -> (d,eig)
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    # (abh,eig) -> (a,b,h,e,i,g) -> (b,e,a,h,i,g) -> (be,ahig)
    W = W.reshape(a,b,h,e,i*g).permute([1,3,0,2,4]).reshape(b*e, -1)
    # (cf,be) @ (be,ahig)
    W = gate_2b.reshape(-1,b*e) @ W
    # (cf,ahig) -> (c,f,a,h,ig) -> (a,c,h,f,ig)
    return W.reshape(c,f,a,h,i*g).permute([2,0,3,1,4]).reshape(a,c,h,f,i,g)


def _apply_2b_gate_mpo_from_bottom(W1:tc.Tensor, W2:tc.Tensor, gate_2b:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      W1        W2                             
               |         |                        |        
              (b)       (e)                      (be)       
               |         |                        |        
        --(a)--◻---(d)---◻--(g)--  ---->  --(a)---◻---(j)--
               |         |                        |        
              (h)       (i)                      (cf)        
               |         |                        |          
               ├-gate2_b-┤
               |         |
              (c)       (f)
               |         |
    """
    a, b, h, d = W1.shape
    d, e, i, g = W2.shape
    h, i, c, f = gate_2b.shape
    # (a,b,h,d) -> (abh,d) @ (d,e,i,g) -> (d,eig)
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    # (abh,eig) -> (a,b,h,e,i,g) -> (a,b,e,g,h,i) -> (abeg,hi)
    W = W.reshape(a,b,h,e,i,g).permute([0,1,3,5,2,4]).reshape(-1, h*i)
    # (abeg,hi) @ (hi,cf)
    W = W @ gate_2b.reshape(h*i, -1)
    # (abeg,cf) -> (a*b,e,g,c,f) -> (a*b,c,e,f,g)
    return W.reshape(a*b,e,g,c,f).permute([0,3,1,4,2]).reshape(a,b,c,e,f,g)


def _apply_2b_gate_mpo_from_topbottom(W1:tc.Tensor, W2:tc.Tensor, gate_2b_tp:tc.Tensor, gate_2b_bt:tc.Tensor) -> tuple[list[tc.Tensor], list[tc.Tensor]]:
    """
    .. code-block:: text
        
        .      |          | 
              (c)        (f)
               |          | 
               ├gate_2b_tp┤                           
               |          |                       |         
              (b)        (e)                    (cf)      
             W1|        W2|                       |       
        --(a)--◻---(d)----◻--(g)--  ---->  --(a)--◻--(g)--
               |          |                       |       
              (h)        (i)                     (jk)     
               |          |                       |        
               ├gate_2b_bt┤
               |          |
              (j)        (k)
               |          |
    """
    a, b, h, d = W1.shape
    d, e, i, g = W2.shape
    c, f, b, e = gate_2b_tp.shape
    h, i, j, k = gate_2b_bt.shape
    # (a,b,h,d) -> (abh,d) @ (d,e,i,g) -> (d,eig)
    W = W1.reshape(-1, d) @ W2.reshape(d, -1)
    # (abh,eig) -> (a,b,h,e,i,g) -> (a,b,e,g,h,i) -> (abeg,hi)
    W = W.reshape(a,b,h,e,i,g).permute([0,1,3,5,2,4]).reshape(-1, h*i)
    # (abeg,hi) @ (hi,jk)
    W = W @ gate_2b_bt.reshape(h*i, -1)
    # (abeg,jk) -> (a,be,g,jk) -> (be,a,jk,g) -> (be,ajkg)
    W = W.reshape(a,b*e,g,j*k).permute([1,0,3,2]).reshape(b*e,-1)
    # (cf,be) @ (be,ajkg)
    W = gate_2b_tp.reshape(-1, b*e) @ W
    # (cf,ajkg) -> (c,f,a,j,k,g) -> (a,c,j,f,k,g)
    return W.reshape(c,f,a,j,k,g).permute([2,0,3,1,4,5]).reshape(a,c,j,f,k,g)

def _resume_canonical_mps(W, V):
    """
    .. code-block:: text
        
        .              V.conj()
                --(e)--⨞--(d)-╮                |     
               |       |      |               (b)
              (b)     (c)     |                | 
               |       |      |   -->   --(a)--⨞--(e)--
        --(a)--╘═══════╛--(d)-╯         
                   W
    
    tc.einsum('abcd,ecd->abe', W, V.conj())
    """
    a, b, c, d = W.shape
    return (W.reshape(-1, c*d) @ V.conj().reshape(-1, c*d).T).reshape(a,b,-1)


def _resume_canonical_mpo(W, V):
    """
    .. code-block:: text
        
        .               ╭╮
                        |
                       (g)
                V.conj()|
                 --(e)--⨞--(d)-╮                |     
                |       |      |               (b)
               (b)     (c)     |                | 
                |       |      |   -->   --(a)--⨞--(e)--
         --(a)--⨞-------⨞--(d)-╯                |
                |       |                      (f)
               (f)     (g)
                |       |
                        ╰╯
                    W
    
    >>> tc.einsum('abfcgd,ecgd->abfe', W, V.conj())
    """
    a,b,f,c,g,d = W.shape
    return (W.reshape(-1, c*g*d) @ V.conj().reshape(-1, c*g*d).T).reshape(a,b,f,-1)


#######################################################################
# 其他操作
#######################################################################

def unitarize(gate_2b:tc.Tensor)->tc.Tensor:
    """
    .. code-block:: text
        
        │         │      └--▽--┘
        ├─gate_2b─┤  -->    │
        │         │      ┌--△--┐
    """
    a, b, c, d = gate_2b.shape
    gate_2b = gate_2b.reshape(a*b, -1)
    u, _, v = tc.linalg.svd(gate_2b, full_matrices=False)
    # u, _, v = tc.svd(gate_2b, compute_uv=True)
    # v = v.conj().T
    return (u @ v).reshape(a,b,c,d)


def _diagonal_contract_step(A, M):
    """
    .. code-block:: text
        
        .       M
        -(a)--┬--(c)------
             (b)
              |
              ├---┐              -->   --(ad)--◻---(ce)--
             (b) (b)           
        -(d)--┼---|--(e)--
             A╰---╯     

        -(a)-┬--(c)----
            (b)
             |
             ├---┐
            (b) (b)
        -(d)-┼---┼--(e)--
             ╰---╯
    
    >>> tc.einsum('dbbe,abc->adce', eig_mpo[n], eig_mps[n])
    """
    d, b, _, e = A.shape
    a, b, c = M.shape
    # (d,b,b,e) -> (d,b,e) -> (d,e,b) -> (de, b)
    res = tc.diagonal(A, dim1=1, dim2=2).reshape(-1, b)
    # (de,b) @ (a,b,c)->(b,ac)
    res = res @ M.permute([1,0,2]).reshape(b, -1)
    # (de,ac) -> (d,e,a,c) -> (a,d,c,e) -> (ad,ce)
    return res.reshape(d,e,a,c).permute([2,0,3,1]).reshape(a*d,-1)


def diagonal_inner(mps, mpo):
    L = len(mps)
    res = _diagonal_contract_step(mpo[0], mps[0])
    for i in range(1,L):
        res = res @ _diagonal_contract_step(mpo[i], mps[i])
    return tc.trace(res)

def mpo_eye(L, local_dims, dtype=tc.complex128, device=None) -> list[tc.Tensor]:
    eyempo = [None] * L
    for i in range(L):
        dim = local_dims[i]
        eyempo[i] = tc.eye(dim, dtype=dtype, device=device).reshape(1, dim, dim, 1)
    return eyempo

def _dm_left2right_mps(Lenv:tc.Tensor, B:tc.Tensor, A:tc.Tensor):
    """
    .. code-block:: text
    
        .         ╭╮          
                  |          
        ╭-╮      (f) B             ╭-╮       
        | ├--(d)--┼--(e)--         | ├--(e)--
        | |      (b)               | |       
        | |       | A              | |       
        | ├--(a)--┴--(c)--         | ├--(c)--
        | ├--(g)--┬--(i)--    -->  | ├--(i)--
        | |       | A.conj()       | |       
        | |      (h)               | |       
        | ├--(j)--┼--(l)--         | ├--(l)--
        ╰-╯       | B.conj()       ╰-╯       
                 (f)         
                  |          
                  ╰╯         
    tc.einsum("adgj,dfbe,abc,ghi,jkhl->ceil", Lenv, B, A, A.conj(), B.conj())
    """
    d,f,b,e = B.shape
    a,d,g,j = Lenv.shape
    a, b, c = A.shape
    # (d,f,b,e) -> (d,f,e,b) -> (dfe,b) @ (a,b,c) -> (b,a,c) -> (b,ac)
    BA = B.permute([0,1,3,2]).reshape(-1,b) @ A.permute([1,0,2]).reshape(b,-1)
    # (dfe,ac) -> (d,f,e,a,c) -> (c,e,f,a,d) -> (cef,ad)
    BA = BA.reshape(d,f,e,a,c).permute([4,2,1,3,0]).reshape(-1, a*d)
    # (cef,ad) @ (ad,gj)
    Lenv = BA @ Lenv.reshape(d*a,g*j)
    # (cef,gj) -> (ce,fgj) @ (ilf,gj) -> (il,fgj) -> (fgj,il)
    Lenv = Lenv.reshape(e*c, -1) @ BA.conj().reshape(e*c, -1).T
    return Lenv.reshape(c,e,c,e)


def _dm_get_R_mps(B:tc.Tensor, A:tc.Tensor, R:tc.Tensor, V:tc.Tensor):
    """
    .. code-block:: text
    
        .                |
                        (c)
                         |
                         ◻ V.conj()
               |         | 
              (g) B     (j)
        --(b)--┼--(h)----┤R          
              (e)        |      -->  --(a,b)--◻--(gc)--
               | A       | 
        --(a)--┴--(f)----╯
    """
    a, e, f = A.shape
    b, g, e, h = B.shape
    f, h, j = R.shape
    
    # (b,g,e,h) -> (b,g,h,e) @ (a,e,f)->(e,a,f)
    res = B.permute([0,1,3,2]).reshape(-1,e) @ A.permute([1,0,2]).reshape(e,-1)
    # (bgh,af) -> (b,g,h,a,f) -> (a,b,g,f,h) @ (fh,j)
    res = res.reshape(b,g,h,a,f).permute([3,0,1,4,2]).reshape(-1,f*h) @ R.reshape(f*h,-1)
    # (abg,j)
    return (res @ V.conj()).reshape(a,b,-1)

def _dm_get_rho(Lenv:tc.Tensor, R:tc.Tensor):
    """
    .. code-block:: text
        
        Lenv               
                  | 
        ╭-╮      (g)
        | ├--(b)--┤
        | |       |R                         |
        | |       |                         (g)
        | ├--(a)--┘                          |
        | ├--(c)--┐                    --->  ◻
        | |       |R.conj()                  |
        | |       |                         (f)
        | ├--(d)--┤                          |
        ╰-╯       |
                 (f)
                  |
    ```
    tc.einsum("abcd,abg,cdf->gf", E[-1], R, R.conj())
    """
    a, b, g = R.shape

    R = R.reshape(-1, g)
    return R.T @ Lenv.reshape(a*b,-1) @ R.conj()

def _dm_get_Lenvs_mps(Ws_mpo, Ws, n, dtype):
    Lenvs = []
    Lenv = tc.tensor(1., dtype=dtype).reshape(1,1,1,1)
    for j in range(n - 1):
        Lenv = _dm_left2right_mps(Lenv, Ws_mpo[j], Ws[j])
        Lenv = Lenv/tc.norm(Lenv)
        Lenvs.append(Lenv)
    return Lenvs

def _apply_on_mps_step(B:tc.Tensor, A:tc.Tensor):
    """
    .. code-block:: text
        
        .      |         
              (e)
               |  B       
        --(d)--┼---(f)---                  |
               |                          (e)
              (b)         -->              |
               |  A                --(ad)--◻--(cf)--
        --(a)--◻---(c)---
    """
    a, b, c = A.shape
    d, e, b, f = B.shape
    
    # (d,e,b,f) -> (d,e,f,b) @ (a,b,c)->(b,a,c)
    res = B.permute([0,1,3,2]).reshape(-1,b) @ A.permute([1,0,2]).reshape(b,-1)
    # (def,ac) -> (d,e,f,a,c) -> (a,d,e,c,f)
    return res.reshape(d,e,f,a,c).permute([3,0,1,4,2]).reshape(a*d,e,-1)


def _apply_on_mpo_step(B:tc.Tensor, A:tc.Tensor):
    """
    .. code-block:: text
        
        .      |         
              (e)
               |  B       
        --(d)--┼---(f)---                  |
               |                          (e)
              (b)         -->              |
               |  A                --(ad)--◻--(cf)--
        --(a)--◻---(c)---                  |
               |                          (g)
              (g)                          |
               |
    """
    a, b, g, c = A.shape
    d, e, b, f = B.shape
    
    # (d,e,b,f) -> (d,e,f,b) @ (a,b,g,c)->(b,a,g,c)
    res = B.permute([0,1,3,2]).reshape(-1,b) @ A.permute([1,0,2,3]).reshape(b,-1)
    # (def,agc) -> (d,e,f,a,g,c) -> (a,d,e,g,c,f)
    return res.reshape(d,e,f,a,g,c).permute([3,0,1,4,5,2]).reshape(a*d,e,g,-1)


def _dm_left2right_mpo(Lenv:tc.Tensor, B:tc.Tensor, A:tc.Tensor):
    """
    .. code-block:: text
        
        .         ╭╮          
                  |          
        ╭-╮      (f) B             ╭-╮       
        | ├--(d)--┼--(e)--         | ├--(e)--
        | |      (b)               | |       
        | |       | A              | |       
        | ├--(a)--┼--(c)--         | ├--(c)--
        | |       |                | |
        | |      (k)               | |
        | |       |                | |
        | ├--(g)--┼--(i)--    -->  | ├--(i)--
        | |       | A.conj()       | |       
        | |      (h)               | |       
        | ├--(j)--┼--(l)--         | ├--(l)--
        ╰-╯       | B.conj()       ╰-╯       
                 (f)         
                  |          
                  ╰╯         
    
    >>> tc.einsum("adgj,dfbe,abc,ghi,jkhl->ceil", Lenv, B, A, A.conj(), B.conj())
    """
    d,f,b,e = B.shape
    a,d,g,j = Lenv.shape
    a, b, k, c = A.shape
    # (d,f,b,e) -> (d,f,e,b) -> (dfe,b) @ (a,b,k,c) -> (b,a,c,k) -> (b,ack)
    BA = B.permute([0,1,3,2]).reshape(-1,b) @ A.permute([1,0,3,2]).reshape(b,-1)
    # (dfe,ack) -> (d,f,e,a,c,k) -> (c,e,f,k,a,d) -> (cefk,ad)
    BA = BA.reshape(d,f,e,a,c,k).permute([4,2,1,5,3,0]).reshape(-1, a*d)
    # (cefk,ad) @ (ad,gj)
    Lenv = BA @ Lenv.reshape(d*a,g*j)
    # (cefk,gj) -> (ce,fkgj) @ (ilfk,gj) -> (il,fkgj) -> (fkgj,il)
    Lenv = Lenv.reshape(e*c, -1) @ BA.conj().reshape(e*c, -1).T
    return Lenv.reshape(c,e,c,e)


def _dm_get_R_mpo(B:tc.Tensor, A:tc.Tensor, R:tc.Tensor, V:tc.Tensor):
    """
    .. code-block:: text
        
        .                |
                        (c)
                         |
                         ◻ V.conj()
               |         | 
              (g) B     (j)
        --(b)--┼--(h)----┤R          
              (e)        |      -->  --(a,b)--◻--(g,k,c)--
               | A       | 
        --(a)--┼--(f)----╯
              (k)
               │
    
    >>> tc.einsum("bgeh,aekf,fhj,jc->abgkc", B, A, R, V.conj)
    """
    a, e, k, f = A.shape
    b, g, e, h = B.shape
    f, h, j = R.shape
    
    # (b,g,e,h) -> (b,g,h,e) @ (a,e,k,f)->(e,a,k,f)
    res = B.permute([0,1,3,2]).reshape(-1,e) @ A.permute([1,0,2,3]).reshape(e,-1)
    # (bgh,akf) -> (b,g,h,a,k,f) -> (a,b,g,k,f,h) @ (fh,j)
    res = res.reshape(b,g,h,a,k,f).permute([3,0,1,4,5,2]).reshape(-1,f*h) @ R.reshape(f*h,-1)
    # (abgk,j)
    return (res @ V.conj()).reshape(a,b,-1)


def _dm_get_Lenvs_mpo(Ws_mpo, Ws, n, dtype):
    Lenvs = []
    Lenv = tc.tensor(1., dtype=dtype).reshape(1,1,1,1)
    for j in range(n - 1):
        Lenv = _dm_left2right_mpo(Lenv, Ws_mpo[j], Ws[j])
        Lenv = Lenv/tc.norm(Lenv)
        Lenvs.append(Lenv)
    return Lenvs


def _up_bottom_tr(tsr):
    """
    .. code-block:: text
        
        .      ╭╮
               │
              (b)
               │
        --(a)--┼--(d)--  ⟶  --(a)--◻--(d)--
               │
              (b)
               │
               ╰╯
    
    >>> tc.einsum('abbc->ac',tsr)
    """
    diag_elements = tsr.permute([0,3,1,2]).diagonal(offset=0, dim1=-2, dim2=-1)
    return diag_elements.sum(-1)


def _ProjMPO_contract_right_env(H:tc.Tensor, psi:tc.Tensor, Renv:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      ╭-╮          psi.conj()╭-╮
        --(a)--┤ │     --(a)--◻--(b)--┤ │
               │ │            │       │ │
               │ │           (c)      │ │
               │ │            │H      │ │
        --(d)--┤ │ <-  --(d)--◻--(e)--┤ │
               │ │            │       │ │
               │ │           (f)      │ │
               │ │            │       │ │
        --(g)--┤ │     --(g)--◻--(h)--┤ │
               ╰-╯          psi       ╰-╯
     
    >>> tc.einsum("acb,dcfe,gfh,beh->adg", psi.conj(), H, psi, Renv)
    """
    b, e, h = Renv.shape
    g, f, h = psi.shape
    d, c, f, e = H.shape
    a = g
    
    # (b,e,h) -> (be,h) @ (g,f,h) -> (h,f,g) -> (h,fg) = (be,fg)
    out = Renv.reshape(-1,h) @ psi.permute([2,1,0]).reshape(h, -1)
    # (d,c,f,e) -> (d,c,e,f) -> (dc,ef) @ (be,fg) -> (b,ef,g) -> (ef,b,g) -> (ef,bg) = (dc,bg)
    out = H.permute([0,1,3,2]).reshape(-1, e*f) @ out.reshape(b, e*f, g).permute([1,0,2]).reshape(e*f, -1)
    # (a,c,b) -> (a,cb) @ (dc,bg) -> (d,cb,g) -> (cb,d,g) -> (cb,dg) = (a,dg)
    out = psi.conj().reshape(a,-1) @ out.reshape(d, -1, g).permute([1,0,2]).reshape(c*b, -1)
    
    return out.reshape(a,d,g)


def _ProjMPS_contract_right_env(H:tc.Tensor, psi:tc.Tensor, Renv:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      ╭-╮          H.conj()  ╭-╮
        --(d)--┤ │ <-  --(d)--◻--(e)--┤ │
               │ │            │       │ │
               │ │           (f)      │ │
               │ │            │       │ │
        --(g)--┤ │     --(g)--◻--(h)--┤ │
               ╰-╯          psi       ╰-╯
     
    >>> tc.einsum("dfe,gfh,eh->dg", H.conj(), psi, Renv)
    """
    e, h = Renv.shape
    g, f, h = psi.shape
    return (H.conj().reshape(-1, e) @ Renv).reshape(-1, f*h) @ psi.reshape(g, -1).T


def _ProjMPO_contract_left_env(H:tc.Tensor, psi:tc.Tensor, Lenv:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        ╭-╮   psi.conj()          ╭-╮       
        │ ├--(a)--◻--(b)--        │ ├--(b)--
        │ │       │               │ │       
        │ │      (c)              │ │       
        │ │       │H              │ │       
        │ ├--(d)--◻--(e)--  --->  │ ├--(e)--
        │ │       │               │ │       
        │ │      (f)              │ │       
        │ │       │               │ │       
        │ ├--(g)--◻--(h)--        │ ├--(h)--
        ╰-╯     psi               ╰-╯       
     
    tc.einsum("adg,acb,dcfe,gfh->beh", Lenv, psi.conj(), H, psi)
    """
    a, d, g = Lenv.shape
    g, f, h = psi.shape
    d, c, f, e = H.shape
    
    # (a,d,g) -> (ad,g) @ (g,f,h) -> (g,fh) = (ad,fh)
    out = Lenv.reshape(-1,g) @ psi.reshape(g, -1)
    # (ad,fh) -> (a,df,h) -> (a,h,df) -> (ah,df) @ (d,c,f,e) -> (d,f,c,e) -> (df,ce) = (ah,ce)
    out = out.reshape(a,-1,h).permute([0,2,1]).reshape(-1, d*f) @ H.permute([0,2,1,3]).reshape(d*f, -1)
    # (ah,ce) -> (a,h,c,e) -> (e,h,a,c) -> (eh,ac) @ (a,c,b) -> (ac,b) = (eh,b) -> (b,eh)
    out = out.reshape(a,h,c,e).permute([3,1,0,2]).reshape(e*h, -1) @ psi.conj().reshape(a*c,-1)
    
    return out.T.reshape(h,e,h)


def _ProjMPS_contract_left_env(H:tc.Tensor, psi:tc.Tensor, Lenv:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        ╭-╮   M.conj()            ╭-╮       
        │ ├--(d)--◻--(e)--  --->  │ ├--(e)--
        │ │       │               │ │       
        │ │      (f)              │ │       
        │ │       │               │ │       
        │ ├--(g)--◻--(h)--        │ ├--(h)--
        ╰-╯     psi               ╰-╯       
     
    tc.einsum("dg,dfe,gfh->eh", Lenv, M.conj(), psi)
    """
    d, g = Lenv.shape
    d, f, e = H.shape
    g, f, h = psi.shape
    
    # (d,g) @ (g,f,h) -> (g,fh) = (d,fh)
    out = Lenv @ psi.reshape(g, -1)
    # (d,fh) -> (df,h) -> (h,df) @ (d,f,e) -> (df,e) = (h,e)
    out = out.reshape(-1,h).T @ H.conj().reshape(d*f, -1)
    return out.T


def _matrix_vector_product0(Lenv, Renv, v):
    """
    .. code-block:: text
    
        ╭-╮                 ╭-╮ 
        │ ├--(a)-     -(c)--┤ │ 
        │ │                 │ │ 
        │ │                 │ │ 
        │ │                 │ │              
        │ ├--(f)-------(h)--┤ │  -->    --(a)--(c)-         
        │ │                 │ │     
        │ │                 │ │ 
        │ │                 │ │ 
        │ ├--(k)---◻---(m)--┤ │ 
        ╰-╯       psi       ╰-╯ 
        Lenv                Renv
    
    输入：
    
    Lenv: (a, fk)
    
    Renv: (m, hc)
    
    v: (km)
    
    前期准备：
    ----------
    >>> Lenv = np.ascontiguousarray(Lenv.reshape(Lenv.shape[0], -1))
    >>> H12 = np.einsum("fdig,gejh->defijh", H1, H2)
    >>> d, e, f, *ijh = H12.shape
    >>> H12 = np.ascontiguousarray(H12.reshape(f*d*e, -1))
    >>> Renv = Renv.transpose([2,1,0])
    >>> Renv = np.ascontiguousarray(Renv.reshape(Renv.shape[0], -1))
    
    np.einsum("afk,fdig,gejh,kijm,chm->adec", Lenv, H1, H2, psi, Renv)
    """
    m, _ = Renv.shape
    a, fa = Lenv.shape
    f = fa // a
    #                 km => khc -> h,k,c -> hk,c => ac
    out = (Lenv @ (v.reshape(-1,m) @ Renv).reshape(a, f, m).swapaxes(0, 1).reshape(fa, -1)).reshape(-1)
    # assert tc.dist(out, tc.einsum("afk,mhc,km->ac", Lenv.reshape(a,f,a), Renv.reshape(a,f,a), v.reshape(a,a)).reshape(-1)) < 1e-10
    return out


def _trace_matrix_vector_product0(Lenv, Renv):
    """
    .. code-block:: text
    
        ╭-╮                 ╭-╮ 
        │ ├--(a)-╯   ╰-(c)--┤ │ 
        │ │                 │ │ 
        │ │                 │ │ 
        │ │                 │ │              
        │ ├--(f)-------(h)--┤ │  -->    --(a)--(c)-         
        │ │                 │ │     
        │ │                 │ │ 
        │ │                 │ │ 
        │ ├--(a)-╮   ╭-(c)--┤ │ 
        ╰-╯                 ╰-╯ 
        Lenv                Renv
    
    输入：
    
    Lenv: (a, fk)
    
    Renv: (m, hc)
    
    v: (km)
    
    前期准备：
    ----------
    >>> Lenv = np.ascontiguousarray(Lenv.reshape(Lenv.shape[0], -1))
    >>> H12 = np.einsum("fdig,gejh->defijh", H1, H2)
    >>> d, e, f, *ijh = H12.shape
    >>> H12 = np.ascontiguousarray(H12.reshape(f*d*e, -1))
    >>> Renv = Renv.transpose([2,1,0])
    >>> Renv = np.ascontiguousarray(Renv.reshape(Renv.shape[0], -1))
    
    np.einsum("afk,fdig,gejh,kijm,chm->adec", Lenv, H1, H2, psi, Renv)
    """
    c, hc = Renv.shape
    h = hc // c
    a, fa = Lenv.shape
    f = fa // a
    
    return tc.einsum("afa->f", Lenv.reshape(a,f,a)) @ tc.einsum("afa->f", Renv.reshape(c, h, c))



def _trace_matrix_vector_product(Lenv, H12, Renv):
    """
    .. code-block:: text
    
        ╭-╮                       ╭-╮ 
        │ ├--(a)-╯         ╰-(c)--┤ │ 
        │ │       │       │       │ │ 
        │ │      (d)     (e)      │ │ 
        │ │       │H1     │H2     │ │              │       │
        │ ├--(f)--◻--(g)--◻--(h)--┤ │  -->        (d)     (e)
        │ │       │       │       │ │       --(a)--┴-------┴--(c)- 
        │ │      (i)     (j)      │ │ 
        │ │       │       │       │ │ 
        │ ├--(k)-╮         ╭-(m)--┤ │ 
        ╰-╯         psi           ╰-╯ 
        Lenv                      Renv
    
    输入：
    
    Lenv: (a, fk)
    
    H12: (def, ijh)
    
    Renv: (m, hc)
    
    v: (kijm)
    
    前期准备：
    ----------
    >>> Lenv = np.ascontiguousarray(Lenv.swapaxes(1,2).reshape(Lenv.shape[0], -1))
    >>> H12 = np.einsum("fdig,gejh->defijh", H1, H2)
    >>> d, e, f, *ijh = H12.shape
    >>> H12 = np.ascontiguousarray(H12.reshape(f*d*e, -1))
    >>> Renv = Renv.transpose([2,1,0])
    >>> Renv = np.ascontiguousarray(Renv.reshape(Renv.shape[0], -1))
    
    np.einsum("afk,fdig,gejh,kijm,chm->adec", Lenv, H1, H2, psi, Renv)
    """
    a, fa = Lenv.shape
    f = fa//a
    Lenv = Lenv.reshape(a, f, a)
    
    m, mh = Renv.shape
    h = mh//m
    Renv = Renv.reshape(m, h, m)
    
    ddf, iih = H12.shape
    dd = ddf//f
    ii = iih//h
    H12 = H12.reshape(dd, f, ii, h)
    
    return tc.einsum("afa->f",Lenv) @ tc.einsum("abac->bc", H12) @ tc.einsum("afa->f",Renv)
    

def _matrix_vector_product(Lenv, H12, Renv, v):
    """
    .. code-block:: text
    
        ╭-╮                       ╭-╮ 
        │ ├--(a)-           -(c)--┤ │ 
        │ │       │       │       │ │ 
        │ │      (d)     (e)      │ │ 
        │ │       │H1     │H2     │ │              │       │
        │ ├--(f)--◻--(g)--◻--(h)--┤ │  -->        (d)     (e)
        │ │       │       │       │ │       --(a)--┴-------┴--(c)- 
        │ │      (i)     (j)      │ │ 
        │ │       │       │       │ │ 
        │ ├--(k)--┴-------┴--(m)--┤ │ 
        ╰-╯         psi           ╰-╯ 
        Lenv                      Renv
    
    输入：
    
    Lenv: (a, fk)
    
    H12: (def, ijh)
    
    Renv: (m, hc)
    
    v: (kijm)
    
    前期准备：
    ----------
    >>> Lenv = np.ascontiguousarray(Lenv.swapaxes(1,2).reshape(Lenv.shape[0], -1))
    >>> H12 = np.einsum("fdig,gejh->defijh", H1, H2)
    >>> d, e, f, *ijh = H12.shape
    >>> H12 = np.ascontiguousarray(H12.reshape(f*d*e, -1))
    >>> Renv = Renv.transpose([2,1,0])
    >>> Renv = np.ascontiguousarray(Renv.reshape(Renv.shape[0], -1))
    
    np.einsum("afk,fdig,gejh,kijm,chm->adec", Lenv, H1, H2, psi, Renv)
    """
    m, _ = Renv.shape
    a, fa = Lenv.shape
                            #    kijhc -> k,ijh,c -> ijh,kc => def,kc -> de,fk,c
    return (Lenv @ (H12 @ (v.reshape(-1,m) @ Renv).reshape(a, -1, m).swapaxes(0, 1).reshape(-1, a*m)).reshape(-1, fa, m).swapaxes(0, 1).reshape(fa, -1)).reshape(-1)


def _projMPS_make_vec(Lenv, H12, Renv):
    """
    .. code-block:: text
    
        ╭-╮                       ╭-╮ 
        │ ├--(f)--◻--(g)--◻--(h)--┤ │
        │ │       │       │       │ │ 
        │ │      (i)     (j)      │ │ 
        │ │       │       │       │ │ 
        │ ├--(k)--         --(m)--┤ │ 
        ╰-╯                       ╰-╯ 
        Lenv                      Renv
    
    输入：
    
    Lenv: (k, f)
    
    H12: (f, ijh)
    
    Renv: (h, m)
    
    前期准备：
    ----------
    >>> M12 = np.einsum("fig,gjh->fijh", M1, M2)
    >>> f, i, j, h = M12.shape
    >>> M12 = np.ascontiguousarray(M12.reshape(f, -1))
    >>> Renv = np.ascontiguousarray(Renv.T)
    
    np.einsum("fk,fijh,kijm,hm->", Lenv, M12, Renv)
    """
    h, _ = Renv.shape
    return (Lenv @ H12).reshape(-1, h) @ Renv


def make_matrix(Lenv:tc.Tensor, H12:tc.Tensor, Renv:tc.Tensor):
    """
    .. code-block:: text
    
        ╭-╮                ╭-╮ 
        │ ├--(a)-    -(c)--┤ │ 
        │ │       │        │ │ 
        │ │      (d)       │ │ 
        │ │       │H12     │ │              │    │    │
        │ ├--(f)--◻--(g)---┤ │  -->        (a)  (d)  (c)
        │ │       │        │ │              ├----┼----┤
        │ │      (i)       │ │             (k)  (i)  (m)
        │ │       │        │ │              │    │    │
        │ ├--(k)--  --(m)--┤ │ 
        ╰-╯                ╰-╯ 
        Lenv                      Renv
    
    a,fk
    df,ig
    m,gc
    
    tc.einsum("afk,fdig,cgm->adckim", Lenv, H1, H2, Renv)
    """
    a, fk = Lenv.shape
    f = fk // a
    df, ig = H12.shape
    d = df // f
    m,gc = Renv.shape
    g = gc // m
    
    # (d,f,i,g) @ (g,m,c) = (dfi,mc)
    out = H12.reshape(-1, g) @ Renv.reshape(m,g,m).swapaxes(0,1).reshape(g, -1)
    
    # (a,fk) @ (f,d,imc) = (ak,dimc)
    out = Lenv.reshape(a,f,a).swapaxes(1,2).reshape(-1, f) @ out.reshape(d,f,-1).swapaxes(0,1).reshape(f,-1)
    
    # (a,k,d,i,c,m) -> (a,d,c,k,i,m) -> (adec,kijn)
    out = out.reshape(a,a,d,d,m,m).permute([0,2,5,1,3,4]).reshape(a*d*m,-1)
    
    # out2 = tc.einsum("afk,fdig,gejh,chm->adeckijm", Lenv, H1, H2, Renv)
    # out2 = out2.reshape(a*d*e*c, k*i*j*m)
    return out


def make_matrix0(Lenv:tc.Tensor, Renv:tc.Tensor):
    """
    .. code-block:: text
    
        ╭-╮                ╭-╮ 
        │ ├--(a)-    -(c)--┤ │ 
        │ │                │ │ 
        │ │                │ │ 
        │ │                │ │              │    │
        │ ├--(f)-----(g)---┤ │  -->        (a)  (c)
        │ │                │ │              ├----┤
        │ │                │ │             (k)  (m)
        │ │                │ │              │    │
        │ ├--(k)-    -(m)--┤ │ 
        ╰-╯                ╰-╯ 
        Lenv                      Renv
    
    a,fk
    m,gc
    
    tc.einsum("afk,fdig,cgm->adckim", Lenv, H1, H2, Renv)
    """
    a, fk = Lenv.shape
    m,gc = Renv.shape
    g = gc // m
    
    # (a,fk) -> (a,k,f) @ (m,gc) -> (g,m,c) = (ak,mc)
    out = Lenv.reshape(a,g,a).swapaxes(1,2).reshape(-1, g) @ Renv.reshape(m,g,m).swapaxes(0,1).reshape(g, -1)
    # (a,k,m,c) -> (a,c,k,m)
    out = out.reshape(a,a,m,m).permute([0,3,1,2]).reshape(a*m,-1)
    
    # assert tc.dist(out.reshape(-1), tc.einsum("afk,mfc->ackm", Lenv.reshape(a,g,a), Renv.reshape(m,g,m)).reshape(-1)).item() < 1e-10
    return out



def _prepare_solve_ground_state(H1:tc.Tensor, H2:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
    
        .      |       |
              (d)     (e)
               |       |
        --(f)--◻--(g)--◻--(h)--
               |       |
              (i)     (j)
               |       |
    
    >>> tc.einsum("fdig,gejh->defijh", H1, H2)
    """
    f,d,i,g = H1.shape
    g,e,j,h = H2.shape
    # (fdi,g) @ (g,ejh) = (fdi,ejh)
    out = H1.reshape(-1, g) @ H2.reshape(g, -1)
    
    # (f,d,i,e,jh) -> (d,e,f,i,jh)
    return out.reshape(f,d,i,e,j*h).permute([1,3,0,2,4]).reshape(d*e*f,-1)


def _projMPS_prepare_solve_ground_state(M1:tc.Tensor, M2:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
    
        --(f)--◻--(g)--◻--(h)--
               |       |
              (i)     (j)
               |       |
    
    >>> tc.einsum("fig,gjh->fijh", M1, M2)
    """
    f,_,g = M1.shape
    return (M1.reshape(-1, g) @ M2.reshape(g, -1)).reshape(f,-1)


def _cholesky_decomp(VL):
    vld = tc.diag(VL)
    tmp = vld[tc.abs(vld)>1e-10][0].conj()
    tmp /= tc.norm(tmp)
    newVL = VL * tmp
    newVL = (newVL + newVL.conj().T)/2
    
    D, WY = tc.linalg.eigh(newVL)

    Y = WY * tc.sqrt(D).reshape(1,-1)
    Yinv = WY.conj().T / tc.sqrt(D).reshape(-1,1)
    
    return Y, Yinv

def canonicalize_infinite(tsr:tc.Tensor):
    r"""
    tsr 是一个三阶张量，将它变换成正则形式
    
    .. code-block:: text
    
        初始平移不变：
           │   │   │
        ───◻───◻───◻───
    
        求解本征问题：
        ╭╮                  ╭╮         
        │├──◻──             │├─   ┌◻─
        ││  │   = \lamdba_l ││  = │
        │├──◻──             │├─   └◻─
        ╰╯                  ╰╯     Y  

             ╭╮              ╭╮      
        ──◻──┤│             ─┤│   ─◻┐
          │  ││ = \lamdba_r  ││ =   │
        ──◻──┤│             ─┤│   ─◻┘
             ╰╯              ╰╯    X 
    
        插入：
    
           │              │              │
        ───◻──◻──◻──◻──◻──◻──◻──◻──◻──◻──◻───
             Y⁻¹ Y  X X⁻¹    Y⁻¹ Y  X X⁻¹
    
        svd 分解：
        ──◻──◻── = ──▷──◇──⨞──
          Y  X       U  S  V
    
        那么：
           │                     │                     │
        ───◻──◻──▷─  ─◇─  ─⨞──◻──◻──◻──▷─  ─◇─  ─⨞──◻──◻───
             Y⁻¹ U    S    V  X⁻¹   Y⁻¹ U    S    V  X⁻¹
                      ↑   └─────────────┘   ↑
                                res
    
    这样 `s * res` 是左正交形式，`res * s` 是右正交形式
    """
    # 拿到转移矩阵：
    tsf_mat = _inner_init(tsr.conj(), tsr)
    a,b,c,d = tsf_mat.shape
    tsf_mat = tsf_mat.reshape(a*b,c*d)
    
    dlsy, VLs = tc.linalg.eig(tsf_mat)
    VR = VLs[:, 0].reshape(c,d)
    # print(tc.dist(tc.einsum("abc,dbe,ce->ad", tsr.conj(), tsr, VR), dlsy[0]*VR))
    
    dlsx, VRs = tc.linalg.eig(tsf_mat.T)
    VL = VRs[:, 0].reshape(a,b)
    # print(tc.dist(tc.einsum("ad,abc,dbe->ce", VL, tsr.conj(), tsr), dlsx[0]*VL))
    
    X, Xinv = _cholesky_decomp(VR)
    # print(tc.dist(VR, X @ X.conj().T))
    X, Xinv = X.conj(), Xinv.conj()
    # print(tc.dist(X @ Xinv, tc.eye(*X.shape)))
    # print(tc.dist(tc.einsum("abc,dbe,cf,ef->ad", tsr.conj(), tsr, X.conj(), X), dlsy[0]*tc.einsum("cf,ef->ce", X.conj(), X)))
    
    Y, Yinv = _cholesky_decomp(VL)
    # print(tc.dist(VL, Y @ Y.conj().T))
    Y, Yinv = Y.conj().T, Yinv.conj().T
    # print(tc.dist(tc.einsum("ba,bc->ac", Y.conj(), Y), VL))
    # print(tc.dist(Y @ Yinv, tc.eye(*Y.shape)))
    # print(tc.dist(tc.einsum("fa,fd,abc,dbe->ce", Y.conj(), Y, tsr.conj(), tsr), dlsx[0]*VL))
    U, sv, V = tc.linalg.svd(Y @ X)
    sVXinv = V @ Xinv
    YinvU = Yinv @ U
    newtsr = tc.einsum("ab,bec,cd->aed", sVXinv, tsr, YinvU)  # todo 用矩阵乘法来计算
    return newtsr, sv, dlsy[0], dlsy[1]


def periodic_trace(tsr:list[tc.Tensor]):
    psis = []
    psis_num = tsr[0].shape[0]
    length = len(tsr)
    for j in range(psis_num):
        psi = []
        for i in range(length):
            if i == 0:
                psi.append(tsr[i][j:j+1, ...])
            elif i == length - 1:
                psi.append(tsr[i][..., j:j+1])
            else:
                psi.append(tsr[i])
        psis.append(psi)
    
    if psis_num == 1:
        return psis[0]
    
    return add(psis)

def _batched_left2right_QR_step(W1:tc.Tensor, W2:tc.Tensor)->tuple[tc.Tensor,tc.Tensor]:
    """
    .. code-block:: text

        .
                |       |                         |       |
               (b)     (d)                       (b)     (d)
                |       |           QR            |       |
         --(a)--⬜--(c)--⬜--(e)--   ---->   --(a)--▷--(f)--⬜--(e)-- 
    
    MPS MPO 都可以
    """
    batch_size, *shp, c = W1.shape
    W1p, S = tc.linalg.qr(W1.reshape(batch_size, -1, c))
    # tc.cuda.synchronize()
    batch_size, c, *e = W2.shape
    W2p = tc.bmm(S, W2.reshape(batch_size, c, -1))
    # tc.cuda.synchronize()
    return W1p.reshape(batch_size, *shp, -1), W2p.reshape(batch_size, -1, *e)

def _batched_left2right_QR(Ws, L, qrnormalize=False)->tuple[tc.Tensor,tc.Tensor]:
    W1 = Ws[0]
    batch_size = W1.shape[0]
    As, lognm = [None] * L, tc.zeros(batch_size, dtype=tc.float64, device=Ws[0].device)
    for i in range(L-1):
        As[i], W1 = _batched_left2right_QR_step(W1, Ws[i+1])
        if qrnormalize:
            nm = tc.norm(W1.reshape(batch_size,-1), dim=1)
            W1 = W1 / nm.reshape(-1,*([1]*(W1.ndim-1)))
            lognm = tc.log(nm.reshape(-1)) + lognm
    As[-1] = W1
    if not qrnormalize:
        nm = tc.norm(As[-1].reshape(batch_size,-1), dim=1)
        As[-1] = As[-1] / nm.reshape(-1,*([1]*(As[-1].ndim-1)))
        lognm = tc.log(nm.reshape(-1))
    return As, lognm


def _batched_right2left_SVD(As, L, trunc_para=(None,None,None)):
    """
    .. code-block:: text
    
        .                       |                        |
                               (b)                      (b)
                                |          SVD           |
         --(a)--▷--(d)--◇--(e)--⨞--(c)--  <----   --(a)--⬜--(c)--
                U       S       B                        W
    """
    cutdim0, svd_min, trunc_cut = trunc_para
    Ss, Bs = [None] * (L + 1), [None] * L
    trunc_err_sum = TruncationError(0.0, 1.0)
    for i in range(L - 1, 0, -1):
        batch_size, a, *shp = As[i].shape
        U, S, B = tc.linalg.svd(As[i].reshape(batch_size, a, -1), full_matrices=False)
        
        cutdim = S.shape[1]
        if svd_min is not None:
            cutdim1 = tc.max(tc.sum(S > svd_min, dim=1))
            cutdim = min(cutdim, cutdim1)
        if trunc_cut is not None:
            normS = S / tc.norm(S, dim=1, keepdim=True)
            cutdim2 = tc.max(tc.sum(tc.flip((tc.cumsum(tc.flip(normS**2, [1]), 1)), [1]) > trunc_cut**2, 1))    
            cutdim = min(cutdim, cutdim2)
        if cutdim0 is not None:
            cutdim = min(cutdim, cutdim0)
        
        eps = tc.max(tc.sum(S[:, cutdim:]**2, dim=1))
        ov = 1. - 2. * eps
        trunc_err_sum += TruncationError(eps, ov)
        
        U = U[:,:,:cutdim]
        S = S[:,:cutdim]
        
        Ss[i] = S
        Bs[i] = B[:,:cutdim,:].reshape(batch_size, cutdim, *shp)
        U = U * S.unsqueeze(1)
        
        batch_size, *shp, a = As[i-1].shape
        As[i - 1] = tc.bmm(As[i - 1].reshape(batch_size, -1, a), U).reshape(batch_size, *shp, -1)
        
    Bs[0] = As[0]
    return Bs, Ss, trunc_err_sum



def batched_canonicalize(
    Ws: list[tc.Tensor], trunc_para=(None,None,None), qrnormalize=False
):
    """
    将任意的 MPS/MPO Ws 变为标准正交的 MPS/MPO (Bs, Ss)
    """
    assert Ws[0].shape[1] == Ws[-1].shape[-1] == 1, "正则形式只对开边界mps有定义！"
    L = len(Ws)
    As, lognm = _batched_left2right_QR(Ws, L, qrnormalize=qrnormalize)
    Bs, Ss, trunc_err = _batched_right2left_SVD(As, L, trunc_para=trunc_para)
    batch_size = Ws[0].shape[0]
    Ss[0] = Ss[-1] = tc.ones(batch_size, dtype=Ss[1].dtype, device=Ss[1].device)
    return Bs, Ss, lognm, trunc_err


def dm_apply_mpo_on_mps(mpo, mps, trunc_para=(None,None,None), normalize=False, updateS=True):
    lognm = 0.0
    n = len(mpo)
    Ss = [None] * (n + 1)

    if n == 1:
        return mpo @ mps

    Lenvs = []
    dtype, device = mpo[0].dtype, mpo[0].device
    Lenv = tc.tensor(1., dtype=dtype, device=device).reshape(1,1,1,1)

    for j in range(n - 1):
        Lenv = _dm_left2right_mps(Lenv, mpo[j], mps[j])
        print(Lenv.shape)
        Lenv = Lenv/tc.norm(Lenv)
        Lenvs.append(Lenv)

    V = tc.tensor(1., dtype=dtype, device=device).reshape(1,1)
    R = tc.tensor(1., dtype=dtype, device=device).reshape(1,1,1)

    R = _dm_get_R_mps(mpo[n-1], mps[n-1], R, V)
    nm = tc.norm(V)
    linkdim = 1
    
    trunc_err_sum = TruncationError(0.0, 1.0)
    for j in range(n - 1, 0, -1):
        # 拿到密度矩阵
        rho = _dm_get_rho(Lenvs[j-1], R)
        
        # 本征分解
        prod_dim = mps[j].shape[0] * mpo[j].shape[0]
        chi_max = trunc_para[0]
        iDc = min(chi_max, prod_dim) if chi_max is not None else prod_dim
        
        S, V = tc.linalg.eigh(rho)
        print(rho.shape)
        S, V = S.flip(0), V.flip(1)
        tc.clamp_(S, min=0)
        tc.sqrt_(S)
        good, trunc_err = truncate(S, iDc, trunc_para[1], trunc_para[2])
        S = S[good]
        V = V[:, good]
        trunc_err_sum += trunc_err
        
        # 替换
        linkdim2 = len(S)
        mps[j] = V.T.reshape(linkdim2, *mps[j].shape[1:-1], linkdim)
        linkdim = linkdim2

        # 前进一步
        R = _dm_get_R_mps(mpo[j-1], mps[j-1], R, V)

        if normalize:
            nm0 = tc.linalg.norm(R)
            R = R / nm0
            nm *= nm0

        if updateS:
            Ss[j] = S
        
    mps[0] = R.reshape(1, *mps[0].shape[1:-1], linkdim)
    lognm = lognm + tc.log(nm)
    return lognm, trunc_err_sum

def _mele_contract_left_env(H:tc.Tensor, psi1:tc.Tensor, psi2:tc.Tensor, Lenv:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .      ╭-╮     psi1                     ╭-╮       
        --(i)--┤ ├--(a)--◻--(b)--        --(i)--┤ ├--(b)--
               │ │       │                      │ │       
               │ │      (c)                     │ │       
               │ │       │H                     │ │       
        --(j)--┤ ├--(d)--◻--(e)--  --->  --(j)--┤ ├--(e)--
               │ │       │                      │ │       
               │ │      (f)                     │ │       
               │ │       │                      │ │       
        --(k)--┤ ├--(g)--◻--(h)--        --(k)--┤ ├--(h)--
               ╰-╯     psi2                     ╰-╯       
     
    tc.einsum("ijkadg,acb,dcfe,gfh->ijkbeh", Lenv, psi1, H, psi2)
    """
    ijk, a, d, g = Lenv.shape
    a, c, b = psi1.shape
    g, f, h = psi2.shape
    d, c, f, e = H.shape
    
    # (ijk,a,d,g) -> (ijkad,g) @ (g,f,h) -> (g,fh) = (ijkad,fh)
    out = Lenv.reshape(-1,g) @ psi2.reshape(g, -1)
    # (ijkad,fh) -> (ijk,a,df,h) -> (ijk,a,h,df) -> (ijkah,df) @ (d,c,f,e) -> (d,f,c,e) -> (df,ce) = (ijkah,ce)
    out = out.reshape(ijk,a,-1,h).swapaxes(2,3).reshape(-1, d*f) @ H.swapaxes(1,2).reshape(d*f, -1)
    # (ijkah,ce) -> (ijk,a,h,c,e) -> (ijk,e,h,a,c) -> (ijkeh,ac) @ (a,c,b) -> (ac,b) = (ijkeh,b)
    out = out.reshape(ijk,a,h,c,e).permute([0,4,2,1,3]).reshape(-1, a*c) @ psi1.reshape(a*c,-1)
    
    return out.reshape(ijk, e, h, b).permute([0,3,1,2])

def _mele_init_left_env(H:tc.Tensor, psi1:tc.Tensor, psi2:tc.Tensor) -> tc.Tensor:
    """
    .. code-block:: text
        
        .     psi1                     ╭-╮       
        --(i)---◻--(b)--        --(i)--┤ ├--(b)--
                │                      │ │       
               (c)                     │ │       
                │H                     │ │       
        --(j)---◻--(e)--  --->  --(j)--┤ ├--(e)--
                │                      │ │       
               (f)                     │ │       
                │                      │ │       
        --(k)---◻--(h)--        --(k)--┤ ├--(h)--
              psi2                     ╰-╯       
     
    tc.einsum("ijkadg,acb,dcfe,gfh->ijkbeh", Lenv, psi1, H, psi2)
    """
    i, c, b = psi1.shape
    k, f, h = psi2.shape
    j, c, f, e = H.shape
    
    # (k,f,h) -> (k,h,f) -> (kh, f) @ (j,c,f,e) -> (f,c,j,e) -> (f,cje) = (kh,cje)
    out = psi2.swapaxes(1,2).reshape(-1, f) @ H.swapaxes(0,2).reshape(f, -1)
    # (kh,cje) -> (kh,c,je) -> (kh,je,c) -> (khje,c) @ (i,c,b) -> (c,i,b) -> (c,ib) = (khje,ib)
    out = out.reshape(-1,c,j*e).swapaxes(1,2).reshape(-1, c) @ psi1.swapaxes(0,1).reshape(c,-1)
    
    return out.reshape(k,h,j,e,i,b).permute([4,2,0,5,3,1]).reshape(-1,b,e,h)