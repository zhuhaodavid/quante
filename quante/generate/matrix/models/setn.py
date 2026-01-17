# -*- coding: utf-8 -*-
# @Author: hzhu
# @Date:   2025-09-19 20:46:34
# @Last Modified by:   hzhu
# @Last Modified time: 2025-10-14 00:25:06

import numpy as np
from tqdm import tqdm
from numba import njit
from ...basis.spin_half.bitsoperation import count_tot_down

def exact_matrix(l, tau, alpha):
    res = np.zeros((2**l, 2**l), dtype=np.float64)
    for i in range(1<<l):
        ci = count_tot_down(i)
        for j in range(1<<l):
            cj = count_tot_down(j)
            x = 2*(ci - cj)
            if x == 0:
                res[i,j] = 1
            else:
                res[i,j] = np.sin(x*alpha*tau) / (x*alpha*tau)
    return res

@njit
def contract(transfer_matrix, lognm, localtsr, tsr):
    chi1 = transfer_matrix.shape[0]
    tsr = tsr.reshape(-1, chi1)
    # (chi2 * LOCAL_DIM, chi1) @ (chi1,) -> (chi2 * LOCAL_DIM,)
    transfer_matrix = (
        np.ascontiguousarray(tsr) @ np.ascontiguousarray(transfer_matrix.real)
        + 1j * (np.ascontiguousarray(tsr) @ np.ascontiguousarray(transfer_matrix.imag))
    )
    # (chi2 * LOCAL_DIM,) @ (LOCAL_DIM,) -> (chi2,)
    transfer_matrix = transfer_matrix.reshape(-1, 4) @ localtsr
    each_nm = np.linalg.norm(transfer_matrix)
    # each_nm = np.max(np.abs(transfer_matrix))
    transfer_matrix[:] /= each_nm
    lognm += np.log(each_nm)
    return lognm, transfer_matrix

def renormalized_tensor(h, tensortranin, tau):
    localtsr = np.array([1, np.exp(2j*h*tau), np.exp(-2j*h*tau), 1])
    transfer_matrix = np.ones((1,), dtype=np.complex128)
    lognm = 0.
    for tsr in tensortranin:
        # ?? do we need group the tsr with the same chi1 and parallelize? 
        lognm, transfer_matrix = contract(transfer_matrix, lognm, localtsr, tsr)
    # (chi,) @ (LOCAL_DIM,) -> (chi * LOCAL_DIM)
    return lognm, np.outer(localtsr, transfer_matrix)

def wv_unify_norm(hlist, tensortrain, tau):
    lognm_list = []
    wv_half_list = []

    # ?? If lognm decreases too much, set the rest to zero. always true?
    close_zero = False
    for h in hlist:
        if close_zero:
            lognm_list.append(-np.inf)
            wv_half_list.append(np.zeros_like(wv_half_list[0]))
        else:
            lognm, wv = renormalized_tensor(h, tensortrain, tau)
            lognm_list.append(lognm)
            wv_half_list.append(wv)
            if lognm_list[0] - lognm > 40:
                close_zero = True

    lognm = np.max(lognm_list) - 2  #? do we need group fft and log-sum-exp？
    for i in range(len(wv_half_list)):
        if np.isfinite(lognm_list[i]):
            wv_half_list[i] *= np.exp(lognm_list[i] - lognm)

    return lognm, wv_half_list

def integrate_func(alpha, tau, m):
    return np.sinc(2 * tau * m * alpha / np.pi)

def fft_integrate(tensortrain, alpha, tau):
    # basic parameters
    l = len(tensortrain) + 1
    M = 2*l
    N = 2*M + 1

    # get sampling points
    hlist = [n*np.pi/(N*tau) for n in range(M+1)]
    lognm, wv = wv_unify_norm(hlist, tensortrain, tau)

    # integrate the renormalized tensor
    wv += [i.conj() for i in wv[-1:0:-1]]
    C = np.fft.fft(wv, axis=0) / N 
    ms = np.arange(-M, M+1)
    c_m = np.array([C[m % N] for m in ms])
    wv_integrated = np.sum(c_m * integrate_func(alpha, tau, ms).reshape(-1,1,1), axis=0)
    wv_integrated = np.real_if_close(wv_integrated)

    # integrate the reduced density matrix
    rdm = [i.conj().reshape(-1,1) @ i.reshape(1,-1) for i in wv]
    C = np.fft.fft(rdm, axis=0) / N 
    ms = np.arange(-M, M+1)
    c_m = np.array([C[m % N] for m in ms])
    rdm_integrated = np.sum(c_m * integrate_func(alpha, tau, ms).reshape(-1,1,1), axis=0)
    rdm_integrated = np.real_if_close(rdm_integrated)
    
    return lognm, wv_integrated, rdm_integrated

def truncate_eig(s, v, trunc):
    s = s[::-1]
    v = v[:, ::-1]
    indx = (s>1e-15) * (s>trunc)
    s = s[indx]
    v = v[:, indx]
    chi = len(s)
    return s, v, chi

def exact_avg(n, alpha=0.5, tau=0.01, trunc=1e-10):
    tensortrain = []
    endtensor = []
    lognm_list = []
    Ss = {}
    chi = 1
    for i in tqdm(range(0, n-1), ascii=True):
        lognm, wv, rdm = fft_integrate(tensortrain, alpha, tau)
        s, v = np.linalg.eigh(rdm)
        s, v, chi = truncate_eig(s, v, trunc)
        Ss[f'n={i}'] = s/s[0]
        tensortrain.append(np.ascontiguousarray(v.T.conj().reshape(chi, 2, 2, -1)))
        if i > 0:
            endtensor.append(wv.reshape(1, 2, 2, -1))
            lognm_list.append(lognm)

    return tensortrain, endtensor, lognm_list, Ss

def todict(tensortrain, endtensor, lognm_list, Ss):
    Vs_out = {}
    for i in range(len(tensortrain)):
        a, *_, b = tensortrain[i].shape
        Vs_out[f"n={i+1}"] = tensortrain[i].reshape(a,-1,b)
    end_tensors = {}
    for i in range(len(endtensor)):
        a, *_, b = endtensor[i].shape
        end_tensors[f"n={i+2}"] = endtensor[i].reshape(a,-1,b)
    lognms = lognm_list[::-1]
    return Vs_out, end_tensors, lognms, Ss

def fromdict(Vs_out, endtensors, lognms):
    tensortrain = [Vs_out[f"n={i+1}"] for i in range(len(Vs_out))]
    endtensor = [endtensors[f"n={i+2}"] for i in range(len(endtensors))]
    lognm_list = lognms[::-1]
    return tensortrain, endtensor, lognm_list

def se_mpo(n, Vs_out, endtensors, lognms):
    import quante.bridge.torch_utils as qtc
    import torch as tc
    tensortrain = []
    for i in range(n-1):
        tsr = Vs_out[f"n={i+1}"]
        a, *_, b = tsr.shape
        tensortrain.append(tsr.reshape(a,2,2,b).swapaxes(0, 3))
    tsr = endtensors[f"n={n}"]
    a, *_, b = tsr.shape
    tensortrain.append(tsr.reshape(a,2,2,b).swapaxes(0, 3))
    lognm = tc.tensor(lognms[-n+1], dtype=tc.float64)
    tt = qtc.MPO(qtc.totc(tensortrain), lognm=lognm)
    return tt


if __name__ == "__main__":
    alpha, tau, trunc = 0.5, 0.01, 1e-10
    N = 10
    n = 8
    tensortrain, endtensor, lognm_list, Ss = exact_avg(
        n=N, alpha=alpha, tau=tau, trunc=trunc
    )
    Vs_out, end_tensors, lognms, Ss = todict(tensortrain, endtensor, lognm_list, Ss)

    tt = se_mpo(n, Vs_out, end_tensors, lognms)
    print(tt)
    print(tt[0])
    vec1 = tt.to_matrix().numpy()

    vec2 = exact_matrix(n, tau, alpha)
    print("norm: ", np.linalg.norm(vec2 - vec1)/np.linalg.norm(vec1))

