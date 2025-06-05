# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-05-28 12:45:11
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-30 16:38:00
import pytest
import numpy as np
import quante as qt
from quante.basicfun import println


def test_init() -> None:
    N = 4
    Ws = [np.random.rand(1, 2, 1) for _ in range(N)]
    llim = 0
    rlim = N - 1
    ## main
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=llim, rlim=rlim)
    assert tt.Ws == Ws
    assert tt.llim == llim
    assert tt.rlim == rlim
    assert tt.N == N
    assert tt.chi == [1, 1, 1]
    

def test_set_mixed_canonical_form_CaseMPS() -> None:
    ## midel case
    d, D, N = 2, 2, 4
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    # main
    oc = 2
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## left case
    oc = 0
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## right case
    oc = N - 1
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    

def test_set_mixed_canonical_form_CaseMPO() -> None:
    ## midel case
    d, D, N = 2, 2, 4
    Ws = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    # main
    oc = 2
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## left case
    oc = 0
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## right case
    oc = N - 1
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    

def test_set_mixed_canonical_form_CaseTT() -> None:
    ## midel case
    d, D, N = 2, 2, 4
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, d, D), np.random.rand(D, d, D)] + [np.random.rand(D, d, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    # main
    oc = 2
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## left case
    oc = 0
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    ## right case
    oc = N - 1
    tt.set_mixed_canonical_form(oc)
    assert tt.llim == oc and tt.rlim == oc
    tt.check_mixed_canonical_form()
    
    
def test_get_llim_rlim_CaseMPS() -> None:
    d, D, N = 2, 2, 4
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    llim, rlim = 0, N-1
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim, rlim)
    ## boundary case
    llim, rlim = tt.get_llim_rlim()
    assert llim == 0 and rlim == N-1
    ## mixed form case
    tt.set_mixed_canonical_form(oc=0)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=N-1)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=2)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim


def test_get_llim_rlim_CaseMPO() -> None:
    d, D, N = 2, 5, 4
    Ws = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
    llim, rlim = 0, N-1
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim, rlim)
    ## boundary case
    llim, rlim = tt.get_llim_rlim()
    assert llim == 0 and rlim == N-1
    ## mixed form case
    tt.set_mixed_canonical_form(oc=0)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=N-1)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=2)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim

    
def test_get_llim_rlim_CaseMPO() -> None:
    d, D, N = 3, 5, 4
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, d, D), np.random.rand(D, d, D)] + [np.random.rand(D, d, d, 1)]
    llim, rlim = 0, N-1
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim, rlim)
    ## boundary case
    llim, rlim = tt.get_llim_rlim()
    assert llim == 0 and rlim == N-1
    ## mixed form case
    tt.set_mixed_canonical_form(oc=0)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=N-1)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    tt.set_mixed_canonical_form(oc=2)
    llim, rlim = tt.get_llim_rlim()
    assert llim == tt.llim and rlim == tt.rlim
    
    
def test_to_tensor() -> None:
    d, D, N = 2, 5, 4
    ## MPS case
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    tt_mat = tt.to_tensor()
    Ws_mat = np.einsum("abc,cde,efg,ghi->abdfhi", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(-1)
    assert np.allclose(tt_mat, Ws_mat)
    ## MPO case
    Ws = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    tt_mat = tt.to_tensor()
    Ws_mat = np.einsum("abcd,defg,ghij,jklm->abehkcfilm", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(d**N, -1)
    assert np.allclose(tt_mat, Ws_mat)
    ## TT case
    Ws = [np.random.rand(D, d, D)] + [np.random.rand(D, d, d, D), np.random.rand(D, d, D)] + [np.random.rand(D, d, d, D)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    tt_mat = tt.to_tensor()
    Ws_mat = np.einsum("abc,cdef,fgh,hijk->abdgiejk", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(D, d**N, -1, D)
    assert np.allclose(tt_mat, Ws_mat)
    

def test_tt_decompose_CaseMPS() -> None:
    d, N = 2, 10
    vector = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(vector, d)
    llim, rlim = tt.get_llim_rlim()
    tt.check_mixed_canonical_form()
    assert llim==tt.llim==N-1 and rlim==tt.rlim==N-1
    vector_tt = tt.to_tensor()
    assert np.allclose(vector_tt, vector)
    
    
def test_inner_CaseMPS() -> None:
    d, N = 2, 11
    A = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    B = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    A_tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(A, d)
    B_tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(B, d)
    overlap = qt.tensor.tensortrain.TensorTrain.inner(A_tt.Ws, B_tt.Ws)
    overlap_tt = np.einsum("a,a->", A.conj(), B)
    assert np.isclose(overlap, overlap_tt)
    
    
def test_norm_CaseMPS() -> None:
    ## vector case
    # oc at the rightmost site
    d, N = 2, 11
    A = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    A_tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(A, d)
    assert A_tt.llim == N-1 and A_tt.rlim == N-1
    norm_tt = A_tt.norm
    norm = np.linalg.norm(A)
    assert np.isclose(norm_tt, norm)
    # oc at random site
    oc = np.random.randint(low=0, high=N)
    A_tt.set_mixed_canonical_form(oc)
    assert A_tt.llim == oc and A_tt.rlim == oc
    norm_tt = A_tt.norm
    assert np.isclose(norm_tt, norm)
    ## random MPS case
    d, D, N = 2, 5, 4
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
    Ws_mat = np.einsum("abc,cde,efg,ghi->abdfhi", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(-1)
    tt_norm = tt.norm
    Ws_norm = np.linalg.norm(Ws_mat)
    assert np.isclose(Ws_norm, tt_norm)  
    
    
def test_add_CaseMPS() -> None:
    c1 = np.random.rand() + 1j * np.random.rand()
    c2 = np.random.rand() + 1j * np.random.rand()
    ## vector case
    d, N = 2, 7
    A = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    B = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    A_tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(A, d)
    B_tt = qt.tensor.tensortrain.TensorTrain.tt_decompose(B, d)
    C_tt = A_tt.add(B_tt, c1=c1, c2=c2)
    assert C_tt.llim == 0 and C_tt.rlim == N-1
    C_vec = C_tt.to_tensor()
    C = c1 * A + c2 * B
    assert np.allclose(C_vec, C)
    ## random MPS case
    d, D, N = 2, 5, 7
    WsA = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    WsB = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    ttA = qt.tensor.tensortrain.TensorTrain(WsA, llim=0, rlim=N-1)
    ttB = qt.tensor.tensortrain.TensorTrain(WsB, llim=0, rlim=N-1)
    C = c1 * ttA.to_tensor() + c2 * ttB.to_tensor()    
    ttC = ttA.add(ttB, c1=c1, c2=c2)
    C_vec = ttC.to_tensor()
    assert np.allclose(C_vec, C)
    
    
# def test_add_CaseMPO() -> None:
#     c1 = np.random.rand() + 1j * np.random.rand()
#     c2 = np.random.rand() + 1j * np.random.rand()
#     ## random MPO case
#     d, D, N = 2, 5, 7
#     WsA = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
#     WsB = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
#     ttA = qt.tensor.tensortrain.TensorTrain(WsA, llim=0, rlim=N-1)
#     ttB = qt.tensor.tensortrain.TensorTrain(WsB, llim=0, rlim=N-1)
#     C = c1 * ttA.to_tensor() + c2 * ttB.to_tensor()    
#     ttC = ttA.add(ttB, c1=c1, c2=c2)
#     C_vec = ttC.to_tensor()
#     assert np.allclose(C_vec, C)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      
# ## MPO case
# Ws = [np.random.rand(1, d, d, D)] + [np.random.rand(D, d, d, D) for _ in range(N-2)] + [np.random.rand(D, d, d, 1)]
# tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
# Ws_mat = np.einsum("abcd,defg,ghij,jklm->abehkcfilm", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(d**N, -1)
# Ws_norm = np.linalg.norm(Ws_mat)
# tt_norm = tt.norm()
# assert np.isclose(Ws_norm, tt_norm)
# ## TT case
# Ws = [np.random.rand(D, d, D)] + [np.random.rand(D, d, d, D), np.random.rand(D, d, D)] + [np.random.rand(D, d, d, D)]
# tt = qt.tensor.tensortrain.TensorTrain(Ws, llim=0, rlim=N-1)
# Ws_mat = np.einsum("abc,cdef,fgh,hijk->abdgiejk", Ws[0], Ws[1], Ws[2], Ws[3]).reshape(D, d**N, -1, D)
# Ws_norm = np.linalg.norm(Ws_mat)
# tt_norm = tt.norm()
# assert np.isclose(Ws_norm, tt_norm)
    
