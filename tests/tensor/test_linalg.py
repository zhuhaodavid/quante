# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-05-28 15:40:20
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-31 15:04:40
import pytest
import numpy as np
import quante as qt
from quante.basicfun import println


def test_generate_delta_tensor() -> None:
    d = 4
    ## uniform dimension case
    shape = (d, d, d)
    delta_T = qt.tensor.generate_delta_tensor(shape)
    A = np.random.rand(d)+1j*np.random.rand(d)
    B = np.random.rand(d)+1j*np.random.rand(d)
    C = np.random.rand(d)+1j*np.random.rand(d)
    result = np.einsum("a,b,c,abc->", A, B, C, delta_T)
    result_ = np.einsum("a,a,a->", A, B, C)
    ## random dimension case
    shape = (3, 4, 5)
    delta_T = qt.tensor.generate_delta_tensor(shape)
    A = np.random.rand(3)+1j*np.random.rand(3)
    B = np.random.rand(4)+1j*np.random.rand(4)
    C = np.random.rand(5)+1j*np.random.rand(5)
    result = np.einsum("a,b,c,abc->", A, B, C, delta_T)
    result_ = sum(A[i]*B[i]*C[i] for i in range(3))
    assert np.allclose(result, result_)
    

def test_invert_transpose_order() -> None:
    ndim = 10
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    order = list(np.random.permutation(ndim))
    B = A.transpose(order)
    ## main
    re_order = qt.tensor.invert_transpose_index(order)
    A_ = B.transpose(re_order)
    assert np.allclose(A, A_)
    
    
def test_tensor2matrix() -> None:
    ### odd dimensional tensor
    ndim = 5
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ## main
    A_mat, l_shape, r_shape = qt.tensor.tensor2matrix(A)
    assert np.allclose(A, A_mat.reshape(*(l_shape+r_shape)))
    ### even dimensional tensor
    ndim = 6
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ## main
    A_mat, l_shape, r_shape = qt.tensor.tensor2matrix(A)
    assert np.allclose(A, A_mat.reshape(*(l_shape+r_shape)))


def test_tensor2matrix() -> None:
    ndim = 5
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ## default case
    A_mat, l_shape, r_shape = qt.tensor.tensor2matrix(A, lr_index=[[0,1], [2,3,4]])
    A_mat_, l_shape, r_shape = qt.tensor.tensor2matrix(A)
    assert np.allclose(A_mat, A_mat_)
    ## main
    A_mat, l_shape, r_shape = qt.tensor.tensor2matrix(A, lr_index=[[4,2], [3,0,1]])
    A_ = A_mat.reshape(*(l_shape+r_shape))
    re_order = qt.tensor.invert_transpose_index([4, 2, 3, 0, 1])
    A_ = A_.transpose(re_order)
    assert np.allclose(A, A_)
    
    
def test_qr() -> None:
    ndim = 4
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ## check default index
    q, r = qt.tensor.qr(A)
    q_, r_ = qt.tensor.qr(A, lr_index=[[0, 1], [2, 3]])
    assert np.allclose(q, q_) and np.allclose(r, r_)
    ## main
    q, r = qt.tensor.qr(A, lr_index=[[0, 2], [3, 1]])
    q_mat = q.reshape(-1, q.shape[-1])
    r_mat = r.reshape(r.shape[0], -1)
    re_order = qt.tensor.invert_transpose_index([0, 2, 3, 1])
    A_ = (q_mat @ r_mat).reshape(q.shape[0], q.shape[1], r.shape[1], r.shape[2])
    A_ = A_.transpose(re_order).reshape(*shape)
    assert np.allclose(A, A_)


def test_rq() -> None:
    ndim = 4
    shape = np.random.randint(low=1, high=5, size=ndim)
    A = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    ## check default index
    r, q = qt.tensor.rq(A)
    r_, q_ = qt.tensor.rq(A, lr_index=[[0, 1], [2, 3]])
    assert np.allclose(r, r_) and np.allclose(q, q_)
    ## main
    r, q = qt.tensor.rq(A, lr_index=[[1, 3], [2, 0]])
    r_mat = r.reshape(-1, r.shape[-1])
    q_mat = q.reshape(q.shape[0], -1)
    re_order = qt.tensor.invert_transpose_index([1, 3, 2, 0])
    B_ = (r_mat @ q_mat).reshape(r.shape[0], r.shape[1], q.shape[1], q.shape[2])
    B_ = B_.transpose(re_order).reshape(*shape)
    assert np.allclose(A, B_)
    
    
def test_left2right_QR_step_CaseMPS() -> None:
    # left-dim < right-dim
    A = np.random.rand(2, 2, 5) + 1j * np.random.rand(2, 2, 5)
    B = np.random.rand(5, 2, 2) + 1j * np.random.rand(5, 2, 2)
    A_, B_ = qt.tensor.left2right_QR_step(A, B, lr_index1=[[0, 1], [2]], lr_index2=[[0], [1, 2]])
    AB = np.einsum("abc,cde->abde", A, B)
    AB_ = np.einsum("abc,cde->abde", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.left2right_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    # left-dim >= right-dim
    A = np.random.rand(2, 2, 3) + 1j * np.random.rand(2, 2, 3)
    B = np.random.rand(3, 2, 2) + 1j * np.random.rand(3, 2, 2)
    A_, B_ = qt.tensor.left2right_QR_step(A, B, lr_index1=[[0, 1], [2]], lr_index2=[[0], [1, 2]])
    AB = np.einsum("abc,cde->abde", A, B)
    AB_ = np.einsum("abc,cde->abde", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.left2right_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    
    
def test_left2right_QR_step_CaseMPO() -> None:
    # left-dim < right-dim
    A = np.random.rand(1, 2, 2, 5) + 1j * np.random.rand(1, 2, 2, 5)
    B = np.random.rand(5, 2, 2, 1) + 1j * np.random.rand(5, 2, 2, 1)
    A_, B_ = qt.tensor.left2right_QR_step(A, B, lr_index1=[[0, 1, 2], [3]], lr_index2=[[0], [1, 2, 3]])
    AB = np.einsum("abcd,defg->abcefg", A, B)
    AB_ = np.einsum("abcd,defg->abcefg", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.left2right_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    # left-dim >= right-dim
    A = np.random.rand(5, 2, 2, 5) + 1j * np.random.rand(5, 2, 2, 5)
    B = np.random.rand(5, 2, 2, 5) + 1j * np.random.rand(5, 2, 2, 5)
    A_, B_ = qt.tensor.left2right_QR_step(A, B, lr_index1=[[0, 1, 2], [3]], lr_index2=[[0], [1, 2, 3]])
    AB = np.einsum("abcd,defg->abcefg", A, B)
    AB_ = np.einsum("abcd,defg->abcefg", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.left2right_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    

def test_right2left_QR_step_CaseMPS() -> None:
    # left-dim > right-dim 
    A = np.random.rand(2, 2, 5) + 1j * np.random.rand(2, 2, 5)
    B = np.random.rand(5, 2, 2) + 1j * np.random.rand(5, 2, 2)
    A_, B_ = qt.tensor.right2left_QR_step(A, B, lr_index1=[[0, 1], [2]], lr_index2=[[0], [1, 2]])
    AB = np.einsum("abc,cde->abde", A, B)
    AB_ = np.einsum("abc,cde->abde", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.right2left_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    # left-dim <= right-dim
    A = np.random.rand(2, 2, 3) + 1j * np.random.rand(2, 2, 3)
    B = np.random.rand(3, 2, 2) + 1j * np.random.rand(3, 2, 2)
    A_, B_ = qt.tensor.right2left_QR_step(A, B, lr_index1=[[0, 1], [2]], lr_index2=[[0], [1, 2]])
    AB = np.einsum("abc,cde->abde", A, B)
    AB_ = np.einsum("abc,cde->abde", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.right2left_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    
    
def test_right2left_QR_step_CaseMPO() -> None:
    # left-dim > right-dim 
    A = np.random.rand(5, 2, 2, 5) + 1j * np.random.rand(5, 2, 2, 5)
    B = np.random.rand(5, 2, 2, 1) + 1j * np.random.rand(5, 2, 2, 1)
    A_, B_ = qt.tensor.right2left_QR_step(A, B, lr_index1=[[0, 1, 2], [3]], lr_index2=[[0], [1, 2, 3]])
    AB = np.einsum("abcd,defg->abcefg", A, B)
    AB_ = np.einsum("abcd,defg->abcefg", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.right2left_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    # left-dim <= right-dim
    A = np.random.rand(1, 2, 2, 3) + 1j * np.random.rand(1, 2, 2, 3)
    B = np.random.rand(3, 2, 2, 3) + 1j * np.random.rand(3, 2, 2, 3)
    A_, B_ = qt.tensor.right2left_QR_step(A, B, lr_index1=[[0, 1, 2], [3]], lr_index2=[[0], [1, 2, 3]])
    AB = np.einsum("abcd,defg->abcefg", A, B)
    AB_ = np.einsum("abcd,defg->abcefg", A_, B_)
    assert np.allclose(AB, AB_)
    ## check default case
    A_default, B_default = qt.tensor.right2left_QR_step(A, B)
    assert np.allclose(A_, A_default) and np.allclose(B_, B_default)
    
    
def test_apply_2b_gate_mps() -> None:
    gate = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    A = np.random.rand(2, 2, 3) + 1j * np.random.rand(2, 2, 3)
    B = np.random.rand(3, 2, 2) + 1j * np.random.rand(3, 2, 2)
    ## main
    theta = qt.tensor.apply_2b_gate_mps(A, B, gate)
    theta_ = np.einsum("abc,cde,xybd->axye", A, B, gate.reshape(2, 2, 2, 2))
    assert np.allclose(theta, theta_)
    
    
def test_update_two_site_CaseMPS() -> None:
    ## right
    d, chi = 2, 4
    theta = np.random.randn(chi, d, d, chi)
    theta /= np.linalg.norm(theta)
    W1, W2, _ = qt.tensor.update_two_site(theta, direction="right", lr_index=[[0,1],[2,3]])
    # check default
    W1_, W2_, _ = qt.tensor.update_two_site(theta, direction="right")
    assert np.allclose(W1, W1_) and np.allclose(W2, W2_)
    theta_ = np.einsum("abc,cde->abde", W1, W2)
    assert np.allclose(theta, theta_)
    ## left
    W1, W2, _ = qt.tensor.update_two_site(theta, direction="left")
    theta_ = np.einsum("abc,cde->abde", W1, W2)
    assert np.allclose(theta, theta_)
    ## mixed
    W1, s, W2, _ = qt.tensor.update_two_site(theta, direction="mixed")
    theta_ = np.einsum("abc,c,cde->abde", W1, s, W2)
    assert np.allclose(theta, theta_)
    
    
def test_inner_initialize() -> None:
    ## Case MPS
    d, D = 2, 5
    W1 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    W2 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    Lenv = qt.tensor.inner_initialize(W1, W2)
    Lenv_ = np.einsum("abc,dbf->adcf", W1.conj(), W2)
    assert np.allclose(Lenv, Lenv_)
    ## Case MPO
    d, D = 2, 5
    W1 = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)
    W2 = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)
    Lenv = qt.tensor.inner_initialize(W1, W2)
    Lenv_ = np.einsum("abcd,ebcf->aedf", W1.conj(), W2)
    assert np.allclose(Lenv, Lenv_)
    

def test_inner_step() -> None:
    d, D = 2, 5
    W1 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    W2 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    Lenv_init = qt.tensor.inner_initialize(W1, W2)
    ## Case MPS
    W1 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    W2 = np.random.rand(D, d, D) + 1j * np.random.rand(D, d, D)
    Lenv = qt.tensor.inner_step(Lenv_init, W1, W2)
    Lenv_ = np.einsum("abcd,ceg,def->abgf", Lenv_init, W1.conj(), W2)
    assert np.allclose(Lenv, Lenv_)
    ## Case MPO
    W1 = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)
    W2 = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)
    Lenv = qt.tensor.inner_step(Lenv_init, W1, W2)
    Lenv_ = np.einsum("abcd,cefh,defg->abhg", Lenv_init, W1.conj(), W2)
    assert np.allclose(Lenv, Lenv_)
    
    
def test_add_mid_CaseMPS() -> None:
    d, D1, D2, D3, D4 = 2, 3, 6, 7, 5
    W1 = np.random.rand(D1, d, D2) + 1j * np.random.rand(D1, d, D2)
    W2 = np.random.rand(D3, d, D4) + 1j * np.random.rand(D3, d, D4)
    W1W2 = qt.tensor.add_mid(W1, W2)
    assert np.allclose(W1W2[:D1, ..., :D2], W1)
    assert np.allclose(W1W2[D1:, ..., D2:], W2)
    
    
def test_add_mid_CaseMPO() -> None:
    d, D1, D2, D3, D4 = 2, 3, 6, 7, 5
    W1 = np.random.rand(D1, d, d, D2) + 1j * np.random.rand(D1, d, d, D2)
    W2 = np.random.rand(D3, d, d, D4) + 1j * np.random.rand(D3, d, d, D4)
    W1W2 = qt.tensor.add_mid(W1, W2)
    assert np.allclose(W1W2[:D1, ..., :D2], W1)
    assert np.allclose(W1W2[D1:, ..., D2:], W2)


def test_add_mid_CaseTT() -> None:
    d, D1, D2, D3, D4 = 2, 3, 6, 7, 5
    W1 = np.random.rand(D1, d, 9, D2) + 1j * np.random.rand(D1, d, 9, D2)
    W2 = np.random.rand(D3, d, 9, D4) + 1j * np.random.rand(D3, d, 9, D4)
    W1W2 = qt.tensor.add_mid(W1, W2)
    assert np.allclose(W1W2[:D1, ..., :D2], W1)
    assert np.allclose(W1W2[D1:, ..., D2:], W2)
    
    
def test_add_left_CaseTT() -> None:
    d, D2, D4 = 2, 3, 6
    W1 = np.random.rand(1, d, 9, D2) + 1j * np.random.rand(1, d, 9, D2)
    W2 = np.random.rand(1, d, 9, D4) + 1j * np.random.rand(1, d, 9, D4)
    W1W2 = qt.tensor.add_left(W1, W2)
    assert np.allclose(W1W2[..., :D2], W1)
    assert np.allclose(W1W2[..., D2:], W2)
    

def test_add_right_CaseTT() -> None:
    d, D1, D3 = 2, 3, 6
    W1 = np.random.rand(D1, d, 9, 1) + 1j * np.random.rand(D1, d, 9, 1)
    W2 = np.random.rand(D3, d, 9, 1) + 1j * np.random.rand(D3, d, 9, 1)
    W1W2 = qt.tensor.add_right(W1, W2)
    assert np.allclose(W1W2[:D1, ...], W1)
    assert np.allclose(W1W2[D1:, ...], W2)
    