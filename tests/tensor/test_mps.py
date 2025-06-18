# -*- coding: utf-8 -*-
# @Author: dzwang
# @Date:   2025-05-29 10:53:36
# @Last Modified by:   dzwang
# @Last Modified time: 2025-05-30 15:27:27
import pytest
import numpy as np
import quante as qt
from quante.basicfun import println


def test_initialization() -> None:
    d, D, N = 2, 3, 5
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    mps = qt.tensor.networks.MPS(Ws, llim=0, rlim=N-1)
    ## main
    assert isinstance(mps, qt.tensor.networks.MPS)
    assert mps.N == len(Ws)
    assert mps.llim == 0 and mps.rlim == N - 1
    assert mps.chi == [D, D, D, D]


def test_to_vector() -> None:
    d, D, N = 2, 3, 5
    Ws = [np.random.rand(1, d, D)] + [np.random.rand(D, d, D) for _ in range(N-2)] + [np.random.rand(D, d, 1)]
    mps = qt.tensor.networks.MPS(Ws, llim=0, rlim=N-1)
    vec_mps = mps.to_vector()
    vec = np.einsum("abc,cde,efg,ghi,ijk->abdfhjk", 
                   mps.Ws[0], mps.Ws[1], mps.Ws[2], mps.Ws[3], mps.Ws[4]).reshape(-1)
    assert np.allclose(vec_mps, vec)
    

def test_generate_product_state() -> None:
    N = 4
    config = np.random.randint(low=0, high=2, size=N)
    config = "".join(str(site) for site in config)
    mps = qt.tensor.networks.MPS.generate_product_state(config, dtype=np.float64)
    assert mps.chi == [1] * (N-1)
    state = qt.generate.state.product_state(config, dtype=np.float64)
    assert np.allclose(mps.to_vector(), state.squeeze())
    llim, rlim = mps.get_llim_rlim()
    assert llim == 0 and rlim == 0
    

def test_generate_product_state_CaseNeel_SingleUp_FullUp():
    # Neel state
    N = 7
    config = "01"*(N//2) if N%2==0 else "01"*(N//2)+"0"
    mps = qt.tensor.networks.MPS.generate_product_state(config, dtype=np.float64)
    neel_state = qt.generate.state.neel(N, down_first=True, dtype=np.float64)
    assert np.allclose(mps.to_vector(), neel_state.squeeze())
    # Neel state
    N = 6
    config = "10"*(N//2) if N%2==0 else "10"*(N//2)+"1"
    mps = qt.tensor.networks.MPS.generate_product_state(config, dtype=np.float64)
    neel_state = qt.generate.state.neel(N, down_first=False, dtype=np.float64)
    assert np.allclose(mps.to_vector(), neel_state.squeeze())
    # Single up state
    N = 10
    config = list("0"*N)
    up_site = np.random.randint(low=0, high=N)
    config[up_site] = "1"
    mps = qt.tensor.networks.MPS.generate_product_state(config, dtype=np.float64)
    single_up_state = qt.generate.state.product_state(config, dtype=np.float64)
    assert np.allclose(mps.to_vector(), single_up_state.squeeze())
    # Full up state
    N = 5
    config = "1"*N
    mps = qt.tensor.networks.MPS.generate_product_state(config, dtype=np.float64)
    full_up_state = qt.generate.state.product_state(config, dtype=np.float64)
    assert np.allclose(mps.to_vector(), full_up_state.squeeze())


def test_generate_from_vector() -> None:
    d, N = 2, 7
    vector = np.random.rand(d**N) + 1j * np.random.rand(d**N)
    mps = qt.tensor.networks.MPS.generate_from_vector(vector, d)
    assert mps.N == N
    assert mps.llim==N-1 and mps.rlim==N-1
    vector_mps = mps.to_vector()
    assert np.allclose(vector_mps, vector)


def test_generate_W_state() -> None:
    N = 7
    ## Case single down
    mps = qt.tensor.networks.MPS.generate_W_state(N, dtype=np.float64, type="single_dn")
    assert mps.chi == [2] * (N-1)
    assert mps.llim == 0 and mps.rlim == N - 1
    state_W = qt.generate.state.w(N, dtype=np.float64)
    assert np.allclose(mps.to_vector()/mps.norm, state_W.squeeze())
    ## Case single up
    N = 5
    mps = qt.tensor.networks.MPS.generate_W_state(N, dtype=np.float64, type="single_up")
    assert mps.chi == [2] * (N-1)
    assert mps.llim == 0 and mps.rlim == N - 1
    steta_W  = qt.generate.state.product_state("00001", dtype=np.float64)
    steta_W += qt.generate.state.product_state("00010", dtype=np.float64)
    steta_W += qt.generate.state.product_state("00100", dtype=np.float64)
    steta_W += qt.generate.state.product_state("01000", dtype=np.float64)
    steta_W += qt.generate.state.product_state("10000", dtype=np.float64)
    assert np.allclose(mps.to_vector(), steta_W.squeeze())
    
    