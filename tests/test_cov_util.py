import numpy as np
from numpy.testing import assert_allclose
import pytest
import torch

from dca.dca import init_coef
from dca.data_util import form_lag_matrix
from dca.synth_data import gen_lorenz_data
from dca.cov_util import (calc_cross_cov_mats_from_cov,
                          calc_cov_from_cross_cov_mats,
                          calc_cross_cov_mats_from_data,
                          calc_pi_from_cross_cov_mats,
                          calc_pi_from_cross_cov_mats_block_toeplitz,
                          calc_pi_from_cov,
                          calc_pi_from_data,
                          project_cross_cov_mats,
                          toeplitzify)


@pytest.fixture
def lorenz_dataset():
    rng = np.random.RandomState(20200129)
    T, d = 20, 31
    X = rng.randn(10000, d)
    XL = gen_lorenz_data(10000)
    X[:, :3] += XL
    ccms = calc_cross_cov_mats_from_data(X, T=T)
    ccov = calc_cov_from_cross_cov_mats(ccms)
    return T, d, X, ccms, ccov


def test_cross_cov_round_trip():
    """Tests whether going from a cov to cross_cov_mats and back works.
    """
    T = 5
    N = 3
    ccms = np.random.randn(T, N, N)
    ccms[0] += ccms[0].T
    cov = calc_cov_from_cross_cov_mats(ccms)
    ccms2 = calc_cross_cov_mats_from_cov(cov, T, N)
    assert_allclose(ccms, ccms2)


def test_cross_cov_mats_from_data_chunks_2d():
    """Test whether chunking the lagged matrix gives the same cross covariance matrices
    with 2d inputs."""
    np.random.seed(0)
    cov = np.random.randn(10, 10)
    cov = cov.T.dot(cov) + np.eye(10)
    X = np.random.multivariate_normal(np.zeros(10), cov, size=10000)
    ccms = calc_cross_cov_mats_from_data(X, 3)
    ccms2 = calc_cross_cov_mats_from_data(X, 3, chunks=1)
    assert_allclose(ccms, ccms2, rtol=1e-2)


def test_cross_cov_mats_from_data_chunks_3d():
    """Test whether chunking the lagged matrix gives the same cross covariance matrices
    with 3d inputs."""
    np.random.seed(1)
    cov = np.random.randn(10, 10)
    cov = cov.T.dot(cov) + np.eye(10)
    X = np.random.multivariate_normal(np.zeros(10), cov, size=10000)
    ccms = calc_cross_cov_mats_from_data(X, 3)
    ccms2 = calc_cross_cov_mats_from_data(X, 3, chunks=1)
    assert_allclose(ccms, ccms2, rtol=1e-2)


def test_compare_ccm_cov_pi(lorenz_dataset):
    """Test whether calculating PI from cross cov mats versus a cov mat
    gives equivalent PI."""
    _, _, _, ccms, ccov = lorenz_dataset
    assert_allclose(calc_pi_from_cross_cov_mats(ccms), calc_pi_from_cov(ccov))
    tccms = torch.tensor(ccms)
    tccov = torch.tensor(ccov)
    assert torch.allclose(calc_pi_from_cross_cov_mats(tccms), calc_pi_from_cov(tccov))
    assert_allclose(calc_pi_from_cross_cov_mats(ccms),
                    calc_pi_from_cov(tccov).numpy())


def test_compare_ccm_data_pi(lorenz_dataset):
    """Test whether calculating PI from cross cov mats versus data
    gives equivalent PI."""
    T, d, X, ccms, _ = lorenz_dataset
    assert_allclose(calc_pi_from_cross_cov_mats(ccms), calc_pi_from_data(X, T))


def test_compare_ccm_block_toeplitz_pi(lorenz_dataset):
    """Test whether calculating PI from cross cov mats with and without
    the block-Toeplitz algorithm gives the same value."""
    _, _, _, ccms, _ = lorenz_dataset
    assert_allclose(calc_pi_from_cross_cov_mats(ccms),
                    calc_pi_from_cross_cov_mats_block_toeplitz(ccms))
    tccms = torch.tensor(ccms)
    assert torch.allclose(calc_pi_from_cross_cov_mats(tccms),
                          calc_pi_from_cross_cov_mats_block_toeplitz(tccms))
    assert_allclose(calc_pi_from_cross_cov_mats(ccms),
                    calc_pi_from_cross_cov_mats_block_toeplitz(tccms).numpy())


def test_compare_ccm_block_toeplitz_pi_grads(lorenz_dataset):
    """Test whether calculating the grad of PI from cross cov mats with and without
    the block-Toeplitz algorithm gives the same value."""
    _, d, _, ccms, _ = lorenz_dataset
    assert_allclose(calc_pi_from_cross_cov_mats(ccms),
                    calc_pi_from_cross_cov_mats_block_toeplitz(ccms))
    tccms = torch.tensor(ccms)
    rng = np.random.RandomState(202001291)
    proj = rng.randn(d, 3)
    tproj = torch.tensor(proj, requires_grad=True)
    PI = calc_pi_from_cross_cov_mats(tccms, tproj)
    PI.backward()
    grad = tproj.grad
    tproj = torch.tensor(proj, requires_grad=True)
    PI_BT = calc_pi_from_cross_cov_mats_block_toeplitz(tccms, tproj)
    PI_BT.backward()
    grad_BT = tproj.grad

    assert torch.allclose(grad, grad_BT)


def test_projected_cov_calc(lorenz_dataset):
    """Test the project_cross_cov_mats function by also directly projecting
    the data."""
    rng = np.random.RandomState(20200226)
    T, d, X, _, _ = lorenz_dataset
    X = X[:, :3]
    N = 3
    d = 2
    T = 6
    V = init_coef(N, d, rng, 'random_ortho')
    tV = torch.tensor(V)

    ccms = calc_cross_cov_mats_from_data(X, T)
    tccms = torch.tensor(ccms)
    pccms = project_cross_cov_mats(ccms, V)
    cov = calc_cov_from_cross_cov_mats(pccms)

    XL = form_lag_matrix(X, T)
    big_V = np.zeros((T * N, T * d))
    for ii in range(T):
        big_V[ii * N:(ii + 1) * N, ii * d:(ii + 1) * d] = V
    Xp = XL.dot(big_V)
    cov2 = np.cov(Xp, rowvar=False)
    cov2 = toeplitzify(cov2, T, d)
    assert_allclose(cov, cov2)

    tpccms = project_cross_cov_mats(tccms, tV)
    tcov = calc_cov_from_cross_cov_mats(tpccms)
    assert torch.allclose(tcov, torch.tensor(cov2))
    assert_allclose(tcov.numpy(), cov2)
