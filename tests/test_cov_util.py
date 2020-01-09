import numpy as np

from numpy.testing import (assert_allclose)

from dca.cov_util import (calc_cross_cov_mats_from_cov,
                          calc_cov_from_cross_cov_mats,
                          calc_cross_cov_mats_from_data)


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
    X = np.random.randn(10000, 10)
    ccms = calc_cross_cov_mats_from_data(X, 5)
    ccms2 = calc_cross_cov_mats_from_data(X, 5, chunks=10)
    assert_allclose(ccms, ccms2, atol=1e-3)


def test_cross_cov_mats_from_data_chunks_3d():
    """Test whether chunking the lagged matrix gives the same cross covariance matrices
    with 3d inputs."""
    np.random.seed(1)
    X = np.random.randn(3, 3000, 10)
    ccms = calc_cross_cov_mats_from_data(X, 5)
    ccms2 = calc_cross_cov_mats_from_data(X, 5, chunks=10)
    assert_allclose(ccms, ccms2, atol=1e-3)
