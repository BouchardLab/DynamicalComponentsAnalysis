import numpy as np
from scipy.linalg import block_diag
from numpy.testing import (assert_allclose)
import pytest

from dca.methods_comparison import (make_block_diag, block_dot_A, block_dot_B,
                                    block_dot_AB, matrix_inversion_identity,
                                    ForecastableComponentsAnalysis as FCA,
                                    GaussianProcessFactorAnalysis as GPFA,
                                    SlowFeatureAnalysis as SFA,
                                    JPCA)


@pytest.fixture
def noise_dataset():
    X = np.random.randn(111, 7)
    return X


def test_FCA(noise_dataset):
    """Test that a FCA model can be fit with no errors.
    """
    X = noise_dataset
    model = FCA(d=3, T=10)
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)


def test_GPFA(noise_dataset):
    """Test that a GPFA model can be fit with no errors.
    """
    X = noise_dataset
    model = GPFA(n_factors=3, tol=1e-4)  # lower tol to make test faster
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)


def test_SFA(noise_dataset):
    """Test that a SFA model can be fit with no errors.
    """
    X = noise_dataset
    model = SFA(n_components=3)
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)

def test_JPCA():
    """ Test that a JPCA model can be fit with no errors.
    """
    X = np.random.rand(60, 100)
    k = 6
    jpca = JPCA(k)
    X_proj = jpca.fit_transform(X)
    assert X_proj.shape[1] == k


def test_JPCA_multiple_conditions():
    """ Test that a JPCA model can be fit on an array with
    multiple conditions."""
    X = []
    k = 6
    # 10 conditions
    for i in range(10):
        X.append(np.random.rand(60, 100))
    jpca = JPCA(k)
    X_proj = jpca.fit_transform(np.array(X))
    assert X_proj.shape[1] == k


def test_JPCA_skew_symmetric():
    """ Test that matrix fit by JPCA is skew symmetric.
    """
    num_rows = 60
    num_cols = 10
    dX = np.random.rand(num_rows, num_cols)
    X_prestate = np.random.rand(num_rows, num_cols)
    jpca = JPCA(num_cols)
    M_skew = jpca._fit_skew(X_prestate, dX)
    assert np.isclose(M_skew.T, -M_skew).all()


def test_make_block_diag():
    T, n, d = 5, 3, 2
    M = np.random.randn(d, n)
    A = block_diag(*[M] * T)
    Ap = make_block_diag(M, T)
    assert_allclose(A, Ap)


def test_block_dot_A():
    T, n, d = 5, 3, 2
    M = np.random.randn(d, n)
    A = make_block_diag(M, T)
    B = np.random.randn(T * n, T * n)
    assert_allclose(A.dot(B), block_dot_A(M, B, T))


def test_block_dot_B():
    T, n, d = 5, 3, 2
    M = np.random.randn(n, d)
    B = make_block_diag(M, T)
    A = np.random.randn(T * n, T * n)
    assert_allclose(A.dot(B), block_dot_B(A, M, T))


def test_block_dot_AB():
    T, n, d = 5, 3, 2
    MA = np.random.randn(d, n)
    A = make_block_diag(MA, T)
    MB = np.random.randn(n, n)
    B = make_block_diag(MB, T)
    assert_allclose(A.dot(B), block_dot_AB(MA, MB, T))


def test_matrix_inversion_identity():
    T, n, d = 5, 3, 2
    X = np.random.randn(10 * n, n)
    R = np.cov(X, rowvar=False)
    R_inv = np.linalg.inv(R)
    X = np.random.randn(10 * T * d, T * d)
    K = np.cov(X, rowvar=False)
    C = np.random.randn(n, d)
    r0 = np.linalg.inv(make_block_diag(R, T) +
                       make_block_diag(C, T).dot(K).dot(make_block_diag(C.T, T)))
    r1 = matrix_inversion_identity(R_inv, K, C, T)
    assert_allclose(r0, r1)
