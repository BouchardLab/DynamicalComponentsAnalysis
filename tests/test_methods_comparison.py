import numpy as np
from scipy.linalg import block_diag
from numpy.testing import (assert_allclose)

from cca.methods_comparison import (make_block_diag, block_dot_A, block_dot_B,
                                    block_dot_AB, matrix_inversion_identity)

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
    assert_allclose(A.dot(B), block_dot_A(A, B, T))

def test_block_dot_B():
    T, n, d = 5, 3, 2
    M = np.random.randn(n, d)
    B = make_block_diag(M, T)
    A = np.random.randn(T * n, T * n)
    assert_allclose(A.dot(B), block_dot_B(A, B, T))

def test_block_dot_AB():
    T, n, d = 5, 3, 2
    M = np.random.randn(d, n)
    A = make_block_diag(M, T)
    M = np.random.randn(n, n)
    B = make_block_diag(M, T)
    assert_allclose(A.dot(B), block_dot_AB(A, B, T))

def test_matrix_inversion_identity():
    T, n, d = 5, 3, 2
    X = np.random.randn(10 * n, n)
    R =  np.cov(X, rowvar=False)
    R_inv = make_block_diag(np.linalg.inv(R), T)
    R = make_block_diag(R, T)
    X = np.random.randn(10 * T * d, T * d)
    K =  np.cov(X, rowvar=False)
    C =  np.random.randn(n, d)
    C = make_block_diag(C, T)
    r0 = np.linalg.inv(R + C.dot(K).dot(C.T))
    R_inv = np.linalg.inv(R)
    r1 = matrix_inversion_identity(R_inv, K, C, T)
    assert_allclose(r0, r1)
