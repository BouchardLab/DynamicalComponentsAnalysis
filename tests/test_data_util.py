import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from dca.data_util import form_lag_matrix


def test_form_lag_matrix_stride_tricks():
    """Test whether stride_tricks gives the same answer as the
    looping version.
    """
    X = np.random.randn(1001, 3)

    T = 1
    X0 = form_lag_matrix(X, T, stride=1, stride_tricks=False)
    assert_array_equal(X, X0)
    X0 = form_lag_matrix(X, T, stride=1, stride_tricks=True)
    assert_array_equal(X, X0)

    T = 5
    X0 = form_lag_matrix(X, T, stride=1, stride_tricks=False)
    X1 = form_lag_matrix(X, T, stride=1, stride_tricks=True)
    assert_array_equal(X0, X1)

    T = 5
    X0 = form_lag_matrix(X, T, stride=7, stride_tricks=False)
    X1 = form_lag_matrix(X, T, stride=7, stride_tricks=True)
    assert_array_equal(X0, X1)


def test_form_lag_matrix_copy():
    """Test whether stride_tricks returns a view if the original matrix
    is C-contiguous and not otherwise. Also effectively tests whether different
    paths are used for stride_tricks for True and False.
    """
    X = np.ones((1001, 3), dtype=float, order='C')

    # X should not be c-contiguous
    X = X.T.copy().T
    T = 5
    X0 = form_lag_matrix(X, T, stride_tricks=False)
    X0 *= 0.
    assert not np.all(np.equal(X, 0.))

    # X should be c-contiguous
    X = np.ones((1001, 3), dtype=float, order='C')
    X0 = form_lag_matrix(X, T, stride_tricks=True, writeable=True)
    X0 *= 0.
    assert_equal(X0, 0.)


def test_form_lag_matrix_errors():
    """Test whether form_lag_matrix raises the correct errors for invalid
    T or stride values.
    """
    X = np.random.randn(11, 3)

    with pytest.raises(ValueError):
        form_lag_matrix(X, 11)
        form_lag_matrix(X, 3, stride=-1)
        form_lag_matrix(X, 3, stride=.5)
