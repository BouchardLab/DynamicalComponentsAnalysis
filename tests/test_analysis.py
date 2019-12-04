import pytest
import numpy as np
from scipy.stats import special_ortho_group as sog
from numpy.testing import (assert_allclose)

from dca.analysis import (random_complement, run_analysis, run_dim_analysis_dca)


def test_random_complement():
    """Test that random complement vector is orthogonal to the projection."""
    rng = np.random.RandomState(20190710)
    for dim in [5, 9, 11]:
        for pdim in [1, 2, 3]:
            for size in [1, 5, 9]:
                proj = sog.rvs(dim, random_state=rng)[:, :pdim]
                vec = random_complement(proj, size, rng)
                assert_allclose(proj.T.dot(vec), 0, atol=1e-15)


@pytest.fixture
def noise_dataset():
    X = np.random.randn(555, 10)
    y = np.random.randn(555, 2)
    return X, y


def test_run_analysis(noise_dataset):
    """Test that run_analysis runs.
    """
    X, y = noise_dataset

    run_analysis(X, y, [2, 3], [1, 2, 3], [0, 1, 2], 2, 3)


def test_run_dim_analysis_dca(noise_dataset):
    """Test that run_dim_analysis_dca runs.
    """
    X, y = noise_dataset

    run_dim_analysis_dca(X, y, 2, [1, 2, 3], 0, 2, 3, n_null=100)
