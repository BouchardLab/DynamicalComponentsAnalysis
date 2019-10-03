import numpy as np
from scipy.stats import special_ortho_group as sog
from numpy.testing import (assert_allclose)

from dca.analysis import random_complement


def test_random_complement():
    """Test that random complement vector is orthogonal to the projection."""
    rng = np.random.RandomState(20190710)
    for dim in [5, 9, 11]:
        for pdim in [1, 2, 3]:
            for size in [1, 5, 9]:
                proj = sog.rvs(dim, random_state=rng)[:, :pdim]
                vec = random_complement(proj, size, rng)
                assert_allclose(proj.T.dot(vec), 0, atol=1e-15)
