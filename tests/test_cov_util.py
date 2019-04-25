import numpy as np

from numpy.testing import (assert_allclose)

from cca.cov_util import (calc_cross_cov_mats_from_cov,
                          calc_cov_from_cross_cov_mats)


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
