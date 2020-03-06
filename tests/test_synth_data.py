import numpy as np

from dca.synth_data import (gen_gp_cov, calc_pi_for_gp, gen_gp_kernel, sample_gp, embed_gp,
                            gen_lorenz_data, oscillators_dynamics_mat, sample_oscillators,
                            oscillators_cross_cov_mats,
                            embedded_lorenz_cross_cov_mats)


def test_gp_kernels():
    """Make sure all GP functions run.
    """
    for ker_type in ['squared_exp', 'exp', 'switch']:
        ker = gen_gp_kernel(ker_type, .3, .5)
        gen_gp_cov(ker, 83, 11)
        calc_pi_for_gp(ker, 10, 11)
        sample_gp(83, 11, ker, 3)
        embed_gp(83, 11, 3, ker, np.eye(11), 10)


def test_lorenz():
    """Make sure all Lorenz functions run.
    """
    gen_lorenz_data(1000)


def test_oscillators():
    """Make sure all oscillation functions run.
    """
    A = oscillators_dynamics_mat()
    sample_oscillators(A, 1000)


def test_oscillators_cross_cov_mats():
    """Test that cross-cov mats can be made for the oscillators.
    """
    A = oscillators_dynamics_mat(N=7)
    ccms = oscillators_cross_cov_mats(A, T=5)
    assert ccms.shape == (5, 14, 14)


def test_embedded_lorenz_cross_crov_mats():
    ccms = embedded_lorenz_cross_cov_mats(11, 7, num_lorenz_samples=1000,
                                          num_subspace_samples=100)
    assert ccms.shape == (7, 11, 11)
