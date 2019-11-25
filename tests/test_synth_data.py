import numpy as np

from dca.synth_data import (gen_gp_cov, calc_pi_for_gp, gen_gp_kernel, sample_gp, embed_gp,
                            gen_lorenz_data, oscillators_dynamics_mat, sample_oscillators)


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
