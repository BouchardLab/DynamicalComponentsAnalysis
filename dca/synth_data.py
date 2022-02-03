import numpy as np
import scipy
from scipy.linalg import expm
from numpy.linalg import matrix_power
from scipy.signal import resample

from .cov_util import (calc_cov_from_cross_cov_mats, calc_cross_cov_mats_from_cov,
                       calc_cross_cov_mats_from_data, calc_pi_from_cov)


def gen_gp_cov(kernel, T, N):
    """Generates a T*N-by-T*N covariance matrix for a spatiotemporal Gaussian
    process (2D Gaussian random field) with a provided kernel.

    Parameters
    ----------
    T : int
        Number of time-steps.
    N : int
        Number of spatial steps.
    kernel : function
        Should be of the form kernel = K(t1, t2, x1, x2).
        The kernel can choose to imlement temporal or spatial stationarity,
        however this is not enfored.

    Returns
    -------
    C : np.ndarray, shape (T*N, T*N)
        Covariance matrix for the Gaussian process. Time is the "outer" variable
        and space is the "inner" variable.
    """

    t1, t2, x1, x2 = np.arange(T), np.arange(T), np.arange(N), np.arange(N)
    t1, t2, x1, x2 = np.meshgrid(t1, t2, x1, x2, indexing="ij")
    C = kernel(t1, t2, x1, x2)
    C = C.swapaxes(1, 2).reshape(T * N, T * N)

    return C


def calc_pi_for_gp(kernel, T_pi, N):
    """Calculates the predictive information in a spatiotemporal Gaussian process
    with a given kernel.

    Parameters
    ----------
    T : int
        Length of temporal windows accross which to compute mutual information.
    N : int
        Number of spatial steps in teh Gaussian process.
    kernel : function
        Should be of the form kernel = K(t1, t2, x1, x2).
        The kernel can choose to imlement temporal or spatial stationarity,
        however this is not enfored.

    Returns
    -------
    PI : float
        (Temporal) predictive information in the Gaussian process.
    """

    cov_2_T_pi = gen_gp_cov(kernel, 2 * T_pi, N)
    cov_T_pi = cov_2_T_pi[:T_pi * N, :T_pi * N]
    sgn_T, logdet_T = np.linalg.slogdet(cov_T_pi)
    sgn_2T, logdet_2T = np.linalg.slogdet(cov_2_T_pi)
    PI = logdet_T - 0.5 * logdet_2T
    return PI


def gen_gp_kernel(kernel_type, spatial_scale, temporal_scale,
                  local_noise=0.):
    """Generates a specified type of Kernel for a spatiotemporal Gaussian
    process.

    Parameters
    ----------
    kernel_type : string
        'squared_exp' or 'exp'
    spatial_scale : float
        Spatial autocorrelation scale.
    temporal_scale : float
        Temporal autocorrelation scale.

    Returns
    -------
    K : function
        Kernel of the form K(t1, t2, x1, x2).
    """

    def squared_exp(t1, t2, x1, x2):
        rval = np.exp(-(t1 - t2)**2 / temporal_scale**2 - (x1 - x2)**2 / spatial_scale**2)
        if local_noise > 0.:
            local_mask = np.logical_and(np.equal(t1, t2), np.equal(x1, x2))
            rval += local_noise * local_mask
        return rval

    def exp(t1, t2, x1, x2):
        rval = np.exp(-np.abs(t1 - t2) / temporal_scale - np.abs(x1 - x2) / spatial_scale)
        if local_noise > 0.:
            local_mask = np.logical_and(np.equal(t1, t2), np.equal(x1, x2))
            rval += local_noise * local_mask
        return rval

    def switch(t1, t2, x1, x2):
        mask = abs(t1 - t2) >= temporal_scale
        ex = exp(t1, t2, x1, x2)
        sq = squared_exp(t1, t2, x1, x2)
        rval = mask * ex + (1. - mask) * sq
        if local_noise > 0.:
            local_mask = np.logical_and(np.equal(t1, t2), np.equal(x1, x2))
            rval += local_noise * local_mask
        return rval

    if kernel_type == "squared_exp":
        K = squared_exp
    elif kernel_type == "exp":
        K = exp
    elif kernel_type == "switch":
        K = switch
    return K


def sample_gp(T, N, kernel, num_to_concat=1):
    """Draw a sample from a spatiotemporal Gaussian process.

    Parameters
    ----------
    T : int
        Length in time of sample.
    N : int
        Size in space of sample.
    kernel : function
        Kernel of the form K(t1, t2, x1, x2).
    num_to_concat : int
        Number of samples of lenght T to concatenate before returning the result.

    Returns
    -------
    sample : np.ndarray, size (T*num_to_concat, N)
        Sample from the Gaussian process.
    """

    t1, t2, x1, x2 = np.arange(T), np.arange(T), np.arange(N), np.arange(N)
    t1, t2, x1, x2 = np.meshgrid(t1, t2, x1, x2, indexing="ij")
    C = kernel(t1, t2, x1, x2)
    C = C.swapaxes(1, 2).reshape(T * N, T * N)

    sample = np.concatenate(np.random.multivariate_normal(mean=np.zeros(C.shape[0]),
                                                          cov=C, size=num_to_concat))
    sample = sample.reshape(T * num_to_concat, N)
    return sample


def embed_gp(T, N, d, kernel, noise_cov, T_pi, num_to_concat=1):
    """Embed a d-dimensional Gaussian process into N-dimensional space, then
    add (potentially) spatially structured white noise.

    Parameters
    ----------
    T : int
        Length in time.
    N : int
        Ambient dimension.
    d : int
        Gaussian process dimension.
    kernel : function
        Kernel of the form K(t1, t2, x1, x2).
    noise_cov : np.ndarray, shape (N, N)
        Covariance matrix from which to sampel Gaussian noise to add to each
        time point in an iid fashion.
    num_to_concat : int
        Number of samples of lenght T to concatenate before returning the result.

    Returns
    -------
    X : np.ndarray, size (T*num_to_concat, N)
        Embedding of GP into high-dimensional space, plus noise.
    """

    # Latent dynamics
    Y = sample_gp(T, d, kernel, num_to_concat)

    # Random orthogonal embedding matrix U
    U = scipy.stats.ortho_group.rvs(N)[:, :d]

    # Data matrix X
    X = np.dot(Y, U.T)

    # Corrupt data by spatially structured white noise
    X += np.random.multivariate_normal(mean=np.zeros(N), cov=noise_cov, size=T * num_to_concat)

    # Compute the PI of the high-dimensional, noisy process
    cov_low_d = gen_gp_cov(T=2 * T_pi, N=d, kernel=kernel)
    low_d_cross_cov_mats = calc_cross_cov_mats_from_cov(cov_low_d, 2 * T_pi, d)
    high_d_cross_cov_mats = np.array([np.dot(U, np.dot(C, U.T)) for C in low_d_cross_cov_mats])
    high_d_cross_cov_mats[0] += noise_cov
    cov_high_d = calc_cov_from_cross_cov_mats(high_d_cross_cov_mats)
    full_pi = calc_pi_from_cov(cov_high_d)

    embedding_cross_cov_mats = np.array([np.dot(U.T, np.dot(C, U)) for C in high_d_cross_cov_mats])
    cov_embedding = calc_cov_from_cross_cov_mats(embedding_cross_cov_mats)
    embedding_pi = calc_pi_from_cov(cov_embedding)

    return X, Y, U, full_pi, embedding_pi, high_d_cross_cov_mats


def gen_lorenz_system(T, seed, integration_dt=0.005):
    """
    Period ~ 1 unit of time (total time is T)
    So make sure integration_dt << 1

    Known-to-be-good chaotic parameters
    See sussillo LFADS paper
    """
    rng = np.random.RandomState(seed)
    rho = 28.0
    sigma = 10.0
    beta = 8 / 3.

    def dx_dt(state, t):
        x, y, z = state
        x_dot = sigma * (y - x)
        y_dot = x * (rho - z) - y
        z_dot = x * y - beta * z
        return (x_dot, y_dot, z_dot)

    x_0 = rng.randn(3)
    t = np.arange(0, T, integration_dt)
    X = scipy.integrate.odeint(dx_dt, x_0, t)
    return X


def gen_lorenz_data(num_samples, normalize=True, seed=20210610):
    integration_dt = 0.005
    data_dt = 0.025
    skipped_samples = 1000
    T = (num_samples + 2 * skipped_samples) * data_dt
    X = gen_lorenz_system(T, seed, integration_dt)
    if normalize:
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
    X_dwn = resample(X, num_samples + 2 * skipped_samples, axis=0)
    X_dwn = X_dwn[skipped_samples:-skipped_samples]
    return X_dwn


def oscillators_dynamics_mat(N=10, omega_sq=.01, alpha_sq=.2, gamma=.05, tau=1.):
    # spring matrix K
    K = np.zeros((N, N))
    K += np.eye(N) * (0.5 * omega_sq + alpha_sq)
    K[1:, :-1] += np.eye(N - 1) * (-0.5 * alpha_sq)  # lower diag
    K[:-1, 1:] += np.eye(N - 1) * (-0.5 * alpha_sq)  # upper diag
    K[0, -1] = -0.5 * alpha_sq
    K[-1, 0] = -0.5 * alpha_sq
    # friction matrix gamma
    gamma_mat = gamma * np.eye(N)
    # continuous-time dynamics matrix B (z^dot = B * z)
    B_top = np.concatenate((np.zeros((N, N)), np.eye(N)), axis=1)
    B_bottom = np.concatenate((-K, -gamma_mat), axis=1)
    B = np.concatenate((B_top, B_bottom), axis=0)
    # discrete-time dynamics matrix A (z_t = A * z_{t-1})
    A = expm(B * tau)
    return A


def oscillators_cross_cov_mats(A, T=10, sigma=1.):
    # get N
    N = A.shape[0] // 2
    # noise matrix sigma (just position noise)
    sigma_mat = np.zeros((2 * N, 2 * N))
    sigma_mat[:N, :N] = np.eye(N) * sigma**2
    # steady-state covariance matrix C
    C = np.ones(2 * N)
    for i in range(100000):
        C = A.dot(C).dot(A.T) + sigma_mat
    # function for generating cross-cov matrices <z_{t+k} z_t^T>

    def gen_cross_cov(k):
        A_k = matrix_power(A, k)
        A_k_C = A_k.dot(C)
        return A_k_C
    # make all cross_cov mats for delta t = {0, .., T-1}
    cross_cov_mats = np.array([gen_cross_cov(k) for k in range(T)])
    return cross_cov_mats


def sample_oscillators(A, T, sigma=1.):
    # generate sample of dynamics
    N = A.shape[0] // 2
    z = np.zeros((T, 2 * N))
    for i in range(1, T):
        z[i] = np.dot(A, z[i - 1])
        z[i, :N] += np.random.normal(0, sigma, N)
    return z


def gen_noise_cov(N, D, var, rng, V_noise=None):
    noise_spectrum = var * np.exp(-2 * np.arange(N) / D)
    if V_noise is None:
        V_noise = scipy.stats.ortho_group.rvs(N, random_state=rng)
    noise_cov = np.dot(V_noise, np.dot(np.diag(noise_spectrum), V_noise.T))
    return noise_cov


def random_basis(N, D, rng):
    return scipy.stats.ortho_group.rvs(N, random_state=rng)[:, :D]


def median_subspace(N, D, rng, num_samples=5000, V_0=None):
    subspaces = np.zeros((num_samples, N, D))
    angles = np.zeros((num_samples, min(D, V_0.shape[1])))
    if V_0 is None:
        V_0 = np.eye(N)[:, :D]
    for i in range(num_samples):
        subspaces[i] = random_basis(N, D, rng)
        angles[i] = np.rad2deg(scipy.linalg.subspace_angles(V_0, subspaces[i]))
    median_angles = np.median(angles, axis=0)
    median_subspace_idx = np.argmin(np.sum((angles - median_angles)**2, axis=1))
    median_subspace = subspaces[median_subspace_idx]
    return median_subspace


def embedded_lorenz_cross_cov_mats(N, T, snr=1., noise_dim=7, return_samples=False,
                                   num_lorenz_samples=10000, num_subspace_samples=5000,
                                   V_dynamics=None, V_noise=None, X_dynamics=None, seed=20200326):
    """Embed the Lorenz system into high dimensions with additive spatially
    structued white noise. Signal and noise subspaces are oriented with the
    median subspace angle.

    Parameters
    ----------
    N : int
        Embedding dimension.
    T : int
        Number of timesteps (2 * T_pi)
    snr : float
        Signal-to-noise ratio. Specifically it is the ratio of the largest
        eigenvalue of the signal covariance to the largest eigenvalue of the
        noise covariance.
    noise_dim : int
        Dimension at which noise eigenvalues fall to 1/e. If noise_dim is
        np.inf then a flat spectrum is used.
    return_samples : bool
        Whether to return cross_cov_mats or data samples.
    num_lorenz_samples : int
        Number of data samples to use.
    num_subspace_samples : int
        Number of random subspaces used to calculate the median relative angle.
    seed : int
        Seed for Numpy random state.
    """

    rng = np.random.RandomState(seed)
    # Generate Lorenz dynamics
    if X_dynamics is None:
        X_dynamics = gen_lorenz_data(num_lorenz_samples)
    dynamics_var = np.max(scipy.linalg.eigvalsh(np.cov(X_dynamics.T)))
    noise_var = dynamics_var / snr
    # Generate dynamics embedding matrix (will remain fixed)
    if V_dynamics is None:
        if N == 3:
            V_dynamics = np.eye(3)
        else:
            V_dynamics = random_basis(N, 3, rng)
    if noise_dim == np.inf:
        noise_cov = np.eye(N) * noise_var
    else:
        # Generate a subspace with median principal angles w.r.t. dynamics subspace
        if V_noise is None:
            V_noise = median_subspace(N, noise_dim, rng, num_samples=num_subspace_samples,
                                      V_0=V_dynamics)
        # Extend V_noise to a basis for R^N
        if V_noise.shape[1] < N:
            V_noise_comp = scipy.linalg.orth(np.eye(N) - np.dot(V_noise, V_noise.T))
            V_noise = np.concatenate((V_noise, V_noise_comp), axis=1)
        # Add noise covariance
        noise_cov = gen_noise_cov(N, noise_dim, noise_var, rng, V_noise=V_noise)
    # Generate actual samples of high-D data
    cross_cov_mats = calc_cross_cov_mats_from_data(X_dynamics, T)
    cross_cov_mats = np.array([V_dynamics.dot(C).dot(V_dynamics.T) for C in cross_cov_mats])
    cross_cov_mats[0] += noise_cov
    if return_samples:
        X_samples = (np.dot(X_dynamics, V_dynamics.T) +
                     rng.multivariate_normal(mean=np.zeros(N),
                                             cov=noise_cov, size=len(X_dynamics)))
        return cross_cov_mats, X_samples
    else:
        return cross_cov_mats
