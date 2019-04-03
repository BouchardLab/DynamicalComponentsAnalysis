import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
from math import floor

from cca.kron_pca import cv_toeplitz

def calc_cross_cov_mats_from_data(X, num_lags, regularization=None, reg_ops=None):
    """Compute a N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to num_lags-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    num_lags: int
        The number of time lags.
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """

    #mean center X
    X = X - X.mean(axis=0)
    N = X.shape[1]

    if regularization is None:
        cross_cov_mats = np.zeros((num_lags, N, N))
        for delta_t in range(num_lags):
            cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t)
            cross_cov_mats[delta_t] = cross_cov
        cov_est = calc_cov_from_cross_cov_mats(cross_cov_mats)
    elif regularization == 'kron':
        if reg_ops is not None:
            if skip in reg_ops.keys():
                skip = reg_ops["skip"]
            else:
                skip = 1
            if "num_folds" in reg_ops.keys():
                num_folds = reg_ops["num_folds"]
            else:
                num_folds = 5
        X_with_lags = form_lag_matrix(X, num_lags, skip=skip)
        r_vals = np.arange(2*T - 1) + 1
        sigma_vals = np.linspace(1, 4*T + 1, 10)
        _, r_opt, sigma_opt = cv_toeplitz(X_with_lags, N, num_lags, r_vals, sigma_vals, num_folds=num_folds)
        cross_cov_mats = calc_cross_cov_mats_from_cov(N, num_lags, cov_est)

    w = scipy.linalg.eigvalsh(cov_est)
    min_eig = np.min(w)
    if min_eig <= 0:
        print("Warning: spatiotemporal covariance matrix not PSD (min eig = " + str(min_eig) + ")")

    return cross_cov_mats


def calc_cross_cov_mats_from_cov(N, num_lags, cov):
    """Calculates num_lags N-by-N cross-covariance matrices given
    a N*num_lags-by-N*num_lags spatiotemporal covariance matrix by
    averaging over off-diagonal cross-covariance blocks with
    constant |t1-t2|.
    Parameters
    ----------
    N : int
        Numbner of spatial dimensions.
    num_lags: int
        Number of time-lags.
    cov : np.ndarray, shape (N*num_lags, N*num_lags)
        Spatiotemporal covariance matrix.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N)
        Cross-covariance matrices.
    """

    use_torch = isinstance(cov, torch.Tensor)

    if use_torch:
        cross_cov_mats = torch.zeros((num_lags, N, N))
    else:
        cross_cov_mats = np.zeros((num_lags, N, N))

    for delta_t in range(num_lags):
        if use_torch:
            to_avg_lower = torch.zeros((num_lags-delta_t, N, N))
            to_avg_upper = torch.zeros((num_lags-delta_t, N, N))
        else:
            to_avg_lower = np.zeros((num_lags-delta_t, N, N))
            to_avg_upper = np.zeros((num_lags-delta_t, N, N))

        for i in range(num_lags-delta_t):
            i_offset = delta_t*N
            to_avg_lower[i, :, :] = cov[i_offset+i*N:i_offset+(i+1)*N, i*N:(i+1)*N]
            to_avg_upper[i, :, :] = cov[i*N:(i+1)*N, i_offset+i*N:i_offset+(i+1)*N]

        if use_torch:
            cross_cov_mats[delta_t, :, :] = 0.5*(torch.mean(to_avg_lower, axis=0) + torch.mean(to_avg_upper, axis=0).T )
        else:
            cross_cov_mats[delta_t, :, :] = 0.5*(np.mean(to_avg_lower, axis=0) + np.mean(to_avg_upper, axis=0).T )
            
    return cross_cov_mats

def calc_cov_from_cross_cov_mats(cross_cov_mats):
    """Calculates the N*num_lags-by-N*num_lags spatiotemporal covariance matrix
    based on num_lags N-by-N cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    Returns
    -------
    cov : np.ndarray, shape (N*num_lags, N*num_lags)
        Big covariance matrix, stationary in time by construction.
    """

    N = cross_cov_mats[0].shape[0]
    num_lags = len(cross_cov_mats)
    use_torch = isinstance(cross_cov_mats[0], torch.Tensor)

    cross_cov_mats_repeated = []
    for i in range(num_lags):
        for j in range(num_lags):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)])
            else:
                if use_torch:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].t())
                else:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].T)

    if use_torch:
        cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated), (num_lags, num_lags, N, N))
        cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1) for cov_ii in cov_tensor])
    else:
        cov_tensor = np.reshape(np.stack(cross_cov_mats_repeated), (num_lags, num_lags, N, N))
        cov = np.concatenate([np.concatenate([cov_ii_jj for cov_ii_jj in cov_ii], axis=1) for cov_ii in cov_tensor])

    return cov


def calc_pi_from_cov(cov_2T):
    """Calculates the mutual information ("predictive information"
    or "PI") between variables  {1,...,T} and {T+1,...,2*T}, which
    are jointly Gaussian with covariance matrix cov_2T.
    Parameters
    ----------
    cov_2T : np.ndarray, shape (2*T, 2*T)
        Covariance matrix.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """

    half = int(cov_2T.shape[0]/2)
    use_torch = isinstance(cov_2T, torch.Tensor)

    cov_T = cov_2T[:half, :half]
    if use_torch:
        logdet_T = torch.logdet(cov_T)
        logdet_2T = torch.logdet(cov_2T)
    else:
        logdet_T = np.linalg.slogdet(cov_T)[1]
        logdet_2T = np.linalg.slogdet(cov_2T)[1]

    PI = (2. * logdet_T - logdet_2T) / np.log(2.)
    return PI


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with num_lags-1 N-by-N cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """
    use_torch = isinstance(cross_cov_mats[0], torch.Tensor)

    if use_torch and isinstance(proj, np.ndarray):
        proj = torch.tensor(proj, device=cross_cov_mats[0].device, dtype=cross_cov_mats[0].dtype)

    T = cross_cov_mats.shape[0] // 2
    cross_cov_mats_proj = []
    if proj is not None:
        for i in range(2*T):
            cross_cov = cross_cov_mats[i]
            if use_torch:
                cross_cov_proj = torch.mm(proj.t(), torch.mm(cross_cov, proj))
            else:
                cross_cov_proj = np.dot(proj.T, np.dot(cross_cov, proj))

            cross_cov_mats_proj.append(cross_cov_proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov_2T = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2T)

    return PI
