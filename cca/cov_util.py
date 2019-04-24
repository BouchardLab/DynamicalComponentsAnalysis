import numpy as np
import scipy as sp
import torch

from .kron_pca import cv_toeplitz, toeplitz_reg_taper_shrink
from .data_util import form_lag_matrix

def rectify_spectrum(cov, epsilon=1e-6, verbose=False):
    min_eig = np.min(scipy.linalg.eigvalsh(cov))
    if min_eig < 0:
        cov += (-min_eig + epsilon)*np.eye(cov.shape[0])
        if verbose:
            print("Warning: non-PSD matrix (had to increase eigenvalues)")

def toeplitzify(C, T, N, symmetrize=True):
    C_toep = np.zeros((T*N, T*N))
    for delta_t in range(T):
        to_avg_lower = np.zeros((T-delta_t, N, N))
        to_avg_upper = np.zeros((T-delta_t, N, N))
        for i in range(T - delta_t):
            to_avg_lower[i] = C[(delta_t + i)*N : (delta_t + i + 1)*N, i*N : (i + 1)*N]
            to_avg_upper[i] = C[i*N : (i + 1)*N, (delta_t + i)*N : (delta_t + i + 1)*N]
        avg_lower = np.mean(to_avg_lower, axis=0)
        avg_upper = np.mean(to_avg_upper, axis=0)
        if symmetrize:
            avg_lower = 0.5*(avg_lower + avg_upper.T)
            avg_upper = 0.5*(avg_lower.T + avg_upper)
        for i in range(T-delta_t):
            C_toep[(delta_t + i)*N : (delta_t + i + 1)*N, i*N : (i + 1)*N] = avg_lower
            C_toep[i*N : (i + 1)*N, (delta_t + i)*N : (delta_t + i + 1)*N] = avg_upper
    return C_toep

def calc_cross_cov_mats_from_data(X, T, regularization=None, reg_ops=None):
    """Compute a N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T: int
        The number of time lags.
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """

    #mean center X
    X = X - X.mean(axis=0)
    _, N = X.shape

    if reg_ops is None:
        reg_ops = dict()
    skip = reg_ops.get('skip', 1)
    X_with_lags = form_lag_matrix(X, T, skip=skip)

    if regularization is None:
        cov_est = np.cov(X_with_lags, rowvar=False)
        cov_est = toeplitzify(cov_est, T, N)
    elif regularization == 'kron':
        num_folds = reg_ops.get('num_folds', 5)
        r_vals = np.arange(1, min(2*T, N**2 + 1))
        sigma_vals = np.concatenate([np.linspace(1, 4*T + 1, 10), [100. * T]])
        alpha_vals = np.concatenate([[0.], np.logspace(-2, -1, 10)])
        ll_vals, opt_idx = cv_toeplitz(X_with_lags, T, N,
                                          r_vals, sigma_vals, alpha_vals,
                                          num_folds=num_folds)
        ri, si, ai = opt_idx
        cov = np.cov(X_with_lags, rowvar=False)
        cov_est = toeplitz_reg_taper_shrink(cov, T, N, r_vals[ri], sigma_vals[si], alpha_vals[ai])
    else:
        raise ValueError

    rectify_spectrum(cov_est, verbose=True)
    cross_cov_mats = calc_cross_cov_mats_from_cov(cov_est, T, N)
    return cross_cov_mats


def calc_cross_cov_mats_from_cov(cov, T, N):
    """Calculates T N-by-N cross-covariance matrices given
    a N*T-by-N*T spatiotemporal covariance matrix by
    averaging over off-diagonal cross-covariance blocks with
    constant |t1-t2|.
    Parameters
    ----------
    N : int
        Numbner of spatial dimensions.
    T: int
        Number of time-lags.
    cov : np.ndarray, shape (N*T, N*T)
        Spatiotemporal covariance matrix.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices.
    """

    use_torch = isinstance(cov, torch.Tensor)

    if use_torch:
        cross_cov_mats = torch.zeros((T, N, N))
    else:
        cross_cov_mats = np.zeros((T, N, N))

    for delta_t in range(T):
        if use_torch:
            to_avg_lower = torch.zeros((T-delta_t, N, N))
            to_avg_upper = torch.zeros((T-delta_t, N, N))
        else:
            to_avg_lower = np.zeros((T-delta_t, N, N))
            to_avg_upper = np.zeros((T-delta_t, N, N))

        for i in range(T-delta_t):
            i_offset = delta_t*N
            to_avg_lower[i, :, :] = cov[(delta_t + i)*N : (delta_t + i + 1)*N, i*N : (i + 1)*N]
            to_avg_upper[i, :, :] = cov[i*N : (i + 1)*N, (delta_t + i)*N : (delta_t + i + 1)*N]

        avg_lower = to_avg_lower.mean(axis=0)
        avg_upper = to_avg_upper.mean(axis=0)

        if use_torch:
            cross_cov_mats[delta_t, :, :] = 0.5*(avg_lower + avg_upper.t())
        else:
            cross_cov_mats[delta_t, :, :] = 0.5*(avg_lower + avg_upper.T)

    return cross_cov_mats

def calc_cov_from_cross_cov_mats(cross_cov_mats):
    """Calculates the N*T-by-N*T spatiotemporal covariance matrix
    based on T N-by-N cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    Returns
    -------
    cov : np.ndarray, shape (N*T, N*T)
        Big covariance matrix, stationary in time by construction.
    """

    N = cross_cov_mats.shape[1]
    T = len(cross_cov_mats)
    use_torch = isinstance(cross_cov_mats, torch.Tensor)

    cross_cov_mats_repeated = []
    for i in range(T):
        for j in range(T):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)])
            else:
                if use_torch:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].t())
                else:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].T)

    if use_torch:
        cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1) for cov_ii in cov_tensor])
    else:
        cov_tensor = np.reshape(np.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = np.concatenate([np.concatenate([cov_ii_jj for cov_ii_jj in cov_ii], axis=1) for cov_ii in cov_tensor])

    return cov


def calc_pi_from_cov(cov_2_T_pi):
    """Calculates the mutual information ("predictive information"
    or "PI") between variables  {1,...,T_pi} and {T_pi+1,...,2*T_pi}, which
    are jointly Gaussian with covariance matrix cov_2_T_pi.
    Parameters
    ----------
    cov_2_T_pi : np.ndarray, shape (2*T_pi, 2*T_pi)
        Covariance matrix.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """

    T_pi = cov_2_T_pi.shape[0] // 2
    use_torch = isinstance(cov_2_T_pi, torch.Tensor)

    cov_T_pi = cov_2_T_pi[:T_pi, :T_pi]
    if use_torch:
        logdet_T_pi = torch.slogdet(cov_T_pi)[1]
        logdet_2T_pi = torch.slogdet(cov_2_T_pi)[1]
    else:
        logdet_T_pi = np.linalg.slogdet(cov_T_pi)[1]
        logdet_2T_pi = np.linalg.slogdet(cov_2_T_pi)[1]

    PI = logdet_T_pi - .5 * logdet_2T_pi
    return PI


def project_cross_cov_mats(cross_cov_mats, proj):
    """Projects the cross covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.
    Returns
    -------
    cross_cov_mats_proj : ndarray, shape (T, d, d)
        Mutual information in bits.
    """
    if isinstance(cross_cov_mats, torch.Tensor):
        use_torch = True
    elif isinstance(cross_cov_mats[0], torch.Tensor):
        cross_cov_mats = torch.stack(cross_cov_mats)
        use_torch = True
    else:
        use_torch = False

    if use_torch and isinstance(proj, np.ndarray):
        proj = torch.tensor(proj, device=cross_cov_mats.device, dtype=cross_cov_mats.dtype)

    T = cross_cov_mats.shape[0] // 2
    if use_torch:
        cross_cov_mats_proj = torch.matmul(proj.t().unsqueeze(0),
                                           torch.matmul(cross_cov_mats,
                                                        proj.unsqueeze(0)))
    else:
        cross_cov_mats_proj = []
        for i in range(2*T):
            cross_cov = cross_cov_mats[i]
            cross_cov_proj = np.dot(proj.T, np.dot(cross_cov, proj))
            cross_cov_mats_proj.append(cross_cov_proj)
        cross_cov_mats_proj = np.stack(cross_cov_mats_proj)

    return cross_cov_mats_proj


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with T-1 N-by-N cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (T, N, N)
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
    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov_2_T_pi = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2_T_pi)

    return PI
