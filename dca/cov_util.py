import numpy as np
import scipy as sp
import collections
import torch
import functools

from .data_util import form_lag_matrix
from sklearn.utils.extmath import randomized_svd


def rectify_spectrum(cov, epsilon=1e-6, verbose=False):
    min_eig = np.min(sp.linalg.eigvalsh(cov))
    if min_eig < 0:
        cov += (-min_eig + epsilon) * np.eye(cov.shape[0])
        if verbose:
            print("Warning: non-PSD matrix (had to increase eigenvalues)")


def toeplitzify(C, T, N, symmetrize=True):
    C_toep = np.zeros((T * N, T * N))
    for delta_t in range(T):
        to_avg_lower = np.zeros((T - delta_t, N, N))
        to_avg_upper = np.zeros((T - delta_t, N, N))
        for i in range(T - delta_t):
            to_avg_lower[i] = C[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i] = C[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]
        avg_lower = np.mean(to_avg_lower, axis=0)
        avg_upper = np.mean(to_avg_upper, axis=0)
        if symmetrize:
            avg_lower = 0.5 * (avg_lower + avg_upper.T)
            avg_upper = 0.5 * (avg_lower.T + avg_upper)
        for i in range(T - delta_t):
            C_toep[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N] = avg_lower
            C_toep[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N] = avg_upper
    return C_toep


def calc_chunked_cov(X, T, stride, chunks, cov_est=None):
    """Calculate an unormalized (by sample count) lagged covariance matrix
    in chunks to save memory.
    """
    if cov_est is None:
        cov_est = 0.
    n_samples = 0
    if X.shape[0] <= T * chunks:
        raise ValueError
    ends = np.linspace(0, X.shape[0], chunks + 1, dtype=int)[1:]
    start = 0
    for chunk in range(chunks):
        X_with_lags = form_lag_matrix(X[start:ends[chunk]], T, stride=stride)
        start = ends[chunk] - T + 1
        ni_samples = X_with_lags.shape[0]
        cov_est = cov_est + np.dot(X_with_lags.T, X_with_lags)
        n_samples += ni_samples
    return cov_est, n_samples


def calc_cross_cov_mats_from_data(X, T, chunks=None, regularization=None, reg_ops=None):
    """Compute a N-by-N cross-covariance matrix, where N is the data dimensionality,
    for each time lag up to T-1.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
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
    if reg_ops is None:
        reg_ops = dict()
    stride = reg_ops.get('stride', 1)
    if chunks is not None and regularization is not None:
        raise NotImplementedError

    if isinstance(X, list) or X.ndim == 3:
        mean = np.concatenate(X).mean(axis=0, keepdims=True)
        X = [Xi - mean for Xi in X]
        N = X[0].shape[-1]
        if chunks is None:
            X_with_lags = np.concatenate([form_lag_matrix(Xi, T, stride=stride) for Xi in X])
        else:
            n_samples = 0
            cov_est = 0.
            for Xi in X:
                cov_est, ni_samples = calc_chunked_cov(Xi, T, stride, chunks, cov_est=cov_est)
                n_samples += ni_samples
            cov_est /= (n_samples - 1.)
            cov_est = toeplitzify(cov_est, T, N)
    else:
        X = X - X.mean(axis=0, keepdims=True)
        N = X.shape[-1]
        if chunks is None:
            X_with_lags = form_lag_matrix(X, T, stride=stride)
        else:
            cov_est, n_samples = calc_chunked_cov(X, T, stride, chunks)
            cov_est /= (n_samples - 1.)
            cov_est = toeplitzify(cov_est, T, N)

    if chunks is not None:
        pass
    elif regularization is None and chunks is None:
        cov_est = np.cov(X_with_lags, rowvar=False)
        cov_est = toeplitzify(cov_est, T, N)
    elif regularization == 'kron':
        num_folds = reg_ops.get('num_folds', 5)
        r_vals = np.arange(1, min(2 * T, N**2 + 1))
        sigma_vals = np.concatenate([np.linspace(1, 4 * T + 1, 10), [100. * T]])
        alpha_vals = np.concatenate([[0.], np.logspace(-2, -1, 10)])
        ll_vals, opt_idx = cv_toeplitz(X_with_lags, T, N, r_vals, sigma_vals, alpha_vals,
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
    constant `|t1-t2|`.
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
            to_avg_lower = torch.zeros((T - delta_t, N, N))
            to_avg_upper = torch.zeros((T - delta_t, N, N))
        else:
            to_avg_lower = np.zeros((T - delta_t, N, N))
            to_avg_upper = np.zeros((T - delta_t, N, N))

        for i in range(T - delta_t):
            to_avg_lower[i, :, :] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i, :, :] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]

        avg_lower = to_avg_lower.mean(axis=0)
        avg_upper = to_avg_upper.mean(axis=0)

        if use_torch:
            cross_cov_mats[delta_t, :, :] = 0.5 * (avg_lower + avg_upper.t())
        else:
            cross_cov_mats[delta_t, :, :] = 0.5 * (avg_lower + avg_upper.T)

    return cross_cov_mats


def calc_cov_from_cross_cov_mats(cross_cov_mats):
    """Calculates the N*T-by-N*T spatiotemporal covariance matrix based on
    T N-by-N cross-covariance matrices.

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
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)])
            else:
                if use_torch:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)].t())
                else:
                    cross_cov_mats_repeated.append(cross_cov_mats[abs(i - j)].T)

    if use_torch:
        cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1)
                         for cov_ii in cov_tensor])
    else:
        cov_tensor = np.reshape(np.stack(cross_cov_mats_repeated), (T, T, N, N))
        cov = np.concatenate([np.concatenate([cov_ii_jj for cov_ii_jj in cov_ii], axis=1)
                              for cov_ii in cov_tensor])

    return cov


def calc_pi_from_data(X, T):
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
        Mutual information in nats.
    """
    ccms = calc_cross_cov_mats_from_data(X, T)

    return calc_pi_from_cross_cov_mats(ccms)


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
        Mutual information in nats.
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
        Mutual information in nats.
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
        for i in range(2 * T):
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
        Mutual information in nats.
    """
    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov_2_T_pi = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2_T_pi)

    return PI


def calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with T-1 N-by-N cross-covariance matrices using the block-Toeplitz
    algorithm.

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
        Mutual information in nats.
    """
    use_torch = isinstance(cross_cov_mats, torch.Tensor)
    if proj is not None:
        ccms = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        ccms = cross_cov_mats
    T, d, d = ccms.shape
    A = dict()
    Ab = dict()

    if use_torch:
        v = ccms[0]
        vb = [ccms[0]]
        D = ccms[1]
        for ii in range(1, T):
            if ii > 1:
                As = torch.stack([A[ii - 2, ii - jj - 1] for jj in range(1, ii)])
                D = ccms[ii] - torch.matmul(As, ccms[1:ii]).sum(dim=0)
            A[(ii - 1, ii - 1)] = torch.solve(D.t(), vb[ii - 1].t())[0].t()
            Ab[(ii - 1, ii - 1)] = torch.solve(D, v.t())[0].t()

            for kk in range(1, ii):
                A[(ii - 1, kk - 1)] = (A[(ii - 2, kk - 1)]
                                       - A[(ii - 1, ii - 1)].mm(Ab[(ii - 2, ii - kk - 1)]))
                Ab[(ii - 1, kk - 1)] = (Ab[(ii - 2, kk - 1)]
                                        - Ab[(ii - 1, ii - 1)].mm(A[(ii - 2, ii - kk - 1)]))

            if ii < T-1:
                As = torch.stack([A[(ii - 1, jj - 1)] for jj in range(1, ii + 1)])
                if ii == 1:
                    cs = ccms[[1]]
                else:
                    cs = ccms[1: ii + 1]
                v = ccms[0] - torch.matmul(As, torch.transpose(cs, 1, 2)).sum(dim=0)
            Abs = torch.stack([Ab[(ii - 1, jj - 1)] for jj in range(1, ii + 1)])
            if ii == 1:
                cs = ccms[[1]]
            else:
                cs = ccms[1: ii + 1]
            vb.append(ccms[0] - torch.matmul(Abs, cs).sum(dim=0))
        logdets = [torch.slogdet(vb[ii])[1] for ii in range(T)]
        return sum(logdets[:T // 2]) - 0.5 * sum(logdets)
    else:
        vb = np.zeros((T, d, d))
        v = ccms[0]
        vb[0] = ccms[0]
        D = ccms[1]
        for ii in range(1, T):
            if ii > 1:
                D = ccms[ii] - sum([A[ii - 2, ii - jj - 1].dot(ccms[jj])
                                    for jj in range(1, ii)])
            A[(ii - 1, ii - 1)] = np.linalg.solve(vb[ii - 1].T, D.T).T
            Ab[(ii - 1, ii - 1)] = np.linalg.solve(v.T, D).T
            for kk in range(1, ii):
                if ii < T-1:
                    A[(ii - 1, kk - 1)] = (A[(ii - 2, kk - 1)]
                                           - A[(ii - 1, ii - 1)].dot(Ab[(ii - 2, ii - kk - 1)]))
                Ab[(ii - 1, kk - 1)] = (Ab[(ii - 2, kk - 1)]
                                        - Ab[(ii - 1, ii - 1)].dot(A[(ii - 2, ii - kk - 1)]))
            if ii < T-1:
                v = ccms[0] - sum([A[(ii - 1, jj - 1)].dot(ccms[jj].T) for jj in range(1, ii + 1)])
            vb[ii] = ccms[0] - sum([Ab[(ii - 1, jj - 1)].dot(ccms[jj]) for jj in range(1, ii + 1)])
        logdets = [np.linalg.slogdet(vb[ii])[1] for ii in range(T)]
        return sum(logdets[:T // 2]) - 0.5 * sum(logdets)


"""
====================================================================================================
====================================================================================================
===================================                                  ===============================
===================================     KronPCA-related methods      ===============================
===================================                                  ===============================
====================================================================================================
====================================================================================================
"""


class memoized(object):
    """Decorator for memoization.
    From: https://wiki.python.org/moin/PythonDecoratorLibrary.

    Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        else:
            value = self.func(*args)
            self.cache[args] = value
            return value

    def __repr__(self):
        # Return the function's docstring.
        return self.func.__doc__

    def __get__(self, obj, objtype):
        # Support instance methods.
        return functools.partial(self.__call__, obj)


@memoized
def pv_permutation(T, N):
    A = np.arange((T * N)**2, dtype=np.int).reshape((T * N, T * N))
    A_perm = np.zeros((T**2, N**2), dtype=np.int)
    for i in range(T):
        for j in range(T):
            row_idx = i * T + j
            A_block = A[i * N:(i + 1) * N, j * N:(j + 1) * N]
            A_perm[row_idx, :] = A_block.T.reshape((N**2,))  # equivalent to I_block.vectorize
    perm = A_perm.ravel()
    perm_inv = perm.argsort()
    return perm, perm_inv


def pv_rearrange(C, T, N):
    perm, _ = pv_permutation(T, N)
    C_prime = C.ravel()[perm].reshape((T**2, N**2))
    return C_prime


def pv_rearrange_inv(C, T, N):
    _, perm_inv = pv_permutation(T, N)
    C_prime = C.ravel()[perm_inv].reshape((T * N, T * N))
    return C_prime


def build_P(T):
    P = np.zeros((2 * T - 1, T**2))
    idx = np.arange(T**2).reshape((T, T)).T + 1
    for offset in range(-T + 1, T):
        diag_idx = np.diagonal(idx, offset=offset)
        P[offset + T - 1, diag_idx - 1] = 1. / np.sqrt(T - np.abs(offset))
    return P


def toeplitz_reg(cov, T, N, r):
    R_C = pv_rearrange(cov, T, N)
    P = build_P(T)
    to_svd = P.dot(R_C)
    U, s, Vt = randomized_svd(to_svd, n_components=r + 1, n_iter=40, random_state=42)
    trunc_svd = U[:, :-1].dot(np.diag(s[:-1] - s[-1])).dot(Vt[:-1, :])
    cov_reg = pv_rearrange_inv(P.T.dot(trunc_svd), T, N)
    return cov_reg


def non_toeplitz_reg(cov, T, N, r):
    R_C = pv_rearrange(cov, T, N)
    U, s, Vt = randomized_svd(R_C, n_components=r + 1, n_iter=40, random_state=42)
    trunc_svd = U[:, :-1].dot(np.diag(s[:-1] - s[-1])).dot(Vt[:-1, :])
    cov_reg = pv_rearrange_inv(trunc_svd, T, N)
    return cov_reg


def toeplitz_reg_taper_shrink(cov, T, N, r, sigma, alpha):
    cov_reg = toeplitz_reg(cov, T, N, r)
    cov_reg_taper = taper_cov(cov_reg, T, N, sigma)
    cov_reg_taper_shrink = (1. - alpha) * cov_reg_taper + alpha * np.eye(T * N)
    return cov_reg_taper_shrink


def gaussian_log_likelihood(cov, sample_cov, num_samples):
    to_trace = np.linalg.solve(cov, sample_cov)
    log_det_cov = np.linalg.slogdet(cov)[1]
    d = cov.shape[1]
    log_likelihood = -0.5 * num_samples * (d * np.log(2. * np.pi) +
                                           log_det_cov + np.trace(to_trace))
    return log_likelihood


def taper_cov(cov, T, N, sigma):
    t = np.arange(T).reshape((T, 1))
    delta_t = t - t.T
    temporal_kernel = np.exp(-(delta_t / sigma)**2)
    full_kernel = np.kron(temporal_kernel, np.ones((N, N)))
    result = full_kernel * cov
    return result


def cv_toeplitz(X_with_lags, T, N, r_vals, sigma_vals, alpha_vals, num_folds=10, verbose=False):
    fold_size = int(np.floor(len(X_with_lags) / num_folds))
    P = build_P(T)
    ll_vals = np.zeros((num_folds, len(r_vals), len(sigma_vals), len(alpha_vals)))

    for cv_iter in range(num_folds):
        if verbose:
            print("fold =", cv_iter + 1)

        X_train = np.concatenate((X_with_lags[:cv_iter * fold_size],
                                  X_with_lags[(cv_iter + 1) * fold_size:]), axis=0)
        X_test = X_with_lags[cv_iter * fold_size:(cv_iter + 1) * fold_size]
        num_samples = len(X_test)
        cov_train, cov_test = np.cov(X_train.T), np.cov(X_test.T)
        cov_train, cov_test = toeplitzify(cov_train, T, N), toeplitzify(cov_test, T, N)
        rectify_spectrum(cov_train)
        rectify_spectrum(cov_test)

        R_C = pv_rearrange(cov_train, T, N)
        to_svd = P.dot(R_C)
        U, s, Vt = randomized_svd(to_svd, n_components=np.max(r_vals), n_iter=40, random_state=42)

        for r_idx in range(len(r_vals)):
            r = r_vals[r_idx]
            if verbose:
                print("r =", r)
            if r_idx == len(r_vals) - 1:
                trunc_svd = to_svd
            else:
                trunc_svd = U[:, :r].dot(np.diag(s[:r] - s[r])).dot(Vt[:r, :])
            cov_kron = pv_rearrange_inv(P.T.dot(trunc_svd), T, N)
            for sigma_idx in range(len(sigma_vals)):
                sigma = sigma_vals[sigma_idx]
                cov_kron_taper = taper_cov(cov_kron, T, N, sigma)
                for alpha_idx in range(len(alpha_vals)):
                    alpha = alpha_vals[alpha_idx]
                    cov_kron_taper_shrunk = ((1. - alpha) * cov_kron_taper + alpha * np.eye(T * N))
                    ll = gaussian_log_likelihood(cov_kron_taper_shrunk, cov_test, num_samples)
                    ll_vals[cv_iter, r_idx, sigma_idx, alpha_idx] = ll

    opt_idx = np.unravel_index(ll_vals.mean(axis=0).argmax(), ll_vals.shape[1:])
    return ll_vals, opt_idx
