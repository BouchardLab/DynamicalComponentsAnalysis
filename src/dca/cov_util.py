import logging
import numpy as np
import scipy as sp
import collections
import torch
import functools
from numpy.lib.stride_tricks import as_strided

from sklearn.utils.extmath import randomized_svd
from sklearn.utils import check_random_state


logging.basicConfig()


def form_lag_matrix(X, T, stride=1, stride_tricks=True, rng=None, writeable=False):
    """Form the data matrix with `T` lags.

    Parameters
    ----------
    X : ndarray (n_time, N)
        Timeseries with no lags.
    T : int
        Number of lags.
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    stride_tricks : bool
        Whether to use numpy stride tricks to form the lagged matrix or create
        a new array. Using numpy stride tricks can can lower memory usage, especially for
        large `T`. If `False`, a new array is created.
    writeable : bool
        For testing. You should not need to set this to True. This function uses stride tricks
        to form the lag matrix which means writing to the array will have confusing behavior.
        If `stride_tricks` is `False`, this flag does nothing.

    Returns
    -------
    X_with_lags : ndarray (n_lagged_time, N * T)
        Timeseries with lags.
    """
    if not isinstance(stride, int) or stride < 1:
        if not isinstance(stride, float) or stride <= 0. or stride >= 1.:
            raise ValueError('stride should be an int and greater than or equal to 1 or a float ' +
                             'between 0 and 1.')
    N = X.shape[1]
    frac = None
    if isinstance(stride, float):
        frac = stride
        stride = 1
    n_lagged_samples = (len(X) - T) // stride + 1
    if n_lagged_samples < 1:
        raise ValueError('T is too long for a timeseries of length {}.'.format(len(X)))
    if stride_tricks:
        X = np.asarray(X, dtype=float, order='C')
        shape = (n_lagged_samples, N * T)
        strides = (X.strides[0] * stride,) + (X.strides[-1],)
        X_with_lags = as_strided(X, shape=shape, strides=strides, writeable=writeable)
    else:
        X_with_lags = np.zeros((n_lagged_samples, T * N))
        for i in range(n_lagged_samples):
            X_with_lags[i, :] = X[i * stride:i * stride + T, :].flatten()
    if frac is not None:
        rng = check_random_state(rng)
        idxs = np.sort(rng.choice(n_lagged_samples, size=int(np.ceil(n_lagged_samples * frac)),
                                  replace=False))
        X_with_lags = X_with_lags[idxs]

    return X_with_lags


def rectify_spectrum(cov, epsilon=1e-6, logger=None):
    """Rectify the spectrum of a covariance matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix
    epsilon : float
        Minimum eigenvalue for the rectified spectrum.
    verbose : bool
        Whethere to print when the spectrum needs to be rectified.
    """
    eigvals = sp.linalg.eigvalsh(cov)
    n_neg = np.sum(eigvals <= 0.)
    if n_neg > 0:
        cov += (-np.min(eigvals) + epsilon) * np.eye(cov.shape[0])
        if logger is not None:
            string = 'Non-PSD matrix, {} of {} eigenvalues were not positive.'
            logger.info(string.format(n_neg, eigvals.size))


def toeplitzify(cov, T, N, symmetrize=True):
    """Make a matrix block-Toeplitz by averaging along the block diagonal.

    Parameters
    ----------
    cov : ndarray (T*N, T*N)
        Covariance matrix to make block toeplitz.
    T : int
        Number of blocks.
    N : int
        Number of features per block.
    symmetrize : bool
        Whether to ensure that the whole matrix is symmetric.
        Optional (default=True).

    Returns
    -------
    cov_toep : ndarray (T*N, T*N)
        Toeplitzified matrix.
    """
    cov_toep = np.zeros((T * N, T * N))
    for delta_t in range(T):
        to_avg_lower = np.zeros((T - delta_t, N, N))
        to_avg_upper = np.zeros((T - delta_t, N, N))
        for i in range(T - delta_t):
            to_avg_lower[i] = cov[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N]
            to_avg_upper[i] = cov[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N]
        avg_lower = np.mean(to_avg_lower, axis=0)
        avg_upper = np.mean(to_avg_upper, axis=0)
        if symmetrize:
            avg_lower = 0.5 * (avg_lower + avg_upper.T)
            avg_upper = avg_lower.T
        for i in range(T - delta_t):
            cov_toep[(delta_t + i) * N:(delta_t + i + 1) * N, i * N:(i + 1) * N] = avg_lower
            cov_toep[i * N:(i + 1) * N, (delta_t + i) * N:(delta_t + i + 1) * N] = avg_upper
    return cov_toep


def calc_chunked_cov(X, T, stride, chunks, cov_est=None, rng=None, stride_tricks=True):
    """Calculate an unormalized (by sample count) lagged covariance matrix
    in chunks to save memory.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The N-dimensional time series data from which the cross-covariance
        matrices are computed.
    T : int
        The number of time lags.
    stride : int
        The number of time-points to skip between samples.
    chunks : int
        Number of chunks to break the data into when calculating the lagged cross
        covariance. More chunks will mean less memory used
    cov_est : ndarray
        Current estimate of unnormalized cov_est to be added to.

    Return
    ------
    cov_est : ndarray
        Current covariance estimate.
    n_samples
        How many samples were used.
    """
    if cov_est is None:
        cov_est = 0.
    n_samples = 0
    if X.shape[0] < T * chunks:
        raise ValueError('Time series is too short to chunk for cov estimation.')
    ends = np.linspace(0, X.shape[0], chunks + 1, dtype=int)[1:]
    start = 0
    for chunk in range(chunks):
        X_with_lags = form_lag_matrix(X[start:ends[chunk]], T, stride=stride,
                                      rng=rng, stride_tricks=stride_tricks)
        start = ends[chunk] - T + 1
        ni_samples = X_with_lags.shape[0]
        cov_est += np.dot(X_with_lags.T, X_with_lags)
        n_samples += ni_samples
    return cov_est, n_samples


def calc_cross_cov_mats_from_data(X, T, mean=None, chunks=None, stride=1,
                                  rng=None, regularization=None, reg_ops=None,
                                  stride_tricks=True, logger=None, method='toeplitzify'):
    """Compute the N-by-N cross-covariance matrix, where N is the data dimensionality,
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
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.
    regularization : string
        Regularization method for computing the spatiotemporal covariance matrix.
    reg_ops : dict
        Paramters for regularization.
    stride_tricks : bool
        Whether to use numpy stride tricks in form_lag_matrix. True will use less
        memory for large T.
    logger : logger
        Logger.
    method : str
        'ML' for EM-based maximum likelihood block toeplitz estimation, 'toeplitzify' for naive
        block averaging.

    Returns
    -------
    cross_cov_mats : np.ndarray, shape (T, N, N), float
        Cross-covariance matrices. cross_cov_mats[dt] is the cross-covariance between
        X(t) and X(t+dt), where X(t) is an N-dimensional vector.
    """
    if reg_ops is None:
        reg_ops = dict()
    if chunks is not None and regularization is not None:
        raise NotImplementedError

    if isinstance(X, list) or X.ndim == 3:
        for Xi in X:
            if len(Xi) <= T:
                raise ValueError('T must be shorter than the length of the shortest ' +
                                 'timeseries. If you are using the DCA model, 2 * DCA.T must be ' +
                                 'shorter than the shortest timeseries.')
        if mean is None:
            mean = np.concatenate(X).mean(axis=0, keepdims=True)
        X = [Xi - mean for Xi in X]
        N = X[0].shape[-1]
        if chunks is None:
            cov_est = np.zeros((N * T, N * T))
            n_samples = 0
            for Xi in X:
                X_with_lags = form_lag_matrix(Xi, T, stride=stride, stride_tricks=stride_tricks,
                                              rng=rng)
                cov_est += np.dot(X_with_lags.T, X_with_lags)
                n_samples += len(X_with_lags)
            cov_est /= (n_samples - 1.)
        else:
            n_samples = 0
            cov_est = np.zeros((N * T, N * T))
            for Xi in X:
                cov_est, ni_samples = calc_chunked_cov(Xi, T, stride, chunks, cov_est=cov_est,
                                                       stride_tricks=stride_tricks, rng=rng)
                n_samples += ni_samples
            cov_est /= (n_samples - 1.)
    else:
        if len(X) <= T:
            raise ValueError('T must be shorter than the length of the shortest '
                             'timeseries. If you are using the DCA model, 2 * DCA.T must be '
                             'shorter than the shortest timeseries.')
        if mean is None:
            mean = X.mean(axis=0, keepdims=True)
        X = X - mean
        N = X.shape[-1]
        if chunks is None:
            X_with_lags = form_lag_matrix(X, T, stride=stride, stride_tricks=stride_tricks,
                                          rng=rng)
            cov_est = np.cov(X_with_lags, rowvar=False)
        else:
            cov_est, n_samples = calc_chunked_cov(X, T, stride, chunks,
                                                  stride_tricks=stride_tricks, rng=rng)
            cov_est /= (n_samples - 1.)

    if regularization is None:
        if method == 'toeplitzify':
            cov_est = toeplitzify(cov_est, T, N)
        elif method == 'ML':
            cov_est = block_toeplitz_covariance(X=None, S_X=cov_est, T=T)
        else:
            raise ValueError('`method` should be "toeplitzify" or "ML".')
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

    rectify_spectrum(cov_est, logger=logger)
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


def calc_pi_from_data(X, T, proj=None, stride=1, rng=None):
    """Calculates the Gaussian Predictive Information between variables
    {1,...,T_pi} and {T_pi+1,...,2*T_pi}..

    Parameters
    ----------
    X : ndarray or torch tensor (time, features) or (batches, time, features)
        Data used to calculate the PI.
    T : int
        This T should be 2 * T_pi. This T sets the joint window length not the
        past or future window length.
    proj : ndarray or torch tensor
        Projection matrix for data (optional). If `proj` is not given, the PI of
        the dataset is given.
    stride : int or float
        If stride is an `int`, it defines the stride between lagged samples used
        to estimate the cross covariance matrix. Setting stride > 1 can speed up the
        calculation, but may lead to a loss in accuracy. Setting stride to a `float`
        greater than 0 and less than 1 will random subselect samples.
    rng : NumPy random state
        Only used if `stride` is a float.

    Returns
    -------
    PI : float
        Mutual information in nats.
    """
    if T % 2 != 0:
        raise ValueError('T must be even (This T sets the joint window length,'
                         ' not the past or future length')

    ccms = calc_cross_cov_mats_from_data(X, T, stride=stride, rng=rng)

    return calc_pi_from_cross_cov_mats(ccms, proj=proj)


def calc_pi_from_cov(cov_2_T_pi):
    """Calculates the Gaussian Predictive Information between variables
    {1,...,T_pi} and {T_pi+1,...,2*T_pi} with covariance matrix cov_2_T_pi.

    Parameters
    ----------
    cov_2_T_pi : np.ndarray, shape (2*T_pi, 2*T_pi)
        Covariance matrix.

    Returns
    -------
    PI : float
        Mutual information in nats.
    """

    if cov_2_T_pi.shape[0] % 2 != 0:
        raise ValueError('cov_2_T_pi must have even shape')

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
        Projected cross covariances matrices.
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

    if len(cross_cov_mats) % 2 != 0:
        raise ValueError('number of cross covariance matrices provided must be even'
                         '(equal to joint window length)')

    if proj is not None:
        cross_cov_mats_proj = project_cross_cov_mats(cross_cov_mats, proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov_2_T_pi = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2_T_pi)

    return PI


def calc_block_toeplitz_logdets(cross_cov_mats, proj=None):
    """Calculates logdets which can be used to calculate predictive information or entropy
    for a spatiotemporal Gaussian process with T N-by-N cross-covariance matrices using
    the block-Toeplitz algorithm.

    Based on:
    Sowell, Fallaw. "A decomposition of block toeplitz matrices with applications
    to vector time series." 1989a). Unpublished manuscript (1989).

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
    lodgets : list
        T logdets.
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
            A[(ii - 1, ii - 1)] = torch.linalg.solve(vb[ii - 1].t(), D.t()).t()
            Ab[(ii - 1, ii - 1)] = torch.linalg.solve(v.t(), D).t()

            for kk in range(1, ii):
                A[(ii - 1, kk - 1)] = (A[(ii - 2, kk - 1)]
                                       - A[(ii - 1, ii - 1)].mm(Ab[(ii - 2, ii - kk - 1)]))
                Ab[(ii - 1, kk - 1)] = (Ab[(ii - 2, kk - 1)]
                                        - Ab[(ii - 1, ii - 1)].mm(A[(ii - 2, ii - kk - 1)]))

            if ii < T - 1:
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
                if ii < T - 1:
                    A[(ii - 1, kk - 1)] = (A[(ii - 2, kk - 1)]
                                           - A[(ii - 1, ii - 1)].dot(Ab[(ii - 2, ii - kk - 1)]))
                Ab[(ii - 1, kk - 1)] = (Ab[(ii - 2, kk - 1)]
                                        - Ab[(ii - 1, ii - 1)].dot(A[(ii - 2, ii - kk - 1)]))
            if ii < T - 1:
                v = ccms[0] - sum([A[(ii - 1, jj - 1)].dot(ccms[jj].T) for jj in range(1, ii + 1)])
            vb[ii] = ccms[0] - sum([Ab[(ii - 1, jj - 1)].dot(ccms[jj]) for jj in range(1, ii + 1)])
        logdets = [np.linalg.slogdet(vb[ii])[1] for ii in range(T)]
    return logdets


def calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with T-1 N-by-N cross-covariance matrices using the block-Toeplitz
    algorithm.

    Based on:
    Sowell, Fallaw. "A decomposition of block toeplitz matrices with applications
    to vector time series." 1989a). Unpublished manuscript (1989).

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
    T = cross_cov_mats.shape[0]
    logdets = calc_block_toeplitz_logdets(cross_cov_mats, proj)
    return sum(logdets[:T // 2]) - 0.5 * sum(logdets)


def extract_diag_blocks(cov, Q):
    """Extract the diagonal blocks from a matrix.

    Parameters
    ----------
    cov : ndarray (Q*P, Q*P)
        Matrix to extract diagonal blocks from.
    Q : int
        The number of blocks.
    """
    P = cov.shape[0] // Q
    blocks = []
    start = 0
    for ii in range(Q):
        start = ii * P
        blocks.append(cov[start:start + P, start:start + P])
    return blocks


def one_step(S_X, R_X, R_Y, A, A_R_Y, Q):
    """Perform one EM step for ML block toeplitz covariance estimation.

    Parameters
    ----------
    S_X : ndarray
        Sample covariance matrix.
    R_X : ndarray
        Observed covariance matrix.
    R_Y : ndarray
        Latent covariance matrix.
    A : ndarray
        Projection from latent to observed variables.
    A_R_Y : ndarray
        Precomputed A @ R_Y.
    """
    R_X_inv_A_R_Y = np.linalg.solve(R_X, A_R_Y)
    R_Y = (R_Y + A_R_Y.T.conj() @ ((np.linalg.solve(R_X, S_X) @ R_X_inv_A_R_Y) - R_X_inv_A_R_Y))
    blocks = extract_diag_blocks(R_Y, Q)
    R_Y = sp.linalg.block_diag(*blocks)
    A_R_Y = A @ R_Y
    R_X = (A_R_Y @ A.T.conj()).real
    return R_X, R_Y, A_R_Y


def block_toeplitz_covariance(X, S_X, T, max_iter=100, tol=1e-6):
    """Estimate the ML block toeplitz covariance matrix using:

    Fuhrmann, Daniel R., and T. A. Barton.
    "Estimation of block-Toeplitz covariance matrices."
    1990 Conference Record Twenty-Fourth Asilomar Conference on Signals, Systems and Computers.

    Parameters
    ----------
    X : ndarray (time, features) or (batches, time, features)
        Data used to calculate the PI. Can be None if S_X is given.
    S_X : ndarray
        Sample covariance matrix. Can be None if X is given.
    T : int
        This T should be 2 * T_pi. This T sets the joint window length not the
        past or future window length.
    max_iter : int
        Maximum number of EM iterations.
    tol : int
        LL tolerance for stopping EM iterations.
    """
    N = T
    if X is not None:
        P = X.shape[1]
        X_N = form_lag_matrix(X, N)
        S_X = np.cov(X_N.T)
    else:
        P = S_X.shape[0] // N

    rectify_spectrum(S_X)
    Q = 4 * N
    NP = N * P
    QP = Q * P

    I_NP = np.eye(NP)
    I_P = np.eye(P)
    W_Q = sp.linalg.dft(Q, scale='sqrtn')
    I_NP_0 = np.concatenate([I_NP, np.zeros((NP, QP - NP))], axis=1)
    W_Q_I_P = np.kron(W_Q, I_P)
    A = I_NP_0 @ W_Q_I_P

    R_Y = np.eye(QP)
    A_R_Y = A @ R_Y
    R_X = np.eye(NP)
    if X is not None:
        ll = sp.stats.multivariate_normal.logpdf(X_N, mean=np.zeros(NP), cov=R_X).mean()
    else:
        d = S_X.shape[0]
        tr = np.trace(np.linalg.solve(R_X, S_X))
        logdets = np.linalg.slogdet(R_X)[1] - np.linalg.slogdet(S_X)[1]
        ll = -.5 * (tr + logdets - d)

    for ii in range(max_iter):
        R_X, R_Y, A_R_Y = one_step(S_X, R_X, R_Y, A, A_R_Y, Q)
        if X is not None:
            new_ll = sp.stats.multivariate_normal.logpdf(X_N, mean=np.zeros(NP), cov=R_X).mean()
        else:
            d = S_X.shape[0]
            tr = np.trace(np.linalg.solve(R_X, S_X))
            logdets = np.linalg.slogdet(R_X)[1] - np.linalg.slogdet(S_X)[1]
            new_ll = -.5 * (tr + logdets - d)
        if abs(new_ll - ll) / max([1., abs(new_ll), abs(ll)]) < tol:
            break
    return R_X


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
        if not isinstance(args, collections.abc.Hashable):
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
