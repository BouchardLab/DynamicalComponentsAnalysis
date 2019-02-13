import numpy as np
import scipy as sp

from .kron_pca import prox_grad_robust_toeplitz_kron_pca

def calc_cross_cov_mats_from_data(X, num_lags, regularization=None, reg_ops=None):
    """Calculates N-by-N cross-covariance matrices for time lags
    up to num_lags - 1, where N is the dimensionality of the time
    series data. The data X are mean-centered prior to the computation.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        cross-covariance matrices are computed.
    num_lags: int
        The number of time-lags.
    regularization : string
        Regularization method for computing cross-covariance matrices.
        Options:
            'Abadir' : Method from 'Design-free estimation of variance
            matrices' (Abadir et al., 2014). We implement the 'Grand Average'
            estimator (see Eq. 11).
    reg_ops : dict
        Paramters for regularization.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N), float
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    """

    #mean-center data
    X = X - np.mean(X, axis=0)
    N = X.shape[1]

    if regularization is None:

        #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=num_lags-1
        cross_cov_mats = np.zeros((num_lags, N, N))
        for delta_t in range(num_lags):
            cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
            cross_cov_mats[delta_t] = cross_cov

        cov_est = calc_cov_from_cross_cov_mats(cross_cov_mats)

    elif regularization == "Abadir":

        #reg_ops:
        #M -- number of m values, uniformly spaced over [0.2, 0.8]
        #S -- number samples for each m
        #skip -- spacing of consecutive samples from series data
                #(see construction of 'X_with_lags')
        M = reg_ops["M"]
        S = reg_ops["S"]
        skip = reg_ops["skip"]

        n = int(np.floor((len(X)-num_lags+1)/skip))

        X_with_lags = np.zeros((n, N*num_lags))
        for i in range(n):
            X_with_lags[i, :] = X[i*skip:i*skip+num_lags].flatten()
        X_with_lags -= X_with_lags.mean(axis=0, keepdims=True)

        #Holds all M*S cov estimates
        results = np.zeros((M, S, N*num_lags, N*num_lags))

        m_vals = np.round(np.linspace(0.2, 0.8,  M)*n)
        idx = np.arange(n)

        #diagonalize all data
        #Note that this code uses 'V' where the paper uses 'P'
        cov = np.dot(X_with_lags.T, X_with_lags)/(n - 1)
        w, V = sp.linalg.eigh(cov)
        w, V = w[::-1], V[:, ::-1]

        for m_idx in range(M):
            m = int(m_vals[m_idx])
            for sample_idx in range(S):

                #seperate data into groups 1 and 2
                idx_1 = np.random.choice(idx, size=m, replace=False)
                idx_2 = np.setdiff1d(idx, idx_1, assume_unique=True)
                X_1, X_2 = X_with_lags[idx_1, :], X_with_lags[idx_2, :]

                #mean-center both
                X_1 = X_1 - np.mean(X_1, axis=0)
                X_2 = X_2 - np.mean(X_2, axis=0)

                #diagonalize X_1
                cov_1 = np.dot(X_1.T, X_1)/(m - 1)
                w_1, V_1 = sp.linalg.eigh(cov_1)
                w_1, V_1 = w_1[::-1], V_1[:, ::-1]

                #project X_2 onto eigenvectors of X_1
                proj_X_2 = np.dot(X_2, V_1)

                #estimate spectrum based on variance of projection
                lambda_est = np.var(proj_X_2, axis=0)

                #form the covariance estimate
                cov_est = np.dot(V, np.dot(np.diag(lambda_est), V.T))

                #store the result
                results[m_idx, sample_idx, :, :] = cov_est

        cov_est = np.mean(results, axis=(0, 1))

        #Note final_cov_est will not be stationary, since stationarity
        #is not preserved by the regularization technique.
        #Thus, we average over the cross-covariance sub-matrices for
        #constant |t1-t2|
        cross_cov_mats = calc_cross_cov_mats_from_cov(N, num_lags, final_cov_est)
        cov_est = calc_cov_from_cross_cov_mats(cross_cov_mats)

    elif regularization == 'kron':
        #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=num_lags-1
        cross_cov_mats = np.zeros((num_lags, N, N))
        for delta_t in range(num_lags):
            cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
            cross_cov_mats[delta_t] = cross_cov

        cov = calc_cov_from_cross_cov_mats(cross_cov_mats)
        cov_est = prox_grad_robust_toeplitz_kron_pca(cov, N, num_lags,
                                                 reg_ops['lambda_L'],
                                                 reg_ops['lambda_S'],
                                                 reg_ops['num_iter'],
                                                 reg_ops['tau'])
        cross_cov_mats = calc_cross_cov_mats_from_cov(N, num_lags, cov_est)

    w, _ = np.linalg.eigh(cov_est)
    min_eig = np.min(w)
    if min_eig <= 0:
        cross_cov_mats[0] += np.eye(N)*(1e-8 - min_eig)
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

    cross_cov_mats = np.zeros((num_lags, N, N))
    for delta_t in range(num_lags):
        to_avg_lower = np.zeros((num_lags-delta_t, N, N))
        to_avg_upper = np.zeros((num_lags-delta_t, N, N))
        for i in range(num_lags-delta_t):
            i_offset = delta_t*N
            to_avg_lower[i, :, :] = cov[i_offset+i*N:i_offset+(i+1)*N, i*N:(i+1)*N]
            to_avg_upper[i, :, :] = cov[i*N:(i+1)*N, i_offset+i*N:i_offset+(i+1)*N]
        cross_cov_mats[delta_t, :, :] = 0.5*(np.mean(to_avg_lower, axis=0) + np.mean(to_avg_upper, axis=0))
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

    N = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]
    num_lags = len(cross_cov_mats)

    cross_cov_mats_repeated = []
    for i in range(num_lags):
        for j in range(num_lags):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)])
            else:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].T)

    cov_tensor = np.reshape(np.array(cross_cov_mats_repeated), (num_lags, num_lags, N, N))
    cov = np.concatenate([np.concatenate(cov_ii, axis=1) for cov_ii in cov_tensor])

    return cov
