import numpy as np
import scipy.stats

from .cov_estimation import calc_cross_cov_mats_from_data

import torch

__all__ = ['ComplexityComponentsAnalysis']


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

    cross_cov_mats_repeated = []
    for i in range(num_lags):
        for j in range(num_lags):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)])
            else:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].t())

    cov_tensor = torch.reshape(torch.stack(cross_cov_mats_repeated),
                               (num_lags, num_lags, N, N))
    cov = torch.cat([torch.cat([cov_ii_jj for cov_ii_jj in cov_ii], dim=1)
                     for cov_ii in cov_tensor])

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

    cov_T = cov_2T[:half, :half]
    logdet_T = torch.logdet(cov_T)
    logdet_2T = torch.logdet(cov_2T)
    PI = (2. * logdet_T - logdet_2T) / np.log(2.)

    return PI


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None,
                                device='cuda:0', dtype=torch.float64):
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
    if not isinstance(cross_cov_mats, torch.Tensor):
        cross_cov_mats = torch.tensor(cross_cov_mats, device=device,
                                      dtype=dtype)
    if proj is not None:
        if not isinstance(proj, torch.Tensor):
            proj = torch.tensor(proj, device=device, dtype=dtype)

    T = cross_cov_mats.shape[0] // 2
    if isinstance(proj, torch.Tensor):
        d = proj.shape[1]
    else:
        d = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]

    cross_cov_mats_proj = []
    if isinstance(proj, torch.Tensor):
        for i in range(2*T):
            cross_cov = cross_cov_mats[i]
            cross_cov_proj = torch.mm(proj.t(), torch.mm(cross_cov, proj))
            cross_cov_mats_proj.append(cross_cov_proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cov_2T = calc_cov_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov(cov_2T)

    return PI


def ortho_reg_fn(V, ortho_lambda, device='cuda:0', dtype=torch.float64):
    """Regularization term which encourages the basis vectors in the
    columns of V to be orthonormal.
    Parameters
    ----------
    V : np.ndarray, shape (N, d)
        Matrix whose columns are basis vectors.
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    reg_val : float
        Value of regularization function.
    """

    d = V.shape[1]
    reg_val = ortho_lambda * torch.sum((torch.mm(V.t(), V) -
                                        torch.eye(d, device=device,
                                                  dtype=dtype))**2)
    return reg_val


def build_loss(cross_cov_mats, d, lambda_param=10,
               device='cuda:0', dtype=torch.float64):
    """Constructs a loss function which gives the (negative) predictive
    information in the projection of multidimensional timeseries data X onto a
    d-dimensional basis, where predictive information is computed using a
    stationary Gaussian process approximation.

    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        mutual information is computed.
    d: int
        Number of basis vectors onto which the data X are projected.
    lambda_param : float
        Regularization hyperparameter.
    Returns
    -------
    loss : function
       Loss function which accepts a (flattened) N-by-d matrix, whose
       columns are basis vectors, and outputs the negative predictive information
       corresponding to that projection (plus regularization term).
    """

    N = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]
    if not isinstance(cross_cov_mats, torch.Tensor):
        cross_cov_mats = torch.tensor(cross_cov_mats, device=device,
                                      dtype=dtype)
    def loss(V_flat):
        if not isinstance(V_flat, torch.Tensor):
            V_flat = torch.tensor(V_flat, device=device, dtype=dtype)
        V = V_flat.reshape(N, d)
        reg_val = ortho_reg_fn(V, lambda_param, device=device, dtype=dtype)
        return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss


class ComplexityComponentsAnalysis(object):
    """Complexity Components Analysis.

    Runs CCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.
    Parameters
    ----------
    d: int
        Number of basis vectors onto which the data X are projected.
    T: int
        Size of time windows accross which to compute mutual information.
    init: string
        Options: "random", "PCA"
        Method for initializing the projection matrix.

    """
    def __init__(self, d=None, T=None, init="random", tol=1e-6, ortho_lambda=10.,
                 verbose=False, device="cuda:0", dtype=torch.float64):
        self.d = d
        self.T = T
        self.init = init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose=verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None

    def estimate_cross_covariance(self, X, T=None, regularization=None,
                                  reg_ops=None):
        if T is None:
            T = self.T
        else:
            self.T = T
        self.cross_covs = calc_cross_cov_mats_from_data(X, T,
                                                        regularization=regularization,
                                                        reg_ops=reg_ops)
        return self

    def fit_projection(self, X, d=None):
        if d is None:
            d = self.d
        if self.cross_covs is None:
            raise ValueError('Call estimate_cross_covariance() first.')

        N = self.cross_covs.shape[1]
        if type(self.init) == str:
            if self.init == "random":
                V_init = np.random.normal(0, 1, (N, d))
            elif self.init == "random_ortho":
                V_init = scipy.stats.ortho_group.rvs(N)[:, :d]
            elif self.init == "uniform":
                V_init = np.ones((N, d)) / np.sqrt(N)
                V_init = V_init + np.random.normal(0, 1e-3, V_init.shape)
            else:
                raise ValueError
        else:
            raise ValueError
        V_init /= np.linalg.norm(V_init, axis=0, keepdims=True)

        v = torch.tensor(V_init, requires_grad=True,
                         device=self.device, dtype=self.dtype)
        c = torch.tensor(self.cross_covs, device=self.device, dtype=self.dtype)

        optimizer = torch.optim.LBFGS([v], max_eval=15000, max_iter=15000)

        def closure():
            optimizer.zero_grad()
            loss = build_loss(c, d, device=self.device, dtype=self.dtype)(v)
            loss.backward()
            if self.verbose:
                reg_val = ortho_reg_fn(v, self.ortho_lambda,
                                       device=self.device, dtype=self.dtype)
                loss_no_reg = loss - reg_val
                pi = -loss_no_reg.detach().cpu().numpy()
                reg_val = reg_val.detach().cpu().numpy()
                print("PI: {} bits, reg: {}".format(str(np.round(pi, 4)),
                                                    str(np.round(reg_val, 4))))
            return loss

        optimizer.step(closure)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v.detach().cpu().numpy())
        self.coef_ = V_opt

        return self

    def fit(self, X, d=None, T=None, regularization=None, reg_ops=None):
        self.estimate_cross_covariance(X, T=T, regularization=regularization,
                                       reg_ops=reg_ops)
        self.fit_projection(X, d=d)
        return self

    def transform(self, X):
        return X.dot(self.coef_)

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None):
        self.fit(X, d=d, T=T, regularization=regularization, reg_ops=reg_ops)
        return self.transform(X)

    def score(self):
        return calc_pi_from_cross_cov_mats(self.cross_covs, self.coef_,
                                           device=self.device, dtype=self.dtype)
