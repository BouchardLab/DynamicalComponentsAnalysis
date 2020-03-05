import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.nn.functional as F

from .cov_util import (calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats,
                       form_lag_matrix, calc_pi_from_cross_cov_mats_block_toeplitz)

__all__ = ['DynamicalComponentsAnalysis',
           'DynamicalComponentsAnalysisFFT',
           'DynamicalComponentsAnalysisKNN',
           'ortho_reg_fn',
           'build_loss',
           'init_coef']


def ortho_reg_fn(V, ortho_lambda):
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

    use_torch = isinstance(V, torch.Tensor)
    d = V.shape[1]

    if use_torch:
        reg_val = ortho_lambda * torch.sum((torch.mm(V.t(), V) -
                                            torch.eye(d, device=V.device, dtype=V.dtype))**2)
    else:
        reg_val = ortho_lambda * np.sum((np.dot(V.T, V) - np.eye(d))**2)

    return reg_val


def build_loss(cross_cov_mats, d, ortho_lambda=1., block_toeplitz=False):
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
    ortho_lambda : float
        Regularization hyperparameter.
    Returns
    -------
    loss : function
       Loss function which accepts a (flattened) N-by-d matrix, whose
       columns are basis vectors, and outputs the negative predictive information
       corresponding to that projection (plus regularization term).
    """
    N = cross_cov_mats.shape[1]

    if block_toeplitz:
        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(V, ortho_lambda)
            return -calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, V) + reg_val
    else:
        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(V, ortho_lambda)
            return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss


class ObjectiveWrapper(object):
    def __init__(self, f_params):
        self.common_computations = None
        self.params = None
        self.f_params = f_params
        self.n_f = 0
        self.n_g = 0

    def core_computations(self, *args):
        params = args[0]
        if not np.array_equal(params, self.params):
            self.common_computations = self.f_params(*args)
            self.params = params.copy()
        return self.common_computations

    def func(self, *args):
        self.n_f += 1
        loss, _ = self.core_computations(*args)
        return loss.detach().cpu().numpy().astype(float)

    def grad(self, *args):
        self.n_g += 1
        loss, params_torch = self.core_computations(*args)
        loss.backward(retain_graph=True)
        grad = params_torch.grad
        return grad.detach().cpu().numpy().astype(float)


def init_coef(N, d, rng, init):
    """Initialize a projection coefficent matrix.

    Parameters
    ----------
    N : int
        Original dimensionality.
    d : int
        Projected dimensionality.
    rng : np.random.RandomState
        Random state for generation.
    init : str or ndarray
        Initialization type.
    """
    if type(init) == str:
        if init == "random":
            V_init = rng.normal(0, 1, (N, d))
        elif init == "random_ortho":
            V_init = scipy.stats.ortho_group.rvs(N, random_state=rng)[:, :d]
        elif init == "uniform":
            V_init = np.ones((N, d)) / np.sqrt(N)
            V_init = V_init + rng.normal(0, 1e-3, V_init.shape)
        else:
            raise ValueError
    elif isinstance(init, np.ndarray):
        V_init = init.copy()
    else:
        raise ValueError
    V_init /= np.linalg.norm(V_init, axis=0, keepdims=True)
    return V_init


class DynamicalComponentsAnalysis(object):
    """Dynamical Components Analysis.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information.
    init : str
        Options: "random_ortho", "random", or "PCA"
        Method for initializing the projection matrix.
    n_init : int
        Number of random restarts. Default is 1.
    tol : float
        Tolerance for stopping optimization. Default is 1e-6.
    ortho_lambda : float
        Coefficient on term that keeps V close to orthonormal.
    verbose : bool
        Verbosity during optimization.
    use_scipy : bool
        Whether to use SciPy or Pytorch L-BFGS-B. Default is True. Pytorch is not well tested.
    block_toeplitz : bool
        If True, uses the block-Toeplitz logdet algorithm which is typically faster and less
        memory intensive on cpu for T>~10.
    device : str
        What device to run the computation on in Pytorch.
    dtype : pytorch.dtype
        What dtype to use for computation.
    """
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False, use_scipy=True, block_toeplitz=None,
                 device="cpu", dtype=torch.float64, rng_or_seed=None):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.use_scipy = use_scipy
        if block_toeplitz is None:
            try:
                if d > 40 and T > 8:
                    self.block_toeplitz = True
                else:
                    self.block_toeplitz = False
            except TypeError:
                self.block_toeplitz = False
        else:
            self.block_toeplitz = block_toeplitz
        self.cross_covs = None
        if rng_or_seed is None:
            self.rng = np.random
        elif isinstance(rng_or_seed, np.random.RandomState):
            self.rng = rng_or_seed
        else:
            self.rng = np.random.RandomState(rng_or_seed)

    def estimate_cross_covariance(self, X, T=None, regularization=None, reg_ops=None):
        """Estimate the cross covariance matrix from data.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        T : int
            T for PI calculation (optional.)
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        """
        if T is None:
            T = self.T
        else:
            self.T = T

        cross_covs = calc_cross_cov_mats_from_data(X, 2 * self.T,
                                                   chunks=10,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops)
        self.cross_covs = torch.tensor(cross_covs, device=self.device, dtype=self.dtype)

        return self

    def fit_projection(self, d=None, n_init=None):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        pis = []
        coefs = []
        for ii in range(n_init):
            coef, pi = self._fit_projection(d=d)
            pis.append(pi)
            coefs.append(coef)
        idx = np.argmax(pis)
        self.coef_ = coefs[idx]

    def _fit_projection(self, d=None, record_V=False):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        record_V : bool
            If True, saves a copy of V at each optimization step. Default is False.
        """
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError
        if self.cross_covs is None:
            raise ValueError('Call estimate_cross_covariance() first.')

        N = self.cross_covs.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)
        v = torch.tensor(V_init, requires_grad=True,
                         device=self.device, dtype=self.dtype)

        c = self.cross_covs
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device, dtype=self.dtype)

        if self.use_scipy:
            if self.verbose or record_V:
                if record_V:
                    self.V_seq = [V_init]

                def callback(v_flat):
                    v_flat_torch = torch.tensor(v_flat,
                                                requires_grad=True,
                                                device=self.device,
                                                dtype=self.dtype)
                    v_torch = v_flat_torch.reshape(N, d)
                    loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
                    reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
                    loss = loss.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    if record_V:
                        self.V_seq.append(v_flat.reshape(N, d))
                    if self.verbose:
                        print("PI: {} nats, reg: {}".format(str(np.round(-loss, 4)),
                                                            str(np.round(reg_val, 4))))

                callback(V_init)
            else:
                callback = None

            """
            def f_df(v_flat):
                v_flat_torch = torch.tensor(v_flat,
                                            requires_grad=True,
                                            device=self.device,
                                            dtype=self.dtype)
                v_torch = v_flat_torch.reshape(N, d)
                loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
                loss.backward()
                grad = v_flat_torch.grad
                return (loss.detach().cpu().numpy().astype(float),
                        grad.detach().cpu().numpy().astype(float))
            opt = minimize(f_df, V_init.ravel(), method='L-BFGS-B', jac=True,
                           options={'disp': self.verbose, 'ftol': self.tol},
                           callback=callback)
            """
            def f_params(v_flat):
                v_flat_torch = torch.tensor(v_flat,
                                            requires_grad=True,
                                            device=self.device,
                                            dtype=self.dtype)
                v_torch = v_flat_torch.reshape(N, d)
                loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
                return loss, v_flat_torch
            objective = ObjectiveWrapper(f_params)
            opt = minimize(objective.func, V_init.ravel(), method='L-BFGS-B', jac=objective.grad,
                           options={'disp': self.verbose, 'ftol': self.tol},
                           callback=callback)
            v = opt.x.reshape(N, d)
        else:
            optimizer = torch.optim.LBFGS([v], max_eval=15000, max_iter=15000,
                                          tolerance_change=self.tol, history_size=10,
                                          line_search_fn='strong_wolfe')

            def closure():
                optimizer.zero_grad()
                loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v)
                loss.backward()
                if self.verbose:
                    reg_val = ortho_reg_fn(v, self.ortho_lambda)
                    loss_no_reg = loss - reg_val
                    pi = -loss_no_reg.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    print("PI: {} nats, reg: {}".format(str(np.round(pi, 4)),
                                                        str(np.round(reg_val, 4))))
                return loss

            optimizer.step(closure)
            v = v.detach().cpu().numpy()

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        final_pi = calc_pi_from_cross_cov_mats(c, V_opt).detach().cpu().numpy()
        return V_opt, final_pi

    def fit(self, X, d=None, T=None, regularization=None, reg_ops=None, n_init=None):
        """Estimate the cross covariance matrix and fit the projection matrix.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional.)
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        n_init : int
            Number of random restarts (optional.)
        """
        self.estimate_cross_covariance(X, T=T, regularization=regularization,
                                       reg_ops=reg_ops)
        self.fit_projection(d=d, n_init=n_init)
        return self

    def transform(self, X):
        """Project the data onto the DCA components.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        """
        if isinstance(X, list):
            y = [(Xi - Xi.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X]
        elif X.ndim == 3:
            y = np.stack([(Xi - Xi.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X])
        else:
            y = (X - X.mean(axis=0, keepdims=True)).dot(self.coef_)
        return y

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None, n_init=None):
        """Estimate the cross covariance matrix and fit the projection matrix. Then
        project the data onto the DCA components.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional.)
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        n_init : int
            Number of random restarts (optional.)
        """
        self.fit(X, d=d, T=T, regularization=regularization, reg_ops=reg_ops, n_init=n_init)
        return self.transform(X)

    def score(self, X=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        if X is None:
            cross_covs = self.cross_covs
        else:
            cross_covs = calc_cross_cov_mats_from_data(X, T=self.T)
        if self.block_toeplitz:
            return calc_pi_from_cross_cov_mats_block_toeplitz(cross_covs, self.coef_)
        else:
            return calc_pi_from_cross_cov_mats(cross_covs, self.coef_)


def make_cepts2(X, T_pi):
    """Calculate the squared real cepstral coefficents."""
    Y = F.unfold(X, kernel_size=[T_pi, 1], stride=T_pi)
    Y = torch.transpose(Y, 1, 2)

    # Compute the power spectral density
    window = torch.Tensor(hann(Y.shape[-1])[np.newaxis, np.newaxis]).type(Y.dtype)
    Yf = torch.rfft(Y * window, 1, onesided=True)
    spect = Yf[:, :, :, 0]**2 + Yf[:, :, :, 1]**2
    spect = spect.mean(dim=1)
    spect = torch.cat([torch.flip(spect[:, 1:], dims=(1,)), spect], dim=1)

    # Log of the DFT of the autocorrelation
    logspect = torch.log(spect) - np.log(float(Y.shape[-1]))

    # Compute squared cepstral coefs (b_k^2)
    cepts = torch.rfft(logspect, 1, onesided=True) / float(Y.shape[-1])
    cepts = torch.sqrt(cepts[:, :, 0]**2 + cepts[:, :, 1]**2)
    return cepts**2


def pi_fft(X, proj, T_pi):
    """
    This is well-tested when X has shape (# time steps, 1).
    Otherwise, behavior has not been considered.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if not isinstance(proj, torch.Tensor):
        proj = torch.Tensor(proj)
    X = torch.mm(X, proj)
    Xp_tensor = X.t()
    Xp_tensor = torch.unsqueeze(Xp_tensor, -1)
    Xp_tensor = torch.unsqueeze(Xp_tensor, 1)
    bs2 = make_cepts2(Xp_tensor, T_pi)
    ks = torch.arange(bs2.shape[-1], dtype=bs2.dtype)
    return .5 * (torch.unsqueeze(ks, 0) * bs2).sum(dim=1).sum()


class DynamicalComponentsAnalysisFFT(object):
    """Dynamical Components Analysis using FFT for PI calculation.

    Currently only well-defined for `d=1`.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the dynamical complexity.

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information.
    init : string
        Options: "random", "PCA"
        Method for initializing the projection matrix.
    """
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False,
                 device="cpu", dtype=torch.float64, rng_or_seed=None):
        self.d = d
        if d > 1:
            raise ValueError('DCAFFT is only defined for d=1.')
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None
        if rng_or_seed is None:
            self.rng = np.random
        elif isinstance(rng_or_seed, np.random.RandomState):
            self.rng = rng_or_seed
        else:
            self.rng = np.random.RandomState(rng_or_seed)

    def fit(self, X, d=None, T=None, n_init=None):
        self.mean_ = X.mean(axis=0, keepdims=True)
        X = X - self.mean_
        if n_init is None:
            n_init = self.n_init
        pis = []
        coefs = []
        for ii in range(n_init):
            coef, pi = self._fit_projection(X, d=d)
            pis.append(pi)
            coefs.append(coef)
        idx = np.argmin(pis)
        self.coef_ = coefs[idx]

    def _fit_projection(self, X, d=None):
        if d is None:
            d = self.d
        if d > 1:
            raise ValueError('DCAFFT is only defined for d=1.')

        N = X.shape[1]
        if type(self.init) == str:
            if self.init == "random":
                V_init = self.rng.normal(0, 1, (N, d))
            elif self.init == "random_ortho":
                V_init = scipy.stats.ortho_group.rvs(N, random_state=self.rng)[:, :d]
            elif self.init == "uniform":
                V_init = np.ones((N, d)) / np.sqrt(N)
                V_init = V_init + self.rng.normal(0, 1e-3, V_init.shape)
            else:
                raise ValueError
        else:
            raise ValueError
        V_init /= np.linalg.norm(V_init, axis=0, keepdims=True)

        v = torch.tensor(V_init, requires_grad=True,
                         device=self.device, dtype=self.dtype)

        Xt = X
        if not isinstance(Xt, torch.Tensor):
            Xt = torch.tensor(Xt, device=self.device, dtype=self.dtype)

        if self.verbose:

            def callback(v_flat):
                v_flat_torch = torch.tensor(v_flat,
                                            requires_grad=True,
                                            device=self.device,
                                            dtype=self.dtype)
                v_torch = v_flat_torch.reshape(N, d)
                pi = pi_fft(Xt, v_torch, self.T)
                reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
                pi = pi.detach().cpu().numpy()
                reg_val = reg_val.detach().cpu().numpy()
                print("PI: {} nats, reg: {}".format(str(np.round(pi, 4)),
                                                    str(np.round(reg_val, 4))))
            callback(V_init)
        else:
            callback = None

        def f_df(v_flat):
            v_flat_torch = torch.tensor(v_flat,
                                        requires_grad=True,
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            pi = pi_fft(Xt, v_torch, self.T)
            reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
            loss = -pi + reg_val
            loss.backward()
            grad = v_flat_torch.grad
            return (loss.detach().cpu().numpy().astype(float),
                    grad.detach().cpu().numpy().astype(float))
        opt = minimize(f_df, V_init.ravel(), method='L-BFGS-B', jac=True,
                       options={'disp': self.verbose, 'ftol': self.tol},
                       callback=callback)
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        v_flat_torch = torch.tensor(V_opt.ravel(),
                                    requires_grad=True,
                                    device=self.device,
                                    dtype=self.dtype)
        v_torch = v_flat_torch.reshape(N, d)
        final_pi = pi_fft(Xt, v_torch, self.T).detach().cpu().numpy()
        return V_opt, final_pi

    def transform(self, X):
        X = X - self.mean_
        return X.dot(self.coef_)

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None):
        self.fit(X, d=d, T=T)
        return self.transform(X)

    def score(self, X):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
        """
        return pi_fft(X, self.coef_, self.T)


class DynamicalComponentsAnalysisKNN(object):
    """Dynamical Components Analysis using MI estimation using the
    k-nearest neighbors methos of Kraskov, et al. This estimator is not
    differentiable and so numerical gradients are taken (very slow!).

    WARNING: This code has not been used or tested and is still being developed.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information.
    init : string
        Options: "random", "PCA"
        Method for initializing the projection matrix.
    """
    def __init__(self, d=None, T=None, init="random", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False, use_scipy=True):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose = verbose
        self.coef_ = None

    def fit(self, X, d=None, T=None, n_init=None):
        self.mean_ = X.mean(axis=0, keepdims=True)
        X -= self.mean_
        if n_init is None:
            n_init = self.n_init
        pis = []
        coefs = []
        for ii in range(n_init):
            coef, pi = self._fit_projection(X, d=d)
            pis.append(pi)
            coefs.append(coef)
        idx = np.argmax(pis)
        self.coef_ = coefs[idx]
        return self

    def _fit_projection(self, X, d=None):
        from info_measures.continuous import kraskov_stoegbauer_grassberger as ksg
        if d is None:
            d = self.d

        N = X.shape[1]
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

        callback = None
        if self.verbose:

            def callback(v_flat):
                v = v_flat.reshape(N, d)
                X_lag = form_lag_matrix(X.dot(v), 2 * self.T)
                mi = ksg.MutualInformation(X_lag[:, :self.T * d],
                                           X_lag[:, self.T * d:])
                pi = mi.mutual_information()
                reg_val = ortho_reg_fn(v, self.ortho_lambda)
                print("PI: {} bits, reg: {}".format(str(np.round(pi, 4)),
                                                    str(np.round(reg_val, 4))))
            callback(V_init)

        def f(v_flat):
            v = v_flat.reshape(N, d)
            X_lag = form_lag_matrix(X.dot(v), 2 * self.T)
            mi = ksg.MutualInformation(X_lag[:, :self.T], X_lag[:, self.T:])
            pi = mi.mutual_information()
            reg_val = ortho_reg_fn(v, self.ortho_lambda)
            loss = -pi + reg_val
            return loss
        opt = minimize(f, V_init.ravel(), method='L-BFGS-B', callback=callback)
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        final_pi = self.score(X, V_opt)
        return V_opt, final_pi

    def transform(self, X):
        return (X - self.mean_).dot(self.coef_)

    def fit_transform(self, X, d=None, T=None):
        self.fit(X, d=d, T=T)
        return self.transform(X)

    def score(self, X, coef=None):
        if coef is None:
            coef = self.coef_
        from info_measures.continuous import kraskov_stoegbauer_grassberger as ksg
        X_lag = form_lag_matrix(X.dot(coef), 2 * self.T)
        mi = ksg.MutualInformation(X_lag[:, :self.T], X_lag[:, self.T:])
        pi = mi.mutual_information()
        return pi
