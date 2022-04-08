import logging, time
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.fft
import torch.nn.functional as F

from .base import SingleProjectionComponentsAnalysis, ortho_reg_fn, init_coef, ObjectiveWrapper
from .cov_util import (calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats,
                       calc_pi_from_cross_cov_mats_block_toeplitz)

__all__ = ['DynamicalComponentsAnalysis',
           'DynamicalComponentsAnalysisFFT',
           'build_loss']


logging.basicConfig()


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
            reg_val = ortho_reg_fn(ortho_lambda, V)
            return -calc_pi_from_cross_cov_mats_block_toeplitz(cross_cov_mats, V) + reg_val
    else:
        def loss(V_flat):
            V = V_flat.reshape(N, d)
            reg_val = ortho_reg_fn(ortho_lambda, V)
            return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss


class DynamicalComponentsAnalysis(SingleProjectionComponentsAnalysis):
    """Dynamical Components Analysis.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace of an N-dimensional space which maximizes the complexity, as
    defined by the Gaussian Predictive Information (PI) of the d-dimensional dynamics over windows
    of length T.

    Parameters
    ----------
    d : int
        Number of basis vectors onto which the data X are projected.
    T : int
        Size of time windows across which to compute mutual information. Total window length will be
        `2 * T`. When fitting a model, the length of the shortest timeseries must be greater than
        `2 * T` and for good performance should be much greater than `2 * T`.
    init : str
        Options: "random_ortho", "random", or "PCA"
        Method for initializing the projection matrix.
    n_init : int
        Number of random restarts. Default is 1.
    stride : int
        Number of samples to skip when estimating cross covariance matrices. Settings stride > 1
        will speedup covariance estimation but may reduce the quality of the covariance estimate
        for small datasets.
    chunk_cov_estimate : None or int
        If `None`, cov is estimated from entire time series. If an `int`, cov is estimated
        by chunking up time series and averaging covariances from chucks. This can use less memory
        and be faster for long timeseries. Requires that the length of the shortest timeseries
        in the batch is longer than `2 * T * chunk_cov_estimate`.
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
        memory intensive on cpu for `T >~ 10` and `d >~ 40`.
    method : str
        'toeplitzify' for naive averaging to compute the covariance, which is faster but less
        accurate for very small datasets. 'ML' for maximum likelihood block toeplitz covariance
        estimation which can be very slow for large datasets.
    device : str
        What device to run the computation on in Pytorch.
    dtype : pytorch.dtype
        What dtype to use for computation.
    rng_or_seed : None, int, or NumPy RandomState
        Random number generator or seed.

    Attributes
    ----------
    T : int
        Default T used for PI.
    T_fit : int
        T used for last cross covariance estimation.
    d : int
        Default d used for fitting the projection.
    d_fit : int
        d used for last projection fit.
    cross covs : torch tensor
        Cross covariance matrices from the last covariance estimation.
    coef_ : ndarray (N, d)
        Projection matrix from fit.
    """
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, stride=1,
                 chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
                 block_toeplitz=None, method='toeplitzify', device="cpu", dtype=torch.float64,
                 rng_or_seed=None):

        super(DynamicalComponentsAnalysis,
              self).__init__(d=d, T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol, verbose=verbose,
                             device=device, dtype=dtype, rng_or_seed=rng_or_seed)

        self.ortho_lambda = ortho_lambda
        if block_toeplitz is None:
            try:
                if d > 40 and T > 10:
                    self.block_toeplitz = True
                else:
                    self.block_toeplitz = False
            except TypeError:
                self.block_toeplitz = False
        else:
            self.block_toeplitz = block_toeplitz
        self.cross_covs = None
        self.method = method

    def estimate_data_statistics(self, X, T=None, regularization=None, reg_ops=None):
        """Estimate the cross covariance matrix from data.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        T : int
            T for PI calculation (optional).
        regularization : str
            Whether to regularize cross covariance estimation.
        reg_ops : dict
            Options for cross covariance regularization.
        """
        if T is None:
            T = self.T
        else:
            self.T = T
        start = time.time()
        self._logger.info('Starting cross covariance estimate.')
        if isinstance(X, list) or X.ndim == 3:
            self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        else:
            self.mean_ = X.mean(axis=0, keepdims=True)

        cross_covs = calc_cross_cov_mats_from_data(X, 2 * self.T, mean=self.mean_,
                                                   chunks=self.chunk_cov_estimate,
                                                   stride=self.stride,
                                                   rng=self.rng,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops,
                                                   logger=self._logger,
                                                   method=self.method)
        self.cross_covs = torch.tensor(cross_covs, device=self.device, dtype=self.dtype)
        delta_time = round((time.time() - start) / 60., 1)
        self._logger.info('Cross covariance estimate took {:0.1f} minutes.'.format(delta_time))

        return self

    def _fit_projection(self, d=None, T=None, record_V=False):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional). Default is `self.T`. If `T` is set here
            it must be less than or equal to `self.T` or self.estimate_cross_covariance() must
            be called with a larger `T`.
        record_V : bool
            If True, saves a copy of V at each optimization step. Default is False.
        """
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError
        self.d_fit = d
        if T is None:
            T = self.T
        if T < 1:
            raise ValueError
        if (2 * T) > self.cross_covs.shape[0]:
            raise ValueError('T must less than or equal to the value when ' +
                             '`estimate_data_statistics()` was called.')
        self.T_fit = T

        if self.cross_covs is None:
            raise ValueError('Call `estimate_cross_covariance()` first.')

        c = self.cross_covs[:2 * T]
        N = c.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)

        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device, dtype=self.dtype)

        def f_params(v_flat, requires_grad=True):
            v_flat_torch = torch.tensor(v_flat,
                                        requires_grad=requires_grad,
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
            return loss, v_flat_torch
        objective = ObjectiveWrapper(f_params)

        def null_callback(*args, **kwargs):
            pass

        if self.verbose or record_V:
            if record_V:
                self.V_seq = [V_init]

            def callback(v_flat, objective):
                if record_V:
                    self.V_seq.append(v_flat.reshape(N, d))
                if self.verbose:
                    loss, v_flat_torch = objective.core_computations(v_flat,
                                                                     requires_grad=False)
                    v_torch = v_flat_torch.reshape(N, d)
                    loss = build_loss(c, d, self.ortho_lambda, self.block_toeplitz)(v_torch)
                    reg_val = ortho_reg_fn(self.ortho_lambda, v_torch)
                    loss = loss.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    PI = -(loss - reg_val)
                    string = "Loss {}, PI: {} nats, reg: {}"
                    self._logger.info(string.format(str(np.round(loss, 4)),
                                                    str(np.round(PI, 4)),
                                                    str(np.round(reg_val, 4))))

            callback(V_init, objective)
        else:
            callback = null_callback

        opt = minimize(objective.func, V_init.ravel(), method='L-BFGS-B', jac=objective.grad,
                       options={'disp': self.verbose, 'ftol': self.tol},
                       callback=lambda x: callback(x, objective))
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        final_pi = calc_pi_from_cross_cov_mats(c, V_opt).detach().cpu().numpy()
        return V_opt, final_pi

    def score(self, X=None):
        """Calculate the PI of data for the DCA projection.

        Parameters
        ----------
        X : ndarray or list
            Optional. If X is none, calculate PI from the training data.
            If X is given, calcuate the PI of X for the learned projections.
        """
        T = self.T_fit
        if T is None:
            T = self.T
        if X is None:
            cross_covs = self.cross_covs.cpu().numpy()
        else:
            cross_covs = calc_cross_cov_mats_from_data(X, 2 * self.T)
        cross_covs = cross_covs[:2 * T]
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
    Yf = torch.fft.rfft(Y * window, dim=-1)
    spect = abs(Yf)**2
    spect = spect.mean(dim=1)
    spect = torch.cat([torch.flip(spect[:, 1:], dims=(1,)), spect], dim=1)

    # Log of the DFT of the autocorrelation
    logspect = torch.log(spect) - np.log(float(Y.shape[-1]))

    # Compute squared cepstral coefs (b_k^2)
    cepts = torch.fft.rfft(logspect, dim=1) / float(Y.shape[-1])
    cepts = abs(cepts)
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
                reg_val = ortho_reg_fn(self.ortho_lambda, v_torch)
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
            reg_val = ortho_reg_fn(self.ortho_lambda, v_torch)
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
