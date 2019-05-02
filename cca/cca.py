import numpy as np
import scipy.stats
from scipy.optimize import minimize
from scipy.signal.windows import hann

import torch
import torch.nn.functional as F

from .cov_util import calc_cross_cov_mats_from_data, calc_pi_from_cross_cov_mats

__all__ = ["ComplexityComponentsAnalysis"]

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
        reg_val = ortho_lambda * torch.sum((torch.mm(V.t(), V) - torch.eye(d, device=V.device, dtype=V.dtype))**2)
    else:
        reg_val = ortho_lambda * np.sum((np.dot(V.T, V) - np.eye(d))**2)

    return reg_val

def build_loss(cross_cov_mats, d, lambda_param=1):
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

    def loss(V_flat):
        V = V_flat.reshape(N, d)
        reg_val = ortho_reg_fn(V, lambda_param)
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
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False, use_scipy=True,
                 device="cpu", dtype=torch.float64):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose=verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None
        self.use_scipy = use_scipy

    def estimate_cross_covariance(self, X, T=None, regularization=None, reg_ops=None):
        if T is None:
            T = self.T
        else:
            self.T = T

        cross_covs = calc_cross_cov_mats_from_data(X, 2*self.T,
                                                   regularization=regularization,
                                                   reg_ops=reg_ops)
        self.cross_covs = torch.tensor(cross_covs, device=self.device, dtype=self.dtype)

        return self

    def fit_projection(self, d=None, n_init=None):
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

    def _fit_projection(self, d=None):
        if d is None:
            d = self.d
        if d < 1:
            raise ValueError
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

        c = self.cross_covs
        if not isinstance(c, torch.Tensor):
        	c = torch.tensor(c, device=self.device, dtype=self.dtype)

        if self.use_scipy:
            if self.verbose:
                def callback(v_flat):
                    v_flat_torch = torch.tensor(v_flat,
                                                requires_grad=True,
                                                device=self.device,
                                                dtype=self.dtype)
                    v_torch = v_flat_torch.reshape(N, d)
                    #optimizer.zero_grad()
                    loss = build_loss(c, d)(v_torch)
                    reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
                    loss_no_reg = loss - reg_val
                    loss = loss.detach().cpu().numpy()
                    reg_val = reg_val.detach().cpu().numpy()
                    print("PI: {} nats, reg: {}".format(str(np.round(-loss, 4)),
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
                #optimizer.zero_grad()
                loss = build_loss(c, d)(v_torch)
                loss.backward()
                grad = v_flat_torch.grad
                return loss.detach().cpu().numpy().astype(float), grad.detach().cpu().numpy().astype(float)
            opt = minimize(f_df, V_init.ravel(), method='L-BFGS-B', jac=True,
                           options={'disp': self.verbose, 'ftol': self.tol, 'gtol': 1e-5, 'maxfun': 15000, 'maxiter': 15000, 'maxls': 20},
                           callback=callback)
            v = opt.x.reshape(N, d)
        else:
            optimizer = torch.optim.LBFGS([v], max_eval=10**10, max_iter=10**10, tolerance_grad=1e-10, tolerance_change=1e-10)
            def closure():
                optimizer.zero_grad()
                loss = build_loss(c, d)(v)
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

    def fit(self, X, d=None, T=None, regularization=None, reg_ops=None):
        self.estimate_cross_covariance(X, T=T, regularization=regularization,
                                       reg_ops=reg_ops)
        self.fit_projection(d=d)
        return self

    def transform(self, X):
        if isinstance(X, list):
            y = [(Xi - X.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X]
        elif X.ndim == 3:
            y = np.stack([(Xi - X.mean(axis=0, keepdims=True)).dot(self.coef_) for Xi in X])
        else:
            y = (X - X.mean(axis=0, keepdims=True)).dot(self.coef_)
        return y

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None):
        self.fit(X, d=d, T=T, regularization=regularization, reg_ops=reg_ops)
        return self.transform(X)

    def score(self):
        return calc_pi_from_cross_cov_mats(self.cross_covs, self.coef_)


def make_cepts2(X, T_pi):
    """Calculate the squared real cepstral coefficents."""
    Y = F.unfold(X, kernel_size=[T_pi, 1], stride=T_pi)
    Y = torch.transpose(Y, 1, 2)

    # Compower the power spectral density
    window = torch.Tensor(hann(Y.shape[-1])[np.newaxis, np.newaxis])
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

def pi_fft_loss_fn(X, proj, T_pi):
    """Power spectrum entropy loss function."""
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
    """Dynamical Components Analysis.

    Runs DCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the dynamical complexity.
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
    def __init__(self, d=None, T=None, init="random_ortho", n_init=1, tol=1e-6,
                 ortho_lambda=10., verbose=False,
                 device="cpu", dtype=torch.float64):
        self.d = d
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.ortho_lambda = ortho_lambda
        self.verbose=verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None

    def fit(self, X, d=None, n_init=None):
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
        idx = np.argmax(pis)
        self.coef_ = coefs[idx]

    def _fit_projection(self, X, d=None):
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
                ent = pi_fft_loss_fn(Xt, v_torch, self.T)
                reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
                ent = ent.detach().cpu().numpy()
                reg_val = reg_val.detach().cpu().numpy()
                print("PI: {} nats, reg: {}".format(str(np.round(ent, 4)),
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
            ent = pi_fft_loss_fn(Xt, v_torch, self.T)
            reg_val = ortho_reg_fn(v_torch, self.ortho_lambda)
            loss = ent + reg_val
            loss.backward()
            grad = v_flat_torch.grad
            return loss.detach().cpu().numpy().astype(float), grad.detach().cpu().numpy().astype(float)
        opt = minimize(f_df, V_init.ravel(), method='L-BFGS-B', jac=True,
                       options={'disp': self.verbose, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 15000, 'maxiter': 15000, 'maxls': 20},
                       callback=callback)
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        v_flat_torch = torch.tensor(V_opt.ravel(),
                                    requires_grad=True,
                                    device=self.device,
                                    dtype=self.dtype)
        v_torch = v_flat_torch.reshape(N, d)
        final_pi = pi_fft_loss_fn(Xt, v_torch, self.T).detach().cpu().numpy()
        return V_opt, final_pi

    def transform(self, X):
        X = X - self.mean_
        return X.dot(self.coef_)

    def fit_transform(self, X, d=None, T=None, regularization=None,
                      reg_ops=None):
        self.fit(X, d=d, T=T, regularization=regularization, reg_ops=reg_ops)
        return self.transform(X)

    def score(self):
        return calc_pi_from_cross_cov_mats(self.cross_covs, self.coef_)
