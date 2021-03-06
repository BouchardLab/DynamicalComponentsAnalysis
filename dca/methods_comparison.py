import warnings

import numpy as np

import scipy
from scipy.optimize import minimize
from sklearn.decomposition import FactorAnalysis as FA, PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state
from functools import partial

import torch
from torch.nn import functional as F


from .base import init_coef


__all__ = ['GaussianProcessFactorAnalysis',
           'SlowFeatureAnalysis',
           'ForecastableComponentsAnalysis',
           'JPCA']


def make_norm_power(X, T_ent):
    """Calculate the normalize power spectrum."""
    Y = F.unfold(X, kernel_size=[T_ent, 1], stride=T_ent)
    Y = torch.transpose(Y, 1, 2)
    Yf = torch.fft.rfft(Y, dim=1)
    Yp = torch.mean(abs(Yf)**2, dim=1)
    return Yp / torch.sum(Yp, dim=1, keepdim=True)


def ent_loss_fn(X, proj, T_ent):
    """Power spectrum entropy loss function."""
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if not isinstance(proj, torch.Tensor):
        proj = torch.Tensor(proj)
    Xp = X.mm(proj)
    Xp_tensor = Xp.t()
    Xp_tensor = torch.unsqueeze(Xp_tensor, -1)
    Xp_tensor = torch.unsqueeze(Xp_tensor, 1)
    YP = make_norm_power(Xp_tensor, T_ent)
    ent = -(YP * torch.log(YP)).sum(dim=1)
    return ent.sum()


class ForecastableComponentsAnalysis(object):
    """Forecastable Components Analysis.

    Runs FCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the entropy of the power spectrum.

    Based on http://proceedings.mlr.press/v28/goerg13.html, but does not calculate
    the gradients in the same way. This implementation uses autograd.

    Note: this has not been carefully tested.

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
    def __init__(self, d, T, init="random_ortho", n_init=1, tol=1e-6,
                 verbose=False, device="cpu", dtype=torch.float64,
                 rng_or_seed=20200818):
        self.d = d
        if d > 1:
            raise ValueError
        self.T = T
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.dtype = dtype
        self.cross_covs = None
        self.rng = check_random_state(rng_or_seed)

    def fit(self, X, d=None, T=None, n_init=None):
        if d is None:
            d = self.d
        if d > 1:
            raise ValueError
        if T is None:
            T = self.T
        self.pca = PCA(whiten=True)
        X = self.pca.fit_transform(X)
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
        if d > 1:
            raise ValueError
        N = X.shape[1]
        V_init = init_coef(N, d, self.rng, self.init)

        v = torch.tensor(V_init, requires_grad=True,
                         device=self.device, dtype=self.dtype)

        Xt = X
        if not isinstance(Xt, torch.Tensor):
            Xt = torch.tensor(Xt, device=self.device, dtype=self.dtype)

        if self.verbose:
            def callback(v_flat):
                with torch.no_grad():
                    v_flat_torch = torch.tensor(v_flat,
                                                device=self.device,
                                                dtype=self.dtype)
                    v_torch = v_flat_torch.reshape(N, d)
                    v_torch = v_torch / torch.norm(v_torch, dim=0, keepdim=True)
                    ent = ent_loss_fn(Xt, v_torch, self.T)
                    ent = ent.detach().cpu().numpy()
                    print("Ent: {} bits".format(str(np.round(ent, 4))))
            callback(V_init)
        else:
            callback = None

        def f_df(v_flat):
            v_flat_torch = torch.tensor(v_flat,
                                        requires_grad=True,
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            v_torch = v_torch / torch.norm(v_torch, dim=0, keepdim=True)
            loss = ent_loss_fn(Xt, v_torch, self.T)
            loss.backward()
            grad = v_flat_torch.grad
            return (loss.detach().cpu().numpy().astype(float),
                    grad.detach().cpu().numpy().astype(float))
        opt = minimize(f_df, V_init.ravel(), method='L-BFGS-B', jac=True,
                       options={'disp': self.verbose, 'ftol': 1e-6, 'gtol': 1e-5,
                                'maxfun': 15000, 'maxiter': 15000, 'maxls': 20},
                       callback=callback)
        v = opt.x.reshape(N, d)

        # Orthonormalize the basis prior to returning it
        V_opt = scipy.linalg.orth(v)
        with torch.no_grad():
            v_flat_torch = torch.tensor(V_opt.ravel(),
                                        device=self.device,
                                        dtype=self.dtype)
            v_torch = v_flat_torch.reshape(N, d)
            final_pi = ent_loss_fn(Xt, v_torch, self.T).detach().cpu().numpy()
        return V_opt, final_pi

    def transform(self, X):
        X = self.pca.transform(X)
        return X.dot(self.coef_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def score(self, X):
        return ent_loss_fn(X, self.coef_, self.T)


def calc_K(tau, delta_t, var_n):
    """Calculates the GP kernel autocovariance.
    """
    var_f = 1. - var_n
    rval = var_f * np.exp(-(delta_t)**2 / (2. * tau**2))
    if delta_t == 0:
        rval += var_n
    return rval


def calc_big_K(T, n_factors, tau, var_n, out=None):
    """Calculates the GP kernel autocorrelation for all latent factors.
    """
    if out is None:
        K = np.zeros((T * n_factors, T * n_factors))
    else:
        K = out
    for delta_t in range(T):
        diag = calc_K(tau, delta_t, var_n)
        diag = np.tile(diag, T - delta_t)
        idxs_0 = np.arange(0, (T - delta_t) * n_factors)
        idxs_1 = np.arange(delta_t * n_factors, T * n_factors)
        K[idxs_0, idxs_1] = diag
        K[idxs_1, idxs_0] = diag
    return K


def make_block_diag(M, num_reps, out=None):
    """Create a block diagonal matrix from M repeated num_reps times.
    """
    if out is None:
        big_M = np.zeros((M.shape[0] * num_reps, M.shape[1] * num_reps))
    else:
        big_M = out
    for i in range(num_reps):
        big_M[i * M.shape[0]:(i + 1) * M.shape[0], i * M.shape[1]:(i + 1) * M.shape[1]] = M
    return big_M


def block_dot_A(A, B, n_blocks, out=None):
    """Computes np.dot(make_block_diag(A, n_blocks), B).
    """
    block_r = A.shape[0]
    block_c = A.shape[1]
    if out is None:
        out = np.empty((block_r * n_blocks, B.shape[1]), dtype=A.dtype)
    for ii in range(n_blocks):
        Bi = B[ii * block_c:(ii + 1) * block_c]
        out[ii * block_r:(ii + 1) * block_r] = A.dot(Bi)
    return out


def block_dot_B(A, B, n_blocks, out=None):
    """Computes np.dot(A, make_block_diag(B, n_blocks)).
    """
    block_r = B.shape[0]
    block_c = B.shape[1]
    if out is None:
        out = np.empty((A.shape[0], block_c * n_blocks), dtype=A.dtype)
    for ii in range(n_blocks):
        Ai = A[:, ii * block_r:(ii + 1) * block_r]
        out[:, ii * block_c:(ii + 1) * block_c] = Ai.dot(B)
    return out


def block_dot_AB(A, B, n_blocks, out=None):
    """Computes np.dot(A, B) when A and B is block diagonal.
    """
    block_r = A.shape[0]
    block_c2 = B.shape[1]
    if out is None:
        out = np.zeros((block_r * n_blocks, block_c2 * n_blocks), dtype=A.dtype)
    for ii in range(n_blocks):
        out[ii * block_r:(ii + 1) * block_r, ii * block_c2:(ii + 1) * block_c2] = A.dot(B)
    return out


def matrix_inversion_identity(R_inv, K, C, T):
    """Computes (R + CKC^T)^{-1} using the matrix inversion identity as
    R^{-1} - R^{-1}C(K^{-1} + C^TR^{-1}C)^{-1}C^TR^{-1}
    Useful when dim(R) > dim(K) and R^{inv} can be easily computed or has been
    previously computed.

    R_inv and C must be block diagonal.
    """
    K_inv = np.linalg.inv(K)
    R_invC = np.dot(R_inv, C)
    sub = K_inv + block_dot_AB(C.T, R_invC, T)
    term1 = block_dot_A(R_invC, np.linalg.solve(sub, make_block_diag(R_invC.T, T)), T)
    return make_block_diag(R_inv, T) - term1


def log_likelihood(mu, sigma, y, T):
    """Log likelihood for a multivariate normal distribution.

    Only works for 1 sample data.
    """
    ll = 0.
    for yi, Ti in zip(y, T):
        d = yi.size
        log_det_cov = np.linalg.slogdet(sigma[Ti])[1]
        y_minus_mean = yi - mu[Ti]
        term3 = np.dot(y_minus_mean.T.ravel(),
                       np.linalg.solve(sigma[Ti], y_minus_mean.T).ravel())
        ll += (-0.5 * d * np.log(2 * np.pi) - 0.5 * log_det_cov - 0.5 * term3)
    return ll


class GaussianProcessFactorAnalysis(object):
    """Gaussian Process Factor Analysis model.

    Based on formulation in https://journals.physiology.org/doi/full/10.1152/jn.90941.2008.

    Parameters
    ----------
    n_factors : int
        Number of latent factors.
    var_n : float
        Independent noise for the factors.
    tol : float
        The EM iterations stop when
        `|L^k - L^{k+1}|/max{|L^k|,|L^{k+1}|,1} <= tol`.
    max_iter : int
        Maximum number of EM steps.
    tau_init : float
        Scale for timescale initialization. Units are in sampling rate units.
    """
    def __init__(self, n_factors, var_n=1e-3, tol=1e-8, max_iter=500,
                 tau_init=10, seed=20190213, verbose=False):
        self.n_factors = n_factors
        self.var_n = var_n
        self.tol = tol
        self.max_iter = max_iter
        self.tau_init = tau_init
        self.verbose = verbose
        if tau_init <= 0:
            raise ValueError
        self.rng = np.random.RandomState(seed)

    def fit(self, y):
        """Fit the GPFA model parameters to the obervations y.

        Parameters
        ----------
        y : ndarray (time, features)
        """
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = [y]
        y_all = np.concatenate(y)
        self.mean_ = y_all.mean(axis=0, keepdims=True)
        y = [yi - self.mean_ for yi in y]
        n = y[0].shape[1]
        T = [yi.shape[0] for yi in y]
        model = FA(self.n_factors, svd_method='lapack')
        model.fit(y_all)

        self.R_ = np.diag(model.noise_variance_)
        self.C_ = model.components_.T
        self.d_ = np.zeros(n)
        self.tau_ = self.tau_init + self.rng.rand(self.n_factors)
        # Allocated and reuse these
        C = self.C_
        R = self.R_
        big_K = {Ti: calc_big_K(Ti, self.n_factors, self.tau_, self.var_n) for Ti in set(T)}
        y_cov = {Ti: block_dot_B(block_dot_A(C, big_K[Ti], Ti), C.T, Ti) + make_block_diag(R, Ti)
                 for Ti in set(T)}
        big_d = {Ti: np.tile(self.d_, Ti) for Ti in set(T)}
        big_y = [yi.ravel() for yi in y]
        ll_pre = log_likelihood(big_d, y_cov, big_y, T)
        if self.verbose:
            print("FA log likelihood:", ll_pre)

        converged = False
        for ii in range(self.max_iter):
            ll = self._em_iter(y, big_K)
            if abs(ll - ll_pre) / np.amax([abs(ll), abs(ll_pre), 1.]) <= self.tol:
                converged = True
                break
            ll_pre = ll
        if not converged:
            warnings.warn("EM max_iter reached.", ConvergenceWarning)
        return self

    def _em_iter(self, y, big_K):
        """One step of EM.

        Exact updates for d, C, and R. Optimization for tau

        Parameters
        ----------
        y : ndarray (time, features)
        """
        T = [yi.shape[0] for yi in y]
        big_d = {Ti: np.tile(self.d_, Ti) for Ti in set(T)}
        big_y = [yi.ravel() for yi in y]
        C = self.C_
        R = self.R_

        mean, big_K, big_dy, KCt, KCt_CKCtR_inv = self._E_mean(y)
        cov = {Ti: big_K[Ti] - KCt_CKCtR_inv[Ti].dot(KCt[Ti].T) for Ti in set(T)}
        y_cov = {Ti: block_dot_A(C, KCt[Ti], Ti) + make_block_diag(R, Ti)
                 for Ti in set(T)}

        if self.verbose == 2:
            # Compute log likelihood under current params
            ll = log_likelihood(big_d, y_cov, big_y, T)
            print("Pre update log likelihood:", ll)

        x = [meani.reshape(Ti, -1) for meani, Ti in zip(mean, T)]
        big_xxp = [cov[Ti] + np.outer(meani, meani) for Ti, meani in zip(T, mean)]
        nf = self.n_factors
        xxp = np.zeros((nf + 1, nf + 1))
        for Ti, big_xxpi in zip(T, big_xxp):
            for t in range(Ti):
                sl = slice(t * nf, (t + 1) * nf)
                xxp[:nf, :nf] += big_xxpi[sl, sl]
        xxp[-1, -1] = sum(T)
        sum_x = sum([xi.sum(axis=0) for xi in x])
        xxp[:-1, -1] = sum_x
        xxp[-1, :-1] = sum_x
        yx = sum([yi.T.dot(np.concatenate((xi, np.ones((Ti, 1))), axis=1))
                  for yi, xi, Ti in zip(y, x, T)])
        Cd = np.linalg.solve(xxp, yx.T).T
        self.C_ = Cd[:, :-1]
        if self.verbose == 2:
            # Compute log likelihood under current params
            ll = self._calc_loglikelihood(y)
            print("C_ update log likelihood:", ll)
        self.d_ = Cd[:, -1]
        if self.verbose == 2:
            # Compute log likelihood under current params
            ll = self._calc_loglikelihood(y)
            print("d_ update log likelihood:", ll)
        dy = [yi - self.d_[np.newaxis] for yi in y]
        self.R_ = sum([np.diag(np.diag(dyi.T.dot(dyi) - dyi.T.dot(xi).dot(self.C_.T)))
                       for dyi, xi in zip(dy, x)]) / sum(T)
        if self.verbose == 2:
            # Compute log likelihood under current params
            ll = self._calc_loglikelihood(y)
            print("Exact update log likelihood:", ll)
        self.tau_ = self._optimize_tau(self.tau_, T, big_xxp, big_K)
        ll = self._calc_loglikelihood(y)
        if self.verbose == 2:
            # Compute log likelihood under current params
            print("tau update log likelihood:", ll)
        if self.verbose:
            # Compute log likelihood under current params
            print("EM update log likelihood:", ll)
            print()
        return ll

    def _calc_loglikelihood(self, y):
        T = [yi.shape[0] for yi in y]
        big_d = {Ti: np.tile(self.d_, Ti) for Ti in set(T)}
        big_y = [yi.ravel() for yi in y]
        C = self.C_
        R = self.R_

        mean, big_K, big_dy, KCt, KCt_CKCtR_inv = self._E_mean(y)
        y_cov = {Ti: block_dot_A(C, KCt[Ti], Ti) + make_block_diag(R, Ti)
                 for Ti in T}
        return log_likelihood(big_d, y_cov, big_y, T)

    def score(self, y):
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = [y]
        return self._calc_loglikelihood(y)

    def _optimize_tau(self, tau_init, T, Sigma_mu_mu_x, big_K=None):
        """Optimization for tau.

        Parameters
        ----------
        tau_init : ndarray
            Inital value for taus.
        T : int
            Number of time points.
        Sigma_mu_mu_x : ndarray (T * n_factors, T * n_factors)
            Sigma + mu mu^T for x.

        Returns
        -------
        opt_tau : ndarray
            Optimal tau parameters from M step.
        """
        if big_K is None:
            big_K = {Ti: None for Ti in set(T)}
        log_tau_init = np.log(tau_init)
        var_f = 1. - self.var_n

        def f_df(big_K, log_tau):
            big_K = {Ti: calc_big_K(Ti, self.n_factors, np.exp(log_tau), self.var_n, big_K[Ti])
                     for Ti in set(T)}
            K_inv = {Ti: np.linalg.inv(big_K[Ti]) for Ti in set(T)}
            f = sum([-.5 * (np.sum(K_inv[Ti] * Sigma_mu_mu_xi) +
                            np.linalg.slogdet(2. * np.pi * big_K[Ti])[1])
                     for Ti, Sigma_mu_mu_xi in zip(T, Sigma_mu_mu_x)])

            df = np.zeros_like(log_tau)
            t_vals = [np.arange(Ti)[np.newaxis] for Ti in T]
            delta_t = [t_valsi - t_valsi.T for t_valsi in t_vals]
            for batch_idx, Tb in enumerate(T):
                Kb = big_K[Tb]
                Sigma_mu_mu_xb = Sigma_mu_mu_x[batch_idx]
                delta_tb = delta_t[batch_idx]
                for ii, lti in enumerate(log_tau):
                    idxs = ii + (np.arange(Tb) * self.n_factors)
                    Ki = Kb[idxs, :][:, idxs]
                    Ki_inv = np.linalg.inv(Ki)
                    xpx = Sigma_mu_mu_xb[idxs, :][:, idxs]
                    dEdKi = .5 * (-Ki_inv + Ki_inv.dot(xpx).dot(Ki_inv))
                    dKidti = (var_f * (delta_tb**2 / np.exp(lti)**3) *
                              np.exp(-delta_tb**2 / (2 * np.exp(lti)**2)))
                    df[ii] += np.trace(np.dot(dEdKi.T, dKidti)) * np.exp(lti)
            if self.verbose == 2:
                print('tau opt', f)

            return -f, -df

        opt_result = minimize(partial(f_df, big_K), x0=log_tau_init, method="L-BFGS-B", jac=True)
        opt_tau = np.exp(opt_result.x)
        return opt_tau

    def _E_mean(self, y, big_K=None):
        """Infer the mean of the latent variables x given obervations y.

        Parameters
        ----------
        y : ndarray (time, features)

        Returns
        -------
        x : ndarray (time, n_factors)
        """
        T = [yi.shape[0] for yi in y]
        big_d = [np.tile(self.d_, Ti) for Ti in T]
        big_y = [yi.ravel() for yi in y]
        if big_K is None:
            big_K = {Ti: None for Ti in set(T)}

        big_K = {Ti: calc_big_K(Ti, self.n_factors, self.tau_, self.var_n, big_K[Ti])
                 for Ti in set(T)}
        R_inv = np.linalg.inv(self.R_)
        big_dy = [big_yi - big_di for big_yi, big_di in zip(big_y, big_d)]
        KCt = {Ti: block_dot_B(big_K[Ti], self.C_.T, Ti) for Ti in set(T)}

        KCt_CKCtR_inv = {Ti: KCt[Ti].dot(matrix_inversion_identity(R_inv, big_K[Ti], self.C_, Ti))
                         for Ti in T}
        mean = [KCt_CKCtR_inv[Ti].dot(big_dyi) for Ti, big_dyi in zip(T, big_dy)]
        return mean, big_K, big_dy, KCt, KCt_CKCtR_inv

    def transform(self, y):
        """Infer the mean of the latent variables x given obervations y.

        Parameters
        ----------
        y : ndarray (time, features)

        Returns
        -------
        x : ndarray (time, n_factors)
        """
        if isinstance(y, np.ndarray) and y.ndim == 2:
            T, n = y.shape
            x = self._E_mean([y - self.mean_])[0]
            x = x[0].reshape(T, self.n_factors)
        else:
            x = self._E_mean([yi - self.mean_ for yi in y])[0]
            x = [xi.reshape(yi.shape[0], self.n_factors) for xi, yi in zip(x, y)]
        return x

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SlowFeatureAnalysis(object):
    """Slow Feature Analysis (SFA)

    Based on https://www.mitpressjournals.org/doi/abs/10.1162/089976602317318938.

    Parameters
    ----------
    n_components : int
        The number of components to learn.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.coef_ = None

    def fit(self, X):
        """Fit the SFA model.

        Parameters
        ----------
        X : ndarray (time, features)
            Data to fit SFA model to.
        """
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [X]
        self.mean_ = np.concatenate(X).mean(axis=0, keepdims=True)
        X_stan = [Xi - self.mean_ for Xi in X]
        uX, sX, vhX = np.linalg.svd(np.concatenate(X_stan), full_matrices=False)
        whiten = vhX.T @ np.diag(1. / sX)
        Xw = [X_stani @ whiten for X_stani in X_stan]
        Xp = [np.diff(Xwi, axis=0) for Xwi in Xw]
        up, sp, vhp = np.linalg.svd(np.concatenate(Xp), full_matrices=False)
        proj = vhp.T
        self.all_coef_ = whiten @ proj[:, ::-1]
        self.all_coef_ /= np.linalg.norm(self.all_coef_, axis=0, keepdims=True)
        self.coef_ = self.all_coef_[:, :self.n_components]
        return self

    def transform(self, X, n_components=None):
        """Transform the data according to the fit SFA model.

        Parameters
        ----------
        X : ndarray (time, features)
            Data to transform using the SFA model.
        """
        if n_components is None:
            n_components = self.n_components
        if self.coef_ is None:
            raise ValueError
        return (X - self.mean_) @ self.all_coef_[:, :n_components]

    def fit_transform(self, X, n_components=None):
        """Fit the SFA model and transform the features.

        Parameters
        ----------
        X : ndarray (time, features)
            Data to fit SFA model to and then transformk.
        """
        if n_components is None:
            n_components = self.n_components
        self.fit(X)
        return self.transform(X, n_components=n_components)


class JPCA(object):
    """ Model for extracting rotational dynamics from timeseries data using jPCA.

    As presented in https://www.nature.com/articles/nature11129.
    Based on code from https://churchland.zuckermaninstitute.columbia.edu/content/code.

    Parameters
    ----------
    n_components : even int (default=6)
        Number of components to reduce X to.

    mean_subtract: boolean (default=True)
        Whether to subtract the cross-condition mean from each condition
        before running jPCA.

    Attributes
    ----------
    eigen_vecs_ : list
        List of numpy eigenvectors from JPCA skew symmetric matrix sorted in
        descending order by magnitude of eigenvalue.

    eigen_vals_ : list
        List of eigenvalues from JPCA skew symmetric matrix. The index
        of each eigenvalue corresponds with the eigenvector in `eigen_vecs_`.

    pca_ : sklearn.decomp.PCA object
        PCA object used to transform X to X_red.

    cross_condition_mean_ : ndarray (time, features)
        Cross condition mean of X during fit.

    proj_vectors_ : list
        List of projection vectors sorted in order by conjugate eigenvalue pairs.


    """
    def __init__(self, n_components=6, mean_subtract=True):
        if n_components // 2 != n_components / 2:
            raise ValueError("n_components must be even int")
        self.n_components_ = n_components
        self.mean_subtract_ = mean_subtract
        self.eigen_vecs_ = None
        self.eigen_vals_ = None
        self.pca_ = None
        self.cross_condition_mean_ = None
        self.proj_vectors_ = None

    def fit(self, X):
        """ Fit a jPCA model to X.

        Parameters
        ----------
        X : ndarray (conditions, time, features)
            Data to fit using jPCA model.

        Returns
        -------
        self
        """
        if self.n_components_ > X.shape[2]:
            raise ValueError("n_components is greater than number of features in X.")

        if len(X.shape) != 3:
            raise ValueError("Data must be in 3 dimensions (conditions, time, features).")

        if self.mean_subtract_:
            self.cross_condition_mean_ = np.mean(X, axis=0, keepdims=True)
            X = X - self.cross_condition_mean_

        X_flat = np.concatenate(X, axis=0)
        self.pca_ = PCA(n_components=self.n_components_)
        self.pca_.fit(X_flat)

        X_red = [self.pca_.transform(Xi) for Xi in X]
        dX = np.concatenate([np.diff(Xi, axis=0) for Xi in X_red], axis=0)
        X_prestate = np.concatenate([Xi[:-1] for Xi in X_red], axis=0)
        M_skew = self._fit_skew(X_prestate, dX)

        self.eigen_vals_, self.eigen_vecs_ = self._get_jpcs(M_skew)

        self.proj_vectors_ = []
        for i in range(len(self.eigen_vecs_) // 2):
            v1 = self.eigen_vecs_[2 * i]
            v2 = self.eigen_vecs_[2 * i + 1]
            real_v1 = np.real(v1 + v2)
            real_v1 /= np.linalg.norm(real_v1)
            real_v2 = np.imag(v1 - v2)
            real_v2 /= np.linalg.norm(real_v2)
            self.proj_vectors_.append(real_v1)
            self.proj_vectors_.append(real_v2)
        self.proj_vectors_ = np.array(self.proj_vectors_)
        return self

    def transform(self, X):
        """ Transform X using JPCA components.

        Parameters
        ----------
        X : ndarray (conditions, time, features)
            Data to fit using jPCA model.

        Returns
        -------
        ndarray (conditions, time, n_components)
            X projected onto jPCA components (conditions are preserved).
            In X_proj, every pair of features correspond to a conjugate pair
            of JPCA eigenvectors. The pairs are sorted by largest magnitude eigenvalue
            (i.e. dimensions 0 and 1 in X_proj contains the projection from
            the conjugate eigenvector pair with the largest eigenvalue magnitude).
            The projection pair is what captures the rotations.

        """

        if self.mean_subtract_:
            X = X - self.cross_condition_mean_

        X_red = [self.pca_.transform(Xi) for Xi in X]
        X_proj = np.stack([X_redi @ self.proj_vectors_.T for X_redi in X_red], axis=0)
        return X_proj

    def fit_transform(self, X):
        """ Fit and transform X using JPCA.

        Parameters
        ----------
        X : ndarray (conditions, time, features)
            Data to be transformed by JPCA.

        Returns
        -------
        ndarray (conditions*time, n_components)
            X projected onto JPCA components.
        """
        self.fit(X)
        return self.transform(X)

    def _fit_skew(self, X_prestate, dX):
        """
        Assume the differential equation dX = M * X_prestate. This function will return
        M_skew, the skew symmetric component of M, that best fits the data. dX and
        X_prestate should be the same shape.
        Note: M is solved for using least squares.

        Parameters
        ----------
        X_prestate : np.array
            Time series matrix with the last time step removed.

        dX : np.array
            Discrete derivative matrix of X obtained by subtracting each row with its
            previous time step. (derivative at time 0 is not included).

        Returns
        -------
        M_skew : np.array
            Optimal skew symmetric matrix that best fits dX and X_prestate.

        """
        # guaranteed to be square
        M0, _, _, _ = np.linalg.lstsq(X_prestate, dX, rcond=None)
        M0_skew = .5 * (M0 - M0.T)
        m_skew = self._mat2vec(M0_skew)
        opt = self._optimize_skew(m_skew, X_prestate, dX)
        return self._vec2mat(opt.x)

    def _optimize_skew(self, m_skew, X_prestate, dX):
        """
        Solves for M_skew using gradient optimization methods.
        The objective function and derivative equations have closed forms.

        Parameters
        ----------
        m_skew : np.array
            Flattened array (1d vector) of initial M_skew guess

        X_prestate : np.array
            Time series matrix with the last time step removed.

        dX : np.array
            Discrete derivative matrix of X obtained by subtracting each row with its
            previous time step. (derivative at time 0 is not included).

        Returns
        -------
        opt : scipy.OptimizeResult object
            SciPy optimization result.
        """
        def objective(x, X_prestate, dX):
            f = np.linalg.norm(dX - X_prestate @ self._vec2mat(x))
            return f**2

        def derivative(x, X_prestate, dX):
            D = dX - X_prestate @ self._vec2mat(x)
            D = D.T @ X_prestate
            return 2 * self._mat2vec(D - D.T)

        return minimize(objective, m_skew, jac=derivative, args=(X_prestate, dX))

    def _get_jpcs(self, M_skew):
        """
        Given optimal M_skew matrix, return the eigenvalues and eigenvectors
        of M_skew. The eigenvectors/values are sorted by eigenvalue magnitude.

        Parameters
        ----------
        M_skew : np.array
            optimal M_skew (2D matrix)

        Returns
        -------
        evecs : np.array
            2D Array where each row is a jPC.

        evals : np.array of floats
            Array where each position contains the correpsonding eigenvalue to the
            jPC in evecs.
        """
        evals, evecs = np.linalg.eig(M_skew)
        evecs = evecs.T
        # get rid of small real number
        evals_j = np.imag(evals)

        # sort in descending order
        sort_indices = np.argsort(-np.absolute(evals_j))
        return evals_j[sort_indices], evecs[sort_indices]

    def _mat2vec(self, mat):
        """
        Convert 2D array into flattened array in column major order.

        Parameters
        ----------
        mat : ndarray (num_rows, num_cols)
            2D matrix to be flattened.

        Returns:
            1D ndarray of size (num_rows*num_cols)
        """
        return mat.flatten('F')

    def _vec2mat(self, vec):
        """
        Convert flattened vector into 2D matrix in column major order.

        Parameters
        ----------
        vec : 1D ndarray (num_rows*num_cols, 1)
            Flattened array to be reshaped into 2D square ndarray.

        Returns
        -------
            2D ndarray (num_rows, num_cols)
        """
        shape = (int(vec.size**(.5)), -1)
        return np.reshape(vec, shape, 'F')
