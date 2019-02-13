import numpy as np

from scipy.optimize import minimize

from sklearn.decomposition import FactorAnalysis as FA


__all__ = ['SlowFeatureAnalysis']


def calc_K(tau, delta_t, var_n):
    var_f = 1. - var_n
    rval = var_f * np.exp(-(delta_t)**2 / (2. * tau**2))
    if delta_t == 0:
        rval += var_n
    return rval

def calc_big_K(T, n_factors, tau, var_n)
    K = np.zeros((T * n_factors, T * n_factors))
    for delta_t in range(T):
        diag = calc_K(tau_, delta_t, var_n)
        diag = np.tile(diag, T - delta_t)
        idxs_0 = np.arange(0, T - delta_t)
        idxs_1 = np.arange(delta_t, T)
        K[idxs_0, idxs_1] = diag
        K[idxs_1, idxs_0] = diag
    return K




class GaussianProcessFactorAnalysis(object):
    """GPFA
    """
    def __init__(self, n_factors, var_n=1e-3, max_iter=100,
                 tau_init=10, seed=20190213):
        self.n_factors = n_factors
        self.var_n = var_n
        self.max_iter = max_iter
        self.tau_init = tau_init
        if tau_init <= 0:
            raise ValueError
        self.rng = np.random.RandomState(seed)

    def fit(self, y):
        T, n = y.shape
        model = FA(self.n_factors)
        model.fit(y)
        self.R_ = np.diag(model.noise_variance_)
        self.C_ = model.components_.T
        self.d_ =  np.zeros(n)
        self.tau_ = self.tau_init + self.rng.rand(self.n_factors)

    def _expectation(self, y):
        T, n = y.shape
        big_y = y.ravel()
        big_d = np.tile(self.d_, T)
        big_K = calc_big_K(T, self.n_factors, self.tau_, self.var_n)
        big_C = np.tile(self.C_, (T, T))
        big_R = np.tile(self.R_, (T, T))
        big_dy = big_y - big_d
        KCt = big_K.dot(big_C.T)
        KCt_CKCtR_inv = KCt.dot(np.linalg.inv(big_C.dot(KCt) - big_R))
        mean = KCt_CKCtR_inv.dot(big_dy)
        cov = big_K - KCt_CKCtR_inv.dot(KCt.T)
        y_cov = big_C.dot(KCt) + big_R
        x = mean.reshape(T, -1)
        # TODO: could do this just along the block diagonal
        big_xxp = cov + np.outer(mean, mean)
        xxp = np.zeros((self.n_factors + 1, self.n_factors + 1))
        for t in range(T):
            xxp[:self.n_factors, :self.n_factors] += big_xxp[t *self.n_factors:(t + 1) * self.n_factors,
                                                             t *self.n_factors:(t + 1) * self.n_factors]
        xxp[-1, -1] = 1.
        xxp[:-1, -1] = x.sum(axis=0)
        xxp[-1, :-1] = x.sum(axis=0)
        Cd = y.T.dot(np.concatenate(x_reshape, np.ones(T, 1), axis=1).dot(np.linalg.inv(xpp))
        self.C_ = Cd[:, :-1]
        self.d_ = Cd[:, -1]
        dy = y - self.d_[np.newaxis]
        self.R_ = np.diag(np.diag(dy.T.dot(dy) - dy.T.dot(x_reshape).dot(self.C_.T))) / T
        self.tau_ = optimize_tau(self.tau_)

    def optimize_tau(self, tau_init, T, Sigma_mu_mu_x):
        log_tau = np.log(tau_init)
        def f_df(log_tau):
            K = calc_big_K(T, self.n_factors, np.exp(log_tau), self.var_n)
            K_inv = np.linalg.inv(K)
            f = -.5 * (np.sum(K_inv * Sigma_mu_mu_x) +
                          np.linalg.slogdet(2. * np.pi * K)[1])

            df = np.zeros_like(log_tau)
            for ii, lti in log_tau:
                idxs = ii + (np.arange(T) * self.n_factors)
                Ki = K[idxs, idxs]
                Ki_inv = np.linalg.inv(Ki)
                xpx = Sigma_mu_mu_x[idxs, idxs]
                dEdK = .5 *(-Ki_inv + Ki_inv.dot(xpx).dot(Ki_inv))
            return f, df






class SlowFeatureAnalysis(object):
    """Slow Feature Analysis (SFA)

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
        X_stan = X - X.mean(axis=0, keepdims=True)
        uX, sX, vhX = np.linalg.svd(X_stan, full_matrices=False)
        whiten = vhX.T @ np.diag(1./sX)
        Xw = X_stan @ whiten
        Xp = np.diff(Xw, axis=0)
        up, sp, vhp = np.linalg.svd(Xp, full_matrices=False)
        proj = vhp.T
        self.coef_ = whiten @ proj[:, ::-1][:, :self.n_components]

    def transform(self, X):
        """Transform the data according to the fit SFA model.

        Parameters
        ----------
        X : ndarray (time, features)
            Data to transform using the SFA model.
        """
        if self.coef_ is None:
            raise ValueError
        return X @ self.coef_

    def fit_transform(self, X):
        """Fit the SFA model and transform the features.

        Parameters
        ----------
        X : ndarray (time, features)
            Data to fit SFA model to and then transformk.
        """
        self.fit(X)
        return X @ self.coef_
