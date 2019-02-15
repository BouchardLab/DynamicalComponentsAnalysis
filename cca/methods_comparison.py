import numpy as np

from scipy.optimize import minimize

from sklearn.decomposition import FactorAnalysis as FA

__all__ = ['GaussianProcessFactorAnalysis',
           'SlowFeatureAnalysis']


def calc_K(tau, delta_t, var_n):
    """Calculates the GP kernel autocorrelation.
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
        idxs_0 = np.arange(0, (T - delta_t)*n_factors)
        idxs_1 = np.arange(delta_t*n_factors, T*n_factors)
        K[idxs_0, idxs_1] = diag
        K[idxs_1, idxs_0] = diag
    return K

def make_block_diag(M, num_reps, out=None):
    """Create a block diagonal matrix from M repeated num_reps times.
    """
    if out is None:
        big_M = np.zeros((M.shape[0]*num_reps, M.shape[1]*num_reps))
    else:
        big_M = out
    for i in range(num_reps):
        big_M[i*M.shape[0]:(i+1)*M.shape[0], i*M.shape[1]:(i+1)*M.shape[1]] = M
    return big_M

def log_likelihood(mu, sigma, y):
    """Log likelihood for a multivariate normal distribution.

    Only works for 1 sample data.
    """
    d = y.size
    log_det_cov = np.linalg.slogdet(sigma)[1]
    y_minus_mean = y - mu
    cov_y = np.dot(y_minus_mean.T, y_minus_mean)
    log_likelihood = (-0.5*d*np.log(2*np.pi)
                      - 0.5*log_det_cov
                      - 0.5*np.trace(np.dot(np.linalg.inv(sigma), cov_y)))
    return log_likelihood


class GaussianProcessFactorAnalysis(object):
    """Gaussian Process Factor Analysis model.

    Parameters
    ----------
    n_factors : int
        Number of latent factors.
    var_n : float
        Independent noise for the factors.
    max_iter : int
        Maximum number of EM steps.
    tau_init : float
        Scale for timescale initialization. Units are in sampling rate units.
    """
    def __init__(self, n_factors, var_n=1e-3, max_iter=100,
                 tau_init=10, seed=20190213, verbose=False):
        self.n_factors = n_factors
        self.var_n = var_n
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
        T, n = y.shape
        model = FA(self.n_factors)
        model.fit(y)
        self.R_ = np.diag(model.noise_variance_)
        self.C_ = model.components_.T
        self.d_ =  np.zeros(n)
        self.tau_ = self.tau_init + self.rng.rand(self.n_factors)
        # Allocated and reuse these
        big_K = calc_big_K(T, self.n_factors, self.tau_, self.var_n)
        big_C = make_block_diag(self.C_, T)
        big_R = make_block_diag(self.R_, T)
        y_cov = big_C.dot(big_K).dot(big_C.T) + big_R
        big_d = np.tile(self.d_, T)
        big_y = y.ravel()
        ll = log_likelihood(big_d, y_cov, big_y)
        if self.verbose:
            print("log likelihood:", ll)

        for ii in range(self.max_iter):
            self._em_iter(y, big_K, big_C, big_R)
        return self

    def _em_iter(self, y, big_K, big_C, big_R):
        """One step of EM.

        Exact updates for d, C, and R. Optimizatino for tau

        Parameters
        ----------
        y : ndarray (time, features)
        """
        T, n = y.shape
        big_y = y.ravel()
        big_d = np.tile(self.d_, T)
        mean, big_K, big_C, big_R, big_dy, KCt, KCt_CKCtR_inv = self._E_mean(y)
        cov = big_K - KCt_CKCtR_inv.dot(KCt.T)
        y_cov = big_C.dot(KCt) + big_R

        if self.verbose:
            #Compute log likelihood under current params
            ll = log_likelihood(big_d, y_cov, big_y)
            print("log likelihood:", ll)
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
        Cd = y.T.dot(np.concatenate((x, np.ones((T, 1))), axis=1)).dot(np.linalg.inv(xxp))
        self.C_ = Cd[:, :-1]
        self.d_ = Cd[:, -1]
        dy = y - self.d_[np.newaxis]
        self.R_ = np.diag(np.diag(dy.T.dot(dy) - dy.T.dot(x).dot(self.C_.T))) / T
        self.tau_ = self._optimize_tau(self.tau_, T, big_xxp)


    def _optimize_tau(self, tau_init, T, Sigma_mu_mu_x):
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
        log_tau_init = np.log(tau_init)
        var_f = 1. - self.var_n
        def f_df(log_tau):
            K = calc_big_K(T, self.n_factors, np.exp(log_tau), self.var_n)
            K_inv = np.linalg.inv(K)
            f = -.5 * (np.sum(K_inv * Sigma_mu_mu_x) +
                          np.linalg.slogdet(2. * np.pi * K)[1])

            df = np.zeros_like(log_tau)
            t_vals = np.arange(T)[np.newaxis]
            delta_t = t_vals - t_vals.T
            for ii, lti in enumerate(log_tau):
                idxs = ii + (np.arange(T) * self.n_factors)
                Ki = K[idxs, :][:, idxs]
                Ki_inv = np.linalg.inv(Ki)
                xpx = Sigma_mu_mu_x[idxs, :][:, idxs]
                dEdKi = .5 *(-Ki_inv + Ki_inv.dot(xpx).dot(Ki_inv))
                dKidti = var_f * (delta_t**2 / np.exp(lti)**3) * np.exp( - delta_t**2 / (2 * np.exp(lti)**2 ))
                df[ii] = np.trace( np.dot(dEdKi.T, dKidti) ) * np.exp(lti)

            return -f, -df

        opt_result = minimize(f_df, x0=log_tau_init, method="L-BFGS-B", jac=True)
        opt_tau = np.exp(opt_result.x)
        return opt_tau

    def _E_mean(self, y, big_K=None, big_C=None, big_R=None):
        """Infer the mean of the latent variables x given obervations y.

        Parameters
        ----------
        y : ndarray (time, features)

        Returns
        -------
        x : ndarray (time, n_factors)
        """
        T, n = y.shape
        big_y = y.ravel()
        big_d = np.tile(self.d_, T)
        big_K = calc_big_K(T, self.n_factors, self.tau_, self.var_n, big_K)
        big_C = make_block_diag(self.C_, T, big_C)
        big_R = make_block_diag(self.R_, T, big_R)
        big_dy = big_y - big_d
        KCt = big_K.dot(big_C.T)

        KCt_CKCtR_inv = KCt.dot(np.linalg.inv(big_C.dot(KCt) + big_R))
        mean = KCt_CKCtR_inv.dot(big_dy)
        return mean, big_K, big_C, big_R, big_dy, KCt, KCt_CKCtR_inv

    def transform(self, y):
        """Infer the mean of the latent variables x given obervations y.

        Parameters
        ----------
        y : ndarray (time, features)

        Returns
        -------
        x : ndarray (time, n_factors)
        """
        T, n = y.shape
        x, _, _, _, _, _, _ = self._E_mean(y)
        return x.reshape(T, self.n_factors)


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
