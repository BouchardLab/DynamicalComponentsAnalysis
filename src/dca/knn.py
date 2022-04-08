import numpy as np
import scipy
from scipy.optimize import minimize

from .cov_util import form_lag_matrix
from .dca import ortho_reg_fn


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
