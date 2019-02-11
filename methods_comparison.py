import numpy as np


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
