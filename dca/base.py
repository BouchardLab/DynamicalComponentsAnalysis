import logging, time
import numpy as np
import scipy.stats
from sklearn.utils import check_random_state

import torch


__all__ = ['BaseComponentsAnalysis',
           'SingleProjectionComponentsAnalysis',
           'ortho_reg_fn',
           'init_coef']


logging.basicConfig()


def ortho_reg_fn(ortho_lambda, *Vs):
    """Regularization term which encourages the basis vectors in the
    columns of the Vs to be orthonormal.

    Parameters
    ----------
    Vs : np.ndarrays, shape (N, d)
        Matrices whose columns are basis vectors.
    ortho_lambda : float
        Regularization hyperparameter.

    Returns
    -------
    reg_val : float
        Value of regularization function.
    """
    return sum([_ortho_reg_fn(ortho_lambda, V) for V in Vs])


def _ortho_reg_fn(ortho_lambda, V):
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


class ObjectiveWrapper(object):
    """Helper object to cache gradient computation for minimization.

    Parameters
    ----------
    f_params : callable
        Function to calculate the loss as a function of the parameters.
    """
    def __init__(self, f_params):
        self.common_computations = None
        self.params = None
        self.f_params = f_params
        self.n_f = 0
        self.n_g = 0
        self.n_c = 0

    def core_computations(self, *args, **kwargs):
        """Calculate the part of the computation that is common to computing
        the loss and the gradient.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        params = args[0]
        if not np.array_equal(params, self.params):
            self.n_c += 1
            self.common_computations = self.f_params(*args, **kwargs)
            self.params = params.copy()
        return self.common_computations

    def func(self, *args):
        """Calculate and return the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
        self.n_f += 1
        loss, _ = self.core_computations(*args)
        return loss.detach().cpu().numpy().astype(float)

    def grad(self, *args):
        """Calculate and return the gradient of the loss.

        Parameters
        ----------
        args
            Any other arguments that self.f_params needs.
        """
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


class BaseComponentsAnalysis(object):
    """Base Components Analysis class.

    Parameters
    ----------
    T : int
        Size of time windows.
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
        in the batch is longer than `T * chunk_cov_estimate`.
    tol : float
        Tolerance for stopping optimization. Default is 1e-6.
    ortho_lambda : float
        Coefficient on term that keeps V close to orthonormal.
    verbose : bool
        Verbosity during optimization.
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
    """
    def __init__(self, T=None, init="random_ortho", n_init=1, stride=1,
                 chunk_cov_estimate=None, tol=1e-6, verbose=False, device="cpu",
                 dtype=torch.float64, rng_or_seed=None):
        self.T = T
        self.T_fit = None
        self.init = init
        self.n_init = n_init
        self.stride = stride
        self.chunk_cov_estimate = chunk_cov_estimate
        self.tol = tol
        self.verbose = verbose
        self._logger = logging.getLogger('Model')
        if verbose:
            self._logger.setLevel(logging.DEBUG)
        else:
            self._logger.setLevel(logging.WARNING)
        self.device = device
        self.dtype = dtype
        self.mean_ = None
        self.rng = check_random_state(rng_or_seed)

    def estimate_data_statistics(self):
        """Estimate the data statistics needed for projection fitting.
        """
        raise NotImplementedError

    def _fit_projection(self):
        """Fit the a single round of projections.
        """
        raise NotImplementedError

    def fit_projection(self):
        """Fit the projections, with `n_init` restarts.
        """
        raise NotImplementedError

    def fit(self):
        """Estimate the data statistics and fit the projections.
        """
        raise NotImplementedError

    def transform(self):
        """Project the data onto the components after removing the training
        mean.
        """
        raise NotImplementedError

    def fit_transform(self, X, d=None, T=None, n_init=None, *args, **kwargs):
        """Estimate the data statistics and fit the projection matrix. Then
        project the data onto the components.
        """
        raise NotImplementedError

    def score(self):
        """Calculate the score of the data.
        """
        raise NotImplementedError('Classes should implement `score`.')


class SingleProjectionComponentsAnalysis(BaseComponentsAnalysis):
    """Base class for Components Analysis with 1 projection.

    Runs a Components Analysis method on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace of an N-dimensional space which maximizes the score of the
    d-dimensional dynamics over windows of length T.

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
        in the batch is longer than `T * chunk_cov_estimate`.
    tol : float
        Tolerance for stopping optimization. Default is 1e-6.
    verbose : bool
        Verbosity during optimization.
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
                 chunk_cov_estimate=None, tol=1e-6, verbose=False, device="cpu",
                 dtype=torch.float64, rng_or_seed=None):

        super(SingleProjectionComponentsAnalysis,
              self).__init__(T=T, init=init, n_init=n_init, stride=stride,
                             chunk_cov_estimate=chunk_cov_estimate, tol=tol,
                             verbose=verbose, device=device, dtype=dtype, rng_or_seed=rng_or_seed)

        self.d = d
        self.d_fit = None
        self.coef_ = None

    def fit_projection(self, d=None, T=None, n_init=None):
        """Fit the projection matrix.

        Parameters
        ----------
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional). Default is `self.T`. If `T` is set here
            it must be less than or equal to `self.T` or self.estimate_cross_covariance() must
            be called with a larger `T`.
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        scores = []
        coefs = []
        for ii in range(n_init):
            start = time.time()
            self._logger.info('Starting projection fig {} of {}.'.format(ii + 1, n_init))
            coef, score = self._fit_projection(d=d, T=T)
            delta_time = round((time.time() - start) / 60., 1)
            self._logger.info('Projection fit {} of {} took {:0.1f} minutes.'.format(ii + 1,
                                                                                     n_init,
                                                                                     delta_time))
            scores.append(score)
            coefs.append(coef)
        idx = np.argmax(scores)
        self.coef_ = coefs[idx]

    def fit(self, X, d=None, T=None, n_init=None, *args, **kwargs):
        """Estimate the data statistics and fit the projection matrix.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional.)
        n_init : int
            Number of random restarts (optional.)
        """
        if n_init is None:
            n_init = self.n_init
        self.estimate_data_statistics(X, T=T, *args, **kwargs)
        self.fit_projection(d=d, n_init=n_init)
        return self

    def transform(self, X):
        """Project the data onto the components after removing the training
        mean.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        """
        if isinstance(X, list):
            y = [(Xi - self.mean_).dot(self.coef_) for Xi in X]
        elif X.ndim == 3:
            y = np.stack([(Xi - self.mean_).dot(self.coef_) for Xi in X])
        else:
            y = (X - self.mean_).dot(self.coef_)
        return y

    def fit_transform(self, X, d=None, T=None, n_init=None, *args, **kwargs):
        """Estimate the data statistics and fit the projection matrix. Then
        project the data onto the components.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            Data to estimate the cross covariance matrix.
        d : int
            Dimensionality of the projection (optional.)
        T : int
            T for PI calculation (optional.)
        n_init : int
            Number of random restarts (optional.)
        """
        self.fit(X, d=d, T=T, n_init=n_init, *args, **kwargs)
        return self.transform(X)
