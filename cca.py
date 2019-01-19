import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from autograd import grad

def calc_cross_cov_mats(X, num_lags):
    """Calculates N-by-N cross-covariance matrices for time lags
    up to num_lags - 1, where N is the dimensionality of the time
    series data. The data X are mean-centered prior to the computation.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        cross-covariance matrices are computed.
    num_lags: int
        The number of time-lags.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N), float
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    """

    #mean-center data
    X = X - np.mean(X, axis=0)

    #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=num_lags-1
    N = X.shape[1]
    cross_cov_mats = np.zeros((num_lags, N, N))
    for delta_t in range(num_lags):
        cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
        cross_cov_mats[delta_t] = cross_cov
        
    return cross_cov_mats
    

def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None, return_cov=False):
    """Calculates the mutual information across consecutive time
    windows of length T in a Gaussian process whose statistics are
    given by the provided cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (2T, N, N)
        A set of 2T N-by-N cross-covariance matrices. The dt-th matrix gives
        the cross-covariance between X(t) and X(t+dt), where X(t) is a N-dimensional
        vector governed by a Gaussian process. 
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.
    return_cov : bool
        If true, returns the big covariance matrix used to compute mutual information.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """

    T = int(cross_cov_mats.shape[0]/2)
    if type(proj) != type(None):
        d = proj.shape[1]
    else:
        d = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]


    cross_cov_mats_proj = []
    if type(proj) != type(None):
        #cross_cov_mats_proj = np.einsum('ij,nil,lk->njk', proj, cross_cov_mats, proj)
        for i in range(2*T):
            cross_cov = cross_cov_mats[i]
            cross_cov_proj = np.dot(proj.T, np.dot(cross_cov, proj))
            cross_cov_mats_proj.append(cross_cov_proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    cross_cov_mats_repeated = []
    for i in range(2*T):
        for j in range(2*T):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats_proj[abs(i-j)])
            else:
                cross_cov_mats_repeated.append(cross_cov_mats_proj[abs(i-j)].T)

    cov_2T_tensor = np.reshape(np.array(cross_cov_mats_repeated), (2*T, 2*T, d, d))
    cov_2T = np.concatenate([np.concatenate(cov_ii, axis=1) for cov_ii in cov_2T_tensor])
    cov_T = cov_2T[:T*d, :T*d]

    sgn_T, logdet_T = np.linalg.slogdet(cov_T)
    sgn_2T, logdet_2T = np.linalg.slogdet(cov_2T)
    PI = (2*logdet_T - logdet_2T)/np.log(2)

    if return_cov:
        return PI, cov_2T
    else:
        return PI

def calc_pi(X, T):
    """Calculates the mutual information across consecutive time
    windows of length T by approximating the multidimensional time-
    series X as a stationary Gaussian process.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        mutual information is computed.
    T: int
        Size of time windows accross which to compute mutual information.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """

    cross_cov_mats = calc_cross_cov_mats(X, 2*T)
    return calc_pi_from_cross_cov_mats(cross_cov_mats)


def compute_cov_mat(X, T):
    cross_cov_mats = calc_cross_cov_mats(X, 2*T)
    pi, cov = calc_pi_from_cross_cov_mats(cross_cov_mats, return_cov=True)
    return cov

def reg_fn(V, lambda_param):
    d = V.shape[1]
    return lambda_param * np.sum((np.dot(V.T, V) - np.eye(d))**2) 

def build_loss(X, T, d, lambda_param=10):
    """Constructs a loss function which gives the (negative) predictive information
    in the projection of multidimensional timeseries data X onto a d-dimensional
    basis, where predictive information is computed using a stationary Gaussian 
    process approximation.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data from which the
        mutual information is computed.
    T: int
        Size of time windows accross which to compute mutual information.
    d: int
        Number of basis vectors onto which the data X are projected.
    Returns
    -------
    loss : function
       Loss function which accepts a (flattened) N-by-d matrix, whose
       columns are basis vectors, and outputs the negative predictive information
       corresponding to that projection.
    """

    cross_cov_mats = calc_cross_cov_mats(X, 2*T)
    N = X.shape[1]

    def loss(V_flat):

        V = V_flat.reshape(N, d)
        reg_part = reg_fn(V, lambda_param)    
        return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_part

    return loss

def pca(X):
    """Performs PCA on the multidimensional data X.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional data.
    Returns
    -------
    w: np.ndarray, shape (N,)
        Eigenvalues (decreasing)
    V np.ndarraym, shape (N, N)
        Eigenvectors (corresponding to sorted eigenvalues)
    """

    X = X - np.mean(X, axis=0)
    cov = np.dot(X.T, X)/(len(X) - 1)
    w, V = np.linalg.eigh(cov)
    w, V = w[::-1], V[:, ::-1]
    return w, V

    

def run_cca(X, T, d, init="random", method="BFGS", tol=1e-6, lambda_param=10, verbose=False):
    """Runs CCA on multidimensional timeseries data X to discover a projection
    onto a d-dimensional subspace which maximizes the complexity of the d-dimensional
    dynamics.
    Parameters
    ----------
    X : np.ndarray, shape (# time-steps, N)
        The multidimensional time series data.
    T: int
        Size of time windows accross which to compute mutual information.
    d: int
        Number of basis vectors onto which the data X are projected.
    init: string
        Options: "random", "PCA"
        Method for initializing the projection matrix.
    Returns
    -------
    V_opt: np.ndarray, shape (N, d)
        Projection matrix.
    """

    loss = build_loss(X, T, d, lambda_param=lambda_param)
    grad_loss = grad(loss)
    
    N = X.shape[1]
    
    if verbose:
        def callback(V_flat):
            loss_val = loss(V_flat)
            V = V_flat.reshape((N, d))
            reg_part = reg_fn(V, lambda_param)
            loss_no_reg = loss_val - reg_part
            pi = -loss_no_reg
            print("PI = " + str(np.round(pi, 4)) + " bits, reg = " + str(np.round(reg_part, 4)))
    else:
        callback = None
              
    if type(init) == str:
        if init == "random":
            V_init = np.random.normal(0, 1, (N, d))
            V_init = V_init / np.sqrt(np.sum(V_init**2, axis=0))
        if init == "random_ortho":
            V_init = scipy.stats.ortho_group.rvs(N)[:, :d]
        if init == "uniform":
            V_init = np.ones((N, d))/np.sqrt(N)
            V_init = V_init + np.random.normal(0, 0.001, V_init.shape)
        elif init == "pca":
            w, V = pca(X)
            V_init = V[:, :d]
    elif type(init) == np.ndarray:
        V_init = init

    opt_result = scipy.optimize.minimize(loss, V_init.flatten(), method=method, jac=grad_loss, callback=callback, tol=tol)
    V_opt_flat = opt_result["x"]
    V_opt = V_opt_flat.reshape((N, d))
    
    V_opt = scipy.linalg.orth(V_opt) #Orhtonormalize

    return V_opt

















