import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from autograd import grad

def calc_cross_cov_mats(X, num_lags, regularization=None, reg_ops=None):
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
    regularization : string
        Regularization method for computing cross-covariance matrices.
        Options:
            'Abadir' : Method from 'Design-free estimation of variance
            matrices' (Abadir et al., 2014). We implement the 'Grand Average'
            estimator (see Eq. 11).
    reg_ops : dict
        Paramters for regularization.
    Returns
    -------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N), float
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    """
    
    #mean-center data
    X = X - np.mean(X, axis=0)
    N = X.shape[1]
    
    if type(regularization) == type(None):

        #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=num_lags-1
        cross_cov_mats = np.zeros((num_lags, N, N))
        for delta_t in range(num_lags):
            cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
            cross_cov_mats[delta_t] = cross_cov

        return cross_cov_mats
    
    elif regularization == "Abadir":
        
        #reg_ops:
        #M -- number of m values, uniformly spaced over [0.2, 0.8]
        #S -- number samples for each m
        #skip -- spacing of consecutive samples from series data
                #(see construction of 'X_with_lags')
        M = reg_ops["M"]
        S = reg_ops["S"]
        skip = reg_ops["skip"]
        
        n = int(np.floor((len(X)-num_lags+1)/skip))
        
        X_with_lags = np.zeros((n, N*num_lags))
        for i in range(n):
            X_with_lags[i, :] = X[i*skip:i*skip+num_lags].flatten()
        
        #Holds all M*S cov estimates
        results = np.zeros((M, S, N*num_lags, N*num_lags))
        
        m_vals = np.round(np.linspace(0.2, 0.8,  M)*n)
        idx = np.arange(n)
        
        #diagonalize all data
        #Note that this code uses 'V' where the paper uses 'P'
        cov = np.dot(X_with_lags.T, X_with_lags)/(n - 1)
        w, V = scipy.linalg.eigh(cov)
        w, V = w[::-1], V[:, ::-1]
        
        for m_idx in range(M):
            m = int(m_vals[m_idx])
            for sample_idx in range(S):
                
                #seperate data into groups 1 and 2
                idx_1 = np.random.choice(idx, size=m, replace=False)
                idx_2 = np.setdiff1d(idx, idx_1, assume_unique=True)
                X_1, X_2 = X_with_lags[idx_1, :], X_with_lags[idx_2, :]
                
                #mean-center both
                X_1 = X_1 - np.mean(X_1, axis=0)
                X_2 = X_2 - np.mean(X_2, axis=0)
                
                #diagonalize X_1
                cov_1 = np.dot(X_1.T, X_1)/(m - 1)
                w_1, V_1 = scipy.linalg.eigh(cov_1)
                w_1, V_1 = w_1[::-1], V_1[:, ::-1]
                
                #project X_2 onto eigenvectors of X_1
                proj_X_2 = np.dot(X_2, V_1)
                
                #estimate spectrum based on variance of projection
                lambda_est = np.var(proj_X_2, axis=0)
                
                #form the covariance estimate
                cov_est = np.dot(V, np.dot(np.diag(lambda_est), V.T))
                
                #store the result
                results[m_idx, sample_idx, :, :] = cov_est
                
        final_cov_est = np.mean(results, axis=(0, 1))
        
        #Note final_cov_est will not be stationary, since stationarity
        #is not preserved by the regularization technique.
        #Thus, we average along the lower diagonal for each delta_t.
        cross_cov_mats = np.zeros((num_lags, N, N))
        for delta_t in range(num_lags):
            to_avg = np.zeros((num_lags-delta_t, N, N))
            for i in range(len(to_avg)):
                i_offset = delta_t*N
                to_avg[i, :, :] = final_cov_est[i_offset+i*N:i_offset+(i+1)*N, i*N:(i+1)*N]
            cross_cov_mats[delta_t, :, :] = np.mean(to_avg, axis=0)
        return cross_cov_mats

    
def calc_cov_from_cross_cov_mats(cross_cov_mats):
    """Calculates a N*num_lags-by-N*num_lags covariance matrix based on
    num_lags N-by-N cross-covariance matrices. This function is 'autograd-safe'
    since is does not use array assignment, only Python list appending.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    Returns
    -------
    cov : np.ndarray, shape (N*num_lags, N*num_lags)
        Big covariance matrix, stationary in time by construction. 
    """
    
    N = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]
    num_lags = len(cross_cov_mats)
    
    cross_cov_mats_repeated = []
    for i in range(num_lags):
        for j in range(num_lags):
            if i > j:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)])
            else:
                cross_cov_mats_repeated.append(cross_cov_mats[abs(i-j)].T)

    cov_tensor = np.reshape(np.array(cross_cov_mats_repeated), (num_lags, num_lags, N, N))
    cov = np.concatenate([np.concatenate(cov_ii, axis=1) for cov_ii in cov_tensor])
    
    return cov


def calc_pi_from_cov(cov_2T):
    """Calculates the mutual information ("predictive information"
    or "PI") between variables  {1,...,T} and {T+1,...,2*T}, which
    are jointly Gaussian with covariance matrix cov_2T.
    Parameters
    ----------
    cov_2T : np.ndarray, shape (2*T, 2*T)
        Covariance matrix.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """
    
    half = int(cov_2T.shape[0]/2)
    
    cov_T = cov_2T[:half, :half]
    sgn_T, logdet_T = np.linalg.slogdet(cov_T)
    sgn_2T, logdet_2T = np.linalg.slogdet(cov_2T)
    PI = (2*logdet_T - logdet_2T)/np.log(2)
    
    return PI


def calc_pi_from_cross_cov_mats(cross_cov_mats, proj=None):
    """Calculates predictive information for a spatiotemporal Gaussian
    process with num_lags-1 N-by-N cross-covariance matrices.
    Parameters
    ----------
    cross_cov_mats : np.ndarray, shape (num_lags, N, N)
        Cross-covariance matrices: cross_cov_mats[dt] is the
        cross-covariance between X(t) and X(t+dt), where each
        of X(t) and X(t+dt) is a N-dimensional vector.
    proj: np.ndarray, shape (N, d), optional
        If provided, the N-dimensional data are projected onto a d-dimensional
        basis given by the columns of proj. Then, the mutual information is
        computed for this d-dimensional timeseries.
    Returns
    -------
    PI : float
        Mutual information in bits.
    """

    T = int(cross_cov_mats.shape[0]/2)
    if type(proj) == np.ndarray:
        d = proj.shape[1]
    else:
        d = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]


    cross_cov_mats_proj = []
    if type(proj) != type(None):
        for i in range(2*T):
            cross_cov = cross_cov_mats[i]
            cross_cov_proj = np.dot(proj.T, np.dot(cross_cov, proj))
            cross_cov_mats_proj.append(cross_cov_proj)
    else:
        cross_cov_mats_proj = cross_cov_mats

    
    cov_2T = calc_cov_mat_from_cross_cov_mats(cross_cov_mats_proj)
    PI = calc_pi_from_cov_mat(cov_2T)

    return PI


def ortho_reg_fn(V, lambda_param):
    """Regularization term which encourages the basis vectors in the
    columns of V to be orthonormal. 
    Parameters
    ----------
    V : np.ndarray, shape (N, d)
        Matrix whose columns are basis vectors.
    lambda_param : float
        Regularization hyperparameter.
    Returns
    -------
    reg_val : float
        Value of regularization function.
    """
    
    d = V.shape[1]
    reg_val = lambda_param * np.sum((np.dot(V.T, V) - np.eye(d))**2) 
    return reg_val


def build_loss(cross_cov_mats, d, lambda_param=10):
    """Constructs a loss function which gives the (negative) predictive information
    in the projection of multidimensional timeseries data X onto a d-dimensional
    basis, where predictive information is computed using a stationary Gaussian 
    process approximation.
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
        reg_val = reg_fn(V, lambda_param)    
        return -calc_pi_from_cross_cov_mats(cross_cov_mats, V) + reg_val

    return loss


def run_cca(cross_cov_mats, d, init="random", method="BFGS", tol=1e-6, lambda_param=10, verbose=False):
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

    loss = build_loss(cross_cov_mats, d, lambda_param=lambda_param)
    grad_loss = grad(loss)
    
    N = cross_cov_mats.shape[1] #or cross_cov_mats.shape[2]
    
    if verbose:
        def callback(V_flat):
            loss_val = loss(V_flat) 
            V = V_flat.reshape((N, d))
            reg_val = reg_fn(V, lambda_param)
            loss_no_reg = loss_val - reg_val
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
            V_init = V_init + np.random.normal(0, 1e-3, V_init.shape)
    elif type(init) == np.ndarray:
        V_init = init

    opt_result = scipy.optimize.minimize(loss, V_init.flatten(), method=method, jac=grad_loss, callback=callback, tol=tol)
    V_opt_flat = opt_result["x"]
    V_opt = V_opt_flat.reshape((N, d))
    
    #Orhtonormalize the basis prior to returning it
    V_opt = scipy.linalg.orth(V_opt)

    return V_opt


