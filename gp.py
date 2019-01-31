import numpy as np
import scipy
import cca


def gen_gp_cov(T, N, kernel):
    """Generates a N*T-by-N*T covariance matrix for a spatiotemporal Gaussian
    process (2D Gaussian random field) with a provided kernel.
    Parameters
    ----------
    T : int
        Number of time-steps.
    N : int
        Number of spatial steps.
    kernel : function
        Should be of the form kernel = K(t1, t2, x1, x2).
        The kernel can choose to imlement temporal or spatial stationarity,
        however this is not enfored.
    Returns
    -------
    C : np.ndarray, shape (N*T, N*T)
        Covariance matrix for the Gaussian process. Time is the "outer" variable
        and space is the "inner" variable.
    """

    t1, t2, x1, x2 = np.arange(T), np.arange(T), np.arange(N), np.arange(N)
    t1, t2, x1, x2 = np.meshgrid(t1, t2, x1, x2, indexing="ij")
    C = kernel(t1, t2, x1, x2)
    C = C.swapaxes(1,2).reshape(N*T, N*T)

    return C


def calc_pi_for_gp(T, N, kernel):
    """Calculates the predictive information in a spatiotemporal Gaussian process
    with a given kernel.
    Parameters
    ----------
    T : int
        Length of temporal windows accross which to compute mutual information.
    N : int
        Number of spatial steps in teh Gaussian process.
    kernel : function
        Should be of the form kernel = K(t1, t2, x1, x2).
        The kernel can choose to imlement temporal or spatial stationarity,
        however this is not enfored.
    Returns
    -------
    PI : float
        (Temporal) predictive information in the Gaussian process.
    """

    cov_2T = gen_gp_cov(2*T, N, kernel)
    cov_T = cov_2T[:N*T, :N*T]
    sgn_T, logdet_T = np.linalg.slogdet(cov_T)
    sgn_2T, logdet_2T = np.linalg.slogdet(cov_2T)
    PI = (2*logdet_T - logdet_2T)/np.log(2)

    return PI


def gen_kernel(kernel_type, spatial_scale, temporal_scale):
    """Generates a specified type of Kernel for a spatiotemporal Gaussian
    process.
    Parameters
    ----------
    kernel_type : string
        'squared_exp' or 'exp'
    spatial_scale : float
        Spatial autocorrelation scale.
    temporal_scale : float
        Temporal autocorrelation scale.
    Returns
    -------
    K : function
        Kernel of the form K(t1, t2, x1, x2).
    """

    if kernel_type == "squared_exp":
        def K(t1, t2, x1, x2):
            return np.exp(-(t1-t2)**2/temporal_scale**2 - (x1-x2)**2/spatial_scale**2)
    elif kernel_type == "exp":
        def K(t1, t2, x1, x2):
            return np.exp(-np.abs(t1-t2)/temporal_scale - np.abs(x1-x2)/spatial_scale)
    return K


def sample_gp(T, N, kernel, num_to_concat=1):
    """Draw a sample from a spatiotemporal Gaussian process.
    Parameters
    ----------
    T : int
        Length in time of sample.
    N : int
        Size in space of sample.
    kernel : function
        Kernel of the form K(t1, t2, x1, x2).
    num_to_concat : int
        Number of samples of lenght T to concatenate before returning the result.
    Returns
    -------
    sample : np.ndarray, size (T*num_to_concat, N)
        Sample from the Gaussian process.
    """

    t1, t2, x1, x2 = np.arange(T), np.arange(T), np.arange(N), np.arange(N)
    t1, t2, x1, x2 = np.meshgrid(t1, t2, x1, x2, indexing="ij")
    C = kernel(t1, t2, x1, x2)
    C = C.swapaxes(1,2).reshape(N*T, N*T)

    sample = np.concatenate(np.random.multivariate_normal(mean=np.zeros(C.shape[0]), cov=C, size=num_to_concat))
    sample = sample.reshape(T*num_to_concat, N)

    return sample

def embed_gp(T, N, d, kernel, noise_cov, T_pi, num_to_concat=1):
    """Embed a d-dimensional Gaussian process into N-dimensional space, then
    add (potentially) spatially structured white noise.
    ----------
    T : int
        Length in time.
    N : int
        Ambient dimension.
    d : int
        Gaussian process dimension.
    kernel : function
        Kernel of the form K(t1, t2, x1, x2).
    noise_cov : np.ndarray, shape (N, N)
        Covariance matrix from which to sampel Gaussian noise to add to each
        time point in an iid fashion.
    num_to_concat : int
        Number of samples of lenght T to concatenate before returning the result.
    Returns
    -------
    X : np.ndarray, size (T*num_to_concat, N)
        Embedding of GP into high-dimensional space, plus noise.
    """

    #Latent dynamics
    Y = sample_gp(T, d, kernel, num_to_concat)

    #Random orthogonal embedding matrix U
    U = scipy.stats.ortho_group.rvs(N)[:, :d]

    #Data matrix X
    X = np.dot(Y, U.T)

    #Corrupt data by spatially structured white noise
    X += np.random.multivariate_normal(mean=np.zeros(N), cov=noise_cov, size=T*num_to_concat)

    #Compute the PI of the high-dimensional, noisy process
    cov_low_d = gen_gp_cov(T=2*T_pi, N=d, kernel=kernel)
    low_d_cross_cov_mats = cca.calc_cross_cov_mats_from_cov(N=d, num_lags=2*T_pi, cov=cov_low_d)
    high_d_cross_cov_mats = np.array([np.dot(U, np.dot(C, U.T)) for C in low_d_cross_cov_mats])
    high_d_cross_cov_mats[0] += noise_cov
    cov_high_d = cca.calc_cov_from_cross_cov_mats(high_d_cross_cov_mats)
    full_pi = cca.calc_pi_from_cov(cov_high_d)

    embedding_cross_cov_mats = np.array([np.dot(U.T, np.dot(C, U)) for C in high_d_cross_cov_mats])
    cov_embedding = cca.calc_cov_from_cross_cov_mats(embedding_cross_cov_mats)
    embedding_pi = cca.calc_pi_from_cov(cov_embedding)

    return X, U , full_pi, embedding_pi
