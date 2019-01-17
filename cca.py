import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from autograd import grad

def calc_cross_cov_mats(X, delta_t_max):
    
    #mean-center data
    X = X - np.mean(X, axis=0)

    #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=delta_t_max-1
    N = X.shape[1]
    cross_cov_mats = np.zeros((2*delta_t_max, N, N))
    for delta_t in range(2*delta_t_max):
        cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
        cross_cov_mats[delta_t] = cross_cov
        
    return cross_cov_mats
    

def calc_pi(X, T):

    cross_cov_mats = calc_cross_cov_mats(X, 2*T)
    N = X.shape[1]

    cross_cov_mats_repeated = []
    for i in range(2*T):
        for j in range(2*T):
            cross_cov = cross_cov_mats[int(np.abs(i-j))]
            cross_cov_mats_repeated.append(cross_cov)

    cov_2T_tensor = np.reshape(np.array(cross_cov_mats_repeated), (2*T, 2*T, N, N))
    cov_2T = np.concatenate([np.concatenate(cov_ii, axis=1) for cov_ii in cov_2T_tensor])
    cov_T = cov_2T[:T*N, :T*N]

    sgn_T, logdet_T = np.linalg.slogdet(cov_T)
    sgn_2T, logdet_2T = np.linalg.slogdet(cov_2T)
    PI = 2*logdet_T - logdet_2T
    
    return PI


def build_loss(X, T, d):
            
    cross_cov_mats = calc_cross_cov_mats(X, 2*T)
    N = X.shape[1]
    
    def loss(V_flat):
        
        #Reshape and normalize V_flat
        V_unnormalized = V_flat.reshape(N, d)
        V = V_unnormalized / np.sqrt(np.sum(V_unnormalized**2, axis=0))
        
        cross_cov_mats_repeated = []
        for i in range(2*T):
            for j in range(2*T):
                cross_cov = cross_cov_mats[int(np.abs(i-j))]
                cross_cov_proj = np.dot(V.T, np.dot(cross_cov, V))
                cross_cov_mats_repeated.append(cross_cov_proj)

        cov_2T_tensor = np.reshape(np.array(cross_cov_mats_repeated), (2*T, 2*T, d, d))
        cov_2T = np.concatenate([np.concatenate(cov_ii, axis=1) for cov_ii in cov_2T_tensor])
        cov_T = cov_2T[:T*d, :T*d]

        sgn_T, logdet_T = np.linalg.slogdet(cov_T)
        sgn_2T, logdet_2T = np.linalg.slogdet(cov_2T)
        PI = 2*logdet_T - logdet_2T

        return -PI

    return loss


def run_cca(X, T, d):
    
    loss = build_loss(X, T, d)
    grad_loss = grad(loss)
    
    N = X.shape[1]
    V_init = np.random.normal(0, 1, (N, d))
    V_init = V_init / np.sqrt(np.sum(V_init**2, axis=0))

    opt_result = scipy.optimize.minimize(loss, V_init.flatten(), method='BFGS', jac=grad_loss)
    V_opt_flat = opt_result["x"]
    V_opt = V_opt_flat.reshape((N, d))
    V_opt = V_opt / np.sqrt(np.sum(V_opt**2, axis=0))
    
    return V_opt




