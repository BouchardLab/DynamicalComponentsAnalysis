import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
from autograd import grad

def calc_pi(X, T):

	#mean-center data
	X = X - np.mean(X, axis=0)

	#Compute N-by-N cross-covariance matrices for all 0<=delta_t<=2*T-1
	N = X.shape[1]
	cross_cov_mats = np.zeros((2*T, N, N))
	for delta_t in range(2*T):
	    cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
	    cross_cov_mats[delta_t] = cross_cov

	vals = []
	for i in range(2*T):
		for j in range(2*T):
			C = cross_cov_mats[int(np.abs(i-j))]
			vals.append(C)

	autocov_2T_tensor = np.reshape(np.array(vals), (2*T, 2*T, N, N))
	autocov_2T = np.concatenate([np.concatenate(autocov_ii, axis=1) for autocov_ii in autocov_2T_tensor])
	autocov_T = autocov_2T[:T * N, :T * N]

	sgn_T, logdet_T = np.linalg.slogdet(autocov_T)
	sgn_2T, logdet_2T = np.linalg.slogdet(autocov_2T)

	PI = 2*logdet_T - logdet_2T
	return PI






def build_loss(X, T, latent_dim, kron=False):
        
    #mean-center data
    X = X - np.mean(X, axis=0)
    
    #Compute N-by-N cross-covariance matrices for all 0<=delta_t<=2*T-1
    N = X.shape[1]
    cross_cov_mats = np.zeros((2*T, N, N))
    for delta_t in range(2*T):
        cross_cov = np.dot(X[delta_t:].T, X[:len(X)-delta_t])/(len(X) - delta_t - 1)
        cross_cov_mats[delta_t] = cross_cov
        
    #Compute PI using big matrices + Kronecker products
    if kron:
        #Compute 2*T*N-by-2*T*N cross-covariance matrix
        C_2T = np.zeros((2*T*N, 2*T*N))
        for delta_t in range(2*T):
            #upper diagonal
            C_2T[:2*T*N-delta_t*N, delta_t*N:] += np.kron(np.eye(2*T - delta_t), cross_cov_mats[delta_t])

            #upper diag = lower diag if delta_t == 0
            if delta_t == 0:
                continue

            #lower-diagonal
            C_2T[delta_t*N:, :2*T*N-delta_t*N] += np.kron(np.eye(2*T - delta_t), cross_cov_mats[delta_t])

        #Get T*N-by-T*N cross-covariance matrix
        C_T = C_2T[:T*N, :T*N]

        #Define loss functon
        def loss(v):

            v_column = np.reshape(v, (len(v), 1))
            v_norm = v_column / np.sqrt(np.sum(v_column**2))

            V_T = np.kron(np.eye(T), v_norm)
            V_2T = np.kron(np.eye(2*T), v_norm)

            autocov_T = np.dot(V_T.T, np.dot(C_T, V_T))
            autocov_2T = np.dot(V_2T.T, np.dot(C_2T, V_2T))

            sgn_T, logdet_T = np.linalg.slogdet(autocov_T)
            sgn_2T, logdet_2T = np.linalg.slogdet(autocov_2T)

            PI = 2*logdet_T - logdet_2T
            return -PI #Loss function (low negative PI = high PI)

        print("Kron loss!")
        return loss
    
    #Compute PI using a Python list --> matrix
    else:
        def loss(v):
            vals = []
            N = X.shape[1]
            v2d_notnorm = v.reshape(latent_dim, N)
            v2d = v2d_notnorm / np.sqrt(np.sum(v2d_notnorm**2, axis=1, keepdims=True))
            for i in range(2*T):
                for j in range(2*T):
                    C = cross_cov_mats[int(np.abs(i-j))]
                    val = np.dot(v2d, np.dot(C, v2d.T))
                    vals.append(val)

            autocov_2T_tensor = np.reshape(np.array(vals), (2*T, 2*T, latent_dim, latent_dim))
            autocov_2T = np.concatenate([np.concatenate(autocov_ii, axis=1) for autocov_ii in autocov_2T_tensor])
            autocov_T = autocov_2T[:T * latent_dim, :T * latent_dim]
            
            sgn_T, logdet_T = np.linalg.slogdet(autocov_T)
            sgn_2T, logdet_2T = np.linalg.slogdet(autocov_2T)
            
            PI = 2*logdet_T - logdet_2T
            return -PI
        
        print("List loss!")
        return loss