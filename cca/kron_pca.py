import numpy as np
import scipy.linalg
import collections
import functools
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from math import floor

class memoized(object):
   """Decorator for memoization.
   From: https://wiki.python.org/moin/PythonDecoratorLibrary.

   Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      #Return the function's docstring.
      return self.func.__doc__
   def __get__(self, obj, objtype):
      #Support instance methods.
      return functools.partial(self.__call__, obj)

@memoized
def pv_permutation(N, T):
    I = np.arange(T**2 * N**2, dtype=np.int).reshape((N*T, N*T))
    I_perm = np.zeros(( T**2, N**2 ), dtype=np.int)
    for i in range(T):
        for j in range(T):
            row_idx =  i*T + j
            I_block = I[i*N:(i+1)*N, j*N:(j+1)*N]
            I_perm[row_idx, :] = I_block.T.reshape((N**2,)) #equivalent to I_block.vectorize
    perm = I_perm.ravel()
    perm_inv = perm.argsort()
    return perm, perm_inv

def pv_rearrange(C, N, T):
    perm, _ = pv_permutation(N, T)
    C_prime = C.ravel()[perm].reshape((T**2, N**2))
    return C_prime

def pv_rearrange_inv(C, N, T):
    _, perm_inv = pv_permutation(N, T)
    C_prime = C.ravel()[perm_inv].reshape((N*T, N*T))
    return C_prime

def build_P(T):
    P = np.zeros((2*T - 1, T**2))
    idx = np.arange(T**2).reshape((T, T)).T + 1
    for offset in range(-T+1, T):
        diag_idx = np.diagonal(idx, offset=offset)
        P[offset+T-1, diag_idx-1] = 1/np.sqrt(T - np.abs(offset))
    return P

def toeplitzify(C, N, T):
    C_toep = np.zeros((N*T, N*T))
    for delta_t in range(T):
        to_avg_lower = np.zeros((T-delta_t, N, N))
        to_avg_upper = np.zeros((T-delta_t, N, N))

        for i in range(T-delta_t):
            to_avg_lower[i, :, :] = C[(delta_t + i)*N : (delta_t + i + 1)*N, i*N : (i + 1)*N]
            to_avg_upper[i, :, :] = C[i*N : (i + 1)*N, (delta_t + i)*N : (delta_t + i + 1)*N]

        block_avg = 0.5*(np.mean(to_avg_lower, axis=0) + np.mean(to_avg_upper, axis=0).T)

        for i in range(T-delta_t):
            C_toep[(delta_t + i)*N : (delta_t + i + 1)*N, i*N : (i + 1)*N] = block_avg
            C_toep[i*N : (i + 1)*N, (delta_t + i)*N : (delta_t + i + 1)*N] = block_avg.T

    return C_toep

def form_lag_matrix(X, num_lags, skip=1):
    n = floor((len(X) - num_lags)/skip)
    N = X.shape[1]
    X_with_lags = np.zeros((n, N*num_lags))
    for i in range(n):
        X_with_lags[i, :] = X[i*skip : i*skip + num_lags, :].ravel()
    return X_with_lags

def calc_cov_local_mean(X, bin_size, stride=1):
    num_windows = floor((len(X) - bin_size) / stride)
    N = X.shape[1]
    cov_sum = np.zeros((N, N))
    for i in range(num_windows):
        X_window = np.copy(X[i*stride : bin_size + i*stride])
        X_window -= X_window.mean(axis=0)
        cov_sum += np.dot(X_window.T, X_window)/len(X_window)
    return cov_sum / num_windows

def toeplitz_reg(cov, N, T, r):
	R_C = pv_rearrange(cov, N, T)
	P = build_P(T)
	to_svd = P.dot(R_C)
	U, s, Vt = randomized_svd(to_svd, n_components=r, n_iter=40, random_state=42)
	trunc_svd = U.dot(np.diag(s)).dot(Vt)
	cov_reg = pv_rearrange_inv(P.T.dot(trunc_svd), N, T)

	return cov_reg

def toeplitz_reg_and_taper(cov, N, T, r, sigma):
    cov_reg = toeplitz_reg_threshold(cov, N, T, r)
    cov_reg_taper = taper_cov(cov_reg, N, T, sigma)
    return cov_reg_taper

def toeplitz_reg_taper_shrink(cov, N, T, r, sigma, alpha):
    cov_reg = toeplitz_reg_threshold(cov, N, T, r)
    cov_reg_taper = taper_cov(cov_reg, N, T, sigma)
    cov_reg_taper_shrink = (1. - alpha)*cov_reg_taper + alpha*np.eye(N*T)
    return cov_reg_taper_shrink

def non_toeplitz_reg(cov, N, T, r):
	R_C = pv_rearrange(cov, N, T)
	U, s, Vt = randomized_svd(R_C, n_components=r, n_iter=40, random_state=42)
	trunc_svd = U.dot(np.diag(s)).dot(Vt)
	cov_reg = pv_rearrange_inv(trunc_svd, N, T)
	return cov_reg

def toeplitz_reg_threshold(cov, N, T, r):
    R_C = pv_rearrange(cov, N, T)
    P = build_P(T)
    to_svd = P.dot(R_C)
    U, s, Vt = randomized_svd(to_svd, n_components=r+1, n_iter=40, random_state=42)
    trunc_svd = U[:, :-1].dot(np.diag(s[:-1] - s[-1])).dot(Vt[:-1, :])
    cov_reg = pv_rearrange_inv(P.T.dot(trunc_svd), N, T)
    return cov_reg

def non_toeplitz_reg_threshold(cov, N, T, r):
    R_C = pv_rearrange(cov, N, T)
    U, s, Vt = randomized_svd(R_C, n_components=r+1, n_iter=40, random_state=42)
    trunc_svd = U[:, :-1].dot(np.diag(s[:-1] - s[-1])).dot(Vt[:-1, :])
    cov_reg = pv_rearrange_inv(trunc_svd, N, T)
    return cov_reg

def gaussian_log_likelihood(cov, sample_cov, num_samples):
    to_trace = np.linalg.solve(cov, sample_cov)
    log_det_cov = np.linalg.slogdet(cov)[1]
    d = cov.shape[1]
    log_likelihood = -0.5*num_samples*(d*np.log(2*np.pi) + log_det_cov + np.trace(to_trace))
    return log_likelihood

def taper_cov(cov, N, T, sigma):
    t = np.arange(T).reshape((T, 1))
    delta_t = t - t.T
    temporal_kernel = np.exp(-(delta_t / sigma)**2)
    full_kernel = np.kron(temporal_kernel, np.ones((N, N)))
    result = full_kernel * cov
    return result

def cv_toeplitz(X_with_lags, N, T, r_vals, sigma_vals, alpha_vals, num_folds=10):
    fold_size = int(np.floor(len(X_with_lags)/num_folds))
    d = N*T
    P = build_P(T)
    ll_vals = np.zeros((num_folds, len(r_vals), len(sigma_vals), len(alpha_vals)))

    for cv_iter in range(num_folds):
        print("fold =", cv_iter+1)

        X_train = np.concatenate((X_with_lags[:cv_iter*fold_size], X_with_lags[(cv_iter+1)*fold_size:]), axis=0)
        X_test = X_with_lags[cv_iter*fold_size : (cv_iter+1)*fold_size]
        num_samples = len(X_test)

        cov_train, cov_test = np.cov(X_train.T), np.cov(X_test.T)
        R_C = pv_rearrange(cov_train, N, T)
        to_svd = P.dot(R_C)
        U, s, Vt = randomized_svd(to_svd, n_components=np.max(r_vals), n_iter=40, random_state=42)
        diag_part = cov_train[:N, :N]

        for r_idx in range(len(r_vals)):
            r = r_vals[r_idx]
            print("r =", r)
            if r_idx == len(r_vals) - 1:
                trunc_svd = to_svd
            else:
                trunc_svd = U[:, :r].dot(np.diag(s[:r] - s[r])).dot(Vt[:r, :])
            cov_kron = pv_rearrange_inv(P.T.dot(trunc_svd), N, T)
            for sigma_idx in range(len(sigma_vals)):
                sigma = sigma_vals[sigma_idx]
                cov_kron_tapered = taper_cov(cov_kron, N, T, sigma)
                for alpha_idx in range(len(alpha_vals)):
                    alpha = alpha_vals[alpha_idx]
                    cov_kron_tapered_shrunk = (1. - alpha)*cov_kron_tapered + alpha*np.eye(N*T)
                    ll = gaussian_log_likelihood(cov_kron_tapered_shrunk, cov_test, num_samples)
                    ll_vals[cv_iter, r_idx, sigma_idx, alpha_idx] = ll

    opt_idx = np.unravel_index(ll_vals.mean(axis=0).argmax(), ll_vals.shape[1:])
    return ll_vals, opt_idx


"""
def shrinkage_likelihood(X, alpha):
    n, p = X.shape
    beta = (1 - alpha)/(n - 1)
    S = np.dot(X.T, X)/n
    G_alpha = n*beta*S + alpha*np.eye(p)
    G_alpha_inv = scipy.linalg.inv(G_alpha)
    log_det_G_alpha = np.linalg.slogdet(G_alpha)[1]
    part_1 = 0.5*(p*np.log(2*np.pi) + log_det_G_alpha)
    part_2 = 0
    #n_prime = n
    #random_idx = np.random.choice(np.arange(n), size=n_prime)
    print("Start...")
    for k in range(n):
        x_k = X[k, :]
        r_k = x_k.dot(G_alpha_inv).dot(x_k)
        part_2 += np.log(1 - beta*r_k) + r_k/(1 - beta*r_k)
    ll = part_1 + (1/(2*n))*part_2
    return ll

alpha_vals = np.linspace(0, 1, 20)
ll_vals = np.array([shrinkage_likelihood(X_lags_downsample, alpha) for alpha in alpha_vals])
"""