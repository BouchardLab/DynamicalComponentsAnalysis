import numpy as np
import scipy.linalg
import collections
import functools
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt

def disp_cov(cov, title):
    plt.imshow(cov, vmin=-1, vmax=1, cmap="RdGy")
    plt.colorbar()
    plt.title(title, fontsize=16)
    plt.show()

class memoized(object):
   """Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   From: https://wiki.python.org/moin/PythonDecoratorLibrary.
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
    n = int(np.floor((len(X) - num_lags)/skip) + 1)
    N = X.shape[1]
    X_with_lags = np.zeros((n, N*num_lags))
    for i in range(n):
        X_with_lags[i, :] = X[i*skip : i*skip + num_lags, :].ravel()
    return X_with_lags

def cv_toeplitz(X_with_lags, N, T, num_folds=10, max_r=10, small_eigval=1e-6):
    fold_size = int(np.floor(len(X_with_lags)/num_folds))
    d = N*T
    P = build_P(T)
    log_likelihood_vals = np.zeros((num_folds, max_r))

    for cv_iter in range(num_folds):

        X_train = np.concatenate((X_with_lags[:cv_iter*fold_size], X_with_lags[(cv_iter+1)*fold_size:]), axis=0)
        X_test = X_with_lags[cv_iter*fold_size : (cv_iter+1)*fold_size]
        num_samples = len(X_test)

        X_train_ctd = X_train - X_train.mean(axis=0)
        X_test_ctd = X_test - X_test.mean(axis=0)
        cov_train = np.dot(X_train_ctd.T, X_train_ctd)/len(X_train)
        cov_test = np.dot(X_test_ctd.T, X_test_ctd)/len(X_test)

        R_C = pv_rearrange(cov_train, N, T)
        to_svd = P.dot(R_C)
        U, s, Vt = randomized_svd(to_svd, n_components=max_r, n_iter=20, random_state=42)

        for r_idx in range(max_r):

            r = r_idx + 1
            trunc_svd = U[:, :r].dot(np.diag(s[:r])).dot(Vt[:r, :])
            cov_reg_train = pv_rearrange_inv(P.T.dot(trunc_svd), N, T)

            w, _ = np.linalg.eigh(cov_reg_train)
            min_eig = np.min(w)
            if min_eig <= small_eigval:
                #print("CV eigvall < 0:", min_eig, "r =", r)
                cov_reg_train = cov_reg_train + (-min_eig + small_eigval)*np.eye(N*T)

            cov_reg_train_inv = np.linalg.inv(cov_reg_train)
            log_det_cov_reg_train = np.linalg.slogdet(cov_reg_train)[1]
            log_likelihood = -0.5*num_samples*(d*np.log(2*np.pi) + log_det_cov_reg_train + np.trace(np.dot(cov_reg_train_inv, cov_test)))
            log_likelihood_vals[cv_iter, r_idx] = log_likelihood

    X_ctd = X_with_lags - X_with_lags.mean(axis=0)
    cov = np.dot(X_ctd.T, X_ctd)/len(X_with_lags)
    opt_r = np.argmax(log_likelihood_vals.mean(axis=0)) + 1
    print(opt_r)
    R_C = pv_rearrange(cov, N, T)
    to_svd = P.dot(R_C)
    U, s, Vt = randomized_svd(to_svd, n_components=opt_r, n_iter=20, random_state=42)
    trunc_svd = U.dot(np.diag(s)).dot(Vt)
    cov_reg = pv_rearrange_inv(P.T.dot(trunc_svd), N, T)
    w, _ = np.linalg.eigh(cov_reg)
    min_eig = np.min(w)
    if min_eig <= small_eigval:
        #print("Final eigval < 0:", min_eig)
        cov_reg = cov_reg + (-min_eig + small_eigval)*np.eye(N*T)

    return log_likelihood_vals, cov_reg
