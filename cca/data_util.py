import numpy as np
import scipy
from math import floor, ceil
import cca.kf_util as kf_util

def sum_over_chunks(X, stride):
    zero_padded = np.zeros((stride*ceil(X.shape[0]/stride), X.shape[1]))
    zero_padded[:len(X), :] = X
    reshaped = zero_padded.reshape((int(len(zero_padded)/stride), stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed[:-1]

def get_active_channels(X, window_size, min_count):
	good_idx = np.ones(X.shape[1], dtype=np.bool)
	for i in range(len(X)):
		if i - (window_size // 2) < 0:
			start, end = 0, window_size
		elif i + (window_size // 2) > len(X):
			start, end = len(X) - window_size, len(X)
		else:
			start, end = i - window_size//2, i + window_size//2
		good_idx *= (X[start:end, :].sum(axis=0) > min_count)
	return good_idx

def sliding_z_score(X, window_size):
	N = X.shape[1]
	X_ctd = np.zeros_like(X)
	for i in range(len(X)):
		if i - (window_size // 2) < 0:
			start, end = 0, window_size
		elif i + (window_size // 2) > len(X):
			start, end = len(X) - window_size, len(X)
		else:
			start, end = i - window_size//2, i + window_size//2
		mu, sigma = X[start:end, :].mean(axis=0), X[start:end, :].std(axis=0)
		X_ctd[i, :] = (X[i] - mu)/sigma

	X_ctd = (X_ctd - X_ctd.mean(axis=0))/X_ctd.std(axis=0)
	return X_ctd

def calc_autocorr_fns(X, T):
	autocorr_fns = np.zeros((X.shape[1], T))
	for dt in range(T):
		autocorr_fns[:, dt] = np.sum((X[dt:]*X[:len(X)-dt]), axis=0)/(len(X)-dt)
	return autocorr_fns

def cv_decoding_score(X, Y, num_folds=5, method="kf"):
    fold_size = int(np.floor(len(X)/num_folds))
    scores = np.zeros(num_folds)
    for fold_idx in range(num_folds):
        i1 = fold_size*fold_idx
        i2 = fold_size*(fold_idx+1)
        Y_train, Y_test = np.vstack((Y[:i1], Y[i2:])), Y[i1:i2]
        X_train, X_test = np.vstack((X[:i1], X[i2:])), X[i1:i2]
        if method == "kf":
            score = decoding_score_kf(X_train, X_test, Y_train, Y_test)
        elif method == "linear":
            score = decoding_score_linear(X_train, X_test, Y_train, Y_test)
        scores[fold_idx] = score
    return np.mean(scores)

def decoding_score_kf(X_train, X_test, Y_train, Y_test):
	A, H, Q, R = kf_util.fit_kf_2d_cursor([Y_train], [X_train], 1.0)
	x_hat_init = np.asarray([0, 0, 0, 0, 1])
	A_sskf, B_sskf = kf_util.steady_state_kf_matrices(A, H, Q, R)
	decoded = kf_util.run_steady_state_kf(A_sskf, B_sskf, X_test, x_hat_init)
	r_xy = [scipy.stats.pearsonr(Y_test[:, i], decoded[:, i])[0] for i in range(2)]
	return np.mean(r_xy)

def decoding_score_linear(X_train, X_test, Y_train, Y_test):
	X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
	X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
	beta = np.dot(np.linalg.inv( np.dot(X_train.T, X_train) ), np.dot(X_train.T, Y_train))
	pred_Y = np.dot(X_test, beta)
	#err = np.mean((pred_Y - Y_test)**2)
	r_xy = [scipy.stats.pearsonr(Y_test[:, i], pred_Y[:, i])[0] for i in range(2)]
	return np.mean(r_xy)





