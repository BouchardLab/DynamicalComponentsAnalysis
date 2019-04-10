import numpy as np
import h5py
from math import ceil

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

def load_sabes_data(filename, bin_width_s=0.1, session=None):
    f = h5py.File(filename, "r")
    sessions = list(f.keys())
    lengths = np.array([f[session]["M1"]["spikes"].shape[0] for session in sessions])
    if session is None:
    	#Use longest session
    	session = sessions[np.argsort(lengths)[::-1][0]]
    X, Y = f[session]["M1"]["spikes"], f[session]["cursor"]
    chunk_size = int(np.round(bin_width_s / .05))
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size)/chunk_size
    return X, Y

def load_miller_data(filename, bin_width_s=0.1):
    file = open(filename, "rb")
    data = pickle.load(file)
    X, Y = data[0], data[1]
    X, Y = X[200:-200, :], Y[200:-200, :]
    chunk_size = int(np.round(bin_width_s / .05))
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size)/chunk_size
    return X, Y