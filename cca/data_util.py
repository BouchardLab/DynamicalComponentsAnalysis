import numpy as np
import h5py
from math import ceil, floor
import pickle

def sum_over_chunks(X, stride):
    X_trunc = X[:len(X)-(len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc)//stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed

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

def load_neural_data(filename, bin_width_s=0.1):
    file = open(filename, "rb")
    data = pickle.load(file)
    X, Y = data[0], data[1]
    good_X_idx = (1 - (np.isnan(X[:, 0]) + np.isnan(X[:, 1]))).astype(np.bool)
    good_Y_idx = (1 - (np.isnan(Y[:, 0]) + np.isnan(Y[:, 1]))).astype(np.bool)
    good_idx = good_X_idx*good_Y_idx
    X, Y = X[good_idx], Y[good_idx]
    chunk_size = int(np.round(bin_width_s / 0.05)) #50 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size)
    return X, Y

def load_mocap_data(filename, z_score=True):
	angles = []
	with open(filename) as f:
		#Skip header
		line = f.readline().strip()
		while line != ":DEGREES":
			line = f.readline().strip()
		#Parse body
		line = f.readline().strip()
		cur_angles = None
		while line:
			if line.isdigit():
				#New time-step
				if cur_angles is not None:
					angles.append(cur_angles)
				cur_angles = []
			else:
				#Continue adding angles to current time-step
				parts = line.split(" ")
				joint_name = parts[0]
				joint_angles = [float(angle_str) for angle_str in parts[1:]]
				cur_angles += joint_angles
			line = f.readline().strip()
	angles = np.array(angles)
	return angles

class CrossValidate:
    def __init__(self, X, Y, num_folds):
        self.X, self.Y = X, Y
        self.num_folds = num_folds
        self.fold_size = len(X) // num_folds

    def __iter__(self):
        self.fold_idx = 0
        return self

    def __next__(self):
        fold_idx, fold_size = self.fold_idx, self.fold_size
        if fold_idx == self.num_folds:
            raise StopIteration
            
        i1 = fold_idx*fold_size
        i2 = (fold_idx + 1)*fold_size
        
        X, Y = self.X, self.Y
        X_test = X[i1:i2]
        Y_test = Y[i1:i2]
        X_train = np.vstack((X[:i1], X[i2:]))
        Y_train = np.vstack((Y[:i1], Y[i2:]))
        
        self.fold_idx += 1
        return X_train, X_test, Y_train, Y_test, fold_idx







