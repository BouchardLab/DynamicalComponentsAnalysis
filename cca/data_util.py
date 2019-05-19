import h5py
from math import ceil, floor
import pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d

def form_lag_matrix(X, T, stride=1):
    N = X.shape[1]
    num_lagged_samples = floor((len(X) - T)/stride) + 1 #number of lagged samples
    X_with_lags = np.zeros((num_lagged_samples, T*N))
    for i in range(num_lagged_samples):
        X_with_lags[i, :] = X[i*stride : i*stride + T, :].flatten()
    return X_with_lags

def sum_over_chunks(X, stride):
    X_trunc = X[:len(X)-(len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc)//stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed

def moving_center(X, n):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd

def calc_autocorr_fns(X, T):
	autocorr_fns = np.zeros((X.shape[1], T))
	for dt in range(T):
		autocorr_fns[:, dt] = np.sum((X[dt:]*X[:len(X)-dt]), axis=0)/(len(X)-dt)
	return autocorr_fns

def load_kording_paper_data(filename, bin_width_s=0.1, min_spike_count=10,
                            preprocess=True):
    file = open(filename, "rb")
    data = pickle.load(file)
    X, Y = data[0], data[1]
    good_X_idx = (1 - (np.isnan(X[:, 0]) + np.isnan(X[:, 1]))).astype(np.bool)
    good_Y_idx = (1 - (np.isnan(Y[:, 0]) + np.isnan(Y[:, 1]))).astype(np.bool)
    good_idx = good_X_idx*good_Y_idx
    X, Y = X[good_idx], Y[good_idx]
    chunk_size = int(np.round(bin_width_s / 0.05)) #50 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size)/chunk_size
    X = X[:, np.sum(X, axis=0) > min_spike_count]
    if preprocess:
        X = np.sqrt(X)
        X = moving_center(X, n=600)
        Y -= Y.mean(axis=0, keepdims=True)
        Y /= Y.std(axis=0, keepdims=True)
    return {'neural': X, 'loc': Y}

def load_weather_data(filename):
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df = df[['Vancouver', 'Portland', 'San Francisco', 'Seattle',
           'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque',
           'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City',
           'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis',
           'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami',
           'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
           'Boston']]
    df = df.dropna(axis=0, how='any')
    dts = (df.index[1:] - df.index[:-1]).to_numpy()
    df = df.iloc[np.nonzero(dts > dts.min())[0].max() + 1:]
    Xfs = df.values.copy()
    ds_factor = 24
    X = resample(Xfs, Xfs.shape[0] // ds_factor, axis=0)
    return X

"""
Download .mat files from
https://zenodo.org/record/583331#.XNtzE5NKjys
Longest session (only has M1): indy_20160627_01.mat

TODO: use downsampling w/ scipy.signal instead of decimation
"""
def load_sabes_data(filename, bin_width_s=.05, preprocess=True):
    #Load MATLAB file
    with h5py.File(filename, "r") as f:
        #Get channel names (e.g. M1 001 or S1 001)
        num_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(num_channels):
            chan_names.append(f[f['chan_names'][0,i]][()].tobytes()[::2].decode())
        #Get M1 and S1 indices
        M1_indices = [i for i in range(num_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(num_channels) if chan_names[i].split(' ')[0] == 'S1']
        #Get time
        t = f['t'][0,:]
        #Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            #Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            #Perform binning
            num_channels = len(indices)
            num_sorted_units = f["spikes"].shape[0] - 1 #The FIRST one is the 'hash' -- ignore!
            d = num_channels * num_sorted_units #d is the output dimension (total # of sorted units)
            max_t = t[-1]
            num_bins = int(np.floor((max_t - t[0]) / bin_width_s))
            binned_spikes = np.zeros((num_bins, d), dtype=np.int)
            for chan_idx in indices: #0,...,95, for example
                for unit_idx in range(1, num_sorted_units): #ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        #ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :] #flatten
                    spike_times = spike_times[spike_times - t[0] < num_bins * bin_width_s] #get rid of extraneous t vals
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int)
                    bin_idx_unique, counts = np.unique(bin_idx, return_counts=True)
                    #make sure to ignore the hash here...
                    binned_spikes[bin_idx_unique, chan_idx * num_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 0]
            if preprocess:
                binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > 5000]
                binned_spikes = np.sqrt(binned_spikes)
                binned_spikes = moving_center(binned_spikes, n=600)
            result[region] = binned_spikes
        #Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        #Line up the binned spikes with the cursor data
        t_mid_bin = np.arange(len(binned_spikes))*bin_width_s + bin_width_s/2
        cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        cursor_interp = cursor_pos_interp(t_mid_bin)
        if preprocess:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        result["cursor"] = cursor_interp
        return result

def load_accel_data(filename, preprocess=True):
    df = pd.read_csv(filename)
    X = df.values
    if preprocess:
        X -= X.mean(axis=0, keepdims=True)
        X /= X.std(axis=0, keepdims=True)
    return X

class CrossValidate:
    def __init__(self, X, Y, num_folds, stack=True):
        self.X, self.Y = X, Y
        self.num_folds = num_folds
        self.fold_size = len(X) // num_folds
        self.stack = stack

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
        if self.stack:
            X_train = np.concatenate((X[:i1], X[i2:]))
            Y_train = np.concatenate((Y[:i1], Y[i2:]))
        else:
            X_train = [X[:i1], X[i2:]]
            Y_train = [Y[:i1], Y[i2:]]

        self.fold_idx += 1
        return X_train, X_test, Y_train, Y_test, fold_idx
