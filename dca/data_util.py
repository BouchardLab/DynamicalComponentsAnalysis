import h5py, pickle
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample
from scipy.ndimage import convolve1d

from .cov_util import form_lag_matrix  # noqa:F401


def sum_over_chunks(X, stride):
    X_trunc = X[:len(X) - (len(X) % stride)]
    reshaped = X_trunc.reshape((len(X_trunc) // stride, stride, X.shape[1]))
    summed = reshaped.sum(axis=1)
    return summed


def moving_center(X, n, axis=0):
    if n % 2 == 0:
        n += 1
    w = -np.ones(n) / n
    w[n // 2] += 1
    X_ctd = convolve1d(X, w, axis=axis)
    return X_ctd


def calc_autocorr_fns(X, T):
    autocorr_fns = np.zeros((X.shape[1], T))
    for dt in range(T):
        autocorr_fns[:, dt] = np.sum((X[dt:] * X[:len(X) - dt]), axis=0) / (len(X) - dt)
    return autocorr_fns


def load_kording_paper_data(filename, bin_width_s=0.05, min_spike_count=10, preprocess=True):
    with open(filename, "rb") as fname:
        data = pickle.load(fname)
    X, Y = data[0], data[1]
    good_X_idx = (1 - (np.isnan(X[:, 0]) + np.isnan(X[:, 1]))).astype(np.bool)
    good_Y_idx = (1 - (np.isnan(Y[:, 0]) + np.isnan(Y[:, 1]))).astype(np.bool)
    good_idx = good_X_idx * good_Y_idx
    X, Y = X[good_idx], Y[good_idx]
    chunk_size = int(np.round(bin_width_s / 0.05))  # 50 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size) / chunk_size
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


def load_sabes_data(filename, bin_width_s=.05, high_pass=True, sqrt=True, thresh=5000,
                    zscore_pos=True):
    # Load MATLAB file
    with h5py.File(filename, "r") as f:
        # Get channel names (e.g. M1 001 or S1 001)
        n_channels = f['chan_names'].shape[1]
        chan_names = []
        for i in range(n_channels):
            chan_names.append(f[f['chan_names'][0, i]][()].tobytes()[::2].decode())
        # Get M1 and S1 indices
        M1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'M1']
        S1_indices = [i for i in range(n_channels) if chan_names[i].split(' ')[0] == 'S1']
        # Get time
        t = f['t'][0, :]
        # Individually process M1 and S1 indices
        result = {}
        for indices in (M1_indices, S1_indices):
            if len(indices) == 0:
                continue
            # Get region (M1 or S1)
            region = chan_names[indices[0]].split(" ")[0]
            # Perform binning
            n_channels = len(indices)
            n_sorted_units = f["spikes"].shape[0] - 1  # The FIRST one is the 'hash' -- ignore!
            d = n_channels * n_sorted_units
            max_t = t[-1]
            n_bins = int(np.floor((max_t - t[0]) / bin_width_s))
            binned_spikes = np.zeros((n_bins, d), dtype=np.int)
            for chan_idx in indices:
                for unit_idx in range(1, n_sorted_units):  # ignore hash!
                    spike_times = f[f["spikes"][unit_idx, chan_idx]][()]
                    if spike_times.shape == (2,):
                        # ignore this case (no data)
                        continue
                    spike_times = spike_times[0, :]
                    # get rid of extraneous t vals
                    spike_times = spike_times[spike_times - t[0] < n_bins * bin_width_s]
                    bin_idx = np.floor((spike_times - t[0]) / bin_width_s).astype(np.int)
                    unique_idxs, counts = np.unique(bin_idx, return_counts=True)
                    # make sure to ignore the hash here...
                    binned_spikes[unique_idxs, chan_idx * n_sorted_units + unit_idx - 1] += counts
            binned_spikes = binned_spikes[:, binned_spikes.sum(axis=0) > thresh]
            if sqrt:
                binned_spikes = np.sqrt(binned_spikes)
            if high_pass:
                binned_spikes = moving_center(binned_spikes, n=600)
            result[region] = binned_spikes
        # Get cursor position
        cursor_pos = f["cursor_pos"][:].T
        # Line up the binned spikes with the cursor data
        t_mid_bin = np.arange(len(binned_spikes)) * bin_width_s + bin_width_s / 2
        cursor_pos_interp = interp1d(t - t[0], cursor_pos, axis=0)
        cursor_interp = cursor_pos_interp(t_mid_bin)
        if zscore_pos:
            cursor_interp -= cursor_interp.mean(axis=0, keepdims=True)
            cursor_interp /= cursor_interp.std(axis=0, keepdims=True)
        result["cursor"] = cursor_interp
        return result


def load_accel_data(filename, preprocess=True):
    df = pd.read_csv(filename)
    X = df.values[:, 1:]
    if preprocess:
        X -= X.mean(axis=0, keepdims=True)
        X /= X.std(axis=0, keepdims=True)
    return X


class CrossValidate:
    def __init__(self, X, Y, num_folds, stack=True):
        self.X, self.Y = X, Y
        self.num_folds = num_folds
        self.idxs = np.array_split(np.arange(len(X)), num_folds)
        self.stack = stack

    def __iter__(self):
        self.fold_idx = 0
        return self

    def __next__(self):
        fold_idx = self.fold_idx
        if fold_idx == self.num_folds:
            raise StopIteration

        test_idxs = self.idxs[fold_idx]
        train_idxs = []
        if fold_idx > 0:
            train_idxs.append(np.concatenate([self.idxs[ii] for ii in range(fold_idx)]))
        if fold_idx < self.num_folds - 1:
            train_idxs.append(np.concatenate([self.idxs[ii]
                                              for ii in range(fold_idx + 1, self.num_folds)]))

        X, Y = self.X, self.Y
        X_test = X[test_idxs]
        Y_test = Y[test_idxs]
        if self.stack:
            X_train = np.concatenate([X[idxs] for idxs in train_idxs])
            Y_train = np.concatenate([Y[idxs] for idxs in train_idxs])
        else:
            X_train = [X[idxs] for idxs in train_idxs]
            Y_train = [Y[idxs] for idxs in train_idxs]

        self.fold_idx += 1
        return X_train, X_test, Y_train, Y_test, fold_idx
