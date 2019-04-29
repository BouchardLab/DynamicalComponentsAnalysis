import numpy as np
import h5py
import pandas as pd
import pickle
import os
import scipy

from cca.data_util import sum_over_chunks, CrossValidate
from cca.cov_util import calc_cross_cov_mats_from_data
from cca.data_util import form_lag_matrix
from cca import ComplexityComponentsAnalysis

#set data filenames
M1_DATA_FILENAME = "/home/davidclark/Projects/DataUtil/nhp_reaches_sorted.hdf5"
HIPPOCAMPUS_DATA_FILENAME = "/home/davidclark/Projects/ComplexityComponentsAnalysis/neuro_data/example_data_hc.pickle"
WEATHER_DATA_FILENMAE = "/home/davidclark/Projects/ComplexityComponentsAnalysis/notebooks/weather/temperature.csv"
CACHE_FILENAME = "data_cache.pickle"

#set results filenames
RESULTS_FILENAME = "good_results.hdf5"
DELETE_OLD_FILE = True

#set analysis params
T_PI_VALS = [1, 2, 3, 4, 5, 6, 7]
DIM_VALS = [1, 5, 10, 20, 30, 40]
OFFSET_VALS = [0, 5, 10]
NUM_CV_FOLDS = 5
DECODING_WINDOW = 3

def pca_proj(X):
    X = X - X.mean(axis=0)
    cov = np.cov(X.T)
    _, V = np.linalg.eigh(cov)
    V = V[::-1]
    return V

def sfa_proj(X):
    X = X - X.mean(axis=0)
    cov = np.cov(X.T)
    whitening_mat = np.linalg.inv(np.real(scipy.linalg.sqrtm(cov)))
    X_white = np.dot(X, whitening_mat)
    X_dot = np.diff(X_white, axis=0)
    cov_dot = np.cov(X_dot.T)
    w_dot, V_dot = np.linalg.eigh(cov_dot) #note: we want the eigenvalues in ASCENDING order
    V_sfa = np.dot(whitening_mat, V_dot)
    V_sfa /= np.sqrt(np.sum(V_sfa**2, axis=0))
    return V_sfa

def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    X_train = X_train - X_train.mean(axis=0)
    Y_train = Y_train - Y_train.mean(axis=0)
    X_test = X_test - X_test.mean(axis=0)
    Y_test = Y_test - Y_test.mean(axis=0)
    
    X_train_lags = form_lag_matrix(X_train, decoding_window)
    X_test_lags = form_lag_matrix(X_test, decoding_window)
    
    Y_train = Y_train[decoding_window // 2 :]
    Y_train = Y_train[: len(X_train_lags)]
    Y_test = Y_test[decoding_window // 2 :]
    Y_test = Y_test[: len(X_test_lags)]
    
    T_train = len(X_train_lags)
    T_test = len(X_test_lags)
    X_train_lags, X_test_lags = X_train_lags[:T_train-offset], X_test_lags[:T_test-offset]
    Y_train, Y_test = Y_train[offset:], Y_test[offset:]
    
    beta = np.linalg.lstsq(X_train_lags, Y_train, rcond=None)[0]
    Y_test_pred = np.dot(X_test_lags, beta)
    r2 = 1 - np.sum((Y_test_pred - Y_test)**2)/np.sum(Y_test**2)
    
    return r2

def load_sabes_data(filename, bin_width_s=0.1, session=None, min_spike_count=1000):
    f = h5py.File(filename, "r")
    sessions = list(f.keys())
    lengths = np.array([f[session]["M1"]["spikes"].shape[0] for session in sessions])
    if session is None:
        #use longest session if none is provided
        session = sessions[np.argsort(lengths)[::-1][0]]
    X, Y = f[session]["M1"]["spikes"], f[session]["cursor"]
    chunk_size = int(np.round(bin_width_s / .004)) #4 ms default bin width
    X, Y = sum_over_chunks(X, chunk_size), sum_over_chunks(Y, chunk_size)/chunk_size
    X = X[:, np.sum(X, axis=0) > min_spike_count]
    return X, Y

def load_kording_paper_data(filename, bin_width_s=0.1, min_spike_count=10):
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
    return X, Y

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
    X = scipy.signal.resample(Xfs, Xfs.shape[0] // ds_factor, axis=0)
    return X

def run_analysis(h5py_group, X, Y, T_pi_vals, dim_vals, offset_vals, num_cv_folds, decoding_window):
    #save params
    h5py_group.attrs["T_pi_vals"] = T_pi_vals
    h5py_group.attrs["dim_vals"] = dim_vals
    h5py_group.attrs["offset_vals"] = offset_vals
    h5py_group.attrs["num_cv_folds"] = num_cv_folds
    h5py_group.attrs["decoding_window"] = decoding_window

    #create results array
    #for the 'T_pi_vals' dimension, indices 0 and 1 correspond to PCA and SFA, respectively
    results_size = (num_cv_folds, len(dim_vals), len(offset_vals), len(T_pi_vals) + 2)
    results = h5py_group.create_dataset("results", results_size)

    #loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds)
    N = X.shape[1]
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        print("fold", fold_idx + 1, "of", num_cv_folds)
        
        #mean-center X and Y
        X_train_ctd = X_train - X_train.mean(axis=0)
        X_test_ctd = X_test - X_test.mean(axis=0)
        Y_train_ctd = Y_train - Y_train.mean(axis=0)
        Y_test_ctd = Y_test - Y_test.mean(axis=0)

        #remove any zero columns from data matrices
        min_std = 1e-6
        good_cols = (X_train_ctd.std(axis=0) > min_std) * (X_test_ctd.std(axis=0) > min_std)
        X_train_ctd = X_train_ctd[:, good_cols]
        X_test_ctd = X_test_ctd[:, good_cols]

        #compute cross-cov mats for DCA
        T_max = 2*np.max(T_pi_vals)
        cross_cov_mats = calc_cross_cov_mats_from_data(X_train_ctd, T_max)
        
        #do PCA/SFA
        V_pca = pca_proj(X_train_ctd)
        V_sfa = sfa_proj(X_train_ctd)
        
        #make DCA object
        opt = ComplexityComponentsAnalysis(verbose=False)
        
        #loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            #print("dim", dim_idx + 1, "of", len(dim_vals))
            
            #compute PCA/SFA R2 vals
            X_train_pca = np.dot(X_train_ctd, V_pca[:, :dim])
            X_test_pca = np.dot(X_test_ctd, V_pca[:, :dim])
            X_train_sfa = np.dot(X_train_ctd, V_sfa[:, :dim])
            X_test_sfa = np.dot(X_test_ctd, V_sfa[:, :dim])
            for offset_idx in range(len(offset_vals)):
                offset = offset_vals[offset_idx]
                r2_pca = linear_decode_r2(X_train_pca, Y_train_ctd, X_test_pca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                r2_sfa = linear_decode_r2(X_train_sfa, Y_train_ctd, X_test_sfa, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                results[fold_idx, dim_idx, offset_idx, 0] = r2_pca
                results[fold_idx, dim_idx, offset_idx, 1] = r2_sfa
            
            #loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):
                T_pi = T_pi_vals[T_pi_idx]
                opt.cross_covs = cross_cov_mats[:2*T_pi]
                opt.fit_projection(d=dim)
                V_dca = opt.coef_
                
                #compute DCA R2 over offsets
                X_train_dca = np.dot(X_train_ctd, V_dca)
                X_test_dca = np.dot(X_test_ctd, V_dca)
                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    results[fold_idx, dim_idx, offset_idx, T_pi_idx + 2] = r2_dca

                
def cache_data():
    X_weather = load_weather_data(WEATHER_DATA_FILENMAE)
    Y_weather = np.copy(X_weather)
    X_m1, Y_m1 = load_sabes_data(M1_DATA_FILENAME, bin_width_s=0.05, min_spike_count=1000)
    X_hc, Y_hc = load_kording_paper_data(HIPPOCAMPUS_DATA_FILENAME, bin_width_s=0.05, min_spike_count=10)
    data = {"weather": (X_weather, Y_weather),
            "m1": (X_m1, Y_m1),
            "hc": (X_hc, Y_hc),}
    with open(CACHE_FILENAME, "wb") as f:
        pickle.dump(data, f)

def load_cached_data():
    with open(CACHE_FILENAME, "rb") as f:
        data = pickle.load(f)
        X_weather, Y_weather = data["weather"]
        X_m1, Y_m1 = data["m1"]
        X_hc, Y_hc = data["hc"]
    return X_weather, Y_weather, X_m1, Y_m1, X_hc, Y_hc

if __name__ == "__main__":
    cache_exists = os.path.isfile(CACHE_FILENAME)
    if not cache_exists:
        cache_data()

    X_weather, Y_weather, X_m1, Y_m1, X_hc, Y_hc = load_cached_data()
    
    #save params
    if DELETE_OLD_FILE:
        try:
            os.remove(RESULTS_FILENAME)
        except OSError:
            pass
    f = h5py.File(RESULTS_FILENAME, "w-")

    print("HC:")
    hc_group = f.create_group("hc")
    run_analysis(hc_group, X_hc, Y_hc, T_PI_VALS, DIM_VALS[:3], OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)

    print("Weather:")
    weather_group = f.create_group("weather")
    run_analysis(weather_group, X_weather, Y_weather, T_PI_VALS, [d for d in DIM_VALS if d < 30], OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)

    print("M1:")
    m1_group = f.create_group("m1")
    run_analysis(m1_group, X_m1, Y_m1, T_PI_VALS, DIM_VALS, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)


