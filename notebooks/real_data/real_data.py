import numpy as np
import h5py
import pandas as pd
import pickle
import os
import scipy
import datetime

from cca.data_util import sum_over_chunks, CrossValidate
from cca.cov_util import calc_cross_cov_mats_from_data
from cca.data_util import form_lag_matrix
from cca.data_util import load_sabes_data, load_kording_paper_data, load_weather_data
from cca import ComplexityComponentsAnalysis
from cca.analysis import run_analysis, linear_decode_r2

#set data filenames
M1_DATA_FILENAME = "/home/davidclark/Projects/DataUtil/nhp_reaches_sorted.hdf5"
HIPPOCAMPUS_DATA_FILENAME = "/home/davidclark/Projects/ComplexityComponentsAnalysis/data/neural/example_data_hc.pickle"
WEATHER_DATA_FILENMAE = "/home/davidclark/Projects/ComplexityComponentsAnalysis/data/weather/temperature.csv"
CACHE_FILENAME = "data_cache_50ms_bins.pickle"

#set results file directory
RESULTS_DIR = ""

#set analysis params
T_PI_VALS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DIM_VALS = [5, 10, 15, 25]
OFFSET_VALS = [0, 5, 10, 15]
NUM_CV_FOLDS = 5
DECODING_WINDOW = 3
N_INIT = 3

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

def write_h5_info(h5py_group, results,
                  T_pi_vals, dim_vals, offset_vals, num_cv_folds, decoding_window):
    #save params
    h5py_group.attrs["T_pi_vals"] = T_pi_vals
    h5py_group.attrs["dim_vals"] = dim_vals
    h5py_group.attrs["offset_vals"] = offset_vals
    h5py_group.attrs["num_cv_folds"] = num_cv_folds
    h5py_group.attrs["decoding_window"] = decoding_window

    #create results array
    #for the 'T_pi_vals' dimension, indices 0 and 1 correspond to PCA and SFA, respectively
    results = h5py_group.create_dataset("results", data=results)


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

    #Add kinematics
    """
    Y_m1_vel = np.diff(Y_m1, axis=0)
    Y_m1 = np.concatenate((Y_m1[:-1], Y_m1_vel), axis=1)
    X_m1 = X_m1[:-1]

    Y_hc_vel = np.diff(Y_hc, axis=0)
    Y_hc = np.concatenate((Y_hc[:-1], Y_hc_vel), axis=1)
    X_hc = X_hc[:-1]
    """

    #save params
    """
    if DELETE_OLD_FILE:
        try:
            os.remove(RESULTS_FILENAME)
        except OSError:
            pass
    """

    date_time_str = "{date:%Y_%m_%d_%H_%M_%S}".format(date=datetime.datetime.now())
    results_filename = RESULTS_DIR + "decode_results" + date_time_str + ".hdf5"
    print(results_filename)
    with h5py.File(results_filename, "w-") as f:

        print("HC:")
        hc_group = f.create_group("hc")
        hc_group.attrs["timestep"] = 50
        hc_group.attrs["timestep_units"] = "ms"
        dim_vals = [d for d in DIM_VALS if d < 48]
        results = run_analysis(X_hc, Y_hc, T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)
        write_h5_info(hc_group, results,
                      T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW):

        print("Weather:")
        weather_group = f.create_group("weather")
        weather_group.attrs["timestep"] = 1
        weather_group.attrs["timestep_units"] = "days"
        dim_vals = [d for d in DIM_VALS if d < X_weather.shape[1]]
        results = run_analysis(X_weather, Y_weather, T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)
        write_h5_info(weather_group, results,
                      T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW):

        print("M1:")
        m1_group = f.create_group("m1")
        m1_group.attrs["timestep"] = 50
        m1_group.attrs["timestep_units"] = "ms"
        dim_vals = [d for d in DIM_VALS if d < X_m1.shape[1]]
        run_analysis(m1_group, X_m1, Y_m1, T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW)
        write_h5_info(m1_group, results,
                      T_PI_VALS, dim_vals, OFFSET_VALS, NUM_CV_FOLDS, DECODING_WINDOW):
