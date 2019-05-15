import numpy as np
import h5py
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA

from .cov_util import calc_cross_cov_mats_from_data
from .data_util import CrossValidate, form_lag_matrix
from .methods_comparison import SlowFeatureAnalysis as SFA
from .cca import ComplexityComponentsAnalysis

def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    X_train_lags = form_lag_matrix(X_train, decoding_window)
    X_test_lags = form_lag_matrix(X_test, decoding_window)

    Y_train = Y_train[decoding_window // 2:]
    Y_train = Y_train[:len(X_train_lags)]
    Y_test = Y_test[decoding_window // 2:]
    Y_test = Y_test[:len(X_test_lags)]

    T_train = len(X_train_lags)
    T_test = len(X_test_lags)
    X_train_lags, X_test_lags = X_train_lags[:T_train-offset], X_test_lags[:T_test-offset]
    Y_train, Y_test = Y_train[offset:], Y_test[offset:]

    model = LR().fit(X_train_lags, Y_train)
    r2 = model.score(X_test_lags, Y_test)
    return r2


def run_analysis(X, Y, T_pi_vals, dim_vals, offset_vals, num_cv_folds, decoding_window,
                 n_init=1):

    results_size = (num_cv_folds, len(dim_vals), len(offset_vals), len(T_pi_vals) + 2)
    results = np.zeros(results_size)
    #loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds)
    N = X.shape[1]
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        print("fold", fold_idx + 1, "of", num_cv_folds)

        #mean-center X and Y
        X_train_ctd = X_train - X_train.mean(axis=0)
        X_test_ctd = X_test - X_train.mean(axis=0)
        Y_train_ctd = Y_train - Y_train.mean(axis=0)
        Y_test_ctd = Y_test - Y_train.mean(axis=0)

        #remove any zero columns from data matrices
        min_std = 1e-6
        good_cols = (X_train_ctd.std(axis=0) > min_std) * (X_test_ctd.std(axis=0) > min_std)
        X_train_ctd = X_train_ctd[:, good_cols]
        X_test_ctd = X_test_ctd[:, good_cols]

        #compute cross-cov mats for DCA
        T_max = 2*np.max(T_pi_vals)
        cross_cov_mats = calc_cross_cov_mats_from_data(X_train_ctd, T_max)

        #do PCA/SFA
        pca_model = PCA(svd_solver='full').fit(X_train_ctd)
        sfa_model = SFA(1).fit(X_train_ctd)

        #make DCA object
        opt = ComplexityComponentsAnalysis()

        #loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            print("dim", dim_idx + 1, "of", len(dim_vals))

            #compute PCA/SFA R2 vals
            X_train_pca = np.dot(X_train_ctd, pca_model.components_[:dim].T)
            X_test_pca = np.dot(X_test_ctd, pca_model.components_[:dim].T)
            X_train_sfa = np.dot(X_train_ctd, sfa_model.all_coef_[:, :dim])
            X_test_sfa = np.dot(X_test_ctd, sfa_model.all_coef_[:, :dim])
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
                opt.fit_projection(d=dim, n_init=n_init)
                V_dca = opt.coef_

                #compute DCA R2 over offsets
                X_train_dca = np.dot(X_train_ctd, V_dca)
                X_test_dca = np.dot(X_test_ctd, V_dca)
                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    results[fold_idx, dim_idx, offset_idx, T_pi_idx + 2] = r2_dca
    return results
