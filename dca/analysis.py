import numpy as np
import h5py
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group as sog

from .cov_util import calc_cross_cov_mats_from_data
from .data_util import CrossValidate, form_lag_matrix
from .methods_comparison import SlowFeatureAnalysis as SFA
from .dca import DynamicalComponentsAnalysis


def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    """Train a linear model on the training set and test on the test set.

    This will work with batched training data, but the testing data has to be
    one sequence.
    """
    if isinstance(X_train, np.ndarray) and X_train.ndim == 2:
        X_train = [X_train]
    if isinstance(Y_train, np.ndarray) and Y_train.ndim == 2:
        Y_train = [Y_train]

    X_train_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_train]
    X_test_lags = form_lag_matrix(X_test, decoding_window)

    Y_train = [Yi[decoding_window // 2:] for Yi in Y_train]
    Y_train = [Yi[:len(Xi)] for Yi, Xi in zip(Y_train, X_train_lags)]
    if offset >= 0:
        Y_train = [Yi[offset:] for Yi in Y_train]
    else:
        Y_train = [Yi[:Yi.shape[0]+offset] for Yi in Y_train]

    Y_test = Y_test[decoding_window // 2:]
    Y_test = Y_test[:len(X_test_lags)]
    if offset >= 0:
        Y_test = Y_test[offset:]
    else:
        Y_test = Y_test[:Y_test.shape[0]+offset]

    if offset >= 0:
        X_train_lags = [Xi[:Xi.shape[0]-offset] for Xi in X_train_lags]
        X_test_lags = X_test_lags[:X_test_lags.shape[0]-offset]
    else:
        X_train_lags = [Xi[-offset:] for Xi in X_train_lags]
        X_test_lags = X_test_lags[-offset:]

    if len(X_train) == 1:
        X_train_lags = X_train_lags[0]
    else:
        X_train_lags = np.concatenate(X_train_lags)
    if len(Y_train) == 1:
        Y_train = Y_train[0]
    else:
        Y_train = np.concatenate(Y_train)

    model = LR().fit(X_train_lags, Y_train)
    r2 = model.score(X_test_lags, Y_test)
    return r2


def run_analysis(X, Y, T_pi_vals, dim_vals, offset_vals, num_cv_folds, decoding_window,
                 n_init=1, verbose=False):

    results_size = (num_cv_folds, len(dim_vals), len(offset_vals), len(T_pi_vals) + 2)
    results = np.zeros(results_size)
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    #loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        X_test = X_test
        Y_test = Y_test
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        #mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        #compute cross-cov mats for DCA
        T_max = 2*np.max(T_pi_vals)
        cross_cov_mats = calc_cross_cov_mats_from_data(X_train_ctd, T_max)

        #do PCA/SFA
        pca_model = PCA(svd_solver='full').fit(np.concatenate(X_train_ctd))
        sfa_model = SFA(1).fit(X_train_ctd)

        #make DCA object
        opt = DynamicalComponentsAnalysis()

        #loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            #compute PCA/SFA R2 vals
            X_train_pca = [np.dot(Xi, pca_model.components_[:dim].T) for Xi in X_train_ctd]
            X_test_pca = np.dot(X_test_ctd, pca_model.components_[:dim].T)
            X_train_sfa = [np.dot(Xi, sfa_model.all_coef_[:, :dim]) for Xi in X_train_ctd]
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
                X_train_dca = [np.dot(Xi, V_dca) for Xi in X_train_ctd]
                X_test_dca = np.dot(X_test_ctd, V_dca)
                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    results[fold_idx, dim_idx, offset_idx, T_pi_idx + 2] = r2_dca
    return results


def random_complement(proj, size=1, random_state=None):
    """Computes a random vector in the orthogonal complement to proj.

    Parameters
    ----------
    proj : ndarray (full dim, low dim)
        Projection matrix.
    random_state : NumPy random state (optional)
        Random state for rng.

    Returns
    -------
    comp_vec : ndarray (full dim, 1)
        Random vector in the complement space.
    """
    dim, pdim = proj.shape
    if pdim >= dim:
        raise ValueError

    # Create complement space
    proj_full = np.concatenate([proj, np.zeros((dim, dim-pdim))], axis=1)
    proj_full_comp = np.concatenate([np.zeros((dim, pdim)),
                                     np.linalg.svd(proj_full)[0][:, pdim:]], axis=1)

    # Sample from random vectors
    rots = sog.rvs(dim-pdim, size=size, random_state=random_state)
    if size == 1:
        rots = rots[np.newaxis]
    comp_vec = np.zeros((dim, size))
    for ii in range(size):
        comp_vec[:, ii] = proj_full_comp[:, pdim:].dot(rots[ii])[:, -1]
    return comp_vec


def run_dim_analysis_dca(X, Y, T_pi, dim_vals, offset, num_cv_folds, decoding_window,
                 n_init=1, verbose=False, n_null=1000, seed=20190710):

    rng = np.random.RandomState(seed)
    results = np.zeros((num_cv_folds, len(dim_vals)))
    null_results = np.zeros((num_cv_folds, len(dim_vals)-2, n_null))
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    #loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        X_test = X_test
        Y_test = Y_test
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        #mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        #compute cross-cov mats for DCA
        cross_cov_mats = calc_cross_cov_mats_from_data(X_train_ctd, 2 * T_pi)

        #make DCA object
        opt = DynamicalComponentsAnalysis()

        #loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            opt.cross_covs = cross_cov_mats
            opt.fit_projection(d=dim, n_init=n_init)
            V_dca = opt.coef_

            #compute DCA R2 over offsets
            X_train_dca = [np.dot(Xi, V_dca) for Xi in X_train_ctd]
            X_test_dca = np.dot(X_test_ctd, V_dca)
            r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
            results[fold_idx, dim_idx] = r2_dca
            if dim_idx < len(dim_vals) - 2:
                comp_vecs = random_complement(V_dca, n_null, rng)
                for ii in range(n_null):
                    vec = comp_vecs[:, [ii]]
                    Vp = np.concatenate([V_dca, vec], axis=1)
                    X_train_dca = [np.dot(Xi, Vp) for Xi in X_train_ctd]
                    X_test_dca = np.dot(X_test_ctd, Vp)
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd, decoding_window=decoding_window, offset=offset)
                    null_results[fold_idx, dim_idx, ii] = r2_dca

    return results, null_results
