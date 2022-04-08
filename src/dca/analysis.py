import scipy
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.decomposition import PCA
from scipy.stats import special_ortho_group as sog

from .base import init_coef
from .cov_util import calc_pi_from_cross_cov_mats, form_lag_matrix
from .data_util import CrossValidate
from .methods_comparison import SlowFeatureAnalysis as SFA
from .dca import DynamicalComponentsAnalysis as DCA, DynamicalComponentsAnalysisFFT as DCAFFT


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def linear_decode_r2(X_train, Y_train, X_test, Y_test, decoding_window=1, offset=0):
    """Train a linear model on the training set and test on the test set.

    This will work with batched training data and/or batched test data.

    X_train : ndarray (time, channels) or (batches, time, channels)
        Feature training data for regression.
    Y_train : ndarray (time, channels) or (batches, time, channels)
        Target training data for regression.
    X_test : ndarray (time, channels) or (batches, time, channels)
        Feature test data for regression.
    Y_test : ndarray (time, channels) or (batches, time, channels)
        Target test data for regression.
    decoding_window : int
        Number of time samples of X to use for predicting Y (should be odd). Centered around
        offset value.
    offset : int
        Temporal offset for prediction (0 is same-time prediction).
    """

    if isinstance(X_train, np.ndarray) and X_train.ndim == 2:
        X_train = [X_train]
    if isinstance(Y_train, np.ndarray) and Y_train.ndim == 2:
        Y_train = [Y_train]

    if isinstance(X_test, np.ndarray) and X_test.ndim == 2:
        X_test = [X_test]
    if isinstance(Y_test, np.ndarray) and Y_test.ndim == 2:
        Y_test = [Y_test]

    X_train_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_train]
    X_test_lags = [form_lag_matrix(Xi, decoding_window) for Xi in X_test]

    Y_train = [Yi[decoding_window // 2:] for Yi in Y_train]
    Y_train = [Yi[:len(Xi)] for Yi, Xi in zip(Y_train, X_train_lags)]
    if offset >= 0:
        Y_train = [Yi[offset:] for Yi in Y_train]
    else:
        Y_train = [Yi[:Yi.shape[0] + offset] for Yi in Y_train]

    Y_test = [Yi[decoding_window // 2:] for Yi in Y_test]
    Y_test = [Yi[:len(Xi)] for Yi, Xi in zip(Y_test, X_test_lags)]
    if offset >= 0:
        Y_test = [Yi[offset:] for Yi in Y_test]
    else:
        Y_test = [Yi[:Yi.shape[0] + offset] for Yi in Y_test]

    if offset >= 0:
        X_train_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_train_lags]
        X_test_lags = [Xi[:Xi.shape[0] - offset] for Xi in X_test_lags]
    else:
        X_train_lags = [Xi[-offset:] for Xi in X_train_lags]
        X_test_lags = [Xi[-offset:] for Xi in X_test_lags]

    if len(X_train_lags) == 1:
        X_train_lags = X_train_lags[0]
    else:
        X_train_lags = np.concatenate(X_train_lags)

    if len(Y_train) == 1:
        Y_train = Y_train[0]
    else:
        Y_train = np.concatenate(Y_train)

    if len(X_test_lags) == 1:
        X_test_lags = X_test_lags[0]
    else:
        X_test_lags = np.concatenate(X_test_lags)

    if len(Y_test) == 1:
        Y_test = Y_test[0]
    else:
        Y_test = np.concatenate(Y_test)

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
    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        # compute cross-cov mats for DCA
        dca_model = DCA(T=np.max(T_pi_vals))
        dca_model.estimate_data_statistics(X_train_ctd)

        # do PCA/SFA
        pca_model = PCA(svd_solver='full').fit(np.concatenate(X_train_ctd))
        sfa_model = SFA(1).fit(X_train_ctd)

        # make DCA object

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            # compute PCA/SFA R2 vals
            X_train_pca = [np.dot(Xi, pca_model.components_[:dim].T) for Xi in X_train_ctd]
            X_test_pca = np.dot(X_test_ctd, pca_model.components_[:dim].T)
            X_train_sfa = [np.dot(Xi, sfa_model.all_coef_[:, :dim]) for Xi in X_train_ctd]
            X_test_sfa = np.dot(X_test_ctd, sfa_model.all_coef_[:, :dim])
            for offset_idx in range(len(offset_vals)):
                offset = offset_vals[offset_idx]
                r2_pca = linear_decode_r2(X_train_pca, Y_train_ctd, X_test_pca, Y_test_ctd,
                                          decoding_window=decoding_window, offset=offset)
                r2_sfa = linear_decode_r2(X_train_sfa, Y_train_ctd, X_test_sfa, Y_test_ctd,
                                          decoding_window=decoding_window, offset=offset)
                results[fold_idx, dim_idx, offset_idx, 0] = r2_pca
                results[fold_idx, dim_idx, offset_idx, 1] = r2_sfa

            # loop over T_pi vals
            for T_pi_idx in range(len(T_pi_vals)):
                T_pi = T_pi_vals[T_pi_idx]
                dca_model.fit_projection(d=dim, T=T_pi, n_init=n_init)
                V_dca = dca_model.coef_

                # compute DCA R2 over offsets
                X_train_dca = [np.dot(Xi, V_dca) for Xi in X_train_ctd]
                X_test_dca = np.dot(X_test_ctd, V_dca)
                for offset_idx in range(len(offset_vals)):
                    offset = offset_vals[offset_idx]
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd,
                                              decoding_window=decoding_window, offset=offset)
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
    proj_full = np.concatenate([proj, np.zeros((dim, dim - pdim))], axis=1)
    proj_full_comp = np.concatenate([np.zeros((dim, pdim)),
                                     np.linalg.svd(proj_full)[0][:, pdim:]], axis=1)

    # Sample from random vectors
    rots = sog.rvs(dim - pdim, size=size, random_state=random_state)
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
    null_results = np.zeros((num_cv_folds, len(dim_vals) - 2, n_null))
    min_std = 1e-6
    good_cols = (X.std(axis=0) > min_std)
    X = X[:, good_cols]
    # loop over CV folds
    cv = CrossValidate(X, Y, num_cv_folds, stack=False)
    for X_train, X_test, Y_train, Y_test, fold_idx in cv:
        X_test = X_test
        Y_test = Y_test
        if verbose:
            print("fold", fold_idx + 1, "of", num_cv_folds)

        # mean-center X and Y
        X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        X_train_ctd = [Xi - X_mean for Xi in X_train]
        X_test_ctd = X_test - X_mean
        Y_mean = np.concatenate(Y_train).mean(axis=0, keepdims=True)
        Y_train_ctd = [Yi - Y_mean for Yi in Y_train]
        Y_test_ctd = Y_test - Y_mean

        # make DCA object
        # compute cross-cov mats for DCA
        dca_model = DCA(T=T_pi)
        dca_model.estimate_data_statistics(X_train_ctd)

        # loop over dimensionalities
        for dim_idx in range(len(dim_vals)):
            dim = dim_vals[dim_idx]
            if verbose:
                print("dim", dim_idx + 1, "of", len(dim_vals))

            dca_model.fit_projection(d=dim, n_init=n_init)
            V_dca = dca_model.coef_

            # compute DCA R2 over offsets
            X_train_dca = [np.dot(Xi, V_dca) for Xi in X_train_ctd]
            X_test_dca = np.dot(X_test_ctd, V_dca)
            r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd,
                                      decoding_window=decoding_window, offset=offset)
            results[fold_idx, dim_idx] = r2_dca
            if dim_idx < len(dim_vals) - 2:
                comp_vecs = random_complement(V_dca, n_null, rng)
                for ii in range(n_null):
                    vec = comp_vecs[:, [ii]]
                    Vp = np.concatenate([V_dca, vec], axis=1)
                    X_train_dca = [np.dot(Xi, Vp) for Xi in X_train_ctd]
                    X_test_dca = np.dot(X_test_ctd, Vp)
                    r2_dca = linear_decode_r2(X_train_dca, Y_train_ctd, X_test_dca, Y_test_ctd,
                                              decoding_window=decoding_window, offset=offset)
                    null_results[fold_idx, dim_idx, ii] = r2_dca

    return results, null_results


def gen_pi_heatmap(calc_pi_fn, N=100):
    theta_vals = np.linspace(0, np.pi, N)
    phi_vals = np.linspace(0, np.pi, N)
    heatmap = np.zeros((N, N))
    for theta_idx in range(N):
        if theta_idx % 10 == 0:
            print("theta_idx =", theta_idx)
        for phi_idx in range(N):
            theta, phi = theta_vals[theta_idx], phi_vals[phi_idx]
            x = np.cos(phi) * np.sin(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(theta)
            V = np.array([x, y, z]).reshape((3, 1))
            heatmap[theta_idx, phi_idx] = calc_pi_fn(V)
    return heatmap


def make_pi_fn_gp(cross_cov_mats):
    def calc_pi_fn_gp(V):
        pi = calc_pi_from_cross_cov_mats(cross_cov_mats, proj=V)
        return pi
    return calc_pi_fn_gp


def make_pi_fn_knn(X, T_pi, n_jobs=-1):
    from info_measures.continuous import kraskov_stoegbauer_grassberger as ksg

    def calc_pi_fn_knn(V):
        X_proj = np.dot(X, V)
        X_proj_lags = form_lag_matrix(X_proj, 2 * T_pi)
        mi = ksg.MutualInformation(X_proj_lags[:, :T_pi], X_proj_lags[:, T_pi:], add_noise=True)
        pi = mi.mutual_information(n_jobs=n_jobs)
        return pi
    return calc_pi_fn_knn


def random_proj_pi_comparison(calc_pi_fn_1, cal_pi_fn_2, N, d=1,
                              n_samples=10000, seed=20210412):
    rng = np.random.RandomState(seed)
    pi_1, pi_2 = np.zeros(n_samples), np.zeros(n_samples)
    for i in range(n_samples):
        if i % 100 == 0:
            print("sample {} of {}".format(i, n_samples))
        V = init_coef(N, d, rng=rng, init='random_ortho')
        pi_1[i] = calc_pi_fn_1(V)
        pi_2[i] = cal_pi_fn_2(V)
    pi_12 = np.vstack((pi_1, pi_2)).T  # (n_samples, 2)
    return pi_12


def gp_knn_trajectories(num_traj, cross_cov_mats, X, T_pi, d):
    f_gp = make_pi_fn_gp(cross_cov_mats)
    f_knn = make_pi_fn_knn(X, T_pi=T_pi)
    trajectories = []
    for traj_idx in range(num_traj):
        print("traj_idx =", traj_idx)
        opt = DCA(d=d, T=T_pi)
        opt.cross_covs = cross_cov_mats
        opt.fit_projection(record_V=True)
        V_seq = opt.V_seq
        num_dca_iter = len(V_seq)
        pi_gp_knn_traj = np.zeros((num_dca_iter, 2))
        for i in range(num_dca_iter):
            if i % 50 == 0:
                print("{} of {}".format(i, num_dca_iter))
            V = V_seq[i]
            pi_gp_knn_traj[i, 0] = f_gp(V)
            pi_gp_knn_traj[i, 1] = f_knn(V)
        trajectories.append(pi_gp_knn_traj)
    return trajectories


def dca_deflation(cross_cov_mats, n_proj, n_init=1):
    N = cross_cov_mats.shape[1]
    T = cross_cov_mats.shape[0] // 2
    F = np.eye(N)
    cov_proj = np.copy(cross_cov_mats)
    basis = np.zeros((N, n_proj))
    opt = DCA(T=T)
    for i in range(n_proj):
        if i % 10 == 0:
            print(i)
        # run DCA
        opt.cross_covs = cov_proj
        opt.fit_projection(d=1, n_init=n_init)
        v = opt.coef_.flatten()
        # get full-dim v
        v_full = np.dot(F, v)
        basis[:, i] = v_full
        # update U, F, cov_proj
        U = scipy.linalg.orth(np.eye(N - i) - np.outer(v, v))
        F = np.dot(F, U)
        cov_proj = np.array([U.T.dot(C).dot(U) for C in cov_proj])
    return basis


def dca_fft_deflation(X, T, n_proj, n_init=1):
    N = X.shape[1]
    F = np.eye(N)
    X_proj = np.copy(X)
    basis = np.zeros((N, n_proj))
    opt = DCAFFT(T=T, d=1)
    for i in range(n_proj):
        if i % 10 == 0:
            print(i)
        # run DCA
        opt.fit(X_proj, n_init=n_init)
        v = opt.coef_.flatten()
        # get full-dim v
        v_full = np.dot(F, v)
        basis[:, i] = v_full
        # update U, F, X
        U = scipy.linalg.orth(np.eye(N - i) - np.outer(v, v))
        F = np.dot(F, U)
        X_proj = np.dot(X_proj, U)
    return basis


def dca_full(cross_cov_mats, n_proj, n_init=1):
    T = cross_cov_mats.shape[0] // 2
    opt = DCA(T=T)
    opt.cross_covs = cross_cov_mats
    V_seq = []
    for i in range(n_proj):
        if i % 10 == 0:
            print(i)
        opt.fit_projection(d=i + 1, n_init=n_init)
        V = opt.coef_
        V_seq.append(V)
    return V_seq


def calc_pi_vs_dim(cross_cov_mats, V=None, V_seq=None):
    if V_seq is None:
        V_seq = [V[:, :i + 1] for i in range(V.shape[1])]
    pi_vals = np.zeros(len(V_seq))
    for i in range(len(V_seq)):
        V = V_seq[i]
        pi_vals[i] = calc_pi_from_cross_cov_mats(cross_cov_mats, proj=V)
    return pi_vals
