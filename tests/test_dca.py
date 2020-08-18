import numpy as np
from numpy.testing import assert_allclose
import pytest

from dca import (DynamicalComponentsAnalysis as DCA,
                 DynamicalComponentsAnalysisFFT as DCAFFT)
from dca.knn import DynamicalComponentsAnalysisKNN as DCAKNN
from dca.synth_data import gen_lorenz_data


@pytest.fixture
def noise_dataset():
    X = np.random.randn(333, 10)
    return X


@pytest.fixture
def lorenz_dataset():
    X = gen_lorenz_data(10000)
    return X


def test_DCA(noise_dataset):
    """Test that a DCA model can be fit with no errors.
    """
    X = noise_dataset
    model = DCA(d=3, T=10)
    model.fit(X)
    assert_allclose(X.mean(axis=0, keepdims=True), model.mean_)
    model.transform(X)
    model.fit_transform(X)
    model.score()

    model = DCA(d=3, T=10, n_init=2)
    model.fit(X)
    model.score()
    model.score(X)

    model = DCA(d=3, T=10, use_scipy=False)
    model.fit(X)

    model = DCA(d=3, T=10, verbose=True)
    model.fit(X)

    model = DCA(d=3, T=10, block_toeplitz=False)
    model.fit(X)


def test_DCA_variable_d(noise_dataset):
    """Test that the DCA projection can be refit with different d.
    """
    X = noise_dataset
    model = DCA(d=3, T=10)
    model.estimate_data_statistics(X)
    model.fit_projection()
    assert model.coef_.shape[1] == 3
    assert model.d_fit == 3
    model.fit_projection(d=2)
    assert model.coef_.shape[1] == 2
    assert model.d_fit == 2


def test_DCA_variable_T(noise_dataset):
    """Test that the DCA projection can be refit with different d.
    """
    X = noise_dataset
    model = DCA(d=3, T=10)
    model.estimate_data_statistics(X)
    model.rng = np.random.RandomState(0)
    model.fit_projection()
    assert model.T_fit == 10
    c0 = model.coef_.copy()
    model.rng = np.random.RandomState(0)
    model.fit_projection()
    c1 = model.coef_.copy()
    model.rng = np.random.RandomState(0)
    model.fit_projection(T=5)
    assert model.T_fit == 5
    c2 = model.coef_.copy()
    assert_allclose(c0, c1)
    assert not np.allclose(c0, c2)
    with pytest.raises(ValueError):
        model.fit_projection(T=11)


def test_DCA_short(noise_dataset):
    """Test that a DCA model raises an error when T would break chunking.
    """
    X = noise_dataset
    model = DCA(d=3, T=20)
    model.fit(X)

    with pytest.raises(ValueError):
        model = DCA(d=3, T=20, chunk_cov_estimate=10)
        model.fit(X)

    model = DCA(d=3, T=20, chunk_cov_estimate=2)
    model.fit(X)


def test_init(noise_dataset):
    X = noise_dataset
    model = DCA(d=3, T=10, init='random')
    model.fit(X)
    model = DCA(d=3, T=10, init='uniform')
    model.fit(X)


def test_input_type():
    """Test that a list of 2d arrays or a 3d array work.
    """
    model = DCA(d=3, T=10)

    X = [np.random.randn(1000, 10) for ii in range(3)]
    model.fit(X)
    assert_allclose(np.concatenate(X).mean(axis=0, keepdims=True), model.mean_)
    model.transform(X)
    model.fit_transform(X)

    X = np.random.randn(3, 1000, 10)
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)


def test_DCAFFT(noise_dataset):
    """Test that a DCAFFT model can be fit with no errors.
    """
    X = noise_dataset
    model = DCAFFT(d=1, T=10)
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)
    model.score(X)


def test_DCAFFT_error(noise_dataset):
    """Test that a DCAFFT raises an error for d>1.
    """
    with pytest.raises(ValueError):
        X = noise_dataset
        model = DCAFFT(d=2, T=10)
        model.fit(X)


@pytest.mark.xfail
def test_DCAKNN(noise_dataset):
    """Test that a DCAKNN model can be fit with no errors.
    """
    X = noise_dataset
    model = DCAKNN(d=1, T=10)
    model.fit(X)
    model.transform(X)
    model.fit_transform(X)


def test_stride_DCA(lorenz_dataset):
    """Check that deterministic and random strides work for DCA.
    """
    X = lorenz_dataset
    model = DCA(T=1)
    model.estimate_data_statistics(X)
    ccms1 = model.cross_covs.numpy()

    model = DCA(T=1, stride=2)
    model.estimate_data_statistics(X)
    ccms2 = model.cross_covs.numpy()
    assert not np.allclose(ccms1, ccms2)
    assert_allclose(ccms1, ccms2, atol=5e-2)

    model = DCA(T=1, stride=.5, rng_or_seed=0)
    model.estimate_data_statistics(X)
    ccms2 = model.cross_covs.numpy()
    assert not np.allclose(ccms1, ccms2)
    assert_allclose(ccms1, ccms2, atol=5e-2)

    model = DCA(T=1, stride=.5, rng_or_seed=1)
    model.estimate_data_statistics(X)
    ccms1 = model.cross_covs.numpy()
    assert not np.allclose(ccms1, ccms2)
    assert_allclose(ccms1, ccms2, atol=5e-2)

    model = DCA(T=1, stride=.5, rng_or_seed=1)
    model.estimate_data_statistics(X)
    ccms2 = model.cross_covs.numpy()
    assert_allclose(ccms1, ccms2)
