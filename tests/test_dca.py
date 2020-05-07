import numpy as np
from numpy.testing import assert_allclose
import pytest

from dca import (DynamicalComponentsAnalysis as DCA,
                 DynamicalComponentsAnalysisFFT as DCAFFT,
                 DynamicalComponentsAnalysisKNN as DCAKNN)


@pytest.fixture
def noise_dataset():
    X = np.random.randn(333, 10)
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
