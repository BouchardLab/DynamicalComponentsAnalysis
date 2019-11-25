import numpy as np
import pytest

from dca import (DynamicalComponentsAnalysis as DCA,
                 DynamicalComponentsAnalysisFFT as DCAFFT,
                 DynamicalComponentsAnalysisKNN as DCAKNN)


@pytest.fixture
def noise_dataset():
    X = np.random.randn(1000, 10)
    return X


def test_DCA(noise_dataset):
    """Test that a DCA model can be fit with no errors.
    """
    X = noise_dataset
    model = DCA(d=3, T=10)
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
