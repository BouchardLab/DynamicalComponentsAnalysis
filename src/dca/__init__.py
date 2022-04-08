from .methods_comparison import (GaussianProcessFactorAnalysis,
                                 SlowFeatureAnalysis,
                                 ForecastableComponentsAnalysis)
from .dca import (DynamicalComponentsAnalysis,
                  DynamicalComponentsAnalysisFFT)

__all__ = ['DynamicalComponentsAnalysis',
           'DynamicalComponentsAnalysisFFT',
           'GaussianProcessFactorAnalysis',
           'SlowFeatureAnalysis',
           'ForecastableComponentsAnalysis']
