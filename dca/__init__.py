from .methods_comparison import (GaussianProcessFactorAnalysis,
                                 SlowFeatureAnalysis,
                                 ForecastableComponentsAnalysis)
from .dca import (DynamicalComponentsAnalysis,
                  DynamicalComponentsAnalysisKNN,
                  DynamicalComponentsAnalysisFFT)

__all__ = ['DynamicalComponentsAnalysis',
           'DynamicalComponentsAnalysisKNN',
           'DynamicalComponentsAnalysisFFT',
           'GaussianProcessFactorAnalysis',
           'SlowFeatureAnalysis',
           'ForecastableComponentsAnalysis']
