.. DynamicalComponentsAnalysis

===================================
Dynamical Components Analysis (DCA)
===================================

Model Formulation
-----------------

Breifly, DCA seeks to find a projection of time series data :math:`y_t=V^T\cdot x_t`
such that the predictive information (PI) [Bialek1999]_ is maximized. The PI of a
stationary, multivariate time series :math:`y_t` is defined as the mutual information
between two consecutive windows of length :math:`T`

.. math::

    \begin{align}
        \text{PI}(y, T) &= \text{MI}(y_{-T+1\ldots 0}, y_{1\ldots T})\\
        &=2H(y_{-T+1\ldots 0})-H(y_{1\ldots T})\\
        &=2H_y(T)-H_y(2T)
    \end{align}

Where :math:`H_y(T)` is the entropy of a length-:math:`T` window of :math:`y`. Estimating
the mutual information or entropy of continuous, high dimensional signals is
difficult. Furthermore, we require a estimator of PI that is differentiable in :math:`V`
so that PI can be maximized.

To solve both of these problems, we can assume that :math:`X` is a stationary, discrete-time
Gaussian process. In this case, :math:`Y` will also be stationary and Gaussian since it is a
linear projection of :math:`X`. In this case, estimating PI simplifies to

.. math::

    \begin{align}
        \text{PI}(y, T) &=2H_y(T)-H_y(2T)\\
        &=\log|\Sigma_T(Y)| - \frac{1}{2}\log|\Sigma_{2T}(Y)|
    \end{align}

where :math:`\Sigma_T(Y)` and :math:`\Sigma_{2T}(Y)` are the space-time cross covariance
matrices for windows of length-:math:`T` and :math:`2T` respectively. The space-time
cross covariance matrix for :math:`X` is

.. math::

  \begin{equation}
      \Sigma_{T}(X)
      = \begin{pmatrix}
      C_0 & C_1 & \ldots & C_{T-1} \\
      C_1^T & C_0 & \ldots & C_{T - 2} \\
      \vdots & \vdots & \ddots & \vdots\\
      C_{T-1}^T & C_{T-2}^T & \ldots & C_{0} \\
      \end{pmatrix} \:\: \text{where} \:\:\:
      C_{\Delta t} = \left\langle x_tx_{t + \Delta t}^T \right\rangle_t.
  \end{equation}

Finally, the space-time cross covariance for :math:`Y` can be computed by taking

.. math::

  C_{\Delta t} \rightarrow V^T C_{\Delta t} V.

This allows us to both compute the Gaussian PI and take derivatives with respect to :math:`V`.
More details can be found in [Clark2019]_.

.. rubric:: References

.. [Bialek1999] W. Bialek, and N. Tishby. Predictive information.
    arXiv preprint cond-mat/9902341 (1999).

Python Implementation
---------------------

The DCA models are designed to mimic `scikit-learn` functionality.


.. code:: python

    import numpy as np
    from dca import DynamicalComponentsAnalysis as DCA

    X = np.random.randn(1000, 9)

    model = DCA(d=3, T=10)
    model.fit(X)

    Y = model.transform(X)
