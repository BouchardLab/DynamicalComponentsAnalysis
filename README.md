# Dynamical Components Analysis

[![Build Status](https://travis-ci.com/BouchardLab/DynamicalComponentsAnalysis.svg?branch=master)](https://travis-ci.com/BouchardLab/DynamicalComponentsAnalysis)
[![Documentation Status](https://readthedocs.org/projects/dynamicalcomponentsanalysis/badge/?version=latest)](https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis)

Implementation of the methods and analyses in [Unsupervised Discovery of Temporal Structure in Noisy Data with Dynamical Components Analysis](https://arxiv.org/abs/1905.09944).

Documentation can be found at https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/index.html

To install, you can clone the repository and `cd` into the DynamicalComponentsAnalysis folder.

```bash
# use ssh
$ git clone git@github.com:BouchardLab/DynamicalComponentsAnalysis.git
# or use https
$ git clone https://github.com/BouchardLab/DynamicalComponentsAnalysis.git
$ cd DynamicalComponentsAnalysis
```

If you are installing into an active conda environment, you can run

```bash
$ conda env update --file environment.yml
$ pip install -e .
```

If you are installing with `pip` you can run

```bash
$ pip install -e . -r requirements.txt
```
