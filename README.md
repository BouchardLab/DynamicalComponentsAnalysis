# Dynamical Components Analysis

[![Build Status](https://travis-ci.com/BouchardLab/DynamicalComponentsAnalysis.svg?branch=master)](https://travis-ci.com/BouchardLab/DynamicalComponentsAnalysis)
[![Documentation Status](https://readthedocs.org/projects/dynamicalcomponentsanalysis/badge/?version=latest)](https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis/branch/master/graph/badge.svg)](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis)

Implementation of the methods and analyses in [Unsupervised Discovery of Temporal Structure in Noisy Data with Dynamical Components Analysis](http://papers.nips.cc/paper/9574-unsupervised-discovery-of-temporal-structure-in-noisy-data-with-dynamical-components-analysis).

Documentation can be found at https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/index.html

## Installation
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

## Datasets
The 4 datasets used in the DCA paper can be found in the following locations
* [M1](https://zenodo.org/record/583331) - We used indy_20160627_01.mat
* [HC](https://github.com/KordingLab/Neural_Decoding) - See link to the datasets in the README
* [Temperature](https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=temperature.csv) - We used the 30 US cities from temperature.csv.
* [Accelerometer](https://github.com/mmalekzadeh/motion-sense/tree/master/data) - We used sub_19.csv from A_DeviceMotion_data.zip
