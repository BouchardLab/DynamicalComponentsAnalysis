# Dynamical Components Analysis

[![Actions Status](https://github.com/BouchardLab/DynamicalComponentsAnalysis/workflows/DCA%20tests/badge.svg)](https://github.com/BouchardLab/DynamicalComponentsAnalysis/actions)
[![Documentation Status](https://readthedocs.org/projects/dynamicalcomponentsanalysis/badge/?version=latest)](https://dynamicalcomponentsanalysis.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis/branch/main/graph/badge.svg?token=atXKol7aHD)](https://codecov.io/gh/BouchardLab/DynamicalComponentsAnalysis)

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

Note: DCA only requires a CPU-only `pytorch`. If you wish to, **before** installing DCA as in the above instructions,
you can install a CPU-only `pytorch` by following the `pytorch` [installation guide](https://pytorch.org/get-started/locally/).


```bash
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
```


## Datasets
The 4 datasets used in the DCA paper can be found in the following locations
* [M1](https://zenodo.org/record/583331) - We used indy_20160627_01.mat
* [HC](https://github.com/KordingLab/Neural_Decoding) - See link to the datasets in the README
* [Temperature](https://www.kaggle.com/selfishgene/historical-hourly-weather-data?select=temperature.csv) - We used the 30 US cities from temperature.csv.
* [Accelerometer](https://github.com/mmalekzadeh/motion-sense/tree/master/data) - We used std_6/sub_19.csv from A_DeviceMotion_data.zip


## Copyright
Dynamical Components Analysis (DCA) Copyright (c) 2021, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals
from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative 
works, and perform publicly and display publicly, and to permit others to do so.
