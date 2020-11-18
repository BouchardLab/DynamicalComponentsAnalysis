.. DynamicalComponentsAnalysis

============
Installation
============

Dynamical Components Analysis (DCA) is available for Python 3. The latest development version
of the code can be installed from https://github.com/BouchardLab/DynamicalComponentsAnalysis

.. code-block:: bash

    # use ssh
    $ git clone git@github.com:BouchardLab/DynamicalComponentsAnalysis.git
    # or use https
    $ git clone https://github.com/BouchardLab/DynamicalComponentsAnalysis.git
    $ cd DynamicalComponentsAnalysis

To install into an active conda environment

.. code-block:: bash

    $ conda env update --file environment.yml
    $ pip install -e .

and with pip

.. code-block:: bash

    $ pip install -e . -r requirements.txt

Requirements
------------

Runtime
^^^^^^^

DCA requires

  * numpy
  * scipy
  * h5py
  * pandas
  * scikit-learn
  * pytorch

to run.

Develop
^^^^^^^

To develop DCA you will additionally need

  * pytest
  * flake8

to run the tests and check formatting.

Docs
^^^^

To build the docs you will additionally need

  * sphinx
  * sphinx_rtd_theme
