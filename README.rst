EMC²: the Earth Model Column Collaboratory
==========================================

.. image:: https://img.shields.io/pypi/v/emc2.svg
    :target: https://pypi.python.org/pypi/emc2
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/columncolab/EMC2.png
   :target: https://travis-ci.org/columncolab/EMC2
   :alt: Latest Travis CI build status

An open source framework for atmospheric model and observational column comparison.
Supported by the Atmospheric Systems Research (ASR) program of the United States Department of Energy.

The Earth Model Column Collaboratory (EMC²) is inspired from past work comparing remotely sensed zenith-pointing
measurements to climate models and their single-column model modes (SCMs)
(e.g., Bodas-Salcedo et al., 2008; Lamer et al. 2018; Swales et al. 2018).

EMC² provides an open source software framework to:

1. Represent both ARM measurements and GCM columns in the Python programming
   language building on the Atmospheric Community Toolkit (ACT, Theisen et. al. 2019)
   and leveraging the EMC² team’s success with Py-ART (Helmus and Collis 2016).
2. Scale GCM outputs (using the cloud fraction) to compare with sub-grid-scale column measurements
   using a modular sub column generator designed to run off-line on time series extracted from
   existing GCM/SCM output.
3. Enable a suite of comparisons between ARM (and other) column measurements and
   the GCM model subcolumns.

Detailed description of EMC² is provided in Silber et al. (GMD, 2022;
https://doi.org/10.5194/gmd-15-901-2022).


Usage
-----

For details on how to use EMC², please see the Documentation (https://columncolab.github.io/EMC2).

Installation
------------

In order to install EMC², you can use either pip or anaconda. In a terminal, simply type either of::

$ pip install emc2
$ conda install -c conda-forge emc2

In addition, if you want to build EMC² from source and install, type in the following commands::

$ git clone https://github.com/columncolab/EMC2
$ cd EMC2
$ pip install .

Requirements
^^^^^^^^^^^^

EMC² requires Python 3.6+ as well as: 
   * Atmoshperic Community Toolkit (https://arm-doe.github.io/ACT). 
   * Numpy (https://numpy.org)
   * Scipy (https://scipy.org)
   * Matplotlib (https://matplotlib.org)
   * Xarray (http://xarray.pydata.org)
   * Pandas (https://pandas.pydata.org/)
   
Licence
-------

Copyright 2021 Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors
-------

`EMC²` was written by `Robert Jackson <rjackson@anl.gov>`_ and `Israel Silber <ixs34@psu.edu>`_.
Collaborators and Contributors include `Scott Collis <scollis@anl.gov>`_, and Ann Fridlind (NASA GISS). 

References
----------

Bodas-Salcedo, A., Webb, M. J., Brooks, M. E., Ringer, M. A., Williams, K. D., Milton, S. F., and Wilson, D. R. (2008), Evaluating cloud systems inthe Met Office global forecast model using simulated CloudSat radar reflectivities, Journal of Geophysical Research: Atmospheres, 113,5https://doi.org/https://doi.org/10.1029/2007JD009620, https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2007JD009620.

Eynard-Bontemps, G., R Abernathey, J. Hamman, A. Ponte, W. Rath, (2019), The Pangeo Big Data Ecosystem and its use at CNES. In P. Soille, S. Loekken, and S. Albani, Proc. of the 2019 conference on Big Data from Space (BiDS’2019), 49-52. EUR 29660 EN, Publications Office of the European Union, Luxembourg. ISBN: 978-92-76-00034-1, doi:10.2760/848593.

Helmus, J., Collis, S. (2016), The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software 4. https://doi.org/10.5334/jors.119

Jupyter et al. (2018), "Binder 2.0 - Reproducible, Interactive, Sharable Environments for Science at Scale," Proceedings of the 17th Python in Science Conference, 10.25080/Majora-4af1f417-011

Lamer, K. (2018), Relative Occurrence of Liquid Water, Ice and Mixed-Phase Conditions within Various Cloud and Precipitation Regimes: Long Term Ground-Based Observations for GCM Model Evaluation, The Pennsylvania State University, PhD dissertation.

Swales, D.J., Pincus, R., Bodas-Salcedo, A. (2018), The Cloud Feedback Model Intercomparison Project Observational Simulator Package: Version 2. Geosci. Model Dev. 11, 77–81. https://doi.org/10.5194/gmd-11-77-2018

Theisen et. al. (2019), Atmospheric Community Toolkit: https://github.com/ANL-DIGR/ACT.
