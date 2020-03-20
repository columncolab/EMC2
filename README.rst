EMC2: the Earth Model Column Collaboratory
==========================================

.. image:: https://img.shields.io/pypi/v/emc2.svg
    :target: https://pypi.python.org/pypi/emc2
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/columncolab/EMC2.png
   :target: https://travis-ci.org/columncolab/EMC2
   :alt: Latest Travis CI build status

An open source framework for atmospheric model and observational column comparison.
Supported by the Atmospheric Systems Research (ASR) program of the United States Department of Energy.

The Earth Model Column Collaboratory (EMC2) will build on past work comparing remotely sensed measurements
in the column to earth system and global climate models (GCMs) and their single-column models (SCMs)
(Lamer et al. 2018; Swales et al. 2018) by building an open source software framework to:

1. Represent both ARM and GCM columns in the Python programming
   language building on the Atmospheric Community Toolkit (ACT, Theisen et. al. 2019)
   and leveraging the EMC2 team’s success with Py-ART (Helmus and Collis 2016).
2. Scale GCM outputs (using the cloud fraction) to compare with sub-grid-scale column measurements
   using a modular sub column generator (Lamer 2018) which will be designed to run off-line on
   time series extracted from existing GCM/SCM output.
3. Enable a suite of comparisons between ARM (and other) column measurements and
   the GCM model subcolumns.

The work is inspired by the (GO)2SIM (Lamer et al. 2018) in which a sample of NASA ModelE3 global
simulation was analyzed over the North Slope of Alaska (NSA) site at Utqiagvik, Alaska.
For this initial development, a forward model was applied directly to the grid cell mean
profiles and areas of different hydrometeors were converted to observational units using a
variety of techniques. The forward-simulated phase identification was then compared with the
actual model phase, demonstrating that significant differences exist between true model phase
and forward-simulated phase owing to a number of factors, including factors such as lidar attenuation,
radar sensitivity, and trace quantities of liquid that would not be observable. It is therefore
particularly important to project true model phase into forward-modeled phase in order to perform
a robust evaluation against available observations.


Usage
-----

Installation
------------

In order to install EMC^2, you can use either pip or anaconda. In a terminal, simply type either of::

$ pip install emc2
$ conda install -c conda-forge emc2

In addition, if you want to build EMC^2 from source and install, type in the following commands::

$ git clone https://github.com/columncolab/EMC2
$ cd EMC2
$ pip install .

Requirements
^^^^^^^^^^^^

EMC^2 requires Python 3.6+ as well as: 
   * Atmoshperic Community Toolkit (https://arm-doe.github.io/ACT). 
   * Numpy (https://numpy.org)
   * Scipy (https://scipy.org)
   * Matplotlib (https://matplotlib.org)
   * Xarray (http://xarray.pydata.org)
   
Licence
-------

Copyright 2019 Authors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors
-------

`EMC2` was written by `Robert Jackson <rjackson@anl.gov>`_, and led by `Scott Collis <scollis@anl.gov>`_.
Collaborators and Contributors include Ann Fridlind (NASA GISS), Israel Silber (Penn State)
and Marcus van-Lier-Walqui (Columbia U.). 

EMC2 was ported from code written in MATLAB by Israel Silber at Penn State.

References
----------

Swales, D.J., Pincus, R., Bodas-Salcedo, A., 2018. The Cloud Feedback Model Intercomparison Project Observational Simulator Package: Version 2. Geosci. Model Dev. 11, 77–81. https://doi.org/10.5194/gmd-11-77-2018

Lamer, K., Fridlind, A.M., Ackerman, A.S., Kollias, P., Clothiaux, E.E., Kelley, M., 2018. (GO)2-SIM: a GCM-oriented ground-observation forward-simulator framework for objective evaluation of cloud and precipitation phase. Geosci. Model Dev. 11, 4195–4214. https://doi.org/10.5194/gmd-11-4195-2018

Lamer, K. Relative Occurrence of Liquid Water, Ice and Mixed-Phase Conditions within Various Cloud and Precipitation Regimes: Long Term Ground-Based Observations for GCM Model Evaluation. 2018. The Pennsylvania State University, PhD dissertation.

Theisen et. al.: Atmospheric Community Toolkit: https://github.com/ANL-DIGR/ACT

Helmus, J., Collis, S., 2016. The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software 4. https://doi.org/10.5334/jors.119

Eynard-Bontemps, G., R Abernathey, J. Hamman, A. Ponte, W. Rath, 2019: The Pangeo Big Data Ecosystem and its use at CNES. In P. Soille, S. Loekken, and S. Albani, Proc. of the 2019 conference on Big Data from Space (BiDS’2019), 49-52. EUR 29660 EN, Publications Office of the European Union, Luxembourg. ISBN: 978-92-76-00034-1, doi:10.2760/848593.

Fridlind, A.M., van Lier-Walqui, M., Collis, S., Giangrande, S.E., Jackson, R.C., Li, X., Matsui, T., Orville, R., Picel, M.H., Rosenfeld, D., Ryzhkov, A., Weitz, R., Zhang, P., 2019. Use of polarimetric radar measurements to constrain simulated convective cell evolution: a pilot study with Lagrangian tracking. Atmos. Meas. Tech. 12, 2979–3000. https://doi.org/10.5194/amt-12-2979-2019

Wang J, R Wood, M Jensen, E Azevedo, C Bretherton, D Chand, C Chiu, X Dong, J Fast, A Gettelman, S Ghan, S Giangrande, M Gilles, A Jefferson, P Kollias, C Kuang, A Laskin, E Lewis, X Liu, Y Liu, E Luke, A McComiskey, F Mei, M Miller, A Sedlacek, and R Shaw. 2019. Aerosol and Cloud Experiments in Eastern North Atlantic (ACE-ENA) Field Campaign Report. Ed. by Robert Stafford, ARM user facility. DOE/SC-ARM-19-012.

Jupyter et al., "Binder 2.0 - Reproducible, Interactive, Sharable
Environments for Science at Scale." Proceedings of the 17th Python
in Science Conference. 2018. 10.25080/Majora-4af1f417-011
