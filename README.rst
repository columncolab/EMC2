EMC²: the Earth Model Column Collaboratory
==========================================

.. image:: https://img.shields.io/pypi/v/emc2.svg
    :target: https://pypi.python.org/pypi/emc2
    :alt: Latest PyPI version

.. image:: https://travis-ci.org/ARM-DOE/EMC2.png
   :target: https://travis-ci.org/ARM-DOE/EMC2
   :alt: Latest Travis CI build status

The Earth Model Column Collaboratory (EMC²) is an open-source framework for
atmospheric model evaluation against observational data and model
intercomparisons. It consisting of an instrument simulator and a sub-column
generator, which enables statistically emulating a higher spatial resolution.
This framework is specifically designed to simulate the `Atmospheric
Radiation Measurement (ARM) User Facility <http://www.arm.gov>`_ remote-
sensing measurements while being faithful to the representation of physical
processes and sub-grid scale assumptions in various state-of-the-art models,
thereby serving as a practical bridge between observations and models.


Detailed description of EMC² is provided in Silber et al. (GMD, 2022;
https://doi.org/10.5194/gmd-15-901-2022).


Useful links
============

- source code repository: https://github.com/ARM-DOE/EMC2
- EMC² Documentation: https://arm-doe.github.io/EMC2
- EMC² tutorial (from the 2022 ARM Open-Science Workshop): https://github.com/ARM-Development/ARM-Notebooks/blob/main/Tutorials/Open-Science-Workshop-2022/tutorials/EMC2_demo_w_E3SM.ipynb


Citing
======

If the Earth Model Column Collaboratory (EMC²) is used in your manuscript,
please cite:

    Silber, I., Jackson, R. C., Fridlind, A. M., Ackerman, A. S., Collis, S.,
    Verlinde, J., and Ding, J.: The Earth Model Column Collaboratory (EMC2)
    v1.1: an open-source ground-based lidar and radar instrument simulator and
    subcolumn generator for large-scale models, Geosci. Model Dev., 15,
    901–927, https://doi.org/10.5194/gmd-15-901-2022, 2022.

and references therein.


Installation
============

In order to install EMC², you can use either pip or anaconda. In a terminal, simply type either of::

$ pip install emc2
$ conda install -c conda-forge emc2

In addition, if you want to build EMC² from source and install, type in the following commands::

$ git clone https://github.com/ARM-DOE/EMC2
$ cd EMC2
$ pip install .


Dependencies
============

EMC² requires Python 3.10+ as well as: 
   * Atmoshperic Community Toolkit (https://arm-doe.github.io/ACT) 
   * Numpy (https://numpy.org)
   * Scipy (https://scipy.org)
   * Matplotlib (https://matplotlib.org)
   * Xarray (http://xarray.pydata.org)
   * Pandas (https://pandas.pydata.org/)
   * matplotlib (https://matplotlib.org/)
   * netCDF4 (https://github.com/Unidata/netcdf4-python)


Contributions
=============

As its acronym suggests, EMC² is a collaboratory...
Contributions are welcome and encouraged, provided that the code can be
distributed under the BSD 3-clause license (see the LICENSE.txt file).
See the  `contributor's guide. <https://github.com/ARM-DOE/EMC2/blob/master/CONTRIBUTING.rst>`_ for more information.
