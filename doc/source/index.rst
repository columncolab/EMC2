.. EMC² documentation master file, created by
   sphinx-quickstart on Thu Oct 17 09:39:03 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EMC²'s documentation!
=================================

The Earth Model Column Collaboratory (EMC²) is an open-source framework for conducting
intercomparisons between models, with an emphasis on climate and earth system models, and
observations from remote sensing instruments such as radars and lidars.

EMC² is inspired from past work comparing remotely-sensed zenith-pointing measurements
to climate models and their single-column model modes (SCMs) (e.g., Bodas-Salcedo et al.,
2008; Lamer, 2018; Swales et al, 2018).

EMC² provides an open source software framework to:

1. Represent both ARM measurements and GCM columns in the Python programming language
building on the Atmospheric Community Toolkit (ACT, Theisen et. al. 2019) and leveraging 
the EMC² team’s success with Py-ART (Helmus and Collis 2016).

2. Scale GCM outputs (using the cloud fraction) to compare with sub-grid-scale column
measurements using a modular sub column generator designed to run off-line on time series
extracted from existing GCM/SCM output.

3. Enable a suite of comparisons between ARM (and other) column measurements and the
GCM model subcolumns.

Detailed description of EMC² is provided in Silber et al. (GMD, 2022;
https://doi.org/10.5194/gmd-15-901-2022).

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage/index.rst
   EMC2example.ipynb
   contributing
   API/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========

Bodas-Salcedo, A., Webb, M. J., Brooks, M. E., Ringer, M. A., Williams, K. D., Milton, S. F., and Wilson, D. R. (2008), Evaluating cloud systems inthe Met Office global forecast model using simulated CloudSat radar reflectivities, Journal of Geophysical Research: Atmospheres, 113,5https://doi.org/https://doi.org/10.1029/2007JD009620, https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2007JD009620.

Eynard-Bontemps, G., R Abernathey, J. Hamman, A. Ponte, W. Rath, (2019), The Pangeo Big Data Ecosystem and its use at CNES. In P. Soille, S. Loekken, and S. Albani, Proc. of the 2019 conference on Big Data from Space (BiDS’2019), 49-52. EUR 29660 EN, Publications Office of the European Union, Luxembourg. ISBN: 978-92-76-00034-1, doi:10.2760/848593.

Helmus, J., Collis, S., (2016), The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather Radar Data in the Python Programming Language. Journal of Open Research Software 4. https://doi.org/10.5334/jors.119

Jupyter et al. (2018), "Binder 2.0 - Reproducible, Interactive, Sharable Environments for Science at Scale." Proceedings of the 17th Python in Science Conference, 10.25080/Majora-4af1f417-011

Lamer, K. (2018), Relative Occurrence of Liquid Water, Ice and Mixed-Phase Conditions within Various Cloud and Precipitation Regimes: Long Term Ground-Based Observations for GCM Model Evaluation, The Pennsylvania State University, PhD dissertation.

Silber, I. and Jackson, R. C. and Fridlind, A. M. and Ackerman, A. S. and Collis, S. Verlinde, J. and Ding, J (2022), The Earth Model Column Collaboratory (EMC$^2$) v1.1: An Open-Source Ground-Based Lidar and Radar Instrument Simulator and Subcolumn Generator for Large-Scale Models, Geoscientific Model Development, https://doi.org/10.5194/gmd-11-77-2018.

Swales, D.J., Pincus, R., Bodas-Salcedo, A., (2018), The Cloud Feedback Model Intercomparison Project Observational Simulator Package: Version 2. Geosci. Model Dev. 11, 77–81. https://doi.org/10.5194/gmd-11-77-2018
