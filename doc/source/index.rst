.. EMC^2 documentation master file, created by
   sphinx-quickstart on Thu Oct 17 09:39:03 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to EMC^2's documentation!
=================================

The Earth Model Column Collaboratory (EMC^2) is an open source framework for conducting
intercomparisons between columns from earth system models and observations from
remote sensing instruments such as radars and lidars.


The Earth Model Column Collaboratory (EMC2) will build on past work comparing remotely sensed
measurements in the column to earth system and global climate models (GCMs) and their single-column models
(SCMs) (Lamer et al. 2018; Swales et al. 2018) by building an open source software framework to:

    Represent both ARM and GCM columns in the Python programming language building on the
    Atmospheric Community Toolkit (ACT, Theisen et. al. 2019) and leveraging the EMC2 team’s
    success with Py-ART (Helmus and Collis 2016).

    Scale GCM outputs (using the cloud fraction) to compare with sub-grid-scale column measurements
    using a modular sub column generator (Lamer 2018) which will be designed to run off-line on time
    series extracted from existing GCM/SCM output.

    Enable a suite of comparisons between ARM (and other) column measurements and the GCM model subcolumns.

The work is inspired by the (GO)2SIM (Lamer et al. 2018) in which a sample of NASA ModelE3
global simulation was analyzed over the North Slope of Alaska (NSA) site at Utqiagvik, Alaska.
For this initial development, a forward model was applied directly to the grid cell mean profiles
and areas of different hydrometeors were converted to observational units using a variety of techniques.
The forward-simulated phase identification was then compared with the actual model phase,
demonstrating that significant differences exist between true model phase and forward-simulated phase
owing to a number of factors, including factors such as lidar attenuation, radar sensitivity, and trace quantities
of liquid that would not be observable. It is therefore particularly important to project true model phase into
forward-modeled phase in order to perform a robust evaluation against available observations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage/index.rst
   contributing
   API/index.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


References
==========
Swales, D.J., Pincus, R., Bodas-Salcedo, A., 2018. The Cloud Feedback Model Intercomparison Project
Observational Simulator Package: Version 2. Geosci. Model Dev. 11, 77–81. https://doi.org/10.5194/gmd-11-77-2018

Lamer, K., Fridlind, A.M., Ackerman, A.S., Kollias, P., Clothiaux, E.E., Kelley, M., 2018.
(GO)2-SIM: a GCM-oriented ground-observation forward-simulator framework for objective evaluation
of cloud and precipitation phase. Geosci. Model Dev. 11, 4195–4214. https://doi.org/10.5194/gmd-11-4195-2018

Lamer, K. Relative Occurrence of Liquid Water, Ice and Mixed-Phase Conditions within Various Cloud
and Precipitation Regimes: Long Term Ground-Based Observations for GCM Model Evaluation. 2018.
The Pennsylvania State University, PhD dissertation.

Theisen et. al.: Atmospheric Community Toolkit: https://github.com/ANL-DIGR/ACT

Helmus, J., Collis, S., 2016. The Python ARM Radar Toolkit (Py-ART), a Library for Working with Weather
Radar Data in the Python Programming Language. Journal of Open Research Software 4. https://doi.org/10.5334/jors.119

Eynard-Bontemps, G., R Abernathey, J. Hamman, A. Ponte, W. Rath, 2019: The Pangeo Big Data Ecosystem
and its use at CNES. In P. Soille, S. Loekken, and S. Albani, Proc. of the 2019 conference on Big Data from
Space (BiDS’2019), 49-52. EUR 29660 EN, Publications Office of the European Union, Luxembourg. ISBN: 978-92-76-00034-1,
doi:10.2760/848593.

Fridlind, A.M., van Lier-Walqui, M., Collis, S., Giangrande, S.E., Jackson, R.C., Li, X.,
Matsui, T., Orville, R., Picel, M.H., Rosenfeld, D., Ryzhkov, A., Weitz, R., Zhang, P., 2019.
Use of polarimetric radar measurements to constrain simulated convective cell evolution: a pilot
study with Lagrangian tracking. Atmos. Meas. Tech. 12, 2979–3000. https://doi.org/10.5194/amt-12-2979-2019

Wang J, R Wood, M Jensen, E Azevedo, C Bretherton, D Chand, C Chiu, X Dong, J Fast, A Gettelman,
S Ghan, S Giangrande, M Gilles, A Jefferson, P Kollias, C Kuang, A Laskin, E Lewis, X Liu, Y Liu,
E Luke, A McComiskey, F Mei, M Miller, A Sedlacek, and R Shaw. 2019. Aerosol and Cloud Experiments
in Eastern North Atlantic (ACE-ENA) Field Campaign Report. Ed. by Robert Stafford, ARM user facility. DOE/SC-ARM-19-012.

Jupyter et al., "Binder 2.0 - Reproducible, Interactive, Sharable Environments for Science at Scale."
Proceedings of the 17th Python in Science Conference. 2018. 10.25080/Majora-4af1f417-011
