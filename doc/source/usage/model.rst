================================
Construction of the Model object
================================

EMC^2 uses the :py:mod:`emc2.core.Model` object to know what fields to load from a given
file in order to obtain the required information. In order to define your
:py:mod:`emc2.core.Model` object, it is highly recommended to use class inheritance. For example::

$ class ModelE(Model):
$    def __init__(self, file_path):
$        """
$        This loads a ModelE simulation with all of the necessary
$        parameters for EMC^2 to run.
$
$        Parameters
$        ----------
$        file_path: str
$            Path to a ModelE simulation.
$        """


In particular, EMC^2 will require information about the mixing ratio and
the number concentration of four
different species: cloud liquid (*cl*), cloud ice (*cl*), precipitating liquid (*pl*),
and precipitating ice (*pi*). These precipitation classes are commonly used
in many models in order to represent the cloud microphysical properties. EMC^2
derives the radar and lidar parameters for all 4 of these species. First,
in order to specify the fields that EMC^2 needs to look for, certain entries
whose keys correspond to the names of these four precipitation species must
be specified::

$   # Names of mixing ratios of species
$   self.q_names = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}
$   # Number concentration of each species
$   self.N_field = {'cl': 'ncl', 'ci': 'nci', 'pl': 'npl', 'pi': 'npi'}
$   # Convective fraction
$   self.conv_frac_names = {'cl': 'cldmccl', 'ci': 'cldmcci', 'pl': 'cldmcpl', 'pi': 'cldmcpi'}
$   # Stratiform fraction
$   self.strat_frac_names = {'cl': 'cldsscl', 'ci': 'cldssci', 'pl': 'cldsspl', 'pi': 'cldsspi'}
$   # Effective radius
$   self.re_fields = {'cl': 're_mccl', 'ci': 're_mcci', 'pi': 're_mcpi', 'pl': 're_mcpl'}
$   # Convective mixing ratio
$   self.q_names_convective = {'cl': 'QCLmc', 'ci': 'QCImc', 'pl': 'QPLmc', 'pi': 'QPImc'}
$   # Stratiform mixing ratio
$   self.q_names_stratiform = {'cl': 'qcl', 'ci': 'qci', 'pl': 'qpl', 'pi': 'qpi'}

In addition, other fields must also be specified::

$   # Water vapor mixing ratio
$   self.q_field = "q"
$   # Pressure
$   self.p_field = "p_3d"
$   # Height
$   self.z_field = "z"
$   # Temperature
$   self.T_field = "t"
$   # Name of height dimension
$   self.height_dim = "plm"
$   # Name of time dimension
$   self.time_dim = "time"

What if your model does not produce output for all 4 species, as is common
in many GCMs? Simply place in zero arrays that are the same shape as your
model fields!

Finally, we need to load in the model dataset. In order to do this, the model
must be loaded into a format that is compatible with **xarray**. If you are loading
a netCDF file, this is quite easy to do as **xarray** has native support for
netCDF files. For example, ModelE's files are in netCDF format, so we can simply do::

$   self.ds = xr.open_dataset(file_path)

One thing to be aware of is that you must ensure that all of your fields that
you load are 64-bit double precision floating point numbers in order to conform
with the assumed data types in ECM^2. Otherwise, underflow
errors are likely for many of the calculations. To ensure that this is the case,
one can simply loop over the variables in the file like this::

$   super().prepare_variables()

Finally, there are many assumptions that go into the calculation of the forward
modelled radar moments. For example, there are various fall-speed relationships
of the form :math:`V = aD^b` for different types of particles. Therefore, if you
are looking at a case where you think a specific kind of ice species may be
dominant, it is important to adjust these :math:`a` and :math:`b` constants. There
are numerous papers on this subject that are included in the references below.
It is best to match these coefficients with what is used in your model for the
best comparison. In order to adjust the constants that are used in the various
routines in EMC^2, you would have to fill in these dictionaries for each
hydrometeor species::

$   # Bulk density
$   self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3),
$                   'ci': 500. * ureg.kg / (ureg.m**3),
$                   'pl': 1000. * ureg.kg / (ureg.m**3),
$                   'pi': 250. * ureg.kg / (ureg.m**3)}
$   # Lidar ratio
$   self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
$                       'ci': 24. * ureg.dimensionless,
$                       'pl': 5.5 * ureg.dimensionless,
$                       'pi': 24.0 * ureg.dimensionless}
$   # Lidar LDR per hydrometeor mass content
$   self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
$                       'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
$                       'pl': 0.1 * 1 / (ureg.kg / (ureg.m**3)),
$                       'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
$   # a, b in V = aD^b
$   self.vel_param_a = {'cl': 3e-7, 'ci': 700., 'pl': 841.997, 'pi': 11.72}
$   self.vel_param_b = {'cl': 2. * ureg.dimensionless,
$                       'ci': 1. * ureg.dimensionless,
$                       'pl': 0.8 * ureg.dimensionless,
$                       'pi': 0.41 * ureg.dimensionless}
$   super()._add_vel_units()


++++++++++
References
++++++++++
Locatelli, J. D., and Hobbs, P. V. (1974), Fall speeds and masses
of solid precipitation particles, J. Geophys. Res., 79( 15), 2185– 2197,
doi:10.1029/JC079i015p02185.

Brown, P.R. and P.N. Francis, 1995: Improved Measurements of the Ice
Water Content in Cirrus Using a Total-Water Probe.
J. Atmos. Oceanic Technol., 12, 410–414,
https://doi.org/10.1175/1520-0426(1995)012<0410:IMOTIW>2.0.CO;2

Heymsfield, A.J., G. van Zadelhoff, D.P. Donovan, F. Fabry, R.J. Hogan,
and A.J. Illingworth, 2007: Refinements to Ice Particle Mass Dimensional
and Terminal Velocity Relationships for Ice Clouds. Part II: Evaluation
and Parameterizations of Ensemble Ice Particle Sedimentation Velocities.
J. Atmos. Sci., 64, 1068–1088, https://doi.org/10.1175/JAS3900.1