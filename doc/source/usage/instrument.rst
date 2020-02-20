===========================================
Construction of the EMC^2 Instrument Object
===========================================

The :py:mod:`emc2.core.Instrument` object contains all of the information that is needed
for EMC^2 to both visualize instrument data as well as the instrument specific parameters
that are needed to simulate a lidar or a radar.

It is recommended that you use class inheritance to make your own **Instrument**
objects. This helps to ensure that your object has the required methods for
EMC^2 to use. For example, the object for the High Spectral Resolution Lidar
is created as follows::

$ class HSRL(Instrument):
$     def __init__(self):
$         """
$         This stores the information for the High Resolution Spectral Lidar.
$         """

Make sure all of the variables are initalized and the proper wavelength is
entered::

$         super().__init__(wavelength=0.532 * ureg.micrometer)

Define whether your instrument is a lidar or a radar::

$         self.instrument_class = "lidar"

Name your instrument::

$         self.instrument_str = "HSRL"

If it's a lidar, we'll tell EMC^2 to consider the signal to be extinct
above a certain optical depth::

$         self.ext_OD = 4

If we have a radar, we will assign the dielectric constant.
Since this is a lidar, we will assign NaN to *K_w*::

$         self.K_w = np.nan

Tell EMC^2 what the index of refraction of water is::

$         self.eps_liq = (1.337273 + 1.7570744e-9j)**2

The rest of these parameters are needed if you are specifying a radar::

$         self.pt = np.nan
$         self.theta = np.nan
$         self.gain = np.nan
$         self.Z_min_1km = np.nan
$         self.lr = np.nan
$         self.pr_noise_ge = np.nan
$         self.pr_noise_md = np.nan
$         self.tau_ge = np.nan
$         self.tau_md = np.nan

In addition, the mie tables for each hydrometeor class must be specified::

$         # Load mie tables
$         data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
$         self.mie_table["cl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
$         self.mie_table["pl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
$         self.mie_table["ci"] = load_mie_file(data_path + "/MieHSRL_ci.dat")
$         self.mie_table["pi"] = load_mie_file(data_path + "/MieHSRL_pi.dat")

===================================================
Addition of more wavelengths - format of Mie tables
===================================================

If you wish to use an instrument with a different wavelength than what is currently used in EMC^2,
then you must generate the Mie scattering parameters using your favorite Mie scattering code for
the given wavelength. The mie scattering tables are xarray datasets with the following variables:

+--------------------+---------------------------------------+
| Variable           |  Description                          |
+====================+=======================================+
| wavelength         |  The wavelength :math:`\lambda` of the|
|                    |  beam in microns.                     |
+--------------------+---------------------------------------+
| p_diam             |  The diameter :math:`D` of the        |
|                    |  particle in microns                  |
+--------------------+---------------------------------------+
| size_parameter     |  The size parameter of the particle.  |
|                    |  This is defined as                   |
|                    |  :math:`\frac{\pi D}{\lambda}`.       |
+--------------------+---------------------------------------+
|  compre_real       |  Real part of complex index of        |
|                    |  refraction of the sphere.            |
+--------------------+---------------------------------------+
|  compre_im         |  Imaginary part of complex index of   |
|                    |  refraction of the sphere             |
+--------------------+---------------------------------------+
|  scat_p            |  Forward scattering cross section in  |
|                    |  :math:`{\mu m}^2`.                   |
+--------------------+---------------------------------------+
|  alpha_p           |  Back scattering cross section in     |
|                    |  :math:`{\mu m}^2`.                   |
+--------------------+---------------------------------------+
|  beta_p            |  Extinction cross section in          |
|                    |  :math:`{\mu m}^2`.                   |
+--------------------+---------------------------------------+
|  scat_eff          |  Scattering efficiency                |
+--------------------+---------------------------------------+
|  ext_eff           |  Extinction efficiency                |
+--------------------+---------------------------------------+
|  backscatt_eff     |  Backscattering efficiency            |
+--------------------+---------------------------------------+

These mie tables must be generated for the four precipitation classes.