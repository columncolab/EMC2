"""
====================
emc2.core.instrument
====================

This module stores the Instrument class.
"""
import numpy as np

from pint import UnitRegistry
from ..io import load_arm_file
from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, UnnormalizedGammaPSD
from pytmatrix import orientation, radar, tmatrix_aux, refractive

ureg = UnitRegistry()
quantity = ureg.Quantity


class Instrument(object):
    """
    This is the base class which holds the information needed to contain the instrument parameters for the
    simulator.

    Attributes
    ----------
    instrument_str: str
        The name of the instrument.
    instrument_class: str
        The class of the instrument. Currently must be one of 'radar,' or 'lidar'.
    freq: float
        The frequency of the instrument.
    wavelength: float
        The wavelength of the instrument
    beta_p_phase_thresh: list of dicts or None
        If a list, each index contains a dictionaly with class name, class integer
        value (mask order), LDR value bounds, and the corresponding beta_p threshold
        (thresholds are linearly interpolated between LDR values). In order for the
        method to operate properly, the list should be arranged from the lowest to
        highest beta_p threshold values for a given LDR, that is,
        beta_p[i+1 | LDR=x] >= beta_p[i | LDR=x].
    ext_OD: float
        The optical depth where we have full extinction of the lidar signal.
    OD_from_sfc: Bool
        If True (default), optical depth will be calculated from the surface. If False,
        optical depth will be calculated from the top of the atmosphere.
    eta: float
        Multiple scattering coefficient.
    K_w: float
        The index of refraction of water used for Ze calculation.
        See the ARM KAZR handbook (Widener et al. 2012)
    eps_liq: float
        The complex dielectric constant for liquid water.
    pt: float
        Transmitting power in Watts.
    theta: float
        3 dB beam width in degrees
    gain: float
        The antenna gain in linear units.
    Z_min_1km: float
        The minimum detectable signal at 1 km in dBZ
    lr: float
        Attenuation based on the the general attributes in the spectra files.
    pr_noise_ge: float
        Minimum detectable signal in mW.
    tau_ge: float
        Pulse width in mus.
    tau_md: float
        Pulse width in mus.
    """

    def __init__(self, frequency=None, wavelength=None):
        self.instrument_str = ""
        self.instrument_class = ""
        self.freq = np.nan
        self.wavelength = np.nan
        self.beta_p_phase_thresh = []
        self.ext_OD = np.nan
        self.OD_from_sfc = True
        self.eta = np.nan
        self.K_w = np.nan
        self.eps_liq = np.nan
        self.location_code = ""
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan
        self.c = 299792458.0  # m/s
        self.R_d = 287.058  # J K^-1 Kg^-1
        self.g = 9.80665  # m/s^2
        self.rho_i = 917 * ureg.kg / (ureg.m**3)  # kg/m^3 (0 C 1013.25 hPa, typical model accuracy)
        self.rho_l = 1000 * ureg.kg / (ureg.m**3)  # kg/m^3 (typical model accuracy)
        self.scatterer = None
        if frequency is None and wavelength is None:
            raise ValueError("Your instrument must have a frequency or wavelength!")
        if frequency is None:
            self.freq = self.c / wavelength.to('meter').magnitude
            self.wavelength = wavelength.to('micrometer').magnitude
        elif wavelength is None:
            self.freq = frequency.to('Hz').magnitude
            self.wavelength = self.c / self.freq * 1e6
        else:
            self.freq = frequency.to('Hz').magnitude
            self.wavelength = wavelength.to('micrometer').magnitude

        self.mie_table = {}
        self.scat_table = {}  # scattering calculation LUTs (e.g., C6 or m-D, A-D relationships).
        self.bulk_table = {}
        self.scatterer = {}
        self.ds = None

    def read_arm_netcdf_file(self, filename, **kwargs):
        """
        Loads a netCDF file that corresponds to ARM standards.

        Parameters
        ----------
        filename: str

        Additional keyword arguments are passed into :py:func:`act.io.armfiles.read_netcdf`

        """
        self.ds = load_arm_file(filename, **kwargs)

    def initialize_tmatrix_scatterer(self, model, band):
        """
        This will initalize the TMatrix scatterers based off of the model parameters.
        This step is required if you are using dual polariation quantities. EMC^2
        will generate the dual polarization moments assuming that the scatterers
        follow oblate spheroids with effective densities and aspect ratios that are
        provided by the :code:`model` object.

        Parameters
        ----------
        model: model object
            The model to pull the hydrometeor assumptions from.
        band: str
            The radar band to simulate
        """
        wavelength = getattr(tmatrix_aux, 'wl_%s' % band)
        self.scatterer["cl"] = Scatterer(wavelength=wavelength,
                                   m=refractive.m_w_0C[wavelength],
                                   axis_ratio_func=lambda x: 1 / model.asp_ratio_func["cl"](x))
        self.scatterer["cl"].psd_integrator = PSDIntegrator()
        self.scatterer["cl"].psd_integrator.D_max = 10.0
        self.scatterer["cl"].psd_integrator.axis_ratio_func = lambda x: 1 / model.asp_ratio_func["cl"](x)
        self.scatterer["cl"].or_pdf = orientation.gaussian_pdf(20.)
        self.scatterer["cl"].orient = orientation.orient_averaged_fixed
        self.scatterer["cl"].psd_integrator.geometries = (
            tmatrix_aux.geom_horiz_back,
            tmatrix_aux.geom_horiz_forw,
        )
        self.scatterer["cl"].psd_integrator.init_scatter_table(self.scatterer["cl"])

        self.scatterer["pl"] = Scatterer(wavelength=wavelength,
                                         m=refractive.m_w_0C[wavelength],
                                         axis_ratio_func=lambda x: 1 / model.asp_ratio_func["pl"](x))
        self.scatterer["pl"].psd_integrator = PSDIntegrator()
        self.scatterer["pl"].psd_integrator.D_max = 10.0
        self.scatterer["pl"].psd_integrator.axis_ratio_func = lambda x: 1 / model.asp_ratio_func["pl"](x)
        self.scatterer["pl"].or_pdf = orientation.gaussian_pdf(20.)
        self.scatterer["pl"].orient = orientation.orient_averaged_fixed
        self.scatterer["pl"].psd_integrator.geometries = (
            tmatrix_aux.geom_horiz_back,
            tmatrix_aux.geom_horiz_forw,
        )
        self.scatterer["pl"].psd_integrator.init_scatter_table(self.scatterer["cl"])
        self.scatterer["ci"] = Scatterer(wavelength=wavelength,
                                         m=refractive.mi(wavelength, model.Rho_hyd["ci"].magnitude * 1e-3),
                                         axis_ratio_func=lambda x: 1 / model.asp_ratio_func["ci"](x))
        self.scatterer["ci"].psd_integrator = PSDIntegrator()
        self.scatterer["ci"].psd_integrator.D_max = 10.0
        self.scatterer["ci"].psd_integrator.axis_ratio_func = lambda x: 1 / model.asp_ratio_func["ci"](x)
        self.scatterer["ci"].or_pdf = orientation.gaussian_pdf(20.)
        self.scatterer["ci"].orient = orientation.orient_averaged_fixed
        self.scatterer["ci"].psd_integrator.geometries = (
            tmatrix_aux.geom_horiz_back,
            tmatrix_aux.geom_horiz_forw,
        )
        self.scatterer["ci"].psd_integrator.init_scatter_table(self.scatterer["ci"])
        self.scatterer["pi"] = Scatterer(wavelength=wavelength,
                                         m=refractive.mi(wavelength, model.Rho_hyd["pi"].magnitude * 1e-3),
                                         axis_ratio_func=lambda x: 1 / model.asp_ratio_func["pi"](x))
        self.scatterer["pi"].psd_integrator = PSDIntegrator()
        self.scatterer["pi"].psd_integrator.D_max = 10.0
        self.scatterer["pi"].psd_integrator.axis_ratio_func = lambda x: 1 / model.asp_ratio_func["pi"](x)
        self.scatterer["pi"].or_pdf = orientation.gaussian_pdf(20.)
        self.scatterer["pi"].orient = orientation.orient_averaged_fixed
        self.scatterer["pi"].psd_integrator.geometries = (
            tmatrix_aux.geom_horiz_back,
            tmatrix_aux.geom_horiz_forw,
        )
        self.scatterer["pi"].psd_integrator.init_scatter_table(self.scatterer["pi"])



