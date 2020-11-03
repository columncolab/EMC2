"""
====================
emc2.core.instrument
====================

This module stores the Instrument class.
"""
import numpy as np

from pint import UnitRegistry
from ..io import load_arm_file

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
    beta_p_phase_thresh: dict
        A dictionary whose keys are the names of hydrometeor classes used for phase
        classification using fixed thresholds.
        sub dict for every key contains the LDR and corresponding beta_p threshold values
    ext_OD: float
        The optical depth where we have full extinction of the lidar signal.
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
        self.c = 299792458.0
        self.R_d = 287.058
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
