import numpy as np

class Instrument(object):
    """
    This is the base class which holds the information needed to contain the instrument parameters for the
    simulator.

    Parameters
    ----------
    instrument_str: str
        The name of the instrument.
    instrument_class: str
        The class of the instrument. Currently must be one of 'radar,' or 'lidar'.
    freq: float
        The frequency of the instrument.
    wavelength: float
        The wavelength of the instrument
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
    instrument_str = ""
    instrument_class = ""
    freq = np.nan
    wavelength = np.nan
    ext_OD = np.nan
    K_w = np.nan
    eps_liq = np.nan
    location_code = ""
    pt = np.nan
    theta = np.nan
    gain = np.nan
    Z_min_1km = np.nan
    lr = np.nan
    pr_noise_ge = np.nan
    pr_noise_md = np.nan
    tau_ge = np.nan
    tau_md = np.nan
    c = 299792458.0
    R_d = 287.058

    def __init__(self, frequency=None, wavelength=None):
        if frequency is None and wavelength is None:
            raise ValueError("Your instrument must have a frequency or wavelength!")
        if frequency is None:
            self.freq = self.c/wavelength*1e-3
            self.wavelength = wavelength
        elif wavelength is None:
            self.freq = frequency
            self.wavelength = self.c/frequency*1e-3
        else:
            self.freq = frequency
            self.wavelength = wavelength



