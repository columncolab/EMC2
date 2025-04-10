"""
=====================
emc2.core.instruments
=====================

This module stores example instruments.
"""
import numpy as np

from .instrument import Instrument, ureg, quantity
from ..io import load_mie_file, load_scat_file, load_bulk_scat_file



class CSAPR(Instrument):
    def __init__(self, supercooled=True, elevation_angle=90., *args):
        """
        This stores the information for the ARM CSAPR.
        """
        super().__init__(frequency=6.25 * ureg.GHz)
        self.instrument_class = "radar"
        self.instrument_str = "CSAPR"
        self.elevation_angle = elevation_angle
        self.ext_OD = np.nan
        self.OD_from_sfc = True
        self.K_w = 0.93
        if supercooled:
            self.eps_liq = (7.434422 + 2.854179j)**2
        else:
            self.eps_liq = (8.474673 + 1.2560015j)**2
        self.pt = 125000.
        self.theta = 0.9
        self.gain = 10**4.51
        self.Z_min_1km = -35  # Based on Oue et al., GMD, 2020
        
        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class XSACR(Instrument):
    def __init__(self, supercooled=True, *args):
        """
        This stores the information for the AMF XSACR.
        """
        super().__init__(frequency=9.71 * ureg.GHz)
        self.instrument_class = "radar"
        self.instrument_str = "XSACR"
        self.ext_OD = np.nan
        self.OD_from_sfc = True
        self.K_w = 0.93
        if supercooled:
            self.eps_liq = (6.257395 + 3.018481j)**2
        else:
            self.eps_liq = (8.112180 + 1.7811075j)**2
        self.pt = 20000.
        self.theta = 1.40
        self.gain = 10**4.2
        self.Z_min_1km = -30
        
        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class KAZR(Instrument):
    def __init__(self, site, supercooled=True, *args):
        """
        This stores the information for the KAZR (Ka-band radar).
        """
        super().__init__(frequency=34.860 * ureg.GHz)
        if site.lower() not in ["sgp", "nsa", "awr", "ena", "mos"]:
            raise ValueError("Site must be one of 'sgp', 'ena', 'nsa', 'awr', or 'mos'!")
        self.instrument_class = "radar"
        self.instrument_str = "KAZR"
        self.ext_OD = np.nan
        self.OD_from_sfc = True
        self.K_w = 0.88
        if supercooled:
            self.eps_liq = (3.658396 + 1.987225j)**2
        else:
            self.eps_liq = (5.489262 + 2.8267679j)**2
        if site.lower() == "ena":
            self.pt = 2200.
        else:
            self.pt = 2000.
        if site.lower() == "sgp":
            self.theta = 0.20  # Widener et al. (2012)
        else:
            self.theta = 0.31  # Valid for all other KAZR and KAZR2
        if site.lower() == "sgp":
            self.gain = 10**5.748
            self.Z_min_1km = -51.5
        elif site.lower() == "ena":
            self.Z_min_1km = -56.5  # KAZR2
        elif site.lower() == "nsa":
            self.gain = 10**5.337
            self.Z_min_1km = -48.5
        elif site.lower() == "awr":
            self.gain = 10**5.273
            self.Z_min_1km = -45.5
        elif site.lower() == "mos":
            self.gain = 10**5.273
            self.Z_min_1km = -41.6
        else:
            self.gain = np.nan
            self.Z_min_1km = -56.5
        if site.lower() == "nsa":
            self.lr = 10**0.4
            self.pr_noise_ge = 10**-6.85
            self.pr_noise_md = 10**-7.23
            self.tau_ge = 0.3
            self.tau_md = 4.0
        elif site.lower() == "ena":
            self.lr = np.nan
            self.pr_noise_ge = np.nan
            self.pr_noise_md = np.nan
            self.tau_ge = 0.2
            self.tau_md = 4.0
        else:
            self.lr = np.nan
            self.pr_noise_ge = np.nan
            self.pr_noise_md = np.nan
            self.tau_ge = np.nan
            self.tau_md = np.nan
        
        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class WACR(Instrument):
    def __init__(self, site, supercooled=True, *args):
        """
        This stores the information for the WACR, M-WACR, and BASTA (W-band radars).
        """
        super().__init__(frequency=95.04 * ureg.GHz)
        if site.lower() not in ["sgp", "awr", "mos", "micre"]:
            raise ValueError("Site must be one of 'sgp' 'awr', 'mos', or 'micre'!")
        self.instrument_class = "radar"
        self.instrument_str = "WACR"
        self.ext_OD = np.nan
        self.OD_from_sfc = True
        self.K_w = 0.84
        if supercooled:
            self.eps_liq = (2.820550 + 1.123154j)**2
        else:
            self.eps_liq = (3.468221 + 2.1423486j)**2
        if site.lower() == "sgp":
            self.pt = 1513.
        else:
            self.pt = 1500.
        if site.lower() == "sgp":
            self.theta = 0.19
        else:
            self.theta = 0.38
        if site.lower() == "sgp":
            self.gain = 10**3.94
            self.Z_min_1km = -46.0
        elif site.lower() == "micre":  # BASTA 95 GHz radar
            self.gain = np.nan  # not determined (see Delanoë et al., 2016)
            self.Z_min_1km = -36.0  # effective sensitivity during MICRE
        else:
            self.gain = 10**3.78
            self.Z_min_1km = -40.0  # 0.2 s increments
        if site.lower() == "sgp":
            self.tau_ge = 0.3
            self.lr = np.nan
            self.pr_noise_ge = np.nan
            self.pr_noise_md = np.nan
            self.tau_md = np.nan
        else:
            self.lr = np.nan
            self.pr_noise_ge = np.nan
            self.pr_noise_md = np.nan
            self.tau_ge = 0.3
            self.tau_md = np.nan

        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class MWACR(WACR):
    pass


class BASTA(WACR):
    pass


class RL(Instrument):
    def __init__(self, *args):
        """
        This stores the information for 355 nm lidars ,e.g., ARM Raman lidar (elastic channel).
        """
        super().__init__(wavelength=0.355 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "RL"
        self.beta_p_phase_thresh = [{'class': 'ice', 'class_ind': 2,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef2', 'class_ind': 4,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 9e-6, 1., 1.]},
                                    {'class': 'undef1', 'class_ind': 3,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 1.41421e-4, 1e-3, 1., 1.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                     'LDR': [0., 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 1e-3, 1., 1.]}]
        self.ext_OD = 4
        self.OD_from_sfc = True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.357247 + 2.4198595e-9j) ** 2
    
        # Load mie tables
        self.load_instrument_scat_files(*args)  # load scattering files for this instrument


class HSRL(Instrument):
    def __init__(self, *args):
        """
        This stores the information for 532 nm lidars ,e.g., the High
        Spectral Resolution Lidar (HSRL), micropulse lidar (MPL).
        """
        super().__init__(wavelength=0.532 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "HSRL"
        self.beta_p_phase_thresh = [{'class': 'ice', 'class_ind': 2,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef2', 'class_ind': 4,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 9e-6, 1., 1.]},
                                    {'class': 'undef1', 'class_ind': 3,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 1.41421e-4, 1e-3, 1., 1.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                     'LDR': [0., 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 1e-3, 1., 1.]}]
        self.ext_OD = 4
        self.OD_from_sfc = True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.337273 + 1.7570744e-9j) ** 2

        # Load mie tables
        self.load_instrument_scat_files(*args)  # load scattering files for this instrument


class MPL(HSRL):
    pass


class CEIL(Instrument):
    def __init__(self, supercooled=True, *args):
        """
        This stores the information for 910 nm lidars ,e.g., the CL31 ceilometer.
        """
        super().__init__(wavelength=0.910 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "CEIL"
        self.ext_OD = 4
        self.OD_from_sfc = True
        self.eta = 1
        self.K_w = np.nan
        if supercooled:
            self.eps_liq = (1.3251203 + 5.1409006e-7j) ** 2
        else:
            self.eps_liq = (1.323434 + 5.6988883e-7j) ** 2

        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class Ten64nm(Instrument):
    def __init__(self, supercooled=True, *args):
        """
        This stores the information for the 1064 nm lidars, e.g., the 2-ch HSRL.
        """
        super().__init__(wavelength=1.064 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "1064nm"
        self.ext_OD = 4
        self.OD_from_sfc = True
        self.eta = 1
        self.K_w = np.nan
        if supercooled:
            self.eps_liq = (1.3235222 + 1.2181699e-6j) ** 2
        else:
            self.eps_liq = (1.320416 + 1.2588968e-6j) ** 2
        
        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class HSRL1064(Ten64nm):
    pass


class NEXRAD(Instrument):
    def __init__(self, supercooled=True, *args):
        """
        This stores the information for the NOAA NEXRAD radar
        Based on  https://www.roc.noaa.gov/WSR88D/Engineering/NEXRADTechInfo.aspx.
        """
        super().__init__(frequency=3.0 * ureg.GHz)
        self.instrument_class = "radar"
        self.instrument_str = "NEXRAD"
        self.ext_OD = np.nan
        self.K_w = 0.92
        if supercooled:
            self.eps_liq = (8.851160 + 1.940795j)**2
        else:
            self.eps_liq = (8.743107 + 0.64089981j)**2
        self.theta = 0.925
        self.pt = 700000.
        self.gain = 10**4.58
        self.Z_min_1km = -50.96  # long pulse at 1 km range (-23.0 dBZ at 25 km)
        self.Z_min_1km_short = -41.48  # short pulse at 1 km range (-7.5 dBZ at 50 km)
        self.lr = np.nan
        self.pr_noise_ge = 0.
        self.tau_ge = 1.57
        self.tau_md = 4.71
        
        # Load mie tables
        self.load_instrument_scat_files(supercooled, *args)  # load scattering files for this instrument


class CALIOP(Instrument):
    def __init__(self, *args):
        """
        This stores the information for 532 nm spaceborne lidars ,e.g.,
        the CALIOP on-board the CALIPSO satellite.
        """
        super().__init__(wavelength=0.532 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "HSRL"
        self.beta_p_phase_thresh = [{'class': 'ice', 'class_ind': 2,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef2', 'class_ind': 4,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 2e-6, 9e-6, 1., 1.]},
                                    {'class': 'undef1', 'class_ind': 3,
                                     'LDR': [0., 0.1, 0.100001, 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 2e-5, 1.41421e-4, 1e-3, 1., 1.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                     'LDR': [0., 0.2, 0.200001, 1.],
                                     'beta_p': [2e-5, 1e-3, 1., 1.]}]
        self.ext_OD = 4
        self.OD_from_sfc = False
        self.eta = 0.7
        self.K_w = np.nan
        self.eps_liq = (1.337273 + 1.7570744e-9j) ** 2

        # Load mie tables
        self.load_instrument_scat_files(*args)  # load scattering files for this instrument
