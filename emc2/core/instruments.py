"""
=====================
emc2.core.instruments
=====================

This module stores example instruments.
"""
import numpy as np
import os

from .instrument import Instrument, ureg, quantity
from ..io import load_mie_file


class KAZR(Instrument):
    def __init__(self, site, *args):
        """
        This stores the information for the KAZR (Ka-band radar).
        """
        super().__init__(frequency=34.860 * ureg.GHz)
        if site.lower() not in ["sgp", "nsa", "awr", "ena"]:
            raise ValueError("Site must be one of 'sgp', 'ena', 'nsa', or 'awr'!")
        self.instrument_class = "radar"
        self.instrument_str = "KAZR"
        self.ext_OD = np.nan
        self.OD_from_sfc=True
        self.K_w = 0.88
        self.eps_liq = (5.489262 + 2.8267679j)**2
        if site.lower() == "ena":
            self.pt = 2200.
        else:
            self.pt = 2000.
        if site.lower() == "sgp":
            self.theta = 0.19
        else:
            self.theta = 0.31
        if site.lower() == "sgp":
            self.gain = 10**5.748
            self.Z_min_1km = -51.5
        elif site.lower() == "nsa":
            self.gain = 10**5.337
            self.Z_min_1km = -48.5
        elif site.lower() == "awr":
            self.gain = 10**5.273
            self.Z_min_1km = -45.5
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
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieKAZR_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieKAZR_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieKAZR_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieKAZR_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieKAZR_pi.dat")


class WACR(Instrument):
    def __init__(self, site, *args):
        """
        This stores the information for the WACR or M-WACR (W-band radars).
        """
        super().__init__(frequency=95.04 * ureg.GHz)
        if site.lower() not in ["sgp", "awr"]:
            raise ValueError("Site must be one of 'sgp' or 'awr'!")
        self.instrument_class = "radar"
        self.instrument_str = "WACR"
        self.ext_OD = np.nan
        self.OD_from_sfc=True
        self.K_w = 0.84
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
        else:
            self.gain = 10**3.78
            self.Z_min_1km = -40.0
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
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieWACR_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieWACR_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieWACR_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieWACR_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieWACR_pi.dat")


class RL(Instrument):
    def __init__(self, *args):
        """
        This stores the information for 355 nm lidars ,e.g., ARM Raman lidar (elastic channel).
        """
        super().__init__(wavelength=0.355 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "RL"
        self.beta_p_phase_thresh = [{'class': 'ice', 'class_ind': 2,
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 1.],
                                            'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef1', 'class_ind': 3,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 0., 0.]},
                                    {'class': 'undef2', 'class_ind': 4,
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 1.41421e-6, 1e-3, 0., 0.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 1e-3, 0., 0.]}]
        self.ext_OD = 4
        self.OD_from_sfc=True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.357247 + 2.4198595e-9j)**2
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan

        # Load mie tables
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieRL_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieRL_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieRL_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieRL_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieRL_pi.dat")


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
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 1.],
                                            'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef1', 'class_ind': 3,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 0., 0.]},
                                    {'class': 'undef2', 'class_ind': 4,
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 1.41421e-6, 1e-3, 0., 0.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 1e-3, 0., 0.]}]
        self.ext_OD = 4
        self.OD_from_sfc=True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.337273 + 1.7570744e-9j)**2
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan

        # Load mie tables
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieHSRL_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieHSRL_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieHSRL_pi.dat")


class CEIL(Instrument):
    def __init__(self, *args):
        """
        This stores the information for 910 nm lidars ,e.g., the CL31 ceilometer.
        """
        super().__init__(wavelength=0.910 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_str = "CEIL"
        self.ext_OD = 4
        self.OD_from_sfc=True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.323434 + 5.6988883e-7j)**2
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan

        # Load mie tables
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieCEIL_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieCEIL_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieCEIL_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieCEIL_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieCEIL_pi.dat")


class Ten64nm(Instrument):
    def __init__(self, *args):
        """
        This stores the information for the 1064 nm lidars, e.g., the 2-ch HSRL.
        """
        super().__init__(wavelength=1.064 * ureg.micrometer)
        self.instrument_class = "lidar"
        self.instrument_name = "1064nm"
        self.ext_OD = 4
        self.OD_from_sfc=True
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.320416 + 1.2588968e-6j)**2
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan
        # Load mie tables
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/Mie1064nm_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/Mie1064nm_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/Mie1064nm_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/Mie1064nm_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/Mie1064nm_pi.dat")


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
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 1.],
                                            'beta_p': [2e-5, 2e-5, 2e-6, 5e-7, 5e-7]},
                                    {'class': 'undef1', 'class_ind': 3,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 0., 0.]},
                                    {'class': 'undef2', 'class_ind': 4,
                                            'LDR': [0., 0.1000, 0.1001, 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 2e-5, 1.41421e-6, 1e-3, 0., 0.]},
                                    {'class': 'liquid', 'class_ind': 1,
                                            'LDR': [0., 0.2000, 0.2001, 1.],
                                            'beta_p': [2e-5, 1e-3, 0., 0.]}]
        self.ext_OD = 4
        self.OD_from_sfc=False
        self.eta = 1
        self.K_w = np.nan
        self.eps_liq = (1.337273 + 1.7570744e-9j)**2
        self.pt = np.nan
        self.theta = np.nan
        self.gain = np.nan
        self.Z_min_1km = np.nan
        self.lr = np.nan
        self.pr_noise_ge = np.nan
        self.pr_noise_md = np.nan
        self.tau_ge = np.nan
        self.tau_md = np.nan

        # Load mie tables
        data_path = os.path.join(os.path.dirname(__file__), 'mie_tables')
        self.mie_table["cl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
        self.mie_table["pl"] = load_mie_file(data_path + "/MieHSRL_liq.dat")
        self.mie_table["ci"] = load_mie_file(data_path + "/MieHSRL_ci.dat")
        if 'DHARMA' in args:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieHSRL_pi1.dat") # pi1 for 100 kg/m^2 (DHARMA)
        else:
            self.mie_table["pi"] = load_mie_file(data_path + "/MieHSRL_pi.dat")
