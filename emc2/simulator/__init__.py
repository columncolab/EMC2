"""
===============================
emc2.simulator (emc2.simulator)
===============================

This module contains all of the calculations used by the radar/lidar
simulator in EMC^2.

.. autosummary::
    :toctree: generated/

    attenuation.calc_radar_atm_attenuation
    attenuation.calc_theory_beta_m
    psd.calc_mu_lambda
    radar_moments.calc_radar_reflectivity_conv
    radar_moments.calc_radar_moments
    lidar_moments.calc_LDR
    lidar_moments.calc_lidar_moments
    subcolumn.set_convective_sub_col_frac
    subcolumn.set_stratiform_sub_col_frac
    subcolumn.set_precip_sub_col_frac
    subcolumn.set_q_n
"""

from . import attenuation
from . import radar_moments
from . import lidar_moments
from . import psd
from . import subcolumn
