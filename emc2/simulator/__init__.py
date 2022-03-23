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
    attenuation.calc_radar_Ze_min
    classification.lidar_classify_phase
    classification.radar_classify_phase
    classification.lidar_emulate_cosp_phase
    classification.calculate_phase_ratio

    psd.calc_mu_lambda
    psd.calc_re_thompson
    radar_moments.calc_total_reflectivity
    radar_moments.accumulate_attenuation
    radar_moments.calc_radar_empirical
    radar_moments.calc_radar_bulk
    radar_moments.calc_radar_micro
    radar_moments.calc_radar_moments
    lidar_moments.calc_total_alpha_beta
    lidar_moments.calc_LDR_and_ext
    lidar_moments.accumulate_OD
    lidar_moments.calc_lidar_empirical
    lidar_moments.calc_lidar_bulk
    lidar_moments.calc_lidar_micro
    lidar_moments.calc_lidar_moments
    main.make_simulated_data
    subcolumn.set_convective_sub_col_frac
    subcolumn.set_stratiform_sub_col_frac
    subcolumn.set_precip_sub_col_frac
    subcolumn.set_q_n
"""

from . import attenuation
from . import classification
from . import radar_moments
from . import lidar_moments
from . import psd
from . import subcolumn
from . import main
