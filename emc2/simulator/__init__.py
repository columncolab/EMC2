"""
===============================
emc2.simulator (emc2.simulator)
===============================

This module contains all of the calculations used by the radar/lidar
simulator in EMC^2.

.. autosummary::
    :toctree: generated/

    attenuation.calc_radar_atm_attenuation
    psd.calc_mu_lambda
    reflectivity.calc_radar_reflectivity_conv
    subcolumn.set_convective_sub_col_frac
"""

from . import attenuation
from . import reflectivity
from . import psd
from . import subcolumn
