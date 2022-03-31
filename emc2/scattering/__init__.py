"""
===============================
emc2.simulator (emc2.simulator)
===============================

This module contains all of the calculations used by the radar/lidar
simulator in EMC^2.

.. autosummary::
    :toctree: generated/

    calc_microwave_ref_index
    calc_microwave_ref_index_ice
    scat_properties_water
    scat_properties_ice
    brandes
"""

from .ref_index import calc_microwave_ref_index, calc_microwave_ref_index_ice
from .mie_scattering import scat_properties_water, scat_properties_ice, brandes
