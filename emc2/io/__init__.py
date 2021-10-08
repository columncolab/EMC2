"""
=================
emc2.io (emc2.io)
=================

This module contains all of the basic I/O functions of EMC^2:

.. autosummary::
    :toctree: generated/

    load_mie_file
    load_scat_file
    load_bulk_scat_file
    load_arm_file
"""

from .load_mie_file import load_mie_file
from .load_scat_file import load_scat_file
from .load_bulk_scat_file import load_bulk_scat_file
from .load_obs import load_arm_file
