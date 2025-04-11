"""
=====================
emc2.core (emc2.core)
=====================

.. currentmodule:: emc2.core

The procedures in this module contain the core data structures of EMC^2.
In particular, the Instrument class that describes the characteristics
of the instrument to be simulated is stored here. In addition, global
constants used by EMC^2 are also stored in this module.

.. autosummary::
    :toctree: generated/

    instruments.CSAPR
    instruments.XSACR
    instruments.KAZR
    instruments.WACR
    instruments.MWACR
    instruments.BASTA
    instruments.RL
    instruments.HSRL
    instruments.MPL
    instruments.CEIL
    instruments.Ten64nm
    instruments.HSRL1064
    instruments.NEXRAD
    instruments.CALIOP
    model.E3SMv1
    model.E3SMv3
    model.CESM2
    model.ModelE
    model.WRF
    model.DHARMA
    model.TestConvection
    model.TestAllStratiform
    model.TestHalfAndHalf
    model.TestModel
    Instrument
    Model

In addition the :func:`emc2.core.quantity` is equivalent to pint's
UnitRegistry().Quantity object. This allows for the use of quantities
with units, making EMC^2 unit aware.
"""

from . import instruments
from .instrument import Instrument
from . import model
from .model import Model
