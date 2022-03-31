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

    instruments.KAZR
    instruments.HSRL
    instruments.CSAPR
    instruments.NEXRAD
    instruments.XSACR
    instruments.Ten64nm
    model.ModelE
    model.TestModel
    model.E3SM
    model.DHARMA
    model.WRF
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
