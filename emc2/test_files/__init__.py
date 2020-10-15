"""
This module contains test files for the continuous integration.
The files here are not recommended for the end user, but only for
development.

"""

import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

TEST_SUBCOL_FILE = os.path.join(DATA_PATH, 'test_subcol.20160816.010000.nc')
TEST_INST_PLOT_FILE = os.path.join(DATA_PATH, 'test_instrument_plot.20160816.100000.nc')
