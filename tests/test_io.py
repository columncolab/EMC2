import emc2
import numpy as np


def test_mie_file():
    KAZR = emc2.core.instruments.KAZR('nsa')
    assert "cl" in KAZR.mie_table.keys()
    assert "alpha_p" in KAZR.mie_table["cl"].variables.keys()
    print(KAZR.mie_table["cl"]["alpha_p"])
    assert KAZR.mie_table["cl"]["alpha_p"][0] == 0.69115511e-19
    assert np.all(KAZR.mie_table["cl"]["wavelength"] == 8.6e3)
