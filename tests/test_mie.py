import emc2
import numpy as np
import pytest


def test_mie_liquid():
    pymiescat = pytest.importorskip('PyMieScatt')
    KAZR = emc2.core.instruments.KAZR('nsa')
    KAZR_mie = emc2.scattering.scat_properties_water(
        KAZR.mie_table["cl"].p_diam * 1e6, KAZR.wavelength * 1e-4, 0.)
    assert np.corrcoef(KAZR_mie.alpha_p.values, KAZR.mie_table["cl"].alpha_p.values)[0, 1] > 0.9
    assert np.corrcoef(KAZR_mie.beta_p.values, KAZR.mie_table["cl"].beta_p.values)[0, 1] > 0.8
    assert np.corrcoef(KAZR_mie.scat_p.values, KAZR.mie_table["cl"].scat_p.values)[0, 1] > 0.8
    np.testing.assert_allclose(KAZR_mie.p_diam.values, KAZR.mie_table["cl"].p_diam.values, rtol=1e-2)


def test_mie_ice():
    pymiescat = pytest.importorskip('PyMieScatt')
    KAZR = emc2.core.instruments.KAZR('nsa')
    KAZR_mie = emc2.scattering.scat_properties_ice(
        KAZR.mie_table["ci"].p_diam * 1e6, KAZR.wavelength * 1e-4, 0.)
    assert np.corrcoef(KAZR_mie.alpha_p.values, KAZR.mie_table["ci"].alpha_p.values)[0, 1] > 0.9
    assert np.corrcoef(KAZR_mie.beta_p.values, KAZR.mie_table["ci"].beta_p.values)[0, 1] > 0.9
    assert np.corrcoef(KAZR_mie.scat_p.values, KAZR.mie_table["ci"].scat_p.values)[0, 1] > 0.9
