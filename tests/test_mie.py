import emc2
import numpy as np


def test_mie_liquid():
    KAZR = emc2.core.instruments.KAZR('nsa')
    KAZR_mie = emc2.scattering.scat_properties_water(
        KAZR.mie_table["cl"].p_diam * 1e6, KAZR.wavelength * 1e-4, 20.)
    np.testing.assert_allclose(KAZR_mie.alpha_p.values, KAZR.mie_table["cl"].alpha_p.values, rtol=0.1)
    np.testing.assert_allclose(KAZR_mie.beta_p.values, KAZR.mie_table["cl"].beta_p.values, rtol=0.1)
    np.testing.assert_allclose(KAZR_mie.scat_p.values, KAZR.mie_table["cl"].scat_p.values, rtol=0.1)


def test_mie_ice():
    KAZR = emc2.core.instruments.KAZR('nsa')
    KAZR_mie = emc2.scattering.scat_properties_ice(
        KAZR.mie_table["ci"].p_diam * 1e6, KAZR.wavelength * 1e-4, 0.)
    np.testing.assert_allclose(KAZR_mie.alpha_p.values, KAZR.mie_table["ci"].alpha_p.values, atol=1e-4)
    np.testing.assert_allclose(KAZR_mie.beta_p.values, KAZR.mie_table["ci"].beta_p.values, atol=1e-4)
    np.testing.assert_allclose(KAZR_mie.scat_p.values, KAZR.mie_table["ci"].scat_p.values, atol=1e-4)
