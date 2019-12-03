import emc2
import xarray as xr
import numpy as np


def test_convective_reflectivity():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestModel()
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "cl")
    assert np.all(my_model.ds["Ze"].values < 30)
    assert my_model.ds["Ze"].values.max() > -20.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "pl")
    assert my_model.ds["Ze"].values.max() > 25.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "ci")
    assert np.all(my_model.ds["Ze"].values < 10)
    assert my_model.ds["Ze"].values.max() > -10.
    my_model = emc2.simulator.radar_moments.calc_radar_reflectivity_conv(instrument, my_model, "pi")
    assert np.all(my_model.ds["Ze"].values < 10)
    assert my_model.ds["Ze"].values.max() > -10


def test_radar_moments_all_convective():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestConvection()
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, True)
    assert my_model.ds["sub_col_Ze_tot_conv"].values.max() > 40
    assert my_model.ds["sub_col_Ze_cl_conv"].values.max() < 30.
    assert my_model.ds["sub_col_Ze_pl_conv"].values.max() > 40.


def test_radar_moments_all_stratiform():
    instrument = emc2.core.instruments.KAZR('nsa')
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, convective=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'ci', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pl', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'pi', is_conv=False, qc_flag=False)
    my_model = emc2.simulator.radar_moments.calc_radar_moments(instrument, my_model, False)
    assert my_model.ds["sub_col_Ze_tot_strat"].values.max() > 40
    assert my_model.ds["sub_col_Ze_cl_strat"].values.max() < 30.
    assert my_model.ds["sub_col_Ze_pl_strat"].values.max() > 40.
