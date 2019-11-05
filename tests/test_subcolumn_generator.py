import emc2
import numpy as np
import xarray as xr


def test_set_convective_profile():
    my_model = emc2.core.model.TestConvection()
    column_ds = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                     'cl', N_columns=8)
    where_gt_1km = np.where(np.logical_and(column_ds.ds['height'] > 1000.,
                                           column_ds.ds['t'] > 273.15))[0]
    where_lt_1km = np.where(np.logical_and(column_ds.ds['height'] < 1000.,
                                           column_ds.ds['t'] > 273.15))[0]
    assert np.all(column_ds.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km, 0])
    assert np.all(~column_ds.ds['conv_frac_subcolumns_cl'].values[:, where_lt_1km, 0])

    # Set convective fraction to half, but still have 8 subcolumns
    my_model.ds[my_model.conv_frac_names['cl']] *= 0.5
    column_ds = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                     'cl')
    assert np.all(column_ds.ds['conv_frac_subcolumns_cl'].values[:4, where_gt_1km, 0])
    assert np.all(~column_ds.ds['conv_frac_subcolumns_cl'].values[4:, where_gt_1km, 0])

    # Zero convection = all false arrays
    my_model.ds[my_model.conv_frac_names['cl']] *= 0
    column_ds = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                     'cl')
    assert np.all(~column_ds.ds['conv_frac_subcolumns_cl'].values)


def test_set_stratiform_profile():
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] > 273.15))[0]
    where_lt_1km = np.where(np.logical_and(my_model.ds['height'] < 1000.,
                                           my_model.ds['t'] > 273.15))[0]
    assert np.all(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km])
    assert np.all(~my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km])

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] > 273.15))[0]
    where_lt_1km = np.where(np.logical_and(my_model.ds['height'] < 1000.,
                                           my_model.ds['t'] > 273.15))[0]

    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km])
    assert num_strat == num_conv
    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    assert num_strat == 4
    assert num_conv == 4