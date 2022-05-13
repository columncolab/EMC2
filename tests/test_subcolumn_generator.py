import emc2
import numpy as np
import xarray as xr


def test_set_convective_profile():
    my_model = emc2.core.model.TestConvection()
    column_ds = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                     'cl', N_columns=8)
    where_gt_1km = np.argwhere(np.logical_and(
        column_ds.ds['z'].values[0] > 1000.,
        column_ds.ds['t'].values[0] >= 273.15))
    where_lt_1km = np.argwhere(np.logical_and(
        column_ds.ds['z'].values[0] < 1000.,
        column_ds.ds['t'].values[0] >= 273.15))
    print(where_lt_1km.shape)
    assert np.all(column_ds.ds['conv_frac_subcolumns_cl'].values[:, 0, where_gt_1km])
    assert np.all(~column_ds.ds['conv_frac_subcolumns_cl'].values[:, 0, where_lt_1km])
    
    # Set convective fraction to half, but still have 8 subcolumns
    my_model.ds[my_model.conv_frac_names['cl']] *= 0.5
    column_ds = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                     'cl')
    assert np.all(column_ds.ds['conv_frac_subcolumns_cl'].values[:4, 0, where_gt_1km])
    assert np.all(~column_ds.ds['conv_frac_subcolumns_cl'].values[4:, 0, where_gt_1km])

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
                                           my_model.ds['t'] >= 273.15))[0]

    assert np.all(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km])
    assert np.all(~my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km])

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    where_gt_1km = np.where(np.logical_and(my_model.ds['z'] > 1000.,
                                           my_model.ds['t'] >= 273.15))[0]

    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km])
    assert num_strat == num_conv
    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    assert num_strat == 480
    assert num_conv == 480

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model.ds[my_model.strat_frac_names['cl']] *= 0.5
    my_model.ds[my_model.strat_frac_names['ci']] *= 0.5
    my_model.ds[my_model.conv_frac_names['cl']] *= 0.5
    my_model.ds[my_model.conv_frac_names['ci']] *= 0.5
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model, 'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model, 'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km[1]])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km[1]])
    assert num_conv == 240
    assert num_strat == 240

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model.ds[my_model.strat_frac_names['cl']] *= 0.25
    my_model.ds[my_model.strat_frac_names['ci']] *= 0.25
    my_model.ds[my_model.conv_frac_names['cl']] *= 0.25
    my_model.ds[my_model.conv_frac_names['ci']] *= 0.25
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model, 'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model, 'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_cl'].values[:, where_gt_1km[0]])
    assert num_conv == 120
    assert num_strat == 120


def test_set_precip_profile():
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] >= 273.15))[0]

    assert np.all(my_model.ds['strat_frac_subcolumns_pl'].values[:, where_gt_1km])
    assert np.all(~my_model.ds['conv_frac_subcolumns_pl'].values[:, where_gt_1km])

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] >= 273.15))[0]
    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_pl'].values[:, where_gt_1km], axis=0)
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_pl'].values[:, where_gt_1km], axis=0)

    assert np.all(num_strat == 4)
    assert np.all(num_conv == 4)

    my_model = emc2.core.model.TestHalfAndHalf()
    my_model.ds[my_model.strat_frac_names['pl']] *= 0.25
    my_model.ds[my_model.strat_frac_names['pi']] *= 0.25
    my_model.ds[my_model.conv_frac_names['pl']] *= 0.5
    my_model.ds[my_model.conv_frac_names['pi']] *= 0.5

    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] >= 273.15))[0]

    num_strat = np.sum(my_model.ds['strat_frac_subcolumns_pl'].values[:, where_gt_1km], axis=0)
    num_conv = np.sum(my_model.ds['conv_frac_subcolumns_pl'].values[:, where_gt_1km], axis=0)

    assert np.all(num_strat == 1)
    assert np.all(num_conv == 2)


def test_set_qn():
    my_model = emc2.core.model.TestAllStratiform()
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'cl', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_convective_sub_col_frac(my_model,
                                                                    'ci', N_columns=8)
    my_model = emc2.simulator.subcolumn.set_stratiform_sub_col_frac(my_model)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=False)
    my_model = emc2.simulator.subcolumn.set_precip_sub_col_frac(my_model, is_conv=True)
    my_model = emc2.simulator.subcolumn.set_q_n(my_model, 'cl', is_conv=False, qc_flag=False)
    where_gt_1km = np.where(np.logical_and(my_model.ds['height'] > 1000.,
                                           my_model.ds['t'] >= 273.15))[0]
    # There should only be a field named
    assert "strat_q_subcolumns_cl" in my_model.ds.variables.keys()
    q_sum = np.mean(my_model.ds["strat_q_subcolumns_cl"].values, axis=0)
    assert np.all(q_sum[where_gt_1km] > 0)
    assert np.all(q_sum[~where_gt_1km] == 0)
    qcl = my_model.ds[my_model.q_names_stratiform["cl"]].values
    np.testing.assert_almost_equal(q_sum, qcl)
