import numpy as np
import xarray as xr


def set_convective_sub_col_frac(model, hyd_type, N_columns=None):
    """
    Sets the hydrometeor fraction due to convection in each subcolumn.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of convective fraction for.
    hyd_type: str
        The hydrometeor type to generate the fraction for.
    N_columns: int or None
        The number of subcolumns to generate. Specifying this will set the number
        of subcolumns in the model parameter when the first subcolumns are generated.
        Therefore, after those are generated this must either be
        equal to None or the number of subcolumns in the model. Setting this to None will
        use the number of subcolumns in the model parameter.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the convective fraction in each subcolumn added.
    """

    if model.num_subcolumns is None and N_columns is None:
        raise RuntimeError("The number of subcolumns must be specified in the model!")

    if model.num_subcolumns != N_columns:
        raise ValueError("The number of subcolumns has already been specified (%d) and != %d" %
                         (model.num_subcolumns, N_columns))

    if model.num_subcolumns is None:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    data_frac = np.zeros(model.ds[model.conv_frac_names[hyd_type]].shape)
    data_frac = np.tile(data_frac, (N_columns, num_hydrometeor_classes, 1, 1)).T

    i = 0
    for hyd_type in model.conv_frac_names.keys():
        data_frac[i] = np.round(model.ds[model.conv_frac_names[hyd_type]] * N_columns)
        i = i + 1

    conv_profs = np.zeros((model.num_subcolumns, data_frac.shape[1], data_frac.shape[2]),
                          dtype=bool)
    for i in range(1, model.num_subcolumns):
        mask = np.where(data_frac == i)
        conv_profs[0:i, mask] = True

    model.ds[("conv_frac_subcolumns_" + hyd_type)] = xr.DataArray(
        conv_profs, dims=('subcolumn', model.time_dim, model.height_dim))

    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["units"] = "boolean"
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["long_name"] = (
        "Is there hydrometeors of type %s in each subcolumn?" % hyd_type)
    return model


def set_stratiform_sub_col_frac(model, N_columns=None):
    """
    Sets the hydrometeor fraction due to stratiform precipitation in each subcolumn.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of convective fraction for.
    N_columns: int or None
        The number of subcolumns to generate. Specifying this will set the number
        of subcolumns in the model parameter when the first subcolumns are generated.
        Therefore, after those are generated this must either be
        equal to None or the number of subcolumns in the model. Setting this to None will
        use the number of subcolumns in the model parameter.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """

    conv_profs1 = model.ds["conv_frac_subcolumns_cl"]
    conv_profs2 = model.ds["conv_frac_subcolumns_ci"]
    data_frac1 = model.ds[model.strat_frac_names["cl"]]
    data_frac1 = np.tile(data_frac1, (N_columns, 1, 1)).T
    data_frac1 = np.round(data_frac1 * N_columns)
    data_frac2 = model.ds[model.strat_frac_names["ci"]]
    data_frac2 = np.tile(data_frac2, (N_columns, 1, 1)).T
    data_frac2 = np.round(data_frac2 * N_columns)

    strat_profs1 = np.zeros_like(data_frac1, dtype=bool)
    strat_profs2 = np.zeros_like(data_frac2, dtype=bool)
    conv_profs = np.logical_and(conv_profs1.values, conv_prof2.values)
    is_cloud = np.logical_or(data_frac1.values > 0, data_frac2.values > 0)
    is_cloud_one_above = np.roll(is_cloud, -1)
    is_cloud_one_above[:, -1] = False
    overlapping_cloud = is_cloud and is_cloud_one_above

    cld_2_assigns = np.stack([data_frac1, data_frac2], axis=0)
    cld_2_assign_min = cld_2_assigns.min(axis=0)
    cld_2_assign_max = cld_2_assigns.max(axis=0)
    I_min = np.argmin(cld_2_assigns, axis=0)
    I_max = np.argmax(cld_2_assigns, axis=0)

    for tt in range(data_frac1.shape[0]):
        for j in range(data_frac1.shape[1], 0, -1):
            cld_2_assign = np.array([data_frac1[tt, j], data_frac2[tt,j]])
            I_min = np.argmin(cld_2_assign)
            I_max = np.argmax(cld_2_assign)
            if overlapping_cloud[tt, j]:
                overlying_locs1 = np.where(np.logical_and(strat_profs1[:, tt, j], ~conv_profs[:, tt, j]))
                overlying_locs2 = np.where(np.logical_and(strat_profs2[:, tt, j], ~conv_profs[:, tt, j]))
                overlying_num = np.array([len(overlying_locs1), len(overlying_locs2)])
                over_diff = abs(overlying_num[1] - overlying_num[0])
                Iover_min = np.argmin(overlying_num)
                Iover_max = np.argmax(overlying_num)
                over_unique_lo = np.setdiff1d(overlying_locs1, overlying_locs2)

                if overlying_num[Iover_min] > cld_2_assign[I_max]:
                    rand_locs = np.random.randint(0, overlying_num.min(), size=cld_2_assign[I_max])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs[0:cld_2_assign[I_min]]]
                    locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign = np.zeros(2)
                elif overlying_num[Iover_min] > cld_2_assign[I_min]:
                    rand_locs = np.random.randint(0, overlying_num.min(), size=cld_2_assign[I_min])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                    cld_2_assign[I_min] = 0
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign[I_max] -= overlying_num[Iover_min]

                    if cld_2_assign[I_max] > 0 and over_diff > 0:
                        rand_locs = np.random.randint(0, over_diff, size=cld_2_assign[I_max])
                        inds = over_unique_lo[rand_locs]
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                        cld_2_assign[I_max] = 0.
                    else:
                        locals()['strat_profs%d' % (I_max + 1)][over_unique_lo, tt, j] = True
                        cld_2_assign[I_max] -= over_diff
                elif overlying_num[Iover_max] > cld_2_assign[I_min]:
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                    locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                    locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign -= overlying_num[Iover_min]

                    if over_diff > cld_2_assign[I_max]:
                        rand_locs = np.random.randint(0, over_diff, size=cld_2_assign[I_max])
                        inds = over_unique_lo[rand_locs[1:cld_2_assign[I_min]]]
                        locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        inds = over_unique_lo[rand_locs]
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                        cld_2_assign = np.zeros(2)
                    else:
                        rand_locs = np.random.randint(0, over_diff, size=cld_2_assign[I_min])
                        inds = over_unique_lo[rand_locs]
                        locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        cld_2_assign[I_min] = 0
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                        cld_2_assign[I_max] -= over_diff
                else:
                    inds = locals()["overlying_locs%d" % (Iover_max + 1)]
                    locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                    locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign -= overlying_num[Iover_max]

            if cld_2_assign.max() > 0:
                sprof = locals()["strat_profs%d" % (I_max + 1)]
                free_locs_max = np.where(np.logical_and(~sprof[:, tt, j], ~conv_profs[:, tt, j]))[0]
                free_num = len(free_locs_max)
                rand_locs = np.random.randint(0, free_num, size=cld_2_assign[I_max])
                locals()["strat_profs%d" % (I_max + 1)][free_locs_max[rand_locs], tt, j] = True
                if cld_2_assign[I_min] > 0.:
                    locals()["strat_profs%d"
                             % (I_min + 1)][free_locs_max[rand_locs[0:cld_2_assign[I_min]]], tt, j] = True


    model_ds['strat_frac_subcolumns_cl'] = strat_profs1
    model_ds['strat_frac_subcolumns_ci'] = strat_profs2

    return model_ds




