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
    np.seterr(divide='ignore', invalid='ignore')
    if model.num_subcolumns == 0 and N_columns is None:
        raise RuntimeError("The number of subcolumns must be specified in the model!")

    if model.num_subcolumns != N_columns and model.num_subcolumns != 0 and N_columns is not None:
        raise ValueError("The number of subcolumns has already been specified (%d) and != %d" %
                         (model.num_subcolumns, N_columns))

    if model.num_subcolumns == 0:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    data_frac = np.round(model.ds[model.conv_frac_names[hyd_type]].values * model.num_subcolumns)

    # In case we only have one time step
    if len(data_frac.shape) == 1:
        data_frac = data_frac[np.newaxis, :]

    conv_profs = np.zeros((model.num_subcolumns, data_frac.shape[0], data_frac.shape[1]),
                          dtype=bool)
    for i in range(1, model.num_subcolumns + 1):
        for k in range(data_frac.shape[1]):
            mask = np.where(data_frac[:, k] == i)[0]
            conv_profs[0:i, mask, k] = True
    my_dims = model.ds[model.conv_frac_names[hyd_type]].dims
    model.ds[("conv_frac_subcolumns_" + hyd_type)] = xr.DataArray(
        conv_profs, dims=('subcolumn', my_dims[0], my_dims[1]))
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["units"] = "boolean"
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["long_name"] = (
        "Is there hydrometeors of type %s in each subcolumn?" % hyd_type)

    return model


def set_stratiform_sub_col_frac(model):
    """
    Sets the hydrometeor fraction due to stratiform cloud particles in each subcolumn.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of stratiform fraction for.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """

    if "conv_frac_subcolumns_cl" not in model.ds.variables.keys():
        raise KeyError("You have to generate the convective fraction in each subcolumn " +
                       "before the stratiform fraction in each subcolumn is generated.")

    if "conv_frac_subcolumns_ci" not in model.ds.variables.keys():
        raise KeyError("You have to generate the convective fraction in each subcolumn " +
                       "before the stratiform fraction in each subcolumn is generated.")
    np.seterr(divide='ignore', invalid='ignore')
    conv_profs1 = model.ds["conv_frac_subcolumns_cl"]
    conv_profs2 = model.ds["conv_frac_subcolumns_ci"]
    N_columns = len(model.ds["subcolumn"])
    subcolumn_dims = conv_profs1.dims
    data_frac1 = model.ds[model.strat_frac_names["cl"]]
    data_frac1 = np.round(data_frac1.values * N_columns).astype(int)
    data_frac2 = model.ds[model.strat_frac_names["ci"]]
    data_frac2 = np.round(data_frac2.values * N_columns).astype(int)

    strat_profs1 = np.zeros((N_columns, data_frac1.shape[0], data_frac1.shape[1]), dtype=bool)
    strat_profs2 = np.zeros_like(strat_profs1, dtype=bool)
    conv_profs = np.logical_or(conv_profs1.values, conv_profs2.values)
    is_cloud = np.logical_or(data_frac1 > 0, data_frac2 > 0)
    is_cloud_one_above = np.roll(is_cloud, -1, axis=1)
    overlapping_cloud = np.logical_and(is_cloud, is_cloud_one_above)

    cld_2_assigns = np.stack([data_frac1, data_frac2], axis=0)
    cld_2_assign_min = cld_2_assigns.min(axis=0)
    cld_2_assign_max = cld_2_assigns.max(axis=0)
    I_min = np.argmin(cld_2_assigns, axis=0)
    I_max = np.argmax(cld_2_assigns, axis=0)
    for tt in range(data_frac1.shape[0]):
        for j in range(data_frac1.shape[1] - 2, -1, -1):
            cld_2_assign = np.array([data_frac1[tt, j], data_frac2[tt, j]])
            I_min = np.argmin(cld_2_assign)
            I_max = np.argmax(cld_2_assign)
            if cld_2_assign[I_max] == 0:
                continue
            if overlapping_cloud[tt, j]:
                overlying_locs1 = np.where(np.logical_and(strat_profs1[:, tt, j + 1], ~conv_profs[:, tt, j]))[0]
                overlying_locs2 = np.where(np.logical_and(strat_profs2[:, tt, j + 1], ~conv_profs[:, tt, j]))[0]
                overlying_num = np.array([len(overlying_locs1), len(overlying_locs2)], dtype=int)
                over_diff = abs(overlying_num[1] - overlying_num[0])
                Iover_min = np.argmin(overlying_num)
                Iover_max = np.argmax(overlying_num)
                over_unique_lo = _setxor(overlying_locs1, overlying_locs2)

                if overlying_num[Iover_min] > cld_2_assign[I_max]:
                    if cld_2_assign[I_max] > 0:
                        rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_max])
                        inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs[0:cld_2_assign[I_min]]]
                        locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign = np.zeros(2)
                elif overlying_num[Iover_min] > cld_2_assign[I_min]:
                    if cld_2_assign[I_min] > 0:
                        rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_min])
                        inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                        locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign[I_min] = 0
                    cld_2_assign[I_max] -= overlying_num[Iover_min]

                    if cld_2_assign[I_max] > 0 and over_diff > 0:
                        rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
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
                        rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
                        inds = over_unique_lo[rand_locs[0:cld_2_assign[I_min]]]
                        locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        inds = over_unique_lo[rand_locs]
                        locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                        cld_2_assign = np.zeros(2)
                    else:
                        if cld_2_assign[I_min] > 0:
                            rand_locs = _randperm(over_diff, size=cld_2_assign[I_min])
                            inds = over_unique_lo[rand_locs]
                            locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                        cld_2_assign[I_min] = 0
                        locals()['strat_profs%d' % (I_max + 1)][over_unique_lo, tt, j] = True
                        cld_2_assign[I_max] -= over_diff
                else:
                    inds = locals()["overlying_locs%d" % (Iover_max + 1)]
                    locals()['strat_profs%d' % (I_min + 1)][inds, tt, j] = True
                    locals()['strat_profs%d' % (I_max + 1)][inds, tt, j] = True
                    cld_2_assign -= overlying_num[Iover_max]

            if cld_2_assign[I_max] > 0:
                sprof = locals()["strat_profs%d" % (I_max + 1)]
                free_locs_max = np.where(np.logical_and(~sprof[:, tt, j], ~conv_profs[:, tt, j]))[0]
                free_num = len(free_locs_max)
                rand_locs = _randperm(free_num, size=int(cld_2_assign[I_max]))
                locals()["strat_profs%d" % (I_max + 1)][free_locs_max[rand_locs], tt, j] = True
                if cld_2_assign[I_min] > 0.:
                    locals()["strat_profs%d"
                             % (I_min + 1)][free_locs_max[rand_locs[0:cld_2_assign[I_min]]], tt, j] = True

    model.ds['strat_frac_subcolumns_cl'] = xr.DataArray(strat_profs1,
                                                        dims=(subcolumn_dims[0],
                                                              subcolumn_dims[1], subcolumn_dims[2]))
    model.ds['strat_frac_subcolumns_ci'] = xr.DataArray(strat_profs2,
                                                        dims=(subcolumn_dims[0],
                                                              subcolumn_dims[1], subcolumn_dims[2]))
    model.ds['strat_frac_subcolumns_cl'].attrs["long_name"] = \
        "Liquid cloud particles present? [stratiform]"
    model.ds['strat_frac_subcolumns_cl'].attrs["units"] = "0 = no, 1 = yes"
    model.ds['strat_frac_subcolumns_ci'].attrs["long_name"] = \
        "Liquid cloud particles present? [stratiform]"
    model.ds['strat_frac_subcolumns_ci'].attrs["units"] = "0 = no, 1 = yes"
    return model


def set_precip_sub_col_frac(model, convective=True, N_columns=None):
    """
    Sets the hydrometeor fraction due to precipitation in each subcolumn. This
    module works for both stratiform and convective precipitation.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of stratiform fraction for.
    convective: bool
        Set to True to generate subcolumns for convective precipitation.
        Set to False to generate subcolumns for stratiform precipitation.
    N_columns: int or None
        Use this to set the number of subcolumns in the model. This can only
        be set once. After the number of subcolumns is set, use None to make
        EMC2 automatically detect the number of subcolumns.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if model.num_subcolumns == 0 and N_columns is None:
        raise RuntimeError("The number of subcolumns must be specified in the model!")

    if model.num_subcolumns != N_columns and model.num_subcolumns != 0 and N_columns is not None:
        raise ValueError("The number of subcolumns has already been specified (%d) and != %d" %
                         (model.num_subcolumns, N_columns))

    if model.num_subcolumns == 0:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    if convective:
        data_frac1 = model.ds[model.conv_frac_names['pl']]
        data_frac2 = model.ds[model.conv_frac_names['pi']]
        out_prof_name1 = 'conv_frac_subcolumns_pl'
        out_prof_name2 = 'conv_frac_subcolumns_pi'
        out_prof_long_name1 = 'Liquid precipitation present? [convective]'
        out_prof_long_name2 = 'Ice precipitation present? [convective]'
        in_prof_cloud_name_liq = 'conv_frac_subcolumns_cl'
        in_prof_cloud_name_ice = 'conv_frac_subcolumns_ci'
    else:
        data_frac1 = model.ds[model.strat_frac_names['pl']]
        data_frac2 = model.ds[model.strat_frac_names['pi']]
        out_prof_name1 = 'strat_frac_subcolumns_pl'
        out_prof_name2 = 'strat_frac_subcolumns_pi'
        in_prof_cloud_name_liq = 'strat_frac_subcolumns_cl'
        in_prof_cloud_name_ice = 'strat_frac_subcolumns_ci'
        out_prof_long_name1 = 'Liquid precipitation present? [stratiform]'
        out_prof_long_name2 = 'Ice precipitation present? [stratiform]'

    if in_prof_cloud_name_liq not in model.ds.variables.keys():
        raise KeyError("%s is not a variable in the model object. Please ensure that you have" +
                       "generated the cloud particle subcolumns before running this routine." %
                       in_prof_cloud_name_liq)

    if in_prof_cloud_name_ice not in model.ds.variables.keys():
        raise KeyError("%s is not a variable in the model object. Please ensure that you have" +
                       "generated the cloud particle subcolumns before running this routine." %
                       in_prof_cloud_name_ice)

    data_frac1 = np.round(data_frac1 * model.num_subcolumns)
    data_frac2 = np.round(data_frac2 * model.num_subcolumns)
    p_strat_profs = np.zeros((model.num_subcolumns, data_frac1.shape[0], data_frac1.shape[1], 2), dtype=bool)
    strat_profs = np.logical_or(model.ds[in_prof_cloud_name_ice].values,
                                model.ds[in_prof_cloud_name_liq].values)
    subcolumn_dims = model.ds[in_prof_cloud_name_ice].dims
    is_cloud = np.logical_or(data_frac1 > 0, data_frac2 > 0)
    is_cloud_one_above = np.roll(is_cloud, -1, axis=1)
    is_cloud_one_above[:, -1] = False
    overlapping_cloud = np.logical_and(is_cloud, is_cloud_one_above)
    precip_exist = np.stack([data_frac1 > 0, data_frac2 > 0])
    PF_val = np.max(np.stack([data_frac1, data_frac2]), axis=0)
    cond = [strat_profs, ~strat_profs]
    for tt in range(data_frac1.shape[0]):
        for j in range(data_frac1.shape[1] - 2, -1, -1):
            if overlapping_cloud[tt, j]:
                overlying_locs = np.where(np.any(p_strat_profs[:, tt, j + 1, :], axis=1))[0]
                overlying_num = len(overlying_locs)
                if overlying_num > PF_val[tt, j]:
                    rand_locs = _randperm(overlying_num, PF_val[tt, j])
                    for i in range(2):
                        if precip_exist[i, tt, j]:
                            p_strat_profs[overlying_locs[rand_locs], tt, j, i] = True
                    PF_val[tt, j] = 0.
                else:
                    for i in range(2):
                        if precip_exist[i, tt, j]:
                            p_strat_profs[overlying_locs, tt, j, i] = True
                    PF_val[tt, j] -= overlying_num

            for ii in range(2):
                if PF_val[tt, j] > 0:
                    free_locs = np.where(np.logical_and(
                        ~np.all(p_strat_profs[:, tt, j, :], axis=1), cond[ii][:, tt, j]))[0]
                    free_num = len(free_locs)
                    if free_num > 0:
                        if free_num > PF_val[tt, j]:
                            rand_locs = _randperm(free_num, PF_val[tt, j])
                            for i in range(2):
                                if precip_exist[i, tt, j]:
                                    p_strat_profs[free_locs[rand_locs], tt, j, i] = True
                            PF_val[tt, j] = 0
                        else:
                            for i in range(2):
                                if precip_exist[i, tt, j]:
                                    p_strat_profs[free_locs, tt, j, i] = True
                            PF_val[tt, j] -= free_num

    model.ds[out_prof_name1] = xr.DataArray(p_strat_profs[:, :, :, 0],
                                            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    model.ds[out_prof_name2] = xr.DataArray(p_strat_profs[:, :, :, 1],
                                            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    model.ds[out_prof_name1].attrs["long_name"] = out_prof_long_name1
    model.ds[out_prof_name1].attrs["units"] = "0 = no, 1 = yes"
    model.ds[out_prof_name2].attrs["long_name"] = out_prof_long_name2
    model.ds[out_prof_name2].attrs["units"] = "0 = no, 1 = yes"
    return model


def set_q_n(model, hyd_type, is_conv=True, qc_flag=True, inv_rel_var=1):
    """

    This function distributes the mixing ratio and number concentration into the subcolumns.
    For :math:`q_c`, the horizontal distribution follows Equation 8 of Morrison and Gettelman (2008).

    Parameters
    ----------
    model: :func:`emc2.core.Model`
        The model to calculate the mixing ratio in each subcolumn for.
    hyd_type: str
        The hydrometeor type.
    is_conv: bool
        Set to True to calculate the mixing ratio assuming convective clouds.
    qc_flag: bool
        Set to True to horizontally distribute the mixing ratio according to
        Morrison and Gettleman (2008)
    inv_rel_var: float
        The inverse of the relative variance for the p.d.f. in Morrison and Gettleman (2008)

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with mixing ratio calculated in each subcolumn.

    References
    ----------
    Morrison, H. and A. Gettelman, 2008: A New Two-Moment Bulk Stratiform Cloud Microphysics Scheme
    in the Community Atmosphere Model, Version 3 (CAM3). Part I: Description and Numerical Tests.
    J. Climate, 21, 3642–3659, https://doi.org/10.1175/2008JCLI2105.1
    """
    np.seterr(divide='ignore', invalid='ignore')
    if model.num_subcolumns == 0:
        raise RuntimeError("The number of subcolumns must be specified in the model!")

    if not is_conv:
        hyd_profs = model.ds[model.strat_frac_names[hyd_type]].astype('float64').values
        N_profs = model.ds[model.N_field[hyd_type]].astype('float64').values
        N_profs = N_profs / hyd_profs
        sub_hyd_profs = model.ds['strat_frac_subcolumns_%s' % hyd_type].values
        N_profs = np.tile(N_profs, (model.num_subcolumns, 1, 1))
        N_profs = np.where(sub_hyd_profs, N_profs, 0)
        q_array = model.ds[model.q_names_stratiform[hyd_type]].astype('float64').values
        q_name = "strat_q_subcolumns_%s" % hyd_type
        n_name = "strat_n_subcolumns_%s" % hyd_type
    else:
        hyd_profs = model.ds[model.conv_frac_names[hyd_type]].astype('float64').values
        sub_hyd_profs = model.ds['conv_frac_subcolumns_%s' % hyd_type]
        q_array = model.ds[model.q_names_convective[hyd_type]].astype('float64').values
        q_name = "conv_q_subcolumns_%s" % hyd_type

    q_ic_mean = q_array / hyd_profs
    q_ic_mean = np.where(np.isnan(q_ic_mean), 0, q_ic_mean)
    if qc_flag:
        tot_hyd_in_sub = sub_hyd_profs.sum(axis=0)
        q_profs = np.zeros_like(sub_hyd_profs, dtype=float)

        for j in range(hyd_profs.shape[1]):
            for tt in range(hyd_profs.shape[0] - 1, -1, -1):
                hyd_in_sub_loc = np.where(sub_hyd_profs[:, tt, j])[0]
                if tot_hyd_in_sub[tt, j] == 1:
                    q_profs[hyd_in_sub_loc, tt, j] = q_ic_mean[tt, j]
                elif tot_hyd_in_sub[tt, j] > 1:
                    alpha = inv_rel_var / q_ic_mean[tt, j]
                    a = inv_rel_var
                    b = 1 / alpha
                    randlocs = np.random.permutation(tot_hyd_in_sub[tt, j])
                    rand_gamma_vals = np.random.gamma(a, b, tot_hyd_in_sub[tt, j] - 1)
                    valid_vals = False
                    counter_4_valid = 0
                    while not valid_vals:
                        counter_4_valid += 1
                        valid_vals = (q_ic_mean[tt, j] * tot_hyd_in_sub[tt, j] -
                                      rand_gamma_vals[0:-counter_4_valid].sum()) > 0
                    q_profs[hyd_in_sub_loc[
                        randlocs[:-(counter_4_valid + 1)]], tt, j] = rand_gamma_vals[:-counter_4_valid]
                    q_profs[hyd_in_sub_loc[randlocs[-counter_4_valid:]], tt, j] = (
                        q_ic_mean[tt, j] * tot_hyd_in_sub[tt, j] - np.sum(rand_gamma_vals[:-counter_4_valid])) / \
                        (1 + counter_4_valid)

    else:
        q_profs = q_array / hyd_profs
        q_profs = np.tile(q_profs, (model.num_subcolumns, 1, 1))
        q_profs = np.where(sub_hyd_profs, q_profs, 0)

    q_profs = np.where(np.isnan(q_profs), 0, q_profs)
    model.ds[q_name] = xr.DataArray(q_profs, dims=('subcolumn', model.time_dim, model.height_dim))
    model.ds[q_name].attrs["long_name"] = "Q in subcolumns"
    model.ds[q_name].attrs["units"] = "kg/kg"
    if not is_conv:
        N_profs = np.where(np.isnan(N_profs), 0, N_profs)
        model.ds[n_name] = xr.DataArray(N_profs, dims=('subcolumn', model.time_dim, model.height_dim))
        model.ds[n_name].attrs["long_name"] = "N in subcolumns"
        model.ds[n_name].attrs["units"] = "cm-3"

    return model


def _randperm(x, size=None):
    if size is None:
        size = len(x)
    return np.random.permutation(x)[0:int(size)].astype(int)


def _setxor(x, y):
    first_set = np.setdiff1d(x, y)
    second_set = np.setdiff1d(y, x)
    return np.concatenate([first_set, second_set])
