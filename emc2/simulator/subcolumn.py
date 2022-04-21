import numpy as np
import xarray as xr
import dask.bag as db
from time import time


def set_convective_sub_col_frac(model, hyd_type, N_columns=None, use_rad_logic=True):
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
    use_rad_logic: bool
        When True using the cloud fraction utilized in a model radiative scheme. Otherwise,
        using the microphysics scheme (note that these schemes do not necessarily
        use exactly the same cloud fraction logic).

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the convective fraction in each subcolumn added.
    """
    np.seterr(divide='ignore', invalid='ignore')
    if N_columns is None and model.num_subcolumns == 0:
        N_columns = model.ds.dims['subcolumn']

    if model.num_subcolumns != N_columns and model.num_subcolumns != 0 and N_columns is not None:
        raise ValueError("The number of subcolumns has already been specified (%d) and != %d" %
                         (model.num_subcolumns, N_columns))

    if model.num_subcolumns == 0:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    if use_rad_logic:
        method_str = "Radiation logic"
        data_frac = np.round(model.ds[model.conv_frac_names_for_rad[hyd_type]].values * model.num_subcolumns)
        data_frac = np.where(model.ds[model.q_names_convective[hyd_type]].values > 0, data_frac, 0)
    else:
        method_str = "Microphysics logic"
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
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["Processing method"] = method_str

    return model


def set_stratiform_sub_col_frac(model, N_columns=None, use_rad_logic=True, parallel=True, chunk=None):
    """
    Sets the hydrometeor fraction due to stratiform cloud particles in each subcolumn.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of stratiform fraction for.
    use_rad_logic: bool
        When True using the cloud fraction utilized in a model radiative scheme. Otherwise,
        using the microphysics scheme (note that these schemes do not necessarily
        use exactly the same cloud fraction logic).
    N_columns: int or None
        The number of subcolumns to generate. Specifying this will set the number
        of subcolumns in the model parameter when the first subcolumns are generated.
        Therefore, after those are generated this must either be
        equal to None or the number of subcolumns in the model. Setting this to None will
        use the number of subcolumns in the model parameter.
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    chunk: int or None
        The number of entries to process in one parallel loop. None will send all of
        the entries to the Dask worker queue at once. Sometimes, Dask will freeze if
        too many tasks are sent at once due to memory issues, so adjusting this number
        might be needed if that happens.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """
    t0 = time()

    if model.process_conv:
        if "conv_frac_subcolumns_cl" not in model.ds.variables.keys():
            raise KeyError("You have to generate the convective fraction in each subcolumn " +
                           "before the stratiform fraction in each subcolumn is generated.")

        if "conv_frac_subcolumns_ci" not in model.ds.variables.keys():
            raise KeyError("You have to generate the convective fraction in each subcolumn " +
                           "before the stratiform fraction in each subcolumn is generated.")
        np.seterr(divide='ignore', invalid='ignore')
        conv_profs1 = model.ds["conv_frac_subcolumns_cl"]
        conv_profs2 = model.ds["conv_frac_subcolumns_ci"]
        conv_profs = np.logical_or(conv_profs1.values, conv_profs2.values)
    else:
        if model.num_subcolumns == 0:
            model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')
        conv_profs = np.zeros((N_columns, *model.ds[model.strat_frac_names["cl"]].shape),
                              dtype=bool)
    N_columns = len(model.ds["subcolumn"])
    subcolumn_dims = ("subcolumn", *model.ds[model.strat_frac_names_for_rad["cl"]].dims)
    if use_rad_logic:
        method_str = "Radiation logic"
        data_frac1 = model.ds[model.strat_frac_names_for_rad["cl"]]
        data_frac1 = data_frac1.where(model.ds[model.q_names_stratiform["cl"]] > 0, 0)
        data_frac2 = model.ds[model.strat_frac_names_for_rad["ci"]]
        data_frac2 = data_frac2.where(model.ds[model.q_names_stratiform["ci"]] > 0, 0)
    else:
        method_str = "Microphysics logic"
        data_frac1 = model.ds[model.strat_frac_names["cl"]]
        data_frac2 = model.ds[model.strat_frac_names["ci"]]
    data_frac1 = np.round(data_frac1.values * N_columns).astype(int)
    data_frac2 = np.round(data_frac2.values * N_columns).astype(int)
    full_overcast_cl_ci = 0

    is_cloud = np.logical_or(data_frac1 > 0, data_frac2 > 0)
    is_cloud_one_above = np.roll(is_cloud, -1, axis=1)
    overlapping_cloud = np.logical_and(is_cloud, is_cloud_one_above)

    cld_2_assigns = np.stack([data_frac1, data_frac2], axis=0)
    I_min = np.argmin(cld_2_assigns, axis=0)
    I_max = np.argmax(cld_2_assigns, axis=0)

    _allocate_strat_sub_cols = lambda x: _allocate_strat_sub_col(
        x, cld_2_assigns, I_min, I_max, conv_profs,
        full_overcast_cl_ci, data_frac1, data_frac2, N_columns, overlapping_cloud)

    t_dim = data_frac1.shape[0]
    if parallel:
        print("Now performing parallel stratiform hydrometeor allocation in subcolumns")
        if chunk is None:
            tt_bag = db.from_sequence(np.arange(0, t_dim, 1))
            my_tuple = tt_bag.map(_allocate_strat_sub_cols).compute()
        else:
            my_tuple = []
            j = 0
            while j < t_dim:
                if j + chunk >= t_dim:
                    ind_max = t_dim
                else:
                    ind_max = j + chunk
                print("Stage 1 of 2: Processing columns %d-%d out of %d" % (j, ind_max, t_dim))
                tt_bag = db.from_sequence(np.arange(j, ind_max, 1))
                my_tuple += tt_bag.map(_allocate_strat_sub_cols).compute()
                j += chunk
    else:
        my_tuple = [x for x in map(_allocate_strat_sub_cols, np.arange(0, t_dim, 1))]

    full_overcast_cl_ci += np.sum([x[0] for x in my_tuple])
    strat_profs1 = np.stack([x[1] for x in my_tuple], axis=1)
    strat_profs2 = np.stack([x[2] for x in my_tuple], axis=1)

    print("Fully overcast cl & ci in %s voxels" % full_overcast_cl_ci)
    model.ds['strat_frac_subcolumns_cl'] = xr.DataArray(strat_profs1,
                                                        dims=(subcolumn_dims[0],
                                                              subcolumn_dims[1], subcolumn_dims[2]))
    model.ds['strat_frac_subcolumns_ci'] = xr.DataArray(strat_profs2,
                                                        dims=(subcolumn_dims[0],
                                                              subcolumn_dims[1], subcolumn_dims[2]))
    model.ds['strat_frac_subcolumns_cl'].attrs["long_name"] = \
        "Liquid cloud particles present? [stratiform]"
    model.ds['strat_frac_subcolumns_cl'].attrs["units"] = "0 = no, 1 = yes"
    model.ds['strat_frac_subcolumns_cl'].attrs["Processing method"] = method_str
    model.ds['strat_frac_subcolumns_ci'].attrs["long_name"] = \
        "Liquid cloud particles present? [stratiform]"
    model.ds['strat_frac_subcolumns_ci'].attrs["units"] = "0 = no, 1 = yes"
    model.ds['strat_frac_subcolumns_ci'].attrs["Processing method"] = method_str

    print("Done! total processing time = %.2fs" % (time() - t0))

    return model


def set_precip_sub_col_frac(model, is_conv, N_columns=None, use_rad_logic=True,
                            parallel=True, chunk=None):
    """
    Sets the hydrometeor fraction due to precipitation in each subcolumn. This
    module works for both stratiform and convective precipitation.

    Parameters
    ----------
    model: :py:func: `emc2.core.Model`
        The model we are generating the subcolumns of stratiform fraction for.
    is_conv: bool
        Set to True to generate subcolumns for convective precipitation.
        Set to False to generate subcolumns for stratiform precipitation.
    N_columns: int or None
        Use this to set the number of subcolumns in the model. This can only
        be set once. After the number of subcolumns is set, use None to make
        EMC2 automatically detect the number of subcolumns.
    use_rad_logic: bool
        When True using the cloud fraction utilized in a model radiative scheme. Otherwise,
        using the microphysics scheme (note that these schemes do not necessarily
        use exactly the same cloud fraction logic).
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    chunk: int or None
        The number of entries to process in one parallel loop. None will send all of
        the entries to the Dask worker queue at once. Sometimes, Dask will freeze if
        too many tasks are sent at once due to memory issues, so adjusting this number
        might be needed if that happens.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """
    t0 = time()

    np.seterr(divide='ignore', invalid='ignore')
    if model.num_subcolumns == 0 and N_columns is None:
        raise RuntimeError("The number of subcolumns must be specified in the model!")

    if np.logical_and(model.num_subcolumns != 0, N_columns is not None):
        raise ValueError("The number of subcolumns has already been specified (%d) and != %d" %
                         (model.num_subcolumns, N_columns))

    if N_columns is None:
        N_columns = model.num_subcolumns

    if model.num_subcolumns == 0:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    if is_conv:
        precip_type = 'conv'
        precip_type_full = 'convective'
        if use_rad_logic:
            method_str = "Radiation logic"
            data_frac1 = model.ds[model.conv_frac_names_for_rad['pl']]
            data_frac1 = data_frac1.where(model.ds[model.q_names_convective["pl"]] > 0, 0)
            data_frac2 = model.ds[model.conv_frac_names_for_rad['pi']]
            data_frac2 = data_frac2.where(model.ds[model.q_names_convective["pi"]] > 0, 0)
        else:
            method_str = "Microphysics logic"
            data_frac1 = model.ds[model.conv_frac_names['pl']]
            data_frac2 = model.ds[model.conv_frac_names['pi']]
    else:
        precip_type = 'strat'
        precip_type_full = 'stratiform'
        if use_rad_logic:
            method_str = "Radiation logic"
            data_frac1 = model.ds[model.strat_frac_names_for_rad['pl']]
            data_frac1.where(model.ds[model.q_names_stratiform["pl"]] > 0, 0)
            data_frac2 = model.ds[model.strat_frac_names_for_rad['pi']]
            data_frac2.where(model.ds[model.q_names_stratiform["pi"]] > 0, 0)
        else:
            method_str = "Microphysics logic"
            data_frac1 = model.ds[model.strat_frac_names['pl']]
            data_frac2 = model.ds[model.strat_frac_names['pi']]
    out_prof_long_name1 = 'Liquid precipitation present? [%s]' % precip_type_full
    out_prof_long_name2 = 'Ice precipitation present? [%s]' % precip_type_full
    out_prof_name1 = precip_type + '_frac_subcolumns_pl'
    out_prof_name2 = precip_type + '_frac_subcolumns_pi'
    in_prof_cloud_name_liq = precip_type + '_frac_subcolumns_cl'
    in_prof_cloud_name_ice = precip_type + '_frac_subcolumns_ci'

    full_overcast_pl_pi = 0
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

    _allocate_precip_sub_cols = lambda x: _allocate_precip_sub_col(
        x, cond, N_columns, data_frac1, data_frac2, PF_val,
        precip_exist, full_overcast_pl_pi, overlapping_cloud)

    t_dim = data_frac1.shape[0]
    if parallel:
        print("Now performing parallel %s precipitation allocation in subcolumns" % precip_type)
        if chunk is None:
            tt_bag = db.from_sequence(np.arange(0, t_dim, 1))
            my_tuple = tt_bag.map(_allocate_precip_sub_cols).compute()
        else:
            my_tuple = []
            j = 0
            while j < t_dim:
                if j + chunk >= t_dim:
                    ind_max = t_dim
                else:
                    ind_max = j + chunk
                print("Stage 1 of 2: Processing columns %d-%d out of %d" % (j, ind_max, t_dim))
                tt_bag = db.from_sequence(np.arange(j, ind_max, 1))
                my_tuple += tt_bag.map(_allocate_precip_sub_cols).compute()
                j += chunk
    else:
        my_tuple = [x for x in map(_allocate_precip_sub_cols, np.arange(0, t_dim, 1))]

    full_overcast_pl_pi += np.sum([x[0] for x in my_tuple])
    p_strat_profs = np.stack([x[1] for x in my_tuple], axis=1)

    print("Fully overcast pl & pi in %s voxels" % full_overcast_pl_pi)
    model.ds[out_prof_name1] = xr.DataArray(p_strat_profs[:, :, :, 0],
                                            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    model.ds[out_prof_name2] = xr.DataArray(p_strat_profs[:, :, :, 1],
                                            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    model.ds[out_prof_name1].attrs["long_name"] = out_prof_long_name1
    model.ds[out_prof_name1].attrs["units"] = "0 = no, 1 = yes"
    model.ds[out_prof_name1].attrs["Processing method"] = method_str
    model.ds[out_prof_name2].attrs["long_name"] = out_prof_long_name2
    model.ds[out_prof_name2].attrs["units"] = "0 = no, 1 = yes"
    model.ds[out_prof_name2].attrs["Processing method"] = method_str

    print("Done! total processing time = %.2fs" % (time() - t0))

    return model


def set_q_n(model, hyd_type, is_conv=True, qc_flag=False, inv_rel_var=1, use_rad_logic=True,
            parallel=True, chunk=None):
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
        Set to True to horizontally distribute the mixing ratio (allowing sub-grid variability)
        according to Morrison and Gettleman (2008). qc_flag is set to False in case use_rad_logic
        and/or is_conv are True (both cases do not follow the Morrison scheme).
    inv_rel_var: float
        The inverse of the relative subgrid qc PDF variance in Morrison and Gettleman (2008)
    use_rad_logic: bool
        When True using the cloud fraction utilized in a model radiative scheme and also implementing
        uniformly distributed qc (setting qc_flag to False) to maintain radiation scheme logic.
        Otherwise, using the microphysics scheme (note that these schemes do not necessarily
        use exactly the same cloud fraction logic).
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    chunk: int or None
        The number of entries to process in one parallel loop. None will send all of
        the entries to the Dask worker queue at once. Sometimes, Dask will freeze if
        too many tasks are sent at once due to memory issues, so adjusting this number
        might be needed if that happens.

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

    if np.logical_or(use_rad_logic, is_conv):
        qc_flag = False

    if not is_conv:
        frac_fieldname = 'strat_frac_subcolumns_%s' % hyd_type
        if use_rad_logic:
            method_str = "Radiation logic"
            data_frac = model.ds[model.strat_frac_names_for_rad[hyd_type]].astype('float64').values
            data_frac = np.where(model.ds[model.q_names_stratiform[hyd_type]].values > 0, data_frac, 0)
        else:
            method_str = "Microphysics logic"
            data_frac = model.ds[model.strat_frac_names[hyd_type]].astype('float64').values
        N_profs = model.ds[model.N_field[hyd_type]].astype('float64').values
        N_profs = N_profs / data_frac
        sub_data_frac = model.ds[frac_fieldname].values
        N_profs = np.tile(N_profs, (model.num_subcolumns, 1, 1))
        N_profs = np.where(sub_data_frac, N_profs, 0)
        q_array = model.ds[model.q_names_stratiform[hyd_type]].astype('float64').values
        q_name = "strat_q_subcolumns_%s" % hyd_type
        n_name = "strat_n_subcolumns_%s" % hyd_type
    else:
        frac_fieldname = 'conv_frac_subcolumns_%s' % hyd_type
        if use_rad_logic:
            method_str = "Radiation logic"
            data_frac = model.ds[model.conv_frac_names_for_rad[hyd_type]].astype('float64').values
            data_frac = np.where(model.ds[model.q_names_convective[hyd_type]].values > 0, data_frac, 0)
        else:
            method_str = "Microphysics logic"
            data_frac = model.ds[model.conv_frac_names[hyd_type]].astype('float64').values
        sub_data_frac = model.ds[frac_fieldname]
        q_array = model.ds[model.q_names_convective[hyd_type]].astype('float64').values
        q_name = "conv_q_subcolumns_%s" % hyd_type

    if qc_flag:
        q_ic_mean = np.where(q_array > 0, q_array / data_frac, 0)
        q_ic_mean = np.where(np.isnan(q_ic_mean), 0, q_ic_mean)
        tot_hyd_in_sub = sub_data_frac.sum(axis=0)

        _distribute_cl_q_n_sub_cols = lambda x: _distribute_cl_q_n(
            x, sub_data_frac, inv_rel_var, model.num_subcolumns, tot_hyd_in_sub, q_ic_mean)

        t_dim = data_frac.shape[0]
        if parallel:
            print("Now distributing q in subcolumns in parallel")
            if chunk is None:
                tt_bag = db.from_sequence(np.arange(t_dim - 1, -1, -1))
                my_tuple = tt_bag.map(_distribute_cl_q_n_sub_cols).compute()
            else:
                my_tuple = []
                j = t_dim - 1
                while j >= 0:
                    if j - chunk < 0:
                        ind_min = 0
                    else:
                        ind_min = j - chunk
                    print("Stage 1 of 2: Processing columns %d-%d out of %d" % (j, t_dim - ind_min, t_dim))
                    tt_bag = db.from_sequence(np.arange(j, ind_min, -1))
                    my_tuple += tt_bag.map(_distribute_cl_q_n_sub_cols).compute()
                    j -= chunk
        else:
            my_tuple = [x for x in map(_distribute_cl_q_n_sub_cols, np.arange(t_dim - 1, -1, -1))]

        q_profs = np.stack([x for x in my_tuple], axis=1)

    else:
        q_profs = np.where(q_array > 0, q_array / data_frac, 0)
        q_profs = np.tile(q_profs, (model.num_subcolumns, 1, 1))
        q_profs = np.where(sub_data_frac, q_profs, 0)

    q_profs = np.where(np.isnan(q_profs), 0, q_profs)
    model.ds[q_name] = xr.DataArray(q_profs, dims=model.ds[frac_fieldname].dims)
    model.ds[q_name].attrs["long_name"] = "q in subcolumns"
    model.ds[q_name].attrs["units"] = r"$kg\ kg^{-1}$"
    model.ds[q_name].attrs["Processing method"] = method_str
    if not is_conv:
        N_profs = np.where(np.isnan(N_profs), 0, N_profs)
        model.ds[n_name] = xr.DataArray(N_profs, dims=model.ds[frac_fieldname].dims)
        model.ds[n_name].attrs["long_name"] = "N in subcolumns"
        model.ds[n_name].attrs["units"] = r"$cm^{-3}$"
        model.ds[n_name].attrs["Processing method"] = method_str

    return model


def _randperm(x, size=None):
    if size is None:
        size = len(x)
    return np.random.permutation(x)[0:int(size)].astype(int)


def _setxor(x, y):
    first_set = np.setdiff1d(x, y)
    second_set = np.setdiff1d(y, x)
    return np.concatenate([first_set, second_set])


def _allocate_strat_sub_col(tt, cld_2_assigns, I_min, I_max, conv_profs,
                            full_overcast_cl_ci, data_frac1, data_frac2, N_columns, overlapping_cloud):
    strat_profs1 = np.zeros((N_columns, data_frac1.shape[1]), dtype=bool)
    strat_profs2 = np.zeros_like(strat_profs1, dtype=bool)
    for j in range(data_frac1.shape[1] - 2, -1, -1):
        cld_2_assign = np.array([data_frac1[tt, j], data_frac2[tt, j]])
        I_min = np.argmin(cld_2_assign)
        I_max = np.argmax(cld_2_assign)
        if cld_2_assign[I_max] == 0:
            continue
        if cld_2_assign[I_min] == N_columns:
            strat_profs1[:, j] = True
            strat_profs2[:, j] = True
            full_overcast_cl_ci += 1
            continue
        if overlapping_cloud[tt, j]:
            overlying_locs1 = np.where(np.logical_and(strat_profs1[:, j + 1], ~conv_profs[:, tt, j]))[0]
            overlying_locs2 = np.where(np.logical_and(strat_profs2[:, j + 1], ~conv_profs[:, tt, j]))[0]
            overlying_num = np.array([len(overlying_locs1), len(overlying_locs2)], dtype=int)
            over_diff = abs(overlying_num[1] - overlying_num[0])
            Iover_min = np.argmin(overlying_num)
            Iover_max = np.argmax(overlying_num)
            over_unique_lo = _setxor(overlying_locs1, overlying_locs2)

            if overlying_num[Iover_min] > cld_2_assign[I_max]:
                if cld_2_assign[I_max] > 0:
                    rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_max])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs[0:cld_2_assign[I_min]]]
                    locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                cld_2_assign = np.zeros(2)
            elif overlying_num[Iover_min] > cld_2_assign[I_min]:
                if cld_2_assign[I_min] > 0:
                    rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_min])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                    locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                cld_2_assign[I_min] = 0
                cld_2_assign[I_max] -= overlying_num[Iover_min]

                if cld_2_assign[I_max] > 0 and over_diff > 0:
                    rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
                    inds = over_unique_lo[rand_locs]
                    locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                    cld_2_assign[I_max] = 0.
                else:
                    locals()['strat_profs%d' % (I_max + 1)][over_unique_lo, j] = True
                    cld_2_assign[I_max] -= over_diff
            elif overlying_num[Iover_max] > cld_2_assign[I_min]:
                inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                cld_2_assign -= overlying_num[Iover_min]

                if over_diff > cld_2_assign[I_max]:
                    rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
                    inds = over_unique_lo[rand_locs[0:cld_2_assign[I_min]]]
                    locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                    inds = over_unique_lo[rand_locs]
                    locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                    cld_2_assign = np.zeros(2)
                else:
                    if cld_2_assign[I_min] > 0:
                        rand_locs = _randperm(over_diff, size=cld_2_assign[I_min])
                        inds = over_unique_lo[rand_locs]
                        locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                    cld_2_assign[I_min] = 0
                    locals()['strat_profs%d' % (I_max + 1)][over_unique_lo, j] = True
                    cld_2_assign[I_max] -= over_diff
            else:
                inds = locals()["overlying_locs%d" % (Iover_max + 1)]
                locals()['strat_profs%d' % (I_min + 1)][inds, j] = True
                locals()['strat_profs%d' % (I_max + 1)][inds, j] = True
                cld_2_assign -= overlying_num[Iover_max]

        if cld_2_assign[I_max] > 0:
            sprof = locals()["strat_profs%d" % (I_max + 1)]
            free_locs_max = np.where(np.logical_and(~sprof[:, j], ~conv_profs[:, tt, j]))[0]
            free_num = len(free_locs_max)
            rand_locs = _randperm(free_num, size=int(cld_2_assign[I_max]))
            locals()["strat_profs%d" % (I_max + 1)][free_locs_max[rand_locs], j] = True
            if cld_2_assign[I_min] > 0.:
                locals()["strat_profs%d"
                         % (I_min + 1)][free_locs_max[rand_locs[0:cld_2_assign[I_min]]], j] = True

    return full_overcast_cl_ci, strat_profs1, strat_profs2


def _allocate_precip_sub_col(tt, cond, N_columns, data_frac1, data_frac2, PF_val,
                             precip_exist, full_overcast_pl_pi, overlapping_cloud):
    p_strat_profs = np.zeros((N_columns, data_frac1.shape[1], 2), dtype=bool)
    for j in range(data_frac1.shape[1] - 2, -1, -1):
        if np.logical_and(data_frac1[tt, j] == N_columns,
                          data_frac2[tt, j] == N_columns):
            p_strat_profs[:, j, :] = True
            full_overcast_pl_pi += 1
            continue
        if overlapping_cloud[tt, j]:
            overlying_locs = np.where(np.any(p_strat_profs[:, j + 1, :], axis=1))[0]
            overlying_num = len(overlying_locs)
            if overlying_num > PF_val[tt, j]:
                rand_locs = _randperm(overlying_num, PF_val[tt, j])
                for i in range(2):
                    if precip_exist[i, tt, j]:
                        p_strat_profs[overlying_locs[rand_locs], j, i] = True
                PF_val[tt, j] = 0.
            else:
                for i in range(2):
                    if precip_exist[i, tt, j]:
                        p_strat_profs[overlying_locs, j, i] = True
                PF_val[tt, j] -= overlying_num

        for ii in range(2):
            if PF_val[tt, j] > 0:
                free_locs = np.where(np.logical_and(
                    ~np.all(p_strat_profs[:, j, :], axis=1), cond[ii][:, tt, j]))[0]
                free_num = len(free_locs)
                if free_num > 0:
                    if free_num > PF_val[tt, j]:
                        rand_locs = _randperm(free_num, PF_val[tt, j])
                        for i in range(2):
                            if precip_exist[i, tt, j]:
                                p_strat_profs[free_locs[rand_locs], j, i] = True
                        PF_val[tt, j] = 0
                    else:
                        for i in range(2):
                            if precip_exist[i, tt, j]:
                                p_strat_profs[free_locs, j, i] = True
                        PF_val[tt, j] -= free_num

    return full_overcast_pl_pi, p_strat_profs


def _distribute_cl_q_n(tt, sub_data_frac, inv_rel_var, N_columns, tot_hyd_in_sub, q_ic_mean):
    q_profs = np.zeros((N_columns, q_ic_mean.shape[1]), dtype=float)
    for j in range(q_ic_mean.shape[1]):
        hyd_in_sub_loc = np.where(sub_data_frac[:, tt, j])[0]
        if tot_hyd_in_sub[tt, j] == 1:
            q_profs[hyd_in_sub_loc, j] = q_ic_mean[tt, j]
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
            q_profs[hyd_in_sub_loc[randlocs[:-(counter_4_valid + 1)]], j] = \
                rand_gamma_vals[:-counter_4_valid]
            q_profs[hyd_in_sub_loc[randlocs[-counter_4_valid:]], j] = (
                q_ic_mean[tt, j] * tot_hyd_in_sub[tt, j] - np.sum(rand_gamma_vals[:-counter_4_valid])) / \
                (1 + counter_4_valid)

    return q_profs
