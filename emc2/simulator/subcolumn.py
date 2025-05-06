import numpy as np
import xarray as xr
import dask.bag as db
from time import time


def set_convective_sub_col_frac(model, hyd_type, N_columns=None, use_rad_logic=True,
                                q_trunc_thresh=1e-18):
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
    q_trunc_thresh: float
        truncation value for q. Smaller values will be treated as 0.

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

    if use_rad_logic:
        method_str = "Radiation logic"
    else:
        method_str = "Microphysics logic"
    my_dims = model.ds[model.conv_frac_names[hyd_type]].dims

    if model.num_subcolumns == 0:
        model.ds['subcolumn'] = xr.DataArray(np.arange(0, N_columns), dims='subcolumn')

    if N_columns == 1:
       print(f"num_subcolumns == 1 (subcolumn generator turned off); setting subcolumns frac "
              f"fields for convective {hyd_type} based on q > {q_trunc_thresh} kg/kg")
       model.ds[("conv_frac_subcolumns_" + hyd_type)] = xr.DataArray(
           np.where(model.ds[model.q_names_convective[hyd_type]].values > q_trunc_thresh, 1., 0.),
           dims=('subcolumn', my_dims[0], my_dims[1]))
    else:  # num_subcolumns > 1
        if use_rad_logic:
            data_frac = np.round(
                model.ds[model.conv_frac_names_for_rad[hyd_type]].values * model.num_subcolumns).astype(int)
            data_frac = np.where(
                model.ds[model.q_names_convective[hyd_type]].values > q_trunc_thresh, data_frac, 0)
        else:
            data_frac = np.round(
                model.ds[model.conv_frac_names[hyd_type]].values * model.num_subcolumns).astype(int)

        # In case we only have one time step
        if len(data_frac.shape) == 1:
            data_frac = data_frac[np.newaxis, :]

        conv_profs = np.zeros((model.num_subcolumns, data_frac.shape[0], data_frac.shape[1]),
                              dtype=bool)
        for i in range(1, model.num_subcolumns + 1):
            for k in range(data_frac.shape[1]):
                mask = np.where(data_frac[:, k] == i)[0]
                conv_profs[0:i, mask, k] = True
        model.ds[("conv_frac_subcolumns_" + hyd_type)] = xr.DataArray(
            conv_profs, dims=('subcolumn', my_dims[0], my_dims[1]))
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["units"] = "0 = no, 1 = yes"
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["long_name"] = (
        "Is there hydrometeors of type %s in each subcolumn?" % hyd_type)
    model.ds[("conv_frac_subcolumns_" + hyd_type)].attrs["Processing method"] = method_str

    return model


def set_stratiform_sub_col_frac(model, N_columns=None, use_rad_logic=True, parallel=True, chunk=None,
                                q_trunc_thresh=1e-18):
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
    q_trunc_thresh: float
        truncation value for q. Smaller values will be treated as 0.

    Returns
    -------
    model: :py:func: `emc2.core.Model`
        The Model object with the stratiform hydrometeor fraction in each subcolumn added.
    """
    t0 = time()

    if use_rad_logic:
        method_str = "Radiation logic"
    else:
        method_str = "Microphysics logic"

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
    subcolumn_dims = ("subcolumn", *model.ds[model.strat_frac_names_for_rad["cl"]].dims)
    if N_columns == 1:
        print(f"num_subcolumns == 1 (subcolumn generator turned off); setting subcolumns frac "
              f"fields to 1 for startiform cl and ci based on q > {q_trunc_thresh} kg/kg")
        model.ds['strat_frac_subcolumns_cl'] = xr.DataArray(
            np.where(np.tile(model.ds[model.q_names_stratiform["cl"]],
                             (1, 1, 1)) > q_trunc_thresh, 1., 0.),
            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
        model.ds['strat_frac_subcolumns_ci'] = xr.DataArray(
            np.where(np.tile(model.ds[model.q_names_stratiform["ci"]],
                             (1, 1, 1)) > q_trunc_thresh, 1., 0.),
            dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    else:  # num_subcolumns > 1
        N_columns = len(model.ds["subcolumn"])
        if use_rad_logic:
            data_frac1 = model.ds[model.strat_frac_names_for_rad["cl"]]
            data_frac1 = data_frac1.where(
                model.ds[model.q_names_stratiform["cl"]] > q_trunc_thresh, 0)
            data_frac2 = model.ds[model.strat_frac_names_for_rad["ci"]]
            data_frac2 = data_frac2.where(
                model.ds[model.q_names_stratiform["ci"]] > q_trunc_thresh, 0)
        else:
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
    model.ds['strat_frac_subcolumns_cl'].attrs["long_name"] = (
        "Liquid cloud particles present? [stratiform]")
    model.ds['strat_frac_subcolumns_cl'].attrs["units"] = "0 = no, 1 = yes"
    model.ds['strat_frac_subcolumns_cl'].attrs["Processing method"] = method_str
    model.ds['strat_frac_subcolumns_ci'].attrs["long_name"] = (
        "Liquid cloud particles present? [stratiform]")
    model.ds['strat_frac_subcolumns_ci'].attrs["units"] = "0 = no, 1 = yes"
    model.ds['strat_frac_subcolumns_ci'].attrs["Processing method"] = method_str

    print("Done! total processing time = %.2fs" % (time() - t0))

    return model


def set_precip_sub_col_frac(model, is_conv, N_columns=None, use_rad_logic=True,
                            parallel=True, chunk=None, 
                            precip_types=["pl", "pi"], q_trunc_thresh=1e-18):
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
    ice_hyd_type: str
        The ice hydrometeor type to include in the subcolumn distribution for precipitation
    q_trunc_thresh: float
        truncation value for q. Smaller values will be treated as 0.

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

    data_frac = []
    # For subcolumn coverage, lump 3 ice precip categories into one
    if is_conv:
        precip_type = 'conv'
        precip_type_full = 'convective'
        if use_rad_logic:
            method_str = "Radiation logic"
            for hyd_type in precip_types:
                data_frac.append(
                    model.ds[model.conv_frac_names_for_rad[hyd_type]])
                data_frac[-1] = data_frac[-1].where(
                    model.ds[model.q_names_convective[hyd_type]] > q_trunc_thresh, 0).values
        else:
            method_str = "Microphysics logic"
            for hyd_type in precip_types:
                data_frac.append(
                    model.ds[model.conv_frac_names[hyd_type]].values)
    else:
        precip_type = 'strat'
        precip_type_full = 'stratiform'
        if use_rad_logic:
            method_str = "Radiation logic"
            for hyd_type in precip_types:
                data_frac.append(
                    model.ds[model.strat_frac_names_for_rad[hyd_type]])
                data_frac[-1] = data_frac[-1].where(
                    model.ds[model.q_names_stratiform[hyd_type]] > q_trunc_thresh, 0).values
        else:
            method_str = "Microphysics logic"
            for hyd_type in precip_types:
                data_frac.append(
                    model.ds[model.strat_frac_names[hyd_type]].values)
    out_prof_long_name = []
    out_prof_name = []
    for hyd_type in precip_types:
        if hyd_type == "pl":
            out_prof_long_name.append(
                'Liquid precipitation present? [%s]' % hyd_type)
        else:
            out_prof_long_name.append(
                'Ice precipitation present? [%s]' % hyd_type)
        out_prof_name.append(precip_type + '_frac_subcolumns_%s' % hyd_type)
    in_prof_cloud_name_liq = precip_type + '_frac_subcolumns_cl'
    in_prof_cloud_name_ice = precip_type + '_frac_subcolumns_ci'
    subcolumn_dims = model.ds[in_prof_cloud_name_ice].dims

    if N_columns == 1:
        print(f"num_subcolumns == 1 (subcolumn generator turned off); setting subcolumns frac "
              f"fields to 1 for {precip_type} precip based on q > {q_trunc_thresh} kg/kg")
        for hyd_type in precip_types:
            if is_conv:
                q_use = np.tile(model.ds[model.q_names_convective[hyd_type]], (1, 1, 1))
            else:
                q_use = np.tile(model.ds[model.q_names_stratiform[hyd_type]], (1, 1, 1))
            model.ds[precip_type + '_frac_subcolumns_%s' % hyd_type] = xr.DataArray(
                np.where(q_use > q_trunc_thresh, 1., 0.),
                dims=(subcolumn_dims[0], subcolumn_dims[1], subcolumn_dims[2]))
    else:
        full_overcast_pl_pi = 0
        if in_prof_cloud_name_liq not in model.ds.variables.keys():
            raise KeyError("%s is not a variable in the model object. Please ensure that you have" +
                           "generated the cloud particle subcolumns before running this routine." %
                           in_prof_cloud_name_liq)

        if in_prof_cloud_name_ice not in model.ds.variables.keys():
            raise KeyError("%s is not a variable in the model object. Please ensure that you have" +
                           "generated the cloud particle subcolumns before running this routine." %
                           in_prof_cloud_name_ice)

        for i in range(len(data_frac)):
            data_frac[i] = np.round(data_frac[i] * model.num_subcolumns).astype(int)
        strat_profs = np.logical_or(model.ds[in_prof_cloud_name_ice].values,
                                    model.ds[in_prof_cloud_name_liq].values)
        is_cloud = data_frac[0] > 0
        for i in range(1, len(data_frac)):
            is_cloud = np.logical_or(is_cloud, data_frac[i] > 0)
        is_cloud_one_above = np.roll(is_cloud, -1, axis=1)
        is_cloud_one_above[:, -1] = False
        overlapping_cloud = np.logical_and(is_cloud, is_cloud_one_above)
        precip_exist = np.stack([frac > 0 for frac in data_frac])
        PF_val = np.max(np.stack(data_frac), axis=0)
        cond = [strat_profs, ~strat_profs]
        _allocate_precip_sub_cols = lambda x: _allocate_precip_sub_col(
            x, cond, N_columns, data_frac, PF_val,
            precip_exist, full_overcast_pl_pi, overlapping_cloud)

        t_dim = data_frac[0].shape[0]
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
        typ_string = ""
        for typ in precip_types:
            typ_string = typ_string + " & " + typ
        typ_string = typ_string[3:]    
        print("Fully overcast %s in %s voxels" % (typ_string, full_overcast_pl_pi))

        for i in range(len(out_prof_name)):
            model.ds[out_prof_name[i]] = xr.DataArray(p_strat_profs[:, :, :, i],
                                                   dims=(
                                                   subcolumn_dims[0],
                                                   subcolumn_dims[1],
                                                   subcolumn_dims[2]))
    for i in range(len(out_prof_name)):
        model.ds[out_prof_name[i]].attrs["long_name"] = out_prof_long_name[i]
        model.ds[out_prof_name[i]].attrs["units"] = "0 = no, 1 = yes"
        model.ds[out_prof_name[i]].attrs["Processing method"] = method_str

    print("Done! total processing time = %.2fs" % (time() - t0))

    return model


def set_q_n(model, hyd_type, is_conv=True, qc_flag=False, inv_rel_var=1, use_rad_logic=True,
            parallel=True, chunk=None, q_trunc_thresh=1e-18):
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
    q_trunc_thresh: float
        truncation value for q. Smaller values will be treated as 0.

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
            data_frac = np.where(
                model.ds[model.q_names_stratiform[hyd_type]].values > q_trunc_thresh, data_frac, 0)
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
            data_frac = np.where(
                model.ds[model.q_names_convective[hyd_type]].values > q_trunc_thresh, data_frac, 0)
        else:
            method_str = "Microphysics logic"
            data_frac = model.ds[model.conv_frac_names[hyd_type]].astype('float64').values
        sub_data_frac = model.ds[frac_fieldname]
        q_array = model.ds[model.q_names_convective[hyd_type]].astype('float64').values
        q_name = "conv_q_subcolumns_%s" % hyd_type

    if model.num_subcolumns == 1:
        print(f"num_subcolumns == 1 (subcolumn generator turned off); setting subcolumns q (and N micro logic) "
              f"fields for {hyd_type} equal to grid-cell in-cloud mean")
        model.ds[q_name] = xr.DataArray(
            np.tile(np.where(q_array > q_trunc_thresh, q_array / data_frac, 0), (1, 1, 1)),
            dims=model.ds[frac_fieldname].dims)
        if not is_conv:
            model.ds[n_name] = xr.DataArray(
                np.tile(model.ds[model.N_field[hyd_type]].astype('float64').values / data_frac, (1, 1, 1)),
                                            dims=model.ds[frac_fieldname].dims)
    else:
        if qc_flag:
            q_ic_mean = np.where(q_array > q_trunc_thresh, q_array / data_frac, 0)
            q_ic_mean = np.where(np.isnan(q_ic_mean), 0, q_ic_mean)
            tot_hyd_in_sub = sub_data_frac.sum(axis=0)

            _distribute_cl_q_n_sub_cols = lambda x: _distribute_cl_q_n(
                x, sub_data_frac, inv_rel_var, model.num_subcolumns, tot_hyd_in_sub, q_ic_mean)

            t_dim = data_frac.shape[0]
            if parallel:
                print("Now distributing q in subcolumns in parallel")
                if chunk is None:
                    tt_bag = db.from_sequence(np.arange(0, t_dim, 1))
                    my_tuple = tt_bag.map(_distribute_cl_q_n_sub_cols).compute()
                else:
                    my_tuple = []
                    j = t_dim - 1
                    while j >= 0:
                        if j + chunk > t_dim:
                            ind_max = t_dim
                        else:
                            ind_max = j + chunk
                        print("Stage 1 of 2: Processing columns %d-%d out of %d" % (j, ind_max, t_dim))
                        tt_bag = db.from_sequence(np.arange(j, ind_max, 1))
                        my_tuple += tt_bag.map(_distribute_cl_q_n_sub_cols).compute()
                        j -= chunk
            else:
                my_tuple = [x for x in map(_distribute_cl_q_n_sub_cols, np.arange(0, t_dim, 1))]

            q_profs = np.stack([x for x in my_tuple], axis=1)

        else:
            q_profs = np.where(q_array > q_trunc_thresh, q_array / data_frac, 0)
            q_profs = np.tile(q_profs, (model.num_subcolumns, 1, 1))
            q_profs = np.where(sub_data_frac, q_profs, 0)
        q_profs = np.where(np.isnan(q_profs), 0, q_profs)
        model.ds[q_name] = xr.DataArray(q_profs, dims=model.ds[frac_fieldname].dims)
        if not is_conv:
            N_profs = np.where(np.isnan(N_profs), 0, N_profs)
            model.ds[n_name] = xr.DataArray(N_profs, dims=model.ds[frac_fieldname].dims)

    model.ds[q_name].attrs["long_name"] = "q in subcolumns"
    model.ds[q_name].attrs["units"] = r"$kg\ kg^{-1}$"
    model.ds[q_name].attrs["Processing method"] = method_str
    if not is_conv:
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

    strat_profs = np.zeros((2, N_columns, data_frac1.shape[1]), dtype=bool)

    for j in range(data_frac1.shape[1] - 2, -1, -1):
        cld_2_assign = np.array([data_frac1[tt, j], data_frac2[tt, j]])
        I_min = np.argmin(cld_2_assign)
        I_max = np.argmax(cld_2_assign)
        if cld_2_assign[I_max] == 0:
            continue
        if cld_2_assign[I_min] == N_columns:
            strat_profs[:, :, j] = True
            full_overcast_cl_ci += 1
            continue
        elif I_min == I_max:  # This is the case of cl_frac == ci_frac != 1
            I_max = 1
        if overlapping_cloud[tt, j]:
            overlying_locs = np.zeros((2, strat_profs.shape[1]))
            overlying_locs1 = np.argwhere(np.logical_and(strat_profs[0, :, j + 1], ~conv_profs[:, tt, j]))
            overlying_locs2 = np.argwhere(np.logical_and(strat_profs[1, :, j + 1], ~conv_profs[:, tt, j]))
            overlying_num = np.array([len(overlying_locs1), len(overlying_locs2)], dtype=int)
            over_diff = abs(overlying_num[1] - overlying_num[0])
            Iover_min = np.argmin(overlying_num)
            Iover_max = np.argmax(overlying_num)
            over_unique_lo = _setxor(overlying_locs1, overlying_locs2)

            if overlying_num[Iover_min] > cld_2_assign[I_max]:
                if cld_2_assign[I_max] > 0:
                    rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_max])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs[0:cld_2_assign[I_min]]]
                    strat_profs[I_min, inds, j] = True
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    strat_profs[I_max, inds, j] = True
                cld_2_assign = np.zeros(2)
            elif overlying_num[Iover_min] > cld_2_assign[I_min]:  # overlying_num[Iover_min] <= cld_2_assign[I_max]
                if cld_2_assign[I_min] > 0: 
                    rand_locs = _randperm(overlying_num.min(), size=cld_2_assign[I_min])
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)][rand_locs]
                    strat_profs[I_min, inds, j] = True
                    inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                    strat_profs[I_max, inds, j] = True
                cld_2_assign[I_min] = 0
                cld_2_assign[I_max] -= overlying_num[Iover_min]

                if over_diff > cld_2_assign[I_max]:  # over_n[Iover_min] < cld_2_assign[I_max] < over_n[Iover_max]
                    rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
                    inds = over_unique_lo[rand_locs]
                    strat_profs[I_max, inds, j] = True
                    cld_2_assign[I_max] = 0.
                else:
                    strat_profs[I_max, over_unique_lo, j] = True
                    cld_2_assign[I_max] -= over_diff
            elif overlying_num[Iover_max] > cld_2_assign[I_min]:
                inds = locals()["overlying_locs%d" % (Iover_min + 1)]
                strat_profs[I_min, inds, j] = True
                strat_profs[I_max, inds, j] = True
                cld_2_assign -= overlying_num[Iover_min]

                if over_diff > cld_2_assign[I_max]:
                    rand_locs = _randperm(over_diff, size=cld_2_assign[I_max])
                    inds = over_unique_lo[rand_locs[0:cld_2_assign[I_min]]]
                    strat_profs[I_min, inds, j] = True
                    inds = over_unique_lo[rand_locs]
                    strat_profs[I_max, inds, j] = True
                    cld_2_assign = np.zeros(2)
                else:
                    if cld_2_assign[I_min] > 0:
                        rand_locs = _randperm(over_diff, size=cld_2_assign[I_min])
                        inds = over_unique_lo[rand_locs]
                        strat_profs[I_min, inds, j] = True
                    cld_2_assign[I_min] = 0
                    strat_profs[I_max, over_unique_lo, j] = True
                    cld_2_assign[I_max] -= over_diff
            else:
                inds = locals()["overlying_locs%d" % (Iover_max + 1)]
                strat_profs[I_min, inds, j] = True
                strat_profs[I_max, inds, j] = True
                cld_2_assign -= overlying_num[Iover_max]

        if cld_2_assign[I_max] > 0:
            sprof = strat_profs[I_max, :, :]
            free_locs_max = np.where(np.logical_and(~sprof[:, j], ~conv_profs[:, tt, j]))[0]
            free_num = len(free_locs_max)
            rand_locs = _randperm(free_num, size=int(cld_2_assign[I_max]))
            strat_profs[I_max, free_locs_max[rand_locs], j] = True
            if cld_2_assign[I_min] > 0.:
                strat_profs[I_min, free_locs_max[rand_locs[0:cld_2_assign[I_min]]], j] = True

    return full_overcast_cl_ci, strat_profs[0, :, :], strat_profs[1, :, :]


def _allocate_precip_sub_col(tt, cond, N_columns, data_frac, PF_val,
                             precip_exist, full_overcast_pl_pi, overlapping_cloud):
    p_strat_profs = np.zeros(
        (N_columns, data_frac[0].shape[1], len(data_frac)), dtype=bool)

    for j in range(data_frac[0].shape[1] - 2, -1, -1):  # loop from 2nd penultimate height to sfc
        all_overlap = True
        for i in range(len(data_frac)):
            all_overlap = np.logical_and(
                all_overlap, data_frac[i][tt, j] == N_columns)
        
        if all_overlap:
            p_strat_profs[:, j, :] = True
            full_overcast_pl_pi += 1
            continue

        PF_per_val = [None] * len(data_frac)  # remaining to allocate per precip class
        for i in range(len(data_frac)):
            PF_per_val[i] = int(data_frac[i][tt, j])
        if overlapping_cloud[tt, j]:  # First allocate to overlying precip (extend vertically)
            overlying_locs = np.where(np.any(p_strat_profs[:, j + 1, :], axis=1))[0]
            overlying_num = len(overlying_locs)
            if overlying_num > PF_val[tt, j]:  # more overlying than the class w/ maximum frac
                rand_locs = _randperm(overlying_num, PF_val[tt, j])
                for i in range(len(data_frac)):
                    if precip_exist[i, tt, j]:
                        p_strat_profs[overlying_locs[rand_locs[:PF_per_val[i]]], j, i] = True
                PF_val[tt, j] = 0
            else:
                rand_locs = np.random.permutation(overlying_num)  # random before loop to ensure max overlap
                for i in range(len(data_frac)):
                    if precip_exist[i, tt, j]:
                        if overlying_num > PF_per_val[i]:  # more overlying than current precip class frac
                            p_strat_profs[overlying_locs[rand_locs[:PF_per_val[i]]], j, i] = True
                            PF_per_val[i] = 0
                        else:
                            p_strat_profs[overlying_locs, j, i] = True
                            PF_per_val[i] -= overlying_num
                PF_val[tt, j] -= overlying_num

        for ii in range(2):  # Second, allocate to cloudy subcol; then to hyd-free subcol
            if PF_val[tt, j] > 0:
                free_locs = np.where(np.logical_and(
                    ~np.any(p_strat_profs[:, j, :], axis=1), cond[ii][:, tt, j]))[0]  # True for remaining allocate
                free_num = len(free_locs)
                if free_num > 0:
                    if free_num > PF_val[tt, j]:
                        rand_locs = _randperm(free_num, PF_val[tt, j])
                        for i in range(len(data_frac)):
                            if precip_exist[i, tt, j]:
                                p_strat_profs[free_locs[rand_locs[:PF_per_val[i]]], j, i] = True
                        PF_val[tt, j] = 0
                    else:
                        rand_locs = np.random.permutation(free_num)  # random before loop to ensure max overlap
                        for i in range(len(data_frac)):
                            if precip_exist[i, tt, j]:
                                if free_num > PF_per_val[i]:  # more free locs than current precip class frac
                                    p_strat_profs[free_locs[rand_locs[:PF_per_val[i]]], j, i] = True
                                    PF_per_val[i] = 0
                                else:
                                    p_strat_profs[free_locs, j, i] = True
                                    PF_per_val[i] -= free_num
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
            rand_gamma_vals = np.random.gamma(a, b, tot_hyd_in_sub[tt, j])  # extra entry 2 prevent indexing issues
            valid_vals = False
            counter_4_valid = 0
            while not valid_vals:  # Finding first index w/ random value sum > cell mean --> randomize up to there
                counter_4_valid += 1
                valid_vals = (q_ic_mean[tt, j] * tot_hyd_in_sub[tt, j] -
                              rand_gamma_vals[:-counter_4_valid].sum()) > 0
            q_profs[hyd_in_sub_loc[randlocs[:-counter_4_valid]], j] = (
                rand_gamma_vals[:-counter_4_valid])
            q_profs[hyd_in_sub_loc[randlocs[-counter_4_valid:]], j] = (
                q_ic_mean[tt, j] * tot_hyd_in_sub[tt, j] -
                np.sum(rand_gamma_vals[:-counter_4_valid])) / float(counter_4_valid)

    return q_profs
