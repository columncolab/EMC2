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
    for tt in range(data_frac1.shape[0]):
        for j in range(data_frac.shape[1], 0, -1):



    



