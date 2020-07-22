import xarray as xr
import numpy as np
import dask.bag as db

from ..core import Instrument, Model
from .attenuation import calc_theory_beta_m
from .psd import calc_mu_lambda
from ..core.instrument import ureg, quantity


def calc_LDR(model, ext_OD=10., OD_from_sfc=True, LDR_per_hyd=None):
    """
    Calculates the lidar extinction mask and linear depolarization ratio for
    the given model and lidar.

    Parameters
    ----------
    model: Model
        The model to generate the parameters for.
    ext_OD: float
        The optical depth threshold for determining if the signal is extinct.
    OD_from_sfc: bool
        If True, optical depth will be calculated from the surface. If False,
        optical depth will be calculated from the top of the atmosphere.
    LDR_per_hyd: dict or None
        If a dict, the amount of LDR per hydrometeor class must be specified in
        a dictionary whose keywords are the model's hydrometeor classes. If None,
        the default settings from the model will be used.
.

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """

    if LDR_per_hyd is None:
        LDR_per_hyd = model.LDR_per_hyd

    for cloud_class in ["conv", "strat"]:
        numerator = 0.
        denominator = 0.

        for hyd_type in model.hydrometeor_classes:
            beta_p_key = "sub_col_beta_p_%s_%s" % (hyd_type, cloud_class)
            numerator += model.ds[beta_p_key] * model.LDR_per_hyd[hyd_type].magnitude
            denominator += model.ds[beta_p_key]

        model.ds["LDR_%s" % cloud_class] = numerator / denominator
        model.ds["LDR_%s" % cloud_class].attrs["long_name"] = "Linear depolarization ratio in %s" % cloud_class
        model.ds["LDR_%s" % cloud_class].attrs["units"] = "1"

    OD_cum_p_tot = model.ds["sub_col_OD_tot_strat"].values + model.ds["sub_col_OD_tot_conv"].values
    OD_cum_p_tot = np.where(OD_cum_p_tot > ext_OD, 2, 0.)
    my_diff = np.diff(OD_cum_p_tot, axis=2, prepend=0)
    ext_tmp = np.where(my_diff > 1., 1, 0)
    ext_mask = OD_cum_p_tot - ext_tmp

    if not OD_from_sfc:
        ext_mask = np.flip(ext_mask, axis=2)

    model.ds["ext_mask"] = xr.DataArray(ext_mask, dims=model.ds["LDR_conv"].dims)
    model.ds["ext_mask"].attrs["long_name"] = "Extinction mask"
    model.ds["ext_mask"].attrs["units"] = ("2 = Signal extinct, 1 = layer where signal becomes " +
                                           "extinct, 0 = signal not extinct")

    return model


def calc_lidar_moments(instrument, model, is_conv,
                       OD_from_sfc=True, parallel=True, **kwargs):
    """
    Calculates the lidar backscatter, extinction, and optical depth
    in a given column for the given lidar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a lidar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    z_field: str
        The name of the altitude field to use.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    parallel: bool
        If True, use parallelism in calculating lidar parameters.
    Additonal keyword arguments are passed into
    :py:func:`emc2.simulator.lidar_moments.calc_LDR`.

    Returns
    -------
    model: Model
        The model with the added simulated lidar parameters.
    """

    hyd_types = ["cl", "ci", "pl", "pi"]
    if not instrument.instrument_class.lower() == "lidar":
        raise ValueError("Instrument must be a lidar!")

    hyd_names_dict = {'cl': 'cloud liquid particles', 'pl': 'liquid precipitation',
                      'ci': 'cloud ice particles', 'pi': 'liquid ice precipitation'}

    p_field = model.p_field
    t_field = model.T_field
    z_field = model.z_field
    column_ds = model.ds
    eta = 1

    # Do unit conversions using pint - pressure in Pa, T in K, z in m
    p_temp = model.ds[p_field].values * getattr(ureg, model.ds[p_field].attrs["units"])
    p_values = p_temp.to('pascal').magnitude
    t_temp = quantity(model.ds[t_field].values, model.ds[t_field].attrs["units"])
    t_values = t_temp.to('kelvin').magnitude
    z_temp = model.ds[z_field].values * getattr(ureg, model.ds[z_field].attrs["units"])
    z_values = z_temp.to('meter').magnitude
    del p_temp, t_temp, z_temp

    if is_conv:
        if "conv_q_subcolumns_cl" not in model.ds.variables.keys():
            raise KeyError("Water mixing ratio in convective subcolumns must be generated first!")
        Dims = model.ds["conv_q_subcolumns_cl"].values.shape
        model.ds['sub_col_beta_p_tot_conv'] = xr.DataArray(
            np.zeros(Dims), dims=model.ds["conv_q_subcolumns_cl"].dims)
        model.ds['sub_col_alpha_p_tot_conv'] = xr.DataArray(
            np.zeros(Dims), dims=model.ds["conv_q_subcolumns_cl"].dims)
        model.ds['sub_col_OD_tot_conv'] = xr.DataArray(
            np.zeros(Dims), dims=model.ds["conv_q_subcolumns_cl"].dims)

        for hyd_type in hyd_types:
            WC = model.ds["conv_q_subcolumns_%s" % hyd_type] * p_values / \
                (instrument.R_d * t_values)
            empr_array = model.ds[model.re_fields[hyd_type]].values
            if hyd_type == "cl" or hyd_type == "pl":
                model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = xr.DataArray(
                    (3 * WC) / (2 * model.Rho_hyd[hyd_type] * 1e-3 *
                                np.tile(empr_array, (model.num_subcolumns, 1, 1))),
                    dims=model.ds["conv_q_subcolumns_cl"].dims)
            else:
                temp = t_values
                a = 0.00532 * ((temp - 273.15) + 90)**2.55
                b = 1.31 * np.exp(0.0047 * (temp - 273.15))
                a = np.tile(a, (model.num_subcolumns, 1, 1))
                b = np.tile(b, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = xr.DataArray(
                    WC / (a**(1 / b)), dims=model.ds["conv_q_subcolumns_cl"].dims)

            model.ds["sub_col_beta_p_%s_conv" % hyd_type] = model.ds["sub_col_alpha_p_%s_conv" % hyd_type] / \
                model.lidar_ratio[hyd_type].magnitude
            model.ds["sub_col_alpha_p_%s_conv" % hyd_type] = \
                model.ds["sub_col_alpha_p_%s_conv" % hyd_type].fillna(0)
            model.ds["sub_col_beta_p_%s_conv" % hyd_type] = \
                model.ds["sub_col_beta_p_%s_conv" % hyd_type].fillna(0)
            if OD_from_sfc:
                dz = np.diff(z_values, axis=0, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_conv" % hyd_type] = np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_conv" % hyd_type])
            else:
                dz = np.diff(z_values, axis=0, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_conv" % hyd_type] = np.flip(np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_conv" % hyd_type]))
            model.ds["sub_col_beta_p_tot_conv"] += model.ds["sub_col_beta_p_%s_conv" % hyd_type]
            model.ds["sub_col_alpha_p_tot_conv"] += model.ds["sub_col_alpha_p_%s_conv" % hyd_type]
            model.ds["sub_col_OD_tot_conv"] += model.ds["sub_col_OD_%s_conv" % hyd_type]
            model.ds["sub_col_beta_p_%s_conv" % hyd_type].attrs["long_name"] = \
                "Backscatter coefficient from %s in convective clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_beta_p_%s_conv" % hyd_type].attrs["units"] = "m^-1"
            model.ds["sub_col_alpha_p_%s_conv" % hyd_type].attrs["long_name"] = \
                "Extinction coefficient from %s in convective clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_alpha_p_%s_conv" % hyd_type].attrs["units"] = "m^-1"
            model.ds["sub_col_OD_%s_conv" % hyd_type].attrs["long_name"] = \
                "Optical depth from %s in convective clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_OD_%s_conv" % hyd_type].attrs["units"] = "1"

        model.ds["sub_col_beta_p_tot_conv"].attrs["long_name"] = \
            "Backscatter coefficient from all hydrometeors in convective clouds"
        model.ds["sub_col_beta_p_tot_conv"].attrs["units"] = "m^-1"
        model.ds["sub_col_alpha_p_tot_conv"].attrs["long_name"] = \
            "Extinction coefficient from all hydrometeors in convective clouds"
        model.ds["sub_col_alpha_p_tot_conv"].attrs["units"] = "m^-1"
        model.ds["sub_col_OD_tot_conv"].attrs["long_name"] = \
            "Optical depth from all hydrometeors in convective clouds"
        model.ds["sub_col_OD_tot_conv"].attrs["units"] = "1"

        model = calc_theory_beta_m(model, instrument.wavelength)
        beta_m = np.tile(model.ds['beta'].values, (model.num_subcolumns, 1, 1))
        T = np.tile(t_values, (model.num_subcolumns, 1, 1))
        model.ds['sub_col_beta_att_tot_conv'] = beta_m + model.ds['sub_col_beta_p_tot_conv'] * \
            T * np.exp(-2 * eta * model.ds['sub_col_OD_tot_conv'])
        model.ds["sub_col_beta_att_tot_conv"].attrs["long_name"] = \
            "Backscatter coefficient from all hydrometeors in convective clouds including gaseous attenuation"
        model.ds["sub_col_beta_att_tot_conv"].attrs["units"] = "m^-1"
        return model
    else:
        if "strat_q_subcolumns_cl" not in model.ds.variables.keys():
            raise KeyError("Water mixing ratio in convective subcolumns must be generated first!")
        Dims = column_ds["strat_q_subcolumns_cl"].values.shape
        for hyd_type in ["pi", "pl", "ci", "cl"]:
            n_names = "strat_n_subcolumns_%s" % hyd_type
            frac_names = "strat_frac_subcolumns_%s" % hyd_type
            print("Generating stratiform lidar variables for hydrometeor class %s" % hyd_type)
            if hyd_type == "pi":
                model.ds["sub_col_beta_p_tot_strat"] = xr.DataArray(
                    np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
                model.ds["sub_col_alpha_p_tot_strat"] = xr.DataArray(
                    np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
                model.ds["sub_col_OD_tot_strat"] = xr.DataArray(
                    np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
            model.ds["sub_col_beta_p_%s_strat" % hyd_type] = xr.DataArray(
                np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type] = xr.DataArray(
                np.zeros(Dims), dims=model.ds.strat_q_subcolumns_cl.dims)
            dD = instrument.mie_table[hyd_type]["p_diam"].values[1] - \
                instrument.mie_table[hyd_type]["p_diam"].values[0]
            fits_ds = calc_mu_lambda(model, hyd_type, subcolumns=True, **kwargs).ds
            total_hydrometeor = column_ds[frac_names] * column_ds[n_names]
            N_0 = fits_ds["N_0"].values
            mu = fits_ds["mu"].values
            num_subcolumns = model.num_subcolumns
            p_diam = instrument.mie_table[hyd_type]["p_diam"].values
            lambdas = fits_ds["lambda"].values
            beta_p = instrument.mie_table[hyd_type]["beta_p"].values
            alpha_p = instrument.mie_table[hyd_type]["alpha_p"].values
            _calc_lidar = lambda x: _calc_strat_lidar_properties(
                x, N_0, lambdas, mu, p_diam, total_hydrometeor, hyd_type, num_subcolumns, dD,
                beta_p, alpha_p)
            if parallel:
                tt_bag = db.from_sequence(np.arange(0, Dims[1], 1))
                lists = tt_bag.map(_calc_lidar).compute()
            else:
                lists = [x for x in map(_calc_lidar, np.arange(0, Dims[1], 1))]
            beta_p_strat = np.stack([x[0] for x in lists], axis=1)
            alpha_p_strat = np.stack([x[1] for x in lists], axis=1)

            model.ds["sub_col_beta_p_%s_strat" % hyd_type][:, :, :] = beta_p_strat
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type][:, :, :] = alpha_p_strat
            model.ds["sub_col_beta_p_%s_strat" % hyd_type] = \
                model.ds["sub_col_beta_p_%s_strat" % hyd_type].fillna(0)
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type] = \
                model.ds["sub_col_alpha_p_%s_strat" % hyd_type].fillna(0)

            if OD_from_sfc:
                dz = np.diff(z_values, axis=1, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_strat" % hyd_type] = np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_strat" % hyd_type], axis=2)
            else:
                dz = np.diff(z_values, axis=1, prepend=0)
                dz = np.tile(dz, (model.num_subcolumns, 1, 1))
                model.ds["sub_col_OD_%s_conv" % hyd_type] = np.flip(np.cumsum(
                    dz * model.ds["sub_col_alpha_p_%s_strat" % hyd_type], axis=2), axis=2)

            model.ds["sub_col_beta_p_%s_strat" % hyd_type].attrs["long_name"] = \
                "Backscatter coefficient from %s in stratiform clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_beta_p_%s_strat" % hyd_type].attrs["units"] = "m^-1"
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type].attrs["long_name"] = \
                "Extinction coefficient from %s in stratiform clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_alpha_p_%s_strat" % hyd_type].attrs["units"] = "m^-1"
            model.ds["sub_col_OD_%s_strat" % hyd_type].attrs["long_name"] = \
                "Optical depth from %s in stratiform clouds" % hyd_names_dict[hyd_type]
            model.ds["sub_col_OD_%s_strat" % hyd_type].attrs["units"] = "1"

            model.ds["sub_col_beta_p_tot_strat"] += model.ds["sub_col_beta_p_%s_strat" % hyd_type].fillna(0)
            model.ds["sub_col_alpha_p_tot_strat"] += model.ds["sub_col_alpha_p_%s_strat" % hyd_type].fillna(0)
            model.ds["sub_col_OD_tot_strat"] += model.ds["sub_col_OD_%s_strat" % hyd_type].fillna(0)

        model.ds["sub_col_beta_p_tot_strat"].attrs["long_name"] = \
            "Backscatter coefficient from all hydrometeors in stratiform clouds"
        model.ds["sub_col_beta_p_tot_strat"].attrs["units"] = "m^-1"
        model.ds["sub_col_alpha_p_tot_strat"].attrs["long_name"] = \
            "Extinction coefficient from all hydrometeors in stratiform clouds"
        model.ds["sub_col_alpha_p_tot_strat"].attrs["units"] = "m^-1"
        model.ds["sub_col_OD_tot_strat"].attrs["long_name"] = \
            "Optical depth from all hydrometeors in stratiform clouds"
        model.ds["sub_col_OD_tot_strat"].attrs["units"] = "1"

        return model


def _calc_strat_lidar_properties(tt, N_0, lambdas, mu, p_diam, total_hydrometeor,
                                 hyd_type, num_subcolumns, dD, beta_p, alpha_p):
    Dims = total_hydrometeor.shape
    num_diam = len(p_diam)
    beta_p_strat = np.zeros((num_subcolumns, Dims[2]))
    alpha_p_strat = np.zeros((num_subcolumns, Dims[2]))

    if tt % 50 == 0:
        print('Stratiform moment for class %s progress: %d/%d' % (hyd_type, tt, Dims[1]))
    for k in range(Dims[2]):
        if np.all(total_hydrometeor.values[:, tt, k] == 0):
            continue
        N_D = []
        for i in range(num_subcolumns):
            N_0_tmp = N_0[i, tt, k]
            lambda_tmp = lambdas[i, tt, k]
            mu_temp = mu[i, tt, k]
            N_D.append(N_0_tmp * p_diam ** mu_temp * np.exp(-lambda_tmp * p_diam))
        N_D = np.stack(N_D, axis=0)

        Calc_tmp = np.tile(beta_p, (num_subcolumns, 1)) * N_D
        beta_p_strat[:, k] = (
            Calc_tmp[:, ::num_diam - 1].sum(axis=1) / 2 + Calc_tmp[:, 1:-1].sum(axis=1)) * dD
        Calc_tmp = np.tile(alpha_p, (num_subcolumns, 1)) * N_D
        alpha_p_strat[:, k] = (
            Calc_tmp[:, ::num_diam - 1].sum(axis=1) / 2 + Calc_tmp[:, 1:-1].sum(axis=1)) * dD

    return beta_p_strat, alpha_p_strat
