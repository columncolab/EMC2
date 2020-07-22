import xarray as xr
import numpy as np
import dask.bag as db
import dask.array as da

from ..core import Instrument, Model
from .attenuation import calc_radar_atm_attenuation
from .psd import calc_mu_lambda
from ..core.instrument import ureg


def calc_radar_reflectivity_conv(instrument, model, hyd_type):
    """
    This estimates the radar reflectivity given a profile of liquid water mixing ratio.
    Convective DSDs are assumed.

    Parameters
    ----------
    instrument: :func:`emc2.core.Instrument` class
        The instrument to calculate the reflectivity parameters for.
    model: :func:`emc2.core.Model` class
        The model to calculate the parameters for.
    hyd_type: str
        The assumed hydrometeor type. Must be one of:
        'cl' (cloud liquid), 'ci' (cloud ice),
        'pl' (liquid precipitation), 'pi' (ice precipitation).

    Returns
    -------
    model: :func:`emc2.core.Model`
        Returns a Model with an added reflectivity field.
    """
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    if hyd_type.lower() not in ['cl', 'ci', 'pl', 'pi']:
        raise ValueError("%s is not a valid hydrometeor type. Valid choices are cl, ci, pl, and pi." % hyd_type)
    q_field = "conv_q_subcolumns_%s" % hyd_type
    p_field = model.p_field
    t_field = model.T_field
    column_ds = model.ds

    WC = column_ds[q_field] * 1e3 * column_ds[p_field] * 1e2 / (instrument.R_d * column_ds[t_field])
    if hyd_type.lower() == "cl":
        column_ds['Ze'] = 0.031 * WC ** 1.56
    elif hyd_type.lower() == "pl":
        column_ds['Ze'] = ((WC * 1e3) / 3.4)**1.75
    else:
        Tc = column_ds[t_field] - 273.15
        if instrument.freq >= 2e9 and instrument.freq < 4e9:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0197 * Tc + 1.7) / 0.060) / 10.)
        elif instrument.freq >= 27e9 and instrument.freq < 40e9:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.699)) / 10.)
        elif instrument.freq >= 75e9 and instrument.freq < 110e9:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.00706 * Tc + 0.992) / (0.000580 * Tc + 0.0923)) / 10.)
        else:
            column_ds['Ze'] = 10**(((np.log10(WC) + 0.0186 * Tc + 1.63) / (0.000242 * Tc + 0.0699)) / 10.)
    column_ds['Ze'] = 10 * np.log10(column_ds["Ze"])
    column_ds['Ze'].attrs["long_name"] = "Radar reflectivity factor"
    column_ds['Ze'].attrs["units"] = "dBZ"
    model.ds = column_ds
    return model


def calc_radar_moments(instrument, model, is_conv,
                       OD_from_sfc=True, parallel=True, **kwargs):
    """
    Calculates the reflectivity, doppler velocity, and spectral width
    in a given column for the given radar.

    Parameters
    ----------
    instrument: Instrument
        The instrument to simulate. The instrument must be a radar.
    model: Model
        The model to generate the parameters for.
    is_conv: bool
        True if the cell is convective
    z_field: str
        The name of the altitude field to use.
    OD_from_sfc: bool
        If True, then calculate optical depth from the surface.
    parallel: bool
        If True, then use parallelism to calculate each column quantity.
    Additional keyword arguments are passed into
    :func:`emc2.simulator.reflectivity.calc_radar_reflectivity_conv` and
    :func:`emc2.simulator.attenuation.calc_radar_atm_attenuation`.

    Returns
    -------
    model: :func:`emc2.core.Model`
        The xarray Dataset containing the calculated radar moments.
    """

    # We don't care about invalid value errors
    np.seterr(divide='ignore', invalid='ignore')
    hyd_types = ["cl", "ci", "pl", "pi"]
    hyd_names_dict = {'cl': 'cloud liquid particles', 'pl': 'liquid precipitation',
                      'ci': 'cloud ice particles', 'pi': 'liquid ice precipitation'}
    if not instrument.instrument_class.lower() == "radar":
        raise ValueError("Reflectivity can only be derived from a radar!")

    p_field = model.p_field
    t_field = model.T_field
    z_field = model.z_field
    column_ds = model.ds

    if is_conv:
        q_names = model.q_names_convective
        for hyd_type in hyd_types:
            temp_ds = calc_radar_reflectivity_conv(instrument, model, hyd_type)

            var_name = "sub_col_Ze_%s_conv" % hyd_type
            column_ds[var_name] = temp_ds.ds["Ze"]
            if "sub_col_Ze_tot_conv" in column_ds.variables.keys():
                column_ds["sub_col_Ze_tot_conv"] += 10**(column_ds[var_name] / 10)
            else:
                column_ds["sub_col_Ze_tot_conv"] = 10**(column_ds[var_name] / 10)
            column_ds[var_name].attrs["long_name"] = \
                "Radar reflectivity factor from %s in convective clouds" % hyd_type
            column_ds[var_name].attrs["units"] = "dBZ"

        column_ds["sub_col_Ze_tot_conv"] = 10 * np.log10(column_ds["sub_col_Ze_tot_conv"])

        kappa_ds = calc_radar_atm_attenuation(instrument, model)
        kappa_f = 6 * np.pi / instrument.wavelength * 1e-6 * model.Rho_hyd["cl"].magnitude
        WC = column_ds[q_names["cl"]] + column_ds[q_names["pl"]] * 1e3 * \
            column_ds[p_field] / (instrument.R_d * column_ds[t_field])
        dz = np.diff(column_ds[z_field].values, axis=0,
                     prepend=0.)

        if OD_from_sfc:
            WC_new = np.zeros_like(WC)
            WC_new[:, 1:] = WC[:, :-1]
            liq_ext = np.cumsum(kappa_f * dz * WC_new, axis=1)
            atm_ext = np.cumsum(kappa_ds.ds["kappa_att"].values * dz, axis=1)
        else:
            WC_new = np.zeros_like(WC)
            WC_new[:, 1:] = WC[:, :-1]
            liq_ext = np.flip(np.cumsum(kappa_f * dz * WC_new), axis=1)
            atm_ext = np.flip(np.cumsum(kappa_ds.ds["kappa_att"].values * dz), axis=1)

        if len(liq_ext.shape) == 1:
            liq_ext = liq_ext[:, np.newaxis]
        if len(atm_ext.shape) == 1:
            atm_ext = atm_ext[:, np.newaxis]
        liq_ext = xr.DataArray(liq_ext, dims=(model.time_dim, model.height_dim))
        atm_ext = xr.DataArray(atm_ext, dims=(model.time_dim, model.height_dim))

        column_ds["sub_col_Ze_att_tot_conv"] = column_ds["sub_col_Ze_tot_conv"] / \
            10**(2 * liq_ext / 10.) / 10**(2 * atm_ext / 10.)
        column_ds["sub_col_Ze_tot_conv"] = column_ds["sub_col_Ze_tot_conv"].where(
            column_ds["sub_col_Ze_tot_conv"] != 0)
        column_ds["sub_col_Ze_att_tot_conv"].attrs["long_name"] = \
            "Radar reflectivity factor from all hydrometeors in convection accounting for gaseous attenuation"
        column_ds["sub_col_Ze_att_tot_conv"].attrs["units"] = "dBZ"
        column_ds["sub_col_Ze_tot_conv"].attrs["long_name"] = \
            "Radar reflectivity factor from all hydrometeors in convection"
        column_ds["sub_col_Ze_tot_conv"].attrs["units"] = "dBZ"
        model.ds = column_ds
        return model


    Dims = column_ds["strat_q_subcolumns_cl"].values.shape

    moment_denom_tot = np.zeros(Dims)
    V_d_numer_tot = np.zeros(Dims)
    sigma_d_numer_tot = np.zeros(Dims)

    for hyd_type in ["cl", "pl", "ci", "pi"]:
        n_names = "strat_n_subcolumns_%s" % hyd_type
        frac_names = "strat_frac_subcolumns_%s" % hyd_type
        if hyd_type == "cl":
            column_ds["sub_col_Ze_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
            column_ds["sub_col_Vd_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
            column_ds["sub_col_sigma_d_tot_strat"] = xr.DataArray(
                np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
        column_ds["sub_col_Ze_%s_strat" % hyd_type] = xr.DataArray(
            np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
        column_ds["sub_col_Vd_%s_strat" % hyd_type] = xr.DataArray(
            np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
        column_ds["sub_col_sigma_d_%s_strat" % hyd_type] = xr.DataArray(
            np.zeros(Dims), dims=column_ds.strat_q_subcolumns_cl.dims)
        dD = instrument.mie_table[hyd_type]["p_diam"].values[1] - \
            instrument.mie_table[hyd_type]["p_diam"].values[0]
        fits_ds = calc_mu_lambda(model, hyd_type, subcolumns=True, **kwargs).ds
        total_hydrometeor = column_ds[frac_names] * column_ds[n_names]
        p_diam = instrument.mie_table[hyd_type]["p_diam"].values
        alpha_p = instrument.mie_table[hyd_type]["alpha_p"].values
        beta_p = instrument.mie_table[hyd_type]["beta_p"].values
        num_subcolumns = model.num_subcolumns
        v_tmp = model.vel_param_a[hyd_type] * p_diam ** model.vel_param_b[hyd_type]
        v_tmp = v_tmp.magnitude
        if hyd_type == "cl":
            N_0 = fits_ds["N_0"].values
            lambdas = fits_ds["lambda"].values
            mu = fits_ds["mu"].values

            _calc_liquid = lambda x: _calculate_observables_liquid(
                x, total_hydrometeor, N_0, lambdas, mu,
                alpha_p, beta_p, v_tmp, num_subcolumns, instrument, dD, p_diam)
            if parallel:
                print("Doing parallel calculation for %s" % hyd_type)
                tt_bag = db.from_sequence(np.arange(0, Dims[1], 1))
                my_tuple = tt_bag.map(_calc_liquid).compute()
            else:
                my_tuple = [x for x in map(_calc_liquid, np.arange(0, Dims[1], 1))]

            V_d_numer_tot = np.nan_to_num(np.stack([x[0] for x in my_tuple], axis=1))
            moment_denom_tot = np.nan_to_num(np.stack([x[1] for x in my_tuple], axis=1))
            od_tot = np.nan_to_num(np.stack([x[2] for x in my_tuple], axis=1))

            column_ds["sub_col_Ze_cl_strat"][:, :, :] = np.stack([x[3] for x in my_tuple], axis=1)
            column_ds["sub_col_Vd_cl_strat"][:, :, :] = np.stack([x[4] for x in my_tuple], axis=1)
            column_ds["sub_col_sigma_d_cl_strat"][:, :, :] = np.stack([x[5] for x in my_tuple], axis=1)
            del my_tuple
        else:
            sub_q_array = column_ds["strat_q_subcolumns_%s" % hyd_type].values
            _calc_other = lambda x: _calculate_other_observables(
                x, total_hydrometeor, fits_ds, model, instrument, sub_q_array, hyd_type, dD)
            if parallel:
                print("Doing parallel calculation for %s" % hyd_type)
                my_tuple = tt_bag.map(_calc_other).compute()
            else:
                my_tuple = [x for x in map(_calc_other, np.arange(0, Dims[1], 1))]
            V_d_numer_tot += np.nan_to_num(np.stack([x[0] for x in my_tuple], axis=1))
            moment_denom_tot += np.nan_to_num(np.stack([x[1] for x in my_tuple], axis=1))
            od_tot = np.nan_to_num(np.stack([x[2] for x in my_tuple], axis=1))
            column_ds["sub_col_Ze_%s_strat" % hyd_type][:, :, :] = np.stack([x[3] for x in my_tuple], axis=1)
            column_ds["sub_col_Vd_%s_strat" % hyd_type][:, :, :] = np.stack([x[4] for x in my_tuple], axis=1)
            column_ds["sub_col_sigma_d_%s_strat" % hyd_type][:, :, :] = np.stack([x[5] for x in my_tuple], axis=1)

        if "sub_col_Ze_tot_strat" in column_ds.variables.keys():
            column_ds["sub_col_Ze_tot_strat"] += column_ds["sub_col_Ze_%s_strat" % hyd_type].fillna(0)
        else:
            column_ds["sub_col_Ze_tot_strat"] = column_ds["sub_col_Ze_%s_strat" % hyd_type].fillna(0)

        #column_ds["sub_col_Ze_%s_strat" % hyd_type] = column_ds["sub_col_Ze_%s_strat" % hyd_type].where(
        #    column_ds["sub_col_Ze_%s_strat" % hyd_type] > 0)
        column_ds["sub_col_Ze_%s_strat" % hyd_type] = 10 * np.log10(column_ds["sub_col_Ze_%s_strat" % hyd_type])
        column_ds["sub_col_Ze_%s_strat" % hyd_type].attrs["long_name"] = \
            "Radar reflectivity factor from %s in stratiform clouds" % hyd_names_dict[hyd_type]
        column_ds["sub_col_Ze_%s_strat" % hyd_type].attrs["units"] = "dBZ"
        column_ds["sub_col_Vd_%s_strat" % hyd_type].attrs["long_name"] = \
            "Doppler velocity from %s in stratiform clouds" % hyd_names_dict[hyd_type]
        column_ds["sub_col_Vd_%s_strat" % hyd_type].attrs["units"] = "m s-1"
        column_ds["sub_col_sigma_d_%s_strat" % hyd_type].attrs["long_name"] = \
            "Spectral width from %s in stratiform clouds" % hyd_names_dict[hyd_type]
        column_ds["sub_col_sigma_d_%s_strat" % hyd_type].attrs["units"] = "m s-1"
        print("Generating stratiform radar moments for hydrometeor class %s" % hyd_type)
    column_ds["sub_col_Vd_tot_strat"] = xr.DataArray(-V_d_numer_tot / moment_denom_tot,
                                                     dims=column_ds["sub_col_Ze_tot_strat"].dims)
    for hyd_type in ["cl", "pl", "ci", "pi"]:
        if hyd_type == "cl":
            Vd_tot = column_ds["sub_col_Vd_tot_strat"].values
            _calc_sigma_d_liq = lambda x: _calc_sigma_d_tot_cl(
                x, fits_ds, instrument, model, total_hydrometeor, dD, Vd_tot)
            if parallel:
                sigma_d_numer = tt_bag.map(_calc_sigma_d_liq).compute()
            else:
                sigma_d_numer = [x for x in map(_calc_sigma_d_liq, np.arange(0, Dims[1], 1))]

            sigma_d_numer = np.stack(sigma_d_numer, axis=1)
            sigma_d_numer_tot = sigma_d_numer
        else:
            sub_q_array = column_ds["strat_q_subcolumns_%s" % hyd_type].values
            _calc_sigma = lambda x: _calc_sigma_d_tot(
                x, model, p_diam, v_tmp, fits_ds, total_hydrometeor, Vd_tot, sub_q_array, dD, beta_p)
            if parallel:
                sigma_d_numer = tt_bag.map(_calc_sigma).compute()
            else:
                sigma_d_numer = [x for x in map(_calc_sigma, np.arange(0, Dims[1], 1))]
            sigma_d_numer = np.stack(sigma_d_numer, axis=1)
            sigma_d_numer_tot += sigma_d_numer
    print(sigma_d_numer_tot.shape)
    column_ds["sub_col_sigma_d_tot_strat"] = xr.DataArray(np.sqrt(sigma_d_numer_tot / moment_denom_tot),
                                                          dims=column_ds["sub_col_Vd_tot_strat"].dims)
    kappa_ds = calc_radar_atm_attenuation(instrument, model)

    if OD_from_sfc:
        dz = np.diff(column_ds[z_field].values, axis=1, prepend=0)
        dz = np.tile(dz, (Dims[0], 1, 1))
        od_tot = np.cumsum(dz * od_tot, axis=1)
        atm_ext = np.cumsum(dz / 1e3 * kappa_ds.ds['kappa_att'].values, axis=2)
    else:
        dz = np.diff(column_ds[z_field].values, prepend=0)
        dz = np.tile(dz, (Dims[0], 1, 1))
        od_tot = np.flip(np.cumsum(np.flip(dz * od_tot, axis=1), axis=1), axis=2)
        atm_ext = np.flip(np.cumsum(np.flip(dz / 1e3 * kappa_ds.ds['kappa_att'].values, axis=1), axis=1), axis=2)

    column_ds['sub_col_Ze_att_tot_strat'] = \
        column_ds['sub_col_Ze_tot_strat'] * np.exp(-2 * od_tot) / 10**(2 * atm_ext / 10)
    column_ds['sub_col_Ze_tot_strat'] = column_ds['sub_col_Ze_tot_strat'].where(
        column_ds['sub_col_Ze_tot_strat'] > 0)
    column_ds['sub_col_Ze_att_tot_strat'] = column_ds['sub_col_Ze_att_tot_strat'].where(
        column_ds['sub_col_Ze_att_tot_strat'] > 0)
    column_ds['sub_col_Ze_tot_strat'] = 10 * np.log10(column_ds['sub_col_Ze_tot_strat'])
    column_ds['sub_col_Ze_att_tot_strat'] = 10 * np.log10(column_ds['sub_col_Ze_att_tot_strat'])
    column_ds['sub_col_Ze_att_tot_strat'].attrs["long_name"] = \
        "Radar reflectivity factor in stratiform clouds factoring in gaseous attenuation"
    column_ds['sub_col_Ze_att_tot_strat'].attrs["units"] = "dBZ"
    column_ds['sub_col_Ze_tot_strat'].attrs["long_name"] = \
        "Radar reflectivity factor in stratiform clouds"
    column_ds['sub_col_Ze_tot_strat'].attrs["units"] = "dBZ"
    column_ds['sub_col_Vd_tot_strat'].attrs["long_name"] = \
        "Doppler velocity in stratiform clouds"
    column_ds['sub_col_Vd_tot_strat'].attrs["units"] = "m s-1"
    column_ds['sub_col_sigma_d_tot_strat'].attrs["long_name"] = \
        "Spectral width in stratiform clouds"
    column_ds['sub_col_sigma_d_tot_strat'].attrs["units"] = "m s-1"
    model.ds = column_ds

    return model


def _calc_sigma_d_tot_cl(tt, fits_ds, instrument, model, total_hydrometeor, dD, Vd_tot):
    hyd_type = "cl"
    sigma_d_numer = np.zeros((model.num_subcolumns, total_hydrometeor.shape[2]))
    if tt % 50 == 0:
        print('Stratiform moment for class cl progress: %d/%d' % (tt, total_hydrometeor.shape[1]))
    p_diam = instrument.mie_table[hyd_type]["p_diam"].values
    num_diam = len(p_diam)
    Dims = Vd_tot.shape
    for k in range(Dims[2]):
        if np.all(total_hydrometeor.values[:, tt, k] == 0):
            continue
        N_0_tmp = fits_ds["N_0"].values[:, tt, k]
        N_0_tmp, d_diam_tmp = np.meshgrid(N_0_tmp, p_diam)
        lambda_tmp = fits_ds["lambda"].values[:, tt, k]
        lambda_tmp, d_diam_tmp = np.meshgrid(lambda_tmp, p_diam)
        mu_temp = fits_ds["mu"].values[:, tt, k] * np.ones_like(lambda_tmp)
        N_D = N_0_tmp * d_diam_tmp ** mu_temp * np.exp(-lambda_tmp * d_diam_tmp)
        Calc_tmp = np.tile(
            instrument.mie_table[hyd_type]["beta_p"].values, (model.num_subcolumns, 1)) * N_D.T
        v_tmp = model.vel_param_a[hyd_type] * p_diam ** model.vel_param_b[hyd_type]
        v_tmp = v_tmp.magnitude
        Calc_tmp2 = (v_tmp - np.tile(Vd_tot[:, tt, k], (num_diam, 1)).T) ** 2 * Calc_tmp
        sigma_d_numer[:, k] = (Calc_tmp2.sum(axis=1) / 2.0 + Calc_tmp2[:, 1:-1].sum(axis=1)) * dD

    return sigma_d_numer


def _calc_sigma_d_tot(tt, model, p_diam, v_tmp, fits_ds, total_hydrometeor, vd_tot, sub_q_array, dD, beta_p):
    Dims = vd_tot.shape
    sigma_d_numer = np.zeros((model.num_subcolumns, total_hydrometeor.shape[2]))

    num_diam = len(p_diam)
    mu = fits_ds["mu"].values.max()
    if tt % 50 == 0:
        print('Stratiform moment for class progress: %d/%d' % (tt, Dims[1]))
    for k in range(Dims[2]):
        if np.all(total_hydrometeor.values[:, tt, k] == 0):
            continue
        N_0_tmp = fits_ds["N_0"][:, tt, k].values.max(axis=0)
        lambda_tmp = fits_ds["lambda"][:, tt, k].values.max(axis=0)
        if np.isnan(N_0_tmp):
            continue
        N_D = []
        for i in range(model.num_subcolumns):
            N_D.append(N_0_tmp * p_diam ** mu * np.exp(-lambda_tmp * p_diam))
        N_D = np.stack(N_D, axis=1)
        Calc_tmp = np.tile(
            beta_p,
            (model.num_subcolumns, 1)) * N_D
        Calc_tmp2 = (v_tmp - np.tile(vd_tot[:, tt, k], (num_diam, 1)).T) ** 2 * Calc_tmp
        Calc_tmp2 = (Calc_tmp2.sum(axis=1) / 2. + Calc_tmp2[:, 1:-1].sum(axis=1)) * dD
        sigma_d_numer[:, k] = np.where(sub_q_array[:, tt, k] == 0, 0, Calc_tmp2)

    return sigma_d_numer


def _calculate_observables_liquid(tt, total_hydrometeor, N_0, lambdas, mu,
                                  alpha_p, beta_p, v_tmp, num_subcolumns, instrument, dD, p_diam):
    height_dims = N_0.shape[2]
    V_d_numer_tot = np.zeros((num_subcolumns, height_dims))
    V_d = np.zeros((num_subcolumns, height_dims))
    Ze = np.zeros_like(V_d)
    sigma_d = np.zeros_like(V_d)
    moment_denom_tot = np.zeros_like(V_d_numer_tot)
    od_tot = np.zeros_like(V_d_numer_tot)
    num_diam = len(p_diam)
    if tt % 50 == 0:
        print("Processing column %d" % tt)
    np.seterr(all="ignore")
    for k in range(height_dims):
        if np.all(total_hydrometeor.values[:, tt, k] == 0):
            continue

        N_0_tmp = np.squeeze(N_0[:, tt, k])
        lambda_tmp = np.squeeze(lambdas[:, tt, k])
        if all(np.isnan(x) for x in N_0_tmp):
            continue
        mu_temp = np.squeeze(mu[:, tt, k])
        N_D = []
        for i in range(num_subcolumns):
            N_D.append(N_0_tmp[i] * p_diam ** mu_temp[i] * np.exp(-lambda_tmp[i] * p_diam))
        N_D = np.stack(N_D, axis=0)

        Calc_tmp = beta_p * N_D
        tmp_od = np.trapz(alpha_p * N_D, dx=dD)
        moment_denom = np.trapz(Calc_tmp, dx=dD, axis=1).astype('float64')
        Ze[:, k] = \
            (moment_denom * instrument.wavelength ** 4) / (instrument.K_w * np.pi ** 5) * 1e-6

        Calc_tmp2 = v_tmp * Calc_tmp.astype('float64')
        V_d_numer = np.trapz(Calc_tmp2, dx=dD, axis=1)
        V_d[:, k] = -V_d_numer / moment_denom
        Calc_tmp2 = (v_tmp - np.tile(V_d[:, k], (num_diam, 1)).T) ** 2 * Calc_tmp
        sigma_d_numer = np.trapz(Calc_tmp2, dx=dD, axis=1)
        sigma_d[:, k] = np.sqrt(sigma_d_numer / moment_denom)
        V_d_numer_tot[:, k] += V_d_numer
        moment_denom_tot[:, k] += moment_denom
        od_tot[:, k] += tmp_od

    return V_d_numer_tot, moment_denom_tot, od_tot, Ze, V_d, sigma_d


def _calculate_other_observables(tt, total_hydrometeor, fits_ds, model, instrument, sub_q_array, hyd_type, dD):
    Dims = sub_q_array.shape
    if tt % 50 == 0:
        print('Stratiform moment for class %s progress: %d/%d' % (hyd_type, tt, Dims[1]))
    Ze = np.zeros((model.num_subcolumns, Dims[2]))
    V_d = np.zeros_like(Ze)
    sigma_d = np.zeros_like(Ze)
    V_d_numer_tot = np.zeros_like(Ze)
    moment_denom_tot = np.zeros_like(Ze)
    od_tot = np.zeros_like(Ze)
    for k in range(Dims[2]):
        if np.all(total_hydrometeor.values[:, tt, k] == 0):
            continue

        p_diam = instrument.mie_table[hyd_type]["p_diam"].values
        num_diam = len(p_diam)
        N_D = []
        for i in range(model.num_subcolumns):
            N_0_tmp = fits_ds["N_0"][i, tt, k].values
            lambda_tmp = fits_ds["lambda"][i, tt, k].values
            N_D.append(N_0_tmp * np.exp(-lambda_tmp * p_diam))
        N_D = np.stack(N_D, axis=0)
        Calc_tmp = np.tile(instrument.mie_table[hyd_type]["beta_p"].values,
                           (model.num_subcolumns, 1)) * N_D
        tmp_od = np.tile(
            instrument.mie_table[hyd_type]["alpha_p"].values, (model.num_subcolumns, 1)) * N_D
        tmp_od = np.trapz(tmp_od, dx=dD, axis=1)
        tmp_od = np.where(sub_q_array[:, tt, k] == 0, 0, tmp_od)
        moment_denom = np.trapz(Calc_tmp, dx=dD, axis=1)
        moment_denom = np.where(sub_q_array[:, tt, k] == 0, 0, moment_denom)
        Ze[:, k] = \
            (moment_denom * instrument.wavelength ** 4) / (instrument.K_w * np.pi ** 5) * 1e-6
        v_tmp = model.vel_param_a[hyd_type] * p_diam ** model.vel_param_b[hyd_type]
        v_tmp = v_tmp.magnitude
        Calc_tmp2 = Calc_tmp * v_tmp
        V_d_numer = np.trapz(Calc_tmp2, axis=1, dx=dD)
        V_d_numer = np.where(sub_q_array[:, tt, k] == 0, 0, V_d_numer)
        V_d[:, k] = -V_d_numer / moment_denom
        Calc_tmp2 = (v_tmp - np.tile(V_d[:, k], (num_diam, 1)).T) ** 2 * Calc_tmp
        Calc_tmp2 = np.trapz(Calc_tmp2, axis=1, dx=dD)
        sigma_d_numer = np.where(sub_q_array[:, tt, k] == 0, 0, Calc_tmp2)
        sigma_d[:, k] = np.sqrt(sigma_d_numer / moment_denom)
        V_d_numer_tot[:, k] += V_d_numer
        moment_denom_tot[:, k] += moment_denom
        od_tot[:, k] += tmp_od

    return V_d_numer_tot, moment_denom_tot, od_tot, Ze, V_d, sigma_d
