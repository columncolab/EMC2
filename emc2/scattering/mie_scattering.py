import numpy as np
import xarray as xr

try:
    from PyMieScatt import MieQ_withDiameterRange
    PYMIESCAT_AVAILABLE = True
except ModuleNotFoundError:
    PYMIESCAT_AVAILABLE = False

from .ref_index import calc_microwave_ref_index_ice, calc_microwave_ref_index


def brandes(D_eq):
    """
    Brandes 2005 Drop shape relationship model.
    Implementation of the Brandes et. al. drop shape model given in [1]. This gives
    the ratio of the major to minor axis as a function of equivalent liquid spherical
    diameter.

    Parameters
    ----------
    D_eq: float or array_like
        Volume Equivalent Drop Diameter

    Returns
    -------
    axis_ratio: float
        The ratio of the semi minor to semi major axis.
    See Also
    tb : Thurai and Bringi DSR
    pb: Pruppacher and Beard DSR
    bc: Beard and Chuang DSR

    References
    ----------
    ..[1] Brandes, etl al. 2005: On the Influence of Assumed Drop Size Distribution Form
    on Radar-Retrieved Thunderstorm Microphysics. J. Appl. Meteor. Climatol., 45, 259-268.
    """
   
    return 0.9951 + 0.0251 * np.power(D_eq, 1) - 0.03644 * np.power(
        D_eq, 2
    ) + 0.005303 * np.power(
        D_eq, 3
    ) - .0002492 * np.power(
        D_eq, 4
    )

def scat_properties_water(diams, wavelength, temperature=0., pressure=1013.):
    """
    Calculate the scattering properties for a range of water spheres.

    Parameters
    ----------
    diams: array floats
        Particle diameter in meters
    wavelength: float
        Wavelength in cm
    temperature: float
        Temperature in degrees C
    pressure: float
        Pressure in hPa

    Returns
    -------
    qext: np.array
        Extinction efficency
    qsca: np.array
        Scattering efficiency
    qabs: np.array
        Absorption efficiency
    g: np.array
        Asymmetry parameter
    qpr: np.array
        Radiation pressure efficiency factor
    qratio: np.array
        The ratio of backscatter/scattering efficiency.
    """
    if PYMIESCAT_AVAILABLE is False:
        raise ModuleNotFoundError("PyMieScat needs to be installed in order to use this feature!")

    m = calc_microwave_ref_index(wavelength, temperature)
    nMedium = 1 + 1e-6 * (77.6 * pressure /
                          (temperature + 273.15 + 3.75e-5 * pressure / (temperature + 273.15)**2))
    diams, qext, qsca, qabs, g, qpr, qback, qratio = MieQ_withDiameterRange(
        m, wavelength * 1e7, nMedium=nMedium, nd=len(diams),
        diameterRange=(diams.min() * 1000., diams.max() * 1000.))

    my_dict = {'ext_eff': qext, 'scat_eff': qsca, 'qabs': qabs,
               'g': g, 'qpr': qpr, 'backscat_eff': qback, 'qratio': qratio}
    my_dict['alpha_p'] = my_dict['ext_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['beta_p'] = my_dict['scat_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['scat_p'] = my_dict['backscat_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['compre_real'] = m.real * np.ones_like(qext) / 1.0003
    my_dict['compre_im'] = m.imag * np.ones_like(qext) / 1.0003
    my_dict['size_parameter'] = np.pi * diams / (wavelength * 1e4)
    my_dict['p_diam'] = diams * 1e-9
    my_dict['wavelength'] = wavelength * 1e4 * np.ones_like(qext)
    my_df = xr.Dataset(my_dict)

    my_df["wavelength"].attrs["units"] = "microns"
    my_df["wavelength"].attrs["long_name"] = "Wavelength of beam"
    my_df["wavelength"].attrs["standard_name"] = "wavelength"

    my_df["p_diam"].attrs["units"] = "meters"
    my_df["p_diam"].attrs["long_name"] = "Diameter of particle"
    my_df['p_diam'].attrs["standard_name"] = "Diameter"

    my_df["size_parameter"].attrs["units"] = "1"
    my_df["size_parameter"].attrs["long_name"] = "Size parameter (pi*diameter / wavelength)"
    my_df['size_parameter'].attrs["standard_name"] = "Size parameter"

    my_df["compre_real"].attrs["units"] = "1"
    my_df["compre_real"].attrs["long_name"] = ("Complex refractive index of the sphere divided " +
                                               "by the real index of the medium (real part)")
    my_df['compre_real'].attrs["standard_name"] = "Complex_over_real_Re"

    my_df["compre_im"].attrs["units"] = "1"
    my_df["compre_im"].attrs["long_name"] = ("Complex refractive index of the sphere divided " +
                                             "by the real index of the medium (imaginary part)")
    my_df['compre_im'].attrs["standard_name"] = "Complex_over_real_Im"

    my_df["scat_p"].attrs["units"] = "meters^2"
    my_df["scat_p"].attrs["long_name"] = "Forward scattering cross section"
    my_df["scat_p"].attrs["standard_name"] = "Scat_cross_section_fwd"

    my_df["alpha_p"].attrs["units"] = "meters^2"
    my_df["alpha_p"].attrs["long_name"] = "Back scattering cross section"
    my_df["alpha_p"].attrs["standard_name"] = "Scat_cross_section_back"

    my_df["beta_p"].attrs["units"] = "meters^2"
    my_df["beta_p"].attrs["long_name"] = "Extinction cross section"
    my_df["beta_p"].attrs["standard_name"] = "Ext_cross_section"

    my_df["scat_eff"].attrs["units"] = "1"
    my_df["scat_eff"].attrs["long_name"] = "Forward scattering efficiency"
    my_df["scat_eff"].attrs["standard_name"] = "Scattering_efficiency"

    my_df["ext_eff"].attrs["units"] = "1"
    my_df["ext_eff"].attrs["long_name"] = "Extinction efficiency"
    my_df["ext_eff"].attrs["standard_name"] = "Extinction_efficiency"

    my_df["backscat_eff"].attrs["units"] = "1"
    my_df["backscat_eff"].attrs["long_name"] = "Backscattering efficiency"
    my_df["backscat_eff"].attrs["standard_name"] = "Backscattering_efficiency"

    return my_df


def scat_properties_ice(diams, wavelength, temperature=0., pressure=1013., rho_d=0.5):
    """
    Calculate the scattering properties for a range of ice spheres.

    Parameters
    ----------
    diams: array floats
        Particle diameter in microns
    wavelength: float
        Wavelength in cm
    temperature: float
        Temperature in degrees C
    pressure: float
        Pressure in hPa
    rho_d: float
        Effective density of ice in g cm-3.

    Returns
    -------
    qext: np.array
        Extinction efficency
    qsca: np.array
        Scattering efficiency
    qabs: np.array
        Absorption efficiency
    g: np.array
        Asymmetry parameter
    qpr: np.array
        Radiation pressure efficiency factor
    qratio: np.array
        The ratio of backscatter/scattering efficiency.
    """
    if PYMIESCAT_AVAILABLE is False:
        raise ModuleNotFoundError("PyMieScat needs to be installed in order to use this feature!")
    m = calc_microwave_ref_index_ice(wavelength, temperature, rho_d)
    nMedium = 1 + 1e-6 * (77.6 * pressure /
                          (temperature + 273.15 + 3.75e-5 * pressure / (temperature + 273.15) ** 2))
    diams, qext, qsca, qabs, g, qpr, qback, qratio = MieQ_withDiameterRange(
        m, wavelength * 1e7, nMedium=nMedium,
        nd=len(diams), diameterRange=(diams.min() * 1000., diams.max() * 1000.))

    my_dict = {'ext_eff': qext, 'scat_eff': qsca, 'qabs': qabs,
               'g': g, 'qpr': qpr, 'backscat_eff': qback, 'qratio': qratio}
    my_dict['alpha_p'] = my_dict['ext_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['beta_p'] = my_dict['backscat_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['scat_p'] = my_dict['scat_eff'] * np.pi / 4 * diams ** 2 * 1e-18
    my_dict['compre_real'] = m.real * np.ones_like(qext) / 1.0003
    my_dict['compre_im'] = m.imag * np.ones_like(qext) / 1.0003
    my_dict['size_parameter'] = np.pi * diams / (wavelength * 1e4)
    my_dict['p_diam'] = diams * 1e-9
    my_dict['wavelength'] = wavelength * 1e4 * np.ones_like(qext)
    my_df = xr.Dataset(my_dict)

    my_df["wavelength"].attrs["units"] = "microns"
    my_df["wavelength"].attrs["long_name"] = "Wavelength of beam"
    my_df["wavelength"].attrs["standard_name"] = "wavelength"

    my_df["p_diam"].attrs["units"] = "meters"
    my_df["p_diam"].attrs["long_name"] = "Diameter of particle"
    my_df['p_diam'].attrs["standard_name"] = "Diameter"

    my_df["size_parameter"].attrs["units"] = "1"
    my_df["size_parameter"].attrs["long_name"] = "Size parameter (pi*diameter / wavelength)"
    my_df['size_parameter'].attrs["standard_name"] = "Size parameter"

    my_df["compre_real"].attrs["units"] = "1"
    my_df["compre_real"].attrs["long_name"] = ("Complex refractive index of the sphere divided " +
                                               "by the real index of the medium (real part)")
    my_df['compre_real'].attrs["standard_name"] = "Complex_over_real_Re"

    my_df["compre_im"].attrs["units"] = "1"
    my_df["compre_im"].attrs["long_name"] = ("Complex refractive index of the sphere divided " +
                                             "by the real index of the medium (imaginary part)")
    my_df['compre_im'].attrs["standard_name"] = "Complex_over_real_Im"

    my_df["scat_p"].attrs["units"] = "microns^2"
    my_df["scat_p"].attrs["long_name"] = "Forward scattering cross section"
    my_df["scat_p"].attrs["standard_name"] = "Scat_cross_section_fwd"

    my_df["alpha_p"].attrs["units"] = "meters^2"
    my_df["alpha_p"].attrs["long_name"] = "Back scattering cross section"
    my_df["alpha_p"].attrs["standard_name"] = "Scat_cross_section_back"

    my_df["beta_p"].attrs["units"] = "meters^2"
    my_df["beta_p"].attrs["long_name"] = "Extinction cross section"
    my_df["beta_p"].attrs["standard_name"] = "Ext_cross_section"

    my_df["scat_eff"].attrs["units"] = "1"
    my_df["scat_eff"].attrs["long_name"] = "Forward scattering efficiency"
    my_df["scat_eff"].attrs["standard_name"] = "Scattering_efficiency"

    my_df["ext_eff"].attrs["units"] = "1"
    my_df["ext_eff"].attrs["long_name"] = "Extinction efficiency"
    my_df["ext_eff"].attrs["standard_name"] = "Extinction_efficiency"

    my_df["backscat_eff"].attrs["units"] = "1"
    my_df["backscat_eff"].attrs["long_name"] = "Backscattering efficiency"
    my_df["backscat_eff"].attrs["standard_name"] = "Backscattering_efficiency"

    return my_df
