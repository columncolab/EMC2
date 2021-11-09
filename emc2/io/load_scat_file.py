import pandas as pd


def load_scat_file(filename, is_radar, param_type="C6"):
    """
    Loads ice scattering LUTs from a file.
    (by default using the PSD used in the C6 collection).

    Parameters
    ----------
    filename: str
        The name of the file storing the scattering parameters
    is_radar: bool
        If True, the first LUT column is treated as the frequency,
        otherwise, wavelength.
    param_type: str
        parameterization type:
        C6 - C6 collection based on Yang et al., JAS, 2013 (as used in the GISS ModelE3).
        mDAD - equivalent V/A spheres after implementing m-D and A-D parameterizations (as used in CESM and E3SM).

    Returns
    -------
    my_df: xarray.Dataset
        The xarray Dataset storing the scattering data, including
        descriptive metadata.
    """
    if is_radar is True:
        first_input = "frequency"
    else:
        first_input = "wavelength"
    if param_type == "C6":
        names = [first_input, "p_diam", "p_diam_eq_A", "p_diam_eq_V",
                 "V", "A", "beta_p", "scat_p", "alpha_p", "beta_p_cross"]
    elif param_type == "mDAD":
        names = [first_input, "p_diam", "p_diam_eq_A", "p_diam_eq_V", "p_diam_eq_VdivA",
                 "V", "A", "beta_p", "scat_p", "alpha_p"]
    my_df = pd.read_csv(filename, names=names)

    my_df["alpha_p"] = my_df["alpha_p"] * 1e-12
    my_df["beta_p"] = my_df["beta_p"] * 1e-12
    my_df["scat_p"] = my_df["scat_p"] * 1e-12
    my_df["p_diam"] = 1e-6 * my_df["p_diam"]
    my_df["p_diam_eq_A"] = 1e-6 * my_df["p_diam_eq_A"]
    my_df["p_diam_eq_V"] = 1e-6 * my_df["p_diam_eq_V"]
    if param_type == "C6":
        my_df["beta_p_cross"] = my_df["beta_p_cross"] * 1e-12
    if param_type == "mDAD":
        my_df["p_diam_eq_VdivA"] = 1e-6 * my_df["p_diam_eq_VdivA"]
    my_df["A"] = my_df["A"] * 1e-12
    my_df["V"] = my_df["V"] * 1e-18
    my_df = my_df.to_xarray()

    if is_radar is True:
        my_df["frequency"].attrs["units"] = "GHz"
        my_df["frequency"].attrs["long_name"] = "Pulse frequency"
        my_df["frequency"].attrs["standard_name"] = "Frequency"
    else:
        my_df["wavelength"].attrs["units"] = "microns"
        my_df["wavelength"].attrs["long_name"] = "Wavelength of beam"
        my_df["wavelength"].attrs["standard_name"] = "Wavelength"

    my_df["p_diam"].attrs["units"] = "meters"
    my_df["p_diam"].attrs["long_name"] = "Maximum dimension of the particle"
    my_df['p_diam'].attrs["standard_name"] = "Maximum dimension"

    my_df["p_diam_eq_A"].attrs["units"] = "meters"
    my_df["p_diam_eq_A"].attrs["long_name"] = "Diameter of equivalent projected area sphere"
    my_df['p_diam_eq_A'].attrs["standard_name"] = "Diameter of equivalent A sphere"

    my_df["p_diam_eq_V"].attrs["units"] = "meters"
    my_df["p_diam_eq_V"].attrs["long_name"] = "Diameter of equivalent volume sphere"
    my_df['p_diam_eq_V'].attrs["standard_name"] = "Diameter of equivalent V sphere"

    if param_type == "C6":
        my_df["beta_p_cross"].attrs["units"] = "meters^2"
        my_df["beta_p_cross"].attrs["long_name"] = "Cross-polar backscattering cross section"
        my_df["beta_p_cross"].attrs["standard_name"] = "Scat_cross_section_back_crosspol"
    if param_type == "mDAD":
        my_df["p_diam_eq_VdivA"].attrs["units"] = "meters"
        my_df["p_diam_eq_VdivA"].attrs["long_name"] = "Diameter of equivalent V/A sphere"
        my_df['p_diam_eq_VdivA'].attrs["standard_name"] = "Diameter of equivalent V/A sphere"

    my_df["A"].attrs["units"] = "meters^2"
    my_df["A"].attrs["long_name"] = "Projected area of particle"
    my_df['A'].attrs["standard_name"] = "Projected area"

    my_df["V"].attrs["units"] = "meters^3"
    my_df["V"].attrs["long_name"] = "Particle volume"
    my_df['V'].attrs["standard_name"] = "Volume"

    my_df["scat_p"].attrs["units"] = "microns^2"
    my_df["scat_p"].attrs["long_name"] = "Scattering cross section"
    my_df["scat_p"].attrs["standard_name"] = "Scat_cross_section"

    my_df["beta_p"].attrs["units"] = "meters^2"
    my_df["beta_p"].attrs["long_name"] = "Backscattering cross section"
    my_df["beta_p"].attrs["standard_name"] = "Scat_cross_section_back"

    my_df["alpha_p"].attrs["units"] = "meters^2"
    my_df["alpha_p"].attrs["long_name"] = "Extinction cross section"
    my_df["alpha_p"].attrs["standard_name"] = "Ext_cross_section"

    return my_df
