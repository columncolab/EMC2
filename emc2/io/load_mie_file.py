import pandas as pd
import numpy as np


def load_mie_file(filename):
    """
    Loads the Mie parameters from a file.

    Parameters
    ----------
    filename: str
        The name of the file storing the Mie scattering parameters

    Returns
    -------
    my_df: xarray.Dataset
        The xarray Dataset storing the Mie parameters, including
        descriptive metadata.
    """

    my_df = pd.read_csv(filename, delim_whitespace=True,
                        names=["wavelength", "p_diam", "size_parameter", "compre_real",
                               "compre_im", "scat_p", "alpha_p", "beta_p", "scat_eff",
                               "ext_eff", "backscat_eff"])

    my_df["alpha_p"] = my_df["alpha_p"] * 1e-12
    my_df["beta_p"] = my_df["beta_p"] * 1e-12 / (4 * np.pi)
    my_df["scat_p"] = my_df["scat_p"] * 1e-12
    my_df["p_diam"] = 2e-6 * my_df["p_diam"]
    my_df["backscat_eff"] = my_df["backscat_eff"] / (4 * np.pi)
    my_df = my_df.to_xarray()

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
    my_df["scat_p"].attrs["long_name"] = "scattering cross section"
    my_df["scat_p"].attrs["standard_name"] = "Scat_cross_section"

    my_df["beta_p"].attrs["units"] = "meters^2"
    my_df["beta_p"].attrs["long_name"] = "Back scattering cross section"
    my_df["beta_p"].attrs["standard_name"] = "Scat_cross_section_back"

    my_df["alpha_p"].attrs["units"] = "meters^2"
    my_df["alpha_p"].attrs["long_name"] = "Extinction cross section"
    my_df["alpha_p"].attrs["standard_name"] = "Ext_cross_section"

    my_df["scat_eff"].attrs["units"] = "1"
    my_df["scat_eff"].attrs["long_name"] = "scattering efficiency"
    my_df["scat_eff"].attrs["standard_name"] = "Scattering_efficiency"

    my_df["ext_eff"].attrs["units"] = "1"
    my_df["ext_eff"].attrs["long_name"] = "Extinction efficiency"
    my_df["ext_eff"].attrs["standard_name"] = "Extinction_efficiency"

    my_df["backscat_eff"].attrs["units"] = "sr^-1"
    my_df["backscat_eff"].attrs["long_name"] = "Backscattering efficiency"
    my_df["backscat_eff"].attrs["standard_name"] = "Backscattering_efficiency"

    return my_df
