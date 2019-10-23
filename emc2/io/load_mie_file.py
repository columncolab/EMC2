import pandas as pd


def load_mie_file(filename):
    """
    Loads the Mie parameters from a file.

    Parameters
    ----------
    filename: str
        The name of the file storing the Mie scattering parameters

    Returns
    -------
    mie_df: pandas DataFrame
        The pandas DataFrame storing the Mie parameters
    """

    my_df = pd.read_csv(filename, delim_whitespace=True,
                        names=["wavelength", "p_diam", "size_parameter", "compre_real",
                               "compre_im", "scat_p", "alpha_p", "beta_p", "scat_eff",
                               "ext_eff", "backscat_p"])
    my_df["alpha_p"] = my_df["alpha_p"] * 1e-12
    my_df["beta_p"] = my_df["beta_p"] * 1e-12
    my_df["p_diam"] = 2e-6 * my_df["p_diam"]

    return my_df
