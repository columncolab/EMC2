import pandas as pd


def load_bulk_c6_file(filename):
    """
    Loads bulk ice or liquid scattering LUTs from a file
    (by default using the PSD used in the C6 collection).

    Parameters
    ----------
    filename: str
        The name of the file storing the bluk scattering parameters

    Returns
    -------
    my_df: xarray DataFrame
        The xarray DataFrame storing the scattering data, including
        descriptive metadata.
    """

    my_df = pd.read_csv(filename,
                        names=["r_e", "Q_scat", "Q_ext", "Q_back", "Q_back_cross",
                               "LDR", "lidar_ratio"])

    my_df = my_df.to_xarray()

    my_df["r_e"].attrs["units"] = "microns"
    my_df["r_e"].attrs["long_name"] = "Effective radius of hydrometeor class"
    my_df['r_e'].attrs["standard_name"] = "Effective radius"

    my_df["Q_scat"].attrs["units"] = "1"
    my_df["Q_scat"].attrs["long_name"] = "Bulk scattering efficiency"
    my_df['Q_scat'].attrs["standard_name"] = "Scattering efficiency"

    my_df["Q_ext"].attrs["units"] = "1"
    my_df["Q_ext"].attrs["long_name"] = "Bulk extinction efficiency"
    my_df['Q_ext'].attrs["standard_name"] = "Extinction efficiency"

    my_df["Q_back"].attrs["units"] = "sr^-1"
    my_df["Q_back"].attrs["long_name"] = "Bulk backscattering efficiency"
    my_df['Q_back'].attrs["standard_name"] = "Backscattering efficiency"

    my_df["Q_back_cross"].attrs["units"] = "sr^-1"
    my_df["Q_back_cross"].attrs["long_name"] = "Bulk cross-polar backscattering efficiency"
    my_df['Q_back_cross'].attrs["standard_name"] = "Cross-pol backcattering efficiency"

    my_df["LDR"].attrs["units"] = "1"
    my_df["LDR"].attrs["long_name"] = "Bulk Linear depolarization ratio"
    my_df['LDR'].attrs["standard_name"] = "Lienar depolarization ratio"

    my_df["lidar_ratio"].attrs["units"] = "sr"
    my_df["lidar_ratio"].attrs["long_name"] = "Bulk Lidar ratio"
    my_df['lidar_ratio'].attrs["standard_name"] = "Lidar ratio"

    return my_df
