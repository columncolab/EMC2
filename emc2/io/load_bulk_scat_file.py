import pandas as pd


def load_bulk_scat_file(filename, param_type="C6"):
    """
    Loads bulk ice or liquid scattering LUTs from a file
    (by default using the PSD used in the C6 collection).

    Parameters
    ----------
    filename: str
        The name of the file storing the bluk scattering parameters
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

    if param_type == "C6":
        names = ["r_e", "Q_scat", "Q_ext", "Q_back", "Q_back_cross", "LDR", "lidar_ratio"]
    elif param_type == "mDAD":
        names = ["D_e", "Q_scat", "Q_ext", "Q_back", "lidar_ratio"]

    my_df = pd.read_csv(filename, names=names)

    my_df = my_df.to_xarray()

    if param_type == "C6":
        my_df["r_e"].attrs["units"] = "microns"
        my_df["r_e"].attrs["long_name"] = "Effective radius of hydrometeor class"
        my_df['r_e'].attrs["standard_name"] = "Effective radius"

        my_df["Q_back_cross"].attrs["units"] = "sr^-1"
        my_df["Q_back_cross"].attrs["long_name"] = "Bulk cross-polar backscattering efficiency"
        my_df['Q_back_cross'].attrs["standard_name"] = "Cross-pol backcattering efficiency"

        my_df["LDR"].attrs["units"] = "1"
        my_df["LDR"].attrs["long_name"] = "Bulk Linear depolarization ratio"
        my_df['LDR'].attrs["standard_name"] = "Lienar depolarization ratio"
    else:
        my_df["D_e"].attrs["units"] = "microns"
        my_df["D_e"].attrs["long_name"] = "Effective diameter of hydrometeor class"
        my_df['D_e'].attrs["standard_name"] = "Effective diameter"

        my_df["r_e"] = my_df["D_e"].copy() / 2.
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

    my_df["lidar_ratio"].attrs["units"] = "sr"
    my_df["lidar_ratio"].attrs["long_name"] = "Bulk Lidar ratio"
    my_df['lidar_ratio'].attrs["standard_name"] = "Lidar ratio"

    return my_df
