from act.io.armfiles import read_netcdf


def load_arm_file(filename, **kwargs):
    """
    Loads an ARM-compliant netCDF file.

    Parameters
    ----------
    filename: str
       The name of the file to load.

    Additional keyword arguments are passed into :py:func:`act.io.armfiles.read_netcdf`

    Returns
    -------
    ds: ACT dataset
        The xarray dataset containing the file data.
    """

    return read_netcdf(filename, **kwargs)
