import numpy as np

def calc_microwave_ref_index(wavelength, temperature=0):
    """
    Calculate the refractive index of water in the microwave spectrum (centimeter wavelengths).

    Parameters
    ----------
    wavelength: float
        Wavelength in cm
    temperature: float
        Temperature in Celsius

    Returns
    -------
    ref_index: complex
        The complex refractive index of water.
    """

    eta_inf = 5.27137 + 0.0216474 * temperature - 0.00131198 * temperature ** 2
    alpha = -16.8129 / (temperature + 273.) + 0.0609265
    lambda_s = 0.00033836 * np.exp(2513.98 / (temperature + 273.))
    sigma = 12.5664e8
    eta_s = 78.54 * (1.0 - 4.579e-3 * (temperature - 25.0) + 1.19e-5 * (temperature - 25.0) ** 2
                     - 2.8e-8 * (temperature - 25.0) ** 3)
    eta_prime = eta_inf + (
                (eta_s - eta_inf) * (1 + (lambda_s / wavelength) ** (1 - alpha) * np.sin(alpha * np.pi / 2))) / \
                (1 + 2 * (lambda_s / wavelength) ** (1 - alpha) * np.sin(alpha * np.pi / 2) + (
                            lambda_s / wavelength) ** (2 - 2 * alpha))
    eta_pprime = ((eta_s - eta_inf) * (lambda_s / wavelength) ** (1 - alpha) * np.cos(alpha * np.pi / 2)) / \
                 (1 + 2 * (lambda_s / wavelength) ** (1 - alpha) * np.sin(alpha * np.pi / 2) + (
                             lambda_s / wavelength) ** (2 - 2 * alpha))
    eta_pprime += sigma * wavelength / 18.8496e10

    n = np.sqrt((np.sqrt(eta_prime ** 2 + eta_pprime ** 2) + eta_prime) / 2)
    k = np.sqrt((np.sqrt(eta_prime ** 2 + eta_pprime ** 2) - eta_prime) / 2)
    return n + k * 1j


def calc_microwave_ref_index_ice(wavelength, temperature=0):
    """
    Calculate the refractive index of ice in the microwave spectrum (centimeter wavelengths).

    Parameters
    ----------
    wavelength: float
        Wavelength in cm
    temperature: float
        Temperature in Celsius

    Returns
    -------
    ref_index: complex
        The complex refractive index of ice.
    """
    c = 299792458.
    eta_prime = 2.1884 + 0.00091 * temperature
    eta_pprime = 0.00041 / (c / (wavelength * 1e-2))

    n = np.sqrt((np.sqrt(eta_prime ** 2 + eta_pprime ** 2) + eta_prime) / 2.)
    k = np.sqrt((np.sqrt(eta_prime ** 2 + eta_pprime ** 2) - eta_prime) / 2.)
    return n + k * 1j
