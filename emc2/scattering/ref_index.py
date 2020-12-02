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
    c = 299792458.
    nu = c / (wavelength * 1e-2) / 1e9
    a = np.array([5.7230e0, 2.2379e-2, -7.1237e-4, 5.0478e0, -7.0315e-2,
                  6.0059e-4, 3.6143e0, 2.8841e-2, 1.3652e-1, 1.4825e-3,
                  2.4166e-4])
    eta_s = (3.70886e4 - 8.2168e1 * temperature) / (4.21854e2 + temperature)
    eta_1 = a[0] + a[1] * temperature + a[2] * temperature**2
    nu_1 = (45. + temperature)/(a[3] + a[4] * temperature + a[5] * temperature ** 2)
    eta_inf = a[6] + a[7] * temperature
    nu_2 = (45. + temperature)/(a[8] + a[9] * temperature + a[10] * temperature ** 2)
    eta = (eta_s - eta_1) / (1 + nu/nu_1 * 1j) + (eta_1 - eta_inf) / (1 + nu/nu_2 * 1j) + eta_inf
    eta_prime = eta.real
    eta_pprime = eta.imag
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
