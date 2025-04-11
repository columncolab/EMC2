import xarray as xr
import numpy as np

from scipy.special import gamma



def set_vt_consts():
    """
    Set constant values for thermodynamic parameters for terminal velocity calculations
    consistent with the P3 implementation in E3SMv3 (via LUTs) where we assume 600 hPa,
    253 K for p and T.
    
    Returns
    =======
    pi: float
        Value of pi.
    g: float
        Gravity constant (m/s^2).
    p: float
        Air pressure (Pa).
    t: float
        Temperature (K).
    rho: float
        Air density (kg/m^3).
    mu: float
        Viscosity of air.
    dv: float
        Diffusivity of water vapor in air.
    dt: float
        Time step for collection (s).
    del0: float
        surface roughness for ice particles (see Mitchell and Heymsfield, 2005)
    c0: float
        surface roughness for ice particles (see Mitchell and Heymsfield, 2005)
    c1: float
        surface roughness for ice particles (see Mitchell and Heymsfield, 2005)
    c2: float
        surface roughness for ice particles (see Mitchell and Heymsfield, 2005)
    """
    pi  = 3.14159265359  #=acos(-1.)
    g   = 9.861                                       # gravity
    p   = 60000.                                      # air pressure (pa)
    t   = 253.15                                      # temp (K)
    rho = p / (287.15 * t)                            # air density (kg m-3)
    mu  = 1.496e-6 * t ** 1.5 / (t + 120.) / rho      # viscosity of air
    dv  = 8.794e-5 * t ** 1.81 / p                    # diffusivity of water vapor in air
    dt  = 10.                                         # time step for collection (s)
    del0 = 5.83                                       # surface roughness for ice particles (see MH2005)
    c0   = 0.6                                        # surface roughness for ice particles (see MH2005)
    c1   = 4./(del0**2*c0**0.5)                       # surface roughness for ice particles (see MH2005)
    c2   = del0**2/4.                                 # surface roughness for ice particles (see MH2005)
    
    return pi, g, p, t, rho, mu, dv, dt, del0, c0, c1, c2


def set_mD_AD_params_and_dcrit(bas=1.88, cs=0.0121, ds=1.9, bag=2.0, dg=3.0):
    """
    Set mass-diameter (m-D) and area-diameter (A-D) parameters and calculate the critical diameter.
    Default parameters are per E3SMv3 implementation but the function provides a basis for other
    unrimed m-D, A-D combinations

    Parameters
    ==========
    bas: float
        Exponent for the unrimed aggregates area-diameter relationship.
    cs: float
        Coefficient for the unrimed aggregates mass-diameter relationship (correction to BF95 from Hogan et al. 2012, JAMC)
    ds: float
        Exponent for the unrimed aggregates mass-diameter relationship.
    bag: float
        Coefficient for the graupel area-diameter relationship (assuming spheres).
    dg: float
        Exponent for the graupel mass-diameter relationship (assuming spheres).

    Returns
    =======
    params: dict
        A dictionary containing the following keys:
        - dcrit: float
            Critical diameter separating spherical ice from unrimed aggregate parameterization (eq. 8 in Morrison, 2012).
        - aas: float
            A-D coefficient for unrimed aggregates of side planes, bullets, et (assuming spheres).
        - bas: float
            Exponent for the unrimed aggregates area-diameter relationship.
        - cs: float
            Coefficient for the mass-diameter relationship.
        - ds: float
            Exponent for the mass-diameter relationship.
        - aag: float
            A coefficient for graupel A-D (assuming spheres).
        - bag: float
            Coefficient for the graupel area-diameter relationship.
        - dg: float
            Exponent for the graupel area-diameter relationship.
    """
    pi, g, p, t, rho, mu, dv, dt, _, _, _, _ = set_vt_consts()

    dcrit = (pi / (6.0 * cs) * 900.) ** (1.0 / (ds - 3.0))  # critical diameter separating spherical ice from unrimed
    aag = pi * 0.25  # a coefficient for graupel A-D (assuming spheres)
    aas = 0.2285 * 100. ** bas / (100.0 ** 2)  # A-D coefficient for unrimed aggregates of side planes, bullets, etc.

    if (pi / 4.0 * dcrit ** 2.0 < aas * dcrit ** bas):
        raise ValueError("area > area of solid ice sphere, unrimed")

    params = {
        "dcrit": dcrit,
        "aas": aas,
        "bas": bas,
        "cs": cs,
        "ds": ds,
        "aag": aag,
        "bag": bag,
        "dg": dg
    }

    return params


def set_ice_params_from_indices(i_Fr=None, i_rhor=None, Fr=None, crp=None):
    """
    Set the P3 ice parameters such as rimed fraction (Fr) and a,b,c,d, of m-D and A-D relationships
    m = a*D^b , A = c*D^d
    The resolved ice particle effective density is calculated following eq. 16 in Morrison, 2012)

    Parameters
    ==========
    i_Fr: int
        Index for rimed fraction. setting if `Fr` is None. Expected values are 1, 2, 3, or 4.
    i_rhor: int
        Index for rime density. setting if `crp` is None. Expected values are 1, 2, 3, 4, or 5.
    Fr: float
        Rimed fraction.
    crp: float
        Alpha for mD relationship in case of partially rimed particle (the rimed portion).

    Returns
    =======
    params: dict
        A dictionary containing the following keys:
        - aas: float
            A-D coefficient for unrimed aggregates of side planes, bullets, et (assuming spheres).
        - bas: float
            Exponent for the unrimed aggregates area-diameter relationship.
        - cs: float
            Coefficient for the mass-diameter relationship.
        - ds: float
            Exponent for the mass-diameter relationship.
        - aag: float
            A coefficient for graupel A-D (assuming spheres).
        - bag: float
            Coefficient for the graupel area-diameter relationship.
        - dg: float
            Exponent for the graupel area-diameter relationship.
        - Fr: float
            Rimed fraction.
        - crp: float
            Alpha for mD relationship in case of partially rimed particle (the rimed portion).
            Equal to the rho_r*pi/6 (rho_r - density of rimed portion)
        - rhodep: float
            Resolved density of the unrimed part of the particle (grown via vapor deposition; eq. 17)
        - cgp: float
            Resolved density of ice particle (considering riming fraction rimed portion density,
            critical D, etc.) (eq. 16 in Morrison).
        - csr: float
            Alpha for mD relationship in mass-dimension relationship for partially-rimed crystals (resolved particle)
        - dsr: float
            Exponent for mD relationship in mass-dimension relationship for partially-rimed crystals (resolved particle)
        - dcrit: float
            (Dth - eq. 8 in Morrison) Critical size separating solid ice (D < dcrit) and unrimed (aggregate) portion
            (rhoeff=rho_i=900, a=pi/4, b=3, c = (pi*900)/6, d=3.0 when D < dcrit). # use Simple Mie for scattering
        - dcrits: float
            (Dgr - eq. 15 in Morrison) Critical size between unrimed and partially rimed
            (rhoeff=variable, a=aas, b=bas, c=cs, d=ds when dcrit < D < dcrits termed "dense nonspherical ice" in Morrison, 2021). # use MG approximation here
        - dcritr: float
            (Dcr - eq. 14 in Morrison) Critical size between partially and fully rimed ice
            (rhoeff=variable, a=aag, b=bag, c=cg, d=dg when dcrits < D < dcritr)
            (rho=variable, a,b,c, and d variable per the fall speed function for  D > dcritr)  # use MG approximation here
    """
    pi, g, p, t, rho, mu, dv, dt, del0, c0, c1, c2 = set_vt_consts()
    params = set_mD_AD_params_and_dcrit()
    dcrit, aas, bas, cs, ds, aag, bag, dg = (
        params["dcrit"], params["aas"], params["bas"], params["cs"],
        params["ds"], params["aag"], params["bag"], params["dg"],
    )

    # Set rimed fraction
    if Fr is None:
        if i_Fr == 1:
            Fr = 0.
        elif i_Fr == 2:
            Fr = 0.333
        elif i_Fr == 3:
            Fr = 0.667
        elif i_Fr == 4:
            Fr = 1.

    # Now set crp (alpha for mD relationship in case of rimed particle)
    if crp is None:
        if i_rhor == 1:
            crp = 50.0
        elif i_rhor == 2:
            crp = 250.0
        elif i_rhor == 3:
            crp = 450.0
        elif i_rhor == 4:
            crp = 650.0
        elif i_rhor == 5:
            crp = 900.0
    crp *= pi / 6.0

    sxth = 1. / 6.0

    # Initial guess for the rime density.
    cgp = crp

    # Handle [i_rhor]
    # Case of no riming (Fr = 0), set dcrits and dcritr to arbitrary large values
    if Fr == 0.0:
        dcrits = 1e6
        dcritr = dcrits
        csr = cs
        dsr = ds
        rhodep = np.nan
    # Case of partial riming (Fr between 0 and 1)
    elif Fr < 1.0:
        while True:
            # Calculate critical sizes separating types of rimed ice
            dcrits = (cs / cgp) ** (1 / (dg - ds))
            dcritr = ((1.0 + Fr / (1.0 - Fr)) * cs / cgp) ** (1 / (dg - ds))
            csr = cs * (1.0 + Fr / (1.0 - Fr))
            dsr = ds

            # Get mean density of vapor deposition/aggregation grown ice
            rhodep = 1.0 / (dcritr - dcrits) * 6.0 * cs / (pi * (ds - 2)) * \
                     (dcritr**(ds - 2.0) - dcrits**(ds - 2.0))

            # Update rime density to weighted density of fully-rimed ice
            cgpold = cgp
            cgp = crp * Fr + rhodep * (1.0 - Fr) * pi * sxth

            # Convergence condition for the iterative calculation
            if abs((cgp - cgpold) / cgp) < 0.01:
                break
    # Case of complete riming (Fr=1.0)
    else:
        # Set threshold size between partially-rimed and fully-rimed ice as arbitrary large
        dcrits = (cs / cgp) ** (1 / (dg - ds))
        dcritr = 1e6
        csr = cgp
        dsr = dg
        rhodep = np.nan

    # Print for debugging: critical size and current critical size for rimed ice mass
    print(f'dcrit,dcrits,dcritr: {i_rhor}, {dcrit}, {dcrits}, {dcritr}')

    params.update({
        "Fr": Fr,
        "crp": crp,
        "rhodep": rhodep,
        "cgp": cgp,
        "csr": csr,
        "dsr": dsr,
        "dcrits": dcrits,
        "dcritr": dcritr
    })

    return params


def calc_mu_n0_from_lambda(lambda_in):
    """
    Calculate the Gamma PSD shape parameter (mu) and normalized n0 from the given lambda
    following E3SMv3 processing (see:
    /E3SM/components/eam/tools/create_p3_lookupTable/create_p3_lookupTable_1.f90-v4.1).

    Parameters
    ==========
    lambda_in: float
        Input slope parameter

    Returns
    =======
    mu_i: float
        Final Gamma PSD shape parameter per Fig. 3B in Heymsfield (2003, Part II)
    n0: float
        Normalized N0.
    """
    mu_i = 0.076 * (lambda_in / 100.) ** 0.8 - 2  # Calculate shape parameter (convert m-1 to cm-1)
    mu_i = np.maximum(mu_i, 0.)
    mu_i = np.minimum(mu_i, 6.)  # final Gamma PSD shape parameter
    no = lambda_in ** (mu_i + 1.) / (gamma(mu_i + 1.))  # Normalized n0

    return mu_i, n0


def calculate_fall_speed(d1, **args):
    """
    Calculate the fall speed of an ice particle based on its properties and environmental conditions.
    Using the parameterization of Mitchell and Heymsfield (2005)

    Parameters
    ==========
    d1: float
        ice particle diameter [m]
    Fr: float
        Rime mass fraction.
    cs: float
        Coefficient for unrimed mass-size relationship.
    ds: float
        Exponent for mass-size relationship.
    dg: float
        Exponent for mass-size relationship of rimed ice.
    bas: float
        Exponent for area-size relationship.
    aas: float
        Coefficient for area-size relationship.
    bag: float
        Exponent for area-size relationship of rimed ice.
    aag: float
        Coefficient for area-size relationship of rimed ice.
    crp: list of float
        List of rime densities.
    i_rhor: int
        Index for rime density in crp.
    rho: float
        Air density.
    mu: float
        Dynamic viscosity of air.
    g: float
        Acceleration due to gravity.
    sxth: float
        Sixth of a unit (1/6).
    pi: float
        Value of pi.
    c1: float
        Coefficient for drag calculation.
    c2: float
        Coefficient for drag calculation.

    Returns
    =======
    fall_speed: float
        Calculated fall speed of the particle.
    """
    pi, g, p, t, rho, mu, dv, dt, del0, c0, c1, c2 = set_vt_consts()
    params = set_ice_params_from_indices(**args)
    sxth = 1.0 / 6.0

    # Determine mass-size and projected area-size relationships for the given size (d1)
    if d1 <= params["dcrit"]:
        cs1 = pi * sxth * 900.0
        ds1 = 3.0
        bas1 = 2.0
        aas1 = pi / 4.0
    elif d1 <= params["dcrits"]:
        cs1 = params["cs"]
        ds1 = params["ds"]
        bas1 = params["bas"]
        aas1 = params["aas"] 
    elif d1 <= params["dcritr"]:
        cs1 = params["cgp"]
        ds1 = params["dg"]
        bas1 = params["bag"]
        aas1 = params["aag"]
    else:
        cs1 = params["csr"]
        ds1 = params["dsr"]
        if params["Fr"] == 0.0:
            aas1 = params["aas"]
            bas1 = params["bas"]
        else:
            # For projected area, keep bas1 constant, but modify aas1 according to rimed fraction
            bas1 = params["bas"]
            dum1 = params["aas"] * d1 ** params["bas"]
            dum2 = params["aag"] * d1 ** params["bag"]
            m1 = cs1 * d1**ds1
            m2 = params["cs"] * d1 ** params["ds"]
            m3 = params["cgp"] * d1 ** params["dg"]

            # Linearly interpolate based on particle mass
            dum3 = dum1 + (m1 - m2) * (dum2 - dum1) / (m3 - m2)
            aas1 = dum3 / (d1 ** params["bas"])
    if params["Fr"] == 0.0:
        params["rhodep"] = cs1 * d1 ** ds1 / ((pi / 6) * d1 ** 3)  # equivalent volume sphere
        rho_rm = np.nan
        rho_f = params["rhodep"]
    else:
        rho_rm = params["crp"] * 6 / pi
        rho_f = params["cgp"] * 6 / pi
    

    # Calculate fall speed for particle
    # Best number for fall speed calculation
    xx = 2.0 * cs1 * g * rho * d1**(ds1 + 2.0 - bas1) / (aas1 * (mu * rho)**2)

    # Coefficients for drag calculations
    a0 = 0.0
    b0 = 0.0
    b1 = c1 * xx**0.5 / (2.0 * ((1.0 + c1 * xx**0.5)**0.5 - 1.0)**2) - \
         a0 * b0 * xx**b0 / (c2 * ((1.0 + c1 * xx**0.5)**0.5 - 1.0)**2)
    a1 = (c2 * ((1.0 + c1 * xx**0.5)**0.5 - 1.0)**2 - a0 * xx**b0) / xx**b1

    # Fall speed in terms of drag terms
    fall_speed = a1 * mu**(1.0 - 2.0 * b1) * (2.0 * cs1 * g / (rho * aas1))**b1 * d1**(b1 * (ds1 - bas1 + 2.0) - 1.0)
    rho_d = params["rhodep"]

    return fall_speed, rho_f, rho_d, rho_rm

