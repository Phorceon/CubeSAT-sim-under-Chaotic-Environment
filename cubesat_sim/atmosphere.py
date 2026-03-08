"""
US Standard Atmosphere 1976 — thermospheric extension for LEO (120-600 km).

Provides density, temperature, and mean molecular weight as functions of
geometric altitude.  Values are from the 1976 US Standard Atmosphere tables
(Weast, 1969) as referenced by Shao et al. (2017).

Below the tabulated range the lowest entry is returned; above, the highest.
Between entries, log-linear interpolation is used for density (exponential
profile) and linear interpolation for temperature and molecular weight.
"""

import numpy as np

# Tabulated data: (altitude_km, density_kg_m3, temperature_K, mol_weight_g_mol)
# Source: US Standard Atmosphere 1976, thermospheric extension
_TABLE = np.array([
    # alt(km)   rho(kg/m^3)       T(K)     M(g/mol)
    # Molecular weights are number-density-weighted means for species mix
    # (O, N2, O2, He, H) from the 1976 US Standard Atmosphere extension.
    [120.0,     2.222e-8,         360.0,    26.98],
    [130.0,     8.152e-9,         469.0,    26.62],
    [140.0,     3.831e-9,         559.6,    25.94],
    [150.0,     2.076e-9,         634.4,    24.89],
    [160.0,     1.233e-9,         696.3,    23.56],
    [170.0,     7.815e-10,        747.4,    22.09],
    [180.0,     5.194e-10,        790.1,    20.56],
    [190.0,     3.581e-10,        825.2,    19.08],
    [200.0,     2.541e-10,        854.6,    21.30],
    [220.0,     1.367e-10,        899.0,    20.18],
    [240.0,     7.858e-11,        929.7,    19.19],
    [260.0,     4.742e-11,        950.4,    18.43],
    [280.0,     2.971e-11,        965.0,    17.73],
    [300.0,     1.916e-11,        975.2,    17.07],
    [320.0,     1.264e-11,        982.4,    16.63],
    [340.0,     8.503e-12,        987.5,    16.30],
    [360.0,     5.805e-12,        991.2,    16.10],
    [380.0,     4.013e-12,        993.8,    16.00],
    [400.0,     2.803e-12,        995.8,    15.98],
    [420.0,     1.973e-12,        997.3,    15.80],
    [440.0,     1.400e-12,        998.4,    15.50],
    [460.0,     9.971e-13,        999.2,    15.10],
    [480.0,     7.132e-13,        999.7,    14.70],
    [500.0,     5.215e-13,       1000.0,    14.33],
    [520.0,     3.831e-13,       1000.3,    13.80],
    [540.0,     2.813e-13,       1000.5,    13.30],
    [560.0,     2.081e-13,       1000.7,    12.85],
    [580.0,     1.546e-13,       1000.8,    12.43],
    [600.0,     1.156e-13,       1000.9,    12.04],
])

_ALT = _TABLE[:, 0]
_RHO = _TABLE[:, 1]
_TEMP = _TABLE[:, 2]
_MOL = _TABLE[:, 3]


def atmosphere(altitude_km: float) -> tuple[float, float, float]:
    """
    Return atmospheric properties at a given geometric altitude.

    Parameters
    ----------
    altitude_km : float
        Geometric altitude above sea level [km].

    Returns
    -------
    rho : float
        Atmospheric density [kg/m^3].
    T : float
        Temperature [K].
    M_mol : float
        Mean molecular weight [g/mol].
    """
    if altitude_km <= _ALT[0]:
        return _RHO[0], _TEMP[0], _MOL[0]
    if altitude_km >= _ALT[-1]:
        return _RHO[-1], _TEMP[-1], _MOL[-1]

    idx = np.searchsorted(_ALT, altitude_km) - 1
    frac = (altitude_km - _ALT[idx]) / (_ALT[idx + 1] - _ALT[idx])

    # Log-linear interpolation for density (exponential atmosphere)
    log_rho = np.log(_RHO[idx]) + frac * (np.log(_RHO[idx + 1]) - np.log(_RHO[idx]))
    rho = np.exp(log_rho)

    T = _TEMP[idx] + frac * (_TEMP[idx + 1] - _TEMP[idx])
    M_mol = _MOL[idx] + frac * (_MOL[idx + 1] - _MOL[idx])

    return rho, T, M_mol
