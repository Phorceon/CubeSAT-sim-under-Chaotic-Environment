"""
Free-molecular-flow aerodynamic model (Tsien, 1946 / Schaaf-Chambre).

Implements Eq. 1-2 from Shao et al. (2017): pressure, shear, lift, and
drag on a flat plate in rarefied gas, using the kinetic-theory formulation.
"""

import numpy as np
from math import erf, sqrt, exp, pi

from .constants import SIGMA_1, SIGMA_2, R_UNIVERSAL
from .atmosphere import atmosphere


def _molecular_speed_ratio(V_sat: float, T_gas: float,
                           M_mol_g: float) -> float:
    """
    Molecular speed ratio  S = V_sat / sqrt(2 * R_specific * T_gas).

    Parameters
    ----------
    V_sat : float   Satellite speed relative to atmosphere [m/s].
    T_gas : float   Gas temperature [K].
    M_mol_g : float Mean molecular weight [g/mol].
    """
    R_specific = R_UNIVERSAL / M_mol_g          # [J/(kg·K)]
    return V_sat / sqrt(2.0 * R_specific * T_gas)


def pressure_and_shear(theta: float, V_sat: float,
                       rho_gas: float, T_gas: float,
                       M_mol_g: float,
                       T_surface: float | None = None
                       ) -> tuple[float, float]:
    """
    Normal pressure *p* and tangential shear *tau* per unit area on a flat
    plate at incidence angle *theta* to the free-stream (Eq. 1).

    Parameters
    ----------
    theta : float
        Angle between incident flow and plate surface [rad].  0 = edge-on.
    V_sat : float
        Satellite velocity relative to atmosphere [m/s].
    rho_gas : float
        Atmospheric mass density [kg/m^3].
    T_gas : float
        Atmospheric temperature [K].
    M_mol_g : float
        Mean molecular weight [g/mol].
    T_surface : float or None
        Plate surface temperature [K].  ``None`` ⇒ T_surface = T_gas
        (large thermal conductivity assumption per paper).

    Returns
    -------
    p : float   Pressure (normal to plate) per unit area [Pa].
    tau : float Shear (tangential to plate) per unit area [Pa].
                Negative means opposing the tangential flow component.
    """
    if T_surface is None:
        T_surface = T_gas

    S = _molecular_speed_ratio(V_sat, T_gas, M_mol_g)
    s = S * np.sin(theta)
    cos_theta = np.cos(theta)

    Tt_Ts = T_gas / T_surface       # ratio (=1 when T_gas==T_surface)
    sqrt_TtTs = sqrt(Tt_Ts)

    s2 = s * s
    exp_s2 = exp(-s2)
    erf_s = erf(s)
    sqrt_pi = sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi

    # -- Pressure (Eq. 1, first expression) --------------------------------
    bracket_exp = ((2.0 - SIGMA_2) * inv_sqrt_pi * s
                   + 0.5 * SIGMA_2 * sqrt_TtTs)

    bracket_erf = ((2.0 - SIGMA_2) * (s2 + 0.5)
                   + 0.5 * SIGMA_2 * sqrt_pi * sqrt_TtTs * s)

    p = (rho_gas * V_sat**2) / (2.0 * S**2) * (
        bracket_exp * exp_s2
        + bracket_erf * (1.0 + erf_s)
    )

    # -- Shear (Eq. 1, second expression) ----------------------------------
    # The Schaaf-Chambre / Tsien formulation gives the tangential force on
    # the plate surface in the direction of the flow's tangential component.
    # This is a positive quantity representing momentum transfer from gas
    # to plate.  The paper's rendered equation carries a leading minus sign
    # (force on the gas); here we use the positive convention consistent
    # with the Eq. 2 decomposition where shear adds to drag.
    tau = (SIGMA_1 * rho_gas * V_sat**2 * cos_theta) / (2.0 * sqrt_pi * S) * (
        exp_s2 + sqrt_pi * s * (1.0 + erf_s)
    )

    return p, tau


def lift_drag_per_unit_area(theta: float, V_sat: float,
                            rho_gas: float, T_gas: float,
                            M_mol_g: float,
                            T_surface: float | None = None
                            ) -> tuple[float, float]:
    """
    Aerodynamic lift and drag **per unit plate area** (Eq. 2).

    Returns
    -------
    Fl : float  Lift per unit area [N/m^2] (perpendicular to flow).
    Fd : float  Drag per unit area [N/m^2] (opposing flow).
    """
    p, tau = pressure_and_shear(theta, V_sat, rho_gas, T_gas,
                                M_mol_g, T_surface)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    Fl = p * cos_t - tau * sin_t
    Fd = p * sin_t + tau * cos_t
    return Fl, Fd


def aero_acceleration(theta: float, altitude_km: float,
                      V_sat: float,
                      K_actuator: float,
                      m_sat: float,
                      use_lift: bool = True) -> float:
    """
    Differential acceleration magnitude from a plate group (Eq. 3).

    Parameters
    ----------
    theta : float       Plate angle of attack [rad].
    altitude_km : float Orbital altitude [km].
    V_sat : float       Satellite velocity [m/s].
    K_actuator : float  Combined actuator coefficient  N_n * k_diff_n * A_n.
    m_sat : float       Satellite mass [kg].
    use_lift : bool     True → acceleration from lift;  False → from drag.

    Returns
    -------
    accel : float       Signed differential acceleration [m/s^2].
    """
    rho, T, M_mol = atmosphere(altitude_km)
    Fl, Fd = lift_drag_per_unit_area(theta, V_sat, rho, T, M_mol)
    force_per_area = Fl if use_lift else Fd
    return K_actuator * force_per_area / m_sat
