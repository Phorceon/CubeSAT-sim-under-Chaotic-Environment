"""
Actuator model: 5-plate configuration with numerical angle inversion.

Implements Eq. 3 (differential acceleration from plate groups) and provides
both numerical root-finding inversion and the paper's fitting formulae
(Eq. 15) for mapping desired accelerations back to plate angles.
"""

import numpy as np
from scipy.optimize import brentq

from .constants import (
    ACTUATOR_GROUPS, M_SAT, MU_EARTH, CHIEF_OE,
    AX_MAX, AY_MAX, AZ_MAX, THETA_MIN, THETA_MAX, E_ACTUATOR,
)
from .aerodynamics import aero_acceleration


# Pre-compute reference orbital velocity at the chief altitude
_ALT_REF_KM = (CHIEF_OE['a'] - 6378.137e3) / 1e3       # ~400 km
_V_REF = np.sqrt(MU_EARTH / CHIEF_OE['a'])               # ~7669 m/s


def _accel_from_angle(theta: float, group_key: str,
                      altitude_km: float | None = None,
                      V_sat: float | None = None) -> float:
    """Compute acceleration for a given plate angle and actuator group."""
    if altitude_km is None:
        altitude_km = _ALT_REF_KM
    if V_sat is None:
        V_sat = _V_REF

    grp = ACTUATOR_GROUPS[group_key]
    use_lift = group_key in ('a', 'c')
    return aero_acceleration(theta, altitude_km, V_sat,
                             grp['K'], M_SAT, use_lift=use_lift)


def max_accel(group_key: str,
              altitude_km: float | None = None,
              V_sat: float | None = None) -> float:
    """Maximum achievable acceleration for a group at theta = 45 deg."""
    return abs(_accel_from_angle(THETA_MAX, group_key, altitude_km, V_sat))


def angle_from_accel(desired_accel: float, group_key: str,
                     altitude_km: float | None = None,
                     V_sat: float | None = None) -> float:
    """
    Numerical inversion: find theta in [0, pi/4] that produces *desired_accel*.

    Parameters
    ----------
    desired_accel : float   Desired absolute acceleration [m/s^2].
    group_key : str         'a', 'b', or 'c'.

    Returns
    -------
    theta : float   Plate angle [rad].
    """
    a_max = max_accel(group_key, altitude_km, V_sat)
    desired_accel = min(abs(desired_accel), a_max)

    if desired_accel < 1e-15:
        return 0.0

    # At theta ≈ 0 the acceleration is near zero; at theta_max it's a_max.
    # If desired exceeds a_max, clamp to theta_max.
    if desired_accel >= a_max * 0.999:
        return THETA_MAX

    th_lo = 1e-6
    th_hi = THETA_MAX

    def residual(th):
        return abs(_accel_from_angle(th, group_key, altitude_km, V_sat)) - desired_accel

    f_lo = residual(th_lo)
    f_hi = residual(th_hi)

    # If both endpoints have the same sign, the desired value may be below
    # the minimum achievable -- return the smallest angle.
    if f_lo * f_hi > 0:
        return th_lo if abs(f_lo) < abs(f_hi) else th_hi

    try:
        theta = brentq(residual, th_lo, th_hi, xtol=1e-10)
    except ValueError:
        theta = th_lo if abs(f_lo) < abs(f_hi) else th_hi
    return theta


def apply_actuator_error(theta: float) -> float:
    """Add relative actuator error  e_actuator = 5 %."""
    noise = np.random.uniform(-E_ACTUATOR, E_ACTUATOR)
    return theta * (1.0 + noise)


# ---------------------------------------------------------------------------
# Fitting formulae (Eq. 15) — optional alternative to numerical inversion.
# Coefficients transcribed from the rendered paper image.
# ---------------------------------------------------------------------------

def _fitting_theta_x(ax: float) -> float:
    """Fitting formula for x-axis plate angle (lift), Eq. 15 line 1."""
    ax = abs(ax)
    if ax < 1e-20:
        return 0.0
    sqrt_a = np.sqrt(ax)
    ln_a = np.log(ax)
    val_low = (3.99 + 6.47e5 * ax + 56.42 * sqrt_a * ln_a
               + 1.08e4 * sqrt_a - 55.96 / ln_a)
    if val_low <= 15.0:
        return np.radians(max(val_low, 0.0))
    val_high = (-5.64e2 + 7.89e7 * ax - 1.98e5 * sqrt_a * ln_a
                - 2.34e6 * sqrt_a + 1.11e-4 / ax)
    return np.radians(np.clip(val_high, 0.0, 45.0))


def _fitting_theta_y(ay: float) -> float:
    """Fitting formula for y-axis plate angle (drag), Eq. 15 line 2."""
    ay = abs(ay)
    if ay < 1e-20:
        return 0.0
    sqrt_a = np.sqrt(ay)
    ln_a = np.log(ay)
    val_low = (-60.64 + 1.03e5 * ay + 8.84e3 * sqrt_a * ln_a
               + 1.01e5 * sqrt_a - 1.19e3 / ln_a)
    if val_low <= 15.0:
        return np.radians(max(val_low, 0.0))
    val_high = (-1.50e2 + 1.42e7 * ay - 3.88e4 * sqrt_a * ln_a
                - 4.33e5 * sqrt_a + 4.79e-5 / ay)
    return np.radians(np.clip(val_high, 0.0, 45.0))


def _fitting_theta_z(az: float) -> float:
    """Fitting formula for z-axis plate angle (lift), Eq. 15 line 3."""
    az = abs(az)
    if az < 1e-20:
        return 0.0
    sqrt_a = np.sqrt(az)
    ln_a = np.log(az)
    val_low = (-3.99 + 6.47e5 * az + 56.42 * sqrt_a * ln_a
               + 1.08e4 * sqrt_a - 55.96 / ln_a)
    if val_low <= 15.0:
        return np.radians(max(val_low, 0.0))
    val_high = (-5.64e2 + 7.89e7 * az - 1.98e5 * sqrt_a * ln_a
                - 2.34e6 * sqrt_a + 1.11e-4 / az)
    return np.radians(np.clip(val_high, 0.0, 45.0))


FITTING_FORMULAE = {
    'a': _fitting_theta_x,
    'b': _fitting_theta_y,
    'c': _fitting_theta_z,
}


def compute_control_accels(u_desired: np.ndarray,
                           altitude_km: float | None = None,
                           V_sat: float | None = None,
                           add_error: bool = True) -> np.ndarray:
    """
    Full actuator pipeline: desired acceleration → plate angles → actual accel.

    Parameters
    ----------
    u_desired : (3,) array  Desired [ux, uy, uz] accelerations [m/s^2].
    altitude_km : float     Orbital altitude [km] (default: chief altitude).
    V_sat : float           Satellite velocity [m/s] (default: chief velocity).
    add_error : bool        Whether to add actuator operation error.

    Returns
    -------
    a_actual : (3,) array   Actual differential accelerations [m/s^2].
    """
    if altitude_km is None:
        altitude_km = _ALT_REF_KM
    if V_sat is None:
        V_sat = _V_REF

    limits = np.array([AX_MAX, AY_MAX, AZ_MAX])
    u_clamped = np.clip(u_desired, -limits, limits)
    groups = ['a', 'b', 'c']
    signs = np.sign(u_clamped)

    a_actual = np.zeros(3)
    for i, gk in enumerate(groups):
        if abs(u_clamped[i]) < 1e-15:
            continue
        theta = angle_from_accel(u_clamped[i], gk, altitude_km, V_sat)
        if add_error:
            theta = apply_actuator_error(theta)
            theta = np.clip(theta, THETA_MIN, THETA_MAX)
        a_actual[i] = signs[i] * abs(
            _accel_from_angle(theta, gk, altitude_km, V_sat)
        )

    return a_actual
