"""
Lyapunov-based formation keeping controller (Eq. 10-14).

The control law drives the tracking errors (Δρ, Δρ̇) to zero using
negative feedback derived from a quadratic Lyapunov candidate, ensuring
global asymptotic stability under the CW approximation.
"""

import numpy as np

from .constants import KR, KV, AX_MAX, AY_MAX, AZ_MAX, N_CHIEF
from .relative_motion import cw_matrices


def lyapunov_control(delta_rho: np.ndarray,
                     delta_rho_dot: np.ndarray,
                     d: np.ndarray,
                     omega: float | None = None) -> np.ndarray:
    """
    Compute desired control acceleration  u  via the Lyapunov control law.

    The paper's Eq. 12 uses the convention ε = ρ_actual - ρ_desired with
    u = -(A1+Kr)ε - (A2+Kv)ε̇ - d.  Since delta_rho = ρ_d - ρ_n = -ε,
    the law in our convention becomes:

        u = (A1 + Kr) Δρ + (A2 + Kv) Δρ̇ - d

    This yields closed-loop error dynamics  ε̈ = -Kr ε - Kv ε̇  which is
    globally asymptotically stable with  V̇ = -ε̇ᵀ Kv ε̇ ≤ 0.

    Parameters
    ----------
    delta_rho : (3,)      Position tracking error  ρ_d - ρ_n  [m].
    delta_rho_dot : (3,)  Velocity tracking error  ρ̇_d - ρ̇_n [m/s].
    d : (3,)              Estimated disturbance acceleration [m/s^2].
    omega : float or None Chief mean motion [rad/s].  None → use default.

    Returns
    -------
    u : (3,) array  Desired control acceleration [m/s^2].
    """
    if omega is None:
        omega = N_CHIEF

    A1, A2 = cw_matrices(omega)

    u = ((A1 + KR) @ delta_rho
         + (A2 + KV) @ delta_rho_dot
         - d)
    return u


def clamp_control(u: np.ndarray) -> np.ndarray:
    """Clamp control acceleration to the physical actuator limits."""
    limits = np.array([AX_MAX, AY_MAX, AZ_MAX])
    return np.clip(u, -limits, limits)


def lyapunov_value(delta_rho: np.ndarray,
                   delta_rho_dot: np.ndarray) -> float:
    """
    Evaluate the Lyapunov function V(x) from Eq. 10:

        V = ½ Δρᵀ Kr Δρ + ½ Δρ̇ᵀ Δρ̇
    """
    return 0.5 * delta_rho @ KR @ delta_rho + 0.5 * delta_rho_dot @ delta_rho_dot


def lyapunov_derivative(delta_rho_dot: np.ndarray) -> float:
    """
    Evaluate V̇ from Eq. 13:

        V̇ = -Δρ̇ᵀ Kv Δρ̇  ≤ 0
    """
    return -delta_rho_dot @ KV @ delta_rho_dot
