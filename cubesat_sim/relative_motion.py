"""
Clohessy-Wiltshire (CW) relative-motion model (Eq. 7-9).

Provides the analytical CW solution for the desired (unperturbed) relative
trajectory, and utilities for disturbance estimation.
"""

import numpy as np


def cw_state_transition(omega: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    CW state-transition matrices  Φ_rr, Φ_rv, Φ_vr, Φ_vv  so that

        ρ(t)   = Φ_rr ρ₀ + Φ_rv ρ̇₀
        ρ̇(t)  = Φ_vr ρ₀ + Φ_vv ρ̇₀

    Returns combined 6×6 matrix Φ such that [ρ; ρ̇](t) = Φ [ρ₀; ρ̇₀].
    """
    nt = omega * t
    sn = np.sin(nt)
    cn = np.cos(nt)
    n = omega

    Phi = np.zeros((6, 6))

    # Position rows
    Phi[0, 0] = 4.0 - 3.0 * cn
    Phi[0, 1] = 0.0
    Phi[0, 2] = 0.0
    Phi[0, 3] = sn / n
    Phi[0, 4] = 2.0 * (1.0 - cn) / n
    Phi[0, 5] = 0.0

    Phi[1, 0] = 6.0 * (sn - nt)
    Phi[1, 1] = 1.0
    Phi[1, 2] = 0.0
    Phi[1, 3] = -2.0 * (1.0 - cn) / n
    Phi[1, 4] = (4.0 * sn - 3.0 * nt) / n
    Phi[1, 5] = 0.0

    Phi[2, 2] = cn
    Phi[2, 5] = sn / n

    # Velocity rows
    Phi[3, 0] = 3.0 * n * sn
    Phi[3, 3] = cn
    Phi[3, 4] = 2.0 * sn

    Phi[4, 0] = 6.0 * n * (cn - 1.0)
    Phi[4, 1] = 0.0
    Phi[4, 3] = -2.0 * sn
    Phi[4, 4] = 4.0 * cn - 3.0

    Phi[5, 2] = -n * sn
    Phi[5, 5] = cn

    return Phi


def cw_propagate(omega: float, rho0: np.ndarray, rho_dot0: np.ndarray,
                 t: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate unperturbed CW relative state to time *t*.

    Parameters
    ----------
    omega : float         Chief mean motion [rad/s].
    rho0 : (3,) array     Initial relative position [m].
    rho_dot0 : (3,) array Initial relative velocity [m/s].
    t : float             Elapsed time [s].

    Returns
    -------
    rho : (3,) array      Desired relative position at t.
    rho_dot : (3,) array  Desired relative velocity at t.
    """
    x0 = np.concatenate([rho0, rho_dot0])
    Phi = cw_state_transition(omega, t)
    x = Phi @ x0
    return x[:3], x[3:]


def cw_matrices(omega: float) -> tuple[np.ndarray, np.ndarray]:
    """
    CW system matrices A1, A2 from Eq. 9.

    The CW dynamics in error coordinates:
        [Δρ̇ ]   [  0   I ] [Δρ ]       [ 0 ]
        [Δρ̈ ] = [ A1  A2 ] [Δρ̇] + B * [ d+u ]

    Returns
    -------
    A1 : (3,3)  stiffness-like matrix.
    A2 : (3,3)  Coriolis-like matrix.
    """
    w2 = omega**2
    A1 = np.array([
        [3.0 * w2, 0.0,  0.0   ],
        [0.0,      0.0,  0.0   ],
        [0.0,      0.0, -w2    ],
    ])
    A2 = np.array([
        [ 0.0,          2.0 * omega, 0.0],
        [-2.0 * omega,  0.0,         0.0],
        [ 0.0,          0.0,         0.0],
    ])
    return A1, A2


def estimate_disturbance(rho: np.ndarray, rho_dot: np.ndarray,
                         rho_ddot_actual: np.ndarray,
                         omega: float) -> np.ndarray:
    """
    Estimate the J2-induced disturbance  d  from Eq. 7:

        ρ̈ = A1 ρ + A2 ρ̇ + d + u

    In the uncontrolled case (u=0):
        d = ρ̈_actual - A1 ρ - A2 ρ̇

    Parameters
    ----------
    rho, rho_dot : (3,) arrays     Current LVLH relative state.
    rho_ddot_actual : (3,) array   Actual relative acceleration (from
                                   differencing propagated ECI states).
    omega : float                  Chief mean motion [rad/s].

    Returns
    -------
    d : (3,) array   Disturbance acceleration [m/s^2].
    """
    A1, A2 = cw_matrices(omega)
    return rho_ddot_actual - A1 @ rho - A2 @ rho_dot
