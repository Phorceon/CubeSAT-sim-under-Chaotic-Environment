"""
Orbit propagation with J2 perturbation via Gauss Variational Equations.

Implements Eq. 5-6 from Shao et al. (2017), plus utility transforms
between Keplerian elements, ECI position/velocity, and the LVLH frame.

For near-circular orbits (e ≈ 0), modified equinoctial elements are used
internally to avoid the 1/e singularity in dω/dt and dM/dt.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import MU_EARTH, R_EARTH, J2, OMEGA_EARTH


# =========================================================================
# Keplerian utilities
# =========================================================================

def mean_to_true_anomaly(M: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation  M = E - e*sin(E)  via Newton iteration."""
    if e < 1e-10:
        return M % (2.0 * np.pi)
    E = M
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    f = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                          np.sqrt(1 - e) * np.cos(E / 2))
    return f % (2.0 * np.pi)


def keplerian_to_eci(a, e, i, RAAN, omega, f):
    """
    Convert Keplerian elements to ECI position and velocity vectors.

    Parameters
    ----------
    a : float       Semi-major axis [m].
    e : float       Eccentricity.
    i : float       Inclination [rad].
    RAAN : float    Right ascension of ascending node [rad].
    omega : float   Argument of perigee [rad].
    f : float       True anomaly [rad].

    Returns
    -------
    r_eci : (3,) array  Position [m].
    v_eci : (3,) array  Velocity [m/s].
    """
    p = a * (1.0 - e**2)
    r_mag = p / (1.0 + e * np.cos(f))

    # Position & velocity in perifocal frame
    r_pf = np.array([r_mag * np.cos(f),
                      r_mag * np.sin(f),
                      0.0])
    mu_p = np.sqrt(MU_EARTH / p)
    v_pf = np.array([-mu_p * np.sin(f),
                      mu_p * (e + np.cos(f)),
                      0.0])

    # Rotation matrix: perifocal -> ECI
    cO, sO = np.cos(RAAN), np.sin(RAAN)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(i), np.sin(i)

    R = np.array([
        [cO * cw - sO * sw * ci,  -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci,  -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                   cw * si,                  ci     ],
    ])

    return R @ r_pf, R @ v_pf


def eci_to_lvlh(r_chief, v_chief):
    """
    Build the ECI → LVLH rotation matrix from the chief's ECI state.

    LVLH (RTN) convention used in the paper:
        x = radial (away from Earth)
        y = along-track
        z = cross-track (completes right-hand frame)

    Returns
    -------
    A : (3,3) array   Rotation matrix  r_lvlh = A @ r_eci.
    """
    r_hat = r_chief / np.linalg.norm(r_chief)
    h = np.cross(r_chief, v_chief)
    z_hat = h / np.linalg.norm(h)
    y_hat = np.cross(z_hat, r_hat)
    return np.array([r_hat, y_hat, z_hat])


# =========================================================================
# J2 perturbing accelerations in RTN frame (Eq. 6)
# =========================================================================

def j2_accel_rtn(r_mag, i, omega, f):
    """
    J2 perturbation accelerations in the radial-transverse-normal frame.

    Returns (F_r, F_t, F_n) in [m/s^2].
    """
    coeff = -1.5 * J2 * MU_EARTH * R_EARTH**2 / r_mag**4
    sin_i = np.sin(i)
    sin2_i = sin_i**2
    u = omega + f                       # argument of latitude

    F_r = coeff * (1.0 - 3.0 * sin2_i * np.sin(u)**2)
    F_t = coeff * sin2_i * np.sin(2.0 * u)
    F_n = coeff * np.sin(2.0 * i) * np.sin(u)
    return F_r, F_t, F_n


# =========================================================================
# Gauss Variational Equations (Eq. 5)
# =========================================================================

def gauss_ve_derivs(t, state, control_rtn=None):
    """
    Right-hand side using a singularity-free formulation for near-circular
    orbits.  State = [a, ex, ey, i, RAAN, u] where:
        ex = e*cos(omega), ey = e*sin(omega),  u = omega + M (mean arg of lat)
    This avoids the 1/e singularity in classical Gauss VE.
    """
    a, ex, ey, inc, RAAN, u_mean = state

    e = np.sqrt(ex**2 + ey**2)
    e_safe = max(e, 1e-14)
    p = a * (1.0 - e**2)
    n = np.sqrt(MU_EARTH / a**3)

    # Recover ω and M from equinoctial elements
    omega = np.arctan2(ey, ex) if e_safe > 1e-12 else 0.0
    M = u_mean - omega

    f = mean_to_true_anomaly(M, e_safe)
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(f)) if e_safe > 1e-12 else a

    # J2 accelerations
    Fr, Ft, Fn = j2_accel_rtn(r, inc, omega, f)

    if control_rtn is not None:
        cr, ct, cn = control_rtn(t, state)
        Fr += cr
        Ft += ct
        Fn += cn

    sin_f = np.sin(f)
    cos_f = np.cos(f)
    h = n * a**2 * np.sqrt(1.0 - e**2)
    u_true = omega + f
    sin_u = np.sin(u_true)
    cos_u = np.cos(u_true)
    sin_i = np.sin(inc)
    cos_i = np.cos(inc)

    # --- da/dt ---
    da_dt = (2.0 * a**2 / h) * (
        Fr * e * sin_f + Ft * p / r
    )

    # --- dex/dt, dey/dt (singularity-free) ---
    dex_dt = (1.0 / h) * (
        p * sin_u * Fr
        + ((p + r) * cos_u + r * ex) * Ft
        - r * ey * sin_u / (np.tan(inc) if abs(sin_i) > 1e-12 else 1e30) * Fn
    )
    dey_dt = (1.0 / h) * (
        -p * cos_u * Fr
        + ((p + r) * sin_u + r * ey) * Ft
        + r * ex * sin_u / (np.tan(inc) if abs(sin_i) > 1e-12 else 1e30) * Fn
    )

    # --- di/dt ---
    di_dt = (r * cos_u / h) * Fn

    # --- dΩ/dt ---
    dRAAN_dt = (r * sin_u / (h * sin_i)) * Fn if abs(sin_i) > 1e-12 else 0.0

    # --- du_mean/dt = n + dω/dt + dM/dt corrections ---
    # For near-circular orbits, du_mean/dt ≈ n plus small J2 corrections.
    # Use the exact expression:  du_mean/dt = n + (dω/dt from full GVE)
    #                                         + (dM_corrections from full GVE)
    # For e≈0, the dominant term is just n, with secular J2 contribution.
    du_dt = n

    # Secular J2 rate on argument of perigee
    if e_safe > 1e-8:
        sqrt_1me2 = np.sqrt(1.0 - e**2)
        domega_perturb = (sqrt_1me2 / (n * a * e_safe)) * (
            -Fr * cos_f
            + Ft * (2.0 + e * cos_f) / (1.0 + e * cos_f) * sin_f
        ) - cos_i * dRAAN_dt
        dM_perturb = -((1.0 - e**2) / (n * a * e_safe)) * (
            Fr * (2.0 * e * r / p - cos_f)
            + Ft * (1.0 + r / p) * sin_f
        )
        du_dt += domega_perturb + dM_perturb
    else:
        du_dt += -cos_i * dRAAN_dt

    return np.array([da_dt, dex_dt, dey_dt, di_dt, dRAAN_dt, du_dt])


# =========================================================================
# High-level propagation
# =========================================================================

def classical_to_equinoctial(oe_classical):
    """Convert [a, e, i, RAAN, omega, M] → [a, ex, ey, i, RAAN, u_mean]."""
    a, e, inc, RAAN, omega, M = oe_classical
    ex = e * np.cos(omega)
    ey = e * np.sin(omega)
    u_mean = omega + M
    return np.array([a, ex, ey, inc, RAAN, u_mean])


def equinoctial_to_classical(oe_eq):
    """Convert [a, ex, ey, i, RAAN, u_mean] → [a, e, i, RAAN, omega, M]."""
    a, ex, ey, inc, RAAN, u_mean = oe_eq
    e = np.sqrt(ex**2 + ey**2)
    omega = np.arctan2(ey, ex) if e > 1e-12 else 0.0
    M = (u_mean - omega) % (2.0 * np.pi)
    return np.array([a, e, inc, RAAN, omega, M])


def propagate_orbit(oe0: np.ndarray, t_span: tuple, t_eval: np.ndarray,
                    control_rtn=None, method='DOP853',
                    rtol=1e-11, atol=1e-13) -> dict:
    """
    Propagate a single satellite from initial classical elements *oe0*.

    Internally converts to equinoctial elements [a, ex, ey, i, RAAN, u_mean]
    to avoid the 1/e singularity for near-circular orbits, then converts
    back to classical elements for output.

    Parameters
    ----------
    oe0 : (6,) array   [a, e, i, RAAN, omega, M0]  (SI + radians).
    t_span : (t0, tf)  Integration limits [s].
    t_eval : array      Output time points [s].
    control_rtn : callable or None
        control_rtn(t, oe_eq) → (a_r, a_t, a_n)  in RTN [m/s^2].
        Note: receives equinoctial state, but for constant control this
        doesn't matter.

    Returns
    -------
    sol : dict with keys 't', 'oe' (classical), 'r_eci', 'v_eci'.
    """
    oe0_eq = classical_to_equinoctial(oe0)

    def rhs(t, y):
        return gauss_ve_derivs(t, y, control_rtn)

    result = solve_ivp(rhs, t_span, oe0_eq, method=method,
                       t_eval=t_eval, rtol=rtol, atol=atol,
                       max_step=30.0)

    n_pts = len(result.t)
    oe_classical = np.zeros((n_pts, 6))
    r_eci = np.zeros((n_pts, 3))
    v_eci = np.zeros((n_pts, 3))

    for k in range(n_pts):
        oe_eq_k = result.y[:, k]
        oe_cl_k = equinoctial_to_classical(oe_eq_k)
        oe_classical[k] = oe_cl_k
        a_k, e_k, i_k, O_k, w_k, M_k = oe_cl_k
        e_k = max(e_k, 0.0)
        f_k = mean_to_true_anomaly(M_k, e_k)
        r_eci[k], v_eci[k] = keplerian_to_eci(a_k, e_k, i_k, O_k, w_k, f_k)

    return {
        't': result.t,
        'oe': oe_classical,      # (n_pts, 6)  classical elements
        'r_eci': r_eci,          # (n_pts, 3)
        'v_eci': v_eci,          # (n_pts, 3)
    }


def get_relative_state_lvlh(r_chief, v_chief, r_deputy, v_deputy):
    """
    Compute relative position and velocity in the chief's LVLH frame.

    Parameters
    ----------
    r_chief, v_chief : (3,) arrays   Chief ECI state.
    r_deputy, v_deputy : (3,) arrays Deputy ECI state.

    Returns
    -------
    rho : (3,) array     Relative position in LVLH [m].
    rho_dot : (3,) array Relative velocity in LVLH [m/s].
    """
    A = eci_to_lvlh(r_chief, v_chief)
    dr = r_deputy - r_chief
    dv = v_deputy - v_chief

    omega_orb = np.linalg.norm(np.cross(r_chief, v_chief)) / np.linalg.norm(r_chief)**2
    omega_vec_lvlh = np.array([0.0, 0.0, omega_orb])

    rho = A @ dr
    rho_dot = A @ dv - np.cross(omega_vec_lvlh, rho)
    return rho, rho_dot
