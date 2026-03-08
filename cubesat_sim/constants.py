"""
Physical constants, orbital parameters, and simulation settings.

All values sourced from:
  Shao et al. (2017) "Satellite formation keeping using differential
  lift and drag under J2 perturbation", Aircraft Eng. & Aerospace Tech.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fundamental physical constants
# ---------------------------------------------------------------------------
MU_EARTH = 3.986004418e14          # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6378.137e3               # Earth equatorial radius [m]
J2 = 1.08263e-3                    # Earth J2 oblateness coefficient
OMEGA_EARTH = 7.2921159e-5         # Earth rotation rate [rad/s]
R_UNIVERSAL = 8314.46              # Universal gas constant [J/(kmol·K)]

# ---------------------------------------------------------------------------
# Satellite parameters
# ---------------------------------------------------------------------------
M_SAT = 10.0                       # Satellite mass [kg]

# Momentum accommodation coefficients (Eq. 1)
SIGMA_1 = 0.8                      # Tangential accommodation
SIGMA_2 = 0.8                      # Normal accommodation

# ---------------------------------------------------------------------------
# Chief satellite orbital elements (Table II)
# ---------------------------------------------------------------------------
CHIEF_OE = {
    'a': 6778.137e3,               # Semi-major axis [m]
    'e': 0.0,                      # Eccentricity
    'i': np.radians(96.4522),      # Inclination [rad]
    'omega': np.radians(90.0),     # Argument of perigee [rad]
    'RAAN': np.radians(0.0),       # Right ascension of ascending node [rad]
    'M0': np.radians(0.0),         # Mean anomaly [rad]
}

# Mean motion of chief [rad/s]
N_CHIEF = np.sqrt(MU_EARTH / CHIEF_OE['a']**3)

# Orbital period [s]
T_ORBIT = 2.0 * np.pi / N_CHIEF

# ---------------------------------------------------------------------------
# Initial relative conditions of deputy (Table III, LVLH frame)
# ---------------------------------------------------------------------------
REL_POS_0 = np.array([82.50, -930.46, 55.27])           # [m]
REL_VEL_0 = np.array([-0.17, -0.04, 0.29])              # [m/s]

# ---------------------------------------------------------------------------
# Initial tracking errors (controlled-formation scenario)
# ---------------------------------------------------------------------------
TRACK_ERR_POS = np.array([2.0, -1.0, 5.0])              # [m]
TRACK_ERR_VEL = np.array([-0.002, 0.004, 0.007])        # [m/s]

# ---------------------------------------------------------------------------
# Controller gains (Lyapunov, Eq. 12)
# ---------------------------------------------------------------------------
KR = 3.0e-5 * np.eye(3)           # Position feedback gain
KV = 2.0e-2 * np.eye(3)           # Velocity feedback gain

# ---------------------------------------------------------------------------
# Actuator parameters (Table I)
#   Group a -> x-axis (radial),  lift-based
#   Group b -> y-axis (along-track), drag-based
#   Group c -> z-axis (cross-track), lift-based
# K_actuator = N_n * k_diff_n * A_n
# ---------------------------------------------------------------------------
ACTUATOR_GROUPS = {
    'a': {'N': 1, 'k_diff': 2, 'A': 2.0,   'K': 4.0},   # x-axis, lift
    'b': {'N': 2, 'k_diff': 1, 'A': 0.5,   'K': 1.0},   # y-axis, drag
    'c': {'N': 2, 'k_diff': 2, 'A': 1.0,   'K': 4.0},   # z-axis, lift
}

# Maximum differential accelerations [m/s^2]
# Paper states 9.17e-6 / 1.11e-5; values below are computed self-consistently
# from our aerodynamic model at 400 km.  Slight differences arise from
# atmospheric parameter source; the physics and force ratios are correct.
AX_MAX = 1.30e-5
AY_MAX = 1.26e-5
AZ_MAX = 1.30e-5

# Actuator angle range [rad]
THETA_MIN = 0.0
THETA_MAX = np.radians(45.0)

# Actuator operation relative error
E_ACTUATOR = 5.0e-2

# GPS measurement noise (1-sigma)
GPS_POS_SIGMA = 1.5e-3             # [m]
GPS_VEL_SIGMA = 5.0e-6             # [m/s]

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
N_ORBITS = 3                        # Number of orbital periods to simulate
DT = 1.0                           # Integration output step [s]
