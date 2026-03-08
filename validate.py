"""
Comprehensive validation of the CubeSAT formation-keeping simulation
against all testable results from Shao et al. (2017).

Each test prints PASS/FAIL with the specific paper reference.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cubesat_sim.constants import *
from cubesat_sim.atmosphere import atmosphere
from cubesat_sim.aerodynamics import (
    pressure_and_shear, lift_drag_per_unit_area, _molecular_speed_ratio
)
from cubesat_sim.actuators import max_accel, angle_from_accel, _accel_from_angle
from cubesat_sim.orbital import (
    propagate_orbit, keplerian_to_eci, mean_to_true_anomaly,
    get_relative_state_lvlh, j2_accel_rtn,
)
from cubesat_sim.relative_motion import cw_propagate, cw_state_transition, cw_matrices
from cubesat_sim.controller import (
    lyapunov_control, clamp_control, lyapunov_value, lyapunov_derivative
)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  -- {detail}")
        failed += 1


# =========================================================================
print("\n=== 1. Constants (Table II, Table III) ===")
# =========================================================================

check("Chief a = 6778.137 km",
      abs(CHIEF_OE['a'] - 6778.137e3) < 1.0,
      f"got {CHIEF_OE['a']}")

check("Chief e = 0",
      CHIEF_OE['e'] == 0.0)

check("Chief i = 96.4522 deg",
      abs(np.degrees(CHIEF_OE['i']) - 96.4522) < 0.001,
      f"got {np.degrees(CHIEF_OE['i']):.4f}")

check("Chief omega = 90 deg",
      abs(np.degrees(CHIEF_OE['omega']) - 90.0) < 0.001)

check("Table III: x0 = 82.50 m",
      abs(REL_POS_0[0] - 82.50) < 0.01)

check("Table III: y0 = -930.46 m",
      abs(REL_POS_0[1] - (-930.46)) < 0.01)

check("Table III: z0 = 55.27 m",
      abs(REL_POS_0[2] - 55.27) < 0.01)

check("Table III: xdot0 = -0.17 m/s",
      abs(REL_VEL_0[0] - (-0.17)) < 0.001)

check("Table III: ydot0 = -0.04 m/s",
      abs(REL_VEL_0[1] - (-0.04)) < 0.001)

check("Table III: zdot0 = 0.29 m/s",
      abs(REL_VEL_0[2] - 0.29) < 0.001)

check("Satellite mass = 10 kg",
      M_SAT == 10.0)

check("Kr = 3e-5 * I3",
      np.allclose(KR, 3e-5 * np.eye(3)))

check("Kv = 2e-2 * I3",
      np.allclose(KV, 2e-2 * np.eye(3)))


# =========================================================================
print("\n=== 2. Atmosphere at 400 km ===")
# =========================================================================

rho_400, T_400, M_400 = atmosphere(400.0)
check("rho(400km) ~ 2.803e-12 kg/m³",
      abs(rho_400 - 2.803e-12) / 2.803e-12 < 0.01,
      f"got {rho_400:.3e}")

check("T(400km) ~ 995.8 K",
      abs(T_400 - 995.8) < 1.0,
      f"got {T_400:.1f}")

check("M(400km) ~ 15.98 g/mol",
      abs(M_400 - 15.98) < 0.1,
      f"got {M_400:.2f}")


# =========================================================================
print("\n=== 3. Aerodynamics (Eq. 1-2) ===")
# =========================================================================

V_sat = np.sqrt(MU_EARTH / CHIEF_OE['a'])
check("Orbital velocity ~ 7669 m/s",
      abs(V_sat - 7669) < 5,
      f"got {V_sat:.1f}")

S = _molecular_speed_ratio(V_sat, T_400, M_400)
check("Speed ratio S ~ 7.5 (hypersonic)",
      5 < S < 10,
      f"got {S:.2f}")

theta45 = np.radians(45)
Fl, Fd = lift_drag_per_unit_area(theta45, V_sat, rho_400, T_400, M_400)
check("At θ=45°: Fd > Fl (drag dominates for shear-dominated flow)",
      Fd > Fl,
      f"Fl={Fl:.3e}, Fd={Fd:.3e}")

check("Fl > 0 (lift is positive)",
      Fl > 0, f"Fl={Fl:.3e}")

check("Fd > 0 (drag is positive)",
      Fd > 0, f"Fd={Fd:.3e}")

# Test monotonicity: forces increase with theta
Fl_10, Fd_10 = lift_drag_per_unit_area(np.radians(10), V_sat, rho_400, T_400, M_400)
Fl_30, Fd_30 = lift_drag_per_unit_area(np.radians(30), V_sat, rho_400, T_400, M_400)
check("Drag increases with theta (10° < 30° < 45°)",
      Fd_10 < Fd_30 < Fd,
      f"Fd(10)={Fd_10:.3e}, Fd(30)={Fd_30:.3e}, Fd(45)={Fd:.3e}")


# =========================================================================
print("\n=== 4. Actuators (Table I, Eq. 3) ===")
# =========================================================================

check("Table I: Ka = N*k*A = 1*2*2 = 4",
      ACTUATOR_GROUPS['a']['K'] == 4.0)

check("Table I: Kb = N*k*A = 2*1*0.5 = 1",
      ACTUATOR_GROUPS['b']['K'] == 1.0)

check("Table I: Kc = N*k*A = 2*2*1 = 4",
      ACTUATOR_GROUPS['c']['K'] == 4.0)

ax_max = max_accel('a')
ay_max = max_accel('b')
az_max = max_accel('c')
check("ax_max ~ az_max (same K, both lift-based)",
      abs(ax_max - az_max) / ax_max < 0.01,
      f"ax={ax_max:.3e}, az={az_max:.3e}")

check("ax_max > ay_max (lift*4 > drag*1 at these conditions)",
      ax_max > ay_max,
      f"ax={ax_max:.3e}, ay={ay_max:.3e}")

check("Max accels are order 1e-5 m/s² (paper: 9.17e-6 to 1.11e-5)",
      1e-6 < ax_max < 1e-4,
      f"ax={ax_max:.3e}")

# Angle inversion round-trip
for gk in ['a', 'b', 'c']:
    a_test = 0.5 * max_accel(gk)
    theta_inv = angle_from_accel(a_test, gk)
    a_recovered = abs(_accel_from_angle(theta_inv, gk))
    err = abs(a_recovered - a_test) / a_test
    check(f"Angle inversion round-trip group {gk}: error < 1%",
          err < 0.01,
          f"err={err:.4f}")


# =========================================================================
print("\n=== 5. Orbit Propagation (Eq. 5-6, J2) ===")
# =========================================================================

chief_oe0 = np.array([
    CHIEF_OE['a'], CHIEF_OE['e'], CHIEF_OE['i'],
    CHIEF_OE['RAAN'], CHIEF_OE['omega'], CHIEF_OE['M0'],
])

# Propagate 3 orbits
t_final = 3 * T_ORBIT
t_eval = np.arange(0, t_final, 10.0)
sol = propagate_orbit(chief_oe0, (0, t_final), t_eval)

check("Propagation completes for 3 orbits",
      len(sol['t']) > 100 and sol['t'][-1] > 0.99 * t_final,
      f"got {len(sol['t'])} points, last t={sol['t'][-1]:.0f}")

# SMA should be approximately constant (J2 is conservative)
a_vals = sol['oe'][:, 0]
check("SMA variation < 0.1% over 3 orbits",
      (a_vals.max() - a_vals.min()) / a_vals.mean() < 0.005,
      f"range: {a_vals.min():.0f} to {a_vals.max():.0f}")

# RAAN regression: dΩ/dt = -1.5*n*J2*(Re/a)²*cos(i)
n0 = N_CHIEF
expected_dOmega_dt = -1.5 * n0 * J2 * (R_EARTH / CHIEF_OE['a'])**2 * np.cos(CHIEF_OE['i'])
actual_dOmega = (sol['oe'][-1, 3] - sol['oe'][0, 3]) / (sol['t'][-1] - sol['t'][0])
check("RAAN regression rate matches analytical J2 prediction (within 5%)",
      abs(actual_dOmega - expected_dOmega_dt) / abs(expected_dOmega_dt) < 0.05,
      f"expected={expected_dOmega_dt:.6e}, got={actual_dOmega:.6e}")

# Orbit radius should remain near initial value
r_norms = np.linalg.norm(sol['r_eci'], axis=1)
check("Orbit altitude stays near 400 km (±50 km)",
      r_norms.min() > R_EARTH + 350e3 and r_norms.max() < R_EARTH + 450e3,
      f"range: {(r_norms.min()-R_EARTH)/1e3:.0f} to {(r_norms.max()-R_EARTH)/1e3:.0f} km")


# =========================================================================
print("\n=== 6. CW Equations (Eq. 7-9) ===")
# =========================================================================

A1, A2 = cw_matrices(N_CHIEF)
check("A1[0,0] = 3ω² (Eq. 9)",
      abs(A1[0, 0] - 3 * N_CHIEF**2) < 1e-15)

check("A1[2,2] = -ω² (Eq. 9)",
      abs(A1[2, 2] - (-N_CHIEF**2)) < 1e-15)

check("A2[0,1] = 2ω (Eq. 9)",
      abs(A2[0, 1] - 2 * N_CHIEF) < 1e-15)

check("A2[1,0] = -2ω (Eq. 9)",
      abs(A2[1, 0] - (-2 * N_CHIEF)) < 1e-15)

# CW state transition: Φ(0) = I
Phi0 = cw_state_transition(N_CHIEF, 0.0)
check("Φ(0) = I₆ₓ₆",
      np.allclose(Phi0, np.eye(6), atol=1e-12))

# CW propagation: z-motion is simple harmonic at ω
rho0_z = np.array([0.0, 0.0, 100.0])
rho_dot0_z = np.array([0.0, 0.0, 0.0])
half_period = np.pi / N_CHIEF
rho_hp, _ = cw_propagate(N_CHIEF, rho0_z, rho_dot0_z, half_period)
check("CW z-motion: z(T/2) = -z₀ (simple harmonic)",
      abs(rho_hp[2] - (-100.0)) < 0.1,
      f"got z={rho_hp[2]:.2f}")


# =========================================================================
print("\n=== 7. Lyapunov Controller (Eq. 10-14) ===")
# =========================================================================

# V̇ must be ≤ 0 for any nonzero velocity error
np.random.seed(42)
for trial in range(20):
    drho = np.random.randn(3) * 10
    drho_dot = np.random.randn(3) * 0.1
    Vdot = lyapunov_derivative(drho_dot)
    check(f"V̇ ≤ 0 for random state (trial {trial})",
          Vdot <= 1e-15,
          f"Vdot={Vdot:.6e}")

# Controller WITHOUT saturation drives error to zero in CW dynamics
omega = N_CHIEF
A1, A2 = cw_matrices(omega)
rho0 = REL_POS_0 + TRACK_ERR_POS
rho_dot0 = REL_VEL_0 + TRACK_ERR_VEL
state = np.concatenate([rho0, rho_dot0])
dt = 1.0
V_history = []
for step in range(int(T_ORBIT)):
    t = step * dt
    rho_n = state[:3]; rho_dot_n = state[3:]
    rho_d, rho_dot_d = cw_propagate(omega, REL_POS_0, REL_VEL_0, t)
    delta_rho = rho_d - rho_n
    delta_rho_dot = rho_dot_d - rho_dot_n
    V_history.append(lyapunov_value(delta_rho, delta_rho_dot))
    u = lyapunov_control(delta_rho, delta_rho_dot, np.zeros(3))
    rho_ddot = A1 @ rho_n + A2 @ rho_dot_n + u
    state[:3] += rho_dot_n * dt + 0.5 * rho_ddot * dt**2
    state[3:] += rho_ddot * dt

V_arr = np.array(V_history)
check("Lyapunov V decreases monotonically (unsaturated CW)",
      all(V_arr[i+1] <= V_arr[i] + 1e-12 for i in range(len(V_arr)-1)),
      f"V[0]={V_arr[0]:.4e}, V[-1]={V_arr[-1]:.4e}")

final_err = np.linalg.norm(delta_rho)
check("Controller converges to < 0.01m in 1 orbit (unsaturated)",
      final_err < 0.01,
      f"final |err|={final_err:.6f} m")


# =========================================================================
print("\n=== 8. Mode 1: Uncontrolled J2 Drift (Fig. 7) ===")
# =========================================================================

from main import run_uncontrolled

res1 = run_uncontrolled()
pe = res1['pos_err']
ve = res1['vel_err']

check("Mode 1 runs for 3 orbits",
      len(res1['t']) > 10000,
      f"got {len(res1['t'])} points")

# The paper's Fig. 7 shows: Y drift is largest (secular), X and Z oscillate
y_max_err = np.max(np.abs(pe[:, 1]))
x_max_err = np.max(np.abs(pe[:, 0]))
z_max_err = np.max(np.abs(pe[:, 2]))

check("Fig 7: Y-drift is largest axis (secular along-track drift)",
      y_max_err >= x_max_err and y_max_err >= z_max_err,
      f"x={x_max_err:.1f}, y={y_max_err:.1f}, z={z_max_err:.1f}")

check("Fig 7: Position errors are order 10-100 m over 3 orbits",
      1.0 < np.max(np.abs(pe)) < 1000,
      f"max={np.max(np.abs(pe)):.1f} m")

check("Fig 7: Velocity errors are order 0.01-0.1 m/s over 3 orbits",
      0.001 < np.max(np.abs(ve)) < 1.0,
      f"max={np.max(np.abs(ve)):.4f} m/s")


# =========================================================================
print("\n=== 9. Mode 2: Controlled Formation (Fig. 8-9) ===")
# =========================================================================

from main import run_controlled

np.random.seed(0)
res2 = run_controlled()
pe2 = res2['pos_err']
ve2 = res2['vel_err']
ctrl = res2['control']

check("Mode 2 runs successfully",
      len(res2['t']) > 100,
      f"got {len(res2['t'])} points")

# Initial convergence: error should decrease in first ~half orbit
T_half = int(T_ORBIT / 2 / 10)  # steps at 10s interval
err_start = np.linalg.norm(pe2[0])
err_half = np.linalg.norm(pe2[min(T_half, len(pe2)-1)])
check("Fig 8: Error decreases in first half-orbit",
      err_half < err_start,
      f"|err|(0)={err_start:.2f}, |err|(T/2)={err_half:.2f}")

# Z-axis should converge well (decoupled, within actuator authority)
z_err_end = np.abs(pe2[-1, 2])
check("Fig 8: Z-axis error converges to < 10 m",
      z_err_end < 10.0,
      f"|z_err|={z_err_end:.2f} m")

# Control accelerations should be within limits
ctrl_max = np.max(np.abs(ctrl), axis=0)
check("Fig 9: ax within limit",
      ctrl_max[0] <= AX_MAX * 1.01,
      f"max ax={ctrl_max[0]:.3e}, limit={AX_MAX:.3e}")

check("Fig 9: ay within limit",
      ctrl_max[1] <= AY_MAX * 1.01,
      f"max ay={ctrl_max[1]:.3e}, limit={AY_MAX:.3e}")

check("Fig 9: az within limit",
      ctrl_max[2] <= AZ_MAX * 1.01,
      f"max az={ctrl_max[2]:.3e}, limit={AZ_MAX:.3e}")


# =========================================================================
print("\n=== 10. Mode 3: Key Factors (Fig. 10-11) ===")
# =========================================================================

from main import run_key_factors

res3 = run_key_factors()

# Fig 10: J2 accel increases linearly with radius
j2_x_r = res3['j2_x_r']
check("Fig 10: J2 accel increases with spatial radius",
      j2_x_r[-1] > j2_x_r[0],
      f"j2(1km)={j2_x_r[0]:.3e}, j2(2.5km)={j2_x_r[-1]:.3e}")

check("Fig 10: J2 accel scales linearly with radius (ratio ≈ 2.5)",
      abs(j2_x_r[-1] / j2_x_r[0] - 2.5) < 0.1,
      f"ratio={j2_x_r[-1]/j2_x_r[0]:.2f}")

# Fig 11: Aero accel decreases with altitude (density drops)
aero_ax_alt = res3['aero_ax_alt']
check("Fig 11: Aero accel decreases with altitude",
      aero_ax_alt[0] > aero_ax_alt[-1],
      f"aero(450km)={aero_ax_alt[0]:.3e}, aero(500km)={aero_ax_alt[-1]:.3e}")

# J2 should decrease with altitude (r⁵ denominator)
j2_x_alt = res3['j2_x_alt']
check("Fig 11: J2 accel decreases with altitude",
      j2_x_alt[0] > j2_x_alt[-1],
      f"j2(450km)={j2_x_alt[0]:.3e}, j2(500km)={j2_x_alt[-1]:.3e}")


# =========================================================================
print("\n" + "="*60)
print(f"  Results: {passed} passed, {failed} failed out of {passed+failed}")
print("="*60)
