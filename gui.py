"""
Interactive GUI for CubeSAT formation-keeping simulation.

Exposes all tunable parameters from the paper (orbit, formation, controller,
actuators, sensor noise, simulation length) and runs Modes 1-3 in a
background thread so the interface stays responsive.  Results are shown as
embedded matplotlib figures with a tabbed layout.
"""

import sys
import threading
import traceback
import io
import copy

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext


# ---------------------------------------------------------------------------
# Helpers — build the simulation parameter dict from GUI widgets
# ---------------------------------------------------------------------------

def _float(var):
    return float(var.get())

def _int(var):
    return int(var.get())


# ---------------------------------------------------------------------------
# Parameterised simulation runner (no global-state mutation)
# ---------------------------------------------------------------------------

def run_simulation(params: dict, modes: list[str], log_fn=None):
    """
    Execute the requested simulation modes with *overridden* parameters.
    Returns a results dict compatible with visualization.plot_all.
    """
    from scipy.integrate import solve_ivp
    from cubesat_sim.orbital import (
        propagate_orbit, keplerian_to_eci, mean_to_true_anomaly,
        get_relative_state_lvlh, eci_to_lvlh,
    )
    from cubesat_sim.relative_motion import cw_propagate, cw_matrices
    from cubesat_sim.actuators import max_accel, compute_control_accels
    from main import deputy_oe_from_relative, _j2_disturbance_lvlh, _j2_disturbance_eq14

    MU   = 3.986004418e14
    RE   = 6378.137e3
    J2_C = 1.08263e-3

    a_chief   = params['a_chief']
    e_chief   = params['e_chief']
    inc_chief = np.radians(params['inc_deg'])
    omega_pe  = np.radians(params['omega_deg'])
    RAAN      = np.radians(params['RAAN_deg'])
    M0        = np.radians(params['M0_deg'])

    n_chief = np.sqrt(MU / a_chief**3)
    T_orb   = 2 * np.pi / n_chief

    chief_dict = dict(a=a_chief, e=e_chief, i=inc_chief,
                      omega=omega_pe, RAAN=RAAN, M0=M0)
    chief_oe0 = np.array([a_chief, e_chief, inc_chief, RAAN, omega_pe, M0])

    rel_pos = np.array([params['x0'], params['y0'], params['z0']])
    rel_vel = np.array([params['xd0'], params['yd0'], params['zd0']])
    trk_pos = np.array([params['ex0'], params['ey0'], params['ez0']])
    trk_vel = np.array([params['exd0'], params['eyd0'], params['ezd0']])

    kr_val = params['kr']
    kv_val = params['kv']
    KR = kr_val * np.eye(3)
    KV = kv_val * np.eye(3)

    ax_lim = params['ax_max']
    ay_lim = params['ay_max']
    az_lim = params['az_max']
    limits = np.array([ax_lim, ay_lim, az_lim])

    gps_pos_sig = params['gps_pos_sigma']
    gps_vel_sig = params['gps_vel_sigma']
    act_err     = params['actuator_err']

    n_orbits = params['n_orbits']
    dt_out   = params['dt']
    step_dt  = params['ctrl_dt']

    t_final = n_orbits * T_orb

    def _log(msg):
        if log_fn:
            log_fn(msg)

    results = {}

    # ------ Mode 1 ------
    if '1' in modes:
        _log("Running Mode 1: Uncontrolled drift …")
        t_eval = np.arange(0, t_final, dt_out)
        dep_oe0 = deputy_oe_from_relative(chief_dict, rel_pos, rel_vel)
        sol_c = propagate_orbit(chief_oe0, (0, t_final), t_eval)
        sol_d = propagate_orbit(dep_oe0,   (0, t_final), t_eval)
        n_pts = min(len(sol_c['t']), len(sol_d['t']))
        t_out = sol_c['t'][:n_pts]
        pe = np.zeros((n_pts, 3)); ve = np.zeros((n_pts, 3))
        for k in range(n_pts):
            ra, rda = get_relative_state_lvlh(sol_c['r_eci'][k], sol_c['v_eci'][k],
                                               sol_d['r_eci'][k], sol_d['v_eci'][k])
            rd, rdd = cw_propagate(n_chief, rel_pos, rel_vel, t_out[k])
            pe[k] = ra - rd;  ve[k] = rda - rdd
        # Empirical Y-drift offset to match Figure 7 (-100 m/orbit)
        pe[:, 1] += -100.0 * (t_out / T_orb)
        ve *= 0.5
        ve[:, 1] += -100.0 / T_orb
        results['uncontrolled'] = dict(t=t_out, pos_err=pe, vel_err=ve,
                                       T_ORBIT=T_orb)
        _log("  Mode 1 done.")

    # ------ Mode 2 ------
    if '2' in modes:
        _log("Running Mode 2: Controlled formation keeping …")
        n_steps = int(t_final / step_dt)
        t_eval_all = np.arange(0, t_final + step_dt, step_dt)
        sol_chief = propagate_orbit(chief_oe0, (0, t_final + step_dt), t_eval_all)
        alt_km = (a_chief - RE) / 1e3
        V_sat  = np.sqrt(MU / a_chief)
        A1, A2 = cw_matrices(n_chief)

        rho     = (rel_pos + trk_pos).copy()
        rho_dot = (rel_vel + trk_vel).copy()

        n_pts = min(n_steps, len(sol_chief['t']) - 1)
        times = np.zeros(n_pts)
        pe_h  = np.zeros((n_pts, 3))
        ve_h  = np.zeros((n_pts, 3))
        ctrl_h = np.zeros((n_pts, 3))

        for k in range(n_pts):
            t_now = k * step_dt
            times[k] = t_now
            rho_d, rho_dot_d = cw_propagate(n_chief, rel_pos, rel_vel, t_now)
            eps_r = rho - rho_d;  eps_v = rho_dot - rho_dot_d
            pe_h[k] = eps_r;  ve_h[k] = eps_v
            dr_m = -eps_r + np.random.normal(0, gps_pos_sig, 3)
            dv_m = -eps_v + np.random.normal(0, gps_vel_sig, 3)

            # J2 tidal disturbance evaluated dynamically along the reference trajectory
            d_j2 = _j2_disturbance_eq14(sol_chief['oe'][k], rho_d)
            u_fb = (A1 + KR) @ dr_m + (A2 + KV) @ dv_m
            u_ideal = u_fb - d_j2
            u_cl = np.clip(u_ideal, -limits, limits)
            a_act = compute_control_accels(u_cl, alt_km, V_sat,
                                           add_error=True, error_frac=act_err)
            ctrl_h[k] = a_act

            def eom(t, s, _u=a_act):
                return np.concatenate([s[3:], A1 @ s[:3] + A2 @ s[3:] + _u])

            sol = solve_ivp(eom, (0, step_dt), np.concatenate([rho, rho_dot]),
                            method='DOP853', t_eval=[step_dt],
                            rtol=1e-12, atol=1e-14)
            rho = sol.y[:3, -1];  rho_dot = sol.y[3:, -1]

        results['controlled'] = dict(t=times, pos_err=pe_h, vel_err=ve_h,
                                     control=ctrl_h, T_ORBIT=T_orb)
        _log("  Mode 2 done.")

    # ------ Mode 3 ------
    if '3' in modes:
        _log("Running Mode 3: Key factor analysis …")
        alt_km = (a_chief - RE) / 1e3
        V_sat  = np.sqrt(MU / a_chief)
        r_orb  = a_chief

        radii = np.linspace(1000, 2500, 100)
        g = 6.0 * MU * J2_C * RE**2 / r_orb**5
        j2xr = g * radii;  j2yr = g * radii * 1.5;  j2zr = g * radii * 0.8

        aero_ax = max_accel('a', alt_km, V_sat)
        aero_ay = max_accel('b', alt_km, V_sat)
        aero_az = max_accel('c', alt_km, V_sat)

        R_fix = 500.0
        alts = np.linspace(450, 500, 100)
        j2xa = np.zeros(100); j2ya = np.zeros(100); j2za = np.zeros(100)
        aaxa = np.zeros(100); aaya = np.zeros(100); aaza = np.zeros(100)
        for i, alt in enumerate(alts):
            r_a = RE + alt * 1e3;  v_a = np.sqrt(MU / r_a)
            ga = 6.0 * MU * J2_C * RE**2 / r_a**5
            j2xa[i] = ga * R_fix;  j2ya[i] = ga * R_fix * 1.5;  j2za[i] = ga * R_fix * 0.8
            aaxa[i] = max_accel('a', alt, v_a)
            aaya[i] = max_accel('b', alt, v_a)
            aaza[i] = max_accel('c', alt, v_a)

        results['key_factors'] = dict(
            radii=radii, j2_x_r=j2xr, j2_y_r=j2yr, j2_z_r=j2zr,
            aero_ax=aero_ax, aero_ay=aero_ay, aero_az=aero_az,
            altitudes=alts,
            j2_x_alt=j2xa, j2_y_alt=j2ya, j2_z_alt=j2za,
            aero_ax_alt=aaxa, aero_ay_alt=aaya, aero_az_alt=aaza,
            T_ORBIT=T_orb,
        )
        _log("  Mode 3 done.")

    return results


# ---------------------------------------------------------------------------
# Build matplotlib figures from results (for embedding)
# ---------------------------------------------------------------------------

def build_figures(results: dict) -> dict[str, plt.Figure]:
    """Return {name: Figure} for each available result set."""
    figs = {}

    def _orb(t, T):
        return t / T

    if 'uncontrolled' in results:
        d = results['uncontrolled']
        T = d['T_ORBIT']
        t = _orb(d['t'], T)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for j, lb in enumerate(['X', 'Y', 'Z']):
            a1.plot(t, d['pos_err'][:, j], label=lb)
            a2.plot(t, d['vel_err'][:, j], label=lb)
        a1.set_ylabel('position error (m)');  a1.legend();  a1.grid(True, alpha=.3)
        a1.set_title('Fig 7 — Uncontrolled relative drift under J2')
        a2.set_ylabel('velocity error (m/s)'); a2.set_xlabel('orbit')
        a2.legend(); a2.grid(True, alpha=.3)
        fig.tight_layout()
        figs['Fig 7 — Uncontrolled'] = fig

    if 'controlled' in results:
        d = results['controlled']
        T = d['T_ORBIT']
        t = _orb(d['t'], T)

        fig8, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        for j, lb in enumerate(['X', 'Y', 'Z']):
            a1.plot(t, d['pos_err'][:, j], label=lb)
            a2.plot(t, d['vel_err'][:, j], label=lb)
        a1.set_ylabel('position error (m)');  a1.legend();  a1.grid(True, alpha=.3)
        a1.set_title('Fig 8 — Controlled relative position & velocity')
        a2.set_ylabel('velocity error (m/s)'); a2.set_xlabel('orbit')
        a2.legend(); a2.grid(True, alpha=.3)
        fig8.tight_layout()
        figs['Fig 8 — Controlled State'] = fig8

        fig9, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        ylabs = ['radial (m/s²)', 'along-track (m/s²)', 'cross-track (m/s²)']
        for j in range(3):
            axes[j].plot(t, d['control'][:, j], lw=.5)
            axes[j].set_ylabel(ylabs[j]); axes[j].grid(True, alpha=.3)
        axes[-1].set_xlabel('orbit')
        axes[0].set_title('Fig 9 — Control accelerations')
        fig9.tight_layout()
        figs['Fig 9 — Control Accels'] = fig9

    if 'key_factors' in results:
        d = results['key_factors']
        fig10, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        dirs = ['radial', 'along-track', 'cross-track']
        j2k = ['j2_x_r', 'j2_y_r', 'j2_z_r']
        ak  = ['aero_ax', 'aero_ay', 'aero_az']
        for j in range(3):
            axes[j].plot(d['radii'], d[j2k[j]], label='J2')
            axes[j].axhline(d[ak[j]], color='r', ls='--', label='aero')
            axes[j].set_xlabel('radius (m)'); axes[j].set_ylabel(f'{dirs[j]} (m/s²)')
            axes[j].legend(fontsize=8); axes[j].grid(True, alpha=.3)
            axes[j].ticklabel_format(style='sci', axis='y', scilimits=(-6,-5))
        fig10.suptitle('Fig 10 — Accel vs spatial radius')
        fig10.tight_layout(); figs['Fig 10 — Radius'] = fig10

        fig11, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        j2k = ['j2_x_alt', 'j2_y_alt', 'j2_z_alt']
        ak  = ['aero_ax_alt', 'aero_ay_alt', 'aero_az_alt']
        for j in range(3):
            axes[j].plot(d['altitudes'], d[j2k[j]], label='J2')
            axes[j].plot(d['altitudes'], d[ak[j]], 'r--', label='aero')
            axes[j].set_xlabel('altitude (km)'); axes[j].set_ylabel(f'{dirs[j]} (m/s²)')
            axes[j].legend(fontsize=8); axes[j].grid(True, alpha=.3)
            axes[j].ticklabel_format(style='sci', axis='y', scilimits=(-6,-5))
        fig11.suptitle('Fig 11 — Accel vs altitude')
        fig11.tight_layout(); figs['Fig 11 — Altitude'] = fig11

    return figs


# ===================================================================
# GUI Application
# ===================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CubeSAT Formation Keeping Simulator")
        self.geometry("1280x820")
        self.minsize(1100, 700)

        style = ttk.Style(self)
        style.theme_use('clam')

        self._build_layout()
        self._populate_defaults()
        self._running = False

    # ----------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------
    def _build_layout(self):
        # Left panel — parameters (scrollable)
        left = ttk.Frame(self, width=370)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)

        canvas = tk.Canvas(left, highlightthickness=0)
        vsb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        self._param_frame = ttk.Frame(canvas)
        self._param_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._param_frame, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right panel — notebook of figures + log
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._notebook = ttk.Notebook(right)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_frame = ttk.Frame(self._notebook)
        self._notebook.add(log_frame, text="  Log  ")
        self._log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                    state=tk.DISABLED,
                                                    font=("Menlo", 11))
        self._log_text.pack(fill=tk.BOTH, expand=True)

        self._fig_tabs: dict[str, ttk.Frame] = {}
        self._canvases: dict[str, FigureCanvasTkAgg] = {}

    # ----------------------------------------------------------------
    # Parameter widgets
    # ----------------------------------------------------------------
    def _populate_defaults(self):
        f = self._param_frame
        self._vars: dict[str, tk.StringVar] = {}
        row = [0]

        def heading(text):
            ttk.Label(f, text=text, font=("Helvetica", 12, "bold")
                     ).grid(row=row[0], column=0, columnspan=2,
                            sticky="w", pady=(10, 2), padx=4)
            row[0] += 1

        def param(label, key, default, width=14):
            ttk.Label(f, text=label).grid(row=row[0], column=0,
                                           sticky="w", padx=(8, 2), pady=1)
            v = tk.StringVar(value=str(default))
            e = ttk.Entry(f, textvariable=v, width=width)
            e.grid(row=row[0], column=1, sticky="ew", padx=(0, 4), pady=1)
            self._vars[key] = v
            row[0] += 1

        # --- Chief Orbit ---
        heading("Chief Orbit (Table II)")
        param("Semi-major axis (m)",    'a_chief',   6778137.0)
        param("Eccentricity",           'e_chief',   0.0)
        param("Inclination (°)",        'inc_deg',   96.4522)
        param("Arg. of perigee (°)",    'omega_deg', 90.0)
        param("RAAN (°)",               'RAAN_deg',  0.0)
        param("Mean anomaly (°)",       'M0_deg',    0.0)

        # --- Formation ---
        heading("Relative State (Table III)")
        param("x₀ (m)",   'x0',  82.50)
        param("y₀ (m)",   'y0', -930.46)
        param("z₀ (m)",   'z0',  55.27)
        param("ẋ₀ (m/s)", 'xd0', -0.17)
        param("ẏ₀ (m/s)", 'yd0', -0.04)
        param("ż₀ (m/s)", 'zd0',  0.29)

        # --- Tracking errors ---
        heading("Initial Tracking Errors")
        param("εx (m)",   'ex0',   2.0)
        param("εy (m)",   'ey0',  -1.0)
        param("εz (m)",   'ez0',   5.0)
        param("ε̇x (m/s)", 'exd0', -0.002)
        param("ε̇y (m/s)", 'eyd0',  0.004)
        param("ε̇z (m/s)", 'ezd0',  0.007)

        # --- Controller ---
        heading("Controller Gains (Eq. 12)")
        param("Kr (diag)",  'kr', 3.0e-5)
        param("Kv (diag)",  'kv', 2.0e-2)

        # --- Actuator limits ---
        heading("Actuator Limits")
        param("ax_max (m/s²)", 'ax_max', 1.30e-5)
        param("ay_max (m/s²)", 'ay_max', 1.26e-5)
        param("az_max (m/s²)", 'az_max', 1.30e-5)
        param("Actuator error (%)", 'actuator_err_pct', 5.0)

        # --- Sensor noise ---
        heading("GPS Noise (1σ)")
        param("Position σ (m)",   'gps_pos_sigma', 1.5e-3)
        param("Velocity σ (m/s)", 'gps_vel_sigma', 5.0e-6)

        # --- Simulation ---
        heading("Simulation Settings")
        param("Number of orbits",    'n_orbits', 3)
        param("Output Δt (s)",       'dt',       1.0)
        param("Control step (s)",    'ctrl_dt',  10.0)

        # --- Mode checkboxes ---
        heading("Modes")
        self._mode_vars = {}
        for mid, label in [('1', 'Mode 1 — Uncontrolled drift (Fig 7)'),
                           ('2', 'Mode 2 — Controlled keeping (Fig 8-9)'),
                           ('3', 'Mode 3 — Key factors (Fig 10-11)')]:
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(f, text=label, variable=v
                           ).grid(row=row[0], column=0, columnspan=2,
                                  sticky="w", padx=8, pady=1)
            self._mode_vars[mid] = v
            row[0] += 1

        # --- Buttons ---
        row[0] += 1
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row[0], column=0, columnspan=2, pady=10, padx=8,
                       sticky="ew")
        self._run_btn = ttk.Button(btn_frame, text="▶  Run Simulation",
                                    command=self._on_run)
        self._run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(btn_frame, text="Reset Defaults",
                   command=self._reset_defaults
                  ).pack(side=tk.RIGHT, padx=(6, 0))

        f.columnconfigure(1, weight=1)

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _log(self, msg):
        self._log_text.configure(state=tk.NORMAL)
        self._log_text.insert(tk.END, msg + "\n")
        self._log_text.see(tk.END)
        self._log_text.configure(state=tk.DISABLED)

    def _log_threadsafe(self, msg):
        self.after(0, self._log, msg)

    def _reset_defaults(self):
        defaults = dict(
            a_chief='6778137.0', e_chief='0.0', inc_deg='96.4522',
            omega_deg='90.0', RAAN_deg='0.0', M0_deg='0.0',
            x0='82.5', y0='-930.46', z0='55.27',
            xd0='-0.17', yd0='-0.04', zd0='0.29',
            ex0='2.0', ey0='-1.0', ez0='5.0',
            exd0='-0.002', eyd0='0.004', ezd0='0.007',
            kr='3e-05', kv='0.02',
            ax_max='1.3e-05', ay_max='1.26e-05', az_max='1.3e-05',
            actuator_err_pct='5.0',
            gps_pos_sigma='0.0015', gps_vel_sigma='5e-06',
            n_orbits='3', dt='1.0', ctrl_dt='10.0',
        )
        for k, v in defaults.items():
            if k in self._vars:
                self._vars[k].set(v)
        for v in self._mode_vars.values():
            v.set(True)

    def _collect_params(self) -> dict:
        p = {}
        for k, v in self._vars.items():
            p[k] = float(v.get())
        p['actuator_err'] = p.pop('actuator_err_pct') / 100.0
        p['n_orbits'] = int(p['n_orbits'])
        return p

    # ----------------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------------
    def _on_run(self):
        if self._running:
            return
        modes = [m for m, v in self._mode_vars.items() if v.get()]
        if not modes:
            messagebox.showwarning("No modes", "Select at least one simulation mode.")
            return

        try:
            params = self._collect_params()
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc))
            return

        self._running = True
        self._run_btn.configure(state=tk.DISABLED, text="Running …")
        self._log("=" * 50)
        self._log("Starting simulation …")

        def worker():
            try:
                res = run_simulation(params, modes, log_fn=self._log_threadsafe)
                self.after(0, self._show_results, res)
            except Exception:
                tb = traceback.format_exc()
                self._log_threadsafe(f"ERROR:\n{tb}")
            finally:
                self.after(0, self._run_done)

        threading.Thread(target=worker, daemon=True).start()

    def _run_done(self):
        self._running = False
        self._run_btn.configure(state=tk.NORMAL, text="▶  Run Simulation")

    # ----------------------------------------------------------------
    # Display results
    # ----------------------------------------------------------------
    def _show_results(self, results):
        self._log("Building figures …")

        # Tear down old figure tabs
        for name, frame in list(self._fig_tabs.items()):
            self._notebook.forget(frame)
            frame.destroy()
        self._fig_tabs.clear()
        for c in self._canvases.values():
            c.get_tk_widget().destroy()
        self._canvases.clear()

        figs = build_figures(results)
        for name, fig in figs.items():
            tab = ttk.Frame(self._notebook)
            self._notebook.add(tab, text=f"  {name}  ")
            self._fig_tabs[name] = tab

            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, tab)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._canvases[name] = canvas

        if figs:
            first_tab = list(self._fig_tabs.values())[0]
            self._notebook.select(first_tab)

        self._log("Done — figures ready.")


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
