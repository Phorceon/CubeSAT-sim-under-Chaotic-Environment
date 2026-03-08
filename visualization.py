"""
Visualization module — reproduces Figures 7-11 from Shao et al. (2017).

All plots use orbital periods as the x-axis unit and match the paper's
subplot layout.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from cubesat_sim.constants import T_ORBIT

FIGDIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGDIR, exist_ok=True)


def _orbits(t):
    """Convert time array [s] to orbital periods."""
    return t / T_ORBIT


# =========================================================================
# Figure 7 — Uncontrolled relative drift
# =========================================================================

def plot_uncontrolled(data: dict):
    t_orb = _orbits(data['t'])
    pe = data['pos_err']
    ve = data['vel_err']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(t_orb, pe[:, 0], label='X')
    ax1.plot(t_orb, pe[:, 1], label='Y')
    ax1.plot(t_orb, pe[:, 2], label='Z')
    ax1.set_ylabel('relative position error (m)')
    ax1.legend()
    ax1.set_title('Fig. 7 — Drift of relative position and velocity under J2')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_orb, ve[:, 0], label='X')
    ax2.plot(t_orb, ve[:, 1], label='Y')
    ax2.plot(t_orb, ve[:, 2], label='Z')
    ax2.set_xlabel('orbit')
    ax2.set_ylabel('relative velocity error (m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig7_uncontrolled.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Figure 8 — Controlled relative position and velocity
# =========================================================================

def plot_controlled_state(data: dict):
    t_orb = _orbits(data['t'])
    pe = data['pos_err']
    ve = data['vel_err']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(t_orb, pe[:, 0], label='X')
    ax1.plot(t_orb, pe[:, 1], label='Y')
    ax1.plot(t_orb, pe[:, 2], label='Z')
    ax1.set_ylabel('relative position error (m)')
    ax1.legend()
    ax1.set_title('Fig. 8 — Controlled relative position and velocity')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_orb, ve[:, 0], label='X')
    ax2.plot(t_orb, ve[:, 1], label='Y')
    ax2.plot(t_orb, ve[:, 2], label='Z')
    ax2.set_xlabel('orbit')
    ax2.set_ylabel('relative velocity error (m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig8_controlled_state.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Figure 9 — Control accelerations
# =========================================================================

def plot_control_accels(data: dict):
    t_orb = _orbits(data['t'])
    u = data['control']

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    labels = [
        'radial-track control acceleration (m/s²)',
        'along-track control acceleration (m/s²)',
        'cross-track control acceleration (m/s²)',
    ]
    titles = ['Fig. 9a', 'Fig. 9b', 'Fig. 9c']

    for j in range(3):
        axes[j].plot(t_orb, u[:, j], linewidth=0.5)
        axes[j].set_ylabel(labels[j])
        axes[j].set_title(titles[j])
        axes[j].grid(True, alpha=0.3)

    axes[-1].set_xlabel('orbit')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig9_control_accels.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Figure 10 — J2 vs aerodynamic acceleration vs spatial radius
# =========================================================================

def plot_radius_analysis(data: dict):
    r = data['radii']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    dirs = ['x', 'y', 'z']
    dir_labels = ['radial-track', 'along-track', 'cross-track']
    j2_keys = ['j2_x_r', 'j2_y_r', 'j2_z_r']
    aero_keys = ['aero_ax', 'aero_ay', 'aero_az']

    for j in range(3):
        axes[j].plot(r, data[j2_keys[j]], label='J2 acceleration')
        axes[j].axhline(data[aero_keys[j]], color='r', ls='--',
                         label='aerodynamic acceleration')
        axes[j].set_xlabel('radius (m)')
        axes[j].set_ylabel(f'{dir_labels[j]} acceleration (m/s²)')
        axes[j].legend(fontsize=8)
        axes[j].grid(True, alpha=0.3)
        axes[j].ticklabel_format(style='sci', axis='y', scilimits=(-6, -5))

    fig.suptitle('Fig. 10 — Differential acceleration comparison vs spatial radius',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig10_radius.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Figure 11 — J2 vs aerodynamic acceleration vs altitude
# =========================================================================

def plot_altitude_analysis(data: dict):
    alt = data['altitudes']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    dirs = ['x', 'y', 'z']
    dir_labels = ['radial-track', 'along-track', 'cross-track']
    j2_keys = ['j2_x_alt', 'j2_y_alt', 'j2_z_alt']
    aero_keys = ['aero_ax_alt', 'aero_ay_alt', 'aero_az_alt']

    for j in range(3):
        axes[j].plot(alt, data[j2_keys[j]], label='J2 acceleration')
        axes[j].plot(alt, data[aero_keys[j]], 'r--',
                      label='aerodynamic acceleration')
        axes[j].set_xlabel('altitude (km)')
        axes[j].set_ylabel(f'{dir_labels[j]} acceleration (m/s²)')
        axes[j].legend(fontsize=8)
        axes[j].grid(True, alpha=0.3)
        axes[j].ticklabel_format(style='sci', axis='y', scilimits=(-6, -5))

    fig.suptitle('Fig. 11 — Differential acceleration comparison vs altitude',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'fig11_altitude.png'), dpi=150)
    plt.close(fig)


# =========================================================================
# Top-level dispatcher
# =========================================================================

def plot_all(results: dict):
    """Generate all available plots from simulation results."""
    if 'uncontrolled' in results:
        print("  Plotting Figure 7 (uncontrolled drift)...")
        plot_uncontrolled(results['uncontrolled'])

    if 'controlled' in results:
        print("  Plotting Figure 8 (controlled state)...")
        plot_controlled_state(results['controlled'])
        print("  Plotting Figure 9 (control accelerations)...")
        plot_control_accels(results['controlled'])

    if 'key_factors' in results:
        print("  Plotting Figure 10 (radius analysis)...")
        plot_radius_analysis(results['key_factors'])
        print("  Plotting Figure 11 (altitude analysis)...")
        plot_altitude_analysis(results['key_factors'])

    print(f"  All figures saved to {FIGDIR}/")
