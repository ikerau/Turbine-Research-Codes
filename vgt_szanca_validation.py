"""
vgt_szanca_validation.py

Validates off_design.py variable-area turbine model against NASA cold-air data:

  Baseline (100% area)  — NASA TN D-4389 (Szanca et al. 1968)
  70% stator area       — NASA TM X-1697 (Schum, Szanca & Prust 1968)

Same turbine geometry; stator pivoted toward tangential for 30% area reduction.
No stator tip/hub sealing in the test (no leakage gap losses modelled).

Experimental benchmarks:
  100% area: eta_ts = 0.923 at PR_ts = 1.751, W_design = 17 Btu/lb
  70%  area: eta_ts = 0.840 at PR_ts = 1.860  (same design work output)
  Delta_eta = -8.3 pp  driven by +19.3 deg positive rotor incidence & reduced reaction.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import cantera as ct

from off_design_v2 import extract_throat, rotate_stator, build_performance_curve_moffitt
from units import Q_

# ── Annulus geometry (TN D-4389, 30-inch tip diameter) ────────────────────────
R_tip  = 0.381           # m  (15 in * 0.0254)
R_hub  = R_tip * 0.7333  # m  (hub/tip = 0.7333)
R_mean = R_tip * 0.8666  # m
span   = R_tip - R_hub   # m  = 0.1016 m

Rm_in  = R_mean / 0.0254  # mean radius [inches] for extract_throat / rotate_stator

# ── Test conditions (cold air, TN D-4389 / TM X-1697) ─────────────────────────
Tt_K   = 306      # K   (midpoint of 303.8–310 K test range)
Pt_Pa  = 1.0159e5   # Pa  (30 in Hg absolute)
N_rpm  = Q_(4407.4, 'rpm')     # rpm (100% equivalent design speed; U_mean = 500 ft/s corrected)
theta_cr  = Tt_K / 288.15
delta_cr  = Pt_Pa / 101325.0


inlet = {"Tt": Q_(Tt_K, 'K'), "Pt": Q_(Pt_Pa, 'Pa')}

gas = ct.Solution('air.yaml')
gas.TP = Tt_K, Pt_Pa
composition = gas.X
RPM = N_rpm * np.sqrt(theta_cr)

# ── TN D-4389 Table II — mean section blade coordinates (r/r_tip = 0.8666) ────
# Format: [x_chord_in, y_lower_in, y_upper_in]
# x runs along the chord direction (at stagger angle phi from axial).
# y_L, y_U are thickness offsets from the chord line (blade-local normal).
# Rows where y_L was missing in the table (LE suction-side sweep) are filled
# by linear interpolation between the LE circle (x=0) and first tabulated point.

raw_stator = np.array([
    # x       y_L      y_U
    [0.000,  0.150,   0.150],   # LE (both surfaces = LE radius)
    [0.100,  0.120,   0.394],   # y_L interpolated (table entry: ---)
    [0.200,  0.091,   0.514],   # y_L interpolated
    [0.300,  0.061,   0.588],
    [0.400,  0.106,   0.636],
    [0.500,  0.145,   0.665],
    [0.600,  0.174,   0.676],
    [0.700,  0.196,   0.675],
    [0.800,  0.210,   0.663],
    [0.900,  0.219,   0.644],
    [1.000,  0.223,   0.619],
    [1.100,  0.221,   0.590],
    [1.200,  0.215,   0.560],
    [1.300,  0.205,   0.527],
    [1.400,  0.191,   0.492],
    [1.500,  0.175,   0.452],
    [1.600,  0.155,   0.410],
    [1.700,  0.133,   0.365],
    [1.800,  0.111,   0.319],
    [1.900,  0.086,   0.267],
    [2.000,  0.060,   0.214],
    [2.100,  0.033,   0.157],
    [2.200,  0.005,   0.096],
    [2.263,  0.035,   0.035],   # TE
])

raw_rotor = np.array([
    [0.000,  0.150,   0.150],   # LE
    [0.100,  0.126,   0.358],   # y_L interpolated
    [0.200,  0.102,   0.490],   # y_L interpolated
    [0.300,  0.078,   0.593],
    [0.400,  0.153,   0.668],
    [0.500,  0.217,   0.722],
    [0.600,  0.267,   0.765],
    [0.700,  0.307,   0.774],
    [0.800,  0.339,   0.781],
    [0.900,  0.360,   0.775],
    [1.000,  0.373,   0.759],
    [1.100,  0.377,   0.734],
    [1.200,  0.373,   0.702],
    [1.300,  0.362,   0.664],
    [1.400,  0.342,   0.620],
    [1.500,  0.315,   0.573],
    [1.600,  0.283,   0.523],
    [1.700,  0.244,   0.466],
    [1.800,  0.203,   0.407],
    [1.900,  0.159,   0.346],
    [2.000,  0.113,   0.277],
    [2.100,  0.067,   0.204],
    [2.200,  0.022,   0.125],
    [2.290,  0.035,   0.035],   # TE
])
Cd = 1.00
# ── Blade parameters (TN D-4389, mean section) ────────────────────────────────
phi_s  = 41.03   # stator stagger [deg]
phi_r  = 22.87   # rotor  stagger [deg]s

# Blade counts derived from Table I solidity at mean radius (σ = c / pitch_mean).
# TN D-4389 Table I: stator σ = 1.385, rotor σ = 1.71; chord in inches.
CHORD_S_IN  = 2.263;  SOLIDITY_S = 1.385   # stator
CHORD_R_IN  = 2.290;  SOLIDITY_R = 1.71    # rotor
pitch_s_in  = CHORD_S_IN / SOLIDITY_S      # = 1.634 in
pitch_r_in  = CHORD_R_IN / SOLIDITY_R      # = 1.339 in
N_s = round(2.0 * np.pi * Rm_in / pitch_s_in)   # ≈ 50
N_r = round(2.0 * np.pi * Rm_in / pitch_r_in)   # ≈ 61
print(f"  Blade counts from solidity:  N_stator={N_s}  N_rotor={N_r}")

# Blade angles — from TN D-4389 velocity diagram (mean section).
# ADJUST these if design-point prediction deviates by more than ~2 pp.
STATOR_BETA2 = 67    # deg from axial; stator exit metal angle
ROTOR_BETA1  = 36.38    # deg from axial; rotor inlet metal angle (design incidence=0)
ROTOR_BETA2  = 58.26   # deg from axial; rotor exit metal angle

# Axial chords from chord × cos(stagger) [cm]
Cax_s = CHORD_S_IN * np.cos(np.radians(phi_s)) * 2.54
Cax_r = CHORD_R_IN * np.cos(np.radians(phi_r)) * 2.54

# ── Extract geometric throat from coordinates ──────────────────────────────────
print("\n" + "="*60)
print("  TN D-4389 Throat Extraction (mean section)")
print("="*60)
throat_s_mm, _ = extract_throat(raw_stator, phi_s, Rm_in, N_s,
                                 plot=False, title="Szanca Stator (mean)")
throat_r_mm, _ = extract_throat(raw_rotor,  phi_r, Rm_in, N_r,
                                 plot=False, title="Szanca Rotor (mean)")

# ── Geometry dicts ─────────────────────────────────────────────────────────────
stator = {
    "Beta_1":     Q_(0.0,          'deg'),
    "Beta_2":     Q_(STATOR_BETA2, 'deg'),
    "stagger":    Q_(phi_s,        'deg'),
    "throat":     Q_(throat_s_mm*Cd,  'mm'),
    "TE_radius":  Q_(0.0889,       'cm'),   # 0.035 in × 2.54
    "LE_radius":  Q_(0.381,        'cm'),   # 0.150 in × 2.54
    "Chord_ax":   Q_(Cax_s,        'cm'),
    "Blade_N":    N_s,
    "Tip_Gap":    Q_(0.0,          'mm'),
    "R_o_inlet":  Q_(R_tip,        'm'),
    "R_i_inlet":  Q_(R_hub,        'm'),
    "R_o_outlet": Q_(R_tip,        'm'),
    "R_i_outlet": Q_(R_hub,        'm'),
    "inlet_wedge": Q_(15,          'deg'),
    "exit_wedge": Q_(3.0,          'deg'),
    "zeta_ung":   Q_(5.0,          'deg'),
    "rotating":   False,
}

rotor = {
    "Beta_1":     Q_(ROTOR_BETA1,  'deg'),
    "Beta_2":     Q_(ROTOR_BETA2,  'deg'),
    "stagger":    Q_(phi_r,        'deg'),
    "throat":     Q_(throat_r_mm*Cd,  'mm'),
    "TE_radius":  Q_(0.0889,       'cm'),
    "LE_radius":  Q_(0.381,        'cm'),
    "Chord_ax":   Q_(Cax_r,        'cm'),
    "Blade_N":    N_r,
    "Tip_Gap":    Q_(0.0762,       'cm'),   
    "R_o_inlet":  Q_(R_tip,        'm'),
    "R_i_inlet":  Q_(R_hub,        'm'),
    "R_o_outlet": Q_(R_tip,        'm'),
    "R_i_outlet": Q_(R_hub,        'm'),
    "inlet_wedge": Q_(15,          'deg'),
    "exit_wedge": Q_(3.0,          'deg'),
    "zeta_ung":   Q_(5.0,          'deg'),
    "rotating":   True,
}

# ── Pressure sweep: covers PR_ts ≈ 1.2 to 2.6 ────────────────────────────────
# p2_vals are rotor exit static pressures [Pa].
# PR stored in curve is total-to-total (Pt_in / Pt_exit).
PR_LO, PR_HI, N_PTS = 1.10, 3.00, 50
p2_vals = Pt_Pa / np.linspace(PR_LO, PR_HI, N_PTS)

# ── Baseline sweep: 100% stator area ──────────────────────────────────────────
print("\n" + "="*60)
print("  Baseline: 100% stator area")
print("="*60)
gas.TPX = Tt_K, Pt_Pa, composition
curve_100 = build_performance_curve_moffitt(
    stator, rotor, inlet, gas, composition, RPM, p2_vals, N_PTS
)

# ── Find rotation angles for 70% and 130% stator throat ───────────────────────
print("\n--- Finding pivot angles for 70% and 130% stator area ---")
throat_design = throat_s_mm

def _throat_residual(theta, frac):
    t_mm, _, _, _, _ = rotate_stator(raw_stator, theta, phi_s, Rm_in, N_s)
    return t_mm / throat_design - frac

# Negative theta = toward tangential = closes (reduces area)
# Positive theta = toward axial     = opens  (increases area)
theta_70  = brentq(_throat_residual, -25.0, -0.5, args=(0.70,), xtol=1e-3)
theta_130 = brentq(_throat_residual,   0.5,  25.0, args=(1.30,), xtol=1e-3)
print(f"  theta_70  = {theta_70:+.3f} deg  (TM X-1697 paper: -7.79 deg)")
print(f"  theta_130 = {theta_130:+.3f} deg  (TM X-1697 paper: +8.44 deg)")

def _make_stator_var(theta):
    t_mm, B1, B2, _, phi_new = rotate_stator(
        raw_stator, theta, phi_s, Rm_in, N_s, Beta_2_design=STATOR_BETA2)
    s = dict(stator)
    s.update({
        "throat":  Q_(t_mm*Cd,    'mm'),
        "Beta_1":  Q_(B1,      'deg'),
        "Beta_2":  Q_(B2,      'deg'),
        "stagger": Q_(phi_new, 'deg'),
    })
    print(f"  theta={theta:+.2f} deg:  throat={t_mm:.3f} mm  "
          f"beta2 {STATOR_BETA2:.1f}->{B2:.2f} deg")
    return s, t_mm

stator_70,  throat_70  = _make_stator_var(theta_70)
stator_130, throat_130 = _make_stator_var(theta_130)

# ── Area sweeps ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  70% stator area  (rotor unchanged)")
print("="*60)
gas.TPX = Tt_K, Pt_Pa, composition
curve_70 = build_performance_curve_moffitt(
    stator_70, rotor, inlet, gas, composition, RPM, p2_vals, N_PTS
)

print("\n" + "="*60)
print("  130% stator area  (rotor unchanged)")
print("="*60)
gas.TPX = Tt_K, Pt_Pa, composition
curve_130 = build_performance_curve_moffitt(
    stator_130, rotor, inlet, gas, composition, RPM, p2_vals, N_PTS
)

# ── Convert total-to-total -> total-to-static using exit Mach number ───────────
# PR_tt = Pt_in / Pt_exit  (stored in curve as 'PR')
# PR_ts = PR_tt * (1 + (gam-1)/2 * M_exit^2)^(gam/(gam-1))
# eta_ts = eta_tt * (1 - PR_tt^(-(gam-1)/gam)) / (1 - PR_ts^(-(gam-1)/gam))
GAM = 1.4
EXP = (GAM - 1.0) / GAM   # = 0.2857

def curve_to_ts(curve):
    PR_tt_arr  = np.array([pt["PR"]   for pt in curve])
    eta_tt_arr = np.array([pt["eta"]  for pt in curve])
    mdot_arr   = np.array([pt["mdot"] for pt in curve])
    M_exit_arr = np.array([float(pt["rotor_out"]["M_abs"]) for pt in curve])
    incid_arr  = np.array([
        float(np.degrees(np.arctan2(
            float(pt["rotor_in"]["W_theta"].to('m/s').magnitude),
            float(pt["rotor_in"]["Cm"].to('m/s').magnitude)
        ))) - ROTOR_BETA1
        for pt in curve
    ])
    # Total-to-static PR
    pr_ratio   = (1.0 + 0.5 * (GAM - 1.0) * M_exit_arr**2) ** (GAM / (GAM - 1.0))
    PR_ts_arr  = PR_tt_arr * pr_ratio
    # Total-to-static efficiency
    dh_tt = 1.0 - PR_tt_arr**(-EXP)   # proportional to isentropic work (tt)
    dh_ts = 1.0 - PR_ts_arr**(-EXP)   # proportional to isentropic work (ts)
    eta_ts_arr = np.where(dh_ts > 0, eta_tt_arr * dh_tt / dh_ts, eta_tt_arr)
    return PR_ts_arr, eta_ts_arr, PR_tt_arr, eta_tt_arr, mdot_arr, incid_arr

PR_ts_100, eta_ts_100, PR_tt_100, eta_tt_100, mdot_100, incid_100 = curve_to_ts(curve_100)
PR_ts_70,  eta_ts_70,  PR_tt_70,  eta_tt_70,  mdot_70,  incid_70  = curve_to_ts(curve_70)
PR_ts_130, eta_ts_130, PR_tt_130, eta_tt_130, mdot_130, incid_130 = curve_to_ts(curve_130)

# Corrected mass flow (standard day, lbm/s — matches paper convention)
KG_TO_LBM = 2.20462
Wc_100 = mdot_100 * np.sqrt(theta_cr) / delta_cr * KG_TO_LBM
Wc_70  = mdot_70  * np.sqrt(theta_cr) / delta_cr * KG_TO_LBM
Wc_130 = mdot_130 * np.sqrt(theta_cr) / delta_cr * KG_TO_LBM

# ── Experimental reference points ──────────────────────────────────────────────
# NASA paper reports total-to-total efficiency (eta_tt) at total-to-total PR (PR_tt).
# TN D-4389: baseline turbine, design-point (17 Btu/lb work output)
#   PR_tt = 1.751  eta_tt = 0.923
# TM X-1697: 70% stator area, same design work output
#   PR_tt = 1.860  eta_tt = 0.840  (Delta_eta = -8.3 pp)
# TM X-1663: 130% stator area, design work output (referenced by TM X-1697)
#   eta_tt = 0.897  (Delta_eta = -2.6 pp); PR_tt not precisely tabulated
EXP_100_PR, EXP_100_ETA = 1.751, 0.923
EXP_70_PR,  EXP_70_ETA  = 1.860, 0.840
EXP_130_ETA              = 0.897   # PR not precisely given in TM X-1697 ref.

# ── Loss breakdown diagnostic at design-point PRs ─────────────────────────────
def _closest_pt(curve, PR_target):
    PRs = np.array([pt["PR"] for pt in curve])
    return curve[int(np.argmin(np.abs(PRs - PR_target)))]

def _print_loss_table(label, pt):
    ls = pt.get("loss_s", {})
    lr = pt.get("loss_r", {})
    PR  = pt["PR"]
    eta = pt["eta"]
    Ms  = pt.get("M_stator_exit", float("nan"))
    Mr  = pt.get("M_rotor_exit",  float("nan"))
    i_r = pt.get("i_rotor",       float("nan"))
    Ps_in  = pt.get("Ps_rotor_in",  float("nan"))
    Ps_out = pt.get("Ps_rotor_out", float("nan"))
    # Degree-of-reaction proxy: static-pressure ratio across rotor
    # Positive reaction → Ps_out < Ps_in; negative → Ps_out > Ps_in
    DoR_proxy = 1.0 - Ps_out / Ps_in if Ps_in > 0 else float("nan")
    print(f"\n  {'='*60}")
    print(f"  {label}   PR_tt={PR:.3f}   eta_tt={eta*100:.2f}%")
    print(f"  {'='*60}")
    print(f"  Stator exit Mach        : {Ms:.3f}")
    print(f"  Rotor exit Mach (rel)   : {Mr:.3f}")
    print(f"  Rotor incidence         : {i_r:+.2f} deg")
    print(f"  Rotor DoR proxy (1-Ps2/Ps1): {DoR_proxy:.3f}  {'[NEGATIVE RXN]' if DoR_proxy < 0 else ''}")
    print(f"  --- Stator losses (zeta) ---")
    print(f"    Profile   : {ls.get('zeta_s', 0):.5f}")
    print(f"    Secondary : {ls.get('zeta_f', 0):.5f}")
    print(f"    Leakage   : {ls.get('zeta_l', 0):.5f}")
    print(f"    TOTAL     : {ls.get('zeta_tot', 0):.5f}   r_loss={ls.get('r_loss_new', 0):.5f}")
    print(f"  --- Rotor losses (zeta) ---")
    print(f"    Profile   : {lr.get('zeta_s', 0):.5f}")
    print(f"    Secondary : {lr.get('zeta_f', 0):.5f}")
    print(f"    Leakage   : {lr.get('zeta_l', 0):.5f}")
    print(f"    TOTAL     : {lr.get('zeta_tot', 0):.5f}   r_loss={lr.get('r_loss_new', 0):.5f}")

print("\n" + "="*60)
print("  LOSS BREAKDOWN DIAGNOSTIC")
print("="*60)
pt100 = _closest_pt(curve_100, EXP_100_PR)
pt70  = _closest_pt(curve_70,  EXP_70_PR)
_print_loss_table("100% area  (exp eta=92.3%)", pt100)
_print_loss_table(" 70% area  (exp eta=84.0%)", pt70)
print()

# NASA TM X-1697 Fig. 8: 70% area corrected flow vs PR_tt at 100% speed
NASA_70_PR_tt = np.array([1.338760812076386, 1.441527356199612, 1.4838218986767862,
                           1.5861601780551953, 1.6752434936352065, 1.845202566917884,
                           1.9223641476372306, 2.2054768845682062, 2.3932029966393116,
                           2.528789575492283,  2.7321337550430056, 2.9485247141411732,
                           3.263992682596707])
NASA_70_Wc    = np.array([18.06574009501214,  21.531738387051547, 23.37927103950706,
                           25.48417405174501,  26.89120946552487,  28.47746420657219,
                           29.28099411318061,  30.059494567483558, 30.112299319182874,
                           30.17144064108611,  30.146728017290833, 30.158239453161276,
                           30.195519607860994])

# ── Interpolate predicted eta_tt at the experimental design-point PR_tt ───────
def _interp_at(PR_arr, eta_arr, PR_target):
    valid = np.isfinite(PR_arr) & np.isfinite(eta_arr)
    if valid.sum() < 2:
        return np.nan
    f = interp1d(PR_arr[valid], eta_arr[valid], bounds_error=False, fill_value=np.nan)
    return float(f(PR_target))

calc_eta_tt_100_at_des = _interp_at(PR_tt_100, eta_tt_100, EXP_100_PR)
calc_eta_tt_70_at_des  = _interp_at(PR_tt_70,  eta_tt_70,  EXP_70_PR)
calc_eta_tt_130_at_des = float(np.nanmax(eta_tt_130))   # peak (no exact PR given)
incid_70_at_des        = _interp_at(PR_tt_70,  incid_70,  EXP_70_PR)

# ── Figure 1: Efficiency overlay (η_tt) ───────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 6))
fig1.suptitle(
    "Variable-Area Turbine — Total-to-Total Efficiency vs PR\n"
    "Szanca et al.  (NASA TN D-4389 / TM X-1697 / TM X-1663)",
    fontsize=11, fontweight='bold'
)

C70  = 'firebrick'
C100 = 'steelblue'
C130 = 'seagreen'

ax1.plot(PR_tt_70,  eta_tt_70  * 100, '-',  color=C70,  lw=2.2,
         label=f'70% area  (θ={theta_70:+.1f}°, −30%)')
ax1.plot(PR_tt_100, eta_tt_100 * 100, '-',  color=C100, lw=2.2,
         label='100% area  (design)')
ax1.plot(PR_tt_130, eta_tt_130 * 100, '-',  color=C130, lw=2.2,
         label=f'130% area  (θ={theta_130:+.1f}°, +30%)')

ax1.plot(EXP_70_PR,  EXP_70_ETA  * 100, 'v', color=C70,  ms=10, zorder=5,
         label=f'Exp 70%: η_tt={EXP_70_ETA*100:.1f}%  PR_tt={EXP_70_PR}  (TM X-1697)')
ax1.plot(EXP_100_PR, EXP_100_ETA * 100, 's', color=C100, ms=10, zorder=5,
         label=f'Exp 100%: η_tt={EXP_100_ETA*100:.1f}%  PR_tt={EXP_100_PR}  (TN D-4389)')
ax1.axhline(EXP_130_ETA * 100, color=C130, ls=':', lw=1.5,
            label=f'Exp 130%: η_tt={EXP_130_ETA*100:.1f}%  (PR_tt not tabulated)')

ax1.set_xlabel('PR  (total-to-total,  Pt_in / Pt_exit)', fontsize=11)
ax1.set_ylabel('η_tt  (%)', fontsize=11)
ax1.set_xlim(1.1, 2.2)
ax1.set_ylim(78, 94)
ax1.legend(fontsize=8.5, loc='lower left')
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('szanca_efficiency_overlay.png', dpi=150)
print("Figure saved: szanca_efficiency_overlay.png")
plt.show()

# ── Figure 2: Corrected mass flow overlay ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))
fig2.suptitle(
    "Variable-Area Turbine — Corrected Mass Flow vs PR\n"
    "Szanca et al.  (NASA TN D-4389 / TM X-1697 / TM X-1663)",
    fontsize=11, fontweight='bold'
)

ax2.plot(PR_tt_70,  Wc_70,  '-', color=C70,  lw=2.2,
         label=f'70%  area  (θ={theta_70:+.1f}°,  −30%)')
ax2.plot(PR_tt_100, Wc_100, '-', color=C100, lw=2.2,
         label='100% area  (design)')
ax2.plot(PR_tt_130, Wc_130, '-', color=C130, lw=2.2,
         label=f'130% area  (θ={theta_130:+.1f}°,  +30%)')
ax2.scatter(NASA_70_PR_tt, NASA_70_Wc, color=C70, marker='o', s=40, zorder=5,
            label='Exp 70%  (TM X-1697 Fig. 8)')

ax2.set_xlabel('PR_tt  (total-to-total,  Pt_in / Pt_exit)', fontsize=11)
ax2.set_ylabel('Wc  (lbm/s,  standard day)', fontsize=11)
ax2.set_xlim(1.1, 3.4)
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('szanca_massflow_overlay.png', dpi=150)
print("Figure saved: szanca_massflow_overlay.png")
plt.show()

# ── Figure 3: Absolute exit swirl angle vs PR_tt (100% area) ─────────────────
# NASA TN D-4389: baseline turbine, absolute exit angle at various PR_tt values.
# Positive = co-rotation direction; negative = counter-rotation (turbine over-turns).
NASA_SWIRL_PR = np.array([
    1.3634, 1.4151, 1.4608, 1.4885,
    1.5884, 1.6413, 1.7112, 1.7365,
    1.7907, 1.8546, 1.8787, 1.9607, 2.0174,
])
NASA_SWIRL_DEG = np.array([
    18.416, 12.135,  7.374,  4.739,
    -3.469, -7.523, -12.288, -14.011,
    -17.357, -20.602, -21.718, -24.764, -26.693,
])

alpha_exit_100 = np.array([pt["alpha_exit"] for pt in curve_100])

fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.suptitle(
    "Absolute Exit Swirl Angle vs PR_tt — 100% Area (Baseline)\n"
    "Szanca et al.  (NASA TN D-4389)",
    fontsize=11, fontweight='bold',
)
ax3.plot(PR_tt_100, alpha_exit_100, '-', color=C100, lw=2.2, label='Model (100% area)')
ax3.scatter(NASA_SWIRL_PR, NASA_SWIRL_DEG, color='k', marker='s', s=45, zorder=5,
            label='Exp (TN D-4389)')
ax3.axhline(0, color='gray', lw=0.8, ls='--')
ax3.set_xlabel('PR_tt  (total-to-total)', fontsize=11)
ax3.set_ylabel('Absolute exit angle  (deg)', fontsize=11)
ax3.set_xlim(1.1, 2.2)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('szanca_swirl_comparison.png', dpi=150)
print("Figure saved: szanca_swirl_comparison.png")
plt.show()

# ── Validation summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  VALIDATION SUMMARY — TN D-4389 / TM X-1697 / TM X-1663")
print("="*60)
print(f"  theta_70  = {theta_70:+.3f} deg  (paper: -7.79 deg)")
print(f"  theta_130 = {theta_130:+.3f} deg  (paper: +8.44 deg)")
print(f"  Throat: {throat_design:.3f} mm (100%)  "
      f"-> {throat_70:.3f} mm (70%)  "
      f"-> {throat_130:.3f} mm (130%)")
print(f"  Rotor incidence at 70% area, design work: {incid_70_at_des:+.1f} deg"
      f"  (paper: +19.3 deg)")
print()
print(f"  {'Case':<12}  {'PR_tt':>8}  {'Calc_eta_tt':>12}  {'Exp_eta_tt':>11}  {'Delta_pp':>9}")
print(f"  {'-'*58}")
for lbl, pr, c_tt, e_tt in [
    ("130% area", None,       calc_eta_tt_130_at_des, EXP_130_ETA),
    ("100% area", EXP_100_PR, calc_eta_tt_100_at_des, EXP_100_ETA),
    ("70%  area", EXP_70_PR,  calc_eta_tt_70_at_des,  EXP_70_ETA),
]:
    pr_str = f"{pr:.3f}" if pr is not None else "  (peak)"
    c_tt_s = f"{c_tt*100:.2f}%" if np.isfinite(c_tt) else "    ---"
    diff   = (c_tt - e_tt) * 100 if np.isfinite(c_tt) else np.nan
    diff_s = f"{diff:+.2f} pp" if np.isfinite(diff) else "    ---"
    print(f"  {lbl:<12}  {pr_str:>8}  {c_tt_s:>12}  {e_tt*100:>10.1f}%  {diff_s:>9}")
print()
if np.isfinite(calc_eta_tt_100_at_des) and np.isfinite(calc_eta_tt_70_at_des):
    calc_delta_70  = (calc_eta_tt_70_at_des  - calc_eta_tt_100_at_des) * 100
    print(f"  Calc Delta_eta_tt (70%-100%)  = {calc_delta_70:+.2f} pp   Exp: -8.3 pp")
if np.isfinite(calc_eta_tt_130_at_des):
    calc_delta_130 = (calc_eta_tt_130_at_des - EXP_100_ETA) * 100
    print(f"  Calc Delta_eta_tt (130%-100%) = {calc_delta_130:+.2f} pp   Exp: -2.6 pp")
print("="*60)
print()
print("  Calibration guidance (if baseline eta_ts deviates by >2 pp from 0.923):")
print(f"    STATOR_BETA2 = {STATOR_BETA2} deg  ->  increase for more stator turning")
print(f"    ROTOR_BETA1  = {ROTOR_BETA1} deg  ->  adjust to match design incidence=0")
print(f"    ROTOR_BETA2  = {ROTOR_BETA2} deg  ->  adjust for exit swirl / reaction")
print("  Then re-run: only calibrate against baseline; 70% area must then match.")

# ══════════════════════════════════════════════════════════════════════════════
# Fit Pritchard metal angles to TN D-4389 blade coordinates
# ══════════════════════════════════════════════════════════════════════════════
from blade_geom import fit_blade_angles, raw_to_chord_local_m

print("\n" + "="*60)
print("  FIT PRITCHARD METAL ANGLES — Stator (TN D-4389)")
print("="*60)

x_s, y_s = raw_to_chord_local_m(raw_stator)

fixed_s = dict(
    LE_radius   = Q_(0.150, "in"),
    TE_radius   = Q_(0.035, "in"),
    Chord_ax    = Q_(Cax_s, "cm"),
    stagger     = Q_(phi_s,  "deg"),
    N_blades    = N_s,
    Radius      = Q_(R_mean, "m"),
)

fit_s = fit_blade_angles(
    x_s, y_s,
    fixed_s,
    beta_in_init       =   0.0,
    beta_out_init      = -66.84,
    beta_in_bounds     = (-10.0,  25.0),
    beta_out_bounds    = (-70, -70),
    inlet_wedge_init   = 18.0,
    inlet_wedge_bounds = (2.0, 30.0),
    exit_wedge_init    = 3.0,
    exit_wedge_bounds  = (1.0, 15.0),
    zeta_ung_init      = 6.5,
    zeta_ung_bounds    = (0.0, 40.0),
    throat_init        = throat_s_mm,
    throat_bounds      = (0.05 * throat_s_mm, 5.0 * throat_s_mm),
    verbose=True,
)

print("\n" + "="*60)
print("  FIT PRITCHARD METAL ANGLES — Rotor (TN D-4389)")
print("="*60)

x_r, y_r = raw_to_chord_local_m(raw_rotor)

fixed_r = dict(
    LE_radius   = Q_(0.150, "in"),
    TE_radius   = Q_(0.035, "in"),
    Chord_ax    = Q_(Cax_r, "cm"),
    stagger     = Q_(phi_r,  "deg"),
    N_blades    = N_r,
    Radius      = Q_(R_mean, "m"),
)

fit_r = fit_blade_angles(
    x_r, y_r,
    fixed_r,
    beta_in_init       =  ROTOR_BETA1,
    beta_out_init      = -ROTOR_BETA2,
    beta_in_bounds     = ( 0.0,  89.0),
    beta_out_bounds    = (-60, -60),
    inlet_wedge_init   = 15.0,
    inlet_wedge_bounds = (1.0, 30.0),
    exit_wedge_init    = 3.0,
    exit_wedge_bounds  = (1.0, 30.0),
    zeta_ung_init      = 6.5,
    zeta_ung_bounds    = (0.0, 40.0),
    throat_init        = throat_r_mm,
    throat_bounds      = (0.05 * throat_r_mm, 5.0 * throat_r_mm),
    verbose=True,
)

# ── Comparison plot (all in chord-local frame) ─────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Pritchard fit to TN D-4389 blade coordinates (chord-local frame)",
              fontsize=12, fontweight='bold')

for ax, (fit, label, beta1_tbl, beta2_tbl) in zip(axes, [
    (fit_s, "Stator", 0.0,         STATOR_BETA2),
    (fit_r, "Rotor",  ROTOR_BETA1, ROTOR_BETA2),
]):
    M2MM = 1e3
    ax.plot(fit["x_ref"] * M2MM, fit["y_ref"] * M2MM,
            'b-', lw=1.5, label="TN D-4389 (chord-local)")
    ax.plot(fit["x_fit"] * M2MM, fit["y_fit"] * M2MM,
            'r-', lw=2.2, label="Pritchard fit")

    bi = float(fit["beta_in_metal"].to('deg').magnitude)
    bo = float(fit["beta_out_metal"].to('deg').magnitude)
    ax.set_title(
        f"{label}\n"
        f"b_in = {bi:+.2f} deg  (table: {beta1_tbl:+.2f} deg)\n"
        f"b_out = {bo:.2f} deg  (table: {-beta2_tbl:.2f} deg)    "
        f"RMS = {fit['rms_mm']:.3f} mm",
        fontsize=9,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x chord [mm]")
    ax.set_ylabel("y normal [mm]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("szanca_blade_fit.png", dpi=150)
print("\nFigure saved: szanca_blade_fit.png")
plt.show()
