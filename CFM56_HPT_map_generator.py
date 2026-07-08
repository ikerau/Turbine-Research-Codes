"""
CFM56_HPT_map_generator.py
--------------------------
Sister file to Turbine_Map_Generator.py.

Generates HPT maps for the CFM56-7B27 engine:
  CFM56_HPT_map.py       — VGT (variable NGV area, 8 alpha slices)
  CFM56_HPT_map_fixed.py — Fixed stator (3 identical alpha slices for slinear)

Design conditions from cfm56_cycle.py DESIGN solve (TOC: 35 kft, M=0.8,
Fn=5961.9 lbf, T4=2857°R, OPR=30.09):

  Station        | Value
  ---------------|-----------------------
  Tt4 (HPT in)   | 2857.0 °R  = 1587.2 K
  Pt4 (HPT in)   | 146.609 psi = 1.011 MPa
  W_HPT_in       | 44.177 lbm/s = 20.04 kg/s  (burner exit, HPT inlet)
  HPT power      | 11,998 hp = 8.945 MW
  N_HP           | 14705.7 rpm
  PR_des         | 3.505 (total-to-total)
  η_des          | 0.903 (adiabatic)
  Wp_des         | 15.94 lbm/s  (pyCycle corrected flow at HPT inlet)
  NpMap_des      | 275.1 rpm    (N/sqrt(Tt_degR))

Tuning guide
------------
ψ (psi)  — loading coefficient = ΔH / U²
  • sets blade speed U and mean radius Rm
  • higher ψ → smaller blade, higher incidence → more loss
  • typical single-stage HPT: ψ ≈ 1.0–2.5

φ (phi)  — flow coefficient = Cm / U
  • sets axial velocity, controls blade angles and throat area
  • typical range: φ ≈ 0.4–0.6

Start with ψ=1.20, φ=0.52.  Run and check:
  • perf['Pt_ratio'] → target ≈ 3.5
  • perf['eta_tt']   → target ≈ 0.903
Adjust ψ/φ and blade geometry params until both are met.
pyCycle s_PR and s_eff scalars will trim remaining differences at run time.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy.interpolate import interp1d

from units import Q_, ureg
from forward_design import (
    design_point_kinematics, forward_design,
    evaluate_at_same_p2, print_same_p2_point,
    compare_fd_od,
)
from off_design import build_performance_curve_moffitt

# ── Constants ─────────────────────────────────────────────────────────────────
KG_TO_LBM  = 2.20462
HP_TO_W    = 745.699872

# =============================================================================
#  STEP 1 — CFM56 HPT CYCLE CONDITIONS  (from cfm56_cycle.py DESIGN solve)
# =============================================================================

# HPT inlet (= burner exit) — read directly from pyCycle DESIGN output
# Updated to match cfm56_cycle.py DESIGN converged values (burner.Fl_O:stat:W, hpt.pwr)
Tt4_K   = 2857.0 * (5.0/9.0)          # 1587.2 K
Pt4_Pa  = 145.626 * 6894.757          # 1,011,100 Pa
mdot_kg = (44.177 + 3.72358) / KG_TO_LBM          # 21.727 kg/s  (burner exit = HPT inlet)
pwr_W   = 11998.248 * HP_TO_W         # 8,945,320 W  (hpt.power from shaft balance)
Nmech   = 14460                      # rpm  (HP shaft design speed)

gas = ct.Solution('air.yaml')
gas.TP = Tt4_K, Pt4_Pa
composition = gas.X

Cp_des    = float(gas.cp_mass)
gamma_des = float(gas.cp_mass / gas.cv_mass)
R_des     = float(ct.gas_constant / gas.mean_molecular_weight)

print("=" * 62)
print("  CFM56 HPT MAP GENERATION PIPELINE")
print("=" * 62)
print(f"\n  CFM56 HPT design conditions (from cfm56_cycle.py):")
print(f"    Tt4  = {Tt4_K:.1f} K  ({2857.0:.0f} °R)")
print(f"    Pt4  = {Pt4_Pa/1e6:.3f} MPa  ({146.609:.3f} psia)")
print(f"    mdot = {mdot_kg:.2f} kg/s  ({mdot_kg*KG_TO_LBM:.2f} lbm/s)")
print(f"    Pwr  = {pwr_W/1e6:.3f} MW  ({pwr_W/HP_TO_W:.0f} hp)")
print(f"    N    = {Nmech:.1f} rpm")
print(f"    Target PR  = 3.505")
print(f"    Target eta = 0.903")
print(f"\n  Gas properties at HPT inlet:")
print(f"    Cp    = {Cp_des:.1f} J/kg/K")
print(f"    gamma = {gamma_des:.4f}")
print(f"    R     = {R_des:.3f} J/kg/K")

Tt0  = Q_(Tt4_K,  'K')
Pt0  = Q_(Pt4_Pa, 'Pa')
mdot = Q_(mdot_kg, 'kg/s')
RPM  = Q_(Nmech,  'rpm')
W_t  = Q_(pwr_W,  'W')

# Corrected flow reference — pyCycle Np convention: N/sqrt(Tt_degR), Tref=1 degR
Tt4_R       = Tt4_K * 9.0 / 5.0            # 2857.0 °R
Np_design   = round(Nmech / np.sqrt(Tt4_R), 4)  # 275.12 rpm
print(f"\n  NpMap design = {Np_design:.4f} rpm  (N/√Tt_degR)")

# =============================================================================
#  STEP 2 — DESIGN POINT KINEMATICS
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 2: Design point kinematics")
print("─" * 62)

# ── Tunable parameters — adjust to hit PR≈3.505, η≈0.903 ──────────────────
PSI_VGT   = 1.493993927   # loading coeff ψ (VGT)
PHI_VGT   = 0.60   # flow coeff    φ (VGT)
PSI_FIXED = 1.493993927   # loading coeff ψ (fixed stator)
PHI_FIXED = 0.60   # flow coeff    φ (fixed stator)
DES_INCIDENCE_R = 0.0# rotor design incidence [deg]

# ── Common blade geometry — shared by both VGT and fixed-stator designs ────
BLADE_CFG = dict(
    composition     = composition,
    AR_stator       = 1.307692308,
    AR_rotor        = 1.727272727,
    Z_stator        = 1.10,
    Z_rotor         = 1.10,
    LE_radius       = Q_(0.005,  'in'),
    TE_radius       = Q_(0.0025, 'in'),
    inlet_wedge     = Q_(15.0,  'deg'),
    exit_wedge_s    = Q_(3.0,   'deg'),
    exit_wedge_r    = Q_(3.0,   'deg'),
    zeta_ung_s      = Q_(6.5,   'deg'),
    zeta_ung_r      = Q_(6.5,   'deg'),
    t_s_rotor       = 0.0265,
    des_incidence_r = DES_INCIDENCE_R,
    tol             = 1e-8,
    max_iter        = 500,
    verbose         = True,
)

gas.TPX = Tt4_K, Pt4_Pa, composition

kin = design_point_kinematics(
    Tt0  = Tt0,
    Pt0  = Pt0,
    gas  = gas,
    mdot = mdot,
    RPM  = RPM,
    psi  = PSI_VGT,
    phi  = PHI_VGT,
    W_t  = W_t,
)

print(f"  Specific work  dH    = {kin['delta_H']:.1f} J/kg")
print(f"  Blade speed    U     = {kin['U']:.2f} m/s")
print(f"  Mean radius    Rm    = {kin['Rm']*1000:.2f} mm")
print(f"  Axial velocity Cm    = {kin['Cm']:.2f} m/s")
print(f"  Stator exit    α2    = {kin['alpha2']:.2f}°")
print(f"  Rotor inlet    β2    = {kin['beta2']:.2f}°")
print(f"  Reaction             = {kin['R_rxn']*100:.1f}%")

# =============================================================================
#  STEP 3 — FORWARD DESIGN  (VGT stator)
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 3: VGT forward design")
print("─" * 62)

gas.TPX = Tt4_K, Pt4_Pa, composition

design = forward_design(
    kin, gas, mdot, RPM,
    **BLADE_CFG,
    t_s_stator = 0.0#0.015/2,  # enable when stator tip-seal leakage is ready
)

if not design['converged']:
    print("  ⚠  forward_design did not converge — proceeding with last iterate")

stator = design['stator']
rotor  = design['rotor']
perf   = design['perf']


span = rotor["R_o_outlet"] - rotor["R_i_outlet"]
print(f"  t/c = {(rotor['Tip_Gap']/span):.3f}")
print(f"\n  VGT design efficiency  η_tt = {perf['eta_tt']*100:.3f}%"
      f"  (target: 90.3%)")
print(f"  VGT design PR (t-t)         = {perf['Pt_ratio']:.4f}"
      f"  (target: 3.505)")

# ── Verify against off-design's rigorous engine at the SAME static back-
#    pressure forward-design assumed (same downstream expansion potential),
#    and display alongside forward-design's own numbers.
inlet = {"Pt": Pt0, "Tt": Tt0}
gas.TPX = Tt4_K, Pt4_Pa, composition
verified = evaluate_at_same_p2(design, gas, composition, RPM, inlet)
print_same_p2_point(design, verified, mdot_kg)
compare_fd_od(design, kin, verified)

# =============================================================================
#  STEP 4 — FORWARD DESIGN  (fixed stator)
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 4: Fixed stator forward design")
print("─" * 62)

gas.TPX = Tt4_K, Pt4_Pa, composition

kin_fixed = design_point_kinematics(
    Tt0  = Tt0,
    Pt0  = Pt0,
    gas  = gas,
    mdot = mdot,
    RPM  = RPM,
    psi  = PSI_FIXED,
    phi  = PHI_FIXED,
    W_t  = W_t,
)

gas.TPX = Tt4_K, Pt4_Pa, composition

design_fixed = forward_design(
    kin_fixed, gas, mdot, RPM,
    **BLADE_CFG,
    # t_s_stator = 0.005,  # enable when stator tip-seal leakage is ready
)

if not design_fixed['converged']:
    print("  ⚠  forward_design (fixed) did not converge — proceeding with last iterate")

stator_fixed = design_fixed['stator']
rotor_fixed  = design_fixed['rotor']
perf_fixed   = design_fixed['perf']

print(f"\n  Fixed design efficiency  η_tt = {perf_fixed['eta_tt']*100:.3f}%")
print(f"  Fixed design PR (t-t)         = {perf_fixed['Pt_ratio']:.4f}")

gas.TPX = Tt4_K, Pt4_Pa, composition
verified_fixed = evaluate_at_same_p2(design_fixed, gas, composition, RPM, inlet)
print_same_p2_point(design_fixed, verified_fixed, mdot_kg)
compare_fd_od(design_fixed, kin_fixed, verified_fixed)

# =============================================================================
#  STEP 5 — OFF-DESIGN PERFORMANCE MAP
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 5: Off-design performance sweep")
print("─" * 62)

# Speed lines: 70% to 110% of design speed
speed_fracs = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05]

# Area fractions: NGV throat area scaling (VGT map dimension)
# 1.0 = design geometry; <1.0 = closed (higher PR); >1.0 = open (lower PR)
area_fracs = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05]

# PR sweep
PR_lo = 1.3
PR_hi = perf['Pt_ratio'] * 2.0
N_pts = 25

Ps3_des = design['states']['Ps3']
Pt3_des = design['perf']['Pt3_abs']
ps_pt   = Ps3_des / Pt3_des

p2_hi  = Pt4_Pa / PR_lo * ps_pt
p2_lo  = Pt4_Pa / PR_hi * ps_pt
p2_vals = np.linspace(p2_hi, p2_lo, N_pts)

# Insert the exact design-PR p2 so off-design evaluates at precisely the design point
p2_des_exact = Pt4_Pa / perf['Pt_ratio'] * ps_pt
p2_vals = np.sort(np.unique(np.append(p2_vals, p2_des_exact)))[::-1]

print(f"  p2 sweep: {p2_hi:.0f} Pa -> {p2_lo:.0f} Pa  (ps/pt={ps_pt:.4f})")
print(f"  PR sweep: {PR_lo:.2f} → {PR_hi:.2f}  ({len(p2_vals)} pts, incl. design PR)")
print(f"  Area fracs: {area_fracs}")

# Corrected flow in pyCycle's turbine Wp convention: W[lbm/s] * sqrt(Tt[degR]) / Pt[psia]
# This matches what pyCycle prints as "Wp" in the turbine table, giving s_Wp ≈ 1.0 at design.
Tt4_R_cr = Tt4_K * 9.0 / 5.0      # °R  (constant: map built at fixed inlet conditions)
Pt4_psia = Pt4_Pa / 6894.757       # psia

def corrected_mdot(mdot_kgs):
    return mdot_kgs * KG_TO_LBM * np.sqrt(Tt4_R_cr) / Pt4_psia

all_curves       = {}
all_curves_fixed = {}

for area_frac in area_fracs:
    stator_var = dict(stator)
    stator_var['throat'] = stator['throat'] * area_frac

    for frac in speed_fracs:
        N_actual = Nmech * frac
        RPM_frac = Q_(N_actual, 'rpm')
        print(f"\n{'='*55}")
        print(f"  {frac*100:.0f}% speed  area={area_frac:.2f}  ({N_actual:.0f} rpm)")
        print(f"{'='*55}")
        gas.TPX = Tt4_K, Pt4_Pa, composition
        all_curves[(frac, area_frac)] = build_performance_curve_moffitt(
            stator_var, rotor, inlet, gas, composition,
            RPM_frac, p2_vals, len(p2_vals)
        )

        if area_frac == 1.0:
            print(f"\n{'='*55}")
            print(f"  {frac*100:.0f}% speed,  Fixed Stator,  ({N_actual:.0f} rpm)")
            print(f"{'='*55}")
            gas.TPX = Tt4_K, Pt4_Pa, composition
            all_curves_fixed[(frac, area_frac)] = build_performance_curve_moffitt(
                stator_fixed, rotor_fixed, inlet, gas, composition,
                RPM_frac, p2_vals, len(p2_vals)
            )

# ── Diagnostic: forward-design vs off-design at design PR ────────────────────
des_PR    = perf['Pt_ratio']
des_curve = all_curves.get((1.00, 1.0), [])
closest   = min(des_curve, key=lambda pt: abs(pt['PR'] - des_PR))

des_states = design['states']
des_losses = design['losses']

# Forward-design quantities
fd_alpha_s  = kin['alpha2']
fd_M_s      = des_states['M_C2']
fd_Ps_s     = des_states['Ps2'] / 1000.0
fd_r_loss_s = perf['r_loss_s']
fd_zp_s     = des_losses['stator']['zeta_s']
fd_ztot_s   = des_losses['stator']['zeta_tot']
fd_beta_r   = kin['beta2']
fd_M_r      = des_states['M_W3']
fd_r_loss_r = perf['r_loss_r']
fd_zp_r     = des_losses['rotor']['zeta_s']
fd_ztot_r   = des_losses['rotor']['zeta_tot']

# Off-design quantities at nearest PR (should now be exactly design PR)
od = closest
od_alpha_s  = float(od['stator_out']['alpha_out'].to('deg').magnitude)
od_M_s      = od['M_stator_exit']
od_Ps_s     = float(od['stator_out']['Ps'].to('kPa').magnitude)
od_r_loss_s = od['loss_s']['r_loss_new']
od_zp_s     = od['loss_s']['zeta_s']
od_ztot_s   = od['loss_s']['zeta_tot']
od_beta_r   = float(np.degrees(np.arctan2(
    float(od['rotor_in']['W_theta'].to('m/s').magnitude),
    float(od['rotor_in']['Cm'].to('m/s').magnitude))))
od_incidence = od['i_rotor']
od_M_r      = od['M_rotor_exit']
od_r_loss_r = od['loss_r']['r_loss_new']
od_zp_r     = od['loss_r']['zeta_s']
od_ztot_r   = od['loss_r']['zeta_tot']

print(f"\n{'='*66}")
print(f"  DIAGNOSTIC: Forward-design vs Off-design at design PR")
print(f"  Forward PR={des_PR:.4f}   Off-design nearest PR={closest['PR']:.4f}")
print(f"{'='*66}")
print(f"  {'Quantity':<30} {'Fwd-Design':>11} {'Off-Design':>11} {'Δ':>10}")
print(f"  {'-'*64}")
print(f"  {'Stator α_exit [°]':<30} {fd_alpha_s:>11.4f} {od_alpha_s:>11.4f} {od_alpha_s-fd_alpha_s:>+10.4f}")
print(f"  {'Stator M_exit (abs)':<30} {fd_M_s:>11.4f} {od_M_s:>11.4f} {od_M_s-fd_M_s:>+10.4f}")
print(f"  {'Stator Ps [kPa]':<30} {fd_Ps_s:>11.3f} {od_Ps_s:>11.3f} {od_Ps_s-fd_Ps_s:>+10.3f}")
print(f"  {'Stator r_loss':<30} {fd_r_loss_s:>11.6f} {od_r_loss_s:>11.6f} {od_r_loss_s-fd_r_loss_s:>+10.6f}")
print(f"  {'Stator ζ_profile':<30} {fd_zp_s:>11.5f} {od_zp_s:>11.5f} {od_zp_s-fd_zp_s:>+10.5f}")
print(f"  {'Stator ζ_total':<30} {fd_ztot_s:>11.5f} {od_ztot_s:>11.5f} {od_ztot_s-fd_ztot_s:>+10.5f}")
print(f"  {'-'*64}")
print(f"  {'Rotor β_in_rel [°]':<30} {fd_beta_r:>11.4f} {od_beta_r:>11.4f} {od_beta_r-fd_beta_r:>+10.4f}")
print(f"  {'Rotor incidence [°] (fwd=0)':<30} {'0.0000':>11} {od_incidence:>11.4f} {od_incidence:>+10.4f}")
print(f"  {'Rotor M_exit_rel':<30} {fd_M_r:>11.4f} {od_M_r:>11.4f} {od_M_r-fd_M_r:>+10.4f}")
print(f"  {'Rotor r_loss':<30} {fd_r_loss_r:>11.6f} {od_r_loss_r:>11.6f} {od_r_loss_r-fd_r_loss_r:>+10.6f}")
print(f"  {'Rotor ζ_profile':<30} {fd_zp_r:>11.5f} {od_zp_r:>11.5f} {od_zp_r-fd_zp_r:>+10.5f}")
print(f"  {'Rotor ζ_total':<30} {fd_ztot_r:>11.5f} {od_ztot_r:>11.5f} {od_ztot_r-fd_ztot_r:>+10.5f}")
print(f"  {'-'*64}")
print(f"  {'η_tt [%]':<30} {perf['eta_tt']*100:>11.3f} {closest['eta']*100:>11.3f} {(closest['eta']-perf['eta_tt'])*100:>+10.3f}")
print(f"{'='*66}")

# =============================================================================
#  STEP 6 — WRITE pyCycle MAP
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 6: Writing pyCycle maps")
print("─" * 62)


def write_pycycle_map(all_curves, speed_fracs, area_fracs, Nmech, Tt4_K, Pt4_Pa,
                      corrected_mdot, des_PR_design, map_obj_name, filename):
    """Write a pyCycle-compatible MapData file."""

    # Common PR grid
    all_PRs = []
    for curve in all_curves.values():
        all_PRs.extend([pt['PR'] for pt in curve])
    PR_min  = max(1.05, min(all_PRs))
    PR_max  = min(9.0,  max(all_PRs))
    N_pr    = 20
    PR_grid = np.linspace(PR_min, PR_max, N_pr)

    # NpMap — normalized to 100 at design speed (pyCycle convention; OD balance upper=200)
    # Physical Np = N/sqrt(Tt_degR); here we store speed_frac*100 so design=100, OD stays in [60,120]
    Tt4_R     = Tt4_K * 9.0 / 5.0
    Np_vals   = np.array([round(100.0 * frac, 4) for frac in speed_fracs])
    Np_design = 100.0

    def build_map_slice(area_frac):
        eff_rows, wp_rows = [], []
        for frac in speed_fracs:
            curve = all_curves.get((frac, area_frac), [])
            if not curve:
                eff_rows.append(np.full(N_pr, np.nan))
                wp_rows.append(np.full(N_pr, np.nan))
                continue
            PRs  = np.array([pt['PR']  for pt in curve])
            effs = np.array([pt['eta'] for pt in curve])
            Wcs  = np.array([corrected_mdot(pt['mdot']) for pt in curve])
            idx  = np.argsort(PRs)
            PRs  = PRs[idx]; effs = effs[idx]; Wcs = Wcs[idx]
            PR_lo_i   = max(PR_grid[0],  PRs[0])
            PR_hi_i   = min(PR_grid[-1], PRs[-1])
            PR_interp = np.clip(PR_grid, PR_lo_i, PR_hi_i)
            f_eff = interp1d(PRs, effs, kind='linear', fill_value='extrapolate')
            f_Wc  = interp1d(PRs, Wcs,  kind='linear', fill_value='extrapolate')
            eff_rows.append(np.clip(f_eff(PR_interp), 0.3, 1.0))
            wp_rows.append(np.clip(f_Wc(PR_interp),  0.1, 500.0))
        return np.array(eff_rows), np.array(wp_rows)

    eff_slices, wp_slices = [], []
    for area_frac in area_fracs:
        eff_arr, wp_arr = build_map_slice(area_frac)
        eff_slices.append(eff_arr)
        wp_slices.append(wp_arr)

    effMap_3d = np.stack(eff_slices, axis=0)   # (n_alpha, n_speed, n_PR)
    WpMap_3d  = np.stack(wp_slices,  axis=0)

    # Design default corrected flow
    des_curve = all_curves.get((1.00, 1.0), [])
    if des_curve:
        prs     = np.array([pt['PR'] for pt in des_curve])
        idx_des = int(np.argmin(np.abs(prs - des_PR_design)))
        des_Wc  = corrected_mdot(des_curve[idx_des]['mdot'])
    else:
        des_Wc = 15.0

    alpha_design = 1.0

    def fmt_row(arr):
        return '[' + ', '.join(f'{v:.4f}' for v in arr) + ']'

    lines = [
        'import numpy as np',
        'from pycycle.maps.map_data import MapData',
        '',
        '',
        f'{map_obj_name} = MapData()',
        '',
        f'# CFM56-7B27 HPT map — generated by CFM56_HPT_map_generator.py',
        f'# Design: Tt4={Tt4_K:.1f} K  Pt4={Pt4_Pa:.0f} Pa  N_HP={Nmech:.1f} rpm',
        f'# NpMap: normalized speed (100 = design), s_Np = {round(Nmech/np.sqrt(Tt4_R),4):.4f} scales to physical N/sqrt(Tt_degR)',
        f'# Target: PR≈3.505  η≈0.903  Wp_des≈15.94 lbm/s',
        '',
        f'{map_obj_name}.defaults = {{}}',
        f"{map_obj_name}.defaults['alphaMap'] = {alpha_design}",
        f"{map_obj_name}.defaults['NpMap']    = {Np_design:.4f}",
        f"{map_obj_name}.defaults['PRmap']    = {des_PR_design:.3f}",
        '',
        f'{map_obj_name}.alphaMap = np.array({[round(float(a), 4) for a in area_fracs]})  # NGV throat area fractions',
        f'{map_obj_name}.NpMap    = np.array({list(np.round(Np_vals, 4))})',
        f'{map_obj_name}.PRmap    = np.array({list(np.round(PR_grid, 4))})',
        '',
    ]

    lines.append(f'{map_obj_name}.effMap = np.array([')
    for a_idx, area_frac in enumerate(area_fracs):
        lines.append(f'    [  # alpha = area_frac = {area_frac:.2f}')
        for i, row in enumerate(effMap_3d[a_idx]):
            comma = ',' if i < len(effMap_3d[a_idx]) - 1 else ''
            lines.append(f'        {fmt_row(row)}{comma}')
        comma = ',' if a_idx < len(area_fracs) - 1 else ''
        lines.append(f'    ]{comma}')
    lines.append('])')
    lines.append('')

    lines.append(f'{map_obj_name}.WpMap = np.array([')
    for a_idx, area_frac in enumerate(area_fracs):
        lines.append(f'    [  # alpha = area_frac = {area_frac:.2f}')
        for i, row in enumerate(WpMap_3d[a_idx]):
            comma = ',' if i < len(WpMap_3d[a_idx]) - 1 else ''
            lines.append(f'        {fmt_row(row)}{comma}')
        comma = ',' if a_idx < len(area_fracs) - 1 else ''
        lines.append(f'    ]{comma}')
    lines.append('])')
    lines.append('')

    lines += [
        f'{map_obj_name}.Npts = {map_obj_name}.NpMap.size',
        '',
        f"{map_obj_name}.units = {{}}",
        f"{map_obj_name}.units['NpMap'] = 'rpm'",
        f"{map_obj_name}.units['WpMap'] = 'lbm/s'",
        '',
        f'{map_obj_name}.param_data = []',
        f'{map_obj_name}.output_data = []',
        '',
        f"{map_obj_name}.param_data.append({{'name': 'alphaMap', 'values': {map_obj_name}.alphaMap,",
        f"                                'default': {alpha_design}, 'units': None}})",
        f"{map_obj_name}.param_data.append({{'name': 'NpMap', 'values': {map_obj_name}.NpMap,",
        f"                                'default': {Np_design:.4f}, 'units': 'rpm'}})",
        f"{map_obj_name}.param_data.append({{'name': 'PRmap', 'values': {map_obj_name}.PRmap,",
        f"                                'default': {des_PR_design:.3f}, 'units': None}})",
        '',
        f"{map_obj_name}.output_data.append({{'name': 'WpMap', 'values': {map_obj_name}.WpMap,",
        f"                                 'default': np.mean({map_obj_name}.WpMap), 'units': 'lbm/s'}})",
        f"{map_obj_name}.output_data.append({{'name': 'effMap', 'values': {map_obj_name}.effMap,",
        f"                                 'default': np.mean({map_obj_name}.effMap), 'units': None}})",
    ]

    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  Written: {filename}")
    print(f"  Shape: effMap={effMap_3d.shape}, WpMap={WpMap_3d.shape}")
    print(f"  alphaMap: {area_fracs}")
    print(f"  NpMap: {np.round(Np_vals, 3)}  (design={Np_design:.4f})")
    print(f"  PRmap: [{PR_grid[0]:.3f} → {PR_grid[-1]:.3f}]  ({N_pr} pts)")
    print(f"  Wp_des: {des_Wc:.3f} lbm/s  (pyCycle target: 15.94)")
    return filename


# VGT map (8 alpha slices)
write_pycycle_map(
    all_curves, speed_fracs, area_fracs,
    Nmech, Tt4_K, Pt4_Pa, corrected_mdot,
    des_PR_design = perf['Pt_ratio'],
    map_obj_name  = 'CFM56HPTMapVGT',
    filename      = 'CFM56_HPT_map.py',
)

# Fixed map (3 identical alpha slices for pyCycle slinear interpolation)
area_fracs_fixed = [0.9, 1.0, 1.1]
design_area_key  = min(area_fracs, key=lambda a: abs(a - 1.0))
for frac in speed_fracs:
    src = all_curves_fixed.get((frac, design_area_key), [])
    all_curves_fixed[(frac, 0.9)] = src
    all_curves_fixed[(frac, 1.0)] = src
    all_curves_fixed[(frac, 1.1)] = src

write_pycycle_map(
    all_curves_fixed, speed_fracs, area_fracs_fixed,
    Nmech, Tt4_K, Pt4_Pa, corrected_mdot,
    des_PR_design = perf_fixed['Pt_ratio'],
    map_obj_name  = 'CFM56HPTMapFixed',
    filename      = 'CFM56_HPT_map_fixed.py',
)

# =============================================================================
#  STEP 7 — VALIDATION PLOTS
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 7: Plotting")
print("─" * 62)

import matplotlib.lines as mlines

unique_speeds = sorted(speed_fracs)
unique_areas  = sorted(area_fracs)
cmap_area     = plt.cm.turbo
area_colors   = {a: cmap_area(i / max(len(unique_areas) - 1, 1))
                 for i, a in enumerate(unique_areas)}
speed_markers = {frac: m for frac, m in zip(
    unique_speeds,
    ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*'][:len(unique_speeds)]
)}

# Plot 1: 100% speed validation (VGT vs Fixed)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('CFM56 HPT — 100% Speed, Design Area Validation', fontsize=11, fontweight='bold')

des_curve_vgt   = all_curves.get((1.00, 1.0), [])
des_curve_fixed = all_curves_fixed.get((1.00, 1.0), [])

for curve, label, style in [
    (des_curve_vgt,   'VGT',   'b-o'),
    (des_curve_fixed, 'Fixed', 'r-s'),
]:
    if curve:
        PRs  = [pt['PR']        for pt in curve]
        Wcs  = [corrected_mdot(pt['mdot']) for pt in curve]
        etas = [pt['eta'] * 100 for pt in curve]
        ax1.plot(PRs, Wcs,  style, lw=2, ms=4, label=label)
        ax2.plot(PRs, etas, style, lw=2, ms=4, label=label)

ax1.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
            label=f"Design PR={perf['Pt_ratio']:.3f}")
ax1.set_xlabel('PR'); ax1.set_ylabel('$W_c$ (lbm/s)')
ax1.set_title('Corrected Mass Flow'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
            label=f"Design PR={perf['Pt_ratio']:.3f}")
ax2.axhline(perf['eta_tt'] * 100, color='gray', ls='--', lw=1,
            label=f"Design η={perf['eta_tt']*100:.1f}% (target 90.3%)")
ax2.axhline(90.3, color='green', ls=':', lw=1.5, label='Target η=90.3%')
ax2.set_xlabel('PR'); ax2.set_ylabel('η (%)'); ax2.set_ylim(60, 100)
ax2.set_title('Efficiency'); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('CFM56_HPT_validation.png', dpi=150)
plt.show()

# Plot 2: Efficiency at 100% speed — all area fracs (VGT)
fig2, ax_eta = plt.subplots(figsize=(10, 6))
fig2.suptitle('CFM56 HPT VGT — Efficiency at 100% Speed', fontsize=11, fontweight='bold')

for area_frac in unique_areas:
    curve = all_curves.get((1.00, area_frac), [])
    if not curve: continue
    PRs  = [pt['PR']        for pt in curve]
    etas = [pt['eta'] * 100 for pt in curve]
    ax_eta.plot(PRs, etas, color=area_colors[area_frac], lw=2, marker='o',
                ms=4, label=f'Area: {area_frac:.2f}')

ax_eta.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
               label=f"Design PR={perf['Pt_ratio']:.3f}")
ax_eta.axhline(perf['eta_tt'] * 100, color='gray', ls='--', lw=1)
ax_eta.axhline(90.3, color='green', ls=':', lw=1.5, label='Target 90.3%')
ax_eta.set_xlabel('PR'); ax_eta.set_ylabel('η (%)'); ax_eta.set_ylim(60, 100)
ax_eta.set_title('Efficiency — All NGV Area Fractions at 100% Speed')
ax_eta.legend(fontsize=9); ax_eta.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('CFM56_HPT_eta_area.png', dpi=150)
plt.show()

# Plot 3: Corrected flow map (all speeds, all areas)
fig3, ax_flow = plt.subplots(figsize=(12, 7))
fig3.suptitle('CFM56 HPT VGT — Corrected Mass Flow Map', fontsize=11, fontweight='bold')

for (frac, area_frac), curve in all_curves.items():
    if not curve: continue
    PRs = [pt['PR'] for pt in curve]
    Wcs = [corrected_mdot(pt['mdot']) for pt in curve]
    ax_flow.plot(PRs, Wcs, color=area_colors[area_frac],
                 marker=speed_markers[frac], linestyle='-', lw=1.5, ms=4)

legend_handles = [
    mlines.Line2D([], [], color='none', label=r'$\mathbf{Area\ Fractions\ (Colors)}$')]
for a in unique_areas:
    legend_handles.append(mlines.Line2D([], [], color=area_colors[a], lw=2,
                                         label=f'Area: {a:.2f}'))
legend_handles.append(
    mlines.Line2D([], [], color='none', label=r'$\mathbf{Speed\ Fractions\ (Markers)}$'))
for frac in unique_speeds:
    legend_handles.append(mlines.Line2D([], [], color='gray',
                                         marker=speed_markers[frac],
                                         linestyle='none', ms=6,
                                         label=f'Speed: {frac*100:.0f}%'))

ax_flow.set_xlabel('PR'); ax_flow.set_ylabel('$W_c$ (lbm/s)')
ax_flow.set_title('Corrected Flow — All Speeds and NGV Areas')
ax_flow.legend(handles=legend_handles, loc='best', fontsize=8)
ax_flow.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('CFM56_HPT_flow_map.png', dpi=150)
plt.show()

print("\n  Done. Files written:")
print(f"    CFM56_HPT_map.py       — VGT pyCycle MapData (CFM56HPTMapVGT)")
print(f"    CFM56_HPT_map_fixed.py — Fixed pyCycle MapData (CFM56HPTMapFixed)")
print(f"    CFM56_HPT_validation.png")
print(f"    CFM56_HPT_eta_area.png")
print(f"    CFM56_HPT_flow_map.png")
print(f"\n  To use in cfm56_cycle.py:")
print(f"    from CFM56_HPT_map import CFM56HPTMapVGT")
print(f"    ...pyc.Turbine(map_data=CFM56HPTMapVGT, ...)")
print(f"\n  If PR or η don't match targets, tune in the file:")
print(f"    PSI_VGT   = {PSI_VGT}  → adjust until PR≈3.505")
print(f"    PHI_VGT   = {PHI_VGT}  → adjust until η≈0.903")
