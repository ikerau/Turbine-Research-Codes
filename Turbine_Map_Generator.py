"""
generate_turbine_map.py
-----------------------
End-to-end pipeline:
  1. Design a turbine matched to the engine cycle using forward_design()
  2. Run off-design performance sweep using build_performance_curve_moffitt()
  3. Write a pyCycle-compatible MapData file


Usage:
  python generate_turbine_map.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import cantera as ct
from scipy.interpolate import interp1d

# ── Import units ──────────────────────────────────────────────────────────────
from units import Q_, ureg

# ── Import forward design tools ───────────────────────────────────────────────
from forward_design import (
    design_point_kinematics,
    forward_design,
)

# ── Import off-design tools ───────────────────────────────────────────────────
from off_design import build_performance_curve_moffitt

# ── Constants ─────────────────────────────────────────────────────────────────
KG_TO_LBM  = 2.20462
HP_TO_W    = 745.699872

# =============================================================================
#  STEP 1 — ENGINE CYCLE CONDITIONS
# =============================================================================

def turbine_power(mdot_kg, Tt4_K, OPR, comp_eff, gamma_c=1.4, gamma_t=1.33,
                  R=287.05, Tt_inlet_K=288.15):
    """
    Estimate HPT shaft power from cycle parameters.
    Assumes ideal Brayton cycle with polytropic efficiencies.
    """
    Cp_c = gamma_c * R / (gamma_c - 1.0)
    Cp_t = gamma_t * R / (gamma_t - 1.0)

    # Compressor work per unit mass
    T_comp_exit = Tt_inlet_K * (1.0 + (OPR**((gamma_c-1.0)/gamma_c) - 1.0) / comp_eff)
    W_comp      = Cp_c * (T_comp_exit - Tt_inlet_K)

    # HPT must provide compressor work (shaft balance)
    pwr_W = mdot_kg * W_comp
    return pwr_W


# From pyCycle DESIGN solve (OPR=13.5, T4=2370°R)
# From pyCycle DESIGN solve (OPR=13.5, T4=2370°R)
Tt4_K   = 1577.65          # K  = 1316.7 K
Pt4_Pa  = 2325.15*1000
mdot_kg = 53.935
pwr_W   = turbine_power(mdot_kg = 52.84, Tt4_K = Tt4_K, OPR = 10.672, comp_eff = 0.867, gamma_c=1.4, gamma_t=1.33,
                  R=287.05, Tt_inlet_K=288.15)
Nmech   = 14460                    # rpm

# =============================================================================
#  STEP 2 — WORKING FLUID SETUP
# =============================================================================

gas = ct.Solution('air.yaml')
gas.TP = Tt4_K, Pt4_Pa
composition = gas.X

print("=" * 62)
print("  TURBINE MAP GENERATION PIPELINE")
print("=" * 62)
print(f"\n  Engine cycle conditions:")
print(f"    Tt4  = {Tt4_K:.1f} K  ({2370:.0f}°R)")
print(f"    Pt4  = {Pt4_Pa/1e6:.3f} MPa  ({198.395:.1f} psia)")
print(f"    mdot = {mdot_kg:.2f} kg/s  ({146.108:.1f} lbm/s)")
print(f"    Pwr  = {pwr_W/1e6:.2f} MW  ({33375:.0f} hp)")
print(f"    N    = {Nmech:.0f} rpm")

gas.TP = Tt4_K, Pt4_Pa
Cp_des    = float(gas.cp_mass)
gamma_des = float(gas.cp_mass / gas.cv_mass)
R_des     = float(ct.gas_constant / gas.mean_molecular_weight)
print(f"\n  Gas properties at turbine inlet:")
print(f"    Cp    = {Cp_des:.1f} J/kg/K")
print(f"    gamma = {gamma_des:.4f}")
print(f"    R     = {R_des:.3f} J/kg/K")

# =============================================================================
#  STEP 3 — DESIGN POINT KINEMATICS  (psi=1.25, phi=0.65)
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 3: Design point kinematics  (ψ=1.00, φ=0.50)")
print("─" * 62)

Tt0  = Q_(Tt4_K,  'K')
Pt0  = Q_(Pt4_Pa, 'Pa')
mdot = Q_(mdot_kg, 'kg/s')
RPM  = Q_(Nmech,  'rpm')
W_t  = Q_(pwr_W,  'W')

gas.TPX = Tt4_K, Pt4_Pa, composition

kin = design_point_kinematics(
    Tt0  = Tt0,
    Pt0  = Pt0,
    gas  = gas,
    mdot = mdot,
    RPM  = RPM,
    psi  = 1.164,
    phi  = 0.536,
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
#  STEP 4 — FORWARD DESIGN  Variable Stator
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 4: Variable Forward design")
print("─" * 62)

gas.TPX = Tt4_K, Pt4_Pa, composition

design = forward_design(
    kin, gas, mdot, RPM,
    composition  = composition,
    AR_stator    = 1.2,
    AR_rotor     = 1.0,
    Z_stator     = 0.85,
    Z_rotor      = 0.80,
    LE_radius    = Q_(0.005,   'in'),
    TE_radius    = Q_(0.0025,   'in'),
    inlet_wedge  = Q_(15.0,  'deg'),
    exit_wedge_s = Q_(3.0,   'deg'),
    exit_wedge_r = Q_(3.0,   'deg'),
    zeta_ung_s   = Q_(6.5,   'deg'),
    zeta_ung_r   = Q_(6.5,   'deg'),
    t_s_rotor    = 0.005,
    t_s_stator=    0.005,
    tol          = 1e-4,
    max_iter     = 500,
    verbose      = True,
)


if not design['converged']:
    print("  ⚠  forward_design did not converge — proceeding with last iterate")

stator = design['stator']
rotor  = design['rotor']
span = rotor["R_o_outlet"] - rotor["R_i_outlet"]
print(f"t/c = {(rotor['Tip_Gap']/span):.3f}")
perf   = design['perf']

print(f"\n  Design efficiency  η_tt = {perf['eta_tt']*100:.3f}%")
print(f"  Design PR (t-t)         = {perf['Pt_ratio']:.4f}")
print(f"  r_loss_s                = {perf['r_loss_s']:.6f}")
print(f"  r_loss_r                = {perf['r_loss_r']:.6f}")

# =============================================================================
#  STEP 4.5 — FORWARD DESIGN  Fixed Stator
# =============================================================================

print("\n" + "─" * 62)
print("  STEP 4.5: Fixed Stator Forward design")
print("─" * 62)

gas.TPX = Tt4_K, Pt4_Pa, composition

design_fixed = forward_design(
    kin, gas, mdot, RPM,
    composition  = composition,
    AR_stator    = 1.2,
    AR_rotor     = 1.0,
    Z_stator     = 0.85,
    Z_rotor      = 0.80,
    LE_radius    = Q_(0.005,   'in'),
    TE_radius    = Q_(0.0025,   'in'),
    inlet_wedge  = Q_(15.0,  'deg'),
    exit_wedge_s = Q_(3.0,   'deg'),
    exit_wedge_r = Q_(3.0,   'deg'),
    zeta_ung_s   = Q_(6.5,   'deg'),
    zeta_ung_r   = Q_(6.5,   'deg'),
    t_s_rotor    = 0.005,
    tol          = 1e-4,
    max_iter     = 500,
    verbose      = True,
)

if not design['converged']:
    print("  ⚠  forward_design did not converge — proceeding with last iterate")

stator_fixed = design_fixed['stator']
rotor_fixed  = design_fixed['rotor']
print(f"rotor Tip_Gap = {rotor_fixed['Tip_Gap'].to('in')}")
perf_fixed   = design_fixed['perf']

print(f"\n  Design efficiency  η_tt = {perf_fixed['eta_tt']*100:.3f}%")
print(f"  Design PR (t-t)         = {perf_fixed['Pt_ratio']:.4f}")
print(f"  r_loss_s                = {perf_fixed['r_loss_s']:.6f}")
print(f"  r_loss_r                = {perf_fixed['r_loss_r']:.6f}")



# =============================================================================
#  STEP 5 — OFF-DESIGN PERFORMANCE MAP
# =============================================================================
 
print("\n" + "─" * 62)
print("  STEP 5: Off-design performance sweep")
print("─" * 62)
 
inlet = {"Pt": Pt0, "Tt": Tt0}
 
# Speed lines: 70% to 110%
speed_fracs = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
 
# Area fractions — alphaMap dimension (NGV throat area scaling)
# 1.0 = design geometry; < 1.0 = closed; > 1.0 = open
area_fracs = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
 
# PR sweep using rotor exit static pressure
PR_lo    = 1.2
PR_hi    = perf['Pt_ratio'] * 2.0
N_pts    = 25
 
Ps3_des  = design['states']['Ps3']
Pt3_des  = design['perf']['Pt3_abs']
ps_pt    = Ps3_des / Pt3_des
 
p2_hi    = Pt4_Pa / PR_lo * ps_pt
p2_lo    = Pt4_Pa / PR_hi * ps_pt
p2_vals  = np.linspace(p2_hi, p2_lo, N_pts)
 
print(f"  p2 sweep: {p2_hi:.0f} Pa -> {p2_lo:.0f} Pa  (ps/pt={ps_pt:.4f})")
print(f"  Stator exit static p1_design = {design['states']['Ps2']:.0f} Pa")
print(f"  Area fracs: {area_fracs}")
 
# Corrected mdot helper — standard day reference
theta_cr   = Tt4_K / 288.15
delta_test = Pt4_Pa / 101325.0
 
def corrected_mdot(mdot_kgs):
    return mdot_kgs * np.sqrt(theta_cr) / delta_test
 
# all_curves keyed by (speed_frac, area_frac)
all_curves = {}
all_curves_fixed = {}
 
for area_frac in area_fracs:
    # Scale stator throat by area_frac — create modified stator geometry
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
            RPM_frac, p2_vals, N_pts
        )
        
        if area_frac == 1.0:
            print(f"\n{'='*55}")
            print(f"  {frac*100:.0f}% speed,  Fixed Stator,  ({N_actual:.0f} rpm)")
            print(f"{'='*55}")
            gas.TPX = Tt4_K, Pt4_Pa, composition
            all_curves_fixed[(frac, area_frac)] = build_performance_curve_moffitt(
                stator_fixed, rotor_fixed, inlet, gas, composition,
                RPM_frac, p2_vals, N_pts
            )

# Find the off-design point closest to design PR
des_PR = perf['Pt_ratio']
des_curve = all_curves.get((1.00, 1.0), [])
closest = min(des_curve, key=lambda pt: abs(pt['PR'] - des_PR))
print(f"\nOff-design at PR={closest['PR']:.4f} (target={des_PR:.4f}):")
print(f"  Stator: zeta_s={closest['loss_s']['zeta_s']:.4f}  "
      f"zeta_f={closest['loss_s']['zeta_f']:.4f}  "
      f"zeta_l={closest['loss_s']['zeta_l']:.4f}  "
      f"zeta_tot={closest['loss_s']['zeta_tot']:.4f}")
print(f"  Rotor:  zeta_s={closest['loss_r']['zeta_s']:.4f}  "
      f"zeta_f={closest['loss_r']['zeta_f']:.4f}  "
      f"zeta_l={closest['loss_r']['zeta_l']:.4f}  "
      f"zeta_tot={closest['loss_r']['zeta_tot']:.4f}")
print(f"  eta={closest['eta']*100:.3f}%")

# =============================================================================
#  STEP 6 — WRITE pyCycle MAP
# =============================================================================
 
print("\n" + "─" * 62)
print("  STEP 6: Writing pyCycle map")
print("─" * 62)
 
def write_pycycle_map(all_curves, speed_fracs, area_fracs, Nmech, Tt4_K, Pt4_Pa,
                      corrected_mdot, des_PR_design=3.683, filename='EngineHPT_map.py'):
    """Write a pyCycle-compatible MapData file.
 
    alphaMap = area_fracs — each slice is a different NGV throat area.
    NpMap uses pyCycle's Np convention: N / sqrt(Tt_degR) with Tref=1 degR.
    slinear requires >= 3 points per dimension — area_fracs must have >= 3.
    """
 
    # ── Common PR grid ────────────────────────────────────────────────────
    all_PRs = []
    for key, curve in all_curves.items():
        all_PRs.extend([pt['PR'] for pt in curve])
    PR_min  = max(1.05, min(all_PRs))
    PR_max  = min(8.0,  max(all_PRs))
    N_pr    = 20
    PR_grid = np.linspace(PR_min, PR_max, N_pr)
 
    # ── NpMap — pyCycle convention: N / sqrt(Tt_degR), Tref=1 degR ───────
    Tt4_R     = Tt4_K * 9.0 / 5.0
    Np_vals   = np.array([round(Nmech * frac / np.sqrt(Tt4_R), 4) for frac in speed_fracs])
    Np_design = round(Nmech / np.sqrt(Tt4_R), 4)
 
    # ── Build (n_alpha, n_speed, n_PR) arrays ─────────────────────────────
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
            Wcs  = np.array([corrected_mdot(pt['mdot']) * KG_TO_LBM for pt in curve])
 
            idx = np.argsort(PRs)
            PRs = PRs[idx]; effs = effs[idx]; Wcs = Wcs[idx]
 
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
 
    effMap_3d = np.stack(eff_slices, axis=0)  # (n_alpha, n_speed, n_PR)
    WpMap_3d  = np.stack(wp_slices,  axis=0)
 
    # ── Design defaults — design area = 1.0 ──────────────────────────────
    des_PR    = des_PR_design
    des_curve = all_curves.get((1.00, 1.0), [])
    if des_curve:
        prs     = np.array([pt['PR'] for pt in des_curve])
        idx_des = int(np.argmin(np.abs(prs - des_PR)))
        des_Wc  = corrected_mdot(des_curve[idx_des]['mdot']) * KG_TO_LBM
    else:
        des_Wc  = 30.0
 
    alpha_design = 1.0   # design area fraction
 
    def fmt_row(arr):
        return '[' + ', '.join(f'{v:.4f}' for v in arr) + ']'
 
    lines = [
        'import numpy as np',
        'from pycycle.maps.map_data import MapData',
        '',
        '',
        'EngineHPTMap = MapData()',
        '',
        f'# Engine HPT map — generated from forward_design() + meanline off-design solver',
        f'# Tt_ref={Tt4_K:.1f} K  Pt_ref={Pt4_Pa:.0f} Pa  N_design={Nmech:.0f} rpm',
        f'# NpMap in pyCycle units: N/sqrt(Tt_degR)  Tref=1 degR',
        '',
        'EngineHPTMap.defaults = {}',
        f"EngineHPTMap.defaults['alphaMap'] = {alpha_design}",
        f"EngineHPTMap.defaults['NpMap']    = {Np_design:.4f}",
        f"EngineHPTMap.defaults['PRmap']    = {des_PR:.3f}",
        '',
        f'EngineHPTMap.alphaMap = np.array({[round(float(a), 4) for a in area_fracs]})  # NGV throat area fractions',
        f'EngineHPTMap.NpMap    = np.array({list(np.round(Np_vals, 4))})',
        f'EngineHPTMap.PRmap    = np.array({list(np.round(PR_grid, 4))})',
        '',
    ]
 
    # effMap — shape (n_alpha, n_speed, n_PR)
    lines.append('EngineHPTMap.effMap = np.array([')
    for a_idx, area_frac in enumerate(area_fracs):
        lines.append(f'    [  # alpha = area_frac = {area_frac:.2f}')
        for i, row in enumerate(effMap_3d[a_idx]):
            comma = ',' if i < len(effMap_3d[a_idx]) - 1 else ''
            lines.append(f'        {fmt_row(row)}{comma}')
        comma = ',' if a_idx < len(area_fracs) - 1 else ''
        lines.append(f'    ]{comma}')
    lines.append('])')
    lines.append('')
 
    # WpMap — shape (n_alpha, n_speed, n_PR)
    lines.append('EngineHPTMap.WpMap = np.array([')
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
        'EngineHPTMap.Npts = EngineHPTMap.NpMap.size',
        '',
        "EngineHPTMap.units = {}",
        "EngineHPTMap.units['NpMap'] = 'rpm'",
        "EngineHPTMap.units['WpMap'] = 'lbm/s'",
        '',
        'EngineHPTMap.param_data = []',
        'EngineHPTMap.output_data = []',
        '',
        "EngineHPTMap.param_data.append({'name': 'alphaMap', 'values': EngineHPTMap.alphaMap,",
        f"                                'default': {alpha_design}, 'units': None}})",
        f"EngineHPTMap.param_data.append({{'name': 'NpMap', 'values': EngineHPTMap.NpMap,",
        f"                                'default': {Np_design:.4f}, 'units': 'rpm'}})",
        f"EngineHPTMap.param_data.append({{'name': 'PRmap', 'values': EngineHPTMap.PRmap,",
        f"                                'default': {des_PR:.3f}, 'units': None}})",
        '',
        "EngineHPTMap.output_data.append({'name': 'WpMap', 'values': EngineHPTMap.WpMap,",
        "                                 'default': np.mean(EngineHPTMap.WpMap), 'units': 'lbm/s'})",
        "EngineHPTMap.output_data.append({'name': 'effMap', 'values': EngineHPTMap.effMap,",
        "                                 'default': np.mean(EngineHPTMap.effMap), 'units': None})",
    ]
 
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
 
    print(f"  Written: {filename}")
    print(f"  Shape: effMap={effMap_3d.shape}, WpMap={WpMap_3d.shape}")
    print(f"  alphaMap (area fracs): {area_fracs}")
    print(f"  NpMap: {np.round(Np_vals, 3)}  (design={Np_design:.4f})")
    print(f"  PRmap: [{PR_grid[0]:.3f} → {PR_grid[-1]:.3f}]  ({N_pr} pts)")
    print(f"  WpMap range: [{WpMap_3d.min():.2f}, {WpMap_3d.max():.2f}] lbm/s")
    print(f"  effMap range: [{effMap_3d.min():.4f}, {effMap_3d.max():.4f}]")
    return filename
 
 
map_file = write_pycycle_map(
    all_curves, speed_fracs, area_fracs,
    Nmech, Tt4_K, Pt4_Pa,
    corrected_mdot,
    des_PR_design = perf['Pt_ratio'],
    filename='EngineHPT_map.py',
)

# Fixed map — duplicate design-area curves to 3 identical slices for slinear
area_fracs_fixed = [0.9, 1.0, 1.1]
design_area_key  = min(area_fracs, key=lambda a: abs(a - 1.0))  # nearest to 1.0
for frac in speed_fracs:
    src = all_curves_fixed.get((frac, design_area_key), [])
    all_curves_fixed[(frac, 0.9)] = src
    all_curves_fixed[(frac, 1.0)] = src
    all_curves_fixed[(frac, 1.1)] = src

map_file_fixed = write_pycycle_map(
    all_curves_fixed, speed_fracs, area_fracs_fixed,
    Nmech, Tt4_K, Pt4_Pa,
    corrected_mdot,
    des_PR_design = perf_fixed['Pt_ratio'],
    filename='EngineHPT_map_fixed.py',
)
 
# =============================================================================
#  STEP 7 — PLOT MAP
# =============================================================================
 
print("\n" + "─" * 62)
print("  STEP 7: Plotting map")
print("─" * 62)
 
import matplotlib.lines as mlines
 
unique_speeds = sorted(speed_fracs)
unique_areas  = sorted(area_fracs)
 
speed_markers = {frac: m for frac, m in zip(
    unique_speeds,
    ['o', 's', '^', 'v', 'D', '<', '>', 'p', '*'][:len(unique_speeds)]
)}
 
cmap_area   = plt.cm.turbo

 
# ── Plot 1: Validation at 100% speed, design area ────────────────────────────
fig_val, (ax_v1, ax_v2) = plt.subplots(1, 2, figsize=(14, 5))
fig_val.suptitle('Engine HPT — 100% Speed, Design Area Validation',
                 fontsize=11, fontweight='bold')
 
des_curve = all_curves.get((1.00, 1.0), [])
des_curve_fixed = all_curves_fixed.get((1.00, 1.0), [])
if des_curve:
    PR_c  = [pt['PR']       for pt in des_curve]
    Wc_c  = [corrected_mdot(pt['mdot']) * KG_TO_LBM for pt in des_curve]
    eta_c = [pt['eta'] * 100 for pt in des_curve]
    
    PR_cf  = [ptf['PR']       for ptf in des_curve_fixed]
    Wc_cf  = [corrected_mdot(ptf['mdot']) * KG_TO_LBM for ptf in des_curve_fixed]
    eta_cf = [ptf['eta'] * 100 for ptf in des_curve_fixed]
 
    ax_v1.plot(PR_c, Wc_c, 'b-o', lw=2, ms=4, label='Calc 100%, area=1.0')
    ax_v1.plot(PR_cf, Wc_cf, lw=2, ms=4, label='Calc - Fixed 100%, area=1.0')
    ax_v1.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
                  label=f"Design PR={perf['Pt_ratio']:.2f}")
    ax_v1.set_xlabel('PR'); ax_v1.set_ylabel('$W_c$ (lbm/s)')
    ax_v1.set_title('Corrected Mass Flow'); ax_v1.legend(fontsize=9)
    ax_v1.grid(True, alpha=0.3)
 
    ax_v2.plot(PR_c, eta_c, 'b-o', lw=2, ms=4, label='Calc 100%, area=1.0')
    ax_v2.plot(PR_cf, eta_cf, lw=2, ms=4, label='Calc - Fixed 100%, area=1.0')
    ax_v2.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
                  label=f"Design PR={perf['Pt_ratio']:.2f}")
    ax_v2.axhline(perf['eta_tt'] * 100, color='gray', ls='--', lw=1,
                  label=f"Design η={perf['eta_tt']*100:.1f}%")
    ax_v2.set_xlabel('PR'); ax_v2.set_ylabel('η (%)')
    ax_v2.set_title('Efficiency'); ax_v2.set_ylim(60, 100)
    ax_v2.legend(fontsize=9); ax_v2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('EngineHPT_validation.png', dpi=150)
plt.show()


# ── Cumulative stacked loss bands — Fixed vs VGT ─────────────────────────────
loss_stack_keys   = ['zeta_s', 'zeta_w', 'zeta_f', 'zeta_l']
loss_stack_labels = ['Profile', 'Wake', 'Endwall', 'Leakage']
loss_stack_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

des_curve_fixed = all_curves_fixed.get((1.00, 1.0), [])
des_curve_vgt   = all_curves.get((1.00, 1.0), [])

for blade_key, blade_title in [('loss_s', 'Stator'), ('loss_r', 'Rotor')]:

    fig, (ax_f, ax_v) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(
        f'{blade_title} Loss Breakdown — Fixed vs Variable Geometry Turbine\n'
        f'100% Speed  |  Design Area (1.0)',
        fontsize=11, fontweight='bold')

    for ax, curve, label in [
        (ax_f, des_curve_fixed, 'Fixed'),
        (ax_v, des_curve_vgt,   'VGT'),
    ]:
        if not curve:
            ax.set_title(f'{label} — no data')
            continue

        PR   = np.array([pt['PR']                    for pt in curve])
        vals = {k: np.array([pt[blade_key][k] for pt in curve])
                for k in loss_stack_keys}

        # Stack bands cumulatively
        bottom = np.zeros_like(PR)
        for k, lbl, color in zip(loss_stack_keys, loss_stack_labels,
                                  loss_stack_colors):
            ax.fill_between(PR, bottom, bottom + vals[k],
                            color=color, alpha=0.80, label=lbl)
            bottom += vals[k]

        # Total line on top
        total = np.array([pt[blade_key]['zeta_tot'] for pt in curve])
        ax.plot(PR, total, 'k-', lw=2, label='Total (ζ_tot)')

        # Design PR marker
        ax.axvline(perf['Pt_ratio'], color='gray', ls='--', lw=1.2,
                   label=f"Design PR={perf['Pt_ratio']:.2f}")

        ax.set_xlabel('PR (total-to-total)', fontsize=10)
        ax.set_ylabel('Loss coefficient  ζ  (—)', fontsize=10)
        ax.set_title(f'{label} Turbine', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fname = f'loss_stacked_{blade_title.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fname}')
    plt.show()

# ── Bar chart at design PR — Fixed vs VGT ────────────────────────────────────
def get_losses_at_PR(curve, des_PR, blade_key):
    """Interpolate each loss component to exactly des_PR."""
    PR_arr = np.array([pt['PR'] for pt in curve])
    result = {}
    for k in loss_stack_keys + ['zeta_tot']:
        vals = np.array([pt[blade_key][k] for pt in curve])
        result[k] = float(np.interp(des_PR, PR_arr, vals))
    return result

des_PR = perf['Pt_ratio']

fig_bar, axes_bar = plt.subplots(1, 2, figsize=(13, 6))
fig_bar.suptitle(
    f'Loss Coefficients at Design PR = {des_PR:.3f} — Fixed vs VGT\n'
    '100% Speed  |  Design Area (1.0)',
    fontsize=11, fontweight='bold')

for ax, blade_key, blade_title in [
    (axes_bar[0], 'loss_s', 'Stator'),
    (axes_bar[1], 'loss_r', 'Rotor'),
]:
    l_fixed = get_losses_at_PR(des_curve_fixed, des_PR, blade_key)
    l_vgt   = get_losses_at_PR(des_curve_vgt,   des_PR, blade_key)

    keys_plot  = loss_stack_keys + ['zeta_tot']
    labels_plot = loss_stack_labels + ['Total']
    colors_plot = loss_stack_colors + ['#8172B3']

    x     = np.arange(len(keys_plot))
    width = 0.35

    bars_f = ax.bar(x - width/2,
                    [l_fixed[k] for k in keys_plot],
                    width, label='Fixed',
                    color=[c + 'BB' for c in colors_plot],
                    edgecolor='k', lw=0.7)
    bars_v = ax.bar(x + width/2,
                    [l_vgt[k]   for k in keys_plot],
                    width, label='VGT',
                    color=colors_plot,
                    edgecolor='k', lw=0.7, hatch='///')

    # Annotate values on each bar
    for bar in bars_f:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color='k')
    for bar in bars_v:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.0003,
                f'{h:.4f}', ha='center', va='bottom', fontsize=7, color='k')

    # Delta annotation on total bar
    delta = l_vgt['zeta_tot'] - l_fixed['zeta_tot']
    x_tot = x[-1]
    y_top = max(l_fixed['zeta_tot'], l_vgt['zeta_tot'])
    ax.annotate(
        f'Δ = {delta:+.4f}\n({delta/l_fixed["zeta_tot"]*100:+.2f}%)',
        xy=(x_tot + width/2, l_vgt['zeta_tot']),
        xytext=(x_tot + width/2 + 0.4, y_top + 0.003),
        fontsize=8, color='C3', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                  ec='C3', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='C3', lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=9)
    ax.set_ylabel('Loss coefficient  ζ  (—)', fontsize=10)
    ax.set_title(f'{blade_title} — Design Point Losses', fontsize=10,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('loss_bar_design_PR.png', dpi=150, bbox_inches='tight')
print('  Saved: loss_bar_design_PR.png')
plt.show()

 
# ── Plot 2: Efficiency at 100% speed — all area fractions ────────────────────
area_colors = {a: cmap_area(i / (len(unique_areas) - 1))
               for i, a in enumerate(unique_areas)}
fig_eta, ax_eta = plt.subplots(figsize=(10, 6))
fig_eta.suptitle('Efficiency at 100% Speed — All Area Fractions',
                 fontsize=11, fontweight='bold')
 
for area_frac in unique_areas:
    curve = all_curves.get((1.00, area_frac), [])
    if not curve: continue
    PRs  = [pt['PR']       for pt in curve]
    etas = [pt['eta'] * 100 for pt in curve]
    ax_eta.plot(PRs, etas, color=area_colors[area_frac], lw=2, marker='o',
                ms=4, label=f'Area: {area_frac:.2f}')
 
ax_eta.axvline(perf['Pt_ratio'], color='gray', ls=':', lw=1.5,
               label=f"Design PR={perf['Pt_ratio']:.2f}")
ax_eta.axhline(perf['eta_tt'] * 100, color='gray', ls='--', lw=1)
ax_eta.set_xlabel('PR'); ax_eta.set_ylabel('η (%)'); ax_eta.set_ylim(60, 100)
ax_eta.set_title('Efficiency — All Area Fractions at 100% Speed')
ax_eta.legend(fontsize=9); ax_eta.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('EngineHPT_eta_area_sweep.png', dpi=150)
plt.show()
 
# ── Plot 3: Stacked corrected flow — markers=speed, colors=area ──────────────
fig_flow, ax_flow = plt.subplots(figsize=(12, 7))
fig_flow.suptitle('Corrected Mass Flow — All Speeds and Area Fractions',
                  fontsize=11, fontweight='bold')
 
for (frac, area_frac), curve in all_curves.items():
    if not curve: continue
    PRs = [pt['PR'] for pt in curve]
    Wcs = [corrected_mdot(pt['mdot']) * KG_TO_LBM for pt in curve]
    ax_flow.plot(PRs, Wcs,
                 color=area_colors[area_frac],
                 marker=speed_markers[frac],
                 linestyle='-', lw=1.5, ms=4)
 
# Legend
legend_handles = [
    mlines.Line2D([], [], color='none',
                  label=r'$\mathbf{Area\ Fractions\ (Colors)}$')]
for area_frac in unique_areas:
    legend_handles.append(
        mlines.Line2D([], [], color=area_colors[area_frac],
                      lw=2, label=f'Area: {area_frac:.2f}'))
legend_handles.append(
    mlines.Line2D([], [], color='none',
                  label=r'$\mathbf{Speed\ Fractions\ (Markers)}$'))
for frac in unique_speeds:
    legend_handles.append(
        mlines.Line2D([], [], color='gray',
                      marker=speed_markers[frac],
                      linestyle='none', ms=6,
                      label=f'Speed: {frac*100:.0f}%'))
 
ax_flow.set_xlabel('PR'); ax_flow.set_ylabel('$W_c$ (lbm/s)')
ax_flow.set_title('Corrected Mass Flow Map (All Configurations)')
ax_flow.legend(handles=legend_handles, loc='best', fontsize=8)
ax_flow.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('EngineHPT_flow_stacked.png', dpi=150)
plt.show()
 
# ── Plot 4: Peak efficiency vs area at design speed ───────────────────────────
eta_vs_area = []
for area_frac in unique_areas:
    curve = all_curves.get((1.00, area_frac), [])
    eta_vs_area.append(max([pt['eta'] * 100 for pt in curve]) if curve else np.nan)

fig_surf, ax_surf = plt.subplots(figsize=(8, 5))
ax_surf.plot(unique_areas, eta_vs_area, 'o-', color='royalblue', lw=2, ms=7)
ax_surf.axvline(1.0, color='gray', ls='--', lw=1.2, label='Design area')
ax_surf.axhline(eta_vs_area[unique_areas.index(1.0)] if 1.0 in unique_areas
                else eta_vs_area[int(len(eta_vs_area)//2)],
                color='gray', ls=':', lw=1.0)
ax_surf.set_xlabel('Area Fraction', fontsize=11)
ax_surf.set_ylabel('Peak η (%)', fontsize=11)
ax_surf.set_title('Peak Efficiency vs NGV Throat Area\n100% Speed', fontsize=11)
ax_surf.legend(fontsize=9)
ax_surf.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('EngineHPT_eta_surface.png', dpi=150)
plt.show()
 
print("\n  Done. Files written:")
print(f"    EngineHPT_map.py             — pyCycle MapData")
print(f"    EngineHPT_validation.png     — 100% speed validation")
print(f"    EngineHPT_eta_area_sweep.png — efficiency by area fraction")
print(f"    EngineHPT_flow_stacked.png   — corrected flow all configs")
print(f"    EngineHPT_eta_surface.png    — peak efficiency surface")
print(f"\n  To use in pyCycle:")
print(f"    from EngineHPT_map import EngineHPTMap")
print(f"    TURB_MAP = EngineHPTMap")
print(f"    # Connect parmGeom to s_Wp or area schedule in off-design")