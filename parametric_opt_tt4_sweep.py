"""
parametric_opr_tt4_sweep.py
----------------------------
2D parametric sweep over OPR × Tt4 comparing Fixed vs VGT turbine.
Physics dP/P only. CdA anchored from fixed engine and shared with VGT
so the combustor model is consistent across both configurations.

Outputs
-------
  vgt_benefit_contour.png   — Δη_th contour map (VGT minus Fixed)
  delta_eta_opr_tt4.npy     — raw array, shape (n_OPR, n_Tt4, n_fn)
  OPR_vals.npy
  Tt4_vals.npy

Usage
-----
  python parametric_opr_tt4_sweep.py

Imports from combustor_pycycle_integration.py — ensure that file and
EngineHPT_map / EngineHPT_map_fixed are importable from the same directory.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import openmdao.api as om
import pycycle.api as pyc
from combustor_py_cycle import (
    _make_mp_single_fn,
    run_physics_dPqP_sweep,
    extract_design,
    calc_CdA_liner,
    map_design_point,
    EngineHPTFixedMap,
    EngineHPTVGTMap,
    psi_to_pa,
    lbm_s_to_kg_s,
    DPQP_FIXED,
    RANKINE_TO_K,
    ETA_COOL,
    estimate_T3_rankine,
    gauntner_cooling_fracs,
    turb_rel_temp_rankine,
    T_METAL_VANE_R,
    T_METAL_BLADE_R,
    PSI_DESIGN,
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_Nmech = 8070.0   # rpm

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP PARAMETERS  ←  edit here
# ══════════════════════════════════════════════════════════════════════════════

OPR_vals       = np.linspace(5.0,  15.0, 6)   # 6 pts, ~2-unit steps
Tt4_vals       = np.linspace(1100, 1600, 6)   # K — converted to °R inside loop
FN_FRACS       = [0.60, 0.80]                        # thrust fractions to evaluate

# A4 ranges
A4_SCALARS_VGT   = list(np.linspace(0.80, 1.00, 9))  # 9 pts, step 0.025
A4_SCALARS_FIXED = [1.0]                              # fixed geometry — single point

# Design point (for contour marker)
OPR_DESIGN = 13.5
Tt4_DESIGN_K = 2370.0 * RANKINE_TO_K   # ≈ 1316.7 K

# ══════════════════════════════════════════════════════════════════════════════
#  BASE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_CFG = dict(
    comp_map     = pyc.AXI5,
    Fn_design    = 11800.0,   # lbf
    T4_design    = Tt4_DESIGN_K,  # degR
    OPR          = 13.5,
    Nmech        = _Nmech,    # rpm
    comp_eff     = 0.83,
    inlet_MN     = 0.60,
    comp_MN      = 0.020,
    burner_MN    = 0.020,
    turb_MN      = 0.40,
    nozz_Cv      = 0.99,
    FAR_guess    = 0.0175,
    W_guess      = 168.0,
    fn_fractions = FN_FRACS,
    a4_scalars   = [1.0],     # single OD point; Brent drives actual A4 for VGT
)

# Size design cooling fracs from Gauntner at the reference design point
_T3_des_R = estimate_T3_rankine(OPR_DESIGN, comp_eff=BASE_CFG['comp_eff'])
_frac_ngv_des, _frac_blade_des = gauntner_cooling_fracs(
    Tt4_DESIGN_K * 9.0 / 5.0, _T3_des_R)
BASE_CFG['ngv_cool_frac'] = _frac_ngv_des
BASE_CFG['bld_cool_frac'] = _frac_blade_des
print(f"  Gauntner design sizing: T3_est={_T3_des_R:.1f}°R")
print(f"  NGV={_frac_ngv_des*100:.2f}%  Blade={_frac_blade_des*100:.2f}%"
      f"  Total={(_frac_ngv_des+_frac_blade_des)*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-POINT PROBLEM BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_problem(cfg: dict, mode: str) -> tuple:
    """
    Construct and set up an MPSingleFn OpenMDAO problem.
    Returns (prob, mp) ready for run_model().
    """
    MPSingleFn = _make_mp_single_fn(cfg)
    prob = om.Problem()
    mp   = prob.model = MPSingleFn()
    prob.setup(check=False)

    prob.set_val('DESIGN.fc.alt',            0.0,              units='ft')
    prob.set_val('DESIGN.fc.MN',             0.000001)
    prob.set_val('DESIGN.balance.Fn_target', cfg['Fn_design'], units='lbf')
    prob.set_val('DESIGN.balance.T4_target', cfg['T4_design'], units='degR')
    prob.set_val('DESIGN.comp.PR',           cfg['OPR'])
    prob.set_val('DESIGN.comp.eff',          cfg['comp_eff'])
    prob.set_val('DESIGN.turb.eff',          cfg['turb_eff'])
    prob.set_val('DESIGN.burner.dPqP',       DPQP_FIXED)

    if mode == 'alpha':
        prob.set_val('DESIGN.turb.map.alphaMap', 1.0)

    prob['DESIGN.balance.FAR']     = cfg.get('FAR_guess',     0.0175)
    prob['DESIGN.balance.W']       = cfg.get('W_guess',       168.0)
    prob['DESIGN.balance.turb_PR'] = cfg.get('turb_PR_guess', 3.5)
    prob['DESIGN.fc.balance.Pt']   = 14.696
    prob['DESIGN.fc.balance.Tt']   = 518.67

    for pt in mp.od_pts:
        prob.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 0

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    return prob, mp


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _quad_peak(a4s, vals):
    """
    Fit a quadratic to (a4s, vals) and return the peak value.
    Falls back to raw max if the fit is not concave or peak is out of range.
    """
    coeffs = np.polyfit(a4s, vals, 2)
    if coeffs[0] >= 0:             # not concave — no interior maximum
        return float(np.max(vals))
    a4_peak = -coeffs[1] / (2.0 * coeffs[0])
    if a4s[0] <= a4_peak <= a4s[-1]:
        return float(np.polyval(coeffs, a4_peak))
    return float(np.max(vals))


def _best_eta(results_by_fn: dict, fn_frac: float) -> float:
    """Peak η_th over A4 sweep via quadratic fit (reduces argmax noise)."""
    pts   = results_by_fn.get(fn_frac, [])
    valid = sorted([(r['a4_scalar'], r['eta_th']) for r in pts
                    if np.isfinite(r.get('eta_th', np.nan))],
                   key=lambda x: x[0])
    if not valid:
        return np.nan
    a4s, etas = zip(*valid)
    if len(valid) < 3:
        return float(np.max(etas))
    return _quad_peak(np.array(a4s), np.array(etas))


def _mean_eta(results_by_fn: dict, fn_frac: float) -> float:
    """Mean η_th over A4 sweep (fixed engine has only one point)."""
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['eta_th'] for r in pts if np.isfinite(r.get('eta_th', np.nan))]
    return float(np.mean(vals)) if vals else np.nan


def _best_T_metal(results_by_fn: dict, fn_frac: float) -> float:
    """T_metal interpolated at the quadratic-peak A4 for η_th."""
    pts   = results_by_fn.get(fn_frac, [])
    valid = sorted([r for r in pts if np.isfinite(r.get('eta_th', np.nan))
                    and np.isfinite(r.get('T_metal', np.nan))],
                   key=lambda r: r['a4_scalar'])
    if not valid:
        return np.nan
    a4s  = np.array([r['a4_scalar'] for r in valid])
    etas = np.array([r['eta_th']    for r in valid])
    tms  = np.array([r['T_metal']   for r in valid])
    if len(valid) < 3:
        return float(max(valid, key=lambda r: r['eta_th'])['T_metal'])
    # Find optimal A4 from quadratic fit to η_th
    coeffs = np.polyfit(a4s, etas, 2)
    if coeffs[0] >= 0 or not (a4s[0] <= -coeffs[1]/(2*coeffs[0]) <= a4s[-1]):
        return float(max(valid, key=lambda r: r['eta_th'])['T_metal'])
    a4_opt = -coeffs[1] / (2.0 * coeffs[0])
    # Interpolate T_metal at that A4
    return float(np.interp(a4_opt, a4s, tms))


def _mean_T_metal(results_by_fn: dict, fn_frac: float) -> float:
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['T_metal'] for r in pts if np.isfinite(r.get('T_metal', np.nan))]
    return float(np.mean(vals)) if vals else np.nan


def _best_T4(results_by_fn: dict, fn_frac: float) -> float:
    """T4 (K) interpolated at the quadratic-peak A4 for η_th."""
    pts   = results_by_fn.get(fn_frac, [])
    valid = sorted([r for r in pts if np.isfinite(r.get('eta_th', np.nan))
                    and np.isfinite(r.get('T4', np.nan))],
                   key=lambda r: r['a4_scalar'])
    if not valid:
        return np.nan
    a4s  = np.array([r['a4_scalar'] for r in valid])
    etas = np.array([r['eta_th']    for r in valid])
    t4s  = np.array([r['T4']        for r in valid]) * RANKINE_TO_K
    if len(valid) < 3:
        return float(max(valid, key=lambda r: r['eta_th'])['T4']) * RANKINE_TO_K
    coeffs = np.polyfit(a4s, etas, 2)
    if coeffs[0] >= 0 or not (a4s[0] <= -coeffs[1]/(2*coeffs[0]) <= a4s[-1]):
        return float(max(valid, key=lambda r: r['eta_th'])['T4']) * RANKINE_TO_K
    a4_opt = -coeffs[1] / (2.0 * coeffs[0])
    return float(np.interp(a4_opt, a4s, t4s))


def _mean_T4(results_by_fn: dict, fn_frac: float) -> float:
    """Mean T4 (K) over A4 sweep (fixed engine has only one point)."""
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['T4'] * RANKINE_TO_K for r in pts if np.isfinite(r.get('T4', np.nan))]
    return float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  BLADE TEMPERATURE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _add_blade_temp(results_by_fn: dict, bld_frac: float):
    """
    Annotate each result dict in-place with T_metal (K) for the rotor blade.
    Uses relative stagnation temperature (T0_rel = T4 - ΔTt/(2ψ)) — what the
    rotating blade actually sees — and blade-only cooling fraction.
    """
    for _, pts in results_by_fn.items():
        for r in pts:
            Tt4_rel_K = turb_rel_temp_rankine(r['T4'], r['T4_exit']) * RANKINE_TO_K
            T3_K      = r['T3'] * RANKINE_TO_K
            r['T_metal'] = Tt4_rel_K - ETA_COOL * bld_frac * (Tt4_rel_K - T3_K)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE (OPR, Tt4) CELL SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_cell(opr: float, tt4_K: float,
               prob_fixed, mp_fixed, prob_vgt, mp_vgt,
               dPqP_prev_fixed=None, dPqP_prev_vgt=None) -> dict | None:
    """
    Re-run design solve + physics sweep for one (OPR, Tt4) cell.
    Problems are pre-built and reused across cells for speed.
    CdA is anchored from the design point (no separate fixed-dP/P sweep).
    Cooling fracs are re-sized via Gauntner for each (OPR, Tt4) cell so that
    every design point has self-consistent hardware cooling.
    """
    tt4_R  = tt4_K * 9.0 / 5.0
    Np_des = _Nmech / np.sqrt(tt4_R)
    eff_fixed, _ = map_design_point(EngineHPTFixedMap, Np_des)
    eff_vgt,   _ = map_design_point(EngineHPTVGTMap,   Np_des)

    # Gauntner cooling sizing for this (OPR, Tt4) — analytical estimate
    # (consistent with cool_frac_gauntner grid computed before the sweep)
    T3_est_R     = estimate_T3_rankine(opr, comp_eff=BASE_CFG['comp_eff'])
    dTt_comp_R   = T3_est_R - 518.67
    T4_exit_est_R = tt4_R - dTt_comp_R
    frac_ngv_cell, frac_blade_cell = gauntner_cooling_fracs(
        tt4_R, T3_est_R, T4_exit_R=T4_exit_est_R)

    cell_out   = {}
    dPqP_out   = {}   # converged dP/P per label, returned for warm-starting next cell

    for label, prob, mp, mode, turb_eff, dPqP_prev in [
        ('fixed',   prob_fixed, mp_fixed, 'sWp',   eff_fixed, dPqP_prev_fixed),
        ('vgt_own', prob_vgt,   mp_vgt,   'alpha', eff_vgt,   dPqP_prev_vgt),
    ]:
        try:
            # Disable OD solvers so only the design point runs
            for pt in mp.od_pts:
                prob.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 0

            # Update per-cell design inputs
            prob.set_val('DESIGN.comp.PR',           opr)
            prob.set_val('DESIGN.balance.T4_target', tt4_R, units='degR')
            prob.set_val('DESIGN.turb.eff',          turb_eff)

            prob.run_model()
            des = extract_design(prob)

            T3_des_K = des['T3_design'] * RANKINE_TO_K

            # CdA anchored directly from design point
            des['CdA_liner'] = calc_CdA_liner(
                T3_des_K,
                psi_to_pa(des['P3_design']),
                lbm_s_to_kg_s(des['W']))
            print(f"    [{label}] CdA = {des['CdA_liner']*1e4:.4f} cm²"
                  f"  cool: NGV={frac_ngv_cell*100:.2f}%  bld={frac_blade_cell*100:.2f}%")

            # A4 optimisation or sweep — warm-start from previous cell's converged dP/P
            phys_dpqp  = {}
            _dPqP_prev = dPqP_prev
            for fn_frac in FN_FRACS:
                phys_dpqp[fn_frac], _dPqP_prev = run_physics_dPqP_sweep(
                    prob, mp, des, fn_frac, BASE_CFG, mode,
                    dPqP_init=_dPqP_prev)
                # Nullify results where dP/P hit a clip bound — nozzle Newton failed
                for r in phys_dpqp[fn_frac]:
                    dpqp = r.get('dPqP', 0.05)
                    if dpqp <= 0.002 or dpqp >= 0.49:
                        r['eta_th'] = np.nan
            dPqP_out[label] = _dPqP_prev

            # Blade temperature — cell-specific blade frac, relative stagnation temperature
            _add_blade_temp(phys_dpqp, frac_blade_cell)

            cell_out[label] = {
                'des':        des,
                'phys_dpqp':  phys_dpqp,
                'frac_ngv':   frac_ngv_cell,
                'frac_blade': frac_blade_cell,
            }

        except Exception as exc:
            print(f"    !! FAILED [{label}] OPR={opr:.1f} Tt4={tt4_K:.0f}K: {exc}")
            cell_out[label] = None

    if cell_out.get('fixed') and cell_out.get('vgt_own'):
        return cell_out, dPqP_out
    return None, {}


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD PROBLEMS ONCE  (reused across the entire OPR × Tt4 grid)
# ══════════════════════════════════════════════════════════════════════════════

_Np_des_template              = _Nmech / np.sqrt(Tt4_DESIGN_K * 9.0 / 5.0)
_eff_fixed_tmpl, _PR_fixed_tmpl = map_design_point(EngineHPTFixedMap, _Np_des_template)
_eff_vgt_tmpl,   _PR_vgt_tmpl   = map_design_point(EngineHPTVGTMap,   _Np_des_template)

_cfg_fixed_tmpl = {
    **BASE_CFG,
    'turb_map'     : EngineHPTFixedMap,
    'turb_eff'     : _eff_fixed_tmpl,
    'turb_PR_guess': _PR_fixed_tmpl,
    'OPR'          : OPR_DESIGN,
    'T4_design'    : Tt4_DESIGN_K * 9.0 / 5.0,
    'a4_scalars'   : A4_SCALARS_FIXED,
}
_cfg_vgt_tmpl = {
    **BASE_CFG,
    'turb_map'     : EngineHPTVGTMap,
    'turb_eff'     : _eff_vgt_tmpl,
    'turb_PR_guess': _PR_vgt_tmpl,
    'OPR'          : OPR_DESIGN,
    'T4_design'    : Tt4_DESIGN_K * 9.0 / 5.0,
    'a4_scalars'   : A4_SCALARS_VGT,
}

print('\n  Building Fixed problem (one-time setup)...')
prob_fixed, mp_fixed = _build_problem(_cfg_fixed_tmpl, 'sWp')
prob_fixed.run_model()

print('\n  Building VGT problem (one-time setup)...')
prob_vgt, mp_vgt = _build_problem(_cfg_vgt_tmpl, 'alpha')
prob_vgt.run_model()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SWEEP LOOP
# ══════════════════════════════════════════════════════════════════════════════

n_opr = len(OPR_vals)
n_tt4 = len(Tt4_vals)
n_fn  = len(FN_FRACS)


# ── Result arrays ─────────────────────────────────────────────────────────────

delta_eta_own     = np.full((n_opr, n_tt4, n_fn), np.nan)
delta_T_metal_own = np.full((n_opr, n_tt4, n_fn), np.nan)
delta_T4_own      = np.full((n_opr, n_tt4, n_fn), np.nan)  # VGT T4 − Fixed T4 (K)

# Gauntner-prescribed cooling fraction grid (analytical — no cycle solve needed)
# cool_frac_gauntner[i, j] = total required cooling at (OPR_vals[i], Tt4_vals[j])
# T4_exit estimated via shaft balance: turbine drops ~same ΔT as compressor rises.
cool_frac_gauntner = np.zeros((n_opr, n_tt4, 2))  # [:,:,0]=NGV  [:,:,1]=blade
for i, opr in enumerate(OPR_vals):
    T3_est_R    = estimate_T3_rankine(opr, comp_eff=BASE_CFG['comp_eff'])
    dTt_comp_R  = T3_est_R - 518.67   # compressor temperature rise
    for j, tt4_K in enumerate(Tt4_vals):
        tt4_R      = tt4_K * 9.0 / 5.0
        tt4_exit_R = tt4_R - dTt_comp_R  # shaft-balance estimate of turbine exit T
        frac_ngv, frac_blade = gauntner_cooling_fracs(tt4_R, T3_est_R,
                                                       T4_exit_R=tt4_exit_R)
        cool_frac_gauntner[i, j, 0] = frac_ngv   * 100
        cool_frac_gauntner[i, j, 1] = frac_blade * 100

# Save analytical data and axis arrays immediately — plotting script can use
# these even during a partial run.
np.save('OPR_vals.npy',           OPR_vals)
np.save('Tt4_vals.npy',           Tt4_vals)
np.save('cool_frac_gauntner.npy', cool_frac_gauntner)
print('  Analytical arrays saved.')

total = n_opr * n_tt4
done  = 0

# Warm-start dP/P seeds — updated after each successful cell
dPqP_prev_fixed = None
dPqP_prev_vgt   = None

for i, opr in enumerate(OPR_vals):
    for j, tt4_K in enumerate(Tt4_vals):
        done += 1
        print(f"\n{'='*60}")
        print(f"  [{done}/{total}]  OPR={opr:.1f}  Tt4={tt4_K:.0f}K"
              f"  ({tt4_K*9/5:.0f}°R)")
        print(f"{'='*60}")

        cell, dPqP_out = solve_cell(opr, tt4_K, prob_fixed, mp_fixed, prob_vgt, mp_vgt,
                                    dPqP_prev_fixed=dPqP_prev_fixed,
                                    dPqP_prev_vgt=dPqP_prev_vgt)
        if cell is None:
            print(f"  → cell skipped (NaN stored)")
            continue

        dPqP_prev_fixed = dPqP_out.get('fixed',   dPqP_prev_fixed)
        dPqP_prev_vgt   = dPqP_out.get('vgt_own', dPqP_prev_vgt)

        for k, fn_frac in enumerate(FN_FRACS):
            eta_f     = _mean_eta(cell['fixed']['phys_dpqp'],   fn_frac)
            eta_v_own = _best_eta(cell['vgt_own']['phys_dpqp'], fn_frac)

            if np.isfinite(eta_f) and np.isfinite(eta_v_own):
                delta_eta_own[i, j, k] = (eta_v_own - eta_f) / eta_f * 100.0

            Tm_f     = _mean_T_metal(cell['fixed']['phys_dpqp'],   fn_frac)
            Tm_v_own = _best_T_metal(cell['vgt_own']['phys_dpqp'], fn_frac)

            if np.isfinite(Tm_f) and np.isfinite(Tm_v_own):
                delta_T_metal_own[i, j, k] = Tm_v_own - Tm_f

            T4_f     = _mean_T4(cell['fixed']['phys_dpqp'],   fn_frac)
            T4_v_own = _best_T4(cell['vgt_own']['phys_dpqp'], fn_frac)

            if np.isfinite(T4_f) and np.isfinite(T4_v_own):
                delta_T4_own[i, j, k] = T4_v_own - T4_f

            print(f"  Fn={fn_frac*100:.0f}%  η_fixed={eta_f*100:.3f}%  "
                  f"η_vgt_own={eta_v_own*100:.3f}%  "
                  f"Δη_own={delta_eta_own[i,j,k]:+.3f}%  "
                  f"ΔTm_own={delta_T_metal_own[i,j,k]:+.1f}K  "
                  f"ΔT4={delta_T4_own[i,j,k]:+.1f}K")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

np.save('delta_eta_own_opr_tt4.npy',     delta_eta_own)
np.save('delta_T_metal_own_opr_tt4.npy', delta_T_metal_own)
np.save('delta_T4_own_opr_tt4.npy',      delta_T4_own)
np.save('cool_frac_gauntner.npy',        cool_frac_gauntner)
np.save('OPR_vals.npy',                  OPR_vals)
np.save('Tt4_vals.npy',                  Tt4_vals)
print('\n  Data saved.')


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

OPR_grid, Tt4_grid = np.meshgrid(OPR_vals, Tt4_vals, indexing='ij')


def _contour_panel(ax, Z, title, cbar_label, cmap='RdBu', symmetric=True):
    """
    Draw a filled contour + zero-crossing on ax.
    Returns the contourf mappable for colorbar attachment.
    """
    valid = Z[np.isfinite(Z)]
    if valid.size == 0:
        ax.set_title(title + '\n(no data)')
        return None

    if symmetric:
        vmax = max(np.abs(valid.min()), np.abs(valid.max()))
        vmin = -vmax
    else:
        vmin, vmax = valid.min(), valid.max()

    # Mask NaN for contourf
    Zm = np.ma.masked_invalid(Z)

    cf = ax.contourf(OPR_grid, Tt4_grid, Zm,
                     levels=20, cmap=cmap, vmin=vmin, vmax=vmax)

    # Zero-crossing (break-even line) — only if data spans zero
    if vmin < 0 < vmax:
        cs = ax.contour(OPR_grid, Tt4_grid, Zm,
                        levels=[0.0], colors='k', linewidths=2.0)
        ax.clabel(cs, fmt='Break-even', fontsize=8, inline=True)

    # Design point marker
    ax.plot(OPR_DESIGN, Tt4_DESIGN_K, '*',
            ms=14, color='gold', markeredgecolor='k',
            zorder=6, label='Design point')

    ax.set_xlabel('OPR',              fontsize=10)
    ax.set_ylabel('Tt4 (K)',          fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.15, color='white')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    return cf


# ── Figure 1: Δη_th contour ───────────────────────────────────────────────────
fig1, axes1 = plt.subplots(n_fn, 1,
                            figsize=(7, 6 * n_fn),
                            squeeze=False)

fig1.suptitle(
    'VGT vs Fixed Turbine — Δη_th contour  (Red = VGT better)',
    fontsize=11, fontweight='bold')

for k, fn_frac in enumerate(FN_FRACS):
    cf = _contour_panel(axes1[k, 0], delta_eta_own[:, :, k],
                        title=f'Δη_th (%) — {fn_frac*100:.0f}% Fn',
                        cbar_label='Δη_th (%)',
                        cmap='RdBu', symmetric=True)
    if cf is not None:
        plt.colorbar(cf, ax=axes1[k, 0], label='Δη_th (%) VGT − Fixed',
                     shrink=0.85, pad=0.03)

fig1.tight_layout()
fig1.savefig('vgt_benefit_contour.png', dpi=150, bbox_inches='tight')
print('  Saved: vgt_benefit_contour.png')


# ── Figure 2: ΔT_metal contour ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(n_fn, 1,
                            figsize=(7, 6 * n_fn),
                            squeeze=False)

fig2.suptitle(
    'VGT vs Fixed — ΔT_metal (K)  (Negative = VGT blade runs cooler)',
    fontsize=11, fontweight='bold')

for k, fn_frac in enumerate(FN_FRACS):
    cf = _contour_panel(axes2[k, 0], delta_T_metal_own[:, :, k],
                        title=f'ΔT_metal (K) — {fn_frac*100:.0f}% Fn',
                        cbar_label='ΔT_metal (K)',
                        cmap='RdBu_r', symmetric=True)
    if cf is not None:
        plt.colorbar(cf, ax=axes2[k, 0], label='ΔT_metal (K) VGT − Fixed',
                     shrink=0.85, pad=0.03)

fig2.tight_layout()
fig2.savefig('vgt_T_metal_contour.png', dpi=150, bbox_inches='tight')
print('  Saved: vgt_T_metal_contour.png')


# ── Figure 3: Gauntner cooling prescription contour ──────────────────────────
# Solid curves = theoretical prescription at each (OPR, Tt4) — analytical
# Dashed contour = design cooling fraction (what the cycle actually uses)
_frac_des_total_pct = (_frac_ngv_des + _frac_blade_des) * 100
cool_total = cool_frac_gauntner[:, :, 0] + cool_frac_gauntner[:, :, 1]

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig3.suptitle(
    'Gauntner Cooling Prescription vs (OPR, Tt4)  —  NASA-TM-81453\n'
    f'T_vane={T_METAL_VANE_R:.0f}°R  T_blade={T_METAL_BLADE_R:.0f}°R  |  '
    f'Dashed = design fraction ({_frac_des_total_pct:.2f}% total)',
    fontsize=11, fontweight='bold')

for ax, Z, title, cmap in [
    (axes3[0], cool_frac_gauntner[:, :, 0], 'NGV prescription (% W_gas)',   'Blues'),
    (axes3[1], cool_frac_gauntner[:, :, 1], 'Blade prescription (% W_gas)', 'Reds'),
    (axes3[2], cool_total,                   'Total prescription (% W_gas)', 'Purples'),
]:
    cf = ax.contourf(OPR_grid, Tt4_grid, Z, levels=20, cmap=cmap)
    plt.colorbar(cf, ax=ax, label='% W_gas', shrink=0.85, pad=0.03)

    # Design prescription as a single contour line
    if Z is cool_total:
        cs = ax.contour(OPR_grid, Tt4_grid, Z,
                        levels=[_frac_des_total_pct],
                        colors='k', linewidths=2.0, linestyles='--')
        ax.clabel(cs, fmt=lambda _: f'Design {_frac_des_total_pct:.2f}%',
                  fontsize=8, inline=True)

    ax.plot(OPR_DESIGN, Tt4_DESIGN_K, '*',
            ms=14, color='gold', markeredgecolor='k', zorder=6, label='Design point')
    ax.set_xlabel('OPR', fontsize=10)
    ax.set_ylabel('Tt4 (K)', fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.15, color='white')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))

fig3.tight_layout()
fig3.savefig('cool_frac_contour.png', dpi=150, bbox_inches='tight')
print('  Saved: cool_frac_contour.png')


plt.show()

# ── Terminal summary ──────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('  PARAMETRIC SWEEP COMPLETE')
print('=' * 70)
print(f"  {'OPR':>6}  {'Tt4 (K)':>8}", end='')
for fn_frac in FN_FRACS:
    print(f"  Δη_own@{fn_frac*100:.0f}%  ΔTm_own@{fn_frac*100:.0f}%", end='')
print()
print('-' * 80)
for i, opr in enumerate(OPR_vals):
    for j, tt4_K in enumerate(Tt4_vals):
        print(f"  {opr:6.1f}  {tt4_K:8.0f}", end='')
        for k in range(n_fn):
            vo = delta_eta_own[i, j, k]
            tm = delta_T_metal_own[i, j, k]
            so  = f"{vo:+10.3f}%" if np.isfinite(vo) else f"  {'NaN':>10}"
            stm = f"{tm:+10.1f}K" if np.isfinite(tm) else f"  {'NaN':>10}"
            print(f"  {so}  {stm}", end='')
        print()