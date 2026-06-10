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
from scipy.interpolate import RegularGridInterpolator

from combustor_py_cycle import (
    _make_mp_single_fn,
    run_fixed_dPqP_sweep,
    run_physics_dPqP_sweep,
    extract_design,
    calc_CdA_liner,
    rankine_to_kelvin,
    psi_to_pa,
    lbm_s_to_kg_s,
    DPQP_FIXED,
    RANKINE_TO_K,
)
from EngineHPT_map       import EngineHPTMap as EngineHPTVGTMap
from EngineHPT_map_fixed import EngineHPTMap as EngineHPTFixedMap

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

ETA_COOL    = 0.65
T_METAL_MAX = 1200.0   # K — blade material limit
_Nmech      = 8070.0   # rpm

# ══════════════════════════════════════════════════════════════════════════════
#  SWEEP PARAMETERS  ←  edit here
# ══════════════════════════════════════════════════════════════════════════════

OPR_vals       = np.linspace(8.0,  15.0, 6)   # 6 pts, ~2-unit steps
Tt4_vals       = np.linspace(1300, 1600, 6)   # K — converted to °R inside loop
FN_FRACS       = [0.50, 1.00]                  # thrust fractions to evaluate

# A4 ranges
A4_SCALARS_VGT   = list(np.linspace(0.80, 1.10, 7))   # VGT sweep
A4_SCALARS_FIXED = [1.0]                                # fixed geometry — single point

# Design point (for contour marker)
OPR_DESIGN = 13.5
Tt4_DESIGN_K = 2370.0 * RANKINE_TO_K   # ≈ 1316.7 K

# ══════════════════════════════════════════════════════════════════════════════
#  BASE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

BASE_CFG = dict(
    comp_map     = pyc.HPCMap,
    Fn_design    = 11800.0,     # lbf
    Nmech        = _Nmech,      # rpm
    comp_eff     = 0.83,
    inlet_MN     = 0.60,
    comp_MN      = 0.020,
    burner_MN    = 0.020,
    turb_MN      = 0.40,
    nozz_Cv      = 0.99,
    FAR_guess    = 0.0175,
    W_guess      = 168.0,
    fn_fractions = FN_FRACS,
    cool_frac    = 0.10,        # placeholder — overridden after design solve
)


# ══════════════════════════════════════════════════════════════════════════════
#  MAP DESIGN POINT HELPER
# ══════════════════════════════════════════════════════════════════════════════

def map_design_point(turb_map, Np_des: float, alpha_design: float = 1.0):
    """Read design efficiency and PR from map at Np_des."""
    a_idx  = int(np.argmin(np.abs(np.array(turb_map.alphaMap) - alpha_design)))
    eff_sl = np.array(turb_map.effMap[a_idx])
    NpMap  = np.array(turb_map.NpMap)
    PRmap  = np.array(turb_map.PRmap)
    PR_des = turb_map.defaults['PRmap']
    Np_q   = float(np.clip(Np_des, NpMap[0], NpMap[-1]))
    interp = RegularGridInterpolator(
        (NpMap, PRmap), eff_sl,
        method='linear', bounds_error=False, fill_value=None)
    return float(interp([[Np_q, PR_des]])), PR_des


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

def _best_eta(results_by_fn: dict, fn_frac: float) -> float:
    """Max η_th over A4 sweep for a given Fn level. Returns NaN on empty."""
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['eta_th'] for r in pts if np.isfinite(r.get('eta_th', np.nan))]
    return float(np.max(vals)) if vals else np.nan


def _mean_eta(results_by_fn: dict, fn_frac: float) -> float:
    """Mean η_th over A4 sweep (fixed engine has only one point)."""
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['eta_th'] for r in pts if np.isfinite(r.get('eta_th', np.nan))]
    return float(np.mean(vals)) if vals else np.nan


def _best_T_metal(results_by_fn: dict, fn_frac: float) -> float:
    """T_metal at the A4 that maximises η_th. Returns NaN on empty."""
    pts = results_by_fn.get(fn_frac, [])
    pts_valid = [r for r in pts if np.isfinite(r.get('eta_th', np.nan))
                                and np.isfinite(r.get('T_metal', np.nan))]
    if not pts_valid:
        return np.nan
    return float(max(pts_valid, key=lambda r: r['eta_th'])['T_metal'])


def _mean_T_metal(results_by_fn: dict, fn_frac: float) -> float:
    pts = results_by_fn.get(fn_frac, [])
    vals = [r['T_metal'] for r in pts if np.isfinite(r.get('T_metal', np.nan))]
    return float(np.mean(vals)) if vals else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  BLADE TEMPERATURE MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _add_blade_temp(results_by_fn: dict, frac_W_cool: float):
    """
    Annotate each result dict in-place with T_metal (K).
    T_metal = Tt4 - eta_cool * frac_W * (Tt4 - T3)
    frac_W is the design cooling fraction — fixed hardware.
    """
    for fn_frac, pts in results_by_fn.items():
        for r in pts:
            Tt4_K = r['T4'] * RANKINE_TO_K
            T3_K  = r['T3'] * RANKINE_TO_K
            r['T_metal'] = Tt4_K - ETA_COOL * frac_W_cool * (Tt4_K - T3_K)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE (OPR, Tt4) CELL SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_cell(opr: float, tt4_K: float) -> dict | None:
    """
    Run fixed and VGT physics-dP/P sweeps at one (OPR, Tt4) point.

    Option B — independent combustor sizing per engine:
      Each engine (fixed and VGT) is treated as a clean-sheet design at this
      (OPR, Tt4) point.  CdA is anchored separately for each from its own
      100% Fn, A4=1.0 operating condition, ensuring dP/P = DPQP_FIXED at
      that reference point for both.  This is appropriate for a parametric
      cycle space study where every cell is a notionally different engine.

    Returns a dict with keys 'fixed' and 'vgt', each containing:
        des        — design-point extract dict
        frac_W     — cooling fraction
        phys_dpqp  — {fn_frac: [result_dicts]} from physics dP/P sweep
    Returns None if either engine failed to converge.
    """
    tt4_R  = tt4_K * 9.0 / 5.0
    Np_des = _Nmech / np.sqrt(tt4_R)

    # Per-map efficiency at this corrected speed — separate lookup per map
    eff_fixed, PR_fixed = map_design_point(EngineHPTFixedMap, Np_des)
    eff_vgt,   PR_vgt   = map_design_point(EngineHPTVGTMap,   Np_des)

    cfg_fixed = {
        **BASE_CFG,
        'turb_map'     : EngineHPTFixedMap,
        'turb_eff'     : eff_fixed,
        'turb_PR_guess': PR_fixed,
        'OPR'          : opr,
        'T4_design'    : tt4_R,
        'a4_scalars'   : A4_SCALARS_FIXED,
    }
    cfg_vgt = {
        **BASE_CFG,
        'turb_map'     : EngineHPTVGTMap,
        'turb_eff'     : eff_vgt,
        'turb_PR_guess': PR_vgt,
        'OPR'          : opr,
        'T4_design'    : tt4_R,
        'a4_scalars'   : A4_SCALARS_VGT,
    }

    cell_out = {}

    for label, cfg, mode in [('fixed', cfg_fixed, 'sWp'),
                              ('vgt',   cfg_vgt,   'alpha')]:
        try:
            prob, mp = _build_problem(cfg, mode)
            prob.run_model()
            des = extract_design(prob)

            # ── Cooling fraction from blade temperature constraint ──────
            Tt4_des_K = des['T4_design'] * RANKINE_TO_K
            T3_des_K  = des['T3_design'] * RANKINE_TO_K
            frac_W = float(np.clip(
                (Tt4_des_K - T_METAL_MAX) / (ETA_COOL * (Tt4_des_K - T3_des_K)),
                0.0, 0.30))
            cfg['cool_frac'] = frac_W

            # ── Initial CdA from design point ──────────────────────────
            des['CdA_liner'] = calc_CdA_liner(
                T3_des_K,
                psi_to_pa(des['P3_design']),
                lbm_s_to_kg_s(des['W']),
            )

            # ── Fixed dP/P sweep — used solely to re-anchor CdA ───────
            # Runs at DPQP_FIXED for all points; anchor picks 100% Fn,
            # A4 closest to 1.0 so both engines start from dP/P=5% at
            # their own 100% Fn, A4=1.0 reference condition.
            fixed_dpqp = {}
            for fn_frac in FN_FRACS:
                fixed_dpqp[fn_frac] = run_fixed_dPqP_sweep(
                    prob, mp, des, fn_frac, cfg, mode)

            if 1.00 in fixed_dpqp:
                anc = min(fixed_dpqp[1.00],
                          key=lambda r: abs(r['a4_scalar'] - 1.0))
                CdA = calc_CdA_liner(
                    rankine_to_kelvin(anc['T3']),
                    psi_to_pa(anc['P3']),
                    lbm_s_to_kg_s(anc['W']))
                des['CdA_liner'] = CdA
                print(f"    [{label}] CdA = {CdA*1e4:.4f} cm²")

            # ── Physics dP/P sweep ─────────────────────────────────────
            phys_dpqp = {}
            for fn_frac in FN_FRACS:
                phys_dpqp[fn_frac] = run_physics_dPqP_sweep(
                    prob, mp, des, fn_frac, cfg, mode)

            # ── Blade temperature annotation ───────────────────────────
            _add_blade_temp(phys_dpqp, frac_W)

            cell_out[label] = {
                'des':       des,
                'frac_W':    frac_W,
                'phys_dpqp': phys_dpqp,
            }

        except Exception as exc:
            print(f"    !! FAILED [{label}] OPR={opr:.1f} Tt4={tt4_K:.0f}K: {exc}")
            cell_out[label] = None

    if cell_out.get('fixed') and cell_out.get('vgt'):
        return cell_out
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SWEEP LOOP
# ══════════════════════════════════════════════════════════════════════════════

n_opr = len(OPR_vals)
n_tt4 = len(Tt4_vals)
n_fn  = len(FN_FRACS)

# Arrays filled during sweep; NaN = failed cell
delta_eta    = np.full((n_opr, n_tt4, n_fn), np.nan)
delta_T_metal = np.full((n_opr, n_tt4, n_fn), np.nan)
cool_frac_arr = np.full((n_opr, n_tt4),        np.nan)

total = n_opr * n_tt4
done  = 0

for i, opr in enumerate(OPR_vals):
    for j, tt4_K in enumerate(Tt4_vals):
        done += 1
        print(f"\n{'='*60}")
        print(f"  [{done}/{total}]  OPR={opr:.1f}  Tt4={tt4_K:.0f}K"
              f"  ({tt4_K*9/5:.0f}°R)")
        print(f"{'='*60}")

        cell = solve_cell(opr, tt4_K)
        if cell is None:
            print(f"  → cell skipped (NaN stored)")
            continue

        cool_frac_arr[i, j] = cell['fixed']['frac_W']

        for k, fn_frac in enumerate(FN_FRACS):
            eta_f = _mean_eta(cell['fixed']['phys_dpqp'], fn_frac)
            eta_v = _best_eta(cell['vgt']['phys_dpqp'],   fn_frac)

            if np.isfinite(eta_f) and np.isfinite(eta_v):
                delta_eta[i, j, k] = (eta_v - eta_f) / eta_f * 100.0

            Tm_f = _mean_T_metal(cell['fixed']['phys_dpqp'], fn_frac)
            Tm_v = _best_T_metal(cell['vgt']['phys_dpqp'],   fn_frac)

            if np.isfinite(Tm_f) and np.isfinite(Tm_v):
                delta_T_metal[i, j, k] = Tm_v - Tm_f

            print(f"  Fn={fn_frac*100:.0f}%  "
                  f"η_fixed={eta_f*100:.3f}%  η_vgt={eta_v*100:.3f}%  "
                  f"Δη={delta_eta[i,j,k]:+.3f}%  "
                  f"ΔT_metal={delta_T_metal[i,j,k]:+.1f}K")

# ══════════════════════════════════════════════════════════════════════════════
#  SAVE RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

np.save('delta_eta_opr_tt4.npy',    delta_eta)
np.save('delta_T_metal_opr_tt4.npy', delta_T_metal)
np.save('cool_frac_opr_tt4.npy',    cool_frac_arr)
np.save('OPR_vals.npy',             OPR_vals)
np.save('Tt4_vals.npy',             Tt4_vals)
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


# ── Figure 1: Δη_th contour (one panel per Fn level) ─────────────────────────
fig1, axes1 = plt.subplots(1, n_fn,
                            figsize=(7 * n_fn, 6),
                            sharey=True)
if n_fn == 1:
    axes1 = [axes1]

fig1.suptitle(
    'VGT vs Fixed Turbine — Δη_th contour\n'
    'Physics dP/P  |  Optimal A4 for VGT  |  Red = VGT better',
    fontsize=11, fontweight='bold')

for k, (ax, fn_frac) in enumerate(zip(axes1, FN_FRACS)):
    cf = _contour_panel(
        ax, delta_eta[:, :, k],
        title=f'Δη_th (%) — {fn_frac*100:.0f}% Fn',
        cbar_label='Δη_th (%)',
        cmap='RdBu', symmetric=True)
    if cf is not None:
        plt.colorbar(cf, ax=ax, label='Δη_th (%) VGT − Fixed',
                     shrink=0.85, pad=0.03)

fig1.tight_layout()
fig1.savefig('vgt_benefit_contour.png', dpi=150, bbox_inches='tight')
print('  Saved: vgt_benefit_contour.png')


# ── Figure 2: ΔT_metal contour ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, n_fn,
                            figsize=(7 * n_fn, 6),
                            sharey=True)
if n_fn == 1:
    axes2 = [axes2]

fig2.suptitle(
    'VGT vs Fixed — ΔT_metal contour  (VGT minus Fixed at opt A4)\n'
    'Negative = VGT blade runs cooler',
    fontsize=11, fontweight='bold')

for k, (ax, fn_frac) in enumerate(zip(axes2, FN_FRACS)):
    cf = _contour_panel(
        ax, delta_T_metal[:, :, k],
        title=f'ΔT_metal (K) — {fn_frac*100:.0f}% Fn',
        cbar_label='ΔT_metal (K)',
        cmap='RdBu_r', symmetric=True)
    if cf is not None:
        plt.colorbar(cf, ax=ax, label='ΔT_metal (K) VGT − Fixed',
                     shrink=0.85, pad=0.03)

fig2.tight_layout()
fig2.savefig('vgt_T_metal_contour.png', dpi=150, bbox_inches='tight')
print('  Saved: vgt_T_metal_contour.png')


# ── Figure 3: cooling fraction contour ───────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 5))
Zm_cool = np.ma.masked_invalid(cool_frac_arr * 100)
if Zm_cool.count() > 0:
    cf3 = ax3.contourf(OPR_grid, Tt4_grid, Zm_cool,
                       levels=15, cmap='YlOrRd')
    plt.colorbar(cf3, ax=ax3, label='Cooling fraction (%)')
    ax3.contour(OPR_grid, Tt4_grid, Zm_cool,
                levels=[10, 15, 20], colors='k',
                linewidths=1.2, linestyles='--')
    ax3.clabel(
        ax3.contour(OPR_grid, Tt4_grid, Zm_cool,
                    levels=[10, 15, 20], colors='k',
                    linewidths=0, linestyles='--'),
        fmt='%.0f%%', fontsize=8)
ax3.plot(OPR_DESIGN, Tt4_DESIGN_K, '*',
         ms=14, color='gold', markeredgecolor='k',
         zorder=6, label='Design point')
ax3.set_xlabel('OPR', fontsize=10)
ax3.set_ylabel('Tt4 (K)', fontsize=10)
ax3.set_title('Cooling Fraction (%) — blade limit constraint\n'
              f'η_cool={ETA_COOL:.2f}  T_metal_max={T_METAL_MAX:.0f}K',
              fontsize=10, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(100))
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
    print(f"  Δη@{fn_frac*100:.0f}% Fn", end='')
print()
print('-' * 70)
for i, opr in enumerate(OPR_vals):
    for j, tt4_K in enumerate(Tt4_vals):
        print(f"  {opr:6.1f}  {tt4_K:8.0f}", end='')
        for k in range(n_fn):
            v = delta_eta[i, j, k]
            print(f"  {v:+10.3f}%" if np.isfinite(v) else f"  {'NaN':>10}", end='')
        print()