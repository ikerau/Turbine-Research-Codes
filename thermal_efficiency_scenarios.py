"""
thermal_efficiency_scenarios.py  —  turboshaft A4 sweep
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

η_th, PSFC, T₄, and T_metal vs A₄ scalar for a simple turboshaft cycle.
Two scenarios (Fixed map vs VGT map) are solved across a grid of
(power fraction × A4 scalar).  Brent optimisation is NOT used — the full
efficiency landscape is exposed so the optimal schedule can be read off directly.

Warm-start strategy (mirrors combustor_turboshaft.py):
  Power axis  — high→low; each level seeds the next.
  A4 axis     — sorted by |a4-1.0| (design anchor first); walks outward.
  Cross-axis  — prev_at_a4[a4] carries the converged result from the
                previous power level at the same a4, giving a 2-D warm grid.

Outputs
-------
  ts_eta_th_vs_a4.png
  ts_psfc_vs_a4.png
  ts_temperatures_vs_a4.png
  ts_a4_schedule.png
"""

import time
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import openmdao.api as om
import pycycle.api as pyc

from EngineHPT_map       import EngineHPTMap as EngineHPTVGTMap
from EngineHPT_map_fixed import EngineHPTMap as EngineHPTFixedMap

from combustor_turboshaft import (
    _make_mp_turboshaft,
    solve_od, extract_design, compute_CdA,
    gauntner_cooling_fracs, turb_rel_temp_rankine, t_metal_rankine,
    estimate_T3_rankine, map_design_point, _std_atm,
    nan_result, draw_compressor_map_background,
    DPQP_DESIGN, RANKINE_TO_K, LBM_S_TO_KG_S, HP_TO_KW, PSI_TO_PA,
    ETA_COOL, T_METAL_VANE_R, T_METAL_BLADE_R,
)
from combustor_py_cycle import lmp_life_ratio

if os.name == 'nt':
    os.system('cls')

# ══════════════════════════════════════════════════════════════════════════════
#  STUDY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

_TIT_R_des  = 2370.0         # design turbine inlet temperature (°R)
_HP_N_des   = 8070.0         # design HP shaft speed (rpm)
_OPR_des    = 13.5           # overall pressure ratio
_comp_eff   = 0.83
_Np_des     = _HP_N_des / np.sqrt(_TIT_R_des)

PWR_FRACS  = list(np.linspace(0.60, 1.00, 15))    # fraction of design SHP
A4_SCALARS = list(np.linspace(0.85, 1.00, 30))   # NGV throat area scalars (closing only)

# Newton convergence controls — tighten for publication, loosen for quick runs
OD_ATOL    = 1e-8   # absolute residual tolerance  (setup default: 1e-6)
OD_RTOL    = 1e-8   # relative residual tolerance  (setup default: 1e-6)
OD_MAXITER = 100     # max Newton iterations         (solve_od default: 30)

_T_METAL_REF_K = T_METAL_BLADE_R * RANKINE_TO_K  # LMP reference (blade allowable)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def _annotate(r):
    """Add LMP life ratio and Gauntner prescription fracs in-place.
    T_metal_blade is already in the result from extract_od()."""
    if not r.get('converged'):
        r.update(life_ratio=np.nan, req_frac_ngv=np.nan, req_frac_blade=np.nan)
        return
    r['life_ratio'] = lmp_life_ratio(_T_METAL_REF_K, r['T_metal_blade'])
    f_ngv, f_bld = gauntner_cooling_fracs(r['T4'], r['T3'], T4_exit_R=r['T4_exit'])
    r['req_frac_ngv']   = f_ngv
    r['req_frac_blade'] = f_bld


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(label, cfg, pwr_fracs, a4_scalars):
    """
    Build turboshaft MPCycle, solve design point, then run the 2-D
    (pwr_frac × a4_scalar) sweep via solve_od().

    Returns
    -------
    dict with keys:
      'label'   : str
      'des'     : design-point result dict (from extract_design)
      'results' : nested dict  results[pwr_frac][a4_scalar]
    """
    print(f"\n{'█'*70}")
    print(f"  {label}")
    print(f"{'█'*70}")

    prob = om.Problem()
    prob.model = _make_mp_turboshaft(cfg)()
    prob.setup(check=False)
    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    Pt_des, Tt_des = _std_atm(cfg['alt'])

    # ── Design solve ──────────────────────────────────────────────────────────
    prob.set_val('DESIGN.fc.alt',                cfg['alt'],           units='ft')
    prob.set_val('DESIGN.fc.MN',                 cfg['MN'])
    prob.set_val('DESIGN.balance.T4_target',      cfg['TIT_R'],         units='degR')
    prob.set_val('DESIGN.balance.pwr_target',     cfg['SHP'],           units='hp')
    prob.set_val('DESIGN.balance.nozz_PR_target', cfg['nozz_PR_target'])
    prob.set_val('DESIGN.comp.PR',                cfg['OPR'])
    prob.set_val('DESIGN.comp.eff',               cfg['comp_eff'])
    prob.set_val('DESIGN.turb.eff',               cfg['turb_eff'])
    prob.set_val('DESIGN.pt.eff',                 cfg['pt_eff'])
    prob.set_val('DESIGN.burner.dPqP',            cfg['dPqP'])

    prob['DESIGN.balance.FAR']     = 0.0175
    prob['DESIGN.balance.W']       = 27.0
    prob['DESIGN.balance.turb_PR'] = 3.87
    prob['DESIGN.balance.pt_PR']   = 2.0
    prob['DESIGN.fc.balance.Pt']   = Pt_des
    prob['DESIGN.fc.balance.Tt']   = Tt_des

    # Seed CoolingCalcs from Gauntner pre-estimate
    prob.set_val('DESIGN.cool_fracs.frac_ngv', cfg['ngv_cool_frac'])
    prob.set_val('DESIGN.cool_fracs.frac_bld', cfg['bld_cool_frac'])

    prob.model._get_subsystem('SWEEP').nonlinear_solver.options['maxiter'] = 0
    prob.model._get_subsystem('DESIGN').nonlinear_solver.options['maxiter'] = 30

    print('\n── Design point ──')
    t0 = time.time()
    prob.run_model()
    print(f'  Solved in {time.time()-t0:.1f}s')

    des = extract_design(prob)
    des['CdA_liner']     = compute_CdA(des)
    des['ngv_cool_frac'] = float(prob.get_val('DESIGN.cool_fracs.frac_ngv')[0])
    des['bld_cool_frac'] = float(prob.get_val('DESIGN.cool_fracs.frac_bld')[0])

    print(f"  OPR={des['OPR']:.3f}  Tt4={des['T4']*RANKINE_TO_K:.1f}K  "
          f"SHP={des['SHP']:.0f}hp  PSFC={des['PSFC']:.5f}")
    print(f"  GGT_PR={des['turb_PR']:.3f}  PT_PR={des['pt_PR']:.3f}  "
          f"turb_s_Wp={des['turb_s_Wp']:.6f}")
    print(f"  Cool NGV={des['ngv_cool_frac']*100:.2f}%  "
          f"blade={des['bld_cool_frac']*100:.2f}%  "
          f"CdA={des['CdA_liner']:.4e} m²")

    # Freeze DESIGN; restore SWEEP solver for OD runs
    prob.model._get_subsystem('DESIGN').nonlinear_solver.options['maxiter'] = 0
    prob.model._get_subsystem('SWEEP').nonlinear_solver.options['maxiter'] = OD_MAXITER

    # ── Prime SWEEP at 100% A4=1.0 to put prob in a physical OD state ────────
    print('\n── Priming SWEEP at 100% power, A4=1.0 ──')
    _prime = solve_od(prob, des,
                      turb_s_Wp=des['turb_s_Wp'],
                      pt_s_Wp=des['pt_s_Wp'],
                      pwr_hp=des['SHP'],
                      warm=des,
                      label='prime',
                      atol=OD_ATOL, rtol=OD_RTOL, maxiter=OD_MAXITER)
    print(f"  {'OK' if _prime['converged'] else 'FAILED'}: "
          f"η_th={_prime.get('eta_th', np.nan)*100:.3f}%  "
          f"Tt4={_prime.get('T4', np.nan)*RANKINE_TO_K:.1f}K")

    # ── 2-D sweep: outer=A4 (1.0→0.85), inner=power (100%→60%) ─────────────────
    _a4_sorted  = sorted(a4_scalars, reverse=True)   # 1.0 → 0.85
    _pf_sorted  = sorted(pwr_fracs,  reverse=True)   # 100% → 60%
    results     = {pf: {} for pf in pwr_fracs}

    for a4 in _a4_sorted:
        print(f"\n  ── A4 = {a4:.3f} ──")
        prev = None   # None → solve_od cold-starts from des (correct at 100%)

        for pf in _pf_sorted:
            pwr_hp = pf * des['SHP']

            r = solve_od(prob, des,
                         turb_s_Wp=des['turb_s_Wp'] * a4,
                         pt_s_Wp=des['pt_s_Wp'],
                         pwr_hp=pwr_hp,
                         warm=prev,
                         label=f'a4={a4:.3f} {pf*100:.0f}%',
                         atol=OD_ATOL, rtol=OD_RTOL, maxiter=OD_MAXITER)

            _annotate(r)
            r['a4_scalar'] = a4
            r['pwr_frac']  = pf
            results[pf][a4] = r

            if r['converged']:
                prev = r
            else:
                prev = None   # force cold-start on next power step

            ok = 'OK' if r.get('converged') else 'FAIL'
            print(f"    {pf*100:.0f}%  {ok}"
                  f"  η_th={r.get('eta_th', np.nan)*100:.3f}%"
                  f"  T4={r.get('T4', np.nan)*RANKINE_TO_K:.1f}K"
                  f"  Tmbl={r.get('T_metal_blade', np.nan):.0f}K"
                  f"  PSFC={r.get('PSFC', np.nan):.5f}")

    return {'label': label, 'des': des, 'results': results, 'prob': prob}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _arr(results, pf, a4s, key):
    return np.array([results[pf][a].get(key, np.nan) for a in a4s])


def _opt_a4_idx(results, pf, a4_scalars, key='eta_th', minimize=False):
    """Index of A4 that maximises (or minimises) the given key."""
    vals = _arr(results, pf, a4_scalars, key)
    if not np.any(np.isfinite(vals)):
        return None
    return int(np.nanargmin(vals) if minimize else np.nanargmax(vals))


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_comp_map(scenarios, pwr_fracs, a4_scalars, out_dir=''):
    """
    Compressor map with operating clouds for each scenario.
    Each line = one power level; dots walk along A4 from 1.0 (left) to 0.85 (right).
    """
    pct  = np.array(pwr_fracs) * 100
    cmap = cm.viridis
    cv   = np.linspace(0.15, 0.90, len(pwr_fracs))
    sm   = plt.cm.ScalarMappable(cmap=cmap,
                                  norm=plt.Normalize(pct.min(), pct.max()))
    sm.set_array([])

    ncols = len(scenarios)
    fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 7), sharey=True)
    if ncols == 1:
        axes = [axes]
    fig.suptitle('AXI5 Compressor Map  —  Operating clouds (A₄ = 1.0 → 0.85)\n'
                 'Each line = power level; colour = power fraction',
                 fontsize=12, fontweight='bold')

    for col, sc in enumerate(scenarios):
        ax   = axes[col]
        prob = sc['prob']
        des  = sc['des']
        res  = sc['results']

        # Background: efficiency contours, Nc lines, R-lines, stall line
        eff_cf = draw_compressor_map_background(
            ax, prob, 'DESIGN.comp', alpha_idx=0,
            eff_vals=np.array([0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]),
            show_rlines=False, show_nclines=True, eff_alpha=0.75)

        # Operating lines: one per power fraction, sweep over A4
        for i, pf in enumerate(pwr_fracs):
            wc = np.array([res[pf][a].get('comp_Wc', np.nan) for a in a4_scalars])
            pr = np.array([res[pf][a].get('comp_PR', np.nan) for a in a4_scalars])
            mask = np.isfinite(wc) & np.isfinite(pr)
            if mask.any():
                ax.plot(wc[mask], pr[mask], '-o', color=cmap(cv[i]),
                        lw=1.8, ms=4, label=f'{pf*100:.0f}%', zorder=8)
                # Mark the design-A4 (a4=1.0) point with a larger marker
                a1_idx = a4_scalars.index(max(a4_scalars))
                if mask[a1_idx]:
                    ax.plot(wc[a1_idx], pr[a1_idx], 'o',
                            color=cmap(cv[i]), ms=9,
                            markeredgecolor='k', markeredgewidth=0.7, zorder=9)

        # Design point star
        ax.plot(des['comp_Wc'], des['comp_PR'],
                '*', ms=18, color='gold', markeredgecolor='k',
                markeredgewidth=1.2, zorder=10, label='Design pt')

        ax.set_xlabel('Corrected mass flow  Wc  (lbm/s)', fontsize=11)
        if col == 0:
            ax.set_ylabel('Compressor PR  πc', fontsize=11)
        ax.set_title(sc['label'], fontsize=10, fontweight='bold')
        ax.legend(fontsize=7, title='Power %', loc='upper left')
        ax.grid(True, alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.colorbar(eff_cf, ax=ax, shrink=0.65, label='Comp. efficiency')

    fig.colorbar(sm, ax=axes[-1], shrink=0.40, pad=0.12,
                 label='Power fraction (%)')
    fig.tight_layout()
    fp = os.path.join(out_dir, 'ts_compressor_map.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fp}')


def _ax_style(ax, col, ylabel, scenario_label):
    ax.axvline(1.0, color='gray', ls='--', lw=1.0, alpha=0.55)
    ax.set_xlabel('A₄ scalar', fontsize=10)
    if col == 0:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(scenario_label, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_scenarios(scenarios, pwr_fracs, a4_scalars, out_dir=''):
    a4s  = np.array(a4_scalars)
    pct  = np.array(pwr_fracs) * 100
    cmap = cm.viridis
    cv   = np.linspace(0.15, 0.90, len(pwr_fracs))

    ncols = len(scenarios)
    sm    = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(pct.min(), pct.max()))
    sm.set_array([])

    SC = ['steelblue', 'firebrick', 'seagreen', 'darkorange']

    # ── Figure 1: η_th vs A4 ─────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(1, ncols, figsize=(8*ncols, 6), sharey=False)
    if ncols == 1:
        axes1 = [axes1]
    fig1.suptitle('Turboshaft Thermal Efficiency vs A₄ Scalar\n'
                  'Dynamic Gauntner cooling · physics combustor dP/P',
                  fontsize=12, fontweight='bold')

    for col, sc in enumerate(scenarios):
        ax  = axes1[col]
        res = sc['results']
        for i, pf in enumerate(pwr_fracs):
            eta = _arr(res, pf, a4_scalars, 'eta_th') * 100
            ax.plot(a4s, eta, '-o', color=cmap(cv[i]), lw=2.0, ms=4,
                    label=f'{pf*100:.0f}%')
            idx = _opt_a4_idx(res, pf, a4_scalars, 'eta_th')
            if idx is not None:
                ax.plot(a4s[idx], eta[idx], '*',
                        color=cmap(cv[i]), ms=12,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=5)
        _ax_style(ax, col, 'η_th  (%)', sc['label'])
        ax.legend(fontsize=7, title='Power %', loc='best')

    fig1.colorbar(sm, ax=axes1[-1], shrink=0.60, label='Power fraction (%)')
    fig1.tight_layout()
    fp = os.path.join(out_dir, 'ts_eta_th_vs_a4.png')
    fig1.savefig(fp, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fp}')

    # ── Figure 2: PSFC vs A4 ─────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, ncols, figsize=(8*ncols, 6), sharey=False)
    if ncols == 1:
        axes2 = [axes2]
    fig2.suptitle('Turboshaft PSFC vs A₄ Scalar', fontsize=12, fontweight='bold')

    for col, sc in enumerate(scenarios):
        ax  = axes2[col]
        res = sc['results']
        for i, pf in enumerate(pwr_fracs):
            psfc = _arr(res, pf, a4_scalars, 'PSFC')
            ax.plot(a4s, psfc, '-o', color=cmap(cv[i]), lw=2.0, ms=4,
                    label=f'{pf*100:.0f}%')
            idx = _opt_a4_idx(res, pf, a4_scalars, 'PSFC', minimize=True)
            if idx is not None:
                ax.plot(a4s[idx], psfc[idx], '*',
                        color=cmap(cv[i]), ms=12,
                        markeredgecolor='k', markeredgewidth=0.5, zorder=5)
        _ax_style(ax, col, 'PSFC  (lbm/hp/hr)', sc['label'])
        ax.legend(fontsize=7, title='Power %', loc='best')

    fig2.colorbar(sm, ax=axes2[-1], shrink=0.60, label='Power fraction (%)')
    fig2.tight_layout()
    fp = os.path.join(out_dir, 'ts_psfc_vs_a4.png')
    fig2.savefig(fp, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fp}')

    # ── Figure 3: T₄ and T_metal_blade vs A4 ─────────────────────────────────
    fig3, axes3 = plt.subplots(2, ncols, figsize=(8*ncols, 10))
    if ncols == 1:
        axes3 = axes3.reshape(2, 1)
    fig3.suptitle('Turboshaft Tt₄ and Blade Metal Temperature vs A₄ Scalar',
                  fontsize=12, fontweight='bold')

    for col, sc in enumerate(scenarios):
        ax_t4 = axes3[0, col]
        ax_tm = axes3[1, col]
        res   = sc['results']

        for i, pf in enumerate(pwr_fracs):
            T4 = _arr(res, pf, a4_scalars, 'T4') * RANKINE_TO_K
            Tm = _arr(res, pf, a4_scalars, 'T_metal_blade')
            c  = cmap(cv[i])
            ax_t4.plot(a4s, T4, '-o', color=c, lw=2.0, ms=4, label=f'{pf*100:.0f}%')
            ax_tm.plot(a4s, Tm, '-o', color=c, lw=2.0, ms=4, label=f'{pf*100:.0f}%')

        T4_des = sc['des']['T4'] * RANKINE_TO_K
        ax_t4.axhline(T4_des, color='crimson', ls='-', lw=1.5,
                      label=f'T₄ design = {T4_des:.0f} K')
        ax_tm.axhline(_T_METAL_REF_K, color='k', ls=':', lw=1.5,
                      label=f'Allowable {_T_METAL_REF_K:.0f} K')

        _ax_style(ax_t4, col, 'Tt₄  (K)', sc['label'])
        _ax_style(ax_tm, col, 'T_metal_blade  (K)', sc['label'])
        ax_t4.legend(fontsize=7, title='Power %')
        ax_tm.legend(fontsize=7, title='Power %')

    fig3.tight_layout()
    fp = os.path.join(out_dir, 'ts_temperatures_vs_a4.png')
    fig3.savefig(fp, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fp}')

    # ── Figure 4: Optimal A4 schedule + Δη_th between scenarios ──────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
    fig4.suptitle('Optimal A₄ Schedule and η_th Benefit  (turboshaft)',
                  fontsize=12, fontweight='bold')

    ax = axes4[0]
    for j, sc in enumerate(scenarios):
        opt = []
        for pf in pwr_fracs:
            idx = _opt_a4_idx(sc['results'], pf, a4_scalars, 'eta_th')
            opt.append(a4_scalars[idx] if idx is not None else np.nan)
        ax.plot(pct, opt, 'o-', color=SC[j % len(SC)], lw=2.2, ms=8,
                label=sc['label'])
    ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='Design A₄ = 1.0')
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('Optimal A₄ scalar', fontsize=11)
    ax.set_title('Optimal A₄ per Power Level  (max η_th)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes4[1]
    if len(scenarios) >= 2:
        sc1, sc2 = scenarios[0], scenarios[1]
        for i, pf in enumerate(pwr_fracs):
            eta1 = _arr(sc1['results'], pf, a4_scalars, 'eta_th') * 100
            eta2 = _arr(sc2['results'], pf, a4_scalars, 'eta_th') * 100
            ax.plot(a4s, eta2 - eta1, '-', color=cmap(cv[i]), lw=2.0,
                    label=f'{pf*100:.0f}%')
        ax.axhline(0.0, color='k', ls='--', lw=1.2)
        ax.axvline(1.0, color='gray', ls='--', lw=1.0, alpha=0.55)
        ax.set_xlabel('A₄ scalar', fontsize=11)
        ax.set_ylabel(f'Δη_th  (%-pts)  [{sc2["label"]} − {sc1["label"]}]',
                      fontsize=9)
        ax.set_title('η_th Difference  (Scenario 2 − Scenario 1)', fontsize=10)
        ax.legend(fontsize=7, title='Power %')
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig4.colorbar(sm, ax=ax, shrink=0.60, label='Power fraction (%)')

    fig4.tight_layout()
    fp = os.path.join(out_dir, 'ts_a4_schedule.png')
    fig4.savefig(fp, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fp}')

    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    eff_vgt,   PR_vgt   = map_design_point(EngineHPTVGTMap,   _Np_des)
    eff_fixed, PR_fixed = map_design_point(EngineHPTFixedMap, _Np_des)

    print('=' * 70)
    print(f'  VGT   map: η={eff_vgt*100:.3f}%   PR={PR_vgt:.4f}')
    print(f'  Fixed map: η={eff_fixed*100:.3f}%   PR={PR_fixed:.4f}')
    print(f'  Leakage penalty: Δη={(eff_fixed-eff_vgt)*100:+.3f}%')
    print('=' * 70)

    _T3_est_R        = estimate_T3_rankine(_OPR_des, comp_eff=_comp_eff)
    _frac_ngv_des, _frac_blade_des = gauntner_cooling_fracs(_TIT_R_des, _T3_est_R)

    print(f'  Gauntner pre-estimate: T4={_TIT_R_des:.0f}°R  T3≈{_T3_est_R:.1f}°R')
    print(f'  NGV={_frac_ngv_des*100:.2f}%  blade={_frac_blade_des*100:.2f}%')
    print(f'  A₄ scalars: {A4_SCALARS[0]:.2f} → {A4_SCALARS[-1]:.2f}'
          f'  ({len(A4_SCALARS)} pts)')
    print(f'  Power levels: {[f"{p*100:.0f}%" for p in PWR_FRACS]}')
    print('=' * 70)

    BASE_CFG = dict(
        alt           = 0.0,
        MN            = 0.001,
        SHP           = 4000.0,
        LP_Nmech      = 5000.0,
        HP_Nmech_init = _HP_N_des,
        TIT_R         = _TIT_R_des,
        OPR           = _OPR_des,
        comp_eff      = _comp_eff,
        pt_eff        = 0.90,
        inlet_MN      = 0.60,
        comp_MN       = 0.20,
        burner_MN     = 0.20,
        turb_MN       = 0.40,
        nozz_Cv       = 0.99,
        nozz_PR_target= 1.2,
        dPqP          = DPQP_DESIGN,
        ngv_cool_frac = _frac_ngv_des,
        bld_cool_frac = _frac_blade_des,
        comp_map      = pyc.AXI5,
        pt_map        = pyc.LPT2269,
    )

    # ── Scenario 1: Fixed turbine map ─────────────────────────────────────────
    scen_fixed = run_scenario(
        'Fixed Turbine (EngineHPT fixed)',
        {**BASE_CFG, 'turb_map': EngineHPTFixedMap, 'turb_eff': eff_fixed},
        PWR_FRACS, A4_SCALARS,
    )

    # ── Scenario 2: VGT turbine map ───────────────────────────────────────────
    scen_vgt = run_scenario(
        'VGT Turbine (EngineHPT VGT)',
        {**BASE_CFG, 'turb_map': EngineHPTVGTMap, 'turb_eff': eff_vgt},
        PWR_FRACS, A4_SCALARS,
    )

    scenarios = [scen_fixed, scen_vgt]

    # ── Summary table ─────────────────────────────────────────────────────────
    print('\n' + '=' * 88)
    print('  OPTIMAL A4 SCHEDULE  (A4 that maximises η_th at each power level)')
    print('=' * 88)
    print(f"  {'Pwr%':>5}  "
          f"{'A4_fix':>7}  {'η_fix%':>8}  {'PSFC_fix':>9}  "
          f"{'A4_vgt':>7}  {'η_vgt%':>8}  {'PSFC_vgt':>9}  "
          f"{'Δη%-pts':>8}")
    print('  ' + '-' * 75)
    for pf in PWR_FRACS:
        a4s = A4_SCALARS
        eta_f  = [scen_fixed['results'][pf][a].get('eta_th', np.nan) for a in a4s]
        eta_v  = [scen_vgt['results'][pf][a].get('eta_th',  np.nan) for a in a4s]
        psfc_f = [scen_fixed['results'][pf][a].get('PSFC',  np.nan) for a in a4s]
        psfc_v = [scen_vgt['results'][pf][a].get('PSFC',   np.nan) for a in a4s]
        idx_f  = int(np.nanargmax(eta_f))  if any(np.isfinite(e) for e in eta_f)  else 0
        idx_v  = int(np.nanargmax(eta_v))  if any(np.isfinite(e) for e in eta_v)  else 0
        d_eta  = (eta_v[idx_v] - eta_f[idx_f]) * 100
        print(f"  {pf*100:5.0f}%  "
              f"{a4s[idx_f]:7.3f}  {eta_f[idx_f]*100:8.4f}  {psfc_f[idx_f]:9.5f}  "
              f"{a4s[idx_v]:7.3f}  {eta_v[idx_v]*100:8.4f}  {psfc_v[idx_v]:9.5f}  "
              f"{d_eta:+8.4f}")
    print('  ' + '-' * 75)

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_comp_map(scenarios, PWR_FRACS, A4_SCALARS)
    plot_scenarios(scenarios, PWR_FRACS, A4_SCALARS)

    print('\n' + '=' * 70)
    print('  All scenarios complete.')
    print('=' * 70)
