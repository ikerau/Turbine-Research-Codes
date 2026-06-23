"""
thermal_efficiency_scenarios.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Three scenarios comparing thermal efficiency vs A₄ scalar:
  1. Zero cooling, fixed turbine
  2. Gauntner-sized cooling, VGT turbine
  3. Zero cooling, VGT turbine

Outputs
-------
  thermal_efficiency_scenarios.pkl
  thermal_efficiency_scenarios.png  — η_th vs A₄
  gauntner_cooling_prescription.png — cooling fraction vs A₄
  blade_life_scenarios.png          — T_metal and LMP life ratio vs A₄

Shared physics (combustor model, cooling sizing, extraction, LMP) is in
combustor_py_cycle.py and imported here to avoid duplication.  The local
cycle model uses fixed cooling fracs (cycle params) so Scenarios 1 and 3
can set them to zero — the dynamic CoolingCalcs model in combustor_py_cycle
cannot represent zero-cooling conditions.
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pint
import pickle
import os

import openmdao.api as om
import pycycle.api as pyc

from EngineHPT_map       import EngineHPTMap as EngineHPTVGTMap
from EngineHPT_map_fixed import EngineHPTMap as EngineHPTFixedMap

# ── Shared utilities from combustor_py_cycle ─────────────────────────────────
from combustor_py_cycle import (
    # constants
    DPQP_FIXED, DPQP_TOL, MAX_OUTER,
    RANKINE_TO_K,
    ETA_COOL, T_METAL_VANE_R, T_METAL_BLADE_R,
    # unit helpers
    rankine_to_kelvin, psi_to_pa, lbm_s_to_kg_s,
    # combustor physics
    physics_pib, calc_CdA_liner,
    # cycle helpers — extract_od_point includes T4_exit
    map_design_point, extract_eta_th,
    extract_design, extract_od_point,
    set_turb_area, _connect_od_scalars, _run_od_sweep_sequential,
    # cooling & life
    estimate_T3_rankine, gauntner_cooling_fracs, turb_rel_temp_rankine,
    lmp_life_ratio,
)


def clear_terminal():
    if os.name == 'nt':
        os.system('cls')


clear_terminal()
ureg = pint.UnitRegistry()
Q_   = ureg.Quantity

# Reference for LMP: Gauntner rotor blade allowable (= "design life" baseline)
_T_METAL_REF_K = T_METAL_BLADE_R * RANKINE_TO_K   # ≈ 1172 K


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLIFIED CYCLE MODEL
#  Uses fixed cooling fracs as cycle params — supports zero-cooling scenarios.
#  combustor_py_cycle uses CoolingCalcs inside the Newton solve (dynamic fracs)
#  which cannot represent zero-cooling baseline conditions.
# ══════════════════════════════════════════════════════════════════════════════

def _make_turbojet_core(comp_map, turb_map):
    def _setup_core(self):
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data']   = pyc.AIR_JETA_TAB_SPEC

        self.add_subsystem('fc',     pyc.FlightConditions())
        self.add_subsystem('inlet',  pyc.Inlet())
        self.add_subsystem('comp',   pyc.Compressor(map_data=comp_map,
                                                     map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('bld3',   pyc.BleedOut(bleed_names=['ngv_cool', 'bld_cool']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type='FAR'))
        self.add_subsystem('turb',   pyc.Turbine(map_data=turb_map,
                                                  bleed_names=['ngv_cool', 'bld_cool'],
                                                  map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('nozz',   pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft',  pyc.Shaft(num_ports=2),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('perf',   pyc.Performance(num_nozzles=1, num_burners=1))

        self.pyc_connect_flow('fc.Fl_O',       'inlet.Fl_I',  connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O',    'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O',     'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O',     'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O',   'turb.Fl_I')
        self.pyc_connect_flow('bld3.ngv_cool', 'turb.ngv_cool', connect_stat=False)
        self.pyc_connect_flow('bld3.bld_cool', 'turb.bld_cool', connect_stat=False)
        self.pyc_connect_flow('turb.Fl_O',     'nozz.Fl_I')

        self.connect('comp.trq',         'shaft.trq_0')
        self.connect('turb.trq',         'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P',   'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P',  'perf.Pt3')
        self.connect('burner.Wfuel',     'perf.Wfuel_0')
        self.connect('inlet.F_ram',      'perf.ram_drag')
        self.connect('nozz.Fg',          'perf.Fg_0')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol']             = 1e-6
        newton.options['rtol']             = 1e-6
        newton.options['iprint']           = 2
        newton.options['maxiter']          = 15
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves']   = 100
        newton.options['reraise_child_analysiserror'] = False
        self.linear_solver = om.DirectSolver()

    return _setup_core


def _make_turbojet(cfg):
    _core = _make_turbojet_core(cfg['comp_map'], cfg['turb_map'])

    class Turbojet(pyc.Cycle):
        def setup(self):
            _core(self)
            design = self.options['design']
            bal = self.add_subsystem('balance', om.BalanceComp())
            if design:
                bal.add_balance('W', units='lbm/s', eq_units='lbf',
                                rhs_name='Fn_target')
                self.connect('balance.W',  'inlet.Fl_I:stat:W')
                self.connect('perf.Fn',    'balance.lhs:W')

                bal.add_balance('FAR', eq_units='degR', lower=1e-4,
                                val=0.017, rhs_name='T4_target')
                self.connect('balance.FAR',       'burner.Fl_I:FAR')
                self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

                bal.add_balance('turb_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', rhs_val=0.)
                self.connect('balance.turb_PR', 'turb.PR')
                self.connect('shaft.pwr_net',   'balance.lhs:turb_PR')
            super().setup()

    return Turbojet


def _make_turbojet_const_fn(cfg):
    _core = _make_turbojet_core(cfg['comp_map'], cfg['turb_map'])

    class TurbojetConstFn(pyc.Cycle):
        def setup(self):
            _core(self)
            bal = self.add_subsystem('balance', om.BalanceComp())

            bal.add_balance('FAR', eq_units='lbf', lower=1e-4,
                            val=0.017, rhs_name='Fn_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn',     'balance.lhs:FAR')

            bal.add_balance('Nmech', val=8070., units='rpm',
                            lower=500., eq_units='hp', rhs_val=0.)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            bal.add_balance('W', val=168.0, units='lbm/s',
                            eq_units='inch**2', rhs_name='nozz_area_target')
            self.connect('balance.W',             'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')
            super().setup()

    return TurbojetConstFn


def _add_design_point_and_params(mp, Turbojet, cfg, include_dPqP=True):
    mp.pyc_add_pnt('DESIGN', Turbojet())
    mp.set_input_defaults('DESIGN.Nmech',     cfg['Nmech'],    units='rpm')
    mp.set_input_defaults('DESIGN.inlet.MN',  cfg['inlet_MN'])
    mp.set_input_defaults('DESIGN.comp.MN',   cfg['comp_MN'])
    mp.set_input_defaults('DESIGN.burner.MN', cfg['burner_MN'])
    mp.set_input_defaults('DESIGN.turb.MN',   cfg['turb_MN'])
    if include_dPqP:
        mp.pyc_add_cycle_param('burner.dPqP', cfg['burner_dPqP'])
    mp.pyc_add_cycle_param('nozz.Cv',              cfg['nozz_Cv'])
    mp.pyc_add_cycle_param('bld3.ngv_cool:frac_W', cfg['ngv_cool_frac'])
    mp.pyc_add_cycle_param('bld3.bld_cool:frac_W', cfg['bld_cool_frac'])
    mp.pyc_add_cycle_param('turb.ngv_cool:frac_P', 1.0)
    mp.pyc_add_cycle_param('turb.bld_cool:frac_P', 0.0)


def _make_mp_single_fn(cfg):
    Turbojet        = _make_turbojet(cfg)
    TurbojetConstFn = _make_turbojet_const_fn(cfg)

    class MPSingleFn(pyc.MPCycle):
        def setup(self):
            _add_design_point_and_params(self, Turbojet, cfg, include_dPqP=False)
            self.a4_scalars = cfg['a4_scalars']
            self.od_pts     = [f'A4_{i:03d}' for i in range(len(self.a4_scalars))]

            for pt in self.od_pts:
                self.pyc_add_pnt(pt, TurbojetConstFn(design=False))
                self.set_input_defaults(pt + '.fc.MN',  0.000001)
                self.set_input_defaults(pt + '.fc.alt', 0.0, units='ft')

            _connect_od_scalars(self, self.od_pts)
            super().setup()

    return MPSingleFn


# ══════════════════════════════════════════════════════════════════════════════
#  SWEEPS
# ══════════════════════════════════════════════════════════════════════════════

def _annotate_result(r, prob, pt, a4, fn_frac, dPqP, bld_cool_frac):
    """Add derived quantities (η_th, cooling req, T_metal, life ratio) in-place."""
    r.update(a4_scalar=a4, dPqP=dPqP, fn_frac=fn_frac,
             eta_th=extract_eta_th(prob, pt))
    req_ngv, req_blade = gauntner_cooling_fracs(r['T4'], r['T3'],
                                                 T4_exit_R=r['T4_exit'])
    r['req_frac_ngv']   = req_ngv
    r['req_frac_blade'] = req_blade

    Tt4_rel_K       = turb_rel_temp_rankine(r['T4'], r['T4_exit']) * RANKINE_TO_K
    T3_K            = r['T3'] * RANKINE_TO_K
    r['T_metal']    = Tt4_rel_K - ETA_COOL * bld_cool_frac * (Tt4_rel_K - T3_K)
    r['life_ratio'] = lmp_life_ratio(_T_METAL_REF_K, r['T_metal'])


def run_fixed_dPqP_sweep(prob, mp, design_results, fn_frac, cfg, mode,
                          bld_cool_frac=0.0):
    print(f"\n  Fixed dP/P={DPQP_FIXED*100:.0f}%  {fn_frac*100:.0f}% Fn")
    Fn_target = fn_frac * cfg['Fn_design']
    for pt, a4 in zip(mp.od_pts, mp.a4_scalars):
        set_turb_area(prob, pt, a4, design_results['s_Wp_design'], mode)
        prob.set_val(f'{pt}.balance.Fn_target', Fn_target, units='lbf')
        prob.set_val(f'{pt}.burner.dPqP', DPQP_FIXED)
        prob[pt + '.balance.Nmech'] = (design_results['Nmech']
                                       * fn_frac**0.5 * (1.0/a4)**0.3)
        prob[pt + '.balance.FAR']   = cfg.get('FAR_guess', 0.017) * fn_frac

    _run_od_sweep_sequential(prob, mp, maxiter=15)

    results = []
    for pt, a4 in zip(mp.od_pts, mp.a4_scalars):
        r = extract_od_point(prob, pt)
        _annotate_result(r, prob, pt, a4, fn_frac, DPQP_FIXED, bld_cool_frac)
        results.append(r)
    return results


def run_physics_dPqP_sweep(prob, mp, design_results, fn_frac, cfg, mode,
                            bld_cool_frac=0.0):
    print(f"\n  Physics dP/P  {fn_frac*100:.0f}% Fn")
    Fn_target = fn_frac * cfg['Fn_design']
    dPqP_cur  = {pt: DPQP_FIXED for pt in mp.od_pts}

    for outer in range(MAX_OUTER):
        print(f"\n  Outer iter {outer+1}/{MAX_OUTER}")
        for pt, a4 in zip(mp.od_pts, mp.a4_scalars):
            set_turb_area(prob, pt, a4, design_results['s_Wp_design'], mode)
            prob.set_val(f'{pt}.balance.Fn_target', Fn_target, units='lbf')
            prob.set_val(f'{pt}.burner.dPqP', dPqP_cur[pt])
            prob[pt + '.balance.Nmech'] = (design_results['Nmech']
                                           * fn_frac**0.5 * (1.0/a4)**0.3)
            prob[pt + '.balance.FAR']   = cfg.get('FAR_guess', 0.017) * fn_frac

        _run_od_sweep_sequential(prob, mp, maxiter=15)

        dPqP_new, max_change = {}, 0.0
        for pt in mp.od_pts:
            r = extract_od_point(prob, pt)
            _, dPqP_phys = physics_pib(
                psi_to_pa(r['P3']), rankine_to_kelvin(r['T3']),
                lbm_s_to_kg_s(r['W']), design_results['CdA_liner'])
            dPqP_phys    = float(np.clip(dPqP_phys, 0.001, 0.50))
            dPqP_new[pt] = dPqP_phys
            change       = abs(dPqP_phys - dPqP_cur[pt]) / max(dPqP_cur[pt], 1e-6)
            max_change   = max(max_change, change)
            print(f'    {pt}  dP/P: {dPqP_cur[pt]*100:.2f}%→{dPqP_phys*100:.2f}%')

        print(f'  max change={max_change*100:.3f}%')
        dPqP_cur = dPqP_new
        if max_change < DPQP_TOL:
            print(f'  Converged in {outer+1} iters')
            break

    # Cleanup pass at converged dPqP values
    print('\n  Final cleanup pass...')
    for pt in mp.od_pts:
        prob.set_val(f'{pt}.burner.dPqP', dPqP_cur[pt])
    _run_od_sweep_sequential(prob, mp, maxiter=20)

    results = []
    for pt, a4 in zip(mp.od_pts, mp.a4_scalars):
        r = extract_od_point(prob, pt)
        _annotate_result(r, prob, pt, a4, fn_frac, dPqP_cur[pt], bld_cool_frac)
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(scenario_name, cfg, mode, turb_map, turb_eff, turb_PR_guess,
                 ngv_cool_frac, bld_cool_frac):
    print(f"\n{'█'*70}")
    print(f" SCENARIO: {scenario_name}")
    print(f"{'█'*70}")
    print(f" Cooling: NGV={ngv_cool_frac*100:.1f}%  blade={bld_cool_frac*100:.1f}%")

    cfg = {
        **cfg,
        'turb_map':      turb_map,
        'turb_eff':      turb_eff,
        'turb_PR_guess': turb_PR_guess,
        'ngv_cool_frac': ngv_cool_frac,
        'bld_cool_frac': bld_cool_frac,
    }

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
    prob['DESIGN.balance.FAR']     = cfg.get('FAR_guess', 0.0175)
    prob['DESIGN.balance.W']       = cfg.get('W_guess', 168.0)
    prob['DESIGN.balance.turb_PR'] = cfg.get('turb_PR_guess', 3.5)
    prob['DESIGN.fc.balance.Pt']   = 14.696
    prob['DESIGN.fc.balance.Tt']   = 518.67
    for pt in mp.od_pts:
        prob.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 0
    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    print('\n── Design point ──')
    t0 = time.time()
    prob.run_model()
    print(f'Design solved in {time.time()-t0:.1f}s')

    des = extract_design(prob)
    print(f"  T4={des['T4_design']*RANKINE_TO_K:.1f}K  "
          f"T3={des['T3_design']*RANKINE_TO_K:.1f}K  "
          f"eff={des['eff_design']*100:.3f}%  s_Wp={des['s_Wp_design']:.6f}")

    # CdA from design point
    des['CdA_liner'] = calc_CdA_liner(
        des['T3_design'] * RANKINE_TO_K,
        psi_to_pa(des['P3_design']),
        lbm_s_to_kg_s(des['W']))

    # Re-anchor CdA at A4=1.0, 100% Fn for better accuracy
    if 1.00 in cfg['fn_fractions']:
        run_fixed_dPqP_sweep(prob, mp, des, 1.00, cfg, mode, bld_cool_frac)
        a4_1 = min(mp.a4_scalars, key=lambda a: abs(a - 1.0))
        idx1 = mp.a4_scalars.index(a4_1)
        pt1  = mp.od_pts[idx1]
        r1   = extract_od_point(prob, pt1)
        des['CdA_liner'] = calc_CdA_liner(
            r1['T3'] * RANKINE_TO_K,
            psi_to_pa(r1['P3']),
            lbm_s_to_kg_s(r1['W']))
        print(f"  CdA (re-anchored) = {des['CdA_liner']*1e4:.4f} cm²")

    print('\n── Fixed dP/P sweep ──')
    fixed_by_fn = {}
    for fn_frac in cfg['fn_fractions']:
        fixed_by_fn[fn_frac] = run_fixed_dPqP_sweep(
            prob, mp, des, fn_frac, cfg, mode, bld_cool_frac)

    print('\n── Physics dP/P sweep ──')
    physics_by_fn = {}
    for fn_frac in cfg['fn_fractions']:
        physics_by_fn[fn_frac] = run_physics_dPqP_sweep(
            prob, mp, des, fn_frac, cfg, mode, bld_cool_frac)

    return {
        'name':          scenario_name,
        'cooling':       f"{ngv_cool_frac*100:.2f}% NGV + {bld_cool_frac*100:.2f}% blade",
        'ngv_cool_frac': ngv_cool_frac,
        'bld_cool_frac': bld_cool_frac,
        'fixed_dpqp':    fixed_by_fn,
        'phys_dpqp':     physics_by_fn,
        'design':        des,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    _Nmech   = 8070.0
    _T4_degR = 2370.0
    _Np_des  = _Nmech / np.sqrt(_T4_degR)

    eff_fixed, PR_fixed = map_design_point(EngineHPTFixedMap, _Np_des)
    eff_vgt,   PR_vgt   = map_design_point(EngineHPTVGTMap,   _Np_des)

    print("=" * 70)
    print(f"  Fixed: eta={eff_fixed*100:.3f}%  PR={PR_fixed:.4f}")
    print(f"  VGT:   eta={eff_vgt*100:.3f}%    PR={PR_vgt:.4f}")
    print(f"  Leakage penalty: Δeta={(eff_fixed-eff_vgt)*100:+.3f}%")
    print("=" * 70)

    BASE_CFG = dict(
        comp_map     = pyc.HPCMap,
        Fn_design    = 11800.0 * 0.5,
        T4_design    = _T4_degR,
        OPR          = 13.5 * 0.5,
        Nmech        = _Nmech,
        comp_eff     = 0.83,
        inlet_MN     = 0.60,
        comp_MN      = 0.020,
        burner_MN    = 0.020,
        turb_MN      = 0.40,
        nozz_Cv      = 0.99,
        FAR_guess    = 0.0175,
        W_guess      = 168.0,
        fn_fractions = [0.50, 1.00],
        a4_scalars   = list(np.linspace(0.75, 1.00, 15)),
    )

    _T3_est   = estimate_T3_rankine(BASE_CFG['OPR'], comp_eff=BASE_CFG['comp_eff'])
    _frac_ngv, _frac_blade = gauntner_cooling_fracs(BASE_CFG['T4_design'], _T3_est)
    print("=" * 70)
    print(f"  Gauntner sizing: T4={BASE_CFG['T4_design']:.0f}°R  T3_est={_T3_est:.1f}°R")
    print(f"  T_metal_vane={T_METAL_VANE_R:.0f}°R  T_metal_blade={T_METAL_BLADE_R:.0f}°R")
    print(f"  NGV={_frac_ngv*100:.2f}%  Blade={_frac_blade*100:.2f}%")
    print("=" * 70)

    # ── Run three scenarios ───────────────────────────────────────────────────

    scen1 = run_scenario(
        "Scenario 1: Zero Cooling, Fixed Turbine",
        BASE_CFG, mode='sWp',
        turb_map=EngineHPTFixedMap, turb_eff=eff_fixed, turb_PR_guess=PR_fixed,
        ngv_cool_frac=0.0, bld_cool_frac=0.0,
    )

    scen2 = run_scenario(
        "Scenario 2: Gauntner Cooling, VGT Turbine",
        BASE_CFG, mode='alpha',
        turb_map=EngineHPTVGTMap, turb_eff=eff_vgt, turb_PR_guess=PR_vgt,
        ngv_cool_frac=_frac_ngv, bld_cool_frac=_frac_blade,
    )

    scen3 = run_scenario(
        "Scenario 3: Zero Cooling, VGT Turbine",
        BASE_CFG, mode='alpha',
        turb_map=EngineHPTVGTMap, turb_eff=eff_vgt, turb_PR_guess=PR_vgt,
        ngv_cool_frac=0.0, bld_cool_frac=0.0,
    )

    all_scenarios = [scen1, scen2, scen3]

    with open('thermal_efficiency_scenarios.pkl', 'wb') as f:
        pickle.dump(all_scenarios, f)
    print("\nSaved: thermal_efficiency_scenarios.pkl")

    # ── Plot helpers ──────────────────────────────────────────────────────────

    fn_fracs = BASE_CFG['fn_fractions']
    ncols    = len(fn_fracs)

    COLORS  = {s['name']: c for s, c in zip(all_scenarios,
                                             ['steelblue', 'darkorange', 'firebrick'])}
    MARKERS = {s['name']: m for s, m in zip(all_scenarios, ['o', 's', '^'])}

    def _arr(scen, fn_frac, key, src='phys_dpqp'):
        return np.array([r[key] for r in scen[src][fn_frac]])

    def _ax_defaults(ax, fn_frac):
        ax.axvline(1.0, color='gray', ls='--', lw=1.2, alpha=0.6)
        ax.set_xlabel('A₄ scalar')
        ax.set_title(f'{fn_frac*100:.0f}% Fn', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── Figure 1: η_th vs A₄ ─────────────────────────────────────────────────

    fig1, axes1 = plt.subplots(1, ncols, figsize=(8*ncols, 6), sharey=False)
    if ncols == 1:
        axes1 = [axes1]
    fig1.suptitle('Thermal Efficiency vs A₄ Scalar', fontsize=12, fontweight='bold')

    for col, fn_frac in enumerate(fn_fracs):
        ax = axes1[col]
        for scen in all_scenarios:
            ax.plot(_arr(scen, fn_frac, 'a4_scalar'),
                    _arr(scen, fn_frac, 'eta_th') * 100,
                    f"{MARKERS[scen['name']]}-",
                    color=COLORS[scen['name']], lw=2.5, ms=6,
                    label=scen['name'])
        if col == 0:
            ax.set_ylabel('η_th (%)')
        _ax_defaults(ax, fn_frac)

    fig1.tight_layout()
    fig1.savefig('thermal_efficiency_scenarios.png', dpi=150, bbox_inches='tight')
    print("Saved: thermal_efficiency_scenarios.png")

    # ── Figure 2: Gauntner cooling prescription vs A₄ ────────────────────────

    _frac_total_des = (_frac_ngv + _frac_blade) * 100

    fig2, axes2 = plt.subplots(1, ncols, figsize=(8*ncols, 6), sharey=False)
    if ncols == 1:
        axes2 = [axes2]
    fig2.suptitle(
        'Gauntner Cooling Prescription vs A₄  (NASA-TM-81453)\n'
        'Solid = prescription at OD conditions  |  Dashed = applied fraction',
        fontsize=11, fontweight='bold')

    for col, fn_frac in enumerate(fn_fracs):
        ax = axes2[col]
        for scen in all_scenarios:
            s     = _arr(scen, fn_frac, 'a4_scalar')
            f_tot = (_arr(scen, fn_frac, 'req_frac_ngv') +
                     _arr(scen, fn_frac, 'req_frac_blade')) * 100
            applied = (scen['ngv_cool_frac'] + scen['bld_cool_frac']) * 100
            ax.plot(s, f_tot, f"{MARKERS[scen['name']]}-",
                    color=COLORS[scen['name']], lw=2, ms=5, label=scen['name'])
            ax.axhline(applied, color=COLORS[scen['name']], ls='--', lw=1.5, alpha=0.55)

        # Shade cooling margin / deficit for Scenario 2
        s2    = _arr(scen2, fn_frac, 'a4_scalar')
        f2    = (_arr(scen2, fn_frac, 'req_frac_ngv') +
                 _arr(scen2, fn_frac, 'req_frac_blade')) * 100
        ax.fill_between(s2, _frac_total_des, f2, where=(f2 <= _frac_total_des),
                        alpha=0.12, color=COLORS[scen2['name']], label='S2 margin')
        ax.fill_between(s2, _frac_total_des, f2, where=(f2 > _frac_total_des),
                        alpha=0.12, color='firebrick', label='S2 deficit')

        if col == 0:
            ax.set_ylabel('Cooling fraction (% W_gas)')
        _ax_defaults(ax, fn_frac)

    fig2.tight_layout()
    fig2.savefig('gauntner_cooling_prescription.png', dpi=150, bbox_inches='tight')
    print("Saved: gauntner_cooling_prescription.png")

    # ── Figure 3: T_metal and LMP life ratio vs A₄ ───────────────────────────
    # life_ratio referenced to T_METAL_BLADE_R (Gauntner allowable ≈ 1172 K)
    # life_ratio > 1 → blade cooler than allowable → creep life extended
    # life_ratio < 1 → blade over-temperature (zero-cooling scenarios)

    fig3, axes3 = plt.subplots(2, ncols, figsize=(8*ncols, 10), sharey='row')
    if ncols == 1:
        axes3 = axes3.reshape(2, 1)
    fig3.suptitle(
        f'Blade Metal Temperature and LMP Life Ratio vs A₄\n'
        f'Reference: T_blade = {_T_METAL_REF_K:.0f} K  '
        f'(C=20, t_des=20 000 hr,  Larson & Miller 1952)',
        fontsize=11, fontweight='bold')

    for col, fn_frac in enumerate(fn_fracs):
        ax_tm  = axes3[0, col]
        ax_lmp = axes3[1, col]

        for scen in all_scenarios:
            s     = _arr(scen, fn_frac, 'a4_scalar')
            tm_K  = _arr(scen, fn_frac, 'T_metal')
            lr    = _arr(scen, fn_frac, 'life_ratio')
            color = COLORS[scen['name']]
            mkr   = MARKERS[scen['name']]

            ax_tm.plot(s, tm_K, f'{mkr}-', color=color, lw=2.5, ms=6,
                       label=scen['name'])
            ax_lmp.plot(s, lr,  f'{mkr}-', color=color, lw=2.5, ms=6,
                        label=scen['name'])

        ax_tm.axhline(_T_METAL_REF_K, color='k', ls=':', lw=1.5,
                      label=f'Allowable {_T_METAL_REF_K:.0f} K')
        ax_lmp.axhline(1.0, color='k', ls=':', lw=1.5, label='Design life')

        if col == 0:
            ax_tm.set_ylabel('T_metal (K)')
            ax_lmp.set_ylabel('Life ratio  t_r / t_r,des')

        _ax_defaults(ax_tm,  fn_frac)
        _ax_defaults(ax_lmp, fn_frac)

    fig3.tight_layout()
    fig3.savefig('blade_life_scenarios.png', dpi=150, bbox_inches='tight')
    print("Saved: blade_life_scenarios.png")

    plt.show()

    print("\n" + "=" * 70)
    print(" All scenarios completed.")
    print("=" * 70)
