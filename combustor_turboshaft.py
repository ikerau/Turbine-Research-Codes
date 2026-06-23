"""
combustor_turboshaft.py
========================
Variable HPT NGV Throat Area (A4) study — simple turboshaft cycle.
Compares Fixed geometry vs VGT (Brent-optimal A4) across a power fraction sweep.

Architecture
------------
  HP spool : Compressor + Gas Generator Turbine (GGT/turb)  → HP_shaft
  LP spool : Power Turbine (PT/pt) → LP_shaft (output power)

Study mode (VARICAP per Rogo & Benstein 1986)
----------------------------------------------
  LP_Nmech held at 100% design speed throughout.
  Power modulated by varying FAR (and A4 for VGT).
  Fixed: turb.s_Wp = design value
  VGT  : Brent minimises PSFC over a4_sc ∈ [0.85, 1.15]
           a4_sc < 1 → close A4 → higher GG speed (constant-TIT mode)
           a4_sc > 1 → open  A4 → lower T4    (constant-OPR mode)
         Roy-Aikins (1990, ASME 90-GT-271) shows constant-OPR gives lower
         SFC for simple turboshaft cycles at part power.
  T4 redline: T4 never permitted to exceed design TIT.

References
----------
  Roy-Aikins (1990, 90-GT-271)  — constant-OPR control for simple cycle
  Rogo & Benstein (1986)        — VARICAP concept; digitised reference curves
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import openmdao.api as om
import pycycle.api as pyc
import cantera as ct
from pycycle.elements.cooling import CoolingCalcs

from EngineHPT_map       import EngineHPTMap as EngineHPTVGTMap
from EngineHPT_map_fixed import EngineHPTMap as EngineHPTFixedMap
from scipy.interpolate import RegularGridInterpolator

np.seterr(divide='ignore', invalid='ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  STUDY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Study sweep parameters ────────────────────────────────────────────────────
DESIGN_ALT  = 0.0     # ft  — sweep always at SLS
DESIGN_MN   = 0.001   # ~static

PWR_FRACS  = np.linspace(0.60, 1.00, 30)   # 50–100% of design SHP
A4_BOUNDS  = (0.80, 1.05)                  # Brent bounds (closing AND opening)

# ── Unit conversion constants ─────────────────────────────────────────────────
LBM_S_TO_KG_S = 0.453592
RANKINE_TO_K  = 5.0 / 9.0
HP_TO_KW      = 0.745700
PSI_TO_PA     = 6894.757

# ── Combustor model ───────────────────────────────────────────────────────────
DPQP_DESIGN   = 0.03   # reference combustor pressure drop at design point

# ── Cooling physics constants — Walsh & Fletcher (2004) §5.4 / Gauntner NASA-TM-81453
ETA_COOL        = 0.65    # film cooling effectiveness
T_METAL_VANE_R  = 2210.0  # allowable NGV bulk metal temperature (°R)
T_METAL_BLADE_R = 2110.0  # allowable rotor blade bulk metal temperature (°R)
PSI_DESIGN      = 1.0   # turbine blade loading coeff ψ (from meanline design)

# ══════════════════════════════════════════════════════════════════════════════
#  MAP / CYCLE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def estimate_T3_rankine(OPR, T2_R=518.67, comp_eff=0.83, gamma=1.4):
    """Analytical estimate of compressor exit total temperature (°R)."""
    T3_ideal = T2_R * OPR ** ((gamma - 1.0) / gamma)
    return T2_R + (T3_ideal - T2_R) / comp_eff


def map_design_point(turb_map, Np_des, alpha_design=1.0):
    """Return (eff_des, PR_des) from turb_map at Np_des and alpha_design."""
    a_idx  = int(np.argmin(np.abs(np.array(turb_map.alphaMap) - alpha_design)))
    eff_sl = np.array(turb_map.effMap[a_idx])
    NpMap  = np.array(turb_map.NpMap)
    PRmap  = np.array(turb_map.PRmap)
    PR_des = turb_map.defaults['PRmap']
    Np_q   = float(np.clip(Np_des, NpMap[0], NpMap[-1]))
    interp = RegularGridInterpolator(
        (NpMap, PRmap), eff_sl, method='linear',
        bounds_error=False, fill_value=None)
    return float(np.asarray(interp([[Np_q, PR_des]])).item()), PR_des


# ══════════════════════════════════════════════════════════════════════════════
#  COOLING MODEL  (Gauntner NASA-TM-81453)
# ══════════════════════════════════════════════════════════════════════════════

def turb_rel_temp_rankine(T4_R, T4_exit_R, psi=PSI_DESIGN):
    """Relative stagnation temperature at rotor inlet (°R).
    T0_rel = T4 - ΔTt_stage/(2ψ) — rotor blade sees T0_rel, not absolute T4."""
    return T4_R - (T4_R - T4_exit_R) / (2.0 * psi)


def gauntner_cooling_fracs(T4_R, T3_R, T4_exit_R=None,
                            T_metal_vane_R=T_METAL_VANE_R,
                            T_metal_blade_R=T_METAL_BLADE_R):
    """Per-row cooling flow fractions from Gauntner's COOLIT algorithm.
    Returns (frac_ngv, frac_blade) as fractions of turbine inlet gas flow."""
    SAFETY = 150.0

    def _row_frac(T_gas_R, T_metal_R, profil):
        T_eff = T_gas_R + SAFETY
        PHI   = (T_eff - T_metal_R) / abs(T_eff - T3_R)
        PHI   = (profil + PHI) / (profil + 1.0)
        PHI   = max(PHI, 0.0)
        if PHI >= 1.0:
            return 0.022 * 10.0 * (4.0 / 3.0)
        return 0.022 * (PHI / (1.0 - PHI)) ** 1.25 * (4.0 / 3.0)

    T_blade_gas_R = (turb_rel_temp_rankine(T4_R, T4_exit_R)
                     if T4_exit_R is not None else 0.92 * T4_R)
    frac_ngv   = _row_frac(T4_R,          T_metal_vane_R,  profil=0.30)
    frac_blade = _row_frac(T_blade_gas_R, T_metal_blade_R, profil=0.13)
    return frac_ngv, frac_blade


def t_metal_rankine(T_gas_R, T3_R, eta_cool=ETA_COOL):
    """Effective blade metal temperature from film cooling effectiveness."""
    return T_gas_R - eta_cool * (T_gas_R - T3_R)


# ══════════════════════════════════════════════════════════════════════════════
#  COMBUSTOR PRESSURE DROP MODEL
# ══════════════════════════════════════════════════════════════════════════════

_GRI30 = ct.Solution("gri30.yaml")

def rho_total(Tt_K, Pt_Pa):
    _GRI30.TPX = Tt_K, Pt_Pa, "O2:0.21, N2:0.79"
    return float(_GRI30.density)


def calc_CdA_liner(Tt3_K, Pt3_Pa, mdot_si, dPqP_ref=DPQP_DESIGN):
    """Back-calculate orifice effective area from design-point flow state."""
    rho   = rho_total(Tt3_K, Pt3_Pa)
    dPt   = dPqP_ref * Pt3_Pa
    return float(mdot_si / np.sqrt(2.0 * rho * dPt))


class PhysicsCombustor(om.ExplicitComponent):
    """
    Orifice liner head-loss model as an OpenMDAO component.
    Connected to burner.dPqP inside the OD Cycle so the Newton solver treats
    combustor pressure drop as an algebraic state — physics-consistent in a
    single Newton solve with no outer iteration.

      dPqP = (ṁ_SI / CdA)² / (2 · ρ_total(Tt3, Pt3) · Pt3)

    Inputs use pyCycle imperial conventions (lbm/s, degR, psi).
    CdA is in SI (m²) — calibrated once from the design-point flow state.
    """
    def setup(self):
        self.add_input('W',   val=10.0,  units='lbm/s', desc='inlet mass flow')
        self.add_input('Tt3', val=800.0, units='degR',  desc='total temperature at burner inlet')
        self.add_input('Pt3', val=100.0, units='psi',   desc='total pressure at burner inlet')
        self.add_input('CdA', val=1e-3,                 desc='orifice effective area  (m²)')
        self.add_output('dPqP', val=DPQP_DESIGN, lower=1e-4, upper=0.49,
                        desc='fractional total-pressure loss (→ burner.dPqP)')
        # FD partials: Cantera inside rho_total can't accept complex perturbations.
        self.declare_partials('dPqP', ['W', 'Tt3', 'Pt3', 'CdA'],
                              method='fd', step=1e-5, step_calc='rel')

    def compute(self, inputs, outputs):
        W_si   = float(inputs['W'][0])   * LBM_S_TO_KG_S
        Tt3_K  = float(inputs['Tt3'][0]) * RANKINE_TO_K
        Pt3_Pa = float(inputs['Pt3'][0]) * PSI_TO_PA
        CdA    = float(inputs['CdA'][0])
        if CdA < 1e-10:
            outputs['dPqP'] = DPQP_DESIGN
            return
        # Clamp to physical bounds: Newton sub-iterations can visit unphysical states
        Tt3_K  = max(Tt3_K,  200.0)
        Pt3_Pa = max(Pt3_Pa, 1000.0)
        W_si   = max(abs(W_si), 1e-6)
        rho  = rho_total(Tt3_K, Pt3_Pa)
        dPt  = (W_si / CdA) ** 2 / (2.0 * rho)
        outputs['dPqP'] = float(np.clip(dPt / Pt3_Pa, 1e-4, 0.49))


# ══════════════════════════════════════════════════════════════════════════════
#  CYCLE DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

def _make_turboshaft(cfg):
    """
    Factory: close over cfg to produce a Cycle class.

    Topology: fc → inlet → comp → bld3 → burner → turb (HP) → pt (LP) → nozz
                                    ↓ ngv_cool, bld_cool
                                    └──────────────────→ turb (bleed reinjection)
    """
    _turb_map  = cfg['turb_map']
    _pt_map    = cfg.get('pt_map',   pyc.LPT2269)
    _comp_map  = cfg.get('comp_map', pyc.AXI5)
    _HP_N_0    = cfg['HP_Nmech_init']

    class SingleSpoolTurboshaft(pyc.Cycle):
        def setup(self):
            design = self.options['design']

            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data']   = pyc.AIR_JETA_TAB_SPEC

            self.add_subsystem('fc',    pyc.FlightConditions())
            self.add_subsystem('inlet', pyc.Inlet())
            self.add_subsystem('comp',
                               pyc.Compressor(map_data=_comp_map, map_extrap=True),
                               promotes_inputs=[('Nmech', 'HP_Nmech')])
            self.add_subsystem('bld3',   pyc.BleedOut(bleed_names=['ngv_cool', 'bld_cool']))
            self.add_subsystem('burner', pyc.Combustor(fuel_type='FAR'))
            self.add_subsystem('turb',
                               pyc.Turbine(map_data=_turb_map,
                                           bleed_names=['ngv_cool', 'bld_cool'],
                                           map_extrap=True),
                               promotes_inputs=[('Nmech', 'HP_Nmech')])
            self.add_subsystem('pt',
                               pyc.Turbine(map_data=_pt_map, map_extrap=True),
                               promotes_inputs=[('Nmech', 'LP_Nmech')])
            self.add_subsystem('nozz',     pyc.Nozzle(nozzType='CV', lossCoef='Cv'))
            self.add_subsystem('HP_shaft', pyc.Shaft(num_ports=2),
                               promotes_inputs=[('Nmech', 'HP_Nmech')])
            self.add_subsystem('LP_shaft', pyc.Shaft(num_ports=1),
                               promotes_inputs=[('Nmech', 'LP_Nmech')])
            self.add_subsystem('perf',
                               pyc.Performance(num_nozzles=1, num_burners=1))
            self.add_subsystem('ngv_calcs', CoolingCalcs(
                n_stages=1, i_row=0,
                T_metal=T_METAL_VANE_R, T_safety=150.0))
            self.add_subsystem('bld_calcs', CoolingCalcs(
                n_stages=1, i_row=1,
                T_metal=T_METAL_BLADE_R, T_safety=150.0))
            self.add_subsystem('cool_fracs', om.ExecComp(
                ['frac_ngv = W_ngv / W_in',
                 'frac_bld = W_bld / W_in'],
                W_ngv={'val': 5.0, 'units': 'lbm/s'},
                W_bld={'val': 3.0, 'units': 'lbm/s'},
                W_in={'val': 100.0, 'units': 'lbm/s'},
            ))

            self.pyc_connect_flow('fc.Fl_O',       'inlet.Fl_I',   connect_w=False)
            self.pyc_connect_flow('inlet.Fl_O',    'comp.Fl_I')
            self.pyc_connect_flow('comp.Fl_O',     'bld3.Fl_I')
            self.pyc_connect_flow('bld3.Fl_O',     'burner.Fl_I')
            self.pyc_connect_flow('burner.Fl_O',   'turb.Fl_I')
            self.pyc_connect_flow('bld3.ngv_cool', 'turb.ngv_cool', connect_stat=False)
            self.pyc_connect_flow('bld3.bld_cool', 'turb.bld_cool', connect_stat=False)
            self.pyc_connect_flow('turb.Fl_O',     'pt.Fl_I')
            self.pyc_connect_flow('pt.Fl_O',       'nozz.Fl_I')

            self.connect('comp.trq',         'HP_shaft.trq_0')
            self.connect('turb.trq',         'HP_shaft.trq_1')
            self.connect('pt.trq',           'LP_shaft.trq_0')
            self.connect('fc.Fl_O:stat:P',   'nozz.Ps_exhaust')
            self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
            self.connect('comp.Fl_O:tot:P',  'perf.Pt3')
            self.connect('burner.Wfuel',     'perf.Wfuel_0')
            self.connect('inlet.F_ram',      'perf.ram_drag')
            self.connect('nozz.Fg',          'perf.Fg_0')
            self.connect('LP_shaft.pwr_net', 'perf.power')

            # CoolingCalcs: compute W_cool and frac_W for tracking / post-processing.
            # For DESIGN (T4 fixed by balance): connect fracs directly into Newton — stable.
            # For OD (T4 free): fracs are fed via set_val in an outer loop inside solve_od(),
            # NOT connected here, to prevent the T4↑→cooling↑→W↓→power↓→FAR↑→T4↑ instability.
            for _calc in ('ngv_calcs', 'bld_calcs'):
                self.connect('burner.Fl_O:tot:T',  f'{_calc}.Tt_primary')
                self.connect('burner.Fl_O:tot:h',  f'{_calc}.ht_primary')
                self.connect('burner.Fl_O:stat:W', f'{_calc}.W_primary')
                self.connect('burner.Fl_O:tot:P',  f'{_calc}.Pt_in')
                self.connect('turb.Fl_O:tot:P',    f'{_calc}.Pt_out')
                self.connect('bld3.Fl_O:tot:T',    f'{_calc}.Tt_cool')
                self.connect('bld3.Fl_O:tot:h',    f'{_calc}.ht_cool')
                self.connect('turb.power',          f'{_calc}.turb_pwr')
            self.connect('ngv_calcs.W_cool', 'cool_fracs.W_ngv')
            self.connect('bld_calcs.W_cool', 'cool_fracs.W_bld')
            self.connect('comp.Fl_O:stat:W', 'cool_fracs.W_in')
            if design:
                self.connect('cool_fracs.frac_ngv', 'bld3.ngv_cool:frac_W')
                self.connect('cool_fracs.frac_bld', 'bld3.bld_cool:frac_W')
            # OD: bld3.ngv_cool:frac_W / bld_cool:frac_W set externally in solve_od()

            balance = self.add_subsystem('balance', om.BalanceComp())

            if design:
                balance.add_balance('W', val=20.0, units='lbm/s',
                                    eq_units=None, rhs_name='nozz_PR_target')
                self.connect('balance.W',  'inlet.Fl_I:stat:W')
                self.connect('nozz.PR',    'balance.lhs:W')

                balance.add_balance('FAR', eq_units='degR',
                                    lower=1e-4, val=0.020, rhs_name='T4_target')
                self.connect('balance.FAR',       'burner.Fl_I:FAR')
                self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

                balance.add_balance('turb_PR', val=3.0, lower=1.001, upper=8,
                                    eq_units='hp', rhs_val=0.)
                self.connect('balance.turb_PR',  'turb.PR')
                self.connect('HP_shaft.pwr_net', 'balance.lhs:turb_PR')

                balance.add_balance('pt_PR', val=3.0, lower=1.001, upper=8,
                                    eq_units='hp', rhs_name='pwr_target')
                self.connect('balance.pt_PR',    'pt.PR')
                self.connect('LP_shaft.pwr_net', 'balance.lhs:pt_PR')

            else:
                # Physics combustor dP: orifice model feeds burner.dPqP each Newton step.
                self.add_subsystem('orifice', PhysicsCombustor())
                self.connect('comp.Fl_O:tot:T',   'orifice.Tt3')
                self.connect('comp.Fl_O:tot:P',   'orifice.Pt3')
                self.connect('inlet.Fl_O:stat:W', 'orifice.W')
                self.connect('orifice.dPqP',      'burner.dPqP')

                balance.add_balance('FAR', eq_units='hp',
                                    lower=1e-4, val=0.020, rhs_name='pwr_target')
                self.connect('balance.FAR',      'burner.Fl_I:FAR')
                self.connect('LP_shaft.pwr_net', 'balance.lhs:FAR')

                balance.add_balance('HP_Nmech', val=_HP_N_0, units='rpm',
                                    lower=500., eq_units='hp', rhs_val=0.)
                self.connect('balance.HP_Nmech', 'HP_Nmech')
                self.connect('HP_shaft.pwr_net', 'balance.lhs:HP_Nmech')

                balance.add_balance('W', val=20.0, units='lbm/s',
                                    eq_units='inch**2')
                self.connect('balance.W',             'inlet.Fl_I:stat:W')
                self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

            if design:
                self.set_order(['fc', 'inlet', 'comp', 'bld3', 'burner', 'turb', 'pt',
                                'nozz', 'HP_shaft', 'LP_shaft', 'perf',
                                'ngv_calcs', 'bld_calcs', 'cool_fracs', 'balance'])
            else:
                self.set_order(['fc', 'inlet', 'comp', 'bld3', 'orifice', 'burner', 'turb', 'pt',
                                'nozz', 'HP_shaft', 'LP_shaft', 'perf',
                                'ngv_calcs', 'bld_calcs', 'cool_fracs', 'balance'])

            newton = self.nonlinear_solver = om.NewtonSolver()
            newton.options['atol']                        = 1e-6
            newton.options['rtol']                        = 1e-6
            newton.options['iprint']                      = 2
            newton.options['maxiter']                     = 30 if design else 50
            newton.options['solve_subsystems']            = True
            newton.options['max_sub_solves']              = 100 if design else 200
            newton.options['reraise_child_analysiserror'] = False

            ls = newton.linesearch = om.ArmijoGoldsteinLS()
            ls.options['iprint']   = -1
            ls.options['maxiter']  = 3
            ls.options['rho']      = 0.75

            self.linear_solver = om.DirectSolver()
            super().setup()

    return SingleSpoolTurboshaft


def _make_mp_turboshaft(cfg):
    """Factory: produce an MPCycle class configured from cfg."""
    Turboshaft = _make_turboshaft(cfg)

    class MPTurboshaft(pyc.MPCycle):
        def setup(self):
            self.pyc_add_pnt('DESIGN', Turboshaft(design=True))
            self.set_input_defaults('DESIGN.HP_Nmech', cfg['HP_Nmech_init'], units='rpm')
            self.set_input_defaults('DESIGN.LP_Nmech', cfg['LP_Nmech'],      units='rpm')
            self.set_input_defaults('DESIGN.inlet.MN',  cfg.get('inlet_MN',  0.10))
            self.set_input_defaults('DESIGN.comp.MN',   cfg.get('comp_MN',   0.20))
            self.set_input_defaults('DESIGN.burner.MN', cfg.get('burner_MN', 0.20))
            self.set_input_defaults('DESIGN.turb.MN',   cfg.get('turb_MN',   0.40))

            # burner.dPqP: DESIGN value set in __main__; SWEEP driven by orifice.
            self.pyc_add_cycle_param('nozz.Cv',              cfg.get('nozz_Cv', 0.99))
            self.pyc_add_cycle_param('turb.ngv_cool:frac_P', 1.0)
            self.pyc_add_cycle_param('turb.bld_cool:frac_P', 0.0)
            # bld3.ngv_cool:frac_W / bld_cool:frac_W:
            #   DESIGN — driven by cool_fracs.frac_ngv/bld (internal Cycle connection)
            #   SWEEP  — set externally via prob.set_val() in solve_od() outer loop

            self.pyc_add_pnt('SWEEP', Turboshaft(design=False))
            self.set_input_defaults('SWEEP.fc.alt',   DESIGN_ALT, units='ft')
            self.set_input_defaults('SWEEP.fc.MN',    DESIGN_MN)
            self.set_input_defaults('SWEEP.LP_Nmech', cfg['LP_Nmech'], units='rpm')

            for sc in ['s_PR', 's_Wc', 's_eff', 's_Nc']:
                self.connect(f'DESIGN.comp.{sc}', f'SWEEP.comp.{sc}')
            for sc in ['s_PR', 's_eff', 's_Np']:
                self.connect(f'DESIGN.turb.{sc}', f'SWEEP.turb.{sc}')
                self.connect(f'DESIGN.pt.{sc}',   f'SWEEP.pt.{sc}')

            self.connect('DESIGN.nozz.Throat:stat:area', 'SWEEP.balance.rhs:W')

            for station, inp in [
                ('comp.Fl_O:stat:area',   'comp.area'),
                ('burner.Fl_O:stat:area', 'burner.area'),
                ('turb.Fl_O:stat:area',   'turb.area'),
            ]:
                self.connect(f'DESIGN.{station}', f'SWEEP.{inp}')

            super().setup()

    return MPTurboshaft


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get(prob, path, units=None):
    return prob.get_val(path, units=units)[0] if units else prob.get_val(path)[0]


def _std_atm(alt_ft):
    Tt = 518.67 - 3.5662e-3 * alt_ft
    Pt = 14.696 * (Tt / 518.67) ** 5.2559
    return Pt, Tt


def _safe_warm(r, baseline, tol=0.3):
    """Accept r as warm-start only if converged and on same branch as baseline."""
    if not r.get('converged'):
        return baseline
    if abs(r.get('OPR', 0) - baseline['OPR']) / max(baseline['OPR'], 1e-6) > tol:
        return baseline
    if r.get('turb_PR', 0) < 1.5:
        return baseline
    return r


def _estimate_W(des, pwr_hp):
    """Rough inlet flow estimate for warm-start seeding."""
    return float(np.clip(des['W'] * pwr_hp / des['SHP'], 0.20 * des['W'], 1.50 * des['W']))


def is_physical(r):
    try:
        return (3.0   < r['OPR']      < 200.0   and
                1000. < r['HP_Nmech'] < 100000.  and
                1.05  < r['turb_PR']  < 15.0     and
                1.05  < r['pt_PR']    < 15.0     and
                200.  < r['T3'] * RANKINE_TO_K < 1200. and
                r['SHP'] > 0. and r['PSFC'] > 0. and
                r['T4']  > 0. and r['T3']   > 0.)
    except (KeyError, TypeError):
        return False


def nan_result():
    r = {k: np.nan for k in ['W', 'OPR', 'T4', 'T4_exit', 'T3', 'P3', 'SHP', 'PSFC',
                               'Wf', 'HP_Nmech', 'turb_PR', 'pt_PR', 'FAR', 'dPqP',
                               'T_metal_ngv', 'T_metal_blade',
                               'comp_Wc', 'comp_PR']}
    r['converged'] = False
    return r


def extract_design(prob):
    _W           = _get(prob, 'DESIGN.inlet.Fl_O:stat:W', 'lbm/s')
    _comp_T_in   = _get(prob, 'DESIGN.inlet.Fl_O:tot:T',  'degR')
    _comp_P_in   = _get(prob, 'DESIGN.inlet.Fl_O:tot:P',  'psi')
    _comp_P_out  = _get(prob, 'DESIGN.comp.Fl_O:tot:P',   'psi')
    _comp_Wc     = _W * np.sqrt(_comp_T_in / 518.67) / (_comp_P_in / 14.696)
    _comp_PR     = _comp_P_out / max(_comp_P_in, 0.001)
    return {
        'W':         _W,
        'OPR':       _get(prob, 'DESIGN.perf.OPR'),
        'T4':        _get(prob, 'DESIGN.burner.Fl_O:tot:T',  'degR'),
        'T4_exit':   _get(prob, 'DESIGN.turb.Fl_O:tot:T',    'degR'),
        'T3':        _get(prob, 'DESIGN.comp.Fl_O:tot:T',    'degR'),
        'P3':        _comp_P_out,
        'SHP':       _get(prob, 'DESIGN.LP_shaft.pwr_net',   'hp'),
        'PSFC':      _get(prob, 'DESIGN.perf.PSFC'),
        'Wf':        _get(prob, 'DESIGN.burner.Wfuel',       'lbm/s'),
        'HP_Nmech':  _get(prob, 'DESIGN.HP_Nmech',           'rpm'),
        'turb_PR':   _get(prob, 'DESIGN.balance.turb_PR'),
        'pt_PR':     _get(prob, 'DESIGN.balance.pt_PR'),
        'FAR':       _get(prob, 'DESIGN.balance.FAR'),
        'turb_s_Wp': _get(prob, 'DESIGN.turb.s_Wp'),
        'pt_s_Wp':   _get(prob, 'DESIGN.pt.s_Wp'),
        'nozz_area': _get(prob, 'DESIGN.nozz.Throat:stat:area', 'inch**2'),
        'comp_Wc':   _comp_Wc,
        'comp_PR':   _comp_PR,
    }


def compute_CdA(des):
    """Back-calculate orifice CdA from design-point flow state at DPQP_DESIGN."""
    return calc_CdA_liner(
        Tt3_K  = des['T3'] * RANKINE_TO_K,
        Pt3_Pa = des['P3'] * PSI_TO_PA,
        mdot_si= des['W']  * LBM_S_TO_KG_S,
        dPqP_ref = DPQP_DESIGN,
    )


_LHV_JETA = 43.2e6   # J/kg, Jet-A lower heating value

def extract_od(prob):
    T4_R    = _get(prob, 'SWEEP.burner.Fl_O:tot:T', 'degR')
    T4ex_R  = _get(prob, 'SWEEP.turb.Fl_O:tot:T',   'degR')
    T3_R    = _get(prob, 'SWEEP.comp.Fl_O:tot:T',   'degR')
    Tmet_ngv   = t_metal_rankine(T4_R,  T3_R) * RANKINE_TO_K
    T0_rel_R   = turb_rel_temp_rankine(T4_R, T4ex_R)
    Tmet_blade = t_metal_rankine(T0_rel_R, T3_R) * RANKINE_TO_K
    _W           = _get(prob, 'SWEEP.inlet.Fl_O:stat:W',  'lbm/s')
    _comp_T_in   = _get(prob, 'SWEEP.inlet.Fl_O:tot:T',   'degR')
    _comp_P_in   = _get(prob, 'SWEEP.inlet.Fl_O:tot:P',   'psi')
    _comp_P_out  = _get(prob, 'SWEEP.comp.Fl_O:tot:P',    'psi')
    _comp_Wc     = _W * np.sqrt(_comp_T_in / 518.67) / (_comp_P_in / 14.696)
    _comp_PR     = _comp_P_out / max(_comp_P_in, 0.001)
    _SHP         = _get(prob, 'SWEEP.LP_shaft.pwr_net', 'hp')
    _Wf          = _get(prob, 'SWEEP.burner.Wfuel',     'lbm/s')
    _Wf_kg       = _Wf * LBM_S_TO_KG_S
    _eta_th      = (_SHP * 745.7) / (_Wf_kg * _LHV_JETA) if _Wf_kg > 0 else np.nan
    return {
        'W':           _W,
        'OPR':         _get(prob, 'SWEEP.perf.OPR'),
        'T4':          T4_R,
        'T4_exit':     T4ex_R,
        'T3':          T3_R,
        'P3':          _comp_P_out,
        'SHP':         _SHP,
        'PSFC':        _get(prob, 'SWEEP.perf.PSFC'),
        'Wf':          _Wf,
        'eta_th':      _eta_th,
        'HP_Nmech':    _get(prob, 'SWEEP.balance.HP_Nmech'),
        'turb_PR':     _get(prob, 'SWEEP.turb.PR'),
        'pt_PR':       _get(prob, 'SWEEP.pt.PR'),
        'FAR':         _get(prob, 'SWEEP.balance.FAR'),
        'dPqP':        _get(prob, 'SWEEP.orifice.dPqP'),
        'T_metal_ngv':   Tmet_ngv,
        'T_metal_blade': Tmet_blade,
        'comp_Wc':       _comp_Wc,
        'comp_PR':       _comp_PR,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  OD SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_od(prob, des, turb_s_Wp, pt_s_Wp, pwr_hp, warm=None, label=''):
    """
    Solve SWEEP at given turbine area scalars and power target.
    Returns result dict with 'converged' bool.
    """
    src   = warm if warm is not None else des
    Pt, Tt = _std_atm(DESIGN_ALT)

    prob['SWEEP.balance.FAR']      = src.get('FAR',  des['FAR'])
    prob['SWEEP.balance.W']        = _estimate_W(des, pwr_hp)
    prob['SWEEP.fc.balance.Pt']    = Pt
    prob['SWEEP.fc.balance.Tt']    = Tt

    # Physics-informed PR seeds
    _a4_ratio  = turb_s_Wp / max(des['turb_s_Wp'], 1e-9)
    _a45_ratio = pt_s_Wp   / max(des['pt_s_Wp'],   1e-9)
    if src is des:
        _turb_PR_seed = float(np.clip(des['turb_PR'], 1.05, 8.0))
        _pt_PR_seed   = float(np.clip(des['pt_PR'],   1.05, 8.0))
    else:
        _turb_PR_seed = float(np.clip(src.get('turb_PR', des['turb_PR']) / _a4_ratio,  1.05, 8.0))
        _pt_PR_seed   = float(np.clip(src.get('pt_PR',   des['pt_PR'])   / _a45_ratio, 1.05, 8.0))

    prob['SWEEP.turb.PR']          = _turb_PR_seed
    prob['SWEEP.pt.PR']            = _pt_PR_seed
    _N_seed = float(np.clip(
        src.get('HP_Nmech', des['HP_Nmech']) * (_turb_PR_seed / max(des['turb_PR'], 1e-6)) ** 0.5,
        5000., 40000.))
    prob['SWEEP.balance.HP_Nmech'] = _N_seed

    prob.set_val('SWEEP.turb.s_Wp', turb_s_Wp)
    prob.set_val('SWEEP.pt.s_Wp',   pt_s_Wp)
    prob.set_val('SWEEP.balance.pwr_target', pwr_hp, units='hp')

    # Inject design-calibrated CdA so the endogenous orifice model uses the
    # correct liner effective area for this combustor geometry.
    if des.get('CdA_liner'):
        prob.set_val('SWEEP.orifice.CdA', des['CdA_liner'])

    # Seed inter-stage temperatures only when starting from design (no prior OD result).
    # If src is a converged OD result, the prob state is already near the solution;
    # overwriting it with design-point temperatures would break convergence.
    if src is des:
        _gam, _eta = 1.33, 0.87
        _Tt_in  = Tt
        _Pt_in  = Pt
        _Tt_c3  = _Tt_in * des['OPR'] ** ((_gam - 1) / (_gam * _eta))
        _Pt_c3  = _Pt_in * des['OPR']
        # T4 must always exceed T3; T4 is a balance output at part power
        _Tt_t4  = max(des['T4'] * RANKINE_TO_K, _Tt_c3 + 300.0)
        _Pt_t4  = _Pt_c3 * 0.97
        _isTt45 = _Tt_t4  * (_turb_PR_seed ** ((1 - _gam) / _gam))
        _Tt_t45 = max(_Tt_t4 - _eta * (_Tt_t4 - _isTt45), 300.)
        _Pt_t45 = max(_Pt_t4 / _turb_PR_seed, 0.1)
        _isTt5  = _Tt_t45 * (_pt_PR_seed ** ((1 - _gam) / _gam))
        _Tt_t5  = max(_Tt_t45 - _eta * (_Tt_t45 - _isTt5), 250.)
        _Pt_t5  = max(_Pt_t45 / _pt_PR_seed, 0.05)
        for _stn, _Tt_est, _Pt_est in [
                ('comp.Fl_O',   _Tt_c3,  _Pt_c3),
                ('burner.Fl_O', _Tt_t4,  _Pt_t4),
                ('turb.Fl_O',   _Tt_t45, _Pt_t45),
                ('pt.Fl_O',     _Tt_t5,  _Pt_t5),
                ('nozz.Fl_I',   _Tt_t5,  _Pt_t5),
        ]:
            try:
                prob.set_val(f'SWEEP.{_stn}:stat:T', max(0.85 * _Tt_est, 200.), units='K')
                prob.set_val(f'SWEEP.{_stn}:stat:P', max(_Pt_est * 0.55,  0.5), units='psi')
                # Also seed tot:T/P — CoolingCalcs reads burner.Fl_O:tot:T as Tt_primary.
                # Without this, CoolingCalcs sees ambient T4 on the first Newton pass,
                # computes max W_cool (~29%/row), bleeds off most flow, and Newton collapses.
                prob.set_val(f'SWEEP.{_stn}:tot:T', max(_Tt_est, 200.), units='K')
                prob.set_val(f'SWEEP.{_stn}:tot:P', max(_Pt_est, 0.5),  units='psi')
            except (KeyError, AttributeError):
                pass
        # Seed bld3 exit too (Tt_cool input to CoolingCalcs)
        try:
            prob.set_val('SWEEP.bld3.Fl_O:tot:T', _Tt_c3, units='K')
            prob.set_val('SWEEP.bld3.Fl_O:tot:P', _Pt_c3, units='psi')
        except (KeyError, AttributeError):
            pass
        # Reset compressor internal temperature balances.
        # Failed Brent probes can leave comp.ideal_flow.balance.T at -2.59e8,
        # which Newton cannot recover from even with good station-T seeds above.
        _T3_ideal = _Tt_in * des['OPR'] ** ((_gam - 1) / _gam)
        try:
            prob.set_val('SWEEP.comp.ideal_flow.balance.T',
                         float(np.clip(_T3_ideal, 300., 2499.)), units='K')
            prob.set_val('SWEEP.comp.real_flow.balance.T',
                         float(np.clip(_Tt_c3,    300., 2499.)), units='K')
        except (KeyError, AttributeError):
            pass

    # Outer cooling fixed-point loop.
    # CoolingCalcs is NOT Newton-coupled in OD (see _make_turboshaft comments):
    # the T4↑→frac↑→W↓→power↓→FAR↑→T4↑ loop destabilises the two-spool balance.
    # Instead: fix fracs during each Newton solve, then update fracs from CoolingCalcs
    # output after convergence. 2-3 outer iterations typically converge to < 0.01%.
    _frac_ngv_cur = src.get('ngv_cool_frac', des.get('ngv_cool_frac', 0.05))
    _frac_bld_cur = src.get('bld_cool_frac', des.get('bld_cool_frac', 0.03))

    prob.model._get_subsystem('SWEEP').nonlinear_solver.options['maxiter'] = 30

    r, converged = nan_result(), False
    for _outer in range(3):
        # Fix fracs for this Newton solve (bld3 in OD is unconnected — driven by set_val)
        prob.set_val('SWEEP.bld3.ngv_cool:frac_W', _frac_ngv_cur)
        prob.set_val('SWEEP.bld3.bld_cool:frac_W', _frac_bld_cur)
        try:
            prob.run_model()
            r = extract_od(prob)
            converged = is_physical(r)
            if not converged:
                print(f'  {label}  [outer {_outer}] unphysical '
                      f'(OPR={r["OPR"]:.2f} HP_N={r["HP_Nmech"]:.0f} '
                      f'tPR={r["turb_PR"]:.2f} pPR={r["pt_PR"]:.2f})')
                break
        except Exception as e:
            print(f'  {label}  [outer {_outer}] exception — {type(e).__name__}: {str(e)[:60]}')
            break

        # Read CoolingCalcs output (computed from converged T4/T3) and update fracs
        _frac_ngv_new = float(prob.get_val('SWEEP.cool_fracs.frac_ngv')[0])
        _frac_bld_new = float(prob.get_val('SWEEP.cool_fracs.frac_bld')[0])
        _delta = abs(_frac_ngv_new - _frac_ngv_cur) + abs(_frac_bld_new - _frac_bld_cur)
        # Damped update to keep fracs physically bounded
        _frac_ngv_cur = 0.5 * _frac_ngv_new + 0.5 * _frac_ngv_cur
        _frac_bld_cur = 0.5 * _frac_bld_new + 0.5 * _frac_bld_cur
        if _delta < 2e-4:
            break  # fracs converged — no further outer iterations needed

    # Second-chance retry with loose tolerances
    if not converged and '[retry' not in label:
        _subsys = prob.model._get_subsystem('SWEEP')
        _orig = (_subsys.nonlinear_solver.options['atol'],
                 _subsys.nonlinear_solver.options['rtol'],
                 _subsys.nonlinear_solver.options['maxiter'])
        _subsys.nonlinear_solver.options['atol']    = 1e-4
        _subsys.nonlinear_solver.options['rtol']    = 1e-4
        _subsys.nonlinear_solver.options['maxiter'] = 50
        prob['SWEEP.balance.FAR']      = warm['FAR']      if warm else des['FAR']
        prob['SWEEP.balance.W']        = warm['W']        if warm else des['W']
        prob['SWEEP.balance.HP_Nmech'] = warm['HP_Nmech'] if warm else des['HP_Nmech']
        prob.set_val('SWEEP.bld3.ngv_cool:frac_W', _frac_ngv_cur)
        prob.set_val('SWEEP.bld3.bld_cool:frac_W', _frac_bld_cur)
        try:
            prob.run_model()
            r2 = extract_od(prob)
            if is_physical(r2):
                r, converged = r2, True
                print(f'  {label}  recovered with loose tolerances')
        except Exception:
            pass
        _subsys.nonlinear_solver.options['atol']    = _orig[0]
        _subsys.nonlinear_solver.options['rtol']    = _orig[1]
        _subsys.nonlinear_solver.options['maxiter'] = _orig[2]

    # Store converged fracs in result for warm-start chaining
    if converged:
        r['ngv_cool_frac'] = float(prob.get_val('SWEEP.cool_fracs.frac_ngv')[0])
        r['bld_cool_frac'] = float(prob.get_val('SWEEP.cool_fracs.frac_bld')[0])

    r['converged'] = converged
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  VGT BRENT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def opt_a4(prob, des, pwr_hp, warm=None):
    """
    Find A4 scalar minimising PSFC at fixed pwr_hp.
    Returns optimal a4_sc ∈ A4_BOUNDS.
    T4 redline hard-enforced: returns inf if T4 > design TIT.

    Pre-solves at a4=1.0 before Brent so the prob state is in the right
    thermodynamic regime for the current power level.  Warm-state updates
    from each converged probe so adjacent Brent evaluations chain together.
    """
    # At design power A4 = 1.0 by definition — the T4 redline is exactly binding
    # and Brent would return a spurious opening (a4 > 1) with no real benefit.
    if abs(pwr_hp / max(des['SHP'], 1e-6) - 1.0) < 0.005:
        _r = solve_od(prob, des,
                      turb_s_Wp = des['turb_s_Wp'],
                      pt_s_Wp   = des['pt_s_Wp'],
                      pwr_hp    = pwr_hp,
                      warm      = warm if warm is not None else des,
                      label     = f'  a4=1.0000 [design power]')
        return 1.0, _r

    # Pre-solve at design A4 for this power level.  Brent's first probe is
    # typically at a4 ≈ 1.035 (golden section); without this the prob state
    # would still be at the last fixed-sweep point (different power level)
    # and the jump causes Newton divergence.
    _ref = solve_od(prob, des,
                    turb_s_Wp = des['turb_s_Wp'],
                    pt_s_Wp   = des['pt_s_Wp'],
                    pwr_hp    = pwr_hp,
                    warm      = warm if warm is not None else des,
                    label     = f'  brent a4=1.0000 [ref]')
    _state = [_ref if _ref.get('converged') else (warm if warm is not None else des)]

    def _eval(a4_sc):
        r = solve_od(prob, des,
                     turb_s_Wp = des['turb_s_Wp'] * a4_sc,
                     pt_s_Wp   = des['pt_s_Wp'],
                     pwr_hp    = pwr_hp,
                     warm      = _state[0],
                     label     = f'  brent a4={a4_sc:.4f}')
        if r.get('converged'):
            _state[0] = r   # chain warm-start through Brent probes
        if not r['converged'] or r['T4'] > des['T4']:   # des['T4'] = design TIT redline
            return np.inf
        return r['PSFC']

    res = minimize_scalar(_eval, bounds=A4_BOUNDS, method='bounded',
                          options={'xatol': 1e-3})
    # Brent's last internal evaluation is NOT necessarily at res.x.
    # Re-solve at the optimal a4 so the prob state is clean for downstream use
    # (Step 4 warm-starts from whatever state the prob is left in after Step 3).
    _final = solve_od(prob, des,
                      turb_s_Wp = des['turb_s_Wp'] * res.x,
                      pt_s_Wp   = des['pt_s_Wp'],
                      pwr_hp    = pwr_hp,
                      warm      = _state[0],
                      label     = f'  brent a4={res.x:.4f} [final]')
    return float(res.x), _final


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

# Rogo & Benstein (1986) digitised reference curves
_ROGO_PWR = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]*100
_ROGO_84  = [0.560, 0.520, 0.490, 0.470, 0.455, 0.445, 0.435]  # best 1984 engine
_ROGO_VC  = [0.455, 0.435, 0.420, 0.415, 0.415, 0.418, 0.430]  # VARICAP


def plot_results(des, pwr_fracs, fixed_res, vgt_res, a4_sched):
    fn_pct   = np.array(pwr_fracs) * 100
    T4_des_K = des['T4'] * RANKINE_TO_K

    def _arr(results, key):
        return np.array([r[key] if r.get('converged') else np.nan for r in results])

    eta_fix   = _arr(fixed_res, 'eta_th') * 100
    eta_vgt   = _arr(vgt_res,   'eta_th') * 100
    T4_fix    = _arr(fixed_res, 'T4') * RANKINE_TO_K
    T4_vgt    = _arr(vgt_res,   'T4') * RANKINE_TO_K
    tPR_fix   = _arr(fixed_res, 'turb_PR')
    tPR_vgt   = _arr(vgt_res,   'turb_PR')
    pPR_fix   = _arr(fixed_res, 'pt_PR')
    pPR_vgt   = _arr(vgt_res,   'pt_PR')
    d_eta     = eta_vgt - eta_fix   # positive = VGT better

    # ─── Figure 1: thermal efficiency comparison ──────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig1.suptitle(
        'Turboshaft Variable A4 — Fixed vs VGT  (VARICAP mode, LP_Nmech = const)\n'
        f'SLS  |  OPR = {des["OPR"]:.1f}  |  Tt4_des = {T4_des_K:.0f} K  |  '
        f'SHP_des = {des["SHP"]:.0f} hp',
        fontsize=10, fontweight='bold')

    # Panel 1: η_th vs power fraction
    ax = axes[0]
    ax.plot(fn_pct, eta_fix, 'o-', color='steelblue', lw=2.2, ms=6, label='Fixed  (A₄ = 1.0)')
    ax.plot(fn_pct, eta_vgt, 's-', color='firebrick', lw=2.2, ms=6, label='VGT  (A₄ optimal)')
    ax.fill_between(fn_pct, eta_fix, eta_vgt,
                    where=(d_eta >= 0), alpha=0.12, color='firebrick', label='VGT benefit')
    ax.fill_between(fn_pct, eta_fix, eta_vgt,
                    where=(d_eta < 0),  alpha=0.12, color='steelblue', label='Fixed benefit')
    ax.axvline(100, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('η_th  (%)', fontsize=11)
    ax.set_title('Thermal Efficiency  (SHP / Q_fuel)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Δη_th improvement
    ax = axes[1]
    ax.plot(fn_pct, d_eta, '^-', color='C2', lw=2.4, ms=7)
    ax.axhline(0, color='k', lw=1.2, ls='--', alpha=0.6)
    ax.fill_between(fn_pct, d_eta, 0,
                    where=~np.isnan(d_eta), alpha=0.15, color='C2')
    for i in range(len(fn_pct)):
        if np.isfinite(d_eta[i]) and abs(d_eta[i]) > 0.03:
            ax.annotate(f'{d_eta[i]:+.2f}%',
                        xy=(fn_pct[i], d_eta[i]),
                        xytext=(0, 6 if d_eta[i] >= 0 else -10),
                        textcoords='offset points',
                        ha='center', fontsize=6.5, color='C2')
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('Δη_th  (%-points)  [VGT − Fixed]', fontsize=11)
    ax.set_title('VGT Thermal Efficiency Gain\n(positive = VGT better)', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3: Tt4 + redline
    ax = axes[2]
    ax.plot(fn_pct, T4_fix, 'o-', color='C0', lw=2.2, ms=6, label='Fixed')
    ax.plot(fn_pct, T4_vgt, 's-', color='C3', lw=2.2, ms=6, label='VGT')
    ax.axhline(T4_des_K, color='crimson', lw=2.0, ls='-',
               label=f'T₄ redline  {T4_des_K:.0f} K')
    ax.axhspan(T4_des_K, T4_des_K + 80, alpha=0.07, color='crimson')
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('Turbine Inlet Temperature  (K)', fontsize=11)
    ax.set_title('Tt4 vs Power Level', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig('fixed_vs_vgt_turboshaft.png', dpi=150, bbox_inches='tight')
    print('  → saved fixed_vs_vgt_turboshaft.png')
    plt.show()

    # ─── Figure 2: A4 schedule + work split ──────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle('VGT A4 Schedule and Turbine Work Split', fontsize=11, fontweight='bold')

    # Panel A: optimal A4 schedule (raw Brent results)
    ax = axes2[0]
    ax.plot(fn_pct, a4_sched, 'D-', color='C4', lw=2.2, ms=7, label='Optimal A4 (Brent)')
    ax.axhline(1.0, color='k', lw=1.2, ls='--', label='Design A4 = 1.0')
    mask_close = a4_sched < 1.0
    mask_open  = a4_sched > 1.0
    if mask_close.any():
        ax.fill_between(fn_pct, 1.0, a4_sched, where=mask_close,
                        alpha=0.15, color='C0', label='Close (<1) → const-TIT mode')
    if mask_open.any():
        ax.fill_between(fn_pct, 1.0, a4_sched, where=mask_open,
                        alpha=0.15, color='C3', label='Open  (>1) → const-OPR mode')
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('A4 scalar', fontsize=11)
    ax.set_title('Optimal A4 Schedule\n(Roy-Aikins 1990: const-OPR better for simple cycle)',
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: GGT PR and PT PR
    ax = axes2[1]
    ax.plot(fn_pct, tPR_fix, 'o--', color='C0', lw=1.8, ms=5,
            label='GGT PR — Fixed', alpha=0.8)
    ax.plot(fn_pct, tPR_vgt, 's-',  color='C0', lw=2.2, ms=5,
            label='GGT PR — VGT')
    ax2b = ax.twinx()
    ax2b.plot(fn_pct, pPR_fix, 'o--', color='C3', lw=1.8, ms=5,
              label='PT PR — Fixed', alpha=0.8)
    ax2b.plot(fn_pct, pPR_vgt, 's-',  color='C3', lw=2.2, ms=5,
              label='PT PR — VGT')
    ax2b.set_ylabel('PT Expansion Ratio  (right axis)', fontsize=10, color='C3')
    ax2b.tick_params(axis='y', labelcolor='C3')
    ax.set_xlabel('Power fraction  (%)', fontsize=11)
    ax.set_ylabel('GGT Expansion Ratio  (left axis)', fontsize=11)
    ax.set_title('Turbine Work Split\nGGT PR (solid/left) vs PT PR (solid/right)', fontsize=10)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig('vgt_schedule_worksplit_turboshaft.png', dpi=150, bbox_inches='tight')
    print('  → saved vgt_schedule_worksplit_turboshaft.png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAP PLOTTING (adapted from combustor_py_cycle.py)
# ══════════════════════════════════════════════════════════════════════════════

def draw_compressor_map_background(
        ax, prob, e_name, alpha_idx=0,
        eff_vals=np.array([0, 0.50, 0.55, 0.60, 0.65,
                           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]),
        show_rlines=True, show_nclines=True, eff_alpha=1.0):
    """Draw compressor map efficiency fill, speed lines, R-lines, and stall line."""
    comp     = prob.model._get_subsystem(e_name)
    map_data = comp.options['map_data']

    s_Wc  = prob.get_val(f'{e_name}.s_Wc')[0]
    s_PR  = prob.get_val(f'{e_name}.s_PR')[0]
    s_eff = prob.get_val(f'{e_name}.s_eff')[0]
    s_Nc  = prob.get_val(f'{e_name}.s_Nc')[0]

    RlineMap, NcMap = np.meshgrid(map_data.RlineMap, map_data.NcMap, sparse=False)
    a       = alpha_idx
    Wc_map  = map_data.WcMap[a, :, :]  * s_Wc
    PR_map  = (map_data.PRmap[a, :, :] - 1.0) * s_PR + 1.0
    eff_map = map_data.effMap[a, :, :] * s_eff
    Nc_map  = NcMap * s_Nc

    eff_cf = ax.contourf(Wc_map, PR_map, eff_map,
                         levels=eff_vals, cmap='viridis',
                         alpha=eff_alpha, zorder=1)

    eff_edge = eff_vals[eff_vals > 0]
    eff_cs = ax.contour(Wc_map, PR_map, eff_map,
                        levels=eff_edge, colors='k', linewidths=0.6, zorder=2)
    ax.clabel(eff_cs, fmt='%.2f', fontsize=6, inline=True, inline_spacing=2)

    if show_nclines:
        nc_cs = ax.contour(Wc_map, PR_map, Nc_map,
                           levels=map_data.NcMap * s_Nc,
                           colors='k', linewidths=0.8, linestyles='solid', zorder=2)
        ax.clabel(nc_cs, fmt='%.2f', fontsize=6, inline=True, inline_spacing=2)

    if show_rlines:
        r_cs = ax.contour(Wc_map, PR_map, RlineMap,
                          levels=map_data.RlineMap,
                          colors='k', linewidths=0.6, linestyles='solid',
                          alpha=1.0, zorder=2)
        ax.clabel(r_cs, fmt='%.2f', fontsize=6, inline=True, inline_spacing=2)

    stall_Wc = Wc_map[:, 0]
    stall_PR = PR_map[:, 0]
    idx      = np.argsort(stall_Wc)
    ax.plot(stall_Wc[idx], stall_PR[idx], 'r-', lw=2.0, zorder=3, label='Stall line')
    return eff_cf


def draw_hpt_map_background(ax, turb_map, alpha_idx,
                             s_Np=1.0, s_PR=1.0, s_eff=1.0,
                             eff_vals=None):
    """
    Draw HPT efficiency contours in physical (PR_actual, Np_actual) space.

    Scaling mirrors pyCycle turbine_map.py convention:
        Np_actual = NpMap * s_Np
        PR_actual = (PRmap - 1) * s_PR + 1
    So operating points plotted at (turb_PR, N/sqrt(T4)) will align with
    the contour background.
    """
    if eff_vals is None:
        eff_vals = np.linspace(0.70, 0.96, 20)

    NpMap  = np.array(turb_map.NpMap)
    PRmap  = np.array(turb_map.PRmap)
    effArr = np.array(turb_map.effMap)[alpha_idx, :, :] * s_eff  # [Np_idx, PR_idx]

    # Scale to physical coordinates (identical convention to compressor)
    Np_phys = NpMap * s_Np
    PR_phys = (PRmap - 1.0) * s_PR + 1.0

    PR_grid, Np_grid = np.meshgrid(PR_phys, Np_phys)

    eff_cf = ax.contourf(PR_grid, Np_grid, effArr,
                         levels=eff_vals, cmap='plasma', alpha=0.85, zorder=1)
    eff_cs = ax.contour(PR_grid, Np_grid, effArr,
                        levels=eff_vals[::4], colors='k', linewidths=0.6, zorder=2)
    ax.clabel(eff_cs, fmt='%.3f', fontsize=6, inline=True, inline_spacing=2)

    return eff_cf, float(Np_phys.min()), float(Np_phys.max()), \
                   float(PR_phys.min()), float(PR_phys.max())


def plot_turboshaft_maps(prob, des, fixed_res, vgt_res, a4_sched, pwr_fracs):
    """
    2-panel map diagnostic: AXI5 compressor map + EngineHPT turbine map.
    Fixed geometry shown as circles, VGT (Step-4 re-solve) as diamonds.
    Colour encodes power fraction (green=100%, red=50%).
    """
    T4_des_K = des['T4'] * RANKINE_TO_K
    fn_pct   = np.array(pwr_fracs) * 100
    cmap_pwr = plt.cm.RdYlGn

    fig, axes = plt.subplots(1, 2, figsize=(17, 8))
    fig.suptitle(
        'Turboshaft Map Operating Lines — Fixed  ○  vs VGT  ◇\n'
        f'AXI5 Compressor  |  EngineHPT GGT  |  '
        f'Design: SLS  OPR={des["OPR"]:.1f}  Tt4={T4_des_K:.0f} K  '
        f'SHP={des["SHP"]:.0f} hp',
        fontsize=11, fontweight='bold')

    # ─── Panel 1: Compressor map ─────────────────────────────────────────────
    ax = axes[0]
    eff_cf = draw_compressor_map_background(
        ax, prob, 'DESIGN.comp',
        alpha_idx=0,
        eff_vals=np.linspace(0.60, 0.88, 20),
        show_rlines=False, show_nclines=True, eff_alpha=0.85)
    cb1 = fig.colorbar(eff_cf, ax=ax, shrink=0.70, pad=0.02)
    cb1.set_label('Compressor  ηc  (—)', fontsize=9)

    # Design point
    ax.plot(des['comp_Wc'], des['comp_PR'],
            '*', ms=18, color='gold', markeredgecolor='k',
            markeredgewidth=1.2, zorder=10, label='Design point')

    # Operating lines: fixed (circles) and VGT (diamonds), coloured by power
    for i, pf in enumerate(pwr_fracs):
        color = cmap_pwr((pf - min(pwr_fracs)) / max(max(pwr_fracs) - min(pwr_fracs), 1e-9))
        fr = fixed_res[i]
        vr = vgt_res[i]
        if fr.get('converged') and np.isfinite(fr.get('comp_Wc', np.nan)):
            ax.plot(fr['comp_Wc'], fr['comp_PR'],
                    'o', ms=9, color=color, markeredgecolor='k',
                    markeredgewidth=0.7, zorder=8)
            ax.annotate(f'{pf*100:.0f}%',
                        xy=(fr['comp_Wc'], fr['comp_PR']),
                        xytext=(fr['comp_Wc'] - 0.3, fr['comp_PR'] + 0.08),
                        fontsize=6, color='white', zorder=11,
                        arrowprops=dict(arrowstyle='-', color='white', lw=0.4))
        if vr.get('converged') and np.isfinite(vr.get('comp_Wc', np.nan)):
            ax.plot(vr['comp_Wc'], vr['comp_PR'],
                    'D', ms=8, color=color, markeredgecolor='navy',
                    markeredgewidth=0.8, zorder=9)

    # Connect converged fixed points with a line (operating line)
    pairs = [(pwr_fracs[i], fixed_res[i]['comp_Wc'], fixed_res[i]['comp_PR'])
             for i in range(len(pwr_fracs))
             if fixed_res[i].get('converged') and np.isfinite(fixed_res[i].get('comp_Wc', np.nan))]
    if len(pairs) > 1:
        pairs.sort(key=lambda x: x[0])
        _, wc_line, pr_line = zip(*pairs)
        ax.plot(wc_line, pr_line, '--', color='white', lw=1.5, zorder=7,
                alpha=0.6, label='Fixed op. line')

    ax.set_xlabel('Corrected Mass Flow  Wc  (lbm/s)', fontsize=11)
    ax.set_ylabel('Compressor Pressure Ratio  πc  (—)', fontsize=11)
    ax.set_title('AXI5 Compressor Map\n○ = Fixed   ◇ = VGT   colour = power level', fontsize=10)

    sm1 = plt.cm.ScalarMappable(cmap=cmap_pwr,
                                 norm=plt.Normalize(min(fn_pct), max(fn_pct)))
    sm1.set_array([])
    cb_pwr = fig.colorbar(sm1, ax=ax, shrink=0.45, pad=0.13)
    cb_pwr.set_label('Power fraction  (%)', fontsize=9)

    ax.legend(fontsize=7.5, loc='upper left',
              facecolor='k', labelcolor='white', edgecolor='white', framealpha=0.8)
    ax.grid(True, alpha=0.15, color='white', zorder=0)

    # ─── Panel 2: GGT (EngineHPT) turbine map ───────────────────────────────
    ax = axes[1]

    ggt_sys  = prob.model._get_subsystem('DESIGN.turb')
    turb_map = ggt_sys.options['map_data']
    # Calibrated scalars from design solve — same convention as compressor
    s_Np_t  = float(prob.get_val('DESIGN.turb.s_Np')[0])
    s_PR_t  = float(prob.get_val('DESIGN.turb.s_PR')[0])
    s_eff_t = float(prob.get_val('DESIGN.turb.s_eff')[0])

    _alpha_arr     = np.array(turb_map.alphaMap)
    _alpha_des_idx = int(np.argmin(np.abs(_alpha_arr - 1.0)))
    _alpha_des_val = _alpha_arr[_alpha_des_idx]

    eff_cf2, _Np_min, _Np_max, _PR_min, _PR_max = draw_hpt_map_background(
        ax, turb_map, alpha_idx=_alpha_des_idx,
        s_Np=s_Np_t, s_PR=s_PR_t, s_eff=s_eff_t)
    cb2 = fig.colorbar(eff_cf2, ax=ax, shrink=0.70, pad=0.02)
    cb2.set_label('HPT efficiency  ηt  (—)', fontsize=9)

    # Design point in physical coordinates: Np = N / sqrt(T4_R)
    Np_des = des['HP_Nmech'] / np.sqrt(des['T4'])
    ax.plot(des['turb_PR'], Np_des,
            '*', ms=18, color='gold', markeredgecolor='k',
            markeredgewidth=1.2, zorder=10, label='Design point')

    for i, pf in enumerate(pwr_fracs):
        color = cmap_pwr((pf - min(pwr_fracs)) / max(max(pwr_fracs) - min(pwr_fracs), 1e-9))
        fr = fixed_res[i]
        vr = vgt_res[i]
        if fr.get('converged') and fr.get('T4', 0) > 0:
            Np_fix = fr['HP_Nmech'] / np.sqrt(max(fr['T4'], 1.0))
            ax.plot(fr['turb_PR'], Np_fix,
                    'o', ms=9, color=color, markeredgecolor='k',
                    markeredgewidth=0.7, zorder=8)
            if fr['turb_PR'] < _PR_min or fr['turb_PR'] > _PR_max or \
               Np_fix < _Np_min or Np_fix > _Np_max:
                ax.annotate('OOB', xy=(fr['turb_PR'], Np_fix),
                             fontsize=6, color='red', zorder=11)
        if vr.get('converged') and vr.get('T4', 0) > 0:
            Np_vgt = vr['HP_Nmech'] / np.sqrt(max(vr['T4'], 1.0))
            ax.plot(vr['turb_PR'], Np_vgt,
                    'D', ms=8, color=color, markeredgecolor='navy',
                    markeredgewidth=0.8, zorder=9)

    # Annotate power fractions on fixed line
    for i, pf in enumerate(pwr_fracs):
        fr = fixed_res[i]
        if fr.get('converged') and fr.get('T4', 0) > 0:
            Np_fix = fr['HP_Nmech'] / np.sqrt(max(fr['T4'], 1.0))
            ax.annotate(f'{pf*100:.0f}%',
                        xy=(fr['turb_PR'], Np_fix),
                        xytext=(fr['turb_PR'] + 0.05, Np_fix + 1.5),
                        fontsize=6, color='white', zorder=11)

    # Physical map boundary (dashed)
    ax.axvline(_PR_min, color='white', lw=1.0, ls=':', alpha=0.6)
    ax.axvline(_PR_max, color='white', lw=1.0, ls=':', alpha=0.6)
    ax.axhline(_Np_min, color='white', lw=1.0, ls=':', alpha=0.6)
    ax.axhline(_Np_max, color='white', lw=1.0, ls=':', alpha=0.6)

    ax.set_xlabel('Expansion Ratio  PR  (—)', fontsize=11)
    ax.set_ylabel('Corrected Speed  Np = N/√Tt4  (rpm / °R½)', fontsize=11)
    ax.set_title(f'EngineHPT Turbine Map   α = {_alpha_des_val:.2f}  (design)\n'
                 f's_Np = {s_Np_t:.3f}   s_PR = {s_PR_t:.3f}   '
                 f'dashed = map boundary', fontsize=10)

    sm2 = plt.cm.ScalarMappable(cmap=cmap_pwr,
                                 norm=plt.Normalize(min(fn_pct), max(fn_pct)))
    sm2.set_array([])
    cb_pwr2 = fig.colorbar(sm2, ax=ax, shrink=0.45, pad=0.13)
    cb_pwr2.set_label('Power fraction  (%)', fontsize=9)

    ax.legend(fontsize=7.5, loc='upper right',
              facecolor='k', labelcolor='white', edgecolor='white', framealpha=0.8)
    ax.grid(True, alpha=0.15, color='white', zorder=0)

    fig.tight_layout()
    fig.savefig('turboshaft_map_oplines.png', dpi=150, bbox_inches='tight')
    print('  → saved turboshaft_map_oplines.png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  ENGINE CONFIG                                               ║
    # ╚══════════════════════════════════════════════════════════════╝

    # Map design-point efficiencies (Np = N/sqrt(Tt4_degR))
    _TIT_R_des  = 2370.0
    _HP_N_des   = 8070.0
    _Np_des_GGT = _HP_N_des / np.sqrt(_TIT_R_des)

    eff_vgt, PR_vgt   = map_design_point(EngineHPTVGTMap,   _Np_des_GGT)
    eff_fixed, PR_fixed = map_design_point(EngineHPTFixedMap, _Np_des_GGT)

    print('=' * 65)
    print('  Map design-point extraction')
    print(f'  VGT  map:  η={eff_vgt*100:.3f}%   PR={PR_vgt:.4f}')
    print(f'  Fixed map: η={eff_fixed*100:.3f}%   PR={PR_fixed:.4f}')
    print(f'  Leakage penalty: Δη={(eff_fixed-eff_vgt)*100:+.3f}%')
    print('=' * 65)

    # Gauntner cooling pre-compute (analytical T3 estimate at design)
    _OPR_des  = 13.5
    _comp_eff = 0.83
    _T3_est_R = estimate_T3_rankine(_OPR_des, comp_eff=_comp_eff)
    _frac_ngv_des, _frac_blade_des = gauntner_cooling_fracs(_TIT_R_des, _T3_est_R)

    print('  Gauntner cooling sizing  (NASA-TM-81453)')
    print(f'  T4={_TIT_R_des:.0f}°R  T3_est={_T3_est_R:.1f}°R')
    print(f'  T_metal: vane={T_METAL_VANE_R:.0f}°R  blade={T_METAL_BLADE_R:.0f}°R')
    print(f'  NGV={_frac_ngv_des*100:.2f}%  Blade={_frac_blade_des*100:.2f}%'
          f'  Total={(_frac_ngv_des+_frac_blade_des)*100:.2f}%')
    print('=' * 65)

    BASE_CFG = dict(
        # Flight condition
        alt           = DESIGN_ALT,
        MN            = DESIGN_MN,
        # Design targets
        SHP           = 4000.0,       # hp
        LP_Nmech      = 5000.0,       # rpm (PT speed, held constant VARICAP mode)
        HP_Nmech_init = _HP_N_des,    # rpm (GG initial guess)
        TIT_R         = _TIT_R_des,   # °R  (2370°R = 1316.7 K)
        OPR           = _OPR_des,
        # Component efficiencies
        comp_eff      = _comp_eff,
        pt_eff        = 0.90,
        # Station Mach numbers
        inlet_MN      = 0.60,
        comp_MN       = 0.20,
        burner_MN     = 0.20,
        turb_MN       = 0.40,
        # Nozzle
        nozz_Cv          = 0.99,
        nozz_PR_target   = 1.2,
        # Combustor
        dPqP          = DPQP_DESIGN,
        # Cooling fracs (Gauntner pre-compute)
        ngv_cool_frac = _frac_ngv_des,
        bld_cool_frac = _frac_blade_des,
        # Component maps
        comp_map      = pyc.AXI5,
        pt_map        = pyc.LPT2269,
    )

    CFG_VGT = {**BASE_CFG, 'turb_map': EngineHPTVGTMap,   'turb_eff': eff_vgt}
    # CFG_FIXED = {**BASE_CFG, 'turb_map': EngineHPTFixedMap, 'turb_eff': eff_fixed}

    cfg = CFG_VGT

    print('\n' + '█' * 68)
    print('  Turboshaft Variable A4 Study — Simple Cycle, VARICAP mode')
    print(f'  Design: SLS  OPR={cfg["OPR"]}  Tt4={cfg["TIT_R"]*RANKINE_TO_K:.0f} K  '
          f'SHP={cfg["SHP"]:.0f} hp  LP_N={cfg["LP_Nmech"]:.0f} rpm')
    print(f'  A4 bounds: {A4_BOUNDS}  '
          f'(close<1=const-TIT, open>1=const-OPR per Roy-Aikins 1990)')
    print('█' * 68)

    prob = om.Problem()
    prob.model = _make_mp_turboshaft(cfg)()
    prob.setup(check=False)
    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    # ── Step 1: Design solve ──────────────────────────────────────────────────
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

    prob['DESIGN.balance.FAR']     = 0.0175506829934
    prob['DESIGN.balance.W']       = 27.265
    prob['DESIGN.balance.turb_PR'] = 3.8768
    prob['DESIGN.balance.pt_PR']   = 2.0
    prob['DESIGN.fc.balance.Pt']   = _std_atm(cfg['alt'])[0]
    prob['DESIGN.fc.balance.Tt']   = _std_atm(cfg['alt'])[1]

    prob.model._get_subsystem('SWEEP').nonlinear_solver.options['maxiter'] = 0
    prob.model._get_subsystem('DESIGN').nonlinear_solver.options['maxiter'] = 30

    # Seed CoolingCalcs initial guess from Gauntner pre-compute
    prob.set_val('DESIGN.cool_fracs.frac_ngv', _frac_ngv_des)
    prob.set_val('DESIGN.cool_fracs.frac_bld', _frac_blade_des)

    print('\n── Step 1: Design point (SLS) ──')
    t0 = time.time()
    prob.run_model()
    print(f'  Solved in {time.time()-t0:.1f} s')

    des = extract_design(prob)
    des['CdA_liner'] = compute_CdA(des)

    # Extract converged cooling fracs from CoolingCalcs (dynamic, in-solver)
    cfg['ngv_cool_frac'] = float(prob.get_val('DESIGN.cool_fracs.frac_ngv')[0])
    cfg['bld_cool_frac'] = float(prob.get_val('DESIGN.cool_fracs.frac_bld')[0])
    # Also put in des dict so solve_od() can use them as warm-start seeds
    des['ngv_cool_frac'] = cfg['ngv_cool_frac']
    des['bld_cool_frac'] = cfg['bld_cool_frac']

    # Gauntner audit against converged CoolingCalcs fracs
    _frac_ngv_g, _frac_blade_g = gauntner_cooling_fracs(
        T4_R=des['T4'], T3_R=des['T3'], T4_exit_R=des['T4_exit'])
    _T0rel_R      = turb_rel_temp_rankine(des['T4'], des['T4_exit'])
    _Tmetal_ngv   = t_metal_rankine(des['T4'], des['T3']) * RANKINE_TO_K
    _Tmetal_blade = t_metal_rankine(_T0rel_R,  des['T3']) * RANKINE_TO_K

    print(f"  OPR       = {des['OPR']:.3f}  (target {cfg['OPR']})")
    print(f"  SHP       = {des['SHP']:.1f} hp  ({des['SHP']*HP_TO_KW:.1f} kW)")
    print(f"  Tt4       = {des['T4']*RANKINE_TO_K:.1f} K  ({des['T4']:.1f}°R)")
    print(f"  Tt3       = {des['T3']*RANKINE_TO_K:.1f} K")
    print(f"  W         = {des['W']:.2f} lbm/s")
    print(f"  PSFC      = {des['PSFC']:.5f} lbm/hp/hr")
    print(f"  GGT PR    = {des['turb_PR']:.3f}")
    print(f"  PT  PR    = {des['pt_PR']:.3f}")
    print(f"  turb s_Wp = {des['turb_s_Wp']:.6f}")
    print(f"  pt   s_Wp = {des['pt_s_Wp']:.6f}")
    print(f"  dP/P_des  = {cfg['dPqP']*100:.1f}%  →  CdA = {des['CdA_liner']:.5e} m²")
    print(f"  Cool NGV  = {cfg['ngv_cool_frac']*100:.2f}%W (cycle)  "
          f"Gauntner={_frac_ngv_g*100:.2f}%  T_metal_NGV  = {_Tmetal_ngv:.1f} K")
    print(f"  Cool Bld  = {cfg['bld_cool_frac']*100:.2f}%W (cycle)  "
          f"Gauntner={_frac_blade_g*100:.2f}%  T_metal_blade= {_Tmetal_blade:.1f} K")

    # Freeze DESIGN for all subsequent OD solves
    prob.model._get_subsystem('DESIGN').nonlinear_solver.options['maxiter'] = 0
    prob.model._get_subsystem('SWEEP').nonlinear_solver.options['maxiter'] = 50

    # ── Prime SWEEP at 100% power so internal states are physical before part-power sweep ──
    print('\n── Priming SWEEP state at 100% power ──')
    _prime = solve_od(prob, des,
                      turb_s_Wp = des['turb_s_Wp'],
                      pt_s_Wp   = des['pt_s_Wp'],
                      pwr_hp    = des['SHP'],
                      warm      = des,
                      label     = 'prime 100%')
    if _prime['converged']:
        print(f"  Prime OK: PSFC={_prime['PSFC']:.5f}  Tt4={_prime['T4']*RANKINE_TO_K:.1f}K")
    else:
        print('  Prime FAILED — will fall back to design-point seeds')

    # ── Step 2: Fixed baseline power sweep ────────────────────────────────────
    # Sweep 100%→50% so each solve warm-starts from a closer state.
    # Results are stored by index (matching PWR_FRACS ascending order) for downstream use.
    print('\n── Step 2: Fixed geometry power sweep ──')
    _pf_idx   = {pf: i for i, pf in enumerate(PWR_FRACS)}
    fixed_res = [nan_result() for _ in PWR_FRACS]
    prev      = _prime if _prime['converged'] else None
    for pf in sorted(PWR_FRACS, reverse=True):
        pwr = pf * des['SHP']
        r = solve_od(prob, des,
                     turb_s_Wp = des['turb_s_Wp'],
                     pt_s_Wp   = des['pt_s_Wp'],
                     pwr_hp    = pwr,
                     warm      = _safe_warm(prev, des) if prev else des,
                     label     = f'fixed {pf*100:.0f}%')
        fixed_res[_pf_idx[pf]] = r
        if r['converged']:
            prev = r
            print(f"  Fixed {pf*100:.0f}%  PSFC={r['PSFC']:.5f}  "
                  f"Tt4={r['T4']*RANKINE_TO_K:.1f}K  "
                  f"OPR={r['OPR']:.3f}  "
                  f"dP/P={r['dPqP']*100:.2f}%  "
                  f"GGT_PR={r['turb_PR']:.3f}  PT_PR={r['pt_PR']:.3f}")
        else:
            print(f"  Fixed {pf*100:.0f}%  FAILED")

    # ── Step 3: VGT Brent optimizer sweep ────────────────────────────────────
    print('\n── Step 3: VGT Brent optimizer sweep ──')
    # vgt_at_opt[i] stores the converged OD result at the optimal a4 for power i.
    # opt_a4 now returns (a4_sc, result_dict) where result_dict is a fresh re-solve
    # at res.x — ensures prob state is clean and gives a good Step 4 warm-start.
    a4_raw, vgt_at_opt, prev_vgt = [], [], None
    for i, pf in enumerate(PWR_FRACS):
        pwr   = pf * des['SHP']
        _warm = _safe_warm(prev_vgt, des) if prev_vgt else des
        print(f"  VGT {pf*100:.0f}% solving...")
        a4_sc, _vgt_ref = opt_a4(prob, des, pwr, warm=_warm)
        a4_raw.append(a4_sc)
        vgt_at_opt.append(_vgt_ref)
        print(f"  VGT {pf*100:.0f}%  a4* = {a4_sc:.4f}")
        # Use the VGT result (at optimal a4) as warm-start for the NEXT Brent search.
        # Fall back to fixed result or design if the VGT final solve failed.
        prev_vgt = (_vgt_ref  if _vgt_ref.get('converged')
                    else fixed_res[i] if fixed_res[i].get('converged')
                    else des)

    # Use Brent results directly — opt_a4 already re-solves at res.x so each
    # vgt_at_opt[i] is a clean converged state.  Polynomial smoothing is omitted:
    # it overrides the a4=1.0 enforcement at design power and risks re-solve
    # divergence when jumping back from 100% to lower power levels.
    fn_pct   = np.array(PWR_FRACS) * 100
    a4_sched = np.array(a4_raw)
    vgt_res  = vgt_at_opt

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '═' * 126)
    print(f"  {'Fn%':>5}  {'Fix_η_th':>9}  {'VGT_η_th':>9}  {'Δη_th':>8}  "
          f"{'Fix_PSFC':>9}  {'VGT_PSFC':>9}  "
          f"{'Fix_T4K':>8}  {'VGT_T4K':>8}  "
          f"{'Fix_dP%':>7}  {'VGT_dP%':>7}  "
          f"{'a4':>7}  {'GGT_PR':>7}")
    print('─' * 126)
    for i, pf in enumerate(PWR_FRACS):
        fr, vr = fixed_res[i], vgt_res[i]
        fc, vc = fr.get('converged'), vr.get('converged')
        if fc and vc:
            d_eta  = (vr['eta_th'] - fr['eta_th']) * 100
            d_psfc = (fr['PSFC'] - vr['PSFC']) / fr['PSFC'] * 100
            print(f"  {pf*100:5.0f}%  "
                  f"{fr['eta_th']*100:9.3f}%  {vr['eta_th']*100:9.3f}%  {d_eta:+8.3f}  "
                  f"{fr['PSFC']:9.5f}  {vr['PSFC']:9.5f}  "
                  f"{fr['T4']*RANKINE_TO_K:8.1f}  {vr['T4']*RANKINE_TO_K:8.1f}  "
                  f"{fr['dPqP']*100:7.2f}%  {vr['dPqP']*100:7.2f}%  "
                  f"{a4_sched[i]:7.4f}  {vr['turb_PR']:7.3f}")
        else:
            _fe = f"{fr['eta_th']*100:.3f}%" if fc else 'FAIL'
            _ve = f"{vr['eta_th']*100:.3f}%" if vc else 'FAIL'
            print(f"  {pf*100:5.0f}%  {_fe:>10}  {_ve:>10}  "
                  f"{'---':>8}  {'---':>9}  {'---':>9}  "
                  f"{'---':>8}  {'---':>8}  {'---':>7}  {'---':>7}  "
                  f"{a4_sched[i]:7.4f}  {'---':>7}")
    print('═' * 126)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(des, PWR_FRACS, fixed_res, vgt_res, a4_sched)
    plot_turboshaft_maps(prob, des, fixed_res, vgt_res, a4_sched, PWR_FRACS)
