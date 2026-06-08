"""
A4 Variation Study — Full Analysis
====================================
Two coupled studies of variable HPT NGV area (A4) effect on cycle performance.

STUDY 1 — Constant Tt4, variable A4
  Fixed TIT, floating thrust. Shows how A4 variation moves the operating
  point and where TSFC is minimised at the design power setting.

STUDY 2 — Constant Fn, variable A4 + Tt4
  Fixed thrust target (multiple levels), floating Tt4. For each thrust level
  finds the optimal (A4, Tt4) pair that minimises TSFC. Directly quantifies
  the variable A4 benefit across the operating envelope.

Physical approach:
  s_Wp is the turbine map corrected flow scalar. Scaling it by a4_scalar
  directly scales the HPT NGV throat area A4 while holding the downstream
  propulsive nozzle area fixed (connected from DESIGN). This is physically
  equivalent to variable A4 with fixed An.

  A4 scalar < 1.0  →  smaller A4  →  higher PR  →  higher pi_c
  A4 scalar > 1.0  →  larger  A4  →  lower  PR  →  lower  pi_c

Flight condition: Sea Level Static (MN ≈ 0, alt = 0 ft) throughout.

Author  : Based on pyCycle turbojet example (Hendricks & Gray, NASA GRC)
Requires: pip install om-pycycle
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import openmdao.api as om
import pycycle.api as pyc


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

A4_SCALARS   = list(np.linspace(0.80, 1.05, 30))
FN_FRACTIONS = [0.80 ,0.90, 1.00]   # fractions of design thrust
FN_DESIGN    = 11800.0                            # lbf

# Unit conversion factors
LBM_S_TO_KG_S = 0.453592
LBF_TO_KN     = 0.00444822
RANKINE_TO_K  = 5.0 / 9.0
PSI_TO_KPA    = 6.89476
COMP_MAP = pyc.HPCMap      # change this one line to switch maps
from  EngineHPT_map import  EngineHPTMap
TURB_MAP =  EngineHPTMap

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN POINT
# ══════════════════════════════════════════════════════════════════════════════

class Turbojet(pyc.Cycle):
    """
    Single-spool turbojet design point.
    Balances: W → Fn_target, FAR → T4_target, turb_PR → shaft pwr = 0
    """

    def setup(self):
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data']   = pyc.AIR_JETA_TAB_SPEC
        FUEL_TYPE = 'FAR'
        design = self.options['design']

        self.add_subsystem('fc',     pyc.FlightConditions())
        self.add_subsystem('inlet',  pyc.Inlet())
        self.add_subsystem('comp',   pyc.Compressor(map_data=COMP_MAP,
                                                     map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('turb',   pyc.Turbine(map_data=TURB_MAP,
                                                  map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('nozz',   pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft',  pyc.Shaft(num_ports=2),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('perf',   pyc.Performance(num_nozzles=1,
                                                      num_burners=1))

        self.pyc_connect_flow('fc.Fl_O',     'inlet.Fl_I',  connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O',  'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O',   'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O',   'nozz.Fl_I')

        self.connect('comp.trq',         'shaft.trq_0')
        self.connect('turb.trq',         'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P',   'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P',  'perf.Pt3')
        self.connect('burner.Wfuel',     'perf.Wfuel_0')
        self.connect('inlet.F_ram',      'perf.ram_drag')
        self.connect('nozz.Fg',          'perf.Fg_0')

        balance = self.add_subsystem('balance', om.BalanceComp())

        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf',
                                 rhs_name='Fn_target')
            self.connect('balance.W',  'inlet.Fl_I:stat:W')
            self.connect('perf.Fn',    'balance.lhs:W')

            balance.add_balance('FAR', eq_units='degR', lower=1e-4,
                                 val=0.017, rhs_name='T4_target')
            self.connect('balance.FAR',       'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance('turb_PR', val=1.5, lower=1.001, upper=8,
                                 eq_units='hp', rhs_val=0.)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net',   'balance.lhs:turb_PR')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol']                        = 1e-6
        newton.options['rtol']                        = 1e-6
        newton.options['iprint']                      = 2
        newton.options['maxiter']                     = 15
        newton.options['solve_subsystems']            = True
        newton.options['max_sub_solves']              = 100
        newton.options['reraise_child_analysiserror'] = False
        self.linear_solver = om.DirectSolver()
        super().setup()


# ══════════════════════════════════════════════════════════════════════════════
#  STUDY 1 — OFF-DESIGN POINT: FIXED Tt4, FLOATING THRUST
# ══════════════════════════════════════════════════════════════════════════════

class TurbojetConstTt4(pyc.Cycle):
    """
    Off-design point for Study 1.
    FAR  → T4_target   (TIT held constant)
    Nmech → pwr_net=0  (shaft power balance)
    W    → An_target   (nozzle area = design An, mass flow floats)
    s_Wp set externally → controls effective A4
    """

    def setup(self):
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data']   = pyc.AIR_JETA_TAB_SPEC
        FUEL_TYPE = 'FAR'
        design = self.options['design']

        self.add_subsystem('fc',     pyc.FlightConditions())
        self.add_subsystem('inlet',  pyc.Inlet())
        self.add_subsystem('comp',   pyc.Compressor(map_data=COMP_MAP,
                                                     map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('turb',   pyc.Turbine(map_data=TURB_MAP,
                                                  map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('nozz',   pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft',  pyc.Shaft(num_ports=2),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('perf',   pyc.Performance(num_nozzles=1,
                                                      num_burners=1))

        self.pyc_connect_flow('fc.Fl_O',     'inlet.Fl_I',  connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O',  'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O',   'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O',   'nozz.Fl_I')

        self.connect('comp.trq',         'shaft.trq_0')
        self.connect('turb.trq',         'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P',   'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P',  'perf.Pt3')
        self.connect('burner.Wfuel',     'perf.Wfuel_0')
        self.connect('inlet.F_ram',      'perf.ram_drag')
        self.connect('nozz.Fg',          'perf.Fg_0')

        balance = self.add_subsystem('balance', om.BalanceComp())

        # FAR drives T4 — TIT held constant
        balance.add_balance('FAR', eq_units='degR', lower=1e-4,
                             val=0.017, rhs_name='T4_target')
        self.connect('balance.FAR',       'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

        # Nmech floats to satisfy shaft power = 0
        balance.add_balance('Nmech', val=8070., units='rpm',
                             lower=500., eq_units='hp', rhs_val=0.)
        self.connect('balance.Nmech', 'Nmech')
        self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

        # W drives nozzle area to design value — An fixed
        balance.add_balance('W', val=168.0, units='lbm/s',
                             eq_units='inch**2', rhs_name='nozz_area_target')
        self.connect('balance.W',             'inlet.Fl_I:stat:W')
        self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol']                        = 1e-6
        newton.options['rtol']                        = 1e-6
        newton.options['iprint']                      = 2
        newton.options['maxiter']                     = 15
        newton.options['solve_subsystems']            = True
        newton.options['max_sub_solves']              = 100
        newton.options['reraise_child_analysiserror'] = False
        self.linear_solver = om.DirectSolver()
        super().setup()


# ══════════════════════════════════════════════════════════════════════════════
#  STUDY 2 — OFF-DESIGN POINT: FIXED THRUST, FLOATING Tt4
# ══════════════════════════════════════════════════════════════════════════════

class TurbojetConstFn(pyc.Cycle):
    """
    Off-design point for Study 2.
    FAR  → Fn_target   (thrust held constant — KEY DIFFERENCE from Study 1)
    Nmech → pwr_net=0  (shaft power balance)
    W    → An_target   (nozzle area = design An, mass flow floats)
    s_Wp set externally → controls effective A4
    Tt4 is a result, not a target.
    """

    def setup(self):
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data']   = pyc.AIR_JETA_TAB_SPEC
        FUEL_TYPE = 'FAR'
        design = self.options['design']

        self.add_subsystem('fc',     pyc.FlightConditions())
        self.add_subsystem('inlet',  pyc.Inlet())
        self.add_subsystem('comp',   pyc.Compressor(map_data=COMP_MAP,
                                                     map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('turb',   pyc.Turbine(map_data=TURB_MAP,
                                                  map_extrap=True),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('nozz',   pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft',  pyc.Shaft(num_ports=2),
                                     promotes_inputs=['Nmech'])
        self.add_subsystem('perf',   pyc.Performance(num_nozzles=1,
                                                      num_burners=1))

        self.pyc_connect_flow('fc.Fl_O',     'inlet.Fl_I',  connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O',  'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O',   'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O',   'nozz.Fl_I')

        self.connect('comp.trq',         'shaft.trq_0')
        self.connect('turb.trq',         'shaft.trq_1')
        self.connect('fc.Fl_O:stat:P',   'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P',  'perf.Pt3')
        self.connect('burner.Wfuel',     'perf.Wfuel_0')
        self.connect('inlet.F_ram',      'perf.ram_drag')
        self.connect('nozz.Fg',          'perf.Fg_0')

        balance = self.add_subsystem('balance', om.BalanceComp())

        # FAR drives thrust — Fn held constant, Tt4 floats
        balance.add_balance('FAR', eq_units='lbf', lower=1e-4,
                             val=0.017, rhs_name='Fn_target')
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('perf.Fn',     'balance.lhs:FAR')

        # Nmech floats to satisfy shaft power = 0
        balance.add_balance('Nmech', val=8070., units='rpm',
                             lower=500., eq_units='hp', rhs_val=0.)
        self.connect('balance.Nmech', 'Nmech')
        self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

        # W drives nozzle area to design value — An fixed
        balance.add_balance('W', val=168.0, units='lbm/s',
                             eq_units='inch**2', rhs_name='nozz_area_target')
        self.connect('balance.W',             'inlet.Fl_I:stat:W')
        self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol']                        = 1e-6
        newton.options['rtol']                        = 1e-6
        newton.options['iprint']                      = 2
        newton.options['maxiter']                     = 15
        newton.options['solve_subsystems']            = True
        newton.options['max_sub_solves']              = 100
        newton.options['reraise_child_analysiserror'] = False
        self.linear_solver = om.DirectSolver()
        super().setup()


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-POINT: STUDY 1 — CONSTANT Tt4
# ══════════════════════════════════════════════════════════════════════════════

class MPConstTt4(pyc.MPCycle):

    def setup(self):
        self.pyc_add_pnt('DESIGN', Turbojet())
        self.set_input_defaults('DESIGN.Nmech',     8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN',  0.60)
        self.set_input_defaults('DESIGN.comp.MN',   0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN',   0.4)
        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozz.Cv',     0.99)

        self.a4_scalars = A4_SCALARS
        self.a4_pts     = [f'A4_{i:03d}' for i in range(len(self.a4_scalars))]

        for pt in self.a4_pts:
            self.pyc_add_pnt(pt, TurbojetConstTt4(design=False))
            self.set_input_defaults(pt + '.fc.MN',  0.000001)
            self.set_input_defaults(pt + '.fc.alt', 0.0, units='ft')

        for pt in self.a4_pts:
            self.connect('DESIGN.nozz.Throat:stat:area',
                         f'{pt}.balance.nozz_area_target')
            for sc in ['s_PR', 's_Wc', 's_eff', 's_Nc']:
                self.connect(f'DESIGN.comp.{sc}', f'{pt}.comp.{sc}')
            for sc in ['s_PR', 's_eff', 's_Np']:
                self.connect(f'DESIGN.turb.{sc}', f'{pt}.turb.{sc}')

        super().setup()


# ══════════════════════════════════════════════════════════════════════════════
#  MULTI-POINT: STUDY 2 — CONSTANT THRUST
# ══════════════════════════════════════════════════════════════════════════════

class MPConstFn(pyc.MPCycle):
    """
    One DESIGN point (frozen after transfer of scalars from Study 1) plus
    len(FN_FRACTIONS) x len(A4_SCALARS) off-design points.

    Point naming: FN{fn_idx:02d}_A4_{a4_idx:03d}
      fn_idx  — index into FN_FRACTIONS
      a4_idx  — index into A4_SCALARS
    """

    def setup(self):
        self.pyc_add_pnt('DESIGN', Turbojet())
        self.set_input_defaults('DESIGN.Nmech',     8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN',  0.60)
        self.set_input_defaults('DESIGN.comp.MN',   0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN',   0.4)
        self.pyc_add_cycle_param('burner.dPqP', 0.03)
        self.pyc_add_cycle_param('nozz.Cv',     0.99)

        self.fn_fractions = FN_FRACTIONS
        self.a4_scalars   = A4_SCALARS
        self.od_pts       = []

        for fi, _ in enumerate(self.fn_fractions):
            for ai, _ in enumerate(self.a4_scalars):
                pt = f'FN{fi:02d}_A4_{ai:03d}'
                self.od_pts.append(pt)
                self.pyc_add_pnt(pt, TurbojetConstFn(design=False))
                self.set_input_defaults(pt + '.fc.MN',  0.000001)
                self.set_input_defaults(pt + '.fc.alt', 0.0, units='ft')

        for pt in self.od_pts:
            self.connect('DESIGN.nozz.Throat:stat:area',
                         f'{pt}.balance.nozz_area_target')
            for sc in ['s_PR', 's_Wc', 's_eff', 's_Nc']:
                self.connect(f'DESIGN.comp.{sc}', f'{pt}.comp.{sc}')
            for sc in ['s_PR', 's_eff', 's_Np']:
                self.connect(f'DESIGN.turb.{sc}', f'{pt}.turb.{sc}')

        super().setup()


# ══════════════════════════════════════════════════════════════════════════════
#  VIEWER
# ══════════════════════════════════════════════════════════════════════════════

def viewer(prob, pt, file=sys.stdout):
    summary = (
        prob[pt + '.fc.Fl_O:stat:MN'],
        prob[pt + '.fc.alt'],
        prob[pt + '.inlet.Fl_O:stat:W'],
        prob[pt + '.perf.Fn'],
        prob[pt + '.perf.Fg'],
        prob[pt + '.inlet.F_ram'],
        prob[pt + '.perf.OPR'],
        prob[pt + '.perf.TSFC'],
    )
    print(file=file, flush=True)
    print('─' * 76, file=file)
    print(f'  POINT: {pt}', file=file)
    print('─' * 76, file=file)
    print('  Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC',
          file=file)
    print(' %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f' % summary,
          file=file, flush=True)
    fs_full = [f'{pt}.{fs}' for fs in
               ['fc.Fl_O', 'inlet.Fl_O', 'comp.Fl_O',
                'burner.Fl_O', 'turb.Fl_O', 'nozz.Fl_O']]
    pyc.print_flow_station(prob, fs_full, file=file)
    pyc.print_compressor(prob, [f'{pt}.comp'],  file=file)
    pyc.print_burner(prob,     [f'{pt}.burner'])
    pyc.print_turbine(prob,    [f'{pt}.turb'],  file=file)
    pyc.print_nozzle(prob,     [f'{pt}.nozz'],  file=file)
    pyc.print_shaft(prob,      [f'{pt}.shaft'], file=file)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING — STUDY 1 (CONSTANT Tt4)
# ══════════════════════════════════════════════════════════════════════════════

def plot_const_tt4(scalars, results, design_results):
    """Six-panel summary of the constant-Tt4 A4 sweep."""

    s     = np.array(scalars)
    pi_c  = np.array([r['pi_c']  for r in results])
    Nmech = np.array([r['Nmech'] for r in results])
    W     = np.array([r['W']     for r in results]) * LBM_S_TO_KG_S
    Fn    = np.array([r['Fn']    for r in results]) * LBF_TO_KN
    TSFC  = np.array([r['TSFC']  for r in results])
    T3    = np.array([r['T3']    for r in results]) * RANKINE_TO_K
    P3    = np.array([r['P3']    for r in results]) * PSI_TO_KPA

    W_des    = design_results['W']    * LBM_S_TO_KG_S
    TSFC_des = design_results['TSFC']
    T4_K     = design_results['T4_design'] * RANKINE_TO_K

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        'Study 1 — Constant TIT, Variable A\u2084\n'
        f'Fixed T\u2084 = {T4_K:.0f} K   |   Sea Level Static',
        fontsize=12, fontweight='bold')

    kw  = dict(lw=2, marker='o', markersize=3)
    kwh = dict(ls=':', lw=1.4, alpha=0.8)

    def vline(ax):
        ax.axvline(1.0, color='gray', ls='--', lw=1.5, label='Design A\u2084')

    def xlabel(ax):
        ax.set_xlabel('A\u2084 scalar  (1.0 = design)', fontsize=9)

    # PR
    ax = axes[0, 0]
    ax.plot(s, pi_c, 'C0', **kw)
    ax.axhline(design_results['pi_c'], color='C0',
               label=f"Design = {design_results['pi_c']:.2f}", **kwh)
    vline(ax); xlabel(ax)
    ax.set_ylabel('Overall Pressure Ratio  \u03c0\u2099', fontsize=9)
    ax.set_title('Compressor PR\nClosing A\u2084 \u2192 higher \u03c0\u2099', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Nmech
    ax = axes[0, 1]
    ax.plot(s, Nmech, 'C1', **kw)
    ax.axhline(design_results['Nmech'], color='C1',
               label=f"Design = {design_results['Nmech']:.0f} rpm", **kwh)
    vline(ax); xlabel(ax)
    ax.set_ylabel('Shaft Speed  N  (rpm)', fontsize=9)
    ax.set_title('Spool Speed\nShaft power balance', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Mass flow
    ax = axes[0, 2]
    ax.plot(s, W, 'C2', **kw)
    ax.axhline(W_des, color='C2',
               label=f"Design = {W_des:.1f} kg/s", **kwh)
    vline(ax); xlabel(ax)
    ax.set_ylabel('Inlet Mass Flow  \u1e40  (kg/s)', fontsize=9)
    ax.set_title('Mass Flow\nNon-monotonic: A\u2084\u2193 raises PR,\nnet \u1e40 peaks at mid-range',
                 fontsize=9)
    peak_idx = int(np.argmax(W))
    ax.annotate(f'Peak {W[peak_idx]:.1f} kg/s',
                xy=(s[peak_idx], W[peak_idx]),
                xytext=(s[peak_idx] + 0.04, W[peak_idx] - 0.5),
                fontsize=7, arrowprops=dict(arrowstyle='->', color='C2'))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Thrust
    ax = axes[1, 0]
    ax.plot(s, Fn, 'C3', **kw)
    Fn_des_kN = FN_DESIGN * LBF_TO_KN
    ax.axhline(Fn_des_kN, color='C3',
               label=f"Design = {Fn_des_kN:.1f} kN", **kwh)
    vline(ax); xlabel(ax)
    ax.set_ylabel('Net Thrust  F\u2099  (kN)', fontsize=9)
    ax.set_title('Net Thrust\nFixed T\u2084 \u2014 thrust varies freely', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # TSFC
    ax = axes[1, 1]
    ax.plot(s, TSFC, 'C4', **kw)
    ax.axhline(TSFC_des, color='C4',
               label=f"Design = {TSFC_des:.5f}", **kwh)
    vline(ax); xlabel(ax)
    ax.set_ylabel('TSFC  (lbm / hr / lbf)', fontsize=9)
    ax.set_title('TSFC\nOptimal A\u2084 \u2260 design', fontsize=9)
    min_idx   = int(np.argmin(TSFC))
    delta_abs = TSFC_des - TSFC[min_idx]
    delta_pct = delta_abs / TSFC_des * 100.0
    ax.annotate(
        f'Optimum A\u2084 = {s[min_idx]:.3f}\n'
        f'TSFC = {TSFC[min_idx]:.5f}\n'
        f'\u2193 {delta_abs:.5f}  ({delta_pct:.2f}%)\n'
        f'vs design ({TSFC_des:.5f})',
        xy=(s[min_idx], TSFC[min_idx]),
        xytext=(s[min_idx] + 0.05, TSFC[min_idx] + 0.003),
        fontsize=7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='C4', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='C4'))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Combustor inlet
    ax  = axes[1, 2]
    ax2 = ax.twinx()
    l1, = ax.plot(s,  T3, 'C5', **kw, label='T\u2083  (K)')
    l2, = ax2.plot(s, P3, 'C6', lw=2, marker='s', markersize=3,
                   ls='--', label='P\u2083  (kPa)')
    vline(ax); xlabel(ax)
    ax.set_ylabel('Combustor Inlet Temp  T\u2083  (K)', fontsize=9, color='C5')
    ax2.set_ylabel('Combustor Inlet Pressure  P\u2083  (kPa)', fontsize=9, color='C6')
    ax.set_title('Combustor Inlet Conditions\n'
                 'Closing A\u2084 \u2192 higher T\u2083 and P\u2083', fontsize=9)
    ax.tick_params(axis='y', labelcolor='C5')
    ax2.tick_params(axis='y', labelcolor='C6')
    ax.legend(handles=[l1, l2], fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.55, wspace=0.38)
    plt.savefig('study1_const_tt4.png', dpi=150)
    print('Plot saved: study1_const_tt4.png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING — STUDY 2 (CONSTANT THRUST)
# ══════════════════════════════════════════════════════════════════════════════

def plot_const_fn(fn_fractions, a4_scalars, const_fn_results, design_results):
    """
    Four-panel summary of the constant-thrust study.
      Panel 1 — TSFC locus at each thrust level
      Panel 2 — Optimal A4 scalar vs % design thrust
      Panel 3 — TSFC improvement vs % design thrust
      Panel 4 — Tt4 at optimal A4 vs % design thrust (thermal benefit)
    """

    s      = np.array(a4_scalars)
    colors = cm.viridis(np.linspace(0.15, 0.85, len(fn_fractions)))
    T4_K   = design_results['T4_design'] * RANKINE_TO_K

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        'Study 2 — Constant Thrust, Variable A\u2084 and T\u2084\n'
        'Sea Level Static   |   Optimal (A\u2084, T\u2084) minimises TSFC at each thrust setting',
        fontsize=12, fontweight='bold')

    # ── Panel 1: TSFC locus ───────────────────────────────────────────────────
    ax = axes[0, 0]
    for fi, frac in enumerate(fn_fractions):
        sweep = const_fn_results[frac]['sweep']
        tsfc  = np.array([r['TSFC'] for r in sweep])
        label = f"{int(frac * 100)}% F\u2099"
        ax.plot(s, tsfc, color=colors[fi], lw=2, marker='o',
                markersize=2, label=label)

        # Mark minimum
        mi = int(np.argmin(tsfc))
        ax.plot(s[mi], tsfc[mi], '*', color=colors[fi],
                markersize=10, markeredgecolor='k', zorder=5)

    # Connect the minima with a dashed curve
    opt_s    = [const_fn_results[f]['opt_scalar']   for f in fn_fractions]
    opt_tsfc = [const_fn_results[f]['opt_tsfc']     for f in fn_fractions]
    ax.plot(opt_s, opt_tsfc, 'k--', lw=1.5, label='Locus of optima')

    ax.axvline(1.0, color='gray', ls='--', lw=1.2, label='Design A\u2084')
    ax.set_xlabel('A\u2084 scalar  (1.0 = design)', fontsize=9)
    ax.set_ylabel('TSFC  (lbm / hr / lbf)', fontsize=9)
    ax.set_title('TSFC vs A\u2084 at Constant Thrust\n'
                 '\u2605 = minimum TSFC  |  dashed = optimal locus', fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Optimal A4 scalar vs thrust ──────────────────────────────────
    ax = axes[0, 1]
    pct_thrust = [f * 100 for f in fn_fractions]
    opt_scalars = [const_fn_results[f]['opt_scalar'] for f in fn_fractions]

    ax.plot(pct_thrust, opt_scalars, 'ko-', lw=2, markersize=8)
    for fi, (pt, os) in enumerate(zip(pct_thrust, opt_scalars)):
        ax.annotate(f'{os:.3f}', xy=(pt, os),
                    xytext=(pt + 1.5, os + 0.003),
                    fontsize=8, color=colors[fi])

    ax.axhline(1.0, color='gray', ls='--', lw=1.2, label='Design geometry')
    ax.set_xlabel('Thrust Setting  (% design F\u2099)', fontsize=9)
    ax.set_ylabel('Optimal A\u2084 Scalar', fontsize=9)
    ax.set_title('Optimal A\u2084 Schedule\n'
                 'Control law: close A\u2084 at part power', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: TSFC improvement vs thrust ───────────────────────────────────
    ax = axes[1, 0]
    # Fixed geometry TSFC — A4 scalar closest to 1.0 for each thrust level
    fixed_geom_tsfc = [const_fn_results[f]['baseline_tsfc'] for f in fn_fractions]
    opt_tsfc_arr    = [const_fn_results[f]['opt_tsfc']      for f in fn_fractions]

    delta_abs = [(fg - op)           for fg, op in zip(fixed_geom_tsfc, opt_tsfc_arr)]
    delta_pct = [(d / fg) * 100.0    for d,  fg in zip(delta_abs, fixed_geom_tsfc)]

    ax2 = ax.twinx()
    bars = ax.bar(pct_thrust, delta_pct, width=4, color=colors,
                  edgecolor='k', alpha=0.85, label='% improvement')
    ax2.plot(pct_thrust, delta_abs, 'k^--', lw=1.5, ms=7,
             label='\u0394TSFC (abs)')

    for i, (pt, dp) in enumerate(zip(pct_thrust, delta_pct)):
        ax.text(pt, dp + 0.01, f'{dp:.2f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Thrust Setting  (% design F\u2099)', fontsize=9)
    ax.set_ylabel('TSFC Improvement  (%)', fontsize=9, color='C0')
    ax2.set_ylabel('\u0394TSFC  (lbm / hr / lbf)', fontsize=9, color='k')
    ax.set_title('TSFC Benefit of Optimal A\u2084\nvs Fixed Geometry (A\u2084 = design)',
                 fontsize=9)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.set_xlim(55, 105); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 4: Tt4 at optimal A4 vs thrust ─────────────────────────────────
    ax = axes[1, 1]

    opt_T4_K        = [const_fn_results[f]['opt_T4'] * RANKINE_TO_K
                       for f in fn_fractions]
    fixed_T4_K      = [const_fn_results[f]['baseline_T4'] * RANKINE_TO_K
                       for f in fn_fractions]

    ax.plot(pct_thrust, opt_T4_K,   'o-',  color='C3', lw=2, ms=8,
            label='Optimal A\u2084')
    ax.plot(pct_thrust, fixed_T4_K, 's--', color='C1', lw=2, ms=8,
            label='Fixed geometry (A\u2084 = design)')
    ax.axhline(T4_K, color='gray', ls=':', lw=1.2,
               label=f'Design T\u2084 = {T4_K:.0f} K')

    # Shade the T4 reduction
    ax.fill_between(pct_thrust, fixed_T4_K, opt_T4_K,
                    alpha=0.15, color='green', label='T\u2084 reduction')

    for i, (pt, ot, ft) in enumerate(zip(pct_thrust, opt_T4_K, fixed_T4_K)):
        delta_T4 = ft - ot
        ax.annotate(f'\u2212{delta_T4:.0f} K',
                    xy=(pt, (ot + ft) / 2),
                    xytext=(pt + 1.5, (ot + ft) / 2),
                    fontsize=7, color='green')

    ax.set_xlabel('Thrust Setting  (% design F\u2099)', fontsize=9)
    ax.set_ylabel('Turbine Inlet Temperature  T\u2084  (K)', fontsize=9)
    ax.set_title('TIT at Optimal A\u2084 vs Fixed Geometry\n'
                 'Variable A\u2084 enables lower T\u2084 at same thrust\n'
                 '(same power, cooler blades)', fontsize=9)
    ax.set_xlim(55, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.subplots_adjust(hspace=0.50, wspace=0.38)
    plt.savefig('study2_const_fn.png', dpi=150)
    print('Plot saved: study2_const_fn.png')
    plt.show()


def draw_compressor_map_background(
        ax, prob, e_name, alpha_idx=0,
        eff_vals=np.array([0, 0.50, 0.55, 0.60, 0.65,
                           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]),
        show_rlines=True, show_nclines=True,
        eff_alpha=1.0):
    """
    Render a full compressor map background onto an existing Axes object.
 
    Layers drawn (bottom to top):
      1. Efficiency fill    — contourf of effMap * s_eff, viridis colormap
      2. Efficiency edges   — black contour lines at same levels, labelled
      3. Speed lines        — contours of NcMap * s_Nc, labelled in rpm
      4. R-lines            — contours of RlineMap, labelled
      5. Stall line         — bold black line at R-line column index 0
 
    Parameters
    ----------
    ax           : matplotlib Axes to draw onto
    prob         : solved OpenMDAO Problem containing the compressor
    e_name       : promoted path to the compressor, e.g. 'DESIGN.comp'
    alpha_idx    : IGV angle index into the map arrays (0 = design angle)
    eff_vals     : efficiency contour levels
    show_rlines  : draw R-line contours and labels
    show_nclines : draw speed-line contours and labels
    eff_alpha    : opacity of the efficiency fill (1.0 = solid)
 
    Returns
    -------
    eff_cf : filled contour object — pass to plt.colorbar() in caller
    """
 
    comp     = prob.model._get_subsystem(e_name)
    map_data = comp.options['map_data']
 
    s_Wc  = prob.get_val(f'{e_name}.s_Wc')[0]
    s_PR  = prob.get_val(f'{e_name}.s_PR')[0]
    s_eff = prob.get_val(f'{e_name}.s_eff')[0]
    s_Nc  = prob.get_val(f'{e_name}.s_Nc')[0]
 
    # 2-D grids — shape (n_Nc, n_Rline)
    RlineMap, NcMap = np.meshgrid(map_data.RlineMap, map_data.NcMap,
                                  sparse=False)
    a = alpha_idx
    Wc_map  = map_data.WcMap[a, :, :]  * s_Wc
    PR_map  = (map_data.PRmap[a, :, :] - 1.0) * s_PR + 1.0
    eff_map = map_data.effMap[a, :, :] * s_eff
    Nc_map  = NcMap * s_Nc
 
    # ── 1. Efficiency fill ────────────────────────────────────────────────────
    eff_cf = ax.contourf(Wc_map, PR_map, eff_map,
                         levels=eff_vals,
                         cmap='viridis',
                         alpha=eff_alpha,
                         zorder=1)
 
    # ── 2. Efficiency island edges — black outlines with value labels ─────────
    eff_edge_levels = eff_vals[eff_vals > 0]
    eff_cs = ax.contour(Wc_map, PR_map, eff_map,
                        levels=eff_edge_levels,
                        colors='k',
                        linewidths=0.6,
                        zorder=2)
    ax.clabel(eff_cs, fmt='%.2f', fontsize=6,
              inline=True, inline_spacing=2)
 
    # ── 3. Speed lines ────────────────────────────────────────────────────────
    if show_nclines:
        nc_cs = ax.contour(Wc_map, PR_map, Nc_map,
                           levels=map_data.NcMap * s_Nc,
                           colors='k',
                           linewidths=0.8,
                           linestyles='solid',
                           zorder=2)
        ax.clabel(nc_cs, fmt='%.0f', fontsize=6,
                  inline=True, inline_spacing=2)
 
    # ── 4. R-lines ────────────────────────────────────────────────────────────
    if show_rlines:
        r_cs = ax.contour(Wc_map, PR_map, RlineMap,
                          levels=map_data.RlineMap,
                          colors='k',
                          linewidths=0.6,
                          linestyles='solid',
                          alpha=1.0,
                          zorder=2)
        ax.clabel(r_cs, fmt='%.2f', fontsize=6,
                  inline=True, inline_spacing=2)
 
    # ── 5. Stall line — R-line column index 0 per speed line ─────────────────
    stall_Wc = Wc_map[:, 0]
    stall_PR = PR_map[:, 0]
    sort_idx = np.argsort(stall_Wc)
    ax.plot(stall_Wc[sort_idx], stall_PR[sort_idx],
            'k-', lw=2.0, zorder=3, label='Stall line')
 
    return eff_cf
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  OPERATING LINE OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_compressor_map_with_oplines(
        prob, e_name,
        const_fn_results, fn_fractions, a4_scalars,
        design_results,
        const_tt4_results=None):
    """
    Full compressor map figure with two operating lines overlaid.
 
    Default operating line
      The locus a fixed-geometry engine traces as power is reduced —
      Study 2 results at A4 scalar closest to 1.0, one point per thrust level.
 
    Variable A4 operating line
      The locus when A4 is scheduled optimally at each power setting —
      Study 2 optimal (minimum TSFC) point per thrust level.
 
    Yellow arrows connect the two lines at each thrust level to show the
    compressor map displacement that variable A4 achieves.
 
    Parameters
    ----------
    prob              : solved OpenMDAO Problem (Study 2, prob2)
    e_name            : compressor path, e.g. 'DESIGN.comp'
    const_fn_results  : Study 2 results dict keyed by fn_fraction
    fn_fractions      : list of thrust fractions, e.g. [0.60, 0.70, ...]
    a4_scalars        : list of A4 scalars used in the sweep
    design_results    : dict from extract_design()
    const_tt4_results : optional Study 1 results list — if provided, shown
                        as a faint blue scatter (full A4 sweep at design Tt4)
    """
 
    T4_K  = design_results['T4_design'] * 5.0 / 9.0
    s_arr = np.array(a4_scalars)
 
    # Index of A4 scalar closest to 1.0 — fixed geometry baseline per point
    baseline_idx = int(np.argmin(np.abs(s_arr - 1.0)))
 
    fig, ax = plt.subplots(figsize=(13, 10))
    fig.suptitle(
        'Compressor Map — Default vs Variable A\u2084 Operating Lines\n'
        f'Design T\u2084 = {T4_K:.0f} K   |   Sea Level Static\n'
        '\u25cf\u2500\u25cf  Default op line: A\u2084 = design, thrust set by T\u2084\u2193     '
        '\u25a0\u254c\u25a0  Variable A\u2084 op line: optimal A\u2084 at each thrust',
        fontsize=11, fontweight='bold')
 
    # ── Map background ────────────────────────────────────────────────────────
    eff_cf = draw_compressor_map_background(
        ax, prob, e_name,
        alpha_idx=0,
        eff_vals=np.array([0, 0.50, 0.55, 0.60, 0.65,
                           0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]),
        show_rlines=True,
        show_nclines=True,
        eff_alpha=1.0)
 
    cb = plt.colorbar(eff_cf, ax=ax, shrink=0.60, pad=0.02)
    cb.set_label('Adiabatic efficiency  \u03b7\u2099  (\u2014)', fontsize=10)
 
    # ── Optional Study 1 scatter — full A4 sweep at design Tt4 ───────────────
    if const_tt4_results is not None:
        Wc_s1 = np.array([r['comp_Wc'] for r in const_tt4_results])
        PR_s1 = np.array([r['comp_PR'] for r in const_tt4_results])
        ax.scatter(Wc_s1, PR_s1,
                   c=s_arr, cmap='Blues', s=18,
                   alpha=0.40, zorder=4, edgecolor='none',
                   vmin=s_arr.min(), vmax=s_arr.max())
        ax.plot(Wc_s1, PR_s1,
                '-', color='steelblue', lw=1.2, alpha=0.45, zorder=4,
                label='Study 1 — const T\u2084 A\u2084 sweep')
 
    # ── Extract operating line points from Study 2 ────────────────────────────
    default_Wc, default_PR = [], []
    opt_Wc,     opt_PR     = [], []
    opt_a4_vals            = []
 
    for frac in fn_fractions:
        sweep   = const_fn_results[frac]['sweep']
        opt_idx = const_fn_results[frac]['opt_idx']
 
        r_def = sweep[baseline_idx]
        default_Wc.append(r_def['comp_Wc'])
        default_PR.append(r_def['comp_PR'])
 
        r_opt = sweep[opt_idx]
        opt_Wc.append(r_opt['comp_Wc'])
        opt_PR.append(r_opt['comp_PR'])
        opt_a4_vals.append(a4_scalars[opt_idx])
 
    default_Wc = np.array(default_Wc)
    default_PR = np.array(default_PR)
    opt_Wc     = np.array(opt_Wc)
    opt_PR     = np.array(opt_PR)
 
    # ── Default operating line ────────────────────────────────────────────────
    # Black halo + white line so it reads on any background region
    ax.plot(default_Wc, default_PR,
            '-', color='k', lw=4.5, zorder=6)
    ax.plot(default_Wc, default_PR,
            '-', color='white', lw=2.5, zorder=7,
            label='Default op line  (A\u2084 = design)')
 
    for wc, pr, frac in zip(default_Wc, default_PR, fn_fractions):
        ax.plot(wc, pr,
                'o', ms=11, color='white',
                markeredgecolor='k', markeredgewidth=1.5, zorder=8)
        ax.annotate(f'{int(frac * 100)}% F\u2099',
                    xy=(wc, pr),
                    xytext=(wc - 5, pr + 0.15),
                    fontsize=8, color='white', fontweight='bold',
                    zorder=9,
                    arrowprops=dict(arrowstyle='-',
                                    color='white', lw=0.6))
 
    # ── Variable A4 operating line ────────────────────────────────────────────
    ax.plot(opt_Wc, opt_PR,
            '--', color='k', lw=4.5, zorder=6)
    ax.plot(opt_Wc, opt_PR,
            '--', color='white', lw=2.5, zorder=7,
            label='Variable A\u2084 op line  (optimal A\u2084 per thrust)')
 
    for wc, pr, frac, a4 in zip(opt_Wc, opt_PR, fn_fractions, opt_a4_vals):
        ax.plot(wc, pr,
                's', ms=11, color='white',
                markeredgecolor='k', markeredgewidth=1.5, zorder=8)
        ax.annotate(f'{int(frac * 100)}% F\u2099\nA\u2084={a4:.2f}',
                    xy=(wc, pr),
                    xytext=(wc + 2, pr - 0.25),
                    fontsize=7, color='white', fontweight='bold',
                    zorder=9,
                    arrowprops=dict(arrowstyle='-',
                                    color='white', lw=0.6))
 
    # ── Yellow arrows connecting default to optimal at each thrust level ──────
    # These arrows make the A4-induced map shift immediately visible
    for i in range(len(fn_fractions)):
        dwc = opt_Wc[i] - default_Wc[i]
        dpr = opt_PR[i] - default_PR[i]
        if np.hypot(dwc, dpr) > 0.3:          # skip negligible shifts
            ax.annotate('',
                        xy=(opt_Wc[i], opt_PR[i]),
                        xytext=(default_Wc[i], default_PR[i]),
                        arrowprops=dict(
                            arrowstyle='->', color='yellow',
                            lw=2.0, mutation_scale=16),
                        zorder=9)
 
    # ── Design point ──────────────────────────────────────────────────────────
    ax.plot(design_results['comp_Wc_design'],
            design_results['comp_PR_design'],
            '*', ms=20, color='gold',
            markeredgecolor='k', markeredgewidth=1.2,
            zorder=10, label='Design point')
 
    # ── Axes labels and legend ────────────────────────────────────────────────
    ax.set_xlabel('Corrected Mass Flow  $W_c$  (lbm/s)', fontsize=11)
    ax.set_ylabel('Compressor Pressure Ratio  $\pi_c$  (\u2014)', fontsize=11)
    ax.legend(fontsize=9, loc='upper left',
              facecolor='k', labelcolor='white',
              edgecolor='white', framealpha=0.85)
    ax.grid(True, alpha=0.15, color='white', zorder=0)
 
    plt.tight_layout()
    plt.savefig('compressor_map_oplines.png', dpi=150)
    print('Plot saved: compressor_map_oplines.png')
    plt.show()


def plot_compressor_map_both_studies(
        a4_scalars, const_tt4_results, const_fn_results,
        fn_fractions, design_results, prob2):
    """
    Compressor map showing operating lines from both studies, overlaid on the
    full compressor map background (efficiency islands, speed lines, R-lines).

    Background rendered by draw_compressor_map_background().
    Study 1 (const Tt4) — scatter coloured by A4 scalar + connecting line
    Study 2 (const Fn)  — one coloured line per thrust level, optimal marked
    """

    T4_K = design_results['T4_design'] * RANKINE_TO_K

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle(
        'Compressor Map — Variable A\u2084 Operating Lines\n'
        f'Design T\u2084 = {T4_K:.0f} K   |   Sea Level Static\n'
        'Background: adiabatic efficiency islands  |  '
        'Dashed blue = R-lines  |  Grey = speed lines',
        fontsize=11, fontweight='bold')

    # ── Full map background ───────────────────────────────────────────────────
    eff_cf = draw_compressor_map_background(
        ax, prob2, 'DESIGN.comp',
        alpha_idx=0,
        eff_vals=(np.linspace(0.70, 0.85, 25)),
        show_rlines=False,
        show_nclines=True,
        eff_alpha=1.00)
    

    cb_eff = plt.colorbar(eff_cf, ax=ax, shrink=0.55, pad=0.01)
    cb_eff.set_label('Adiabatic efficiency  \u03b7\u2099  (—)', fontsize=9)

    # ── Design point ──────────────────────────────────────────────────────────
    ax.plot(design_results['comp_Wc_design'],
            design_results['comp_PR_design'],
            '*', ms=18, color='gold', markeredgecolor='k',
            zorder=7, label='Design point')

    # ── Study 1 — const Tt4, A4 sweep ────────────────────────────────────────
    Wc1 = np.array([r['comp_Wc'] for r in const_tt4_results])
    PR1 = np.array([r['comp_PR'] for r in const_tt4_results])
    s1  = np.array(a4_scalars)

    # Line first (lower zorder), then scatter on top so dots are readable
    ax.plot(Wc1, PR1, '-', color='royalblue', lw=2.0, zorder=4,
            label='Study 1 — const T\u2084  (A\u2084 sweep)')
    sc1 = ax.scatter(Wc1, PR1, c=s1, cmap='Blues_r',
                     s=25, zorder=5, edgecolor='none',
                     vmin=s1.min(), vmax=s1.max())
    cb1 = plt.colorbar(sc1, ax=ax, shrink=0.35, pad=0.08)
    cb1.set_label('A\u2084 scalar\n(Study 1)', fontsize=8)

    # ── Study 2 — const Fn, one line per thrust level ─────────────────────────
    colors_fn = cm.autumn(np.linspace(0.15, 0.85, len(fn_fractions)))

    for fi, frac in enumerate(fn_fractions):
        sweep = const_fn_results[frac]['sweep']
        Wc2   = np.array([r['comp_Wc'] for r in sweep])
        PR2   = np.array([r['comp_PR'] for r in sweep])

        ax.plot(Wc2, PR2, '-', color=colors_fn[fi], lw=2.0, zorder=4,
                label=f'Study 2 — {int(frac * 100)}% F\u2099')

        # Optimal point — star marker
        mi = int(np.argmin([r['TSFC'] for r in sweep]))
        ax.plot(Wc2[mi], PR2[mi], '*', ms=13,
                color=colors_fn[fi], markeredgecolor='k',
                zorder=6,
                label=f'  \u2605 opt A\u2084={a4_scalars[mi]:.2f}  '
                      f'TSFC={sweep[mi]["TSFC"]:.4f}')

    ax.set_xlabel('Corrected Mass Flow  $W_c$  (lbm/s)', fontsize=11)
    ax.set_ylabel('Compressor Pressure Ratio  $\pi_c$  (—)', fontsize=11)
    ax.legend(fontsize=7.5, loc='upper left',
              facecolor='white', framealpha=0.9,
              ncol=2)
    ax.grid(True, alpha=0.2, zorder=0)

    plt.tight_layout()
    plt.savefig('compressor_map_both_studies.png', dpi=150)
    print('Plot saved: compressor_map_both_studies.png')
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def extract_design(prob, pt='DESIGN'):
    W_des  = prob.get_val(f'{pt}.inlet.Fl_O:stat:W',  units='lbm/s')[0]
    Pt_des = prob.get_val(f'{pt}.inlet.Fl_O:tot:P',   units='psi')[0]
    Tt_des = prob.get_val(f'{pt}.inlet.Fl_O:tot:T',   units='degR')[0]
    Wc_des = W_des * np.sqrt(Tt_des / 518.67) / (Pt_des / 14.696)
    return {
        'pi_c':           prob.get_val(f'{pt}.perf.OPR')[0],
        'Nmech':          prob.get_val(f'{pt}.shaft.Nmech',       units='rpm')[0],
        'W':              W_des,
        'TSFC':           prob.get_val(f'{pt}.perf.TSFC',         units='lbm/(h*lbf)')[0],
        'T4_design':      prob.get_val(f'{pt}.burner.Fl_O:tot:T', units='degR')[0],
        's_Wp_design':    prob.get_val(f'{pt}.turb.s_Wp')[0],
        'eff_design':    prob.get_val(f'{pt}.turb.eff')[0],
        'comp_PR_design': prob.get_val(f'{pt}.comp.PR')[0],
        'comp_Wc_design': Wc_des,
        'Fn_design':      prob.get_val(f'{pt}.perf.Fn',           units='lbf')[0],
        'T3_design':      prob.get_val(f'{pt}.comp.Fl_O:tot:T',   units='degR')[0],
        'P3_design':      prob.get_val(f'{pt}.comp.Fl_O:tot:P',   units='lbf/inch**2')[0],
    }


def extract_od_point(prob, pt):
    W_od  = prob.get_val(f'{pt}.inlet.Fl_O:stat:W', units='lbm/s')[0]
    Pt_od = prob.get_val(f'{pt}.inlet.Fl_O:tot:P',  units='psi')[0]
    Tt_od = prob.get_val(f'{pt}.inlet.Fl_O:tot:T',  units='degR')[0]
    Wc_od = W_od * np.sqrt(Tt_od / 518.67) / (Pt_od / 14.696)
    return {
        'pi_c':      prob.get_val(f'{pt}.perf.OPR')[0],
        'Nmech':     prob.get_val(f'{pt}.shaft.Nmech',        units='rpm')[0],
        'W':         W_od,
        'Fn':        prob.get_val(f'{pt}.perf.Fn',            units='lbf')[0],
        'TSFC':      prob.get_val(f'{pt}.perf.TSFC',          units='lbm/(h*lbf)')[0],
        'T3':        prob.get_val(f'{pt}.comp.Fl_O:tot:T',    units='degR')[0],
        'P3':        prob.get_val(f'{pt}.comp.Fl_O:tot:P',    units='lbf/inch**2')[0],
        'T4':        prob.get_val(f'{pt}.burner.Fl_O:tot:T',  units='degR')[0],
        'comp_PR':   prob.get_val(f'{pt}.comp.PR')[0],
        'comp_Wc':   Wc_od,
        'turb_PR':   prob.get_val(f'{pt}.turb.PR')[0],        # ← add
        'turb_eff':  prob.get_val(f'{pt}.turb.eff')[0],       # ← add
        'turb_Wp':   prob.get_val(f'{pt}.turb.Wp')[0],        # ← add
        'turb_NpMap': prob.get_val(f'{pt}.turb.map.NpMap')[0], # ← add
        'turb_PRmap': prob.get_val(f'{pt}.turb.map.PRmap')[0], # ← add
    }


def summarise_const_fn(fn_fractions, a4_scalars, const_fn_results,
                       design_results):
    """Print the Study 2 summary table."""
    print('\n' + '=' * 80)
    print(' STUDY 2 SUMMARY — OPTIMAL A4 AT EACH THRUST SETTING')
    print('=' * 80)
    print(f"{'Fn (%)':>7} {'Fn_tgt (kN)':>12} {'Opt A4':>8} "
          f"{'Opt TSFC':>10} {'T4 opt (K)':>11} "
          f"{'T4 fixed (K)':>13} {'\u0394T4 (K)':>8} {'\u0394TSFC (%)':>10}")
    print('-' * 80)
    for frac in fn_fractions:
        d = const_fn_results[frac]
        print(f"  {int(frac*100):5d}  "
              f"{frac * FN_DESIGN * LBF_TO_KN:11.2f}  "
              f"{d['opt_scalar']:8.3f}  "
              f"{d['opt_tsfc']:10.5f}  "
              f"{d['opt_T4'] * RANKINE_TO_K:11.1f}  "
              f"{d['baseline_T4'] * RANKINE_TO_K:13.1f}  "
              f"{(d['baseline_T4'] - d['opt_T4']) * RANKINE_TO_K:8.1f}  "
              f"{(d['baseline_tsfc'] - d['opt_tsfc']) / d['baseline_tsfc'] * 100:10.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── PROBLEM 1: STUDY 1 — CONSTANT Tt4 ────────────────────────────────────
    print('\n' + '█' * 60)
    print(' PROBLEM 1 — Constant Tt4, Variable A4')
    print('█' * 60)

    prob1 = om.Problem()
    mp1   = prob1.model = MPConstTt4()
    prob1.setup(check=False)

    # Design inputs
    prob1.set_val('DESIGN.fc.alt',            0.0,    units='ft')
    prob1.set_val('DESIGN.fc.MN',             0.000001)
    prob1.set_val('DESIGN.balance.Fn_target', FN_DESIGN, units='lbf')
    prob1.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
    prob1.set_val('DESIGN.comp.PR',           13.5)
    prob1.set_val('DESIGN.comp.eff',          0.83)
    prob1.set_val('DESIGN.turb.eff',          0.90)
    prob1['DESIGN.balance.FAR']     = 0.0175506829934
    prob1['DESIGN.balance.W']       = 168.453135137
    prob1['DESIGN.balance.turb_PR'] = 4.46138725662
    prob1['DESIGN.fc.balance.Pt']   = 14.6955113159
    prob1['DESIGN.fc.balance.Tt']   = 518.665288153

    # OD initial guesses
    for pt in mp1.a4_pts:
        prob1[pt + '.balance.FAR']   = 0.017
        prob1[pt + '.balance.Nmech'] = 8070.0
        prob1[pt + '.balance.W']     = 168.0
        prob1[pt + '.fc.balance.Pt'] = 14.696
        prob1[pt + '.fc.balance.Tt'] = 518.67
        prob1.set_val(pt + '.balance.T4_target', 2370.0, units='degR')

    prob1.set_solver_print(level=-1)
    prob1.set_solver_print(level=2, depth=1)

    # Step 1a — solve design, freeze OD
    print('\n── Step 1a: Design point ──')
    for pt in mp1.a4_pts:
        prob1.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 0
    t0 = time.time()
    prob1.run_model()
    print(f'Design solved in {time.time() - t0:.1f} s')

    design_results = extract_design(prob1)
    viewer(prob1, 'DESIGN')
    print(f"\nDesign PR = {design_results['pi_c']:.3f}")
    print(f"Design s_Wp = {design_results['s_Wp_design']:.6f}")

    # Step 1b — set s_Wp per point and run sweep
    print('\n── Step 1b: A4 sweep ──')
    for pt in mp1.a4_pts:
        prob1.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 15

    for pt, a4_scalar in zip(mp1.a4_pts, mp1.a4_scalars):
        scaled_s_Wp = design_results['s_Wp_design'] * a4_scalar
        prob1.set_val(f'{pt}.turb.s_Wp', scaled_s_Wp)
        prob1[pt + '.balance.Nmech'] = (design_results['Nmech']
                                        * (1.0 / a4_scalar) ** 0.3)

    t1 = time.time()
    prob1.run_model()
    print(f'Sweep solved in {time.time() - t1:.1f} s')

    # Extract Study 1 results
    const_tt4_results = []
    for pt, a4_scalar in zip(mp1.a4_pts, mp1.a4_scalars):
        r = extract_od_point(prob1, pt)
        r['scalar'] = a4_scalar
        const_tt4_results.append(r)

    print('\n' + '=' * 80)
    print(' STUDY 1 RESULTS — Constant Tt4')
    print('=' * 80)
    print(f"{'A4 scl':>7} {'PR':>8} {'Nmech':>9} {'W kg/s':>8} "
          f"{'Fn kN':>8} {'TSFC':>9} {'T3 K':>7} {'T4 K':>7}")
    print('-' * 80)
    for r in const_tt4_results:
        print(f"  {r['scalar']:5.3f}  {r['pi_c']:8.3f}  {r['Nmech']:8.1f}  "
            f"{r['W'] * LBM_S_TO_KG_S:7.2f}  "
            f"{r['Fn'] * LBF_TO_KN:7.2f}  "
            f"{r['TSFC']:9.5f}  "
            f"{r['T3'] * RANKINE_TO_K:6.1f}  "
            f"{r['T4'] * RANKINE_TO_K:6.1f}  "
            f"turb_PR={r['turb_PR']:.3f}  eff={r['turb_eff']:.4f}  "
            f"Wp={r['turb_Wp']:.3f}  NpMap={r['turb_NpMap']:.1f}  PRmap={r['turb_PRmap']:.3f}")

    # ── PROBLEM 2: STUDY 2 — CONSTANT THRUST ─────────────────────────────────
    print('\n' + '█' * 60)
    print(' PROBLEM 2 — Constant Fn, Variable A4 and Tt4')
    print('█' * 60)

    prob2 = om.Problem()
    mp2   = prob2.model = MPConstFn()
    prob2.setup(check=False)

    # Transfer design point values from prob1 so DESIGN can be frozen
    for v in ['DESIGN.balance.FAR', 'DESIGN.balance.W',
              'DESIGN.balance.turb_PR', 'DESIGN.fc.balance.Pt',
              'DESIGN.fc.balance.Tt']:
        prob2[v] = prob1[v]
    prob2.set_val('DESIGN.fc.alt',            0.0,    units='ft')
    prob2.set_val('DESIGN.fc.MN',             0.000001)
    prob2.set_val('DESIGN.balance.Fn_target', FN_DESIGN, units='lbf')
    prob2.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
    prob2.set_val('DESIGN.comp.PR',           13.5)
    prob2.set_val('DESIGN.comp.eff',          0.83)
    prob2.set_val('DESIGN.turb.eff',          0.86)

    # OD initial guesses
    for pt in mp2.od_pts:
        prob2[pt + '.balance.FAR']   = 0.017
        prob2[pt + '.balance.Nmech'] = 8070.0
        prob2[pt + '.balance.W']     = 168.0
        prob2[pt + '.fc.balance.Pt'] = 14.696
        prob2[pt + '.fc.balance.Tt'] = 518.67

    prob2.set_solver_print(level=-1)
    prob2.set_solver_print(level=2, depth=1)

    # Step 2a — solve DESIGN in prob2, freeze all OD
    print('\n── Step 2a: Design point (prob2) ──')
    for pt in mp2.od_pts:
        prob2.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 0
    t2 = time.time()
    prob2.run_model()
    print(f'Design solved in {time.time() - t2:.1f} s')

    design_results2 = extract_design(prob2)
    # Should match prob1 design exactly
    print(f"Design PR (prob2) = {design_results2['pi_c']:.3f}  "
          f"(expect {design_results['pi_c']:.3f})")

    # Step 2b — set s_Wp and Fn_target per point and run sweep
    print('\n── Step 2b: Constant-Fn sweep ──')
    for pt in mp2.od_pts:
        prob2.model._get_subsystem(pt).nonlinear_solver.options['maxiter'] = 15

    for fi, frac in enumerate(FN_FRACTIONS):
        Fn_target = frac * FN_DESIGN
        for ai, a4_scalar in enumerate(A4_SCALARS):
            pt = f'FN{fi:02d}_A4_{ai:03d}'
            scaled_s_Wp = design_results2['s_Wp_design'] * a4_scalar
            prob2.set_val(f'{pt}.turb.s_Wp', scaled_s_Wp)
            prob2.set_val(f'{pt}.balance.Fn_target', Fn_target, units='lbf')
            # Warm-start: scale Nmech and FAR with thrust fraction and A4
            prob2[pt + '.balance.Nmech'] = (design_results2['Nmech']
                                            * frac ** 0.5
                                            * (1.0 / a4_scalar) ** 0.3)
            prob2[pt + '.balance.FAR']   = 0.017 * frac

    t3 = time.time()
    prob2.run_model()
    print(f'Sweep solved in {time.time() - t3:.1f} s')

    # Extract Study 2 results — organised by thrust level
    const_fn_results = {}
    # Index of A4 scalar closest to 1.0 — the fixed geometry baseline
    baseline_ai = int(np.argmin(np.abs(np.array(A4_SCALARS) - 1.0)))

    for fi, frac in enumerate(FN_FRACTIONS):
        sweep = []
        for ai, a4_scalar in enumerate(A4_SCALARS):
            pt = f'FN{fi:02d}_A4_{ai:03d}'
            r  = extract_od_point(prob2, pt)
            r['scalar'] = a4_scalar
            sweep.append(r)

        tsfc_arr = np.array([r['TSFC'] for r in sweep])
        opt_idx  = int(np.argmin(tsfc_arr))

        const_fn_results[frac] = {
            'sweep':          sweep,
            'opt_idx':        opt_idx,
            'opt_scalar':     A4_SCALARS[opt_idx],
            'opt_tsfc':       tsfc_arr[opt_idx],
            'opt_T4':         sweep[opt_idx]['T4'],
            'baseline_tsfc':  tsfc_arr[baseline_ai],
            'baseline_T4':    sweep[baseline_ai]['T4'],
        }

    summarise_const_fn(FN_FRACTIONS, A4_SCALARS, const_fn_results,
                       design_results)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_const_tt4(A4_SCALARS, const_tt4_results, design_results)
    plot_const_fn(FN_FRACTIONS, A4_SCALARS, const_fn_results, design_results)
    plot_compressor_map_both_studies(
        A4_SCALARS, const_tt4_results, const_fn_results,
        FN_FRACTIONS, design_results, prob2)

    plot_compressor_map_with_oplines(
        prob2, 'DESIGN.comp',
        const_fn_results, FN_FRACTIONS, A4_SCALARS,
        design_results,
        const_tt4_results=const_tt4_results   # optional — pass None to omit
)