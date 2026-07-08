import numpy as np
import cantera as ct
import openmdao.api as om
import pint

ureg = pint.UnitRegistry()
Q_   = ureg.Quantity

DPQP_FIXED = 0.06

_GRI30 = ct.Solution("gri30.yaml")

def rho_total(Tt_K, Pt_Pa):
    Tt_K  = max(float(Tt_K),  1.0)
    Pt_Pa = max(float(Pt_Pa), 1.0)
    _GRI30.TPX = Tt_K, Pt_Pa, "O2:0.21, N2:0.79"
    return float(_GRI30.density)

def calc_CdA_liner(Tt3_K, Pt3_Pa, mdot_si, dPqP_ref=DPQP_FIXED):
    rho_t3 = rho_total(Tt3_K, Pt3_Pa)
    dPt    = dPqP_ref * Pt3_Pa
    return float(mdot_si / np.sqrt(2.0 * rho_t3 * dPt))

class PhysicsCombustor(om.ExplicitComponent):
    def setup(self):
        self.add_input('W',   val=100.0, units='lbm/s')
        self.add_input('Tt3', val=800.0, units='degR')
        self.add_input('Pt3', val=100.0, units='psi')
        self.add_input('CdA', val=1e-3)
        self.add_output('dPqP', val=DPQP_FIXED, lower=1e-4, upper=0.49)
        self.declare_partials('dPqP', ['W', 'Tt3', 'Pt3', 'CdA'],
                              method='fd', step=1e-5, step_calc='rel')

    def compute(self, inputs, outputs):
        W_si   = Q_(float(inputs['W'][0]),   'lb/s').to('kg/s').magnitude
        Tt3_K  = Q_(float(inputs['Tt3'][0]), 'degR').to('K').magnitude
        Pt3_Pa = Q_(float(inputs['Pt3'][0]), 'psi').to('Pa').magnitude
        CdA    = float(inputs['CdA'][0])
        if CdA < 1e-10:
            outputs['dPqP'] = DPQP_FIXED
            return
        rho  = rho_total(Tt3_K, Pt3_Pa)
        dPt  = (W_si / CdA) ** 2 / (2.0 * rho)
        outputs['dPqP'] = float(np.clip(dPt / Pt3_Pa, 1e-4, 0.49))
