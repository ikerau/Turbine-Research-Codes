"""
combustor_A4_sweep.py
---------------------
Combustor model with full HPT/HPC power balance loop.

Two analyses:
  1. A4 sweep at design FAR — 9 separate figures
  2. Lean blowout sweep at 3 A4 values — FAR decreased until blowout

Fixed inputs:  Tt2, Pt2  (ambient, sea-level ISA takeoff)
Variable:      A4, FAR
All float:     mdot_air, Pt4, Pt3, Tt3, Tt4

Three coupled constraints:
  1. Liner orifice:     mdot_air = CdA_liner * sqrt(2 * rho_ann * (Pt3-Pt4))
  2. Choked nozzle:     mdot_tot = choked_factor(A4) * Pt4 / sqrt(Tt4)
  3. Power balance:     mdot_tot * cp_t * Tt4 * tau_t = mdot_air * cp_c * (Tt3-Tt2)
     + isentropic HPC:  Pt3/Pt2 = (Tt3/Tt2)^(gamma_c * eta_pc / (gamma_c-1))
     + turbine:         tau_t = 1 - (A4/An)^exponent

Blowout criterion: eta_b < 0.50  OR  T_PZ < Tt3 + 200K

Design point: CFM56-7B27 take-off (Villette et al. 2024, Table A1)
Mechanism:    nDodecane_Reitz.yaml (nDodecane_IG), fuel c12h26
              CO, UHC, eta_b available. NOx not available.
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import cantera as ct
from scipy.optimize import brentq
import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MECH  = "nDodecane_Reitz.yaml"
PHASE = "nDodecane_IG"
FUEL  = "c12h26"

# CFM56-7B27 geometry
A_REF = 0.160; L = 0.178
LR_PZ = 0.2029; LR_SZ = 0.1979; LR_DZ = 0.5992
AR_PZ = 0.3082; AR_SZ = 0.2302; AR_DZ = 0.4615
V_PZ  = A_REF * L * LR_PZ
V_SZ  = A_REF * L * LR_SZ
V_DZ  = A_REF * L * LR_DZ

ALPHA   = 0.70               # liner-to-reference area ratio A_L / A_ref
A_FLAME = A_REF * ALPHA      # flame tube cross-sectional area [m²]
L_PZ    = L * LR_PZ          # PZ axial length [m]

# Design point (takeoff, Table A1)
PT3_DES  = 28.50e5   # Pa
TT3_DES  = 795.0     # K
MDOT_DES = 47.47     # kg/s  (for CdA back-calc only)
FAR_DES  = 0.026
DPQP_DES = 0.0501
TT4_EST  = 1749.0    # K  initial Tt4 estimate

# Ambient
TT2    = 288.0
PT2    = 101325.0
ETA_PT = 0.89

# Blowout thresholds
ETA_B_BLOWOUT = 0.50
DT_PZ_BLOWOUT = 200.0   # K above Tt3

# ---------------------------------------------------------------------------
# Cantera helpers
# ---------------------------------------------------------------------------

def _gas():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ct.Solution(MECH, PHASE)


def _rgas(reactor):
    for attr in ("thermo", "phase"):
        try:
            return getattr(reactor, attr)
        except AttributeError:
            continue
    raise AttributeError("Reactor has neither .thermo nor .phase")


def _far_stoich():
    g = _gas()
    g.set_equivalence_ratio(1.0, fuel=FUEL, oxidizer="O2:0.21, N2:0.79")
    Yf = g.mass_fraction_dict().get(FUEL, 0.0)
    return Yf / (1.0 - Yf)


def _compute_LHV():
    g = _gas()
    g.set_equivalence_ratio(1.0, fuel=FUEL, oxidizer="O2:1.0, N2:3.76")
    g.TP = 298.15, 101325.0
    Yf = g.mass_fraction_dict().get(FUEL, 0.0)
    hR = g.enthalpy_mass
    g.equilibrate("HP")
    g.TP = 423.15, 101325.0
    return float((hR - g.enthalpy_mass) / Yf)


FAR_ST = _far_stoich()
LHV    = _compute_LHV()

# ---------------------------------------------------------------------------
# Module 1: Design-point back-calculations
# ---------------------------------------------------------------------------

def _air_props(Tt, Pt):
    g = ct.Solution("gri30.yaml")
    g.TPX = Tt, Pt, "O2:0.21, N2:0.79"
    return float(g.cp_mass), float(g.cp_mass / g.cv_mass)


def _hot_gas_props(Tt4, Pt4, FAR):
    g = _gas()
    g.set_equivalence_ratio(FAR / FAR_ST, fuel=FUEL, oxidizer="O2:0.21, N2:0.79")
    g.TP = Tt4, Pt4
    g.equilibrate("TP")
    cp    = float(g.cp_mass)
    gamma = float(g.cp_mass / g.cv_mass)
    R     = float(ct.gas_constant / g.mean_molecular_weight)
    return cp, gamma, R


def _turbine_exponent(gamma_t):
    return 2.0 * ETA_PT * (gamma_t - 1.0) / ((2.0 - ETA_PT) * gamma_t + ETA_PT)


def _annulus_rho(Tt3, Pt3):
    g = ct.Solution("gri30.yaml")
    g.TPX = Tt3, Pt3, "O2:0.21, N2:0.79"
    return float(g.density)


def _choked_factor(Pt4_est, FAR, Tt4_est=TT4_EST):
    _, gamma, R = _hot_gas_props(Tt4_est, Pt4_est, FAR)
    exp    = (gamma + 1.0) / (2.0 * (gamma - 1.0))
    factor = (2.0 / (gamma + 1.0)) ** exp
    return float(np.sqrt(gamma / R) * factor)


def back_calc_design(verbose=True):
    """Back-calculate CdA_liner, A4_nom, An, eta_pc from design point."""
    Pt4_nom  = PT3_DES * (1.0 - DPQP_DES)
    rho_ann  = _annulus_rho(TT3_DES, PT3_DES)
    dPt_nom  = PT3_DES - Pt4_nom
    CdA_liner = MDOT_DES / np.sqrt(2.0 * rho_ann * dPt_nom)

    mdot_tot = MDOT_DES * (1.0 + FAR_DES)
    cf       = _choked_factor(Pt4_nom, FAR_DES, TT4_EST)
    A4_nom   = mdot_tot * np.sqrt(TT4_EST) / (cf * Pt4_nom)

    cp_c, gamma_c = _air_props((TT2 + TT3_DES) / 2.0, (PT2 + PT3_DES) / 2.0)
    eta_pc = (math.log(PT3_DES / PT2) /
              (gamma_c / (gamma_c - 1.0) * math.log(TT3_DES / TT2)))

    cp_t, gamma_t, _ = _hot_gas_props(TT4_EST, Pt4_nom, FAR_DES)
    exp_t     = _turbine_exponent(gamma_t)
    tau_t_des = (MDOT_DES * cp_c * (TT3_DES - TT2) /
                 (MDOT_DES * (1.0 + FAR_DES) * cp_t * TT4_EST))
    An = A4_nom / (1.0 - tau_t_des) ** (1.0 / exp_t)

    if verbose:
        print(f"  CdA_liner = {CdA_liner:.5f} m²")
        print(f"  A4_nom    = {A4_nom*1e4:.2f} cm²")
        print(f"  An        = {An*1e4:.2f} cm²")
        print(f"  eta_pc    = {eta_pc:.4f}")
        print(f"  tau_t_des = {tau_t_des:.4f}  exp_t = {exp_t:.5f}")

    return {
        "CdA_liner": float(CdA_liner),
        "A4_nom":    float(A4_nom),
        "An":        float(An),
        "eta_pc":    float(eta_pc),
        "gamma_c":   float(gamma_c),
        "cp_c":      float(cp_c),
        "Pt4_nom":   float(Pt4_nom),
    }


# ---------------------------------------------------------------------------
# Module 2: Inner loop — solve (mdot_air, Pt4) given (Pt3, Tt3, FAR)
# ---------------------------------------------------------------------------

def solve_mdot_Pt4(A4, Pt3, Tt3, CdA_liner, FAR, Tt4_est=TT4_EST, tol=1e-6):
    cf      = _choked_factor(Pt3 * 0.95, FAR, Tt4_est)
    rho_ann = _annulus_rho(Tt3, Pt3)

    def get_Pt4(mdot_air):
        return float(mdot_air * (1.0 + FAR) * np.sqrt(Tt4_est) / (cf * A4))

    def residual(mdot_air):
        Pt4 = get_Pt4(mdot_air)
        dPt = Pt3 - Pt4
        if dPt <= 0.0:
            return 1e6
        return mdot_air - CdA_liner * np.sqrt(2.0 * rho_ann * dPt)

    mdot_hi = None
    for m in np.linspace(0.5, 400.0, 800):
        if get_Pt4(m) >= Pt3:
            mdot_hi = m * 0.999
            break
    if mdot_hi is None:
        mdot_hi = 300.0

    mdot_sol = brentq(residual, 0.1, mdot_hi, xtol=tol)
    return float(mdot_sol), float(get_Pt4(mdot_sol))


# ---------------------------------------------------------------------------
# Module 3: Power balance
# ---------------------------------------------------------------------------

def power_balance(Tt4, Pt4, mdot_air, A4, design, FAR):
    An      = design["An"]
    eta_pc  = design["eta_pc"]
    gamma_c = design["gamma_c"]
    cp_c    = design["cp_c"]

    cp_t, gamma_t, _ = _hot_gas_props(Tt4, Pt4, FAR)
    exp_t  = _turbine_exponent(gamma_t)
    tau_t  = 1.0 - (A4 / An) ** exp_t
    Tt45   = Tt4 * (1.0 - tau_t)

    mdot_tot = mdot_air * (1.0 + FAR)
    Tt3_new  = TT2 + mdot_tot * cp_t * Tt4 * tau_t / (mdot_air * cp_c)
    OPR_new  = (Tt3_new / TT2) ** (gamma_c * eta_pc / (gamma_c - 1.0))
    Pt3_new  = OPR_new * PT2

    return float(Tt3_new), float(Pt3_new), float(tau_t), float(Tt45)


# ---------------------------------------------------------------------------
# Module 4: Equilibrium Tt4 (fast proxy for power balance loop)
# ---------------------------------------------------------------------------

def _equil_Tt4(Tt3, Pt4, FAR):
    g = _gas()
    g.set_equivalence_ratio(FAR / FAR_ST, fuel=FUEL, oxidizer="O2:0.21, N2:0.79")
    g.TP = Tt3, Pt4
    g.equilibrate("HP")
    return float(g.T)


# ---------------------------------------------------------------------------
# Module 5: Ignition delay
# ---------------------------------------------------------------------------

def ignition_delay(Tt3, Pt4, phi_PZ, t_max=1.0, dTdt_threshold=1e4):
    g = _gas()
    g.set_equivalence_ratio(phi_PZ, fuel=FUEL, oxidizer="O2:0.21, N2:0.79")
    g.TP = Tt3, Pt4
    r   = ct.IdealGasConstPressureReactor(g)
    net = ct.ReactorNet([r])
    net.atol = 1e-14; net.rtol = 1e-10

    checkpoints = np.unique(np.concatenate([
        np.logspace(-8, -3, 100), np.logspace(-3, np.log10(t_max), 80)
    ]))
    T_prev = _rgas(r).T; t_prev = 0.0
    dTdt_max = 0.0; t_ign = np.nan

    for t_next in checkpoints:
        try:
            net.advance(t_next)
        except Exception:
            break
        T_cur = _rgas(r).T
        dt    = t_next - t_prev
        dTdt  = (T_cur - T_prev) / dt if dt > 0 else 0.0
        if dTdt > dTdt_max:
            dTdt_max = dTdt; t_ign = t_next
        if T_cur > Tt3 + 500.0 and dTdt < dTdt_max * 0.1:
            break
        T_prev = T_cur; t_prev = t_next

    return float(t_ign) if dTdt_max > dTdt_threshold else np.nan


# ---------------------------------------------------------------------------
# Module 6: CRN helpers
# ---------------------------------------------------------------------------

def _emission_indices(phase_obj, FAR):
    Y    = phase_obj.mass_fraction_dict()
    Y_CO = Y.get("co", Y.get("CO", 0.0))
    skip = {"co2","co","CO2","CO","n2","N2","o2","O2","h2o","H2O",
            "h2","H2","oh","OH","o","O","h","H","ho2","HO2","h2o2","H2O2"}
    Y_UHC = sum(v for k, v in Y.items()
                if k not in skip and any(c in k for c in ["c","C"]))
    scale = (1.0 + FAR) / FAR * 1000.0
    return {"EI_CO": float(Y_CO * scale), "EI_UHC": float(Y_UHC * scale)}


def _eta_b(phase_obj, FAR):
    X    = phase_obj.mole_fraction_dict()
    MW   = phase_obj.mean_molecular_weight
    X_CO = X.get("co", X.get("CO", 0.0))
    skip = {"co2","co","CO2","CO","n2","N2","o2","O2","h2o","H2O",
            "h2","H2","oh","OH","o","O","h","H","ho2","HO2","h2o2","H2O2"}
    X_UHC = sum(v for k, v in X.items()
                if k not in skip and any(c in k for c in ["c","C"]))
    n_per_kgfuel = (1000.0 / MW) * (1.0 + FAR) / FAR
    return float(np.clip(
        (LHV - X_CO*n_per_kgfuel*282965. - X_UHC*n_per_kgfuel*802396.) / LHV,
        0.0, 1.0
    ))


# ---------------------------------------------------------------------------
# Module 7: CRN solver
# ---------------------------------------------------------------------------

def run_crn(Pt4, Tt3, mdot_air, FAR, t_end=2.0, verbose=False):
    """Three-zone serial PSR CRN at fixed CFM56 geometry."""
    mdot_fuel  = mdot_air * FAR
    mdot_tot   = mdot_air + mdot_fuel
    ma_PZ      = mdot_air * AR_PZ
    ma_SZ      = mdot_air * AR_SZ
    ma_DZ      = mdot_air * AR_DZ
    mdot_in_PZ = ma_PZ + mdot_fuel
    mdot_in_SZ = mdot_in_PZ + ma_SZ
    mdot_in_DZ = mdot_tot
    phi_PZ = FAR / (AR_PZ * FAR_ST)
    phi_SZ = FAR / ((AR_PZ + AR_SZ) * FAR_ST)
    phi_DZ = FAR / FAR_ST

    def equil_T(phi):
        g = _gas()
        g.set_equivalence_ratio(min(phi, 2.0), fuel=FUEL,
                                oxidizer="O2:0.21, N2:0.79")
        g.TP = Tt3, Pt4; g.equilibrate("HP")
        return float(g.T)

    def make_r(phi, T_init, V):
        g = _gas()
        g.set_equivalence_ratio(min(phi, 2.0), fuel=FUEL,
                                oxidizer="O2:0.21, N2:0.79")
        g.TP = T_init, Pt4
        return ct.IdealGasReactor(g, volume=V)

    r_PZ = make_r(phi_PZ, equil_T(phi_PZ), V_PZ)
    r_SZ = make_r(phi_SZ, equil_T(phi_SZ), V_SZ)
    r_DZ = make_r(phi_DZ, equil_T(phi_DZ), V_DZ)

    def res_rv(X, T=None):
        g = _gas(); g.TPX = (T or Tt3), Pt4, X
        return ct.Reservoir(g)

    air_res  = res_rv("O2:0.21, N2:0.79")
    fuel_res = res_rv(f"{FUEL}:1.0")
    exh_res  = res_rv("O2:0.21, N2:0.79")
    ign_res  = res_rv("H2:1.0")

    ct.MassFlowController(air_res,  r_PZ, mdot=ma_PZ)
    ct.MassFlowController(air_res,  r_SZ, mdot=ma_SZ)
    ct.MassFlowController(air_res,  r_DZ, mdot=ma_DZ)
    ct.MassFlowController(fuel_res, r_PZ, mdot=mdot_fuel)
    ct.MassFlowController(r_PZ,    r_SZ, mdot=mdot_in_PZ)
    ct.MassFlowController(r_SZ,    r_DZ, mdot=mdot_in_SZ)
    ct.Valve(r_DZ, exh_res, K=1.0)

    A_ign = 0.05 * mdot_fuel; t0 = 5e-4; tau = 2e-4
    ct.MassFlowController(ign_res, r_PZ,
                          mdot=lambda t: A_ign * np.exp(-((t-t0)/tau)**2))

    net = ct.ReactorNet([r_PZ, r_SZ, r_DZ])
    net.atol = 1e-14; net.rtol = 1e-9

    checkpoints = np.unique(np.concatenate([
        np.logspace(-6, -2, 30), np.linspace(0.01, min(0.5, t_end), 30), [t_end]
    ]))
    T_prev = np.array([_rgas(r_PZ).T, _rgas(r_SZ).T, _rgas(r_DZ).T])
    for t_next in checkpoints:
        net.advance(t_next)
        T_cur = np.array([_rgas(r_PZ).T, _rgas(r_SZ).T, _rgas(r_DZ).T])
        if t_next > 0.05 and np.max(np.abs(T_cur - T_prev)) < 0.5:
            if verbose:
                print(f"  CRN t={t_next:.3f}s "
                      f"T=[{T_cur[0]:.0f},{T_cur[1]:.0f},{T_cur[2]:.0f}]K")
            break
        T_prev = T_cur

    zone_out = {}
    for name, reactor, mdot_in, V in [
        ("PZ", r_PZ, mdot_in_PZ, V_PZ),
        ("SZ", r_SZ, mdot_in_SZ, V_SZ),
        ("DZ", r_DZ, mdot_in_DZ, V_DZ),
    ]:
        g = _rgas(reactor)
        tau_res = g.density * V / mdot_in
        try:
            phi_z = g.equivalence_ratio(fuel=FUEL, oxidizer="O2:0.21, N2:0.79",
                                        basis="mole")
        except Exception:
            phi_z = 0.0
        zone_out[name] = {
            "T": float(g.T), "P": float(g.P), "phi": float(phi_z),
            "rho": float(g.density), "tau_flow": float(tau_res),
            **_emission_indices(g, FAR),
        }

    g_exit = _rgas(r_DZ)
    return {
        "zones":  zone_out, "T_exit": float(g_exit.T),
        "eta_b":  _eta_b(g_exit, FAR), "phi_PZ": phi_PZ,
        **_emission_indices(g_exit, FAR),
    }


# ---------------------------------------------------------------------------
# Module 8: Full coupled solver at one (A4, FAR) point
# ---------------------------------------------------------------------------

def solve_A4_point(A4, design, FAR, relax=0.5, tol_T=0.05, tol_P=50.0,
                   max_iter=40, verbose=False):
    """
    Solve full HPT/HPC/combustor system at given (A4, FAR).

    Outer loop: iterate (Pt3, Tt3) using equilibrium Tt4 (fast).
    Once converged: single CRN call for accurate Tt4, emissions, Da.
    """
    CdA_liner = design["CdA_liner"]
    Pt3_k = PT3_DES; Tt3_k = TT3_DES

    for k in range(max_iter):
        mdot_air, Pt4 = solve_mdot_Pt4(A4, Pt3_k, Tt3_k, CdA_liner, FAR)
        Tt4           = _equil_Tt4(Tt3_k, Pt4, FAR)
        Tt3_new, Pt3_new, tau_t, Tt45 = power_balance(
            Tt4, Pt4, mdot_air, A4, design, FAR
        )

        dT = abs(Tt3_new - Tt3_k)
        dP = abs(Pt3_new - Pt3_k)

        if verbose:
            print(f"  k={k:2d}  Tt3={Tt3_new:.2f}K  Pt3={Pt3_new/1e5:.3f}bar  "
                  f"mdot={mdot_air:.3f}  Tt4={Tt4:.1f}  dT={dT:.3f}")

        if dT < tol_T and dP < tol_P:
            # Single CRN call at converged state
            crn     = run_crn(Pt4, Tt3_new, mdot_air, FAR, verbose=False)
            Tt4_crn = crn["T_exit"]

            # Final power balance with CRN Tt4
            Tt3_f, Pt3_f, tau_t_f, Tt45_f = power_balance(
                Tt4_crn, Pt4, mdot_air, A4, design, FAR
            )

            phi_PZ  = crn["phi_PZ"]

            # Overall flow Damkohler across full liner length
            #
            # tau_flow = L / U_ref
            #   L     = full liner length
            #   U_ref = mdot_tot / (rho_inlet * A_FLAME)
            #   rho_inlet at cold inlet conditions (Tt3, Pt4)
            #
            # tau_chem = ignition delay at PZ EXIT temperature and Pt4
            #   Using T_PZ (hot, partially reacted) rather than cold Tt3
            #   because SZ/DZ see already-burning gas — chemistry is faster
            #   phi_overall = FAR / FAR_ST (overall equivalence ratio)
            #
            # Da > 1: sufficient residence time — stable combustion
            # Da < 1: flow too fast for chemistry — approaching blowout
            g_inlet = ct.Solution("gri30.yaml")
            g_inlet.TPX = Tt3_f, Pt4, "O2:0.21, N2:0.79"
            rho_inlet = float(g_inlet.density)
            mdot_tot_comb = mdot_air * (1.0 + FAR)
            U_ref    = mdot_tot_comb / (rho_inlet * A_FLAME)
            tau_flow = L / U_ref

            T_PZ_exit  = crn["zones"]["PZ"]["T"]
            phi_overall = FAR / FAR_ST
            tau_ign    = ignition_delay(T_PZ_exit, Pt4, phi_overall)
            Da = tau_flow / tau_ign if not np.isnan(tau_ign) else np.nan

            return {
                "A4":       A4,       "FAR":    FAR,
                "Pt3":      Pt3_f,    "Tt3":    Tt3_f,
                "Pt4":      Pt4,      "Tt4":    Tt4_crn,
                "Tt45":     Tt45_f,   "tau_t":  tau_t_f,
                "dP_P":     (Pt3_f - Pt4) / Pt3_f,
                "OPR":      Pt3_f / PT2,
                "mdot_air": mdot_air,
                "T_PZ":     crn["zones"]["PZ"]["T"],
                "T_SZ":     crn["zones"]["SZ"]["T"],
                "T_DZ":     crn["zones"]["DZ"]["T"],
                "tau_flow": tau_flow,
                "U_ref":    U_ref,
                "tau_ign":  tau_ign,
                "Da":       Da,
                "eta_b":    crn["eta_b"],
                "EI_CO":    crn["EI_CO"],
                "EI_UHC":   crn["EI_UHC"],
                "n_iter":   k + 1,
                "blown_out": False,
            }

        Pt3_k = relax * Pt3_new + (1.0 - relax) * Pt3_k
        Tt3_k = relax * Tt3_new + (1.0 - relax) * Tt3_k

    raise RuntimeError(
        f"Power balance did not converge after {max_iter} iterations "
        f"at A4={A4*1e4:.1f}cm²  FAR={FAR:.4f}  dT={dT:.3f}K"
    )


# ---------------------------------------------------------------------------
# Module 9: A4 sweep at design FAR
# ---------------------------------------------------------------------------

def sweep_A4(design, n_points=11, A4_range=(0.75, 1.25), verbose=False):
    """Sweep A4 at design FAR with full power balance."""
    A4_nom   = design["A4_nom"]
    A4_sweep = np.linspace(A4_nom * A4_range[0],
                           A4_nom * A4_range[1], n_points)
    results  = []

    hdr = (f"{'A4/A4n':>7} {'mdot':>7} {'Pt3[bar]':>9} {'Tt3[K]':>7} "
           f"{'Pt4[bar]':>9} {'Tt4[K]':>7} {'Tt45[K]':>8} "
           f"{'OPR':>6} {'dP/P%':>6} {'Da':>7} {'EI_CO':>7} {'eta_b':>7}")
    print(hdr); print("-" * len(hdr))

    for A4 in A4_sweep:
        ratio = A4 / A4_nom
        try:
            r = solve_A4_point(A4, design, FAR_DES, verbose=verbose)
        except Exception as e:
            print(f"  ratio={ratio:.3f}: failed — {e}")
            continue
        results.append({**r, "ratio": ratio})
        da = f"{r['Da']:.3f}" if not np.isnan(r['Da']) else "  n/a"
        print(f"{ratio:>7.3f} {r['mdot_air']:>7.3f} {r['Pt3']/1e5:>9.3f} "
              f"{r['Tt3']:>7.1f} {r['Pt4']/1e5:>9.3f} {r['Tt4']:>7.1f} "
              f"{r['Tt45']:>8.1f} {r['OPR']:>6.1f} {r['dP_P']*100:>6.2f} "
              f"{da:>7} {r['EI_CO']:>7.3f} {r['eta_b']:>7.4f}")

    return results


# ---------------------------------------------------------------------------
# Module 10: Lean blowout sweep at 3 A4 values
# ---------------------------------------------------------------------------

def blowout_sweep(design, A4_ratios=(0.85, 1.00, 1.15),
                  FAR_start=None, FAR_step=0.001, FAR_min=0.004,
                  verbose=False):
    """
    At each of 3 A4 values, decrease FAR from design until blowout.
    Full power balance at each (A4, FAR) point.

    Blowout: eta_b < ETA_B_BLOWOUT  OR  T_PZ < Tt3 + DT_PZ_BLOWOUT
    """
    if FAR_start is None:
        FAR_start = FAR_DES

    A4_nom   = design["A4_nom"]
    all_data = {}

    for ratio in A4_ratios:
        A4    = ratio * A4_nom
        label = f"A4/A4n = {ratio:.2f}"
        print(f"\n{label}  (A4={A4*1e4:.1f}cm²)")
        print(f"  {'FAR':>6} {'mdot':>7} {'Pt3':>8} {'Tt3':>7} "
              f"{'Tt4':>7} {'T_PZ':>7} {'eta_b':>7} {'Da':>7}  status")
        print(f"  {'-'*70}")

        FAR_current = FAR_start
        points = []

        while FAR_current >= FAR_min:
            try:
                r = solve_A4_point(A4, design, FAR_current, verbose=False)
            except Exception as e:
                print(f"  {FAR_current:.4f}  solver failed: {e}")
                break

            # Check blowout
            blown = (r["eta_b"] < ETA_B_BLOWOUT or
                     r["T_PZ"] < r["Tt3"] + DT_PZ_BLOWOUT)
            r["blown_out"] = blown

            da_str = f"{r['Da']:.3f}" if not np.isnan(r["Da"]) else "  n/a"
            status = "BLOWOUT" if blown else "ok"
            print(f"  {FAR_current:.4f} {r['mdot_air']:>7.3f} "
                  f"{r['Pt3']/1e5:>8.3f} {r['Tt3']:>7.1f} "
                  f"{r['Tt4']:>7.1f} {r['T_PZ']:>7.1f} "
                  f"{r['eta_b']:>7.4f} {da_str:>7}  {status}")

            points.append({**r, "FAR_current": FAR_current})

            if blown:
                break

            FAR_current = round(FAR_current - FAR_step, 4)

        all_data[ratio] = points

    return all_data


# ---------------------------------------------------------------------------
# Module 11: Separate figures for A4 sweep
# ---------------------------------------------------------------------------

def plot_A4_sweep(results, design):
    """Display one figure per quantity from the A4 sweep."""
    if len(results) < 2:
        print("Not enough results to plot."); return

    A4_nom = design["A4_nom"]
    ratios = np.array([r["ratio"]       for r in results])
    nom    = np.argmin(np.abs(ratios - 1.0))

    def get(key, scale=1.0):
        return np.array([r[key] * scale for r in results])

    def finalize(ax, fig):
        ax.axvline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("A4 / A4_nom  [-]")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    print("Displaying A4 sweep figures...")

    # Pt3 and Tt3 on twin axes
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    ax.plot(ratios, get("Pt3", 1e-5), "o-", color="steelblue", lw=1.8, ms=5, label="Pt3")
    ax2.plot(ratios, get("Tt3"), "s--", color="firebrick", lw=1.8, ms=5, label="Tt3")
    ax.scatter([ratios[nom]], [get("Pt3", 1e-5)[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("Pt3 [bar]", color="steelblue")
    ax2.set_ylabel("Tt3 [K]", color="firebrick")
    ax.set_title("Compressor Delivery vs A4", fontsize=10)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs  = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, fontsize=9)
    finalize(ax, fig)

    # OPR
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, get("OPR"), "o-", color="darkgreen", lw=1.8, ms=5)
    ax.scatter([ratios[nom]], [get("OPR")[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("OPR [-]"); ax.set_title("Overall Pressure Ratio vs A4", fontsize=10)
    finalize(ax, fig)

    # mdot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, get("mdot_air"), "o-", color="purple", lw=1.8, ms=5)
    ax.scatter([ratios[nom]], [get("mdot_air")[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("ṁ_air [kg/s]"); ax.set_title("Air Mass Flow vs A4", fontsize=10)
    finalize(ax, fig)

    # Tt4 and Tt45
    fig, ax = plt.subplots(figsize=(7, 5))
    ax2 = ax.twinx()
    ax.plot(ratios, get("Tt4"),  "o-",  color="firebrick",  lw=1.8, ms=5, label="Tt4")
    ax2.plot(ratios, get("Tt45"), "s--", color="darkorange", lw=1.8, ms=5, label="Tt4.5")
    ax.scatter([ratios[nom]], [get("Tt4")[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("Tt4 [K]", color="firebrick")
    ax2.set_ylabel("Tt4.5 [K]", color="darkorange")
    ax.set_title("Turbine Temperatures vs A4", fontsize=10)
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labs  = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labs, fontsize=9)
    finalize(ax, fig)

    # Zone temperatures
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, get("T_PZ"), "o-", color="firebrick",  label="PZ", lw=1.8, ms=5)
    ax.plot(ratios, get("T_SZ"), "s-", color="darkorange", label="SZ", lw=1.8, ms=5)
    ax.plot(ratios, get("T_DZ"), "^-", color="steelblue",  label="DZ", lw=1.8, ms=5)
    ax.scatter([ratios[nom]], [get("T_PZ")[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("Temperature [K]"); ax.set_title("Zone Temperatures vs A4", fontsize=10)
    ax.legend(fontsize=9)
    finalize(ax, fig)

    # dP/P
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, get("dP_P", 100), "o-", color="slategray", lw=1.8, ms=5)
    ax.scatter([ratios[nom]], [get("dP_P", 100)[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("dP/P [%]"); ax.set_title("Combustor Pressure Loss vs A4", fontsize=10)
    finalize(ax, fig)

    # Timescales
    tau_f = np.array([r["tau_flow"]*1e3 for r in results])
    tau_i = np.array([r["tau_ign"]*1e3 if not np.isnan(r["tau_ign"])
                      else np.nan for r in results])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, tau_f, "o-",  color="steelblue", label="τ_flow = L/U_ref  (full liner)", lw=1.8, ms=5)
    ax.plot(ratios, tau_i, "s--", color="firebrick", label="τ_ign (inlet)",        lw=1.8, ms=5)
    ax.set_ylabel("Time [ms]"); ax.set_title("PZ Timescales vs A4", fontsize=10)
    ax.set_yscale("log"); ax.legend(fontsize=9)
    finalize(ax, fig)

    # Da
    Da = np.array([r["Da"] if not np.isnan(r["Da"]) else np.nan for r in results])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, Da, "o-", color="purple", lw=1.8, ms=5)
    ax.axhline(1.0, color="black", ls=":", lw=1.5, alpha=0.6, label="Da=1")
    ax.scatter([ratios[nom]], [Da[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("Da = τ_flow/τ_ign"); ax.set_title("Overall Flow Damköhler  (L/U_ref) / τ_ign(T_PZ, ϕ_overall)", fontsize=10)
    ax.set_yscale("log"); ax.legend(fontsize=9)
    finalize(ax, fig)

    # EI_CO
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ratios, get("EI_CO"), "o-", color="saddlebrown", lw=1.8, ms=5)
    ax.scatter([ratios[nom]], [get("EI_CO")[nom]], color="red", s=60, zorder=5)
    ax.set_ylabel("EI_CO [g/kg fuel]"); ax.set_title("CO Emission Index vs A4", fontsize=10)
    finalize(ax, fig)


# ---------------------------------------------------------------------------
# Module 12: Blowout sweep figure
# ---------------------------------------------------------------------------

def plot_blowout(all_data):
    """Three-panel figure: eta_b, T_PZ, Da vs FAR for each A4 value."""
    colors = {"0.85": "steelblue", "1.00": "firebrick", "1.15": "darkgreen"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CFM56-7B27 Take-off: Lean Blowout Sweep\n"
                 "Full power balance at each (A4, FAR) point",
                 fontsize=11, fontweight="bold")

    for ratio, points in all_data.items():
        if not points:
            continue
        label  = f"A4/A4n={ratio:.2f}"
        color  = colors.get(f"{ratio:.2f}", "gray")
        FAR_v  = np.array([p["FAR_current"] for p in points])
        eta_v  = np.array([p["eta_b"]       for p in points])
        T_PZ_v = np.array([p["T_PZ"]        for p in points])
        Tt3_v  = np.array([p["Tt3"]         for p in points])
        Da_v   = np.array([p["Da"] if not np.isnan(p["Da"]) else np.nan
                           for p in points])

        # Find blowout FAR
        blown_idx = next((i for i, p in enumerate(points) if p["blown_out"]), None)
        FAR_bo = points[blown_idx]["FAR_current"] if blown_idx is not None else None

        for ax, y, ylabel, title, logy in [
            (axes[0], eta_v,          "η_b [-]",         "Combustion Efficiency", False),
            (axes[1], T_PZ_v - Tt3_v, "T_PZ - Tt3 [K]",  "PZ Temperature Rise",   False),
            (axes[2], Da_v,            "Da [-]",           "Damköhler Number",      True),
        ]:
            ax.plot(FAR_v, y, "o-", color=color, lw=1.8, ms=5, label=label)
            if FAR_bo is not None:
                ax.axvline(FAR_bo, color=color, ls="--", lw=1.2, alpha=0.6)
            ax.set_xlabel("FAR [-]"); ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3)
            if logy: ax.set_yscale("log")

    # Add threshold lines
    axes[0].axhline(ETA_B_BLOWOUT, color="black", ls=":", lw=1.5,
                    label=f"Blowout threshold ({ETA_B_BLOWOUT})")
    axes[1].axhline(DT_PZ_BLOWOUT, color="black", ls=":", lw=1.5,
                    label=f"Blowout threshold (+{DT_PZ_BLOWOUT}K)")
    axes[2].axhline(1.0, color="black", ls=":", lw=1.5, label="Da=1")

    for ax in axes:
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CFM56-7B27 Take-off — Design geometry back-calculation")
    print("=" * 60)
    design = back_calc_design(verbose=True)
    print()

    # --- Analysis 1: A4 sweep at design FAR ---
    print("=" * 60)
    print(f"A4 sweep at FAR={FAR_DES:.3f} (design)")
    print("=" * 60)
    sweep_results = sweep_A4(design, n_points=11, A4_range=(0.75, 1.25))

    # --- Analysis 2: Lean blowout at 3 A4 values ---
    print()
    print("=" * 60)
    print("Lean blowout sweep at A4/A4nom = 0.85, 1.00, 1.15")
    print("=" * 60)
    blowout_data = blowout_sweep(
        design,
        A4_ratios  = (0.85, 1.00, 1.15),
        FAR_start  = FAR_DES,
        FAR_step   = 0.001,
        FAR_min    = 0.004,
    )
    plot_A4_sweep(sweep_results, design)
    plot_blowout(blowout_data)

    # Summary
    nom = min(sweep_results, key=lambda r: abs(r["ratio"] - 1.0))
    print(f"\nDesign point summary:")
    print(f"  Pt3={nom['Pt3']/1e5:.3f}bar  Tt3={nom['Tt3']:.1f}K  "
          f"mdot={nom['mdot_air']:.3f}kg/s")
    print(f"  Tt4={nom['Tt4']:.1f}K  Tt45={nom['Tt45']:.1f}K  OPR={nom['OPR']:.1f}")
    print(f"  Da={nom['Da']:.3f}  eta_b={nom['eta_b']:.4f}  "
          f"EI_CO={nom['EI_CO']:.3f}g/kg")
    print(f"\nNote: NOx not available (nDodecane_Reitz has no N-radical chemistry)")