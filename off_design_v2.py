from matplotlib.pylab import gamma
import pint
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, brentq, least_squares
from scipy import interpolate
import cantera as ct
def clear_terminal():
    if os.name == 'nt':
        _ = os.system('cls')
# ---------------- Pint setup ----------------
ureg = pint.UnitRegistry()
from units import Q_, ureg
clear_terminal()

def mach_to_lambda(M, gamma):
    M = np.asarray(float(M) if not hasattr(M, '__len__') else M)
    gp1 = gamma+1
    gm1 = gamma-1
    gm1_gp1 = gm1/gp1
    gp1_2 = gp1/2
    gm1_2 = gm1/2
    term1 = gp1_2*M**2
    term2 = 1 + gm1_2*M**2
    return np.sqrt(term1/term2)
    
def lambda_to_mach(lmbd, gamma, guess=0.5):
    gp1 = gamma+1
    gm1 = gamma-1
    gm1_gp1 = gm1/gp1
    term1 = 2/gp1
    term2_den = 1 - gm1_gp1*lmbd**2
    term2 = lmbd**2 / term2_den
    return np.sqrt(term1*term2)

def a_crit(R, T0, gamma):
    """Return a* (m/s) as magnitude (float). R in J/kg/K, T0 in K."""
    return np.sqrt(2 * (gamma / (gamma + 1)) * R * T0).to('m/s')

def isentropic_p_P0(lmbd, gamma):
    gm1_gp1 = (gamma - 1.0) / (gamma + 1.0)
    g_gm1   = gamma / (gamma - 1.0)
    return float((1.0 - gm1_gp1 * float(lmbd)**2) ** g_gm1)

def isentropic_t_T0(lmbd, gamma):
    gm1_gp1 = (gamma - 1.0) / (gamma + 1.0)
    return float(1.0 - gm1_gp1 * float(lmbd)**2)

# === Compressible flowrate (magnitudes) ===
def compressible_flowrate(Area, P0, T0, R, Mach, gamma):
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    gm1_2 = 0.5 * gm1
    expn  = gp1 / (2.0 * gm1)

    term = 1.0 + gm1_2 * Mach * Mach

    return (Area * P0 / np.sqrt(T0)) * np.sqrt(gamma / R) * Mach * term**(-expn)


def P0_from_mdot(Area, T0, R, Mach, mdot, gamma):
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    gm1_2 = 0.5 * gm1
    expn  = gp1 / (2.0 * gm1)

    # Convert units once
    Rm    = R.to('J/kg/K').magnitude
    md    = mdot.to('kg/s').magnitude
    A     = Area.to('m^2').magnitude
    T0m   = T0.to('K').magnitude

    term = 1.0 + gm1_2 * Mach * Mach

    denom = A * np.sqrt(gamma / (Rm * T0m)) * Mach * term**(-expn)

    return Q_(md / denom, 'Pa')


def area_from_flowrate(mdot, vel, P0, T0, R, gamma):
    """Return Area magnitude (m^2)."""

    # Convert units once
    md  = mdot.to('kg/s').magnitude
    P0m = P0.to('Pa').magnitude
    T0m = T0.to('K').magnitude
    Rm  = R.to('J/kg/K').magnitude

    acrit = a_crit(R, T0, gamma).to('m/s').magnitude
    velm  = vel.to('m/s').magnitude

    lmbd = velm / acrit
    Mach = lambda_to_mach(lmbd, gamma)

    def fun(A):
        return compressible_flowrate(A, P0m, T0m, Rm, Mach, gamma) - md

    area = float(fsolve(fun, 0.1)[0])

    return Q_(area, 'm^2')


def velocity_from_flowrate(mdot, Area, P0, T0, R, gamma, supersonic=False):
    md  = mdot.to('kg/s').magnitude
    A   = Area.to('m^2').magnitude
    P0m = P0.to('Pa').magnitude
    T0m = T0.to('K').magnitude
    Rm  = R.to('J/kg/K').magnitude

    def fun(M):
        return compressible_flowrate(A, P0m, T0m, Rm, M, gamma) - md

    if supersonic:
        M = float(brentq(fun, 1.00001, 10.0, xtol=1e-6, maxiter=100))
    else:
        # Check for choke before attempting brentq
        mdot_max = compressible_flowrate(A, P0m, T0m, Rm, 1.0, gamma)
        if md >= mdot_max:
            M = 1.0   # choked — return sonic velocity
        else:
            M = float(brentq(fun, 1e-6, 0.9999999, xtol=1e-6, maxiter=100))

    lmbd = mach_to_lambda(M, gamma)
    return lmbd * a_crit(R, T0, gamma)


def mean_radius(blade, outlet = True):
    """Return the mean radius of a bladed section (m)."""
    if outlet:
        return np.sqrt((blade["R_o_outlet"]**2 + blade["R_i_outlet"]**2) / 2)
    else:
        return np.sqrt((blade["R_o_inlet"]**2 + blade["R_i_inlet"]**2) / 2)


def profile_loss(W1, W2, blade, mu, rho, gamma, gamma_in, M1_mean, M2_mean, Ps1, Ps2, inlet, outlet):
    zeta_s = base_profile_loss(W2, M2_mean, gamma, blade, inlet,
                      Cd=0.002, n_integ=120, DF=0.28, Speak_over_S0=0.42, LEI=0.45)

    W1_mag = float(W1.to('m/s').magnitude)
    W2_mag = float(W2.to('m/s').magnitude)

    # ── Hub inlet Mach — free vortex scaling ──────────────────────────────────
    Rm       = float(mean_radius(blade, outlet=False).to('m').magnitude)
    R_hub    = float(blade["R_i_inlet"].to('m').magnitude)
    R_tip    = float(blade["R_o_inlet"].to('m').magnitude)
    W1_ax    = W1_mag * float(np.cos(blade["Beta_1"].to('rad').magnitude))
    W1_theta = W1_mag * float(np.sin(blade["Beta_1"].to('rad').magnitude))
    W1_theta_hub = W1_theta * Rm / R_hub
    W1_hub   = np.sqrt(W1_ax**2 + W1_theta_hub**2)
    a_in = float(np.sqrt(inlet["gamma"]*inlet["Ts"]*inlet["R"]).to('m/s').magnitude)
    M1_hub   = (W1_hub / a_in)
    zeta_shock_2 = 0
    if M1_hub > 0.4:
        zeta_shock_1 = 0.75*(M1_hub - 0.4)**1.75 * (R_hub / R_tip)  # KO Eq. 5: radial averaging
        # term1 is the inlet-station dynamic-pressure fraction (M1_mean) --
        # uses gamma_in (the row's true local inlet gamma), not the exit
        # gamma used everywhere else in this function for M2_mean terms.
        term1 = 1 - (1 + ((gamma_in - 1)/2)*M1_mean**2)**(gamma_in/(gamma_in - 1))
        term2 = 1 - (1 + ((gamma - 1)/2)*M2_mean**2)**(gamma/(gamma - 1))
        zeta_shock_2 = zeta_shock_1*(Ps1/Ps2)*(term1/term2)
    
    #Convert zeta to Yp
    den_conv = (1 + (gamma-1)/2 * M2_mean**2)**(gamma/(gamma-1)) - 1
    Yp_s = zeta_s * (gamma/2 * M2_mean**2) / den_conv
    
    K1 = 1
    if M2_mean > 0.2:
        K1 = 1 - 1.25*(M2_mean - 0.2)
    K2 = (M1_mean/M2_mean)**2
    Kp = 1 - K2*(1 - K1)
    Yp_s = 0.914*((2/3)*Yp_s*Kp + zeta_shock_2)
    
    true_chord = blade["Chord_ax"] / np.cos(blade["stagger"].to('rad').magnitude)
    Re     = rho * W2 * true_chord / mu
    Re_mag = max(float(Re.to('').magnitude), 100.0)
    
    if Re_mag <= 2*10**5:
        X_re = (Re_mag/(2*10**5))**(-0.4)
    elif 2*10**5 < Re_mag < 10**6:
        X_re = (1.0)
    else:
        X_re = (Re_mag/(10**6))**(-0.2)
    Yp_s = Yp_s * X_re
    
    CFM = 1.0
    if M2_mean > 1:
        CFM = 1 + 49.5*(M2_mean - 1)**3 + 3.3*(M2_mean - 1)**2
     
    
    # At the end, convert corrected Yp back to zeta
    zeta_final = CFM * Yp_s * ((1 + (gamma-1)/2 * M2_mean**2)**(gamma/(gamma-1)) - 1) / (gamma/2 * M2_mean**2)
    return float(zeta_final)


def bladed_throat_area(blade):
    TE  = float(blade["TE_radius"].to('m').magnitude)
    t   = float(blade["throat"].to('m').magnitude)
    Cax = float(blade["Chord_ax"].to('m').magnitude)
    
    bout    = float(blade["Beta_2"].to('rad').magnitude)
    ew      = float(blade["exit_wedge"].to('rad').magnitude)
    zeta    = float(blade["zeta_ung"].to('rad').magnitude)
    
    beta_2  = bout - ew + zeta
    x_2     = Cax - TE + (t + TE) * np.sin(beta_2)
    x_frac  = float(np.clip(x_2 / Cax, 0.0, 1.0))
    
    span_in  = float((blade["R_o_inlet"]  - blade["R_i_inlet"]).to('m').magnitude)
    span_out = float((blade["R_o_outlet"] - blade["R_i_outlet"]).to('m').magnitude)
    span_throat = span_in + x_frac * (span_out - span_in)
    
    # all magnitudes now plain floats in meters
    area = blade["Blade_N"] * t * span_throat
    return Q_(area, 'm^2')

def axial_exit_area_nb(blade):
    annulus_area = np.pi * (blade["R_o_outlet"]**2 - blade["R_i_outlet"]**2)
    exit_blade_span = (blade["R_o_outlet"] - blade["R_i_outlet"])
    TE_blockage = blade["Blade_N"] * 2 * blade["TE_radius"] * exit_blade_span
    return annulus_area# - TE_blockage

def axial_exit_area(blade):
    annulus_area = np.pi * (blade["R_o_outlet"]**2 - blade["R_i_outlet"]**2)
    exit_blade_span = (blade["R_o_outlet"] - blade["R_i_outlet"])
    TE_blockage = blade["Blade_N"] * 2 * blade["TE_radius"] * exit_blade_span
    return annulus_area - TE_blockage

def axial_inlet_area(blade):
    annulus_area = np.pi * (blade["R_o_inlet"]**2 - blade["R_i_inlet"]**2)
    inlet_blade_span = (blade["R_o_inlet"] - blade["R_i_inlet"])
    LE_blockage = blade["Blade_N"] * 2 * blade["LE_radius"] * inlet_blade_span
    return annulus_area# - LE_blockage

def incidence_moustapha(inlet, blade, M1, M2, zeta_s, gamma):
    Cm = inlet["Cm"]
    We = blade["inlet_wedge"].to('deg').magnitude

    if "W_theta" in inlet:
        # Rotor: signed relative flow angle via arctan2
        W_theta = float(inlet["W_theta"].to('m/s').magnitude)
        Cm_val  = float(Cm.to('m/s').magnitude)
        alpha_1 = np.arctan2(W_theta, Cm_val)
    else:
        # Stator: absolute frame, C_theta from vector triangle (signed positive)
        W1      = inlet["W"]
        Cm_val  = float(Cm.to('m/s').magnitude)
        W1_val  = float(W1.to('m/s').magnitude)
        C_theta = np.sqrt(max(W1_val**2 - Cm_val**2, 0.0))
        alpha_1 = np.arctan2(C_theta, Cm_val)

    beta_metal  = blade["Beta_1"].to('rad').magnitude
    i_des_rad   = float(blade.get("i_des", Q_(0.0, 'deg')).to('rad').magnitude)
    # i_eff = α₁ - α₁(des) per Moustapha (1989) Eq.13 x-axis
    # cos β₁/cos β₂ below still uses actual metal angle for geometry
    i = (alpha_1 - beta_metal) - i_des_rad

    if i == 0:
        return zeta_s
    i = np.degrees(i)
    

    cosb2 = np.cos(blade["Beta_2"].to('rad'))
    cosb1 = np.cos(blade["Beta_1"].to('rad'))
    Rm    = mean_radius(blade, outlet=True)
    pitch = (2 * np.pi * Rm) / blade["Blade_N"] 
    d_s   = (2 * blade["LE_radius"] / pitch).to('').magnitude

    X = (d_s**-0.05) * (We**-0.2) * ((cosb1/cosb2)**-1.4) * i
    
    
    if X >= 0:
        a1 = -6.149*10**-5
        a2 =  1.327*10**-3
        a3 = -2.506*10**-4
        a4 = -1.542*10**-4
        a5 =  9.017*10**-5
        a6 =  1.106*10**-5
        a7 = -5.318*10**-6
        a8 =  3.711*10**-7
        dphi2 = a8 * X**8 + a7*X**7 + a6*X**6 + a5*X**5 + a4*X**4 + a3*X**3 + a2*X**2 + a1*X
    elif X < 0:
        a2 =  1.358*10**-4
        a1 = -8.720*10**-4
        dphi2 = a2*X**2 + a1*X

    return float(zeta_s + dphi2)


def secondary_incidence_moustapha(inlet, blade):
    """
    """
    Cm = inlet["Cm"]
    Cm_val = float(Cm.to('m/s').magnitude)

    if "W_theta" in inlet:
        W_theta = float(inlet["W_theta"].to('m/s').magnitude)
        alpha_1 = np.degrees(np.arctan2(W_theta, Cm_val))
    else:
        W1_val  = float(inlet["W"].to('m/s').magnitude)
        C_theta = np.sqrt(max(W1_val**2 - Cm_val**2, 0.0))
        alpha_1 = np.degrees(np.arctan2(C_theta, Cm_val))

    beta1 = float(blade["Beta_1"].to('deg').magnitude)
    beta2 = float(blade["Beta_2"].to('deg').magnitude)
    cosb1 = np.cos(np.radians(beta1))
    cosb2 = np.cos(np.radians(beta2))

    d     = 2.0 * float(blade["LE_radius"].to('m').magnitude)
    c     = float(blade["Chord_ax"].to('m').magnitude)

    chi_dbl = ((alpha_1 - beta2) / (beta1 + beta2)
               * (cosb1 / cosb2) ** (-1.5)
               * (d / c) ** (-0.3))

    if 0.0 < chi_dbl <= 0.3:
        ratio = (np.exp(0.9 * chi_dbl)
                 + 13.0 * chi_dbl**2
                 + 400.0 * chi_dbl**4)
    elif -0.1 <= chi_dbl < 0.0:
        ratio = np.exp(0.9 * chi_dbl)
    else:
        ratio = 1.0   # outside Moustapha validity range — no correction

    return float(ratio)


def arc_length(blade):
    b2 = blade["Beta_2"].to('rad').magnitude
    b1 = blade["Beta_1"].to('rad').magnitude
    tanb2 = -1*np.tan(b2)
    tanb1 = np.tan(b1)
    F2 = tanb2 * np.sqrt(1 + tanb2**2) + np.arcsinh(tanb2)
    F1 = tanb1 * np.sqrt(1 + tanb1**2) + np.arcsinh(tanb1)
    return ((blade["Chord_ax"])/(2*(tanb2 - tanb1)))*(F2 - F1)

def suction_profile(U2, blade, DF=0.28, Speak_over_S0=0.42, LEI=0.45):
    S0 = 1.15 * arc_length(blade).to('m').magnitude
    Rm = mean_radius(blade, outlet=True).to('m')
    t = 2*blade["TE_radius"] 
    pitch = ((2 * np.pi * Rm) / blade["Blade_N"]).to('m').magnitude
    
    #Trailing-edge depression correlation (Eq. 31) ---
    s_over_S0 = pitch / S0
    s_over_S0 = min(s_over_S0, 0.73)
    U_TE        =  U2
    
    #Peak velocity from diffusion factor
    U_peak = (1 + DF) * U_TE
    S_peak = Speak_over_S0 * S0
    n_LE = (1/LEI - 1) / 5

    N1   = 60
    S1   = np.linspace(0, S_peak, N1)
    x_LE = S1 / S_peak
    U1   = U_peak * (x_LE**n_LE)
    
    #Decelerating region (S_peak → S0) using P*(S*) polynomial
    N2     = 60
    S2     = np.linspace(S_peak, S0, N2)
    Sstar  = (S2 - S_peak) / (S0 - S_peak)
    Pstar  = 1.513*Sstar**4 - 4.464*Sstar**3 + 3.956*Sstar**2
    UT_over_Up = U_TE / U_peak
    U2_suction = U_peak * np.sqrt( 1 - Pstar * (1 - UT_over_Up**2) )
    
    S_suction = [S1,         S2[1:]]
    U_suction = [U1, U2_suction[1:]]
    
    return S_suction, U_suction

def base_profile_loss(W2, M2_mean, gamma, blade, inlet,
                      Cd=0.002, n_integ=120, DF=0.28, Speak_over_S0=0.42, LEI=0.45):

    stagger_deg = float(blade["stagger"].to('deg').magnitude)
    Cax         = float(blade["Chord_ax"].to('m').magnitude)
    blade_chord = Cax / np.cos(np.radians(stagger_deg))

    Rm    = float(mean_radius(blade, outlet=True).to('m').magnitude)
    pitch = (2.0 * np.pi * Rm / blade["Blade_N"])
    t = float((2 * blade["TE_radius"]).to('m').magnitude)

    a1 = np.radians(blade["Beta_1"])
    a2 = np.radians(blade["Beta_2"])
    Cm = inlet["Cm"].to('m/s').magnitude

    S0 = 1.15 * arc_length(blade).to('m').magnitude
    W2mag = W2.to('m/s').magnitude
    # ── Suction surface ───────────────────────────────────────────────────
    S_parts, U_parts = suction_profile(W2mag, blade, DF=DF,
                                       Speak_over_S0=Speak_over_S0, LEI=LEI)
    S_suction = np.concatenate([S_parts[0], S_parts[1]])
    U_suction = np.concatenate([U_parts[0], U_parts[1]])
    z_s       = S_suction / S0
    Vs        = U_suction
    
    # ── Pressure surface ──────────────────────────────────────────────────
    Vp = W2mag * z_s **3

    # ── Profile loss ──────────────────────────────────────────────────────
    inter1 = Cd * (Vs / W2mag) ** 3
    inter2 = Cd * (Vp / W2mag) ** 3

    factor       = S0 / (pitch * np.cos(a2))
    zeta_profile = 2.0 * factor * ((np.trapezoid(inter1, z_s)) + (np.trapezoid(inter2, z_s)))
    
    # ── Trailing Edge ──────────────────────────────────────────────────────
    H23 = 0.57
    H12 = 1.4
    Cpb = -0.1
    
    den_conv = (1 + (gamma-1)/2 * M2_mean**2)**(gamma/(gamma-1)) - 1
    Yp_s = zeta_profile * (gamma/2 * M2_mean**2) / den_conv
    
    Yp_tot = ( -(Cpb * t) / (pitch * np.cos(a2)) 
             + 2*H23*Yp_s
             + (H12*H23*Yp_s + (t)/(pitch * np.cos(a2)))**2
    )
    
    zeta_final = Yp_tot * ((1 + (gamma-1)/2 * M2_mean**2)**(gamma/(gamma-1)) - 1) / (gamma/2 * M2_mean**2)
    
    return zeta_final

def coull_secondary(W1, W2, rho_ratio, blade, inlet, Cd=0.002, X_us=0.5, X_ds=0.5, Tmax_Cx=0.10,
                    DF=0.28, Speak_over_S0=0.52, LEI=0.45):
    Cax   = float(blade["Chord_ax"].to('m').magnitude)
    Rm    = float(mean_radius(blade, outlet=True).to('m').magnitude)
    pitch = 2.0 * np.pi * Rm / blade["Blade_N"]
    Cm    = inlet["Cm"].to('m/s')
    if "W_theta" in inlet:
        W_theta = float(inlet["W_theta"].to('m/s').magnitude)
        Cm_val  = float(Cm.to('m/s').magnitude)
        alpha_1 = np.arctan2(W_theta, Cm_val)
    else:
        W1_loc  = inlet["W"]
        Cm_val  = float(Cm.to('m/s').magnitude)
        W1_val  = float(W1_loc.to('m/s').magnitude)
        C_theta = np.sqrt(max(W1_val**2 - Cm_val**2, 0.0))
        alpha_1 = np.arctan2(C_theta, Cm_val)
    alpha_2     = blade["Beta_2"].to('rad').magnitude
    V1_V2       = (W1 / W2).to('').magnitude
    XUS_Cx      = X_us
    XDS_Cx      = X_ds
    span        = (blade["R_o_outlet"] - blade["R_i_outlet"]).to('m').magnitude
    Cx_hcosa2 = Cax / (span * np.cos(alpha_2))   # Cx/(h·cos α₂), paper Eqs. (7) and (8)
    Cx_pcosa2 = Cax / (pitch * np.cos(alpha_2))
    Cx_h      = Cax / span

    # Wetted Area Loss — paper Eqs. (7), (8), (9)
    zeta_US   = 4 * Cd * Cx_hcosa2 * V1_V2**3 * XUS_Cx
    zeta_DS   = 4 * Cd * Cx_hcosa2 * XDS_Cx
    zeta_pass = 4.363 * Cd * Cx_h
    zeta_wet  = zeta_US + zeta_DS + zeta_pass

    # Coull secondary model requires V1 < V2 (accelerating passage)
    if V1_V2 >= 1.0:
        return zeta_wet

    # Secondary Flow: Mixing-Induced Loss
    V_star    = 1 - np.sqrt(1 - V1_V2**2)
    dT_star   = transit_time_integral(W1, W2, blade, DF=DF, Speak_over_S0=Speak_over_S0, LEI=LEI)
    Gamma_sec = (V_star * dT_star * Cx_pcosa2
                 + np.abs(V1_V2 * np.sin(alpha_1) / np.cos(alpha_2)
                          - V_star * np.tan(alpha_2)))
    zeta_secmix = 2 * (pitch * np.cos(alpha_2) / span) * 0.01442 * Gamma_sec

    # Secondary Flow: Secondary Kinetic Energy (PI_SKE, Coull GT2025 coefficients)
    H12     = 1.4
    H12_lim = 1.23
    H_eff   = max(H12, H12_lim)
    disp_th_Cx  = 0.05
    CF = (1 + min(2.62, H12)) * (1 / V1_V2 - 1) * (np.cos(alpha_2) / np.cos(alpha_1))
    CF = min(max(CF, 0), 5)
    disp_th_Eff = disp_th_Cx * np.exp(-0.345 * CF)
    D      = disp_th_Eff / (pitch * np.cos(alpha_2))
    term   = D**((0.170 / (H_eff - 1)) + 0.593)
    PI_SKE = (0.0102 * (H_eff - 1) * term) / (D**1.636 + 0.0986 * (H_eff - 1) - 0.119)
    zeta_SKE = 2 * (pitch * np.cos(alpha_2) / span) * Gamma_sec**2 * PI_SKE

    return zeta_wet + zeta_secmix + zeta_SKE


def transit_time_integral(W1, W2, blade,
                          DF=0.28, Speak_over_S0=0.52, LEI=0.45, n_integ=120):
    """Direct Eq. (15) integral: dT* = T*_PS - T*_SS = ∮ (V2/V) d(S/Cx)."""
    Cax   = float(blade["Chord_ax"].to('m').magnitude)
    W2mag = float(W2.to('m/s').magnitude)
    W1mag = float(W1.to('m/s').magnitude)

    # Suction surface
    S0_SS   = float(arc_length(blade).to('m').magnitude)
    S_parts, U_parts = suction_profile(W2mag, blade, DF=DF,
                                       Speak_over_S0=Speak_over_S0, LEI=LEI)
    S_ss = np.concatenate([S_parts[0], S_parts[1]])[1:]
    V_ss = np.concatenate([U_parts[0], U_parts[1]])[1:]
    T_star_SS = np.trapezoid(W2mag / V_ss, S_ss / Cax)

    # Pressure surface: constant at W1
    S0_PS = S0_SS * 0.86
    z_ps  = np.linspace(0.0, S0_PS / Cax, n_integ)
    T_star_PS = np.trapezoid(np.full(n_integ, W2mag / W1mag), z_ps)

    dT_star = T_star_PS - T_star_SS
    return min(dT_star, 12.0)                # §A.3 limiter


def zeta_leak_denton(blade, W1, W2, n_integ=40, Cd=0.8):
    Rm         = mean_radius(blade, outlet=True)
    pitch      = float((2*np.pi*Rm / blade["Blade_N"]).to('m').magnitude)
    span       = float(((blade["R_o_inlet"] - blade["R_i_inlet"])).to('m').magnitude)
    true_chord = float((blade["Chord_ax"] /
                  np.cos(blade["stagger"].to('rad').magnitude)).to('m').magnitude)
    g          = float(blade["Tip_Gap"].to('m').magnitude)

    a1 = float(blade["Beta_1"].to('rad').magnitude)
    a2 = float(blade["Beta_2"].to('rad').magnitude)

    W2_m = float(W2.to('m/s').magnitude)

    # ── Suction surface from Coull & Hodson ───────────────────────────────
    S_parts, U_parts = suction_profile(W2_m, blade)
    S_suction = np.concatenate([S_parts[0], S_parts[1]])
    U_suction = np.concatenate([U_parts[0], U_parts[1]])
    S0        = float((1.15 * arc_length(blade)).to('m').magnitude)
    z_s       = S_suction / S0
    Vs        = U_suction

    # ── Pressure surface — quadratic from stagnation to W2 ───────────────
    z  = np.linspace(0.0, 1.0, n_integ)
    Vp = W2_m * z**3

    # ── Interpolate Vs onto same z grid for ratio ─────────────────────────
    Vs_interp = np.interp(z, z_s, Vs)
    Vs_interp = np.maximum(Vs_interp, 1e-6)
    Vp        = np.maximum(Vp, 0.0)
    ratio     = np.clip(Vp / Vs_interp, 0.0, 1.0)

    # ── Denton Eq. A6.7 integrand ─────────────────────────────────────────
    integrand = (Vs_interp / W2_m)**3 * (1.0 - ratio) * np.sqrt(1.0 - ratio**2)

    I = np.trapezoid(integrand, z)

    return float(2.0 * Cd * g * true_chord / (span * pitch * np.cos(a2)) * I)

def zeta_leak_Lyu(blade, M2_mean, gamma):
    """Variable geometry turbine leakage loss model modification
        You, Lyu; Dong, Xuezhi; Liu, Xiyang; Tan, Chunqing
        ISSN: 0360-5442 , 1873-6785; DOI: 10.1016/j.energy.2025.139829
        Energy. , 2026, Vol.344, p.139829"""
    
    tau_LE = blade["Tip_Gap"]
    chord = blade["Chord_ax"]
    tau_TE = tau_LE
    k_theta = (abs(tau_LE - tau_TE)/chord).to('').magnitude
    Cd = 0.8
    d_c = 0.25 # Quarter OF BLADE
    d = chord * d_c
    z_d = 0.25 # Quarter POINT OF BLADE
    phi_d = 1 - (2*d*(tau_LE - k_theta*z_d * chord)) / (chord * (tau_LE + tau_TE))
    
    alpha_1 = blade["Beta_1"]
    alpha_2 = blade["Beta_2"]
    term = 0.5 * (np.tan(alpha_1) + np.tan(alpha_2))
    alpha_m = np.atan(term)
    Cl_sc = 2 * (np.tan(alpha_1) + np.tan(alpha_2))*np.cos(alpha_m)
    
    Rm    = mean_radius(blade, outlet=True).to('m')
    pitch = 2.0 * np.pi * Rm / blade["Blade_N"]
    s_c = pitch / chord
    Cl = Cl_sc * s_c
    
    blade_height = blade["R_o_outlet"] - blade["R_i_outlet"]
    C = 0.007
    K_g = 0.943
    
    term1 = 2 * (0.566*tau_LE - 0.404*k_theta*chord)
    term2 = phi_d * Cd * Cl**1.5
    term3 = ((np.cos(alpha_2)**2)/(np.cos(alpha_m)**3))*(chord/(blade_height*pitch))
    term4 = phi_d * C * K_g * Cd * np.sqrt(Cl) * (chord**2/(blade_height*pitch)) * (1/np.cos(alpha_m))
    
    Y_leak = term1 * term2 * term3 + term4
    
    zeta_final = Y_leak * ((1 + (gamma-1)/2 * M2_mean**2)**(gamma/(gamma-1)) - 1) / (gamma/2 * M2_mean**2)
    
    return float(zeta_final.to('').magnitude)
    
def q_func(lam, gamma):
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0
    gp1_2 = gp1 / 2.0
    odin_gm1 = 1.0 / gm1
    return lam*(gp1_2*(1 - (gm1/gp1)*lam**2))**odin_gm1

def lam_crit_polytropic(n, gamma):
    gm1_gp1 = (gamma - 1.0) / (gamma + 1.0)
    gp1_2   = (gamma + 1.0) / 2.0
    odin_gm1 = 1/(gamma - 1)
    dva_nm1 = 2/(n - 1)
    odin_nm1 = 1/(n - 1)
    def func(lam):
        term1 = gp1_2 ** odin_gm1
        term2 = (1 - gm1_gp1*lam**2)**(odin_nm1 - 1)
        term3 = (1 - (lam**2)*(gm1_gp1)*(1 + dva_nm1))
        return term1 * term2 * term3
    lam_crit = fsolve(func, x0=1.0)[0]
    return lam_crit

def zeta_leak_impulse(blade, inlet, outlet, Pt0_stage, phi_nozzle=0.975,
                      lambda_seal=1.0, tau_y=1.0):

    Delta = float(blade['Tip_Gap'].to('m').magnitude)
    h1    = float((blade['R_o_outlet'] - blade['R_i_outlet']
                   - blade['Tip_Gap']).to('m').magnitude)
    D_cp  = 2.0 * float(mean_radius(blade, outlet=True).to('m').magnitude)

    if Delta <= 0 or h1 <= 0:
        return 0.0, 1.0, 0.0

    # ── Ovsyannikov Eqs. 1-4: leakage mass fraction ──────────────────────
    # alpha_1T: stator exit angle from tangential
    # Beta_2 is stored in axial convention → alpha_1T = 90° − |Beta_2|
    alpha_1T_rad = np.radians(
        90.0 - abs(float(blade['Beta_2'].to('deg').magnitude))
    )

    # Eq. 4: flow coefficient for open unshrouded gap
    mu_zaz = np.sqrt(4.0 * Delta / (lambda_seal * tau_y))

    # Eq. 3: geometric factor
    r_g = (1.0 + h1 / D_cp) * (2.0 * Delta / h1)

    # Eq. 2: leakage flow factor
    # rho_T = stage total-to-static PR = Pt0_stage / Ps_rotor_exit
    Pt0_Pa = float(Pt0_stage.to('Pa').magnitude)
    Ps2_Pa = float(outlet['Ps'].to('Pa').magnitude)
    rho_T  = Pt0_Pa / Ps2_Pa

    sin2_a  = np.sin(alpha_1T_rad)**2
    bracket = max(rho_T * ((1.0 / phi_nozzle**2) * sin2_a - 1.0), 0.0)
    f_y     = np.sqrt(1.0 + bracket) * r_g

    # Eq. 1: leakage fraction
    mdot_frac = float(np.clip(mu_zaz * f_y, 0.0, 0.5))
    eta_vol   = 1.0 - mdot_frac

    # ── Denton mixing loss ────────────────────────────────────────────────
    W1_m      = float(inlet['W'].to('m/s').magnitude)
    W2_m      = float(outlet['W'].to('m/s').magnitude)
    beta1_rad = float(blade['Beta_1'].to('rad').magnitude)
    beta2_rad = float(blade['Beta_2'].to('rad').magnitude)

    # Leakage exits at inlet swirl (unturned, positive direction)
    # Main flow exits at outlet swirl (turned, opposite direction)
    V_thetaL  =  W1_m * np.sin(abs(beta1_rad))   # positive: inlet direction
    V_theta2  = -W2_m * np.sin(abs(beta2_rad))   # negative: exit direction (turned)
    delta_Vt  = V_thetaL - V_theta2              # = W_theta_in + W_theta_out

    if W2_m < 1.0:
        return 0.0, eta_vol, mdot_frac

    alpha2_rad = abs(beta2_rad)
    TdS        = mdot_frac * W2_m**2 * (1.0 - V_thetaL/V_theta2)**2 \
                * np.sin(alpha2_rad)**2
    zeta_mix   = float(max(TdS / (0.5 * W2_m**2), 0.0))

    return zeta_mix, eta_vol, mdot_frac

def q_n_func(lam, n, gamma):
    gm1_gp1 = (gamma - 1.0) / (gamma + 1.0)
    gp1_2   = (gamma + 1.0) / 2.0
    odin_gm1 = 1/(gamma - 1)
    odin_nm1 = 1/(n - 1)
    
    term1  = gp1_2**odin_gm1
    term2  = lam * (1 - gm1_gp1*lam**2)**odin_nm1
    
    return term1 * term2
    
def deviation(blade, lam, lam_crit, gamma):
    """
    Deviation angle model, matching Zou et al. 2026 Appendix B (Eq. B1-B4)
    exactly. The smoothstep blend is defined there purely in terms of Mach
    number (Ma from 0.5 to 1.0) -- NOT lam/lam_crit. That distinction
    matters: lam_crit is the polytropic-adjusted choking threshold, which is
    generically < 1 for a lossy row (this is itself validated by the paper
    -- the qn(lambda) mass-flow-function peak shifts below lambda=1 as
    losses increase), so blending on lam/lam_crit would relax the deviation
    to zero at a lower ACTUAL Mach than the reference model intends.
    Choking (mass-flow pinning, lam >= lam_crit) is a separate decision from
    which deviation formula to use -- see the lam >= 1.0 routing at the
    caller (sup_deviation only applies once the flow is actually
    supersonic; lam_crit <= lam < 1 still uses this Mach-blended formula).
    """
    # Pitch here must pair with the annulus-area formula (pi*(Ro^2-Ri^2) =
    # 2*pi*b*Rm_arith, exact algebra), so it needs the ARITHMETIC mean
    # radius -- not mean_radius()'s RMS definition (used elsewhere for
    # blade speed / loss correlations, a separate convention). Using RMS
    # here breaks Dixon Eq. 3.60's A_exit*cos(alpha2)|_{Ma=1} = A_throat
    # identity, since the two terms would then reference different radii.
    Rm    = (blade["R_o_outlet"] + blade["R_i_outlet"]) / 2
    t     = float(((2 * np.pi * Rm) / blade["Blade_N"]).to('m').magnitude)
    o     = float(blade["throat"].to('m').magnitude)

    sinb   = float(np.clip(o / t, 0.0, 1.0))
    beta_g_circ = np.degrees(np.arcsin(sinb))

    arg    = sinb * (1.0 + (1.0 - sinb) * (beta_g_circ / 90.0)**2)
    delta_0 = np.degrees(np.arcsin(arg)) - beta_g_circ

    # Smoothstep on Ma from 0.5 to 1.0 (Eq. B2), not lam/lam_crit.
    Ma = lambda_to_mach(lam, gamma)
    
    #Mc = lambda_to_mach(lam_crit, gamma)
    if Ma <= 0.5:
        delta = delta_0
    elif Ma <= 1.0:
        X     = 2.0 * Ma - 1.0   # 0 to 1 as Ma goes 0.5 to 1.0
        delta = delta_0 * (1.0 - 10*X**3 + 15*X**4 - 6*X**5)
    else:
        delta = 0.0   # Ma slightly above 1.0 due to numerics — use choke value

    # Hard limit: the flow can never turn MORE than the fully-guided
    # (gauging-angle) limit -- alpha_out <= arccos(o/t), i.e. delta >= 0.
    # Dixon & Hall Ch.3 (Eq. 3.60): alpha2 = arccos(o/s) exactly at Ma=1,
    # this is the most-turned condition the geometry can produce.
    delta = max(delta, 0.0)

    beta_g_axial = np.degrees(np.arcsin(float(o / t)))

    alpha_out = beta_g_axial + delta
    alpha_out = 90 - alpha_out

    return Q_(alpha_out, 'deg')

def mdot_out(A, Pt, Tt, lam_c, n_poly, gamma, R):
    dva_gp1 = 2/(gamma + 1)
    gp1_gm1 = (gamma + 1.0) / (gamma - 1.0)
    qn = q_n_func(lam_c, n_poly, gamma)
    term1 = (gamma / R) * dva_gp1**gp1_gm1
    term2 = (Pt/np.sqrt(Tt))*A*qn
    return np.sqrt(term1)*term2
    
def sup_deviation(blade, lam_out, Pt_mag, Tt_mag, gamma, R_mag, n_poly, mdot_choke=None):

    gp1      = gamma + 1.0
    gm1      = gamma - 1.0
    mf_term  = np.sqrt((gamma / R_mag) * (2.0 / gp1) ** (gp1 / gm1))

    # isentropic q_n at lam_out
    q_n_out  = q_n_func(lam_out, n_poly, gamma)

    # isentropic lam_crit = 1.0, q_n_crit
    lam_crit = lam_crit_polytropic(n_poly, gamma)
    alpha = deviation(blade, lam_crit, lam_crit, gamma)
    cos_b = np.cos(alpha)
    A_out_nb    = float(axial_exit_area_nb(blade).to('m**2').magnitude)

    if mdot_choke is None:
        mdot_choke = mdot_out(A_out_nb * cos_b, Pt_mag, Tt_mag, lam_crit, n_poly, gamma, R_mag)

    arg      = (mdot_choke * np.sqrt(Tt_mag)) / \
                   (mf_term * Pt_mag * A_out_nb * q_n_out)

    arg = float(np.clip(arg, 0.0, 1.0))
    return Q_(np.degrees(np.arccos(arg)), 'deg')

def blade_passage_new(blade, input, gas, P_out, r_loss, rpm=Q_(0,'rpm'), mdot_choke=None):
    # ── Unpack inlet ──────────────────────────────────────────────────────
    Pt_in     = input["Pt_rel"]
    Tt_in     = input["Tt_rel"]
    gamma     = input["gamma"]
    R         = input["R"]
    W1        = input["W"]

    Pt_mag    = float(Pt_in.to('Pa').magnitude)
    Tt_mag    = float(Tt_in.to('K').magnitude)
    P_out_mag = float(P_out.to('Pa').magnitude)
    Rmag      = float(R.to('J/kg/K').magnitude)

    # ── Inlet static state ────────────────────────────────────────────────
    lam_w1 = float((W1 / a_crit(R, Tt_in, gamma)).to('').magnitude)
    Ts1    = Tt_in * isentropic_t_T0(lam_w1, gamma)
    Ps1    = Pt_in * isentropic_p_P0(lam_w1, gamma)
    gas.TP = float(Ts1.to('K').magnitude), float(Ps1.to('Pa').magnitude)
    Cp_in  = float(gas.cp_mass)

    # ── Blade speed ───────────────────────────────────────────────────────
    if blade["rotating"]:
        omega = rpm.to('rad/s')
        Rm    = mean_radius(blade, outlet=True)
        U_mag = float((omega * Rm).to('m/s').magnitude)
    else:
        U_mag = 0.0

    # ── Outlet total pressure ─────────────────────────────────────────────
    Pt_out_mag = Pt_mag * r_loss
    if Pt_out_mag <= P_out_mag:
        raise ValueError(f"Pt_out ({Pt_out_mag:.0f} Pa) <= P_out ({P_out_mag:.0f} Pa) — unphysical")

    # ── Iterate gamma_out and n_poly to convergence ───────────────────────
    conv      = False
    gamma_out = gamma
    R_out     = R
    R_out_mag = Rmag

    for _ in range(30):
        # n_poly and gp1_gm1 BOTH reference the INLET gamma (fixed, not
        # gamma_out): n_poly characterizes entropy generation relative to
        # the inlet reference state, and using the same inlet gamma in
        # gp1_gm1 keeps lam_out a closed-form function of fixed inlet gas
        # properties + the given pressures -- no dependence on the
        # not-yet-converged gamma_out at all, so no circularity (mixing in
        # gamma_out here previously broke mass-flow convergence once both
        # rows choke). gamma_out is still iterated below, but only feeds
        # LOCAL/exit-state conversions (T_out_mag, Cp_out, rho_out, Mach),
        # matching forward_design.py's _build_blade_geometry, which uses the
        # same inlet/exit split for n_poly vs local-state gamma.
        gp1_gm1 = (gamma + 1.0) / (gamma - 1.0)

        Pt_in_over_Ps  = Pt_mag     / P_out_mag
        Pt_out_over_Ps = Pt_out_mag / P_out_mag
        gm1_g = (gamma - 1.0)/gamma

        n_poly = 1/(1 - (gm1_g)*(np.log(Pt_out_over_Ps)/np.log(Pt_in_over_Ps)))

        # lam_out from polytropic relation
        omn_n   = (1.0 - n_poly) / n_poly
        lam_out = float(np.sqrt(gp1_gm1 * (1.0 - (Pt_mag / P_out_mag) ** omn_n)))

        # Static temperature from lam_out
        T_out_mag = float(Tt_mag * isentropic_t_T0(lam_out, gamma_out))

        # Update gamma and R at exit static conditions
        gamma_new, R_new = thermo_update_quantities(
            Q_(T_out_mag, 'K'), Q_(P_out_mag, 'Pa'), gas)
        R_new_mag = float(R_new.to('J/kg/K').magnitude)

        err = max(abs(float(gamma_new) - gamma_out),
                abs(R_new_mag - R_out_mag) / max(R_out_mag, 1e-6))

        gamma_out = float(gamma_new)
        R_out     = R_new
        R_out_mag = R_new_mag

        if err < 1e-6:
            break

    gas.TP  = T_out_mag, P_out_mag
    Cp_out  = float(gas.cp_mass)
    rho_out = float(gas.density)

    # ── Choke detection ─────────────────────────────────────────────────────
    # lam_out is already a polytropic (loss-referenced) quantity, not an
    # isentropic one -- everything about the actual exit state (Ts, Ps, W2)
    # is built from n_poly. Comparing it against an isentropic-only threshold
    # (n=gamma) compares two different reference frames and produced a real
    # ~5% mass-flow discontinuity at the choke transition on the TP1680
    # sweep (confirmed). lam_crit_polytropic(n_poly, gamma) is the correct,
    # continuous threshold: it's the peak of the SAME mass-flow function
    # (parameterized by the SAME n_poly) that lam_out/deviation() already use
    # on the unchoked side, so choking and the unchoked branch stay in the
    # same reference frame by construction. See sup_deviation() for how mass
    # continuity is enforced once choked.
    lam_crit = lam_crit_polytropic(n_poly, gamma)
    choked   = lam_out >= lam_crit

    # ── Exit velocity ─────────────────────────────────────────────────────
    a_crit_mag = float(a_crit(R_out, Q_(Tt_mag, 'K'), gamma_out).to('m/s').magnitude)
    W2_mag     = lam_out * a_crit_mag
    M_w2       = lambda_to_mach(lam_out, gamma_out)

    # ── Outlet flow angle ─────────────────────────────────────────────────
    if not choked:
        alpha_out = deviation(blade, lam_out, lam_crit, gamma)
    else:
        mdot_choke_val = (float(mdot_choke.to('kg/s').magnitude)
                         if mdot_choke is not None else None)
        alpha_out = sup_deviation(blade, lam_out, Pt_mag, Tt_mag,
                                  gamma, Rmag, n_poly,
                                  mdot_choke=mdot_choke_val)

    cos_a  = float(np.cos(float(alpha_out.to('rad').magnitude)))
    tan_a  = float(np.tan(float(alpha_out.to('rad').magnitude)))
    Cm_out = W2_mag * cos_a

    # ── Absolute frame ────────────────────────────────────────────────────
    # Beta_2 is stored as a positive magnitude; for a turbine rotor the relative
    # exit tangential velocity is in the OPPOSITE direction to blade motion, so
    # W_theta_rel is negative.  C_theta = W_theta + U vectorially.
    if blade["rotating"]:
        W_theta_rel = -Cm_out * tan_a
        C_theta_abs = W_theta_rel + U_mag
        C2          = np.sqrt(Cm_out**2 + C_theta_abs**2)
        Tt_abs_mag  = T_out_mag + 0.5 * C2**2 / Cp_out
        a_crit_abs  = float(a_crit(R_out, Q_(Tt_abs_mag, 'K'),
                                   gamma_out).to('m/s').magnitude)
        lam_abs     = C2 / a_crit_abs
        M_abs_val   = lambda_to_mach(lam_abs, gamma_out)
        Pt_abs_mag  = P_out_mag / isentropic_p_P0(lam_abs, gamma_out)
        gas.TP      = T_out_mag, P_out_mag
        rho_out     = float(gas.density)
        hs_out      = Q_(gas.enthalpy_mass, 'J/kg')
        ht_abs      = hs_out + Q_(0.5 * C2**2, 'J/kg')
        C_out       = Q_(C2,          'm/s')
        W_theta     = Q_(W_theta_rel, 'm/s')
        C_theta     = Q_(C_theta_abs, 'm/s')
    else:
        Tt_abs_mag = Tt_mag
        Pt_abs_mag = Pt_out_mag
        C2         = W2_mag
        M_abs_val  = M_w2   # absolute = relative for stationary blade
        gas.TP     = T_out_mag, P_out_mag
        rho_out    = float(gas.density)
        hs_out     = Q_(gas.enthalpy_mass, 'J/kg')
        ht_abs     = hs_out + Q_(0.5 * C2**2, 'J/kg')
        C_out      = Q_(C2,           'm/s')
        C_theta    = Q_(Cm_out * tan_a,'m/s')
        W_theta    = Q_(0.0,          'm/s')

    # ── Mass flow ─────────────────────────────────────────────────────────
    A_exit  = float(axial_exit_area_nb(blade).to('m**2').magnitude)
    mdot = mdot_out(A_exit * cos_a, Pt_mag, Tt_mag, lam_out, n_poly, gamma, Rmag)

    return {
        "Pt_rel":   Q_(Pt_out_mag, 'Pa'),
        "Tt_rel":   Q_(Tt_mag,     'K'),
        "Ps":       Q_(P_out_mag,  'Pa'),
        "Ts":       Q_(T_out_mag,  'K'),
        "W":        Q_(W2_mag,     'm/s'),
        "Cm":       Q_(Cm_out,     'm/s'),
        "W_theta":  W_theta,
        "Pt_abs":   Q_(Pt_abs_mag, 'Pa'),
        "Tt_abs":   Q_(Tt_abs_mag, 'K'),
        "C":        C_out,
        "C_theta":  C_theta,
        "gamma":    gamma_out,
        "R":        R_out,
        "ht":       ht_abs,
        "hs":       hs_out,
        "lam_exit": lam_out,
        "M_exit":   M_w2,
        "M_abs":    M_abs_val,
        "lam_crit": lam_crit,
        "choked":   choked,
        "alpha_out": alpha_out,
        "mdot":     Q_(mdot, 'kg/s'),
        "n_poly":   n_poly,
    }

def stator_to_rotor_inlet(s_out, rotor, gas, rpm):
    """
    Velocity triangle only — no thermodynamic calculation.
    Stator absolute frame → rotor relative frame.
    """
    gamma  = s_out["gamma"]
    R      = s_out["R"]
    Cm_mag = float(s_out["Cm"].to('m/s').magnitude)
    C_mag  = float(s_out["C"].to('m/s').magnitude)
    Ts_mag = float(s_out["Ts"].to('K').magnitude)
    Ps_mag = float(s_out["Ps"].to('Pa').magnitude)

    omega   = rpm.to('rad/s')
    U       = float((omega * mean_radius(rotor, outlet=False)).to('m/s').magnitude)

    # Replace the sqrt (which loses sign) with direct use of C_theta from s_out:
    C_theta = float(s_out["C_theta"].to('m/s').magnitude)   # signed, from blade_passage_new
    W_theta = C_theta - U
    W_mag   = np.sqrt(Cm_mag**2 + W_theta**2)

    gas.TP     = Ts_mag, Ps_mag
    Cp         = float(gas.cp_mass)
    Tt_rel_mag = Ts_mag + 0.5 * W_mag**2 / Cp
    a_crit_rel = float(a_crit(R, Q_(Tt_rel_mag,'K'), gamma).to('m/s').magnitude)
    lam_rel    = W_mag / a_crit_rel
    Pt_rel_mag = Ps_mag / isentropic_p_P0(lam_rel, gamma)

    hs     = Q_(gas.enthalpy_mass, 'J/kg')
    ht_rel = hs + Q_(0.5 * W_mag**2, 'J/kg')

    return {
        "Pt_rel":  Q_(Pt_rel_mag, 'Pa'),
        "Pt_abs":  s_out["Pt_abs"],
        "Tt_rel":  Q_(Tt_rel_mag, 'K'),
        "Tt_abs":  s_out["Tt_abs"],
        "Ps":      s_out["Ps"],
        "Ts":      s_out["Ts"],
        "W":       Q_(W_mag,      'm/s'),
        "Cm":      s_out["Cm"],
        "C":       s_out["C"],
        "W_theta": Q_(W_theta,    'm/s'),
        "gamma":   gamma,
        "R":       R,
        "ht":      ht_rel,
        "hs":      hs,
    }

def compute_blade_losses(blade, inlet, outlet, gas, rpm=Q_(150000,'rpm')):
    """
    Pure loss evaluation at converged flow state.
    inlet, outlet are state dicts from blade_passage_new.
    Returns r_loss_new = Pt_out/Pt_in
    """
    
    # ── Unpack inlet ──────────────────────────────────────────────────────
    Pt_in  = inlet["Pt_rel"]
    Tt_in  = inlet["Tt_rel"]
    
    gamma  = inlet["gamma"]
    R      = inlet["R"]
    W1     = inlet["W"]
    
    # ── Unpack outlet ─────────────────────────────────────────────────────
    W2     = outlet["W"]
    Ps2    = outlet["Ps"]
    Ts2    = outlet["Ts"]
    lam_w2 = outlet["lam_exit"]
    
    Pt_mag = float(Pt_in.to('Pa').magnitude)
    
    lam_w1 = float((W1 / a_crit(R, Tt_in, gamma)).to('').magnitude)
    M_w1   = lambda_to_mach(lam_w1, gamma)
    Ts1    = Tt_in * isentropic_t_T0(lam_w1, gamma)
    Ps1    = Pt_in * isentropic_p_P0(lam_w1, gamma)
    
    # ── Inlet static state for loss models ───────────────────────────────
    gas.TP    = float(Ts1.to('K').magnitude), float(Ps1.to('Pa').magnitude)
    rho_in   = Q_(gas.density,   'kg/m^3')

    # ── Outlet static state for loss models ───────────────────────────────
    gas.TP    = float(Ts2.to('K').magnitude), float(Ps2.to('Pa').magnitude)
    mu_out    = Q_(gas.viscosity, 'Pa*s')
    rho_out   = Q_(gas.density,   'kg/m^3')
    gamma_out, R_out = thermo_update_quantities(Ts2, Ps2, gas)
    Cp_out    = float(gas.cp_mass)
    R_mag_out     = float(R_out.to('J/kg/K').magnitude)
    
    try:
        M_w2   = outlet["M_exit"]
    except:
        M_w2   = (W2 / np.sqrt(Ts2 * R_out * gamma_out)).to('').magnitude

    # ── Profile loss ──────────────────────────────────────────────────────
    zeta_s = profile_loss(W1, W2, blade, mu_out, rho_out, gamma_out, gamma, M_w1, M_w2, Ps1, Ps2, inlet, outlet)

    # ── Incidence correction ──────────────────────────────────────────────
    zeta_s = float(incidence_moustapha(inlet, blade, M_w1, M_w2, zeta_s, gamma_out))

    # ── Wake loss ─────────────────────────────────────────────────────────
    zeta_w = 0.0


    # ── Endwall loss (Coull 2025) + Moustapha secondary incidence correction ──
    rho_ratio = (rho_in / rho_out).to('').magnitude
    zeta_f = coull_secondary(W1, W2, rho_ratio, blade, inlet)
    zeta_f *= secondary_incidence_moustapha(inlet, blade)

    # ── Leakage loss ──────────────────────────────────────────────────────
    if blade["Tip_Gap"] > Q_(0, 'mm'):
        if blade["rotating"]:
            zeta_l = zeta_leak_denton(blade, W1, W2, Cd = 0.800)
        else:
            zeta_l = zeta_leak_Lyu(blade, M_w2, gamma_out)
    else:
        zeta_l = 0.0

    # ── Total loss ────────────────────────────────────────────────────────
    zeta_tot = zeta_s + zeta_w + zeta_f + zeta_l

    # ── r_loss via entropy definition ─────────────────────────────────────
    exponent   = -(zeta_tot * Cp_out * (gamma_out - 1.0) * M_w2**2) / (2.0 * R_mag_out)
    Pt_out_new = Pt_mag * np.exp(exponent)
    r_loss_new = float(Pt_out_new / Pt_mag)   # ← explicit float
    return {
        "r_loss_new": r_loss_new,        # plain float
        "zeta_s":     float(zeta_s),
        "zeta_w":     float(zeta_w),
        "zeta_f":     float(zeta_f),
        "zeta_l":     float(zeta_l),
        "zeta_tot":   float(zeta_tot),
    }

def compute_blade_losses_impulse(blade, inlet, outlet, gas, rpm=Q_(150000, 'rpm')):

    # ── Unpack inlet / outlet ─────────────────────────────────────────────
    Pt_in  = inlet["Pt_rel"]
    Tt_in  = inlet["Tt_rel"]
    gamma  = inlet["gamma"]
    R      = inlet["R"]
    W1     = inlet["W"]

    W2     = outlet["W"]
    Ts2    = outlet["Ts"]
    Ps2    = outlet["Ps"]

    Pt_mag = float(Pt_in.to('Pa').magnitude)

    # ── Thermo at outlet static — need Cp, gamma, R_out for the entropy
    #    → Pt-ratio conversion used by compute_blade_losses ───────────────
    gas.TP = float(Ts2.to('K').magnitude), float(Ps2.to('Pa').magnitude)
    gamma_out, R_out = thermo_update_quantities(Ts2, Ps2, gas)
    Cp_out  = float(gas.cp_mass)
    R_mag   = float(R_out.to('J/kg/K').magnitude)

    # ── Relative Mach at rotor INLET (paper Fig. 14 uses M_w2 = M_w_inlet) ─
    # Paper station convention: 2 = nozzle exit = rotor inlet.
    # In this code base: inlet = rotor inlet, so M_w1_code = M_w2_paper.
    lam_w1 = float((W1 / a_crit(R, Tt_in, gamma)).to('').magnitude)
    M_w1   = lambda_to_mach(lam_w1, gamma)

    # Also compute exit M for the r_loss / entropy conversion (uses exit state)
    lam_w2_exit = outlet["lam_exit"]
    M_w2_exit   = lambda_to_mach(lam_w2_exit, gamma_out)

    # ── Blade geometry → β (tangential, deg) and chord/span ──────────────
    # Paper Eq. [13] uses blade angle β measured from TANGENTIAL
    # (Fig. 1 of Linhardt & Silvern).  This code stores β from axial,
    # so convert:  β_tang = 90° − β_axial
    # For a symmetric blade β_1 = β_2; the average handles non-symmetric
    # cases sanely.
    beta_1_ax = abs(float(blade["Beta_1"].to('deg').magnitude))    # from axial
    beta_2_ax = abs(float(blade["Beta_2"].to('deg').magnitude))    # from axial
    beta_1_tg = 90.0 - beta_1_ax                                    # from tangential
    beta_2_tg = 90.0 - beta_2_ax                                    # from tangential
    beta_tg   = 0.5 * (beta_1_tg + beta_2_tg)

    chord_m = float(blade["Chord_ax"].to('m').magnitude)
    span_m  = float((blade["R_o_outlet"] - blade["R_i_outlet"]).to('m').magnitude)
    chord_over_span = chord_m / span_m if span_m > 0 else 0.0

    # ── Rotor velocity coefficient — Linhardt & Silvern 1961, Eq. [13] ────
    # ψ_R = [1 − 0.228·(1 − (β°/90°)³)] · [1 − 0.05·(M_w − 1)²] · [1 − 0.06·c/h]
    # β is from tangential.  M_w is the relative Mach at rotor inlet
    # (paper station 2 = rotor inlet).  Naturally bounded:
    # f_turning ∈ [0.772, 1.0] for β_tang ∈ [0°, 90°]; β_tang → 0° is
    # maximum turning (max loss), β_tang → 90° is zero turning.
    f_turning = 1.0 - 0.228 * ((1.0 - beta_tg / 90.0)**3)
    f_mach    = 1.0 - 0.05  * (M_w1 - 1.0)**2
    f_ar      = 1.0 - 0.06  * chord_over_span
    phi       = f_turning * f_mach * f_ar

    # ── Velocity coefficient → enthalpy-loss coefficient ──────────────────
    zeta_cascade = 1.0 / phi**2 - 1.0
    zeta_cascade = max(zeta_cascade, 0.0)

    # ── Tip leakage loss (separate — not captured by cascade correlation) ─
    if blade["Tip_Gap"] > Q_(0, 'mm'):
        zeta_l, b, c = zeta_leak_impulse(blade, inlet, outlet, inlet['Pt0_stage'])
    else:
        zeta_l = 0.0

    zeta_tot = zeta_cascade + zeta_l

    # ── r_loss via same entropy route as compute_blade_losses ─────────────
    # The Pt ratio is evaluated at the exit station, so use M at exit.
    exponent   = -(zeta_tot * Cp_out * (gamma_out - 1.0) * M_w2_exit**2) / (2.0 * R_mag)
    Pt_out_new = Pt_mag * np.exp(exponent)
    r_loss_new = float(Pt_out_new / Pt_mag)

    return {
        "r_loss_new": r_loss_new,
        "zeta_s":   float(zeta_cascade),
        "zeta_w":   0.0,
        "zeta_f":   0.0,
        "zeta_l":   float(zeta_l),
        "zeta_tot": float(zeta_tot),
        # Diagnostics specific to this loss model
        "phi":             float(phi),
        "M_w1":            float(M_w1),        # rotor INLET Mach (correlation arg)
        "M_w2":            float(M_w2_exit),   # rotor EXIT Mach (r_loss arg)
        "beta_tang_deg":   float(beta_tg),
        "f_turning":       float(f_turning),
        "f_mach":          float(f_mach),
        "f_ar":            float(f_ar),
        "chord_over_span": float(chord_over_span),
    }

def smooth_duct(Rm_in, Rm_out, A2, input, gas, mdot):
    gamma = input["gamma"]
    R_gas = input["R"]
    Pt = input["Pt_abs"]
    Tt = input["Tt_abs"]
    Cm_out = velocity_from_flowrate(mdot, A2, Pt, Tt, R_gas, gamma)
    C_in = input["C"]
    Cm_in = input["Cm"]
    C_theta1 = np.sqrt(C_in**2 - Cm_in**2)
    C_theta2 = C_theta1*Rm_in/Rm_out
    C_out = np.sqrt(Cm_out**2 + C_theta2**2)
    lam_c2 = (C_out / a_crit(R_gas, Tt, gamma)).to('')
    if not np.isreal(lam_c2):
        print()
    Ts_out = Tt * isentropic_t_T0(lam_c2, gamma)
    
    Ps_out = Pt * isentropic_p_P0(lam_c2, gamma)
    gas.TP = Tt.to('K').magnitude, Pt.to('Pa').magnitude
    ht = Q_(gas.enthalpy_mass,"J/kg")
    hs = ht - 0.5*C_out**2
    output = {
        "Pt_rel": Pt,
        "Ps_rel": Ps_out,
        "Tt_rel": Tt,
        "Ts_rel": Ts_out,
        "Pt_abs": Pt,
        "Tt_abs": Tt,
        "C": C_out,
        "Cm": Cm_out,
        "W": C_out,
        "gamma": gamma,
        "R": R_gas ,
        "ht": ht,
        "hs": hs
    }
    
    return output
    
def initialize_stator_inlet(inlet, gas):
    """
    Build stator_inlet dict from basic inlet conditions.
    Velocity seeded as zero — updated after first mdot convergence.
    """
    Pt   = inlet["Pt"]
    Tt   = inlet["Tt"]
    
    gamma, R = thermo_update_quantities(Tt, Pt, gas)
    
    # At total conditions — zero velocity seed
    gas.TP = float(Tt.to('K').magnitude), float(Pt.to('Pa').magnitude)
    ht     = Q_(gas.enthalpy_mass, 'J/kg')
    
    return {
        "Pt_rel": Pt,
        "Pt_abs": Pt,
        "Tt_rel": Tt,
        "Tt_abs": Tt,
        "Ps":     Pt,       # seed — updated below
        "Ts":     Tt,       # seed — updated below
        "W":      Q_(0.0, 'm/s'),   # seed
        "Cm":     Q_(0.0, 'm/s'),   # seed
        "C":      Q_(0.0, 'm/s'),   # seed
        "W_theta": Q_(0.0, 'm/s'),
        "gamma":  gamma,
        "R":      R,
        "ht":     ht,
        "hs":     ht,       # seed — static = total at zero velocity
    }
    
def update_stator_inlet(stator_inlet, mdot, stator, gas):
    """
    Update velocity-dependent quantities in stator_inlet
    given converged mdot from current iteration.
    """
    Pt    = stator_inlet["Pt_rel"]
    Tt    = stator_inlet["Tt_rel"]
    gamma = stator_inlet["gamma"]
    R     = stator_inlet["R"]
    
    conv_cm = False
    Cm = velocity_from_flowrate(mdot, axial_inlet_area(stator), Pt, Tt, R, gamma)
    while not conv_cm:
        lam  = float((Cm / a_crit(R, Tt, gamma)).to('').magnitude)
        Ts   = Tt * isentropic_t_T0(lam, gamma)
        Ps   = Pt * isentropic_p_P0(lam, gamma)
        gas.TP = float(Ts.to('K').magnitude), float(Ps.to('Pa').magnitude)
        gamma_new, R_new = thermo_update_quantities(Ts, Ps, gas)
        Cm_new = velocity_from_flowrate(mdot, axial_inlet_area(stator), 
                                        Pt, Tt, R_new, gamma_new)
        err = abs(float((Cm_new - Cm).to('m/s').magnitude))
        Cm    = Cm_new
        gamma = gamma_new
        R     = R_new
        
        gas.TP = float(Ts.to('K').magnitude), float(Ps.to('Pa').magnitude)
        hs     = Q_(gas.enthalpy_mass, 'J/kg')
        ht     = hs + 0.5 * Cm**2
        
        if err < 1e-6:
            conv_cm = True
    
    stator_inlet.update({
        "Ps":    Ps,
        "Ts":    Ts,
        "W":     Cm,
        "Cm":    Cm_new,
        "C":     Cm_new,
        "gamma": gamma_new,
        "R":     R_new,
        "ht":    ht,
        "hs":    hs,
    })
    
    return stator_inlet

def thermo_update_quantities(T_q, P_q, gas_solution):
    """
    Accepts T_q (Quantity), P_q (Quantity) or magnitudes.
    Updates Cantera gas (using magnitudes in K and Pa) and returns (gamma, R) as magnitudes.
    """
    # Ensure magnitudes for Cantera (Pa, K)
    T = (T_q.to('K').magnitude)
    P = (P_q.to('Pa').magnitude)

    # Update Cantera state
    gas_solution.TP = T, P

    cp = gas_solution.cp_mass   # J/kg/K (magnitude)
    cv = gas_solution.cv_mass
    gamma = float(cp / cv)
    R = float(ct.gas_constant / gas_solution.mean_molecular_weight)  # J/kg/K
    return gamma, Q_(R,'J/kg/K')

def build_performance_curve_moffitt(stator, rotor, inlet, gas,
                                    composition, rpm, p2_sweep, N_points):

    Pt_mag = float(inlet["Pt"].to('Pa').magnitude)
    Tt_mag = float(inlet["Tt"].to('K').magnitude)

    gas.TPX = Tt_mag, Pt_mag, composition

    def compute_eta(rotor_out):
        gas.TPX = Tt_mag, Pt_mag, composition
        ht_in = Q_(gas.enthalpy_mass, 'J/kg')
        s_in  = gas.entropy_mass
        gas.SP = s_in, float(rotor_out["Pt_abs"].to('Pa').magnitude)
        ht_is  = Q_(gas.enthalpy_mass, 'J/kg')
        dh_act = ht_in - rotor_out["ht"]
        dh_is  = ht_in - ht_is
        return float((dh_act / dh_is).to('').magnitude)

    stator_inlet = initialize_stator_inlet(inlet, gas)
    gamma        = stator_inlet["gamma"]
    curve        = []

    # ── Geometry diagnostics ──────────────────────────────────────────────────────
    A_exit_s = float(axial_exit_area(stator).to('m**2').magnitude)
    A_exit_r = float(axial_exit_area(rotor).to('m**2').magnitude)
    A_thr_s  = float(bladed_throat_area(stator).to('m**2').magnitude)
    A_thr_r  = float(bladed_throat_area(rotor).to('m**2').magnitude)
    print(f"  Stator: A_exit={A_exit_s:.6f} m²  A_throat={A_thr_s:.6f} m²")
    print(f"  Rotor:  A_exit={A_exit_r:.6f} m²  A_throat={A_thr_r:.6f} m²")

    # ── Which orifice choked first — set permanently once detected ────────────
    # "stator" → stator choked first; p1 is free, p2 drives expansion
    # "rotor"  → rotor choked first;  mdot set by rotor, p1 is irrelevant to mdot
    # None     → neither choked yet
    first_choke  = None   # "stator" | "rotor" | None
    mdot_choke_s = None   # kg/s, stator choked mass flow for sup_deviation
    mdot_choke_r = None   # kg/s, rotor choked mass flow for sup_deviation

    # Warm-start losses and p1 from previous converged point
    r_loss_s_prev = 1.0
    r_loss_r_prev = 1.0
    p1_prev       = 0.5 * (float(p2_sweep[0]) + Pt_mag)   # midpoint of first p2 and Pt_in
    n_choked_s = False
    mdot_frozen = None
    for p2_idx, p2_mag in enumerate(p2_sweep):
        p2        = Q_(p2_mag, 'Pa')
        r_loss_s  = r_loss_s_prev
        r_loss_r  = r_loss_r_prev
        s_out     = None
        r_out     = None
        r_inlet   = None
        choke_mode = "none"
        conv_loss  = False
        err        = 1e6
        alpha      = 0.3
        _invalid     = [False]
        _invalid_msg = [""]
        loss_iter = 0
        

        while loss_iter <= 50:
            # Only break if invalid was set by final state eval (not residual search)
            if _invalid[0]:
                break

            # ── Residual — always ms - mr unless frozen ──────────────────────
            def mass_flow_residual(p1_mag):
                p1_mag = float(np.asarray(p1_mag).flat[0])
                try:
                    p1_q     = Q_(float(p1_mag), 'Pa')
                    s_out_   = blade_passage_new(stator, stator_inlet, gas,
                                                 p1_q, r_loss_s, rpm=Q_(0,'rpm'),
                                                 mdot_choke=Q_(mdot_frozen,'kg/s') if mdot_frozen is not None else None)
                    r_inlet_ = stator_to_rotor_inlet(s_out_, rotor, gas, rpm)
                    r_out_   = blade_passage_new(rotor, r_inlet_, gas,
                                                 p2, r_loss_r, rpm=rpm,
                                                 mdot_choke=Q_(mdot_frozen,'kg/s') if mdot_frozen is not None else None)
                    ms = float(s_out_["mdot"].to('kg/s').magnitude)
                    mr = float(r_out_["mdot"].to('kg/s').magnitude)
                    return ms - mr
                except Exception as e:
                    _invalid_msg[0] = str(e)[:80]
                    return 1e6

            # ── Solve — fsolve warm-started from previous p1 ─────────────────
            if (choke_mode == "rotor" or choke_mode == "both") and mdot_frozen is not None:
                # Rotor choked: stator must also pass exactly mdot_frozen.
                # Solve for p1 that gives stator ms = mdot_frozen.
                def stator_choked_residual(p1_mag):
                    p1_mag = float(np.asarray(p1_mag).flat[0])
                    try:
                        p1_q = Q_(float(p1_mag), 'Pa')
                        s_out_ = blade_passage_new(stator, stator_inlet, gas,
                                                   p1_q, r_loss_s, rpm=Q_(0,'rpm'),
                                                   mdot_choke=Q_(mdot_frozen,'kg/s'))
                        ms = float(s_out_["mdot"].to('kg/s').magnitude)
                        return ms - mdot_frozen
                    except Exception:
                        return 1e6
                p1_sol = float(fsolve(stator_choked_residual, p1_prev,
                                      full_output=False)[0])
            elif choke_mode == "rotor" or choke_mode == "both":
                p1_sol = p1_prev
            else:
                p1_sol = float(fsolve(mass_flow_residual, p1_prev,
                                    full_output=False)[0])

            # Reset invalid flag — residual may have hit bad p1 during search
            _invalid[0] = False
            p1_sol_clipped = float(np.clip(float(p1_sol), 
                                        Pt_mag * 0.01,   # min: 1% of inlet
                                        Pt_mag * 0.999)) # max: just below inlet
            try:
                p1      = Q_(p1_sol_clipped, 'Pa')
                s_out   = blade_passage_new(stator, stator_inlet, gas,
                                            p1, r_loss_s, rpm=Q_(0,'rpm'),
                                            mdot_choke=Q_(mdot_frozen,'kg/s') if mdot_frozen is not None else None)
                r_inlet = stator_to_rotor_inlet(s_out, rotor, gas, rpm)
                r_out   = blade_passage_new(rotor, r_inlet, gas,
                                            p2, r_loss_r, rpm=rpm,
                                            mdot_choke=Q_(mdot_frozen,'kg/s') if mdot_frozen is not None else None)
            except Exception as e:
                _invalid[0] = True
                _invalid_msg[0] = str(e)[:60]
                break

            # ── W2 < W1 check — skip loss update but don't break ─────────────
            W1_r = float(r_inlet["W"].to('m/s').magnitude)
            W2_r = float(r_out["W"].to('m/s').magnitude)
            w2_lt_w1 = W2_r < W1_r

            # ── Choke detection — only update first_choke once ────────────────
            s_choked = s_out["choked"]
            r_choked = r_out["choked"]

            if s_choked and r_choked:
                choke_mode = "both"
            elif s_choked:
                choke_mode = "stator"
            elif r_choked:
                choke_mode = "rotor"
            else:
                choke_mode = "none"

            if os.environ.get('DEBUG_PEAK') and 10 <= p2_idx <= 20:
                print(f"    [DBG idx={p2_idx} iter={loss_iter}] p1_sol={p1_sol_clipped:.3f} "
                      f"r_loss_s={r_loss_s:.6f} r_loss_r={r_loss_r:.6f} "
                      f"ms={float(s_out['mdot'].to('kg/s').magnitude):.6f} "
                      f"mr={float(r_out['mdot'].to('kg/s').magnitude):.6f} "
                      f"s_choked={s_choked} r_choked={r_choked} choke_mode={choke_mode} err={err:.3e}")

            err_prev     = err
            # ── Update stator inlet ───────────────────────────────────────────
            if choke_mode == "rotor" or choke_mode == "both":
                loss_r = compute_blade_losses(rotor, r_inlet, r_out,
                                          gas, rpm=rpm)
                r_loss_r_new = float(loss_r["r_loss_new"])
                err = abs(r_loss_r_new - r_loss_r)
                r_loss_r = alpha * r_loss_r_new + (1.0 - alpha) * r_loss_r
            
            else:
                
                stator_inlet = update_stator_inlet(stator_inlet, s_out["mdot"],
                                                stator, gas)
                loss_s = compute_blade_losses(stator, stator_inlet, s_out,
                                                gas, rpm=Q_(0,'rpm'))
                loss_r = compute_blade_losses(rotor, r_inlet, r_out,
                                            gas, rpm=rpm)

                
                r_loss_s_new = float(loss_s["r_loss_new"])
                r_loss_r_new = float(loss_r["r_loss_new"])
                
                err = max(abs(r_loss_s_new - r_loss_s),
                            abs(r_loss_r_new - r_loss_r))

                r_loss_s = alpha * r_loss_s_new + (1.0 - alpha) * r_loss_s
                r_loss_r = alpha * r_loss_r_new + (1.0 - alpha) * r_loss_r


            # ── Convergence — loss only, mdot guaranteed by p1 solver ─────────
            if err < 1e-4:
                if (s_out['choked'] or r_out['choked']) and not n_choked_s:
                    if s_out['choked']:
                        mdot_frozen = float(s_out["mdot"].to('kg/s').magnitude)
                    else:
                        mdot_frozen = float(r_out["mdot"].to('kg/s').magnitude)
                    n_choked_s  = True
                    loss_iter = 1
                else:
                    conv_loss = True
                    break
            loss_iter += 1

        # ── Save warm start ───────────────────────────────────────────────────
        if conv_loss:
            r_loss_s_prev = r_loss_s
            r_loss_r_prev = r_loss_r
            p1_prev       = p1_sol_clipped


        # ── Store result ──────────────────────────────────────────────────────
        if s_out is not None and r_out is not None and conv_loss:
            
            # after each converged point:
            
            mdot_final = float(s_out["mdot"].to('kg/s').magnitude)
            
            eta          = compute_eta(r_out)
            PR           = Pt_mag / float(r_out["Pt_abs"].to('Pa').magnitude)
            Cm_exit      = float(r_out["Cm"].to('m/s').magnitude)
            C_theta_exit = float(r_out["C_theta"].to('m/s').magnitude)
            alpha_exit   = float(np.degrees(np.arctan2(C_theta_exit, Cm_exit)))
            M_EXITR    = float(r_out["M_exit"])
            M_EXITS    = float(s_out["M_exit"])

            beta_in_deg = float(np.degrees(np.arctan2(
                float(r_inlet['W_theta'].to('m/s').magnitude),
                float(r_inlet['Cm'].to('m/s').magnitude))))

            theta_cr   = Tt_mag / 288.15
            delta_test = Pt_mag / 101325
            mdot_corr  = mdot_final * np.sqrt(theta_cr) / delta_test
            i_rotor    = beta_in_deg - float(rotor['Beta_1'].to('deg').magnitude)

            mdot_s_raw = float(s_out["mdot"].to('kg/s').magnitude)
            mdot_r_raw = float(r_out["mdot"].to('kg/s').magnitude)

            #clear_terminal()
            print(f"  [{p2_idx+1}/{N_points}]"
                  f"  PR={PR:.3f}"
                  f"  mdot_corr={mdot_corr:.3f}"
                  f"  eta={eta*100:.2f}%"
                  f"  choke={choke_mode}"
                  f"  1st={first_choke}"
                  f"  alpha_s={s_out["alpha_out"]:4f}°"
                  f"  Mach_stator={M_EXITS:.3f}"
                  f"  p1={float(p1.to('Pa').magnitude)/1000:.1f}kPa"
                  f"  ms - mr={mdot_s_raw - mdot_r_raw:.3f}")

            Ps_rin  = float(r_inlet["Ps"].to('Pa').magnitude)
            Ps_rout = float(r_out["Ps"].to('Pa').magnitude)
            Pt_rin  = float(r_inlet["Pt_rel"].to('Pa').magnitude)

            curve.append({
                "PR":              PR,
                "mdot":            mdot_final,
                "eta":             eta,
                "alpha_exit":      alpha_exit,
                "choke":           choke_mode,
                "first_choke":     first_choke,
                "loss_s":          loss_s,
                "loss_r":          loss_r,
                "stator_inlet":    stator_inlet,
                "stator_out":      s_out,
                "rotor_in":        r_inlet,
                "rotor_out":       r_out,
                "M_stator_exit":   M_EXITS,
                "M_rotor_exit":    M_EXITR,
                "i_rotor":         i_rotor,
                "Ps_rotor_in":     Ps_rin,
                "Ps_rotor_out":    Ps_rout,
                "Pt_rotor_in":     Pt_rin,
            })
        else:
            PR_skip = Pt_mag / p2_mag
            skip_reason = f"invalid:{_invalid_msg[0]}" if _invalid[0] else f"iter={loss_iter+1}"
            print(f"  [{p2_idx+1}/{N_points}]"
                  f"  PR={PR_skip:.3f}  SKIPPED"
                  f"  conv={'Y' if conv_loss else 'N'}"
                  f"  choke={choke_mode}"
                  f"  1st={first_choke}"
                  f"  why={skip_reason}")

    return curve

def point_to_segment_distance(P, A, B):
    AB = B - A; AP = P - A
    t  = np.clip(np.dot(AP, AB) / np.dot(AB, AB), 0.0, 1.0)
    return np.linalg.norm(P - (A + t * AB)), A + t * AB
 
def geometric_throat(x_ss, y_ss, x_ps_adj, y_ps_adj):
    """Minimum distance from SS blade 1 to PS of adjacent blade."""
    min_dist = float('inf')
    throat_pt_ss = throat_pt_ps = None
    ps_points = np.column_stack([x_ps_adj, y_ps_adj])
    for i in range(len(x_ss)):
        P = np.array([x_ss[i], y_ss[i]])
        for j in range(len(x_ps_adj) - 1):
            d, closest = point_to_segment_distance(P, ps_points[j], ps_points[j+1])
            if d < min_dist:
                min_dist = d; throat_pt_ss = P; throat_pt_ps = closest
    return min_dist, throat_pt_ss, throat_pt_ps
 
def extract_throat(
    raw,            # Nx3 array: columns [X, Y_L, Y_U] in inches
    phi_deg,        # orientation angle [deg]
    Rm_in,          # mean radius [inches]
    N_blades,       # blade count
    plot=True,
    title="Blade throat",
    savepath=None,
):
    """
    Extract geometric throat from NASA blade coordinate table.
 
    Convention (TN D-4389 / TP-1680):
        X    : axial in blade-local frame (chord direction)
        Y_L  : lower (pressure) surface offset from chord line
        Y_U  : upper (suction) surface offset from chord line
        phi  : orientation angle — blade X-axis relative to machine axial
 
    The throat is computed entirely in blade-local frame.
    A circumferential pitch (machine frame) maps to blade-local shifts:
        dX_local = -pitch_circ * sin(phi)
        dY_local = +pitch_circ * cos(phi)
    Rotation to machine frame is applied only for plotting.
 
    Returns
    -------
    throat_mm : float  — geometric throat in mm
    throat_in : float  — geometric throat in inches
    """
    X_raw  = raw[:, 0]
    YL_raw = raw[:, 1]   # pressure surface
    YU_raw = raw[:, 2]   # suction surface
    X_in = np.linspace(X_raw[0], X_raw[-1], 100)
    FU = interpolate.interp1d(X_raw, YU_raw)
    FL = interpolate.interp1d(X_raw, YL_raw)
    YL_in = FL(X_in)
    YU_in = FU(X_in)
 
    phi = np.radians(phi_deg)
    pitch_circ_in = 2.0 * np.pi * Rm_in / N_blades
 
    # Pitch in blade-local frame
    pitch_local_X = -pitch_circ_in * np.sin(phi)
    pitch_local_Y = pitch_circ_in * np.cos(phi)
 
    # Surfaces in blade-local frame
    x_ss = X_in;  y_ss = YU_in          # SS blade 1
    x_ps = X_in;  y_ps = YL_in          # PS blade 1
    x_ps_adj = x_ps + pitch_local_X     # PS adjacent blade
    y_ps_adj = y_ps + pitch_local_Y
    x_ss_adj = x_ss + pitch_local_X     # SS adjacent blade (plot only)
    y_ss_adj = y_ss + pitch_local_Y
 
    # Throat in blade-local frame
    throat_in, pt_ss, pt_ps = geometric_throat(x_ss, y_ss, x_ps_adj, y_ps_adj)
    throat_mm = throat_in * 25.4
    
    print(f"Blade 1 LE: ({x_ss[0]:.4f}, {y_ss[0]:.4f})")
    print(f"Blade 1 TE: ({x_ss[-1]:.4f}, {y_ss[-1]:.4f})")
    print(f"Blade 2 LE: ({x_ps_adj[0]:.4f}, {y_ps_adj[0]:.4f})")
    print(f"Blade 2 TE: ({x_ps_adj[-1]:.4f}, {y_ps_adj[-1]:.4f})")
    print(f"Pitch shift: ({pitch_local_X:.4f}, {pitch_local_Y:.4f})")
    print(f"Expected tangential pitch: {pitch_circ_in:.4f} in")
    print(f"Throat point SS: ({pt_ss[0]:.4f}, {pt_ss[1]:.4f})")
    print(f"Throat point PS: ({pt_ps[0]:.4f}, {pt_ps[1]:.4f})")
    print(f"Throat distance: {throat_in:.4f} in = {throat_mm:.4f} mm")
 
    print(f"{'-'*52}")
    print(f"  {title}")
    print(f"  Circumferential pitch : {pitch_circ_in*25.4:.3f} mm")
    print(f"  Geometric throat      : {throat_mm:.4f} mm  ({throat_in:.5f} in)")
    if pt_ss is not None:
        print(f"  Throat SS (local)     : X={pt_ss[0]:.4f} in, Y={pt_ss[1]:.4f} in")
        print(f"  Throat PS (local)     : X={pt_ps[0]:.4f} in, Y={pt_ps[1]:.4f} in")
    print(f"{'-'*52}")
 
    if plot:
        # Rotate to machine frame for plotting only
        def rotate(x, y):
            return (x*np.cos(phi) + y*np.sin(phi),
                -x*np.sin(phi) + y*np.cos(phi))
 
        x_ss_m,     y_ss_m     = rotate(x_ss,     y_ss)
        x_ps_m,     y_ps_m     = rotate(x_ps,     y_ps)
        x_ps_adj_m, y_ps_adj_m = rotate(x_ps_adj, y_ps_adj)
        x_ss_adj_m, y_ss_adj_m = rotate(x_ss_adj, y_ss_adj)
        pt_ss_m = np.array(rotate(pt_ss[0], pt_ss[1]))
        pt_ps_m = np.array(rotate(pt_ps[0], pt_ps[1]))
 
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(x_ss_m,     y_ss_m,     'b-',  lw=2.5, label='SS blade 1')
        ax.plot(x_ps_m,     y_ps_m,     'b--', lw=2.5, label='PS blade 1')
        ax.plot(x_ss_adj_m, y_ss_adj_m, 'r-',  lw=2.5, label='SS blade 2')
        ax.plot(x_ps_adj_m, y_ps_adj_m, 'r--', lw=2.5, label='PS blade 2')
        ax.plot([pt_ss_m[0], pt_ps_m[0]], [pt_ss_m[1], pt_ps_m[1]],
                'g-', lw=3, label=f'Throat = {throat_mm:.3f} mm')
        ax.plot(*pt_ss_m, 'go', ms=10)
        ax.plot(*pt_ps_m, 'gs', ms=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_xlabel('Axial [in]'); ax.set_ylabel('Tangential [in]')
        ax.set_title(f'{title}  —  Throat = {throat_mm:.3f} mm  (phi={phi_deg}°)')
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
            print(f"  Plot saved: {savepath}")
        plt.show()
 
    return throat_mm, throat_in

def rotate_stator(raw, theta_deg, phi_deg_design, Rm_in, N_blades,
                    Beta_2_design=73.05, plot=False):
        X_in  = raw[:, 0]
        YL_in = raw[:, 1]
        YU_in = raw[:, 2]

        X_pivot = X_in[-1]
        Y_pivot = 0.5 * (YL_in[-1] + YU_in[-1])

        theta = np.radians(theta_deg)

        X_rel  = X_in  - X_pivot
        YL_rel = YL_in - Y_pivot
        YU_rel = YU_in - Y_pivot

        X_rot  = X_rel * np.cos(theta) - YL_rel * np.sin(theta) + X_pivot
        YL_rot = X_rel * np.sin(theta) + YL_rel * np.cos(theta) + Y_pivot
        X_rot2 = X_rel * np.cos(theta) - YU_rel * np.sin(theta) + X_pivot
        YU_rot = X_rel * np.sin(theta) + YU_rel * np.cos(theta) + Y_pivot

        raw_rot = np.column_stack([X_rot, YL_rot, YU_rot])
        # Negative theta_deg closes the blade (toward tangential); Beta_2 and
        # phi both use the same convention: subtract theta_deg so that closing
        # (theta < 0) increases Beta_2 and phi (more tangential) — matching
        # the paper's description of 7.79° toward tangential for 70% area.
        phi_new = phi_deg_design - theta_deg

        phi_rad = np.radians(phi_deg_design)   # design phi for blade-local pitch offset
        pitch   = 2.0 * np.pi * Rm_in / N_blades
        plX     = -pitch * np.sin(phi_rad)
        plY     =  pitch * np.cos(phi_rad)

        x_ps_adj = X_rot  + plX
        y_ps_adj = YL_rot + plY

        t_in, _, _ = geometric_throat(X_rot2, YU_rot, x_ps_adj, y_ps_adj)
        throat_mm  = t_in * 25.4

        Beta_1_new = 0.0           - theta_deg
        Beta_2_new = Beta_2_design - theta_deg

        return throat_mm, Beta_1_new, Beta_2_new, raw_rot, phi_new

"""
NASA TP-1680 (Moffitt 1980) — pyCycle Map Generation
Multi-speed sweep for turbine performance map
"""

if __name__ == "__main__":

    gas = ct.Solution('air.yaml')
    gas.TP = 378.0, 241325.0
    composition = gas.X

    inlet = {
        "Tt": Q_(378.0,    'K'),
        "Pt": Q_(241325.0, 'Pa'),
    }

    Dm    = Q_(46.99, 'cm').to('m').magnitude
    b     = Q_(3.81,  'cm').to('m').magnitude
    Rm    = Dm / 2                                      # RMS mean radius
    R_i   = Rm - b/2#(-b + np.sqrt(4*Rm**2 - b**2)) / 2          # solve (Ro²+Ri²)/2=Rm², Ro-Ri=b
    R_o   = R_i + b
    Rm_in = (Dm * 100 / 2) / 2.54

    raw_stator = np.array([
        [0,     0.100,  0.100], [0.05,  0.013,  0.364], [0.10,  0.000,  0.461],
        [0.15,  0.013,  0.526], [0.20,  0.050,  0.573], [0.25,  0.082,  0.607],
        [0.30,  0.109,  0.633], [0.40,  0.151,  0.663], [0.50,  0.182,  0.676],
        [0.60,  0.203,  0.676], [0.70,  0.217,  0.666], [0.80,  0.226,  0.649],
        [0.90,  0.229,  0.627], [1.00,  0.228,  0.601], [1.10,  0.223,  0.572],
        [1.20,  0.216,  0.539], [1.30,  0.205,  0.503], [1.40,  0.192,  0.465],
        [1.50,  0.176,  0.425], [1.60,  0.159,  0.382], [1.70,  0.139,  0.337],
        [1.80,  0.118,  0.290], [1.90,  0.095,  0.242], [2.00,  0.070,  0.192],
        [2.10,  0.045,  0.141], [2.20,  0.017,  0.089], [2.25,  0.003,  0.062],
        [2.291, 0.025,  0.025],
    ])

    raw_rotor = np.array([
        [0,      0.074,  0.074], [0.01,   0.033,  0.138], [0.02,   0.019,  0.176],
        [0.03,   0.010,  0.208], [0.04,   0.005,  0.234], [0.05,   0.002,  0.257],
        [0.075,  0.002,  0.309], [0.100,  0.012,  0.349], [0.125,  0.030,  0.385],
        [0.150,  0.055,  0.415], [0.200,  0.107,  0.470], [0.250,  0.152,  0.512],
        [0.300,  0.190,  0.546], [0.350,  0.221,  0.574], [0.400,  0.247,  0.596],
        [0.450,  0.269,  0.611], [0.500,  0.285,  0.620], [0.550,  0.296,  0.623],
        [0.600,  0.303,  0.621], [0.650,  0.307,  0.615], [0.700,  0.306,  0.602],
        [0.750,  0.302,  0.585], [0.800,  0.293,  0.564], [0.850,  0.282,  0.538],
        [0.900,  0.267,  0.508], [0.950,  0.250,  0.474], [1.000,  0.229,  0.436],
        [1.050,  0.206,  0.395], [1.100,  0.180,  0.350], [1.150,  0.152,  0.305],
        [1.200,  0.121,  0.256], [1.250,  0.088,  0.205], [1.300,  0.052,  0.152],
        [1.350,  0.013,  0.096], [1.400,  0.025,  0.025],
    ])

    throat_s_mm, _ = extract_throat(raw_stator, phi_deg=48.633, Rm_in=Rm_in,
                                    N_blades=36, plot=False, title="Moffitt Stator")
    throat_r_mm, _ = extract_throat(raw_rotor,  phi_deg=14.25,  Rm_in=Rm_in,
                                    N_blades=64, plot=False, title="Moffitt Rotor")

    Cd = 1.00
    stator = {
        "Beta_1":     Q_(0.0,             'deg'),
        "Beta_2":     Q_(73.05,           'deg'),
        "stagger":    Q_(48.633,          'deg'),
        "throat":     Q_(throat_s_mm*Cd,  'mm'),
        "TE_radius":  Q_(0.0635,          'cm'),
        "LE_radius":  Q_(0.254,           'cm'),
        "Chord_ax":   Q_(3.81,            'cm'),
        "Blade_N":    36,
        "Tip_Gap":    Q_(0.0,             'mm'),
        "R_o_inlet":  Q_(R_o,             'm'),
        "R_i_inlet":  Q_(R_i,             'm'),
        "R_o_outlet": Q_(R_o,             'm'),
        "R_i_outlet": Q_(R_i,             'm'),
        "exit_wedge": Q_(3.0,             'deg'),
        "inlet_wedge": Q_(15,             'deg'),
        "zeta_ung":   Q_(6.5,             'deg'),
        "rotating":   False,
    }

    rotor = {
        "Beta_1":     Q_(50.83,           'deg'),
        "Beta_2":     Q_(59.63,           'deg'),
        "stagger":    Q_(14.25,           'deg'),
        "throat":     Q_(throat_r_mm*Cd,  'mm'),
        "TE_radius":  Q_(0.0635,          'cm'),
        "LE_radius":  Q_(0.1905,          'cm'),
        "Chord_ax":   Q_(3.429,           'cm'),
        "Blade_N":    64,
        "Tip_Gap":    Q_(0.030,           'cm'),
        "R_o_inlet":  Q_(R_o,             'm'),
        "R_i_inlet":  Q_(R_i,             'm'),
        "R_o_outlet": Q_(R_o,             'm'),
        "R_i_outlet": Q_(R_i,             'm'),
        "exit_wedge": Q_(3.0,             'deg'),
        "inlet_wedge": Q_(15,             'deg'),
        "zeta_ung":   Q_(6.5,             'deg'),
        "rotating":   True,
    }

    # ── Map sweep parameters ──────────────────────────────────────────────────────
    N_equiv_design = 8081.0          # rpm at standard conditions
    Tt_test        = 378.0           # K
    Pt_test        = 241325.0        # Pa
    Pt_mag         = Pt_test

    speed_fracs = [1.00]#[0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10]
    area_fracs  = [1.00]   # stator throat area multipliers (1.0 = design)
    PR_lo       = 1.5
    PR_hi       = 7.0
    N_pts       = 100

    theta_cr   = Tt_test / 288.15
    delta_test = Pt_mag / 101325.0

    def corrected_mdot(mdot_kg):
        return mdot_kg * np.sqrt(theta_cr) / delta_test

    p2_vals = Pt_mag / np.linspace(PR_lo, PR_hi, N_pts)

    # ── Solve theta_deg for each area fraction ───────────────────────────────────
    throat_design = throat_s_mm / Cd   # geometric throat before Cd correction

    def throat_residual(theta_deg, target_throat_mm):
        t_mm, _, _, _, _ = rotate_stator(raw_stator, theta_deg, 48.633,
                                          Rm_in, N_blades=36, plot=False)
        return t_mm - target_throat_mm

    area_theta = {}   # area_frac -> theta_deg
    for area_frac in area_fracs:
        target = throat_design * area_frac
        if abs(area_frac - 1.0) < 1e-6:
            area_theta[area_frac] = 0.0
        else:
            from scipy.optimize import brentq
            theta = brentq(throat_residual, -20.0, 20.0,
                           args=(target,), xtol=1e-4)
            area_theta[area_frac] = theta
        print(f"  area_frac={area_frac:.3f}  theta={area_theta[area_frac]:+.3f}°"
              f"  target_throat={target:.4f} mm")

    # ── Run all speed lines ───────────────────────────────────────────────────────
    all_curves = {}   # keyed by (speed_frac, area_frac)
    for area_frac in area_fracs:
        theta_deg = area_theta[area_frac]
        throat_s_mm_var, Beta_1_s, Beta_2_s, _, phi_new = rotate_stator(
            raw_stator, theta_deg, 48.633, Rm_in, N_blades=36, plot=False)

        stator_var = {
            "Beta_1":     Q_(Beta_1_s,            'deg'),
            "Beta_2":     Q_(Beta_2_s,            'deg'),
            "stagger":    Q_(phi_new,             'deg'),
            "throat":     Q_(throat_s_mm_var*Cd,  'mm'),
            "TE_radius":  Q_(0.0635,              'cm'),
            "LE_radius":  Q_(0.254,               'cm'),
            "Chord_ax":   Q_(3.81,                'cm'),
            "Blade_N":    36,
            "Tip_Gap":    Q_(0.0,                 'mm'),
            "R_o_inlet":  Q_(R_o,                 'm'),
            "R_i_inlet":  Q_(R_i,                 'm'),
            "R_o_outlet": Q_(R_o,                 'm'),
            "R_i_outlet": Q_(R_i,                 'm'),
            "inlet_wedge": Q_(15,             'deg'),
            "exit_wedge": Q_(3.0,                 'deg'),
            "zeta_ung":   Q_(6.5,                 'deg'),
            "rotating":   False,
        }

        for frac in speed_fracs:
            N_actual = N_equiv_design * frac * np.sqrt(Tt_test / 288.15)
            RPM      = Q_(N_actual, 'rpm')
            print(f"\n{'='*60}\n  {frac*100:.0f}% speed  area={area_frac:.2f}"
                  f"  theta={theta_deg:+.2f}°  ({N_actual:.0f} rpm)\n{'='*60}")
            gas.TPX = Tt_test, Pt_mag, composition
            all_curves[(frac, area_frac)] = build_performance_curve_moffitt(
                stator_var, rotor, inlet, gas, composition, RPM, p2_vals, N_pts
            )

    # ── Experimental data (TP-1680 validation) ───────────────────────────────────
    PR_eta_exp = np.array([
        2.0884849535570686, 2.1833890115068226, 2.2762437570431224,
        2.3953128090916564, 2.52847366233466,   2.6697072950632017,
        2.831153994047659,  3.0229314058238703, 3.1764120140384566,
        3.3198505468908017, 3.424947822549994,  3.4552785353417486,
        3.5664926306992335, 3.6797649291964785, 3.784991120387102,
        3.872035776282654,
    ])
    eta_exp = np.array([
        0.9078658478307738, 0.9055361811811478, 0.9046470492790732,
        0.9047669555411821, 0.9065213625630778, 0.9082699609771204,
        0.9085649552759422, 0.9061655853320797, 0.9015323907681662,
        0.8946450414567159, 0.8875796998046263, 0.884885335561942,
        0.8749374722627862, 0.8631379077601964, 0.8501106524978125,
        0.8379187995148627,
    ])
    PR_ang_exp = np.array([
        1.9767769039400702, 2.063630057416988,  2.2745696511799594,
        2.491754630829106,  2.5817335417171607, 2.696540772747407,
        2.7989675528551095, 2.8982808334238745, 2.979005050800523,
        3.0379776739573146, 3.246012152008759,  3.4044451502007225,
        3.5629333088175894, 3.7742896702353788, 3.83959348438375,
    ])
    angle_exp = np.array([
        -21.201745075260558, -17.275660044629795, -8.011383105866226,
          0.4625953815680539,  3.9146350636434164,  8.150287084938697,
         11.12627558482587,   14.260724941705462,  15.979055753817349,
         17.702066844405813,  21.759868283591423,  22.83031483230394,
         22.48063116898313,   21.015035394605952,  19.73865659292443,
    ])

    # ── Plot map — one figure per area fraction ──────────────────────────────────
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(speed_fracs)))

    for area_frac in area_fracs:
        theta_deg = area_theta[area_frac]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]
        fig.suptitle(f'TP-1680 — Area Frac={area_frac:.2f}  θ={theta_deg:+.2f}°',
                     fontsize=13, fontweight='bold')

        for idx, frac in enumerate(speed_fracs):
            curve = all_curves.get((frac, area_frac), [])
            if not curve: continue
            PRs   = [pt["PR"]  for pt in curve]
            Wcs   = [corrected_mdot(pt["mdot"]) for pt in curve]
            etas  = [pt["eta"] * 100 for pt in curve]
            label = f'{frac*100:.0f}%'
            ax1.plot(PRs, Wcs,  color=colors[idx], lw=2, marker='o', ms=3, label=label)
            ax2.plot(PRs, etas, color=colors[idx], lw=2, marker='o', ms=3, label=label)

        ax1.axhline(3.856, color='k', ls='--', lw=1.5, label='Exp choked 3.856')
        ax1.axvline(3.44,  color='gray', ls=':', lw=1)
        ax2.axvline(3.44,  color='gray', ls=':', lw=1, label='Design PR=3.44')
        ax2.axhline(88.6,  color='gray', ls='--', lw=1, label='Design η=88.6%')
        ax1.set_xlabel('PR'); ax1.set_ylabel('Wc (kg/s)')
        ax1.set_title('Mass Flow Map')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('PR'); ax2.set_ylabel('η (%)')
        ax2.set_title('Efficiency Map')
        ax2.set_ylim(60, 100)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # 100% speed validation
        des_curve = all_curves.get((1.00, area_frac), [])
        if des_curve:
            ax3.plot([pt["PR"] for pt in des_curve],
                     [corrected_mdot(pt["mdot"]) for pt in des_curve],
                     'b-o', lw=2, ms=4, label='Calc 100%')
            ax3.axhline(3.708, color='k', ls='--', lw=1.5, label='Des choked 3.708')
            ax3.axvline(3.44,  color='gray', ls=':', lw=1)
            ax4.plot([pt["PR"] for pt in des_curve],
                     [pt["eta"]*100 for pt in des_curve],
                     'b-o', lw=2, ms=4, label='Calc 100%')
            ax4.plot(PR_eta_exp, eta_exp*100, 'ks--', lw=2, ms=7, label='Exp (Fig 11)')
            ax4.axvline(3.44, color='gray', ls=':', lw=1.5, label='Design PR=3.44')
            ax4.axhline(89.2, color='gray', ls='--', lw=1,  label='Design η=89.2%')

        ax3.set_xlabel('PR'); ax3.set_ylabel('Wc (kg/s)')
        ax3.set_title('100% Speed — Mass Flow Validation')
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
        ax4.set_xlabel('PR'); ax4.set_ylabel('η (%)')
        ax4.set_title('100% Speed — Efficiency Validation')
        ax4.set_ylim(75, 100)
        ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'tp1680_map_area{area_frac:.2f}.png'
        plt.savefig(fname, dpi=150)
        plt.show()

    # ── Print map table ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TP-1680 Performance Map")
    print(f"  {'area_frac':>10} {'Nc_frac':>8} {'PR':>8} {'Wc':>10} {'eta':>8} {'choke':>8}")
    print(f"  {'-'*60}")
    for area_frac in area_fracs:
        for frac in speed_fracs:
            for pt in all_curves.get((frac, area_frac), []):
                print(f"  {area_frac:>10.2f} {frac:>8.2f} {pt['PR']:>8.3f} "
                      f"{corrected_mdot(pt['mdot']):>10.4f} "
                      f"{pt['eta']:>8.4f} "
                      f"{pt['choke']:>8}")
    print(f"{'='*70}")

    # ── pyCycle map generation ────────────────────────────────────────────────────
    def write_pycycle_map(all_curves, speed_fracs, area_frac,
                          N_equiv_design, Tt_test, Pt_test,
                          corrected_mdot, filename='TP1680_map.py'):
        """
        Write a pyCycle-compatible MapData file for a single area fraction.

        pyCycle conventions:
          NpMap  : corrected speed  [rpm]  = N * sqrt(Tref/Tt) / (Pref/Pt)  (but pyCycle
                   normalises internally, so we provide actual corrected rpm)
          WpMap  : corrected flow   [lbm/s]
          effMap : total-to-total efficiency [-]
          PRmap  : total-to-total pressure ratio [-]
          alphaMap: single value = 1.0 (no bleed variation)
        """
        from scipy.interpolate import interp1d

        KG_TO_LBM = 2.20462

        # Design corrected speed at 100%
        N_corr_design = N_equiv_design  # rpm — already at standard conditions

        # Common PR grid — union of all computed PR points, trimmed to valid range
        all_PRs = []
        for frac in speed_fracs:
            curve = all_curves.get((frac, area_frac), [])
            all_PRs.extend([pt['PR'] for pt in curve])
        PR_min = max(1.05, min(all_PRs))
        PR_max = min(7.0,  max(all_PRs))
        N_pr   = 20
        PR_grid = np.linspace(PR_min, PR_max, N_pr)

        # NpMap — corrected speed for each speed line [rpm]
        Np_vals = []
        for frac in speed_fracs:
            N_corr = N_equiv_design * frac   # corrected rpm at standard day
            Np_vals.append(round(N_corr, 1))
        Np_vals = np.array(Np_vals)

        # Interpolate each speed line onto common PR grid
        effMap_rows = []
        WpMap_rows  = []

        for frac in speed_fracs:
            curve = all_curves.get((frac, area_frac), [])
            if not curve:
                effMap_rows.append(np.full(N_pr, np.nan))
                WpMap_rows.append(np.full(N_pr, np.nan))
                continue

            PRs  = np.array([pt['PR']  for pt in curve])
            effs = np.array([pt['eta'] for pt in curve])
            Wcs  = np.array([corrected_mdot(pt['mdot']) * KG_TO_LBM for pt in curve])

            # Sort by PR
            idx  = np.argsort(PRs)
            PRs  = PRs[idx]; effs = effs[idx]; Wcs = Wcs[idx]

            # Clamp PR grid to available range for this speed line
            PR_lo_i = max(PR_grid[0],  PRs[0])
            PR_hi_i = min(PR_grid[-1], PRs[-1])
            PR_interp = np.clip(PR_grid, PR_lo_i, PR_hi_i)

            f_eff = interp1d(PRs, effs, kind='linear', fill_value='extrapolate')
            f_Wc  = interp1d(PRs, Wcs,  kind='linear', fill_value='extrapolate')

            effMap_rows.append(f_eff(PR_interp))
            WpMap_rows.append(f_Wc(PR_interp))

        effMap_arr = np.array(effMap_rows)   # shape (n_speed, n_PR)
        WpMap_arr  = np.array(WpMap_rows)    # shape (n_speed, n_PR)

        # pyCycle wants shape (n_alpha, n_Np, n_PR)
        # Replicate single slice 3 times — slinear needs >= 3 points per dim
        # alpha=1.0 is no-cooling (solid turbine); engine always uses alpha=1.0
        effMap_3d = np.stack([effMap_arr, effMap_arr, effMap_arr], axis=0)
        WpMap_3d  = np.stack([WpMap_arr,  WpMap_arr,  WpMap_arr],  axis=0)

        # Design point values (100% speed)
        des_curve = all_curves.get((1.00, area_frac), [])
        if des_curve:
            des_PR  = max(pt['PR'] for pt in des_curve if pt['choke'] != 'none') \
                      if any(pt['choke'] != 'none' for pt in des_curve) else des_curve[-1]['PR']
            des_Wc  = corrected_mdot(des_curve[-1]['mdot']) * KG_TO_LBM
            des_eff = des_curve[-1]['eta']
        else:
            des_PR = 3.44; des_Wc = 8.5; des_eff = 0.886

        # Format numpy array as compact string
        def fmt_arr1d(arr, indent=8):
            vals = ', '.join(f'{v:.4f}' for v in arr)
            return f'{" "*indent}[{vals}]'

        def fmt_arr3d(arr, name, indent=8):
            lines = [f'{" "*indent}{name} = np.array([']
            lines.append(f'{" "*(indent+4)}[')   # alpha dim
            for i, row in enumerate(arr[0]):
                comma = ',' if i < len(arr[0]) - 1 else ''
                lines.append(f'{" "*(indent+8)}{fmt_arr1d(row).strip()}{comma}')
            lines.append(f'{" "*(indent+4)}]')
            lines.append(f'{" "*indent}])')
            return '\n'.join(lines)

        lines = [
            f'import numpy as np',
            f'from pycycle.maps.map_data import MapData',
            f'',
            f'',
            f'TP1680Map = MapData()',
            f'',
            f'# TP-1680 (Moffitt 1980) turbine map — area_frac={area_frac:.2f}',
            f'# Generated from meanline off-design solver',
            f'# Tt_ref={Tt_test:.1f} K  Pt_ref={Pt_test:.0f} Pa',
            f'',
            f'TP1680Map.defaults = {{}}',
            f"TP1680Map.defaults['alphaMap'] = 1.0",
            f"TP1680Map.defaults['NpMap']    = {N_corr_design:.1f}",
            f"TP1680Map.defaults['PRmap']    = {des_PR:.3f}",
            f'',
            f'TP1680Map.alphaMap = np.array([0.0, 1.0, 2.0])  # replicated — no cooling model yet',
            f'TP1680Map.NpMap    = np.array({list(np.round(Np_vals, 1))})',
            f'TP1680Map.PRmap    = np.array({list(np.round(PR_grid, 4))})',
            f'',
        ]

        # effMap — 3 identical alpha slices
        lines.append('TP1680Map.effMap = np.array([')
        for a_idx, alpha_label in enumerate([0.0, 1.0, 2.0]):
            lines.append(f'    [  # alpha = {alpha_label:.1f}')
            for i, row in enumerate(effMap_3d[a_idx]):
                comma = ',' if i < len(effMap_3d[a_idx]) - 1 else ''
                lines.append(f'        {fmt_arr1d(row).strip()}{comma}')
            comma = ',' if a_idx < 2 else ''
            lines.append(f'    ]{comma}')
        lines.append('])')
        lines.append('')

        # WpMap — 3 identical alpha slices
        lines.append('TP1680Map.WpMap = np.array([')
        for a_idx, alpha_label in enumerate([0.0, 1.0, 2.0]):
            lines.append(f'    [  # alpha = {alpha_label:.1f}')
            for i, row in enumerate(WpMap_3d[a_idx]):
                comma = ',' if i < len(WpMap_3d[a_idx]) - 1 else ''
                lines.append(f'        {fmt_arr1d(row).strip()}{comma}')
            comma = ',' if a_idx < 2 else ''
            lines.append(f'    ]{comma}')
        lines.append('])')
        lines.append('')

        lines += [
            f'TP1680Map.Npts = TP1680Map.NpMap.size',
            f'',
            f"TP1680Map.units = {{}}",
            f"TP1680Map.units['NpMap'] = 'rpm'",
            f"TP1680Map.units['WpMap'] = 'lbm/s'",
            f'',
            f'TP1680Map.param_data = []',
            f'TP1680Map.output_data = []',
            f'',
            f"TP1680Map.param_data.append({{'name': 'alphaMap', 'values': TP1680Map.alphaMap,",
            f"                              'default': 1.0, 'units': None}})",
            f"TP1680Map.param_data.append({{'name': 'NpMap', 'values': TP1680Map.NpMap,",
            f"                              'default': {N_corr_design:.1f}, 'units': 'rpm'}})",
            f"TP1680Map.param_data.append({{'name': 'PRmap', 'values': TP1680Map.PRmap,",
            f"                              'default': {des_PR:.3f}, 'units': None}})",
            f'',
            f"TP1680Map.output_data.append({{'name': 'WpMap', 'values': TP1680Map.WpMap,",
            f"                               'default': np.mean(TP1680Map.WpMap), 'units': 'lbm/s'}})",
            f"TP1680Map.output_data.append({{'name': 'effMap', 'values': TP1680Map.effMap,",
            f"                               'default': np.mean(TP1680Map.effMap), 'units': None}})",
        ]

        with open(filename, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"  pyCycle map written: {filename}")
        print(f"  Shape: effMap={effMap_3d.shape}, WpMap={WpMap_3d.shape}")
        print(f"  NpMap: {np.round(Np_vals,1)}")
        print(f"  PRmap: {np.round(PR_grid[[0,-1]],3)} ({N_pr} points)")

    # Write map for area_frac=1.00
    write_pycycle_map(
        all_curves, speed_fracs, area_frac=1.00,
        N_equiv_design=N_equiv_design,
        Tt_test=Tt_test, Pt_test=Pt_test,
        corrected_mdot=corrected_mdot,
        filename='TP1680_map.py'
    )