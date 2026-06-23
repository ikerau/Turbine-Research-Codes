"""
sfc_pr_debug.py
---------------
Variable-γ turbojet cycle sweep (Cantera) across OPR for several T04 values.

Combined cost:  J = Fn_spec / SFC  =  (1+f)² · u_e² / f

Maximising J trades off specific thrust (high = smaller/lighter engine)
against fuel consumption (low SFC = long range).  The peak of J sits
between the OPR that maximises Fn_spec and the OPR that minimises SFC.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cantera as ct

# ── Parameters ────────────────────────────────────────────────────────────────

PA       = 101325.0   # Pa    — ambient pressure (SLS)
TA       = 288.15     # K     — ambient temperature (SLS)
ETA_C    = 0.83       # compressor isentropic efficiency
ETA_T    = 0.88       # turbine isentropic efficiency
ETA_N    = 0.98       # nozzle efficiency
ETA_COMB = 0.999      # combustion efficiency
DP       = 0.05       # combustor total-pressure loss (fraction)
Q_R      = 43.2e6    # J/kg  — Jet-A lower heating value

PR_RANGE = (2.0, 30.0)
N_PR     = 300

T04_VALS = [1000, 1100, 1200, 1300, 1400, 1500, 1600]  # K

# ── Cantera setup ─────────────────────────────────────────────────────────────

_GAS = ct.Solution("gri30.yaml")
AIR  = "O2:0.21, N2:0.79"


def _gas_props(T_K, P_Pa):
    _GAS.TPX = T_K, P_Pa, AIR
    cp  = float(_GAS.cp_mass)
    cv  = float(_GAS.cv_mass)
    return cp, cp / cv, cp - cv   # cp, gamma, R


# ── Cycle ─────────────────────────────────────────────────────────────────────

def run_cycle(pr, T04,
              eta_c=ETA_C, eta_t=ETA_T, eta_n=ETA_N,
              eta_comb=ETA_COMB, dp=DP, Q_r=Q_R,
              Pa=PA, Ta=TA, verbose=False):
    """
    Variable-γ turbojet via Cantera.  Returns dict or None if non-physical.

    Stations: 2=inlet  3=comp exit  4=burner exit  5=turb exit  9=nozzle exit
    """
    # ── 2 → 3  compressor ────────────────────────────────────────────────────
    cp2, g2, _ = _gas_props(Ta, Pa)
    P3   = pr * Pa
    T3s  = Ta * pr ** ((g2 - 1.0) / g2)
    T3   = Ta + (T3s - Ta) / eta_c

    cp3, _, _ = _gas_props(0.5 * (Ta + T3), 0.5 * (Pa + P3))
    dH_comp   = cp3 * (T3 - Ta)               # J/kg_air — compressor work

    # ── 3 → 4  combustor ─────────────────────────────────────────────────────
    P4       = P3 * (1.0 - dp)
    cp4, g4, _ = _gas_props(T04, P4)

    denom = eta_comb * Q_r - cp4 * T04
    if denom <= 0.0 or T3 >= T04:
        return None

    f = (cp4 * T04 - cp3 * T3) / denom        # fuel-air ratio [kg/kg]

    # ── 4 → 5  turbine ───────────────────────────────────────────────────────
    T5 = T04 - dH_comp / ((1.0 + f) * cp4)
    if not (0.0 < T5 < T04):
        return None

    # Isentropic turbine exit temp from efficiency definition
    T5_is = T04 - (T04 - T5) / eta_t          # eta_t = (T04−T5)/(T04−T5_is)
    term  = T5_is / T04
    if term < 0.01:
        return None

    cp5, g5, _ = _gas_props(0.5 * (T04 + T5), P4)
    P5 = P4 * term ** (g5 / (g5 - 1.0))       # P5 = P4 · (T5_is/T04)^(γ/(γ-1))
    if P5 <= Pa:
        return None

    # ── 5 → 9  nozzle ────────────────────────────────────────────────────────
    cp9, g9, _ = _gas_props(T5, P5)
    val = 2.0 * eta_n * cp9 * T5 * (1.0 - (Pa / P5) ** ((g9 - 1.0) / g9))
    if val <= 0.0:
        return None
    u_e = np.sqrt(val)

    # ── Performance ───────────────────────────────────────────────────────────
    Fn_spec = (1.0 + f) * u_e                 # N per kg/s inlet air
    sfc     = f / Fn_spec                     # kg_fuel / (N·s)
    eta_th  = 0.5 * Fn_spec ** 2 / (f * Q_r) # thermal efficiency (static)
    combined = Fn_spec / sfc                  # = Fn_spec² / f  [m²/s²]

    d = dict(T3=T3, T3s=T3s, f=f, T5=T5, T5_is=T5_is, term=term,
             P5=P5, u_e=u_e, Fn_spec=Fn_spec, sfc=sfc,
             eta_th=eta_th, combined=combined,
             g2=g2, g4=g4, g5=g5, g9=g9)

    if verbose:
        print(f"\n  PR={pr:.2f}  T04={T04:.0f}K")
        print(f"    T3={T3:.1f}K (T3s={T3s:.1f}K)   dH_comp={dH_comp/1e3:.2f} kJ/kg")
        print(f"    f={f*100:.4f}%   T5={T5:.1f}K   T5_is={T5_is:.1f}K   term={term:.4f}")
        print(f"    g2={g2:.4f}  g4={g4:.4f}  g5={g5:.4f}  g9={g9:.4f}")
        print(f"    PR_turb={(1/term)**(g5/(g5-1)):.3f}   P5/Pa={P5/PA:.3f}")
        print(f"    u_e={u_e:.1f} m/s   Fn_spec={Fn_spec:.1f} N/(kg/s)")
        print(f"    SFC={sfc*1e6:.3f} µg/(N·s)   η_th={eta_th*100:.3f}%")
        print(f"    Combined J = Fn/SFC = {combined/1e6:.3f} ×10⁶ m²/s²")

    return d


# ── Sweep ─────────────────────────────────────────────────────────────────────

def sweep_T04(T04, pr_arr=None):
    if pr_arr is None:
        pr_arr = np.linspace(PR_RANGE[0], PR_RANGE[1], N_PR)
    out = {'pr': [], 'sfc': [], 'Fn': [], 'combined': [], 'eta': [],
           'f': [], 'ue': [], 'T3': [], 'T5': []}
    for pr in pr_arr:
        d = run_cycle(pr, T04)
        if d is not None:
            out['pr'].append(pr)
            out['sfc'].append(d['sfc'] * 1e6)       # µg/(N·s)
            out['Fn'].append(d['Fn_spec'])
            out['combined'].append(d['combined'] / 1e6)  # ×10⁶ m²/s²
            out['eta'].append(d['eta_th'] * 100)
            out['f'].append(d['f'] * 100)
            out['ue'].append(d['u_e'])
            out['T3'].append(d['T3'])
            out['T5'].append(d['T5'])
    return {k: np.array(v) for k, v in out.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Verbose check at one point ────────────────────────────────────────────
    print("=" * 60)
    print("  VERBOSE: PR=10, T04=1300 K")
    print("=" * 60)
    run_cycle(10.0, 1300.0, verbose=True)

    # ── Run all sweeps ────────────────────────────────────────────────────────
    colors  = cm.plasma(np.linspace(0.15, 0.85, len(T04_VALS)))
    sweeps  = {T04: sweep_T04(T04) for T04 in T04_VALS}

    # ── Optimum table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  OPTIMA  —  max J = Fn_spec / SFC")
    print(f"  {'T04 K':>6}  {'OPR@maxJ':>9}  {'max J ×10⁶':>11}"
          f"  {'OPR@minSFC':>11}  {'OPR@maxFn':>10}")
    print("  " + "-" * 73)
    for T04 in T04_VALS:
        s = sweeps[T04]
        if s['pr'].size == 0:
            continue
        i_j  = int(np.argmax(s['combined']))
        i_sfc = int(np.argmin(s['sfc']))
        i_fn  = int(np.argmax(s['Fn']))
        print(f"  {T04:6.0f}  {s['pr'][i_j]:9.2f}  {s['combined'][i_j]:11.3f}"
              f"  {s['pr'][i_sfc]:11.2f}  {s['pr'][i_fn]:10.2f}")
    print("=" * 75)

    # ── Intermediates table at T04=1300 K ─────────────────────────────────────
    T04_diag = 1300.0
    s = sweeps[T04_diag]
    print(f"\n  INTERMEDIATES  T04={T04_diag:.0f} K")
    print(f"  {'OPR':>5}  {'T3 K':>7}  {'T5 K':>7}  {'f %':>6}"
          f"  {'u_e':>7}  {'Fn':>8}  {'SFC µg':>8}  {'J×10⁶':>8}  {'η_th%':>7}")
    print("  " + "-" * 70)
    pr_diag = np.linspace(PR_RANGE[0], PR_RANGE[1], 20)
    for pr in pr_diag:
        d = run_cycle(pr, T04_diag)
        if d:
            print(f"  {pr:5.2f}  {d['T3']:7.1f}  {d['T5']:7.1f}  {d['f']*100:6.3f}"
                  f"  {d['u_e']:7.1f}  {d['Fn_spec']:8.1f}  "
                  f"{d['sfc']*1e6:8.3f}  {d['combined']/1e6:8.3f}  "
                  f"{d['eta_th']*100:7.3f}")

    # ── Figure 1: SFC and Fn_spec vs OPR ─────────────────────────────────────
    fig1, (ax_sfc, ax_fn) = plt.subplots(1, 2, figsize=(14, 5))
    fig1.suptitle(
        f'Turbojet Sweep — Variable γ (Cantera)  |  SLS\n'
        f'η_c={ETA_C}  η_t={ETA_T}  η_n={ETA_N}  dP/P={DP*100:.0f}%  '
        f'Q_R={Q_R/1e6:.1f} MJ/kg',
        fontsize=11, fontweight='bold')

    for i, T04 in enumerate(T04_VALS):
        s   = sweeps[T04]
        c   = colors[i]
        lbl = f'T₄={T04:.0f}K'
        if s['pr'].size == 0:
            continue

        ax_sfc.plot(s['pr'], s['sfc'], '-', color=c, lw=2, label=lbl)
        idx = int(np.argmin(s['sfc']))
        ax_sfc.plot(s['pr'][idx], s['sfc'][idx],
                    'v', color=c, ms=9, markeredgecolor='k', zorder=5)

        ax_fn.plot(s['pr'], s['Fn'], '-', color=c, lw=2, label=lbl)
        idx = int(np.argmax(s['Fn']))
        ax_fn.plot(s['pr'][idx], s['Fn'][idx],
                   '^', color=c, ms=9, markeredgecolor='k', zorder=5)

    for ax, ylabel, sym, title in [
        (ax_sfc, 'SFC  (µg fuel / N·s)',    '▼ min',  'SFC vs OPR'),
        (ax_fn,  'Specific thrust  (N per kg/s)', '▲ max', 'Specific thrust vs OPR'),
    ]:
        ax.set_xlabel('OPR', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{title}  ({sym})', fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(PR_RANGE)
        ax.axvline(13.5, color='gray', ls='--', lw=1.0, alpha=0.5)

    fig1.tight_layout()
    fig1.savefig('sfc_fn_vs_opr.png', dpi=150, bbox_inches='tight')

    # ── Figure 2: combined J = Fn / SFC ──────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(9, 6))
    fig2.suptitle(
        'Combined Cost  J = Fn_spec / SFC  (×10⁶ m²/s²)\n'
        'Maximising J balances specific thrust against fuel consumption  '
        '(★ = peak per T₄)',
        fontsize=11, fontweight='bold')

    for i, T04 in enumerate(T04_VALS):
        s   = sweeps[T04]
        c   = colors[i]
        lbl = f'T₄={T04:.0f}K'
        if s['pr'].size == 0:
            continue

        ax.plot(s['pr'], s['combined'], '-', color=c, lw=2.2, label=lbl)
        idx = int(np.argmax(s['combined']))
        ax.plot(s['pr'][idx], s['combined'][idx],
                '*', color=c, ms=16, markeredgecolor='k',
                markeredgewidth=0.8, zorder=6,
                label=f'  peak OPR={s["pr"][idx]:.1f}')

    ax.axvline(13.5, color='gray', ls='--', lw=1.2, alpha=0.6,
               label='Design OPR = 13.5')
    ax.set_xlabel('OPR', fontsize=12)
    ax.set_ylabel('J = Fn_spec / SFC  (×10⁶ m²/s²)', fontsize=12)
    ax.legend(fontsize=8.5, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(PR_RANGE)

    fig2.tight_layout()
    fig2.savefig('combined_cost_vs_opr.png', dpi=150, bbox_inches='tight')

    # ── Figure 3: decomposition at one T04 ───────────────────────────────────
    T04_diag = 1300.0
    s        = sweeps[T04_diag]

    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9))
    fig3.suptitle(
        f'Cycle Decomposition — T04={T04_diag:.0f}K  (Cantera)',
        fontsize=12, fontweight='bold')

    quants = [
        (s['T3'],      'T3  (K)',               'C0'),
        (s['T5'],      'T5  (K)',               'C1'),
        (s['f'],       'Fuel-air ratio  f (%)', 'C2'),
        (s['ue'],      'Nozzle velocity  u_e (m/s)', 'C3'),
        (s['sfc'],     'SFC  (µg / N·s)',       'C4'),
        (s['combined'],'J = Fn/SFC  (×10⁶)',   'C5'),
    ]
    for ax, (data, ylabel, color) in zip(axes3.flat, quants):
        ax.plot(s['pr'], data, '-', color=color, lw=2.2)
        if 'SFC' in ylabel:
            idx = int(np.argmin(data))
            ax.plot(s['pr'][idx], data[idx], 'v', color=color,
                    ms=10, markeredgecolor='k', zorder=5,
                    label=f'min @ OPR={s["pr"][idx]:.1f}')
        elif 'J =' in ylabel or 'Fn/SFC' in ylabel:
            idx = int(np.argmax(data))
            ax.plot(s['pr'][idx], data[idx], '*', color=color,
                    ms=14, markeredgecolor='k', zorder=5,
                    label=f'max @ OPR={s["pr"][idx]:.1f}')
        elif 'velocity' in ylabel or 'u_e' in ylabel:
            idx = int(np.argmax(data))
            ax.plot(s['pr'][idx], data[idx], '^', color=color,
                    ms=10, markeredgecolor='k', zorder=5,
                    label=f'max @ OPR={s["pr"][idx]:.1f}')
        ax.axvline(13.5, color='gray', ls='--', lw=1.0, alpha=0.5)
        ax.set_xlabel('OPR')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(PR_RANGE)

    fig3.tight_layout()
    fig3.savefig('sfc_decomposition.png', dpi=150, bbox_inches='tight')

    # ── Figure 4: OPR locus vs T04 for each objective ─────────────────────────
    # Sweep a fine T04 grid; for each T04 find the OPR that
    # minimises SFC, maximises Fn_spec, and maximises J = Fn/SFC.
    T04_fine = np.linspace(800, 1800, 60)   # K
    pr_arr   = np.linspace(PR_RANGE[0], PR_RANGE[1], N_PR)

    locus_sfc = []   # (T04, OPR_at_min_SFC)
    locus_fn  = []   # (T04, OPR_at_max_Fn)
    locus_j   = []   # (T04, OPR_at_max_J)

    print("\n  Computing OPR locus vs T04 …")
    for T04 in T04_fine:
        s = sweep_T04(T04, pr_arr)
        if s['pr'].size < 3:
            continue
        locus_sfc.append((T04, s['pr'][int(np.argmin(s['sfc']))]))
        locus_fn.append( (T04, s['pr'][int(np.argmax(s['Fn']))]))
        locus_j.append(  (T04, s['pr'][int(np.argmax(s['combined']))]))

    locus_sfc = np.array(locus_sfc)
    locus_fn  = np.array(locus_fn)
    locus_j   = np.array(locus_j)

    fig4, ax4 = plt.subplots(figsize=(9, 6))
    fig4.suptitle(
        'Optimal OPR Locus vs T₄ — Variable γ (Cantera)  |  SLS\n'
        f'η_c={ETA_C}  η_t={ETA_T}  η_n={ETA_N}  dP/P={DP*100:.0f}%',
        fontsize=11, fontweight='bold')

    ax4.plot(locus_sfc[:, 1], locus_sfc[:, 0],
             'v-', color='steelblue', lw=2.2, ms=5,
             label='Min SFC  (best range)')
    ax4.plot(locus_fn[:, 1],  locus_fn[:, 0],
             '^-', color='firebrick', lw=2.2, ms=5,
             label='Max specific thrust  (smallest engine)')
    ax4.plot(locus_j[:, 1],   locus_j[:, 0],
             '*-', color='darkorange', lw=2.2, ms=7,
             label='Max J = Fn / SFC  (combined)')

    # Shade the feasible band between min-SFC and max-Fn
    if locus_sfc.size and locus_fn.size:
        T_common = np.intersect1d(
            np.round(locus_sfc[:, 0], 1),
            np.round(locus_fn[:, 0], 1))
        if T_common.size:
            sfc_interp = np.interp(T_common, locus_sfc[:, 0], locus_sfc[:, 1])
            fn_interp  = np.interp(T_common, locus_fn[:, 0],  locus_fn[:, 1])
            ax4.fill_between(T_common, sfc_interp, fn_interp,
                             alpha=0.08, color='gray',
                             label='Feasible band (min-SFC ↔ max-Fn)')

    # Mark the design point
    ax4.axvline(13.5, color='gray', ls='--', lw=1.2, alpha=0.7,
                label='Design OPR = 13.5')
    ax4.axhline(2370 * 5/9, color='dimgray', ls=':', lw=1.2, alpha=0.7,
                label=f'Design T₄ = {2370*5/9:.0f} K')

    ax4.set_xlabel('Optimal OPR', fontsize=12)
    ax4.set_ylabel('T₄  (K)', fontsize=12)
    ax4.set_xlim(PR_RANGE[0], PR_RANGE[1])
    ax4.set_ylim(T04_fine[0], T04_fine[-1])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig('opr_locus_vs_T4.png', dpi=150, bbox_inches='tight')

    plt.show()
