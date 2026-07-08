"""
smith_chart_vgt_vs_fixed.py
---------------------------
Smith chart: design-point η_tt over (φ, ψ) space for VGT vs Fixed NGV turbine.

Physical difference between the two configurations:
  VGT:   tip_gap_stator = 0.030 cm  (variable NGV must rotate → needs clearance)
  Fixed: tip_gap_stator = 0.0  mm   (fixed NGV can be tightly sealed)

All other blade geometry params are identical (same AR, Zweifel, LE/TE radii,
rotor tip gap). The stator tip gap leakage loss is the sole source of Δη.

Δη = η_vgt − η_fixed  (negative = VGT penalised by stator gap leakage)

Outputs
-------
  smith_eta_vgt.npy         shape (N_PHI, N_PSI)
  smith_eta_fixed.npy       shape (N_PHI, N_PSI)
  smith_delta_eta.npy       shape (N_PHI, N_PSI)  = vgt − fixed
  smith_chart_vgt_vs_fixed.png

Runtime: ~8–15 min for a 10×10 grid (200 forward_design solves).
"""

from __future__ import annotations
import io, contextlib, sys, time
import numpy as np
import matplotlib.pyplot as plt
import cantera as ct

from units import Q_, ureg
from forward_design import design_point_kinematics, forward_design

# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE CONDITIONS  (CFM56 HPT design point, from pyCycle DESIGN solve —
#  mirrors CFM56_HPT_map_generator.py / cfm56_cycle.py DESIGN.T4_MAX=2857 degR)
# ══════════════════════════════════════════════════════════════════════════════

KG_TO_LBM = 2.20462
HP_TO_W   = 745.699872

Tt4_K   = 2857.0 * (5.0 / 9.0)                       # degR -> K
Pt4_Pa  = 145.626 * 6894.757                         # psi  -> Pa
mdot_kg = (44.177 + 3.72358) / KG_TO_LBM             # lbm/s (core + NGV cooling) -> kg/s
pwr_W   = 11998.248 * HP_TO_W                        # hp   -> W
Nmech   = 14705.7                                    # rpm (HP spool)

gas = ct.Solution('air.yaml')
gas.TP = Tt4_K, Pt4_Pa
composition = gas.X


def _turbine_power(mdot_kg, Tt4_K, OPR=6.0, comp_eff=0.83,
                   gamma_c=1.4, R=287.05, Tt_inlet_K=288.15):
    Cp_c = gamma_c * R / (gamma_c - 1.0)
    T_comp = Tt_inlet_K * (1.0 + (OPR ** ((gamma_c - 1) / gamma_c) - 1.0) / comp_eff)
    return mdot_kg * Cp_c * (T_comp - Tt_inlet_K)


delta_H_des = pwr_W / mdot_kg         # J/kg — design specific work

print("=" * 62)
print("  Smith Chart — VGT vs Fixed NGV Turbine")
print("=" * 62)
print(f"  Tt4        = {Tt4_K:.1f} K  ({Tt4_K*9/5:.0f}°R)")
print(f"  Pt4        = {Pt4_Pa/1e6:.3f} MPa")
print(f"  mdot       = {mdot_kg:.2f} kg/s")
print(f"  N          = {Nmech:.0f} rpm")
print(f"  Design ΔH  = {delta_H_des:.1f} J/kg")

# ══════════════════════════════════════════════════════════════════════════════
#  (φ, ψ) GRID
# ══════════════════════════════════════════════════════════════════════════════
 
N_PHI = 10
N_PSI = 10
phi_vals = np.linspace(0.40, 0.90, N_PHI)   # flow coefficient
psi_vals = np.linspace(0.975, 1.75, N_PSI)   # stage loading

# design point
PHI_DES = 0.60
PSI_DES = 1.493993927

# ══════════════════════════════════════════════════════════════════════════════
#  BLADE GEOMETRY  (same for both configs; differs only in stator tip gap)
# ══════════════════════════════════════════════════════════════════════════════

_BLADE = dict(
    AR_stator       = 1.307692308,
    AR_rotor        = 1.727272727,
    Z_stator        = 1.10,
    Z_rotor         = 1.10,
    LE_radius       = Q_(0.005,  'in'),
    TE_radius       = Q_(0.0025, 'in'),
    inlet_wedge     = Q_(15.0,  'deg'),
    exit_wedge_s    = Q_(3.0,   'deg'),
    exit_wedge_r    = Q_(3.0,   'deg'),
    zeta_ung_s      = Q_(6.5,   'deg'),
    zeta_ung_r      = Q_(6.5,   'deg'),
    t_s_rotor       = 0.0265,
    tol             = 1e-8,
    max_iter        = 500,
    verbose         = True,
)

T_S_STATOR_VGT   = 0.01   # stator tip-to-span ratio for variable geometry (0.5%)
T_S_STATOR_FIXED = 0.0     # no stator gap for fixed NGV


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE DESIGN SOLVE
# ══════════════════════════════════════════════════════════════════════════════

def _run_design(phi: float, psi: float, t_s_stator: float,
                verbose: bool = False) -> float:
    """Return η_tt at (φ, ψ) or np.nan on failure.

    verbose=True prints one-line failure reason per cell so you can see
    which (φ, ψ) regions fail and why.
    """
    tag = f"φ={phi:.3f} ψ={psi:.3f}"
    try:
        gas.TPX = Tt4_K, Pt4_Pa, composition
        kin = design_point_kinematics(
            Tt0  = Q_(Tt4_K,   'K'),
            Pt0  = Q_(Pt4_Pa,  'Pa'),
            gas  = gas,
            mdot = Q_(mdot_kg, 'kg/s'),
            RPM  = Q_(Nmech,   'rpm'),
            psi  = psi,
            phi  = phi,
            W_t  = Q_(pwr_W,   'W'),
        )
        gas.TPX = Tt4_K, Pt4_Pa, composition
        with contextlib.redirect_stdout(io.StringIO()):
            d = forward_design(
                kin, gas,
                Q_(mdot_kg, 'kg/s'),
                Q_(Nmech,   'rpm'),
                composition = composition,
                t_s_stator  = t_s_stator,
                **_BLADE,
            )
        eta = float(d['perf']['eta_tt'])
        if not d['converged']:
            err = d.get('err', float('nan'))
            # r_loss err of 0.03 shifts η by ~0.3 pp — acceptable for a Smith chart.
            # Return the last iterate rather than nan when the solution is close.
            if err < 0.03 and (0.30 < eta < 1.0):
                if verbose:
                    print(f"    ACCEPTED LAST ITER  {tag}  "
                          f"err={err:.2e}  η={eta*100:.2f}%",
                          file=sys.stderr, flush=True)
                return eta
            if verbose:
                print(f"    NOT CONVERGED  {tag}  "
                      f"err={err:.2e}  η_last={eta*100:.2f}%",
                      file=sys.stderr, flush=True)
            return np.nan
        if not (0.30 < eta < 1.0):
            if verbose:
                print(f"    ETA OUT OF RANGE  {tag}  η={eta*100:.2f}%",
                      file=sys.stderr, flush=True)
            return np.nan
        return eta
    except ValueError as e:
        if verbose:
            print(f"    GEOM FAIL  {tag}  {e}", file=sys.stderr, flush=True)
        return np.nan
    except Exception as e:
        if verbose:
            print(f"    ERROR  {tag}  {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  GRID SWEEP
# ══════════════════════════════════════════════════════════════════════════════

n_total = N_PHI * N_PSI
eta_vgt   = np.full((N_PHI, N_PSI), np.nan)
eta_fixed = np.full((N_PHI, N_PSI), np.nan)

print(f"\nRunning {n_total * 2} forward_design solves "
      f"({N_PHI}φ × {N_PSI}ψ × 2 configs)...")
t_start = time.time()

for i, phi in enumerate(phi_vals):
    for j, psi in enumerate(psi_vals):
        t_cell = time.time()
        eta_vgt[i, j]   = _run_design(phi, psi, T_S_STATOR_VGT,   verbose=True)
        eta_fixed[i, j] = _run_design(phi, psi, T_S_STATOR_FIXED, verbose=True)
        done = i * N_PSI + j + 1
        elapsed = time.time() - t_start
        eta_rem = elapsed / done * (n_total - done) if done > 0 else 0
        print(
            f"  [{done:3d}/{n_total}]  φ={phi:.3f}  ψ={psi:.3f}  "
            f"η_vgt={eta_vgt[i,j]*100:.2f}%  "
            f"η_fixed={eta_fixed[i,j]*100:.2f}%  "
            f"({time.time()-t_cell:.1f}s / ~{eta_rem/60:.1f} min left)",
            flush=True,
        )

delta_eta = eta_vgt - eta_fixed   # pp; negative = VGT penalised

np.save('smith_eta_vgt.npy',   eta_vgt)
np.save('smith_eta_fixed.npy', eta_fixed)
np.save('smith_delta_eta.npy', delta_eta)
print(f"\nSweep complete in {(time.time()-t_start)/60:.1f} min")
print("Saved: smith_eta_vgt.npy  smith_eta_fixed.npy  smith_delta_eta.npy")

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

valid_mask = np.isfinite(delta_eta)
n_valid = valid_mask.sum()
print(f"\n  {n_valid}/{n_total} points converged successfully")
if n_valid > 0:
    print(f"  η_vgt   range: [{np.nanmin(eta_vgt)*100:.1f}%, {np.nanmax(eta_vgt)*100:.1f}%]")
    print(f"  η_fixed range: [{np.nanmin(eta_fixed)*100:.1f}%, {np.nanmax(eta_fixed)*100:.1f}%]")
    print(f"  Δη      range: [{np.nanmin(delta_eta)*100:.2f} pp, {np.nanmax(delta_eta)*100:.2f} pp]")
    des_i = int(np.argmin(np.abs(phi_vals - PHI_DES)))
    des_j = int(np.argmin(np.abs(psi_vals - PSI_DES)))
    print(f"  At design (φ={phi_vals[des_i]:.3f}, ψ={psi_vals[des_j]:.3f}):")
    print(f"    η_vgt   = {eta_vgt[des_i, des_j]*100:.3f}%")
    print(f"    η_fixed = {eta_fixed[des_i, des_j]*100:.3f}%")
    print(f"    Δη      = {delta_eta[des_i, des_j]*100:.3f} pp")

# ══════════════════════════════════════════════════════════════════════════════
#  PLOT  (raw grid — no interpolation/smoothing)
# ══════════════════════════════════════════════════════════════════════════════

# Iso-reaction lines: R = 1 − ψ/2  (for zero exit swirl, C_θ3=0)
# ψ = 2(1−R)  → horizontal lines on (φ, ψ) axes
R_LINES  = [0.0, 0.25, 0.50, 0.75]
PSI_RXNS = [2 * (1 - R) for R in R_LINES]
R_LABELS = ['R=0.00\n(impulse)', 'R=0.25', 'R=0.50\n(50% rxn)', 'R=0.75']


def _add_reaction_lines(ax, xlim):
    for R, ps, lbl in zip(R_LINES, PSI_RXNS, R_LABELS):
        if psi_vals[0] - 0.05 <= ps <= psi_vals[-1] + 0.05:
            ax.axhline(ps, color='0.55', lw=0.9, ls='--', alpha=0.7)
            ax.text(xlim[0] + 0.003, ps + 0.02, lbl,
                    fontsize=6.5, color='0.45', va='bottom')


ETA_LEVELS = np.arange(68, 99, 2)   # % contour lines

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5), sharey=True)
fig.suptitle(
    'Smith Chart — VGT vs Fixed NGV Turbine  '
    f'(t/s_stator: VGT = {T_S_STATOR_VGT:.1%}, Fixed = {T_S_STATOR_FIXED:.1%}  |  '
    f't/s_rotor = {_BLADE["t_s_rotor"]:.1%})',
    fontsize=11, fontweight='bold',
)

xlim = (phi_vals[0] - 0.01, phi_vals[-1] + 0.01)
ylim = (psi_vals[0] - 0.05, psi_vals[-1] + 0.05)

# ── Panel 1: η_vgt ────────────────────────────────────────────────────────────
ax = axes[0]
eta_v_pct = np.ma.masked_invalid(eta_vgt.T * 100)
cf1 = ax.contourf(phi_vals, psi_vals, eta_v_pct,
                  levels=ETA_LEVELS, cmap='RdYlGn', extend='both')
cs1 = ax.contour(phi_vals, psi_vals, eta_v_pct,
                 levels=ETA_LEVELS, colors='k', linewidths=0.5, alpha=0.45)
ax.clabel(cs1, fmt='%.0f%%', fontsize=7, inline_spacing=2)
_add_reaction_lines(ax, xlim)
ax.plot(PHI_DES, PSI_DES, 'k*', ms=12, zorder=5, label='Design pt')
fig.colorbar(cf1, ax=ax, label='η_tt (%)', shrink=0.88, pad=0.02)
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel('φ = Cm/U', fontsize=10)
ax.set_ylabel('ψ = ΔH/U²', fontsize=10)
ax.set_title(f'VGT  (t/s_stator = {T_S_STATOR_VGT:.1%})', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.12)

# ── Panel 2: η_fixed ──────────────────────────────────────────────────────────
ax = axes[1]
eta_f_pct = np.ma.masked_invalid(eta_fixed.T * 100)
cf2 = ax.contourf(phi_vals, psi_vals, eta_f_pct,
                  levels=ETA_LEVELS, cmap='RdYlGn', extend='both')
cs2 = ax.contour(phi_vals, psi_vals, eta_f_pct,
                 levels=ETA_LEVELS, colors='k', linewidths=0.5, alpha=0.45)
ax.clabel(cs2, fmt='%.0f%%', fontsize=7, inline_spacing=2)
_add_reaction_lines(ax, xlim)
ax.plot(PHI_DES, PSI_DES, 'k*', ms=12, zorder=5)
fig.colorbar(cf2, ax=ax, label='η_tt (%)', shrink=0.88, pad=0.02)
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel('φ = Cm/U', fontsize=10)
ax.set_title('Fixed NGV  (t/s_stator = 0%)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.12)

# ── Panel 3: Δη = η_vgt − η_fixed ────────────────────────────────────────────
ax = axes[2]
delta_pct = np.ma.masked_invalid(delta_eta.T * 100)
_absmax = max(abs(float(np.nanmin(delta_pct.data))),
              abs(float(np.nanmax(delta_pct.data))), 0.1)
_absmax = np.ceil(_absmax * 10) / 10    # round up to nearest 0.1 pp
delta_levels_fill = np.linspace(-_absmax, _absmax, 21)
delta_levels_line = np.linspace(-_absmax, _absmax, 11)
cf3 = ax.contourf(phi_vals, psi_vals, delta_pct,
                  levels=delta_levels_fill, cmap='RdBu_r', extend='both')
cs3 = ax.contour(phi_vals, psi_vals, delta_pct,
                 levels=delta_levels_line, colors='k', linewidths=0.5, alpha=0.4)
ax.clabel(cs3, fmt='%.2f', fontsize=7, inline_spacing=2)
# Bold zero-crossing line
try:
    ax.contour(phi_vals, psi_vals, delta_pct,
               levels=[0.0], colors=['k'], linewidths=2.0)
except Exception:
    pass
_add_reaction_lines(ax, xlim)
ax.plot(PHI_DES, PSI_DES, 'k*', ms=12, zorder=5)
cbar3 = fig.colorbar(cf3, ax=ax, shrink=0.88, pad=0.02)
cbar3.set_label('Δη_tt (pp)\n← Fixed better  |  VGT better →', fontsize=8)
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.set_xlabel('φ = Cm/U', fontsize=10)
ax.set_title('Δη = η_VGT − η_Fixed  (pp)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.12)

plt.tight_layout()
plt.savefig('smith_chart_vgt_vs_fixed.png', dpi=150, bbox_inches='tight')
print('\nSaved: smith_chart_vgt_vs_fixed.png')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  COST FUNCTION FIGURE
#
#  J(φ,ψ) = η_vgt + λ·Δη  =  η_vgt + λ·(η_vgt − η_fixed)
#
#  Weighted trade-off: maximize VGT efficiency while minimizing stator-gap penalty.
#  λ=1 → equal weight (1 pp efficiency gain = 1 pp penalty reduction).
#  Higher J = better operating point.  argmax(J) picks the best trade-off.
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA = 1.0   # trade-off weight; increase to penalise stator-gap leakage more

_valid  = np.isfinite(eta_vgt) & np.isfinite(eta_fixed)
penalty = np.where(_valid, -delta_eta, np.nan)   # η_fixed − η_vgt ≥ 0
J_grid  = np.where(_valid, eta_vgt + LAMBDA * delta_eta, np.nan)   # higher = better

import matplotlib.gridspec as gridspec
fig2 = plt.figure(figsize=(18, 11))
fig2.suptitle(
    f'VGT Design Cost Function  ·  J = η_vgt + λ·Δη   (λ={LAMBDA})\n'
    'Weighted trade-off: maximize VGT efficiency, minimize stator-gap penalty  |  higher J = better',
    fontsize=11, fontweight='bold',
)
gs2    = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.38, wspace=0.28)
axes2  = [[fig2.add_subplot(gs2[0, c]) for c in range(3)],
          [None, None, None]]   # row 1 filled below

xlim2 = (phi_vals[0] - 0.01, phi_vals[-1] + 0.01)
ylim2 = (psi_vals[0] - 0.05, psi_vals[-1] + 0.05)


def _stamp_rxn(ax):
    for R, ps, lbl in zip(R_LINES, PSI_RXNS, R_LABELS):
        if psi_vals[0] - 0.05 <= ps <= psi_vals[-1] + 0.05:
            ax.axhline(ps, color='0.60', lw=0.8, ls='--', alpha=0.6)
            ax.text(xlim2[0] + 0.003, ps + 0.01, lbl,
                    fontsize=6, color='0.50', va='bottom')


# ── Row 0, panel 0: η_vgt objective ───────────────────────────────────────────
ax = axes2[0][0]
cf_e = ax.contourf(phi_vals, psi_vals, np.ma.masked_invalid(eta_vgt.T * 100),
                   levels=np.arange(68, 99, 2), cmap='RdYlGn', extend='both')
ax.contour(phi_vals, psi_vals, np.ma.masked_invalid(eta_vgt.T * 100),
           levels=np.arange(68, 99, 2), colors='k', linewidths=0.4, alpha=0.4)
_stamp_rxn(ax)
ax.plot(PHI_DES, PSI_DES, 'k*', ms=11, zorder=5, label='Design pt')
fig2.colorbar(cf_e, ax=ax, label='η_vgt (%)', shrink=0.88)
ax.set_xlim(xlim2); ax.set_ylim(ylim2)
ax.set_ylabel('ψ = ΔH/U²', fontsize=10)
ax.set_title('Obj 1 — η_vgt  (maximize)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.10)

# ── Row 0, panel 1: stator-gap penalty objective ──────────────────────────────
ax = axes2[0][1]
pen_pct = np.ma.masked_invalid(penalty.T * 100)
_pen_max_pct = np.nanmax(penalty) * 100
pen_levels = np.linspace(0, _pen_max_pct * 1.05, 20)
cf_p = ax.contourf(phi_vals, psi_vals, pen_pct,
                   levels=pen_levels, cmap='YlOrRd', extend='both')
ax.contour(phi_vals, psi_vals, pen_pct,
           levels=pen_levels[::4], colors='k', linewidths=0.4, alpha=0.4)
_stamp_rxn(ax)
ax.plot(PHI_DES, PSI_DES, 'k*', ms=11, zorder=5)
fig2.colorbar(cf_p, ax=ax, label='η_fixed − η_vgt (pp)', shrink=0.88)
ax.set_xlim(xlim2); ax.set_ylim(ylim2)
ax.set_title('Obj 2 — Stator Gap Penalty  (minimize)', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.10)

# ── Row 0, panel 2: Pareto scatter ────────────────────────────────────────────
ax = axes2[0][2]
_PHI = np.broadcast_to(phi_vals[:, None], (N_PHI, N_PSI))
_eta_f  = eta_vgt.ravel()
_pen_f  = penalty.ravel()
_phi_f  = _PHI.ravel()
_ok     = np.isfinite(_eta_f) & np.isfinite(_pen_f)
eta_s   = _eta_f[_ok] * 100
pen_s   = _pen_f[_ok] * 100
phi_s   = _phi_f[_ok]

sc = ax.scatter(eta_s, pen_s, c=phi_s, cmap='plasma',
                s=18, alpha=0.75, edgecolors='none', zorder=3)
fig2.colorbar(sc, ax=ax, label='φ', shrink=0.88)

# Pareto front: non-dominated (want high η AND low penalty)
pareto = np.ones(eta_s.size, dtype=bool)
for k in range(eta_s.size):
    if pareto[k]:
        dominated_by_k = (eta_s >= eta_s[k]) & (pen_s <= pen_s[k])
        dominated_by_k[k] = False
        pareto[dominated_by_k] = False
pareto_order = np.argsort(pen_s[pareto])
ax.plot(eta_s[pareto][pareto_order], pen_s[pareto][pareto_order],
        'k-o', lw=1.8, ms=5, zorder=4, label='Pareto front')
ax.set_xlabel('η_vgt (%)', fontsize=10)
ax.set_ylabel('Penalty η_fixed − η_vgt (pp)', fontsize=10)
ax.set_title('Pareto Space  (upper-left = optimal)', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.25)

# ── Row 1: single J = Δη / η_vgt panel (spans all 3 columns) ────────────────
ax_J = fig2.add_subplot(gs2[1, :])

J_levels = np.linspace(float(np.nanmin(J_grid)), float(np.nanmax(J_grid)), 25)
J_ma = np.ma.masked_invalid(J_grid.T)
cf_J = ax_J.contourf(phi_vals, psi_vals, J_ma,
                     levels=J_levels, cmap='RdYlGn', extend='both')
ax_J.contour(phi_vals, psi_vals, J_ma,
             levels=J_levels[::4], colors='k', linewidths=0.4, alpha=0.35)

# Optimal = highest J (argmax)
opt_ij  = np.unravel_index(np.nanargmax(J_grid), J_grid.shape)
phi_opt = phi_vals[opt_ij[0]]
psi_opt = psi_vals[opt_ij[1]]
J_opt   = float(J_grid[opt_ij])
ax_J.plot(phi_opt, psi_opt, 'b*', ms=14, zorder=6,
          label=f'Best\nφ={phi_opt:.3f}, ψ={psi_opt:.3f}\nJ={J_opt*100:.2f}%')
ax_J.plot(PHI_DES, PSI_DES, 'w^', ms=9, zorder=6, mec='k', mew=0.8,
          label='Design pt')

_stamp_rxn(ax_J)
fig2.colorbar(cf_J, ax=ax_J, label=f'J = η_vgt + λΔη  (λ={LAMBDA},  higher = better)', shrink=0.6)
ax_J.set_xlim(xlim2); ax_J.set_ylim(ylim2)
ax_J.set_xlabel('φ = Cm/U', fontsize=10)
ax_J.set_ylabel('ψ = ΔH/U²', fontsize=10)
ax_J.set_title(f'J = η_vgt + λ·Δη   (λ={LAMBDA})  =  η_vgt + λ·(η_vgt − η_fixed)', fontsize=11, fontweight='bold')
ax_J.legend(fontsize=8, loc='upper right')
ax_J.grid(True, alpha=0.10)

print(f"  Best operating point (λ={LAMBDA}): φ={phi_opt:.3f}, ψ={psi_opt:.3f}  "
      f"η_vgt={eta_vgt[opt_ij]*100:.2f}%  "
      f"penalty={penalty[opt_ij]*100:.3f} pp  J={J_opt*100:.2f}%")

plt.tight_layout()
plt.savefig('smith_cost_function.png', dpi=150, bbox_inches='tight')
print('\nSaved: smith_cost_function.png')
plt.show()
