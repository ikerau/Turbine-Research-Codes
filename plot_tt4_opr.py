"""
plot_tt4_opr.py
---------------
Reload saved sweep data and generate separate figures for each result type.
Y-axis: gas-generator temperature ratio  TR = Tt4 / T2  (dimensionless).
Data is bicubic-spline interpolated to a fine grid before plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import RectBivariateSpline
import cantera as ct

# ── Cycle model for OPR locus (variable γ via Cantera) ───────────────────────

_GAS = ct.Solution("gri30.yaml")
_AIR = "O2:0.21, N2:0.79"

def _gas_props(T_K, P_Pa):
    _GAS.TPX = T_K, P_Pa, _AIR
    cp = float(_GAS.cp_mass)
    cv = float(_GAS.cv_mass)
    return cp, cp / cv

def _run_cycle(pr, T04,
               eta_c=0.83, eta_t=0.88, eta_n=0.98,
               eta_comb=0.999, dp=0.05, Q_r=43.2e6,
               Pa=101325.0, Ta=288.15):
    _, g2 = _gas_props(Ta, Pa)
    P3  = pr * Pa
    T3s = Ta * pr ** ((g2 - 1.0) / g2)
    T3  = Ta + (T3s - Ta) / eta_c
    cp3, _  = _gas_props(0.5 * (Ta + T3), 0.5 * (Pa + P3))
    dH_comp = cp3 * (T3 - Ta)

    P4       = P3 * (1.0 - dp)
    cp4, _   = _gas_props(T04, P4)
    denom    = eta_comb * Q_r - cp4 * T04
    if denom <= 0.0 or T3 >= T04:
        return None
    f = (cp4 * T04 - cp3 * T3) / denom

    T5 = T04 - dH_comp / ((1.0 + f) * cp4)
    if not (0.0 < T5 < T04):
        return None
    T5_is = T04 - (T04 - T5) / eta_t
    term  = T5_is / T04
    if term < 0.01:
        return None

    _, g5 = _gas_props(0.5 * (T04 + T5), P4)
    P5 = P4 * term ** (g5 / (g5 - 1.0))
    if P5 <= Pa:
        return None

    cp9, g9 = _gas_props(T5, P5)
    val = 2.0 * eta_n * cp9 * T5 * (1.0 - (Pa / P5) ** ((g9 - 1.0) / g9))
    if val <= 0.0:
        return None
    u_e     = np.sqrt(val)
    Fn_spec = (1.0 + f) * u_e
    sfc     = f / Fn_spec
    return dict(sfc=sfc, Fn_spec=Fn_spec, combined=Fn_spec / sfc)


def _compute_loci(T04_arr, pr_arr=np.linspace(2.0, 30.0, 300)):
    """Return (locus_sfc, locus_fn, locus_j) each shaped (N, 2) = [T04, OPR_opt]."""
    locus_sfc, locus_fn, locus_j = [], [], []
    for T04 in T04_arr:
        rows = [(pr, _run_cycle(pr, T04)) for pr in pr_arr]
        rows = [(pr, d) for pr, d in rows if d is not None]
        if len(rows) < 3:
            continue
        prs  = np.array([r[0] for r in rows])
        sfcs = np.array([r[1]['sfc']      for r in rows])
        fns  = np.array([r[1]['Fn_spec']  for r in rows])
        js   = np.array([r[1]['combined'] for r in rows])
        locus_sfc.append([T04, prs[np.argmin(sfcs)]])
        locus_fn.append( [T04, prs[np.argmax(fns)]])
        locus_j.append(  [T04, prs[np.argmax(js)]])
    return (np.array(locus_sfc), np.array(locus_fn), np.array(locus_j))


print('Computing OPR loci …')
_T04_locus  = np.linspace(800.0, 1900.0, 80)
locus_sfc, locus_fn, locus_j = _compute_loci(_T04_locus)
print('  done.')

# ── Load data ────────────────────────────────────────────────────────────────
def _try_load(fname, shape_fallback=None):
    try:
        return np.load(fname)
    except FileNotFoundError:
        print(f'  Warning: {fname} not found — skipping panel')
        return np.full(shape_fallback, np.nan) if shape_fallback else None

OPR_vals           = np.load('OPR_vals.npy')
Tt4_vals           = np.load('Tt4_vals.npy')
_shape = (len(OPR_vals), len(Tt4_vals), 1)

delta_eta_own      = _try_load('delta_eta_own_opr_tt4.npy',     _shape)
delta_T_metal_own  = _try_load('delta_T_metal_own_opr_tt4.npy', _shape)
delta_T4_own       = _try_load('delta_T4_own_opr_tt4.npy',      _shape)
cool_frac_gauntner = _try_load('cool_frac_gauntner.npy',
                                (len(OPR_vals), len(Tt4_vals), 2))

T2      = 288.15
TR_vals = Tt4_vals / T2

FN_FRACS = [0.50]

# ── Fine interpolation grid ───────────────────────────────────────────────────
N_INTERP = 10
N_LEVELS = 20

OPR_fine = np.linspace(OPR_vals[0], OPR_vals[-1], N_INTERP)
TR_fine  = np.linspace(TR_vals[0],  TR_vals[-1],  N_INTERP)
OPR_grid_fine, TR_grid_fine = np.meshgrid(OPR_fine, TR_fine, indexing='ij')

SAVED = []


def _interp2d(Z):
    """Bicubic spline onto the fine grid. NaN cells filled with nanmean before fit."""
    Zf = Z.copy().astype(float)
    mask = ~np.isfinite(Zf)
    if mask.all():
        return np.full((N_INTERP, N_INTERP), np.nan)
    if mask.any():
        Zf[mask] = np.nanmean(Zf)
    spl = RectBivariateSpline(OPR_vals, TR_vals, Zf, kx=3, ky=3)
    return spl(OPR_fine, TR_fine)


# ── Axis helpers ──────────────────────────────────────────────────────────────

def _style_ax(ax, title):
    ax.set_xlabel('Overall Pressure Ratio (OPR)', fontsize=11)
    ax.set_ylabel(r'Temperature Ratio  $T_{t4}/T_2$  (—)', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.15, color='white')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))


def _fill_ax(ax, Z, title, levels, cmap='RdBu', zero_line=True):
    """Interpolate Z to fine grid and draw filled contour into ax."""
    Zi = _interp2d(Z)
    Zm = np.ma.masked_invalid(Zi)
    cf = ax.contourf(OPR_grid_fine, TR_grid_fine, Zm,
                     levels=levels, cmap=cmap, extend='both')
    if zero_line and levels[0] < 0 < levels[-1]:
        cs = ax.contour(OPR_grid_fine, TR_grid_fine, Zm,
                        levels=[0.0], colors='k', linewidths=2.0)
        ax.clabel(cs, fmt='Break-even', fontsize=8, inline=True)
    _style_ax(ax, title)
    return cf


def _overlay_loci(ax):
    """Overlay min-SFC, max-Fn, max-J OPR locus lines onto an OPR×TR axes."""
    specs = [
        (locus_sfc, 'steelblue',  '--', 'Min SFC'),
        (locus_fn,  'firebrick',  ':',  'Max Sp. Thrust'),
        (locus_j,   'darkorange', '-',  'Max J = Fn/SFC'),
    ]
    for locus, color, ls, label in specs:
        if locus.size == 0:
            continue
        opr = locus[:, 1]
        tr  = locus[:, 0] / T2
        mask = ((opr >= OPR_fine[0]) & (opr <= OPR_fine[-1]) &
                (tr  >= TR_fine[0])  & (tr  <= TR_fine[-1]))
        if mask.sum() < 2:
            continue
        ax.plot(opr[mask], tr[mask], ls, color=color, lw=2.2,
                label=label, zorder=6)
    ax.legend(fontsize=7.5, loc='lower right',
              facecolor='k', labelcolor='white',
              edgecolor='white', framealpha=0.75)


def _make_contour(Z, title, cbar_label, cmap='RdBu', symmetric=True,
                  zero_line=True):
    """Single-panel figure: interpolate, auto-range levels, add colorbar."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    valid = Z[np.isfinite(Z)]
    if valid.size == 0:
        ax.set_title(title + '\n(no data)')
        return fig
    if symmetric:
        vmax = max(np.abs(valid.min()), np.abs(valid.max()))
        vmin = -vmax
    else:
        vmin, vmax = valid.min(), valid.max()
    levels = np.linspace(vmin, vmax, N_LEVELS + 1)
    cf = _fill_ax(ax, Z, title, levels, cmap=cmap, zero_line=zero_line)
    fig.colorbar(cf, ax=ax, label=cbar_label)
    _overlay_loci(ax)
    return fig


# ── Per-Fn figures ───────────────────────────────────────────────────────────

for k, fn_frac in enumerate(FN_FRACS):
    fn_pct = f'{fn_frac*100:.0f}pct'
    fn_lbl = f'{fn_frac*100:.0f}% Fn'

    # Δη_th
    Z_own      = delta_eta_own[:, :, k]
    valid_eta  = Z_own[np.isfinite(Z_own)]
    eta_vmax   = max(np.abs(valid_eta.min()), np.abs(valid_eta.max())) if valid_eta.size else 1.0
    eta_levels = np.linspace(-eta_vmax, eta_vmax, N_LEVELS + 1)

    fig_eta, ax_own = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    cf_own = _fill_ax(ax_own, Z_own,
                      f'VGT own eff — {fn_lbl}\n(scheduling + hardware benefit)',
                      eta_levels, cmap='RdBu')
    fig_eta.colorbar(cf_own, ax=ax_own, label='Δη_th (%)  VGT − Fixed')
    fig_eta.suptitle(f'Thermal Efficiency Benefit — {fn_lbl}\n'
                     'Red = VGT better  |  Blue = Fixed better',
                     fontsize=12, fontweight='bold')
    _overlay_loci(ax_own)
    SAVED.append((fig_eta, f'vgt_delta_eta_{fn_pct}.png'))

    # ΔT_metal
    fig = _make_contour(
        delta_T_metal_own[:, :, k],
        title=f'ΔT_metal (K) — VGT own eff  |  {fn_lbl}\nBlue = VGT blade runs cooler',
        cbar_label='ΔT_metal (K)  VGT − Fixed',
        cmap='RdBu_r', symmetric=True)
    SAVED.append((fig, f'vgt_delta_Tmetal_own_{fn_pct}.png'))

    # ΔT4
    fig = _make_contour(
        delta_T4_own[:, :, k],
        title=f'ΔT4 (K) — VGT own eff  |  {fn_lbl}\nBlue = VGT allows lower T4',
        cbar_label='ΔT4 (K)  VGT − Fixed',
        cmap='RdBu_r', symmetric=True)
    SAVED.append((fig, f'vgt_delta_T4_{fn_pct}.png'))


# ── Gauntner cooling prescription ────────────────────────────────────────────

cool_total = cool_frac_gauntner[:, :, 0] + cool_frac_gauntner[:, :, 1]

for Z, label, fname, cmap in [
    (cool_frac_gauntner[:, :, 0], 'NGV prescription (% W_gas)',   'cool_frac_ngv.png',   'Blues'),
    (cool_frac_gauntner[:, :, 1], 'Blade prescription (% W_gas)', 'cool_frac_blade.png', 'Reds'),
    (cool_total,                   'Total prescription (% W_gas)', 'cool_frac_total.png', 'Purples'),
]:
    fig = _make_contour(
        Z,
        title=f'Gauntner Cooling — {label}',
        cbar_label='% W_gas',
        cmap=cmap, symmetric=False, zero_line=False)
    SAVED.append((fig, fname))


# ── Save all ─────────────────────────────────────────────────────────────────

for fig, fname in SAVED:
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'  Saved: {fname}')

plt.show()
