import numpy as np
import matplotlib.pyplot as plt

delta_eta = np.load('delta_eta_opr_tt4.npy')
OPR_vals  = np.load('OPR_vals.npy')
Tt4_vals  = np.load('Tt4_vals.npy')

T2        = 288.15
TR_vals   = Tt4_vals / T2

fn_fracs_sweep = [0.60, 1.00]

OPR_grid, TR_grid = np.meshgrid(OPR_vals, TR_vals, indexing='ij')

vmax_global = max(
    np.nanmax(np.abs(delta_eta[:, :, k]))
    for k in range(len(fn_fracs_sweep))
)

fig, axes = plt.subplots(1, len(fn_fracs_sweep),
                          figsize=(7*len(fn_fracs_sweep), 6), sharey=True)
if len(fn_fracs_sweep) == 1:
    axes = [axes]

cf_last = None
for ax, k, fn_frac in zip(axes, range(len(fn_fracs_sweep)), fn_fracs_sweep):
    Z      = delta_eta[:, :, k]
    cf_last = ax.contourf(OPR_grid, TR_grid, Z, levels=20,
                          cmap='RdBu', vmin=-vmax_global, vmax=vmax_global)

    if not np.all(np.isnan(Z)):
        try:
            cs = ax.contour(OPR_grid, TR_grid, Z, levels=[0.0],
                            colors='k', linewidths=2.5)
            ax.clabel(cs, fmt='Break-even', fontsize=9)
        except Exception:
            pass

    ax.plot(13.5, 1316.7/T2, '*', ms=14, color='gold',
            markeredgecolor='k', zorder=5, label=f'Design ({1316.7/T2:.2f})')

    ax.set_xlabel('Overall Pressure Ratio (OPR)', fontsize=11)
    ax.set_ylabel('Temperature Ratio  T₄/T₂  (—)', fontsize=11)
    ax.set_title(f'VGT Benefit — {fn_frac*100:.0f}% Thrust\n'
                 'Red = VGT better  |  Blue = Fixed better',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, color='white')

# single shared colorbar on the right
fig.colorbar(cf_last, ax=axes[-1], label='Δη_th (%) — VGT minus Fixed',
             fraction=0.046, pad=0.04)

fig.suptitle(
    'VGT vs Fixed Turbine — Δη_th Contour Map\n'
    'Physics dP/P  |  Optimal A4 scheduling for VGT',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('vgt_benefit_contour_TR.png', dpi=150, bbox_inches='tight')
plt.show()