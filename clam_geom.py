"""
NGV Clamshell Variable Geometry Analysis
=========================================
Splits the Pritchard profile into SS and PS, then sweeps pivot angle for:
  - PS rotation about its LE point (your clamshell concept)
  - SS rotation about its LE point (for comparison)

For each rotation angle the script finds:
  - Throat location (x, y) — minimum passage width point
  - Throat height (o) — the minimum normal distance between rotated surface
    and the fixed opposing surface

Convention: passage is between SS (lower) and PS (upper, shifted +pitch).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


# ── unit helpers ──────────────────────────────────────────────────────────────
def meter(x): return x.to("m").magnitude
def rad(a):   return a.to("rad").magnitude


# ── polyline utilities ────────────────────────────────────────────────────────
def dedup(x, y, tol=1e-12):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = [0]
    for i in range(1, len(x)):
        if abs(x[i]-x[keep[-1]]) > tol or abs(y[i]-y[keep[-1]]) > tol:
            keep.append(i)
    return x[keep], y[keep]


def split_ss_ps(x, y, tol=1e-12):
    """Split closed profile into SS (high y) and PS (low y), both LE->TE."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if abs(x[0]-x[-1]) > tol or abs(y[0]-y[-1]) > tol:
        x, y = np.r_[x, x[0]], np.r_[y, y[0]]

    i_le = np.argmin(x)
    i_te = np.argmax(x)
    n    = len(x)

    def fwd(i0, i1):
        if i0 <= i1: return np.arange(i0, i1+1)
        return np.r_[np.arange(i0, n), np.arange(0, i1+1)]

    idxA = fwd(i_le, i_te)
    idxB = fwd(i_te, i_le)[::-1]

    xA, yA = x[idxA], y[idxA]
    xB, yB = x[idxB], y[idxB]

    if np.mean(yA) >= np.mean(yB):
        return (xA, yA), (xB, yB)
    return (xB, yB), (xA, yA)


# ── Pritchard profile builder ─────────────────────────────────────────────────
def build_pritchard(params, n=80):
    LE  = meter(params["LE_radius"])
    TE  = meter(params["TE_radius"])
    Cax = meter(params["Chord_ax"])
    zeta = rad(params["zeta_ung"])
    bin_ = rad(params["beta_in"])
    bout = rad(params["beta_out"])
    iw   = rad(params["inlet_wedge"])
    ew   = rad(params["exit_wedge"])
    t    = meter(params["throat"])
    R    = meter(params["Radius"])
    pitch = 2*np.pi*R / params["N_blades"]

    b1 = bout - ew
    x1 = Cax - TE*(1 + np.sin(b1));  y1 = TE*np.cos(b1)

    b2 = bout - ew + zeta
    x2 = Cax - TE + (t+TE)*np.sin(b2);  y2 = pitch - (t+TE)*np.cos(b2)

    b3 = bin_ + iw
    x3 = LE*(1 - np.sin(b3))
    y3 = y2 + ((x2-x3)/(b2-b3))*np.log(np.cos(b2)/np.cos(b3))

    b4 = bin_ - iw
    x4 = LE*(1 + np.sin(b4));  y4 = (y3 - LE*np.cos(b3)) - LE*np.cos(b4)

    b5 = bout + ew
    x5 = Cax - TE*(1 - np.sin(b5));  y5 = -TE*np.cos(b5)

    # curve 1->2 (quadratic)
    A = np.array([[x1**2,x1,1],[x2**2,x2,1],[2*x2,1,0]], float)
    a12,b12,c12 = np.linalg.solve(A, [y1, y2, np.tan(b2)])
    x12 = np.linspace(x1, x2, n);  y12 = a12*x12**2 + b12*x12 + c12

    # curve 2->3 (log)
    x23 = np.linspace(x2, x3, n)
    bx  = (b2-b3)/(x2-x3)*(x23-x3) + b3
    y23 = y2 + ((x2-x3)/(b2-b3))*np.log(np.cos(b2)/np.cos(bx))

    # LE arc 3->4
    cx_le = x3 + LE*np.sin(b3);  cy_le = y3 - LE*np.cos(b3)
    a3 = np.arctan2(y3-cy_le, x3-cx_le)
    a4 = np.arctan2(y4-cy_le, x4-cx_le)
    d  = (a4-a3) % (2*np.pi); d = d-2*np.pi if d>np.pi else d
    th_le = a3 + np.linspace(0, d, n)
    x_le  = cx_le + LE*np.cos(th_le);  y_le = cy_le + LE*np.sin(th_le)

    # curve 4->5 (cubic)
    x45 = np.linspace(x4, x5, n)
    dd_ = (np.tan(b4)+np.tan(b5))/(x4-x5)**2 - 2*(y4-y5)/(x4-x5)**3
    cc_ = (y4-y5)/(x4-x5)**2 - np.tan(b5)/(x4-x5) - dd_*(x4+2*x5)
    bb_ = np.tan(b5) - 2*cc_*x5 - 3*dd_*x5**2
    aa_ = y5 - bb_*x5 - cc_*x5**2 - dd_*x5**3
    y45 = aa_ + bb_*x45 + cc_*x45**2 + dd_*x45**3

    # TE arc 5->1
    if TE > 0:
        cx_te = x1 + TE*np.sin(b1);  cy_te = y1 - TE*np.cos(b1)
        a5 = np.arctan2(y5-cy_te, x5-cx_te)
        a1 = np.arctan2(y1-cy_te, x1-cx_te)
        d  = (a1-a5) % (2*np.pi); d = d-2*np.pi if d>np.pi else d
        th_te = a5 + np.linspace(0, d, n)
        x_te  = cx_te + TE*np.cos(th_te);  y_te = cy_te + TE*np.sin(th_te)
    else:
        x_te, y_te = np.array([x5,x1]), np.array([y5,y1])

    xp = np.concatenate([x12, x23[1:], x_le[1:], x45[1:], x_te[1:]])
    yp = np.concatenate([y12, y23[1:], y_le[1:], y45[1:], y_te[1:]])
    xp, yp = dedup(xp, yp)
    return xp, yp, pitch, Cax


# ── throat finder ─────────────────────────────────────────────────────────────
def find_throat(x_ss, y_ss, x_ps_shifted, y_ps_shifted, n_sample=400):
    """
    Minimum distance from each point on PS (shifted) to the SS polyline,
    and vice versa.  Returns (x_throat, y_throat, throat_height).
    The throat point is on the PS (the moving surface in the clamshell concept).
    """
    # resample both curves to n_sample points by arc-length
    def resample(x, y, n):
        ds  = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s   = np.r_[0, np.cumsum(ds)]
        s_q = np.linspace(0, s[-1], n)
        return np.interp(s_q, s, x), np.interp(s_q, s, y)

    xs, ys = resample(x_ss, y_ss, n_sample)
    xp, yp = resample(x_ps_shifted, y_ps_shifted, n_sample)

    # for each point on PS, find min dist to SS
    min_dist = np.full(len(xp), np.inf)
    for i in range(len(xp)):
        d = np.sqrt((xs - xp[i])**2 + (ys - yp[i])**2)
        min_dist[i] = np.min(d)

    idx = np.argmin(min_dist)
    return xp[idx], yp[idx], min_dist[idx]


# ── rotation helper ───────────────────────────────────────────────────────────
def rotate_curve(x, y, pivot_x, pivot_y, angle_rad):
    dx, dy = x - pivot_x, y - pivot_y
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return pivot_x + c*dx - s*dy, pivot_y + s*dx + c*dy


# ── NGV parameters (typical HP turbine NGV) ───────────────────────────────────
params = dict(
    LE_radius   = Q_(0.06,  "in"),
    TE_radius   = Q_(0.020, "in"),
    Chord_ax    = Q_(1.2,   "in"),
    zeta_ung    = Q_(5.0,   "deg"),
    beta_in     = Q_(0.0,   "deg"),
    beta_out    = -Q_(68.0, "deg"),
    inlet_wedge = Q_(18.0,  "deg"),
    exit_wedge  = Q_(2.5,   "deg"),
    N_blades    = 48,
    throat      = Q_(0.32,  "in"),
    Radius      = Q_(9.5,   "in"),
)

x_prof, y_prof, pitch, Cax = build_pritchard(params, n=1500)
(x_ss, y_ss), (x_ps, y_ps) = split_ss_ps(x_prof, y_prof)

# PS shifted into passage (the upper wall in the passage view)
x_ps_base = x_ps.copy()
y_ps_base = y_ps + pitch

# Pivot points: TE of each surface
ps_pivot = (x_ps_base[-1], y_ps_base[-1])  # TE of shifted PS
ss_pivot = (x_ss[-1],      y_ss[-1])       # TE of SS

# ── sweep ─────────────────────────────────────────────────────────────────────
angles_deg = np.linspace(0, 5, 1500)   # negative = closing (throat shrinks)
angles_rad = np.radians(angles_deg)

throat_x_ps, throat_y_ps, throat_h_ps = [], [], []
throat_x_ss, throat_y_ss, throat_h_ss = [], [], []

for a in angles_rad:
    # --- PS rotation (your clamshell concept) ---
    xpr, ypr = rotate_curve(x_ps_base, y_ps_base, *ps_pivot, a)
    tx, ty, th = find_throat(x_ss, y_ss, xpr, ypr)
    throat_x_ps.append(tx); throat_y_ps.append(ty); throat_h_ps.append(th)

    # --- SS rotation (comparison) — negated so closing is also negative angle ---
    xsr, ysr = rotate_curve(x_ss, y_ss, *ss_pivot, -a)
    tx, ty, th = find_throat(xsr, ysr, x_ps_base, y_ps_base)
    throat_x_ss.append(tx); throat_y_ss.append(ty); throat_h_ss.append(th)

throat_x_ps  = np.array(throat_x_ps)
throat_y_ps  = np.array(throat_y_ps)
throat_h_ps  = np.array(throat_h_ps)
throat_x_ss  = np.array(throat_x_ss)
throat_y_ss  = np.array(throat_y_ss)
throat_h_ss  = np.array(throat_h_ss)

# convert to mm for display
M2MM = 1000.0
pitch_mm = pitch * M2MM
Cax_mm   = Cax   * M2MM

# design (0°) throat — normalisation reference
idx_0_arr        = np.argmin(np.abs(angles_deg))
throat_design_ps = throat_h_ps[idx_0_arr]
throat_design_ss = throat_h_ss[idx_0_arr]
throat_norm_ps   = throat_h_ps / throat_design_ps
throat_norm_ss   = throat_h_ss / throat_design_ss

print(f"Pitch:               {pitch_mm:.2f} mm")
print(f"Axial chord:         {Cax_mm:.2f} mm")
print(f"Design throat (PS):  {throat_design_ps*M2MM:.3f} mm")
print(f"Design throat (SS):  {throat_design_ss*M2MM:.3f} mm")

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# ── (A) two full blades showing PS pivot at 0°, ±10° ─────────────────────────
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_title("NGV cascade — two blades, PS clamshell pivot at TE (0°, ±10°)", fontsize=11)

idx_m10 = np.argmin(np.abs(angles_deg + 10))
idx_p10 = np.argmin(np.abs(angles_deg - 10))
idx_0   = np.argmin(np.abs(angles_deg))

# ── Blade 0: full profile at y as-is ──────────────────────────────────────
ax0.plot(x_ss*M2MM, y_ss*M2MM, color="steelblue", lw=2.2, label="SS — fixed")
ax0.plot(x_ps*M2MM, y_ps*M2MM, color="gray",      lw=1.5, ls="--", label="PS nominal")

ps_pivot_b0 = (x_ps[-1], y_ps[-1])
for idx, col, lbl in [(idx_m10,"#d62728","PS −10° (closing)"),
                       (idx_p10,"#2ca02c","PS +10° (opening)")]:
    xr, yr = rotate_curve(x_ps, y_ps, *ps_pivot_b0, angles_rad[idx])
    ax0.plot(xr*M2MM, yr*M2MM, color=col, lw=1.8, label=lbl)

# ── Blade 1: full profile shifted +pitch ──────────────────────────────────
ax0.plot(x_ss*M2MM, (y_ss+pitch)*M2MM, color="steelblue", lw=2.2)
ax0.plot(x_ps*M2MM, (y_ps+pitch)*M2MM, color="gray",      lw=1.5, ls="--")

ps_pivot_b1 = (x_ps[-1], y_ps[-1]+pitch)
for idx, col in [(idx_m10,"#d62728"),(idx_p10,"#2ca02c")]:
    xr, yr = rotate_curve(x_ps, y_ps+pitch, *ps_pivot_b1, angles_rad[idx])
    ax0.plot(xr*M2MM, yr*M2MM, color=col, lw=1.8)

# ── throat markers ─────────────────────────────────────────────────────────
ax0.scatter([throat_x_ps[idx_0]*M2MM], [throat_y_ps[idx_0]*M2MM],
            marker="D", s=70, color="k", zorder=6, label="Throat (nominal)")
for idx, col in [(idx_m10,"#d62728"),(idx_p10,"#2ca02c")]:
    ax0.scatter([throat_x_ps[idx]*M2MM], [throat_y_ps[idx]*M2MM],
                marker="x", s=90, color=col, zorder=6, linewidths=2)

ax0.set_aspect("equal"); ax0.grid(True, ls="--", alpha=0.3)
ax0.set_xlabel("x [mm]"); ax0.set_ylabel("y [mm]")
ax0.legend(fontsize=8, loc="upper left")

# ── (B) throat height vs angle ────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])
ax1.set_title("Throat height vs pivot angle", fontsize=11)
ax1.plot(angles_deg, throat_norm_ps, "steelblue", lw=2, label="PS pivot (clamshell)")
ax1.plot(angles_deg, throat_norm_ss, "coral",     lw=2, ls="--", label="SS pivot")
ax1.axvline(0, color="k", lw=0.8, ls=":")
ax1.axhline(1, color="k", lw=0.5, ls=":")
ax1.set_xlabel("Pivot angle [deg]"); ax1.set_ylabel("o / o_design  [—]")
ax1.legend(fontsize=9); ax1.grid(True, ls="--", alpha=0.3)

# ── (C) throat x-location vs angle ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("Throat x-location vs pivot angle", fontsize=11)
ax2.plot(angles_deg, throat_x_ps*M2MM, "steelblue", lw=2, label="PS pivot")
ax2.plot(angles_deg, throat_x_ss*M2MM, "coral",     lw=2, ls="--", label="SS pivot")
ax2.axvline(0, color="k", lw=0.8, ls=":")
ax2.set_xlabel("Pivot angle [deg]"); ax2.set_ylabel("x_throat [mm]")
ax2.legend(fontsize=9); ax2.grid(True, ls="--", alpha=0.3)

# ── (D) throat y-location vs angle ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title("Throat y-location vs pivot angle", fontsize=11)
ax3.plot(angles_deg, throat_y_ps*M2MM, "steelblue", lw=2, label="PS pivot")
ax3.plot(angles_deg, throat_y_ss*M2MM, "coral",     lw=2, ls="--", label="SS pivot")
ax3.axvline(0, color="k", lw=0.8, ls=":")
ax3.set_xlabel("Pivot angle [deg]"); ax3.set_ylabel("y_throat [mm]")
ax3.legend(fontsize=9); ax3.grid(True, ls="--", alpha=0.3)

# ── (E) dA/dtheta — authority ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title("Throat height sensitivity dο/dθ", fontsize=11)
dh_ps = np.gradient(throat_norm_ps, angles_deg)
dh_ss = np.gradient(throat_norm_ss, angles_deg)
ax4.plot(angles_deg, dh_ps, "steelblue", lw=2, label="PS pivot")
ax4.plot(angles_deg, dh_ss, "coral",     lw=2, ls="--", label="SS pivot")
ax4.axvline(0, color="k", lw=0.8, ls=":")
ax4.axhline(0, color="k", lw=0.5)
ax4.set_xlabel("Pivot angle [deg]"); ax4.set_ylabel("d(o/o_design)/dθ  [1/deg]")
ax4.legend(fontsize=9); ax4.grid(True, ls="--", alpha=0.3)

plt.suptitle("NGV Clamshell: Pressure-side vs Suction-side Pivot Comparison", fontsize=13, y=1.01)

plt.show()
print("Saved: ngv_clamshell_analysis.png")