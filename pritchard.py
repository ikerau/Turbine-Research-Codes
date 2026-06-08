import numpy as np
import gmsh
import math
import pint
import matplotlib.pyplot as plt
import os
import gmsh
import math

gmsh.initialize()
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

def clear_terminal():
    if os.name == 'nt':
        _ = os.system('cls')
clear_terminal()

# -------------------------
# Inputs
# -------------------------
LE_radius = Q_(0.02,'inch')
TE_radius = Q_(0.00000,'inch')
Chord_ax  = Q_(0.5,'inch')
zeta_ung  = Q_(6.5, 'deg') # guess
beta_in   = Q_(35, 'deg')
beta_out  = -Q_(57, 'deg')
inlet_wedge = Q_(9, 'deg')
exit_wedge  = 0.5 * zeta_ung
N_blades  = 35
throat    = Q_(0.175,'inch')
Radius    = Q_(2.08661,'inch')

# -------------------------
# Helpers: strip units
# -------------------------
def meter(x):  # pint quantity -> float inches
    return x.to('m').magnitude

def rad(a):   # pint quantity -> float radians
    return a.to('rad').magnitude

LE = meter(LE_radius)
TE = meter(TE_radius)
Cax = meter(Chord_ax)
zeta = rad(zeta_ung)
bin_ = rad(beta_in)
bout = rad(beta_out)
iw = rad(inlet_wedge)
ew = rad(exit_wedge)
t = meter(throat)
R = meter(Radius)

pitch = (2*np.pi*R)/N_blades

# -------------------------
# Point 1
# -------------------------
beta_1 = bout - ew
x_1 = Cax - TE * (1 + np.sin(beta_1))
y_1 = TE * np.cos(beta_1)

# -------------------------
# Point 2
# -------------------------
beta_2 = bout - ew + zeta
x_2 = Cax - TE + (t + TE) * np.sin(beta_2)
y_2 = pitch - (t + TE) * np.cos(beta_2)

# -------------------------
# Point 3 (your "dbeta/dx const" + log-cos relation)
# -------------------------
beta_3 = bin_ + iw
x_3 = LE * (1 - np.sin(beta_3))

# Same expression you wrote, but with all angles in radians (NO 180/pi factor)
y_3 = y_2 + ((x_2 - x_3)/(beta_2 - beta_3)) * np.log(np.cos(beta_2)/np.cos(beta_3))
Chord_tan = y_3 - LE * np.cos(beta_3)
stagger = np.arctan(Chord_tan/Cax)

# -------------------------
# Point 4
# -------------------------
beta_4 = bin_ - iw
x_4 = LE * (1 + np.sin(beta_4))
y_4 = Chord_tan - LE * np.cos(beta_4)

# -------------------------
# Point 5  (likely TE_radius here, not LE_radius)
# -------------------------
beta_5 = bout + ew
x_5 = Cax - TE * (1 - np.sin(beta_5))
y_5 = -TE * np.cos(beta_5)

# -------------------------
# Curve / line 2 -> 3 (beta varies linearly with x)
# -------------------------
n = 20
x23 = np.linspace(x_2, x_3, n)

# beta(x) linear between (x3,beta3) and (x2,beta2)
beta_x = (beta_2 - beta_3)/(x_2 - x_3) * (x23 - x_3) + beta_3

# y(x) from your log-cos relationship (consistent radians)
y23 = y_2 + ((x_2 - x_3)/(beta_2 - beta_3)) * np.log(np.cos(beta_2)/np.cos(beta_x))

beta_prime = (beta_2 - beta_3)/(x_2 - x_3)      # rad/m
ypp_23_at2 = (1/np.cos(beta_2)**2) * beta_prime # 1/m

# -------------------------
# Curve 4 - 5
# -------------------------
x45 = np.linspace(x_4, x_5, n)
d45 = ((np.tan(beta_4) + np.tan(beta_5))/((x_4 - x_5)**2)
       - (2*(y_4 - y_5))/((x_4 - x_5)**3))

c45 = ((y_4 - y_5)/((x_4 - x_5)**2)
       - (np.tan(beta_5))/(x_4 - x_5)
       - d45 * (x_4 + 2 * x_5))

b45 = np.tan(beta_5) - 2*c45*x_5 - 3*d45*x_5**2
a45 = y_5 - b45*x_5 - c45*x_5**2 - d45*x_5**3

y45 = a45 + b45*x45 + c45*x45**2 + d45*x45**3

# -------------------------
# Curve 1 - 2 (cubic, same style as 4-5)
# -------------------------
m2 = np.tan(beta_2)  # beta_2 is radians float

A = np.array([
    [x_1**2, x_1, 1.0],
    [x_2**2, x_2, 1.0],
    [2.0*x_2, 1.0, 0.0],
], dtype=float)

rhs = np.array([y_1, y_2, m2], dtype=float)

a12, b12, c12 = np.linalg.solve(A, rhs)

x12 = np.linspace(x_1, x_2, n)
y12 = a12*x12**2 + b12*x12 + c12

ypp_12_at2 = 2.0*a12

# -------------------------
# Curve LE
# -------------------------

x0_le = x_3 + LE * np.sin(beta_3)
y0_le = y_3 - LE * np.cos(beta_3)

ang3 = np.arctan2(y_3 - y0_le, x_3 - x0_le)   # angle of point 3 about center
ang4 = np.arctan2(y_4 - y0_le, x_4 - x0_le)   # angle of point 4 about center

def angle_linspace_minor(a, b, n):
    d = (b - a) % (2*np.pi)
    if d > np.pi:
        d -= 2*np.pi
    return a + np.linspace(0.0, d, n)

th = angle_linspace_minor(ang3, ang4, n)

x_le_circ = x0_le + LE * np.cos(th)
y_le_circ = y0_le + LE * np.sin(th)

# -------------------------
# Curve LE
# -------------------------

# TE center (your normal-offset definition based on point 1 and beta_1)
x0_te = x_1 + TE * np.sin(beta_1)
y0_te = y_1 - TE * np.cos(beta_1)

ang5 = np.arctan2(y_5 - y0_te, x_5 - x0_te)
ang1 = np.arctan2(y_1 - y0_te, x_1 - x0_te)

# For TE, you may want the short arc or the long arc depending on your geometry.
# Start with "short". If it draws the wrong side, switch to prefer="long".
th_te = angle_linspace_minor(ang5, ang1, n)

x_te_circ = x0_te + TE * np.cos(th_te)
y_te_circ = y0_te + TE * np.sin(th_te)


# Ensure your segment arrays are oriented as:
# x12: x_1 -> x_2
# x23: x_2 -> x_3
# x45: x_4 -> x_5
# LE arc: point 3 -> point 4
# TE arc: point 5 -> point 1

x_profile = np.concatenate([
    x12,
    x23[1:],
    x_le_circ[1:],
    x45[1:],
    x_te_circ[1:]
])

y_profile = np.concatenate([
    y12,
    y23[1:],
    y_le_circ[1:],
    y45[1:],
    y_te_circ[1:]
])

def to_in(x_m):
    # x_m can be a float, list, or numpy array of meter magnitudes
    return (Q_(np.asarray(x_m), 'meter')).to('inch').magnitude

# -------------------------
# Plot: Two blades with SS/PS highlighted
# -------------------------

import numpy as np

def split_profile_le_te(x, y, prefer_te_unique=True, tol=1e-12):
    """
    Split closed profile (x,y) into two open curves from LE(min x) to TE(max x).

    Returns:
        (x_ss, y_ss), (x_ps, y_ps), info dict with indices and points
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)

    # 0) Ensure closed (optional)
    if (abs(x[0]-x[-1]) > tol) or (abs(y[0]-y[-1]) > tol):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    # 1) Handle multiple points with same min/max x (common with TE/LE arcs)
    # Choose a representative by picking extreme y among candidates
    xmin = np.min(x)
    xmax = np.max(x)

    le_candidates = np.where(np.isclose(x, xmin, atol=tol, rtol=0.0))[0]
    te_candidates = np.where(np.isclose(x, xmax, atol=tol, rtol=0.0))[0]

    # Pick LE as candidate with y closest to mid (stable), TE similarly
    ymid = 0.5*(np.max(y) + np.min(y))
    i_le = le_candidates[np.argmin(np.abs(y[le_candidates] - ymid))]

    # TE: choose mid-y too (avoid picking top/bottom of TE circle)
    i_te = te_candidates[np.argmin(np.abs(y[te_candidates] - ymid))]

    # 2) Build forward path from i_le to i_te (wrapping)
    n = len(x)
    def forward_indices(i0, i1):
        if i0 <= i1:
            return np.arange(i0, i1+1)
        else:
            return np.r_[np.arange(i0, n), np.arange(0, i1+1)]

    idx_A = forward_indices(i_le, i_te)          # LE -> TE going "forward"
    idx_B = forward_indices(i_te, i_le)          # TE -> LE going "forward"
    idx_B = idx_B[::-1]                          # reverse to get LE -> TE

    xA, yA = x[idx_A], y[idx_A]
    xB, yB = x[idx_B], y[idx_B]

    # 3) Decide which one is suction vs pressure
    # Simple rule: suction side tends to have higher y on average in your coordinates
    if np.mean(yA) >= np.mean(yB):
        (x_ss, y_ss), (x_ps, y_ps) = (xA, yA), (xB, yB)
        ss_from = "A"
    else:
        (x_ss, y_ss), (x_ps, y_ps) = (xB, yB), (xA, yA)
        ss_from = "B"

    info = dict(
        i_le=int(i_le), i_te=int(i_te),
        le_point=(float(x[i_le]), float(y[i_le])),
        te_point=(float(x[i_te]), float(y[i_te])),
        ss_path=ss_from,
        n_total=int(n)
    )
    return (x_ss, y_ss), (x_ps, y_ps), info

def dedup_consecutive(x, y, tol=1e-12):
    x = np.asarray(x); y = np.asarray(y)
    keep = [0]
    for i in range(1, len(x)):
        if (abs(x[i]-x[keep[-1]]) > tol) or (abs(y[i]-y[keep[-1]]) > tol):
            keep.append(i)
    return x[keep], y[keep]

# ==========================================================
# Build AE523-style hybrid boundary (NO Gmsh yet)
# Periodic straight segments in inlet/outlet, blade walls in middle
# ==========================================================

# 1) Split blade into SS/PS curves (LE->TE)
(x_ss, y_ss), (x_ps, y_ps), info = split_profile_le_te(x_profile, y_profile)

x_ss, y_ss = dedup_consecutive(x_ss, y_ss, tol=1e-12)
x_ps, y_ps = dedup_consecutive(x_ps, y_ps, tol=1e-12)

# 2) Key x-locations
x_le = x_ss[0]
x_te = x_ss[-1]

L_in  = 1.0 * Cax
L_out = 1.0 * Cax
x_inlet_plane  = x_le - L_in
x_outlet_plane = x_te + L_out

# 3) Blade endpoints we want to connect to
LE_ss = (x_le, y_ss[0])
LE_ps = (x_le, y_ps[0] + pitch)

TE_ss = (x_te, y_ss[-1])
TE_ps = (x_te, y_ps[-1] + pitch)

# 4) Define TRUE periodic straight-line levels in inlet/outlet
#    (Bottom line is anchored to SS endpoint; top is bottom + pitch)
y_in_bot  = LE_ss[1]
y_in_top  = y_in_bot + pitch

y_out_bot = TE_ss[1]
y_out_top = y_out_bot + pitch

# 5) Resolution for straight segments/joins
nseg = n

# -------------------------
# INLET BLOCK BOUNDARY PIECES
# -------------------------

# Inlet plane (bottom->top)
INLET_x = np.full(nseg, x_inlet_plane)
INLET_y = np.linspace(y_in_bot, y_in_top, nseg)

# Inlet periodic top (left->right)
PER_IN_TOP_x = np.linspace(x_inlet_plane, x_le, nseg)
PER_IN_TOP_y = np.full(nseg, y_in_top)

# Inlet periodic bottom (right->left for loop continuity later)
PER_IN_BOT_x = np.linspace(x_le, x_inlet_plane, nseg)
PER_IN_BOT_y = np.full(nseg, y_in_bot)

# Join at LE on top: from periodic top down/up to the PS(+pitch) LE point
# If LE_ps happens to lie exactly at y_in_top, this is zero-length (fine).
LE_JOIN_TOP_x = np.full(nseg, x_le)
LE_JOIN_TOP_y = np.linspace(y_in_top, LE_ps[1], nseg)

# (Bottom at LE already meets SS point by construction; no LE_JOIN_BOT needed)

# -------------------------
# BLADE PASSAGE WALLS (middle)
# -------------------------

# Upper wall: PS shifted by pitch (LE->TE)
WALL_PS_x = x_ps
WALL_PS_y = y_ps + pitch

# Lower wall: SS (TE->LE for loop closure later)
WALL_SS_x = x_ss[::-1]
WALL_SS_y = y_ss[::-1]

# -------------------------
# OUTLET BLOCK BOUNDARY PIECES
# -------------------------

# Join at TE on top: from PS(+pitch) TE point up/down to outlet periodic top
TE_JOIN_TOP_x = np.full(nseg, x_te)
TE_JOIN_TOP_y = np.linspace(TE_ps[1], y_out_top, nseg)

# Outlet periodic top (left->right)
PER_OUT_TOP_x = np.linspace(x_te, x_outlet_plane, nseg)
PER_OUT_TOP_y = np.full(nseg, y_out_top)

# Outlet plane (top->bottom for loop continuity)
OUTLET_x = np.full(nseg, x_outlet_plane)
OUTLET_y = np.linspace(y_out_top, y_out_bot, nseg)

# Outlet periodic bottom (right->left for loop continuity)
PER_OUT_BOT_x = np.linspace(x_outlet_plane, x_te, nseg)
PER_OUT_BOT_y = np.full(nseg, y_out_bot)

# (Bottom at TE already meets SS TE point by construction; no TE_JOIN_BOT needed)

# ==========================================================
# Assemble one closed outer loop (for plotting / sanity)
# Order around the boundary:
# inlet plane -> inlet top periodic -> LE top join -> PS wall -> TE top join
# -> outlet top periodic -> outlet plane -> outlet bottom periodic -> SS wall -> inlet bottom periodic
# ==========================================================

loop_x = np.concatenate([
    INLET_x,
    PER_IN_TOP_x[1:],
    LE_JOIN_TOP_x[1:],
    WALL_PS_x[1:],
    TE_JOIN_TOP_x[1:],
    PER_OUT_TOP_x[1:],
    OUTLET_x[1:],
    PER_OUT_BOT_x[1:],
    WALL_SS_x[1:],
    PER_IN_BOT_x[1:],
    [INLET_x[0]]
])

loop_y = np.concatenate([
    INLET_y,
    PER_IN_TOP_y[1:],
    LE_JOIN_TOP_y[1:],
    WALL_PS_y[1:],
    TE_JOIN_TOP_y[1:],
    PER_OUT_TOP_y[1:],
    OUTLET_y[1:],
    PER_OUT_BOT_y[1:],
    WALL_SS_y[1:],
    PER_IN_BOT_y[1:],
    [INLET_y[0]]
])

loop_x, loop_y = dedup_consecutive(loop_x, loop_y, tol=1e-12)

# ==========================================================
# Plot (verify it matches your intended sketch)
# ==========================================================
fig, ax = plt.subplots(figsize=(10,4))

ax.plot(loop_x, loop_y, lw=2, label="Full outer boundary loop")

# Blade walls
ax.plot(x_ss, y_ss, lw=2, label="SS (LE->TE)")
ax.plot(x_ps, y_ps + pitch, lw=2, label="PS + pitch (LE->TE)")

# Periodic segments explicitly
ax.plot(PER_IN_TOP_x,  PER_IN_TOP_y,  lw=2, label="PER_IN_TOP")
ax.plot(PER_IN_BOT_x,  PER_IN_BOT_y,  lw=2, label="PER_IN_BOT")
ax.plot(PER_OUT_TOP_x, PER_OUT_TOP_y, lw=2, label="PER_OUT_TOP")
ax.plot(PER_OUT_BOT_x, PER_OUT_BOT_y, lw=2, label="PER_OUT_BOT")

# Inlet/outlet planes
ax.plot(INLET_x,  INLET_y,  lw=2, label="INLET")
ax.plot(OUTLET_x, OUTLET_y, lw=2, label="OUTLET")

# Joins
ax.plot(LE_JOIN_TOP_x, LE_JOIN_TOP_y, lw=2, label="LE_JOIN_TOP")
ax.plot(TE_JOIN_TOP_x, TE_JOIN_TOP_y, lw=2, label="TE_JOIN_TOP")

# Key points
ax.scatter([LE_ss[0], LE_ps[0], TE_ss[0], TE_ps[0]],
           [LE_ss[1], LE_ps[1], TE_ss[1], TE_ps[1]],
           s=40, zorder=5, label="LE/TE endpoints (SS, PS+pitch)")

ax.set_aspect('equal')
ax.grid(True, ls='--', alpha=0.3)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Hybrid cascade geometry: periodic inlet/outlet + blade walls in middle")
ax.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

# Optional quick checks (print)
print("Inlet periodic gap (should equal pitch):", y_in_top - y_in_bot, " pitch:", pitch)
print("Outlet periodic gap (should equal pitch):", y_out_top - y_out_bot, " pitch:", pitch)
print("LE top join length:", abs(y_in_top - LE_ps[1]))
print("TE top join length:", abs(y_out_top - TE_ps[1]))

# ==========================================================
# Gmsh: build + tag + mesh the hybrid cascade geometry
# ==========================================================

# ---- User knobs ----
mesh_name = "cascade_passage"
msh_file  = f"{mesh_name}.msh"

# base mesh size (tune later; fields are better eventually)
lc = 0.02 * Cax

# snapping tolerance for point reuse (in meters)
snap = 1e-12

# ----------------------------------------------------------
# 0) Init model
# ----------------------------------------------------------
gmsh.initialize()
gmsh.model.add(mesh_name)

# (optional) make meshing more deterministic
gmsh.option.setNumber("General.Terminal", 1)
gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
gmsh.option.setNumber("Mesh.Algorithm", 6)   # 6 = Frontal-Delaunay for 2D (often good)
gmsh.option.setNumber("Mesh.Optimize", 1)

# ----------------------------------------------------------
# 1) Point cache + polyline builder (connected topology)
# ----------------------------------------------------------
_point_cache = {}

def _key(x, y, tol):
    # snap to grid for stable hashing
    sx = float(round(x / tol) * tol)
    sy = float(round(y / tol) * tol)
    return (sx, sy)

def add_point_cached(x, y, lc, tol=snap):
    k = _key(float(x), float(y), tol)
    if k in _point_cache:
        return _point_cache[k]
    tag = gmsh.model.geo.addPoint(k[0], k[1], 0.0, float(lc))
    _point_cache[k] = tag
    return tag

def build_polyline(name, x, y, lc, tol=snap):
    """
    Build a polyline as a list of gmsh line tags.
    Returns (pt_tags, line_tags).
    If the polyline is degenerate (all points same), returns empty lists.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # drop consecutive duplicates to avoid zero-length lines
    xx, yy = dedup_consecutive(x, y, tol=tol)

    if len(xx) < 2:
        return [], []

    pts = [add_point_cached(xx[i], yy[i], lc, tol=tol) for i in range(len(xx))]
    lines = []
    for i in range(len(pts) - 1):
        if pts[i] != pts[i+1]:
            lines.append(gmsh.model.geo.addLine(pts[i], pts[i+1]))

    return pts, lines

# ----------------------------------------------------------
# 2) Build each boundary segment as its own polyline
# ----------------------------------------------------------
pts_INLET,       lines_INLET       = build_polyline("INLET",       INLET_x,       INLET_y,       lc)
pts_PER_IN_TOP,  lines_PER_IN_TOP  = build_polyline("PER_IN_TOP",  PER_IN_TOP_x,  PER_IN_TOP_y,  lc)
pts_LE_JOIN_TOP, lines_LE_JOIN_TOP = build_polyline("LE_JOIN_TOP", LE_JOIN_TOP_x, LE_JOIN_TOP_y, lc)
pts_WALL_PS,     lines_WALL_PS     = build_polyline("WALL_PS",     WALL_PS_x,     WALL_PS_y,     lc)
pts_TE_JOIN_TOP, lines_TE_JOIN_TOP = build_polyline("TE_JOIN_TOP", TE_JOIN_TOP_x, TE_JOIN_TOP_y, lc)
pts_PER_OUT_TOP, lines_PER_OUT_TOP = build_polyline("PER_OUT_TOP", PER_OUT_TOP_x, PER_OUT_TOP_y, lc)
pts_OUTLET,      lines_OUTLET      = build_polyline("OUTLET",      OUTLET_x,      OUTLET_y,      lc)
pts_PER_OUT_BOT, lines_PER_OUT_BOT = build_polyline("PER_OUT_BOT", PER_OUT_BOT_x, PER_OUT_BOT_y, lc)
pts_WALL_SS,     lines_WALL_SS     = build_polyline("WALL_SS",     WALL_SS_x,     WALL_SS_y,     lc)
pts_PER_IN_BOT,  lines_PER_IN_BOT  = build_polyline("PER_IN_BOT",  PER_IN_BOT_x,  PER_IN_BOT_y,  lc)

# ----------------------------------------------------------
# 3) Connectivity sanity checks (topological closure)
#    (These must match the order used in your loop assembly.)
# ----------------------------------------------------------
def p0(pts): return pts[0]
def p1(pts): return pts[-1]

# If any join polyline is empty (zero length), skip it in checks and loop
segments = [
    ("INLET",       pts_INLET,       lines_INLET),
    ("PER_IN_TOP",  pts_PER_IN_TOP,  lines_PER_IN_TOP),
    ("LE_JOIN_TOP", pts_LE_JOIN_TOP, lines_LE_JOIN_TOP),
    ("WALL_PS",     pts_WALL_PS,     lines_WALL_PS),
    ("TE_JOIN_TOP", pts_TE_JOIN_TOP, lines_TE_JOIN_TOP),
    ("PER_OUT_TOP", pts_PER_OUT_TOP, lines_PER_OUT_TOP),
    ("OUTLET",      pts_OUTLET,      lines_OUTLET),
    ("PER_OUT_BOT", pts_PER_OUT_BOT, lines_PER_OUT_BOT),
    ("WALL_SS",     pts_WALL_SS,     lines_WALL_SS),
    ("PER_IN_BOT",  pts_PER_IN_BOT,  lines_PER_IN_BOT),
]

# filter out degenerate segments with no lines
segments = [(nm, pts, lns) for (nm, pts, lns) in segments if len(lns) > 0]

# ensure end of i equals start of i+1, and last ends at first start
for (nmA, ptsA, _), (nmB, ptsB, _) in zip(segments, segments[1:]):
    if p1(ptsA) != p0(ptsB):
        raise RuntimeError(f"Connectivity break: {nmA} end != {nmB} start")

if p1(segments[-1][1]) != p0(segments[0][1]):
    raise RuntimeError("Loop does not close: last end != first start")

# ----------------------------------------------------------
# 4) Curve loop + surface
# ----------------------------------------------------------
all_lines = []
for _, _, lns in segments:
    all_lines += lns

cloop = gmsh.model.geo.addCurveLoop(all_lines)
surf  = gmsh.model.geo.addPlaneSurface([cloop])

gmsh.model.geo.synchronize()

# ----------------------------------------------------------
# 5) Physical groups (BC tags)
# ----------------------------------------------------------
def add_phys(name, dim, tags):
    pg = gmsh.model.addPhysicalGroup(dim, tags)
    gmsh.model.setPhysicalName(dim, pg, name)

# Boundary groups
add_phys("INLET",       1, lines_INLET)
add_phys("OUTLET",      1, lines_OUTLET)
add_phys("WALL_SS",     1, lines_WALL_SS)
add_phys("WALL_PS",     1, lines_WALL_PS)
add_phys("PER_IN_TOP",  1, lines_PER_IN_TOP)
add_phys("PER_IN_BOT",  1, lines_PER_IN_BOT)
add_phys("PER_OUT_TOP", 1, lines_PER_OUT_TOP)
add_phys("PER_OUT_BOT", 1, lines_PER_OUT_BOT)

# Optional: keep join lines tagged for debugging if non-degenerate
if len(lines_LE_JOIN_TOP) > 0:
    add_phys("LE_JOIN_TOP", 1, lines_LE_JOIN_TOP)
if len(lines_TE_JOIN_TOP) > 0:
    add_phys("TE_JOIN_TOP", 1, lines_TE_JOIN_TOP)

# Domain
add_phys("FLUID", 2, [surf])

# ----------------------------------------------------------
# 6) Mesh + write
# ----------------------------------------------------------
gmsh.model.mesh.generate(2)


gmsh.write(msh_file)

# Optional GUI inspect
gmsh.fltk.run()

gmsh.finalize()
print(f"Wrote: {msh_file}")

