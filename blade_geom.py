# geometry.py
import numpy as np
import pint
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


# ==========================================================
# Unit helpers
# ==========================================================
def meter(x: Q_) -> float:
    return x.to("m").magnitude

def rad(a: Q_) -> float:
    return a.to("rad").magnitude


# ==========================================================
# Basic polyline utilities
# ==========================================================
def dedup_consecutive(x, y, tol=1e-12):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = [0]
    for i in range(1, len(x)):
        if (abs(x[i]-x[keep[-1]]) > tol) or (abs(y[i]-y[keep[-1]]) > tol):
            keep.append(i)
    return x[keep], y[keep]

def split_profile_le_te(x, y, tol=1e-12):
    """
    Split closed profile (x,y) into two open curves from LE(min x) to TE(max x).
    Returns:
      (x_ss,y_ss), (x_ps,y_ps), info
    SS/PS classification uses mean(y) (your convention).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # ensure closed
    if (abs(x[0]-x[-1]) > tol) or (abs(y[0]-y[-1]) > tol):
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]

    xmin = np.min(x)
    xmax = np.max(x)

    le_candidates = np.where(np.isclose(x, xmin, atol=tol, rtol=0.0))[0]
    te_candidates = np.where(np.isclose(x, xmax, atol=tol, rtol=0.0))[0]

    ymid = 0.5*(np.max(y) + np.min(y))
    i_le = le_candidates[np.argmin(np.abs(y[le_candidates] - ymid))]
    i_te = te_candidates[np.argmin(np.abs(y[te_candidates] - ymid))]

    n = len(x)

    def forward_indices(i0, i1):
        if i0 <= i1:
            return np.arange(i0, i1+1)
        else:
            return np.r_[np.arange(i0, n), np.arange(0, i1+1)]

    idx_A = forward_indices(i_le, i_te)   # LE->TE forward
    idx_B = forward_indices(i_te, i_le)   # TE->LE forward
    idx_B = idx_B[::-1]                   # reverse => LE->TE

    xA, yA = x[idx_A], y[idx_A]
    xB, yB = x[idx_B], y[idx_B]

    # suction side tends to have higher y in your plotting convention
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


# ==========================================================
# Blade profile builder (your exact method, just packaged)
# ==========================================================
def build_pritchard_profile(params, n=25):
    """
    Returns:
      x_profile, y_profile (closed polyline, meters)
      pitch (meters)
      extra dict with intermediate arrays if you want debugging
    """
    # ---- unpack params (Quantities) ----
    LE_radius   = params["LE_radius"]
    TE_radius   = params["TE_radius"]
    Chord_ax    = params["Chord_ax"]
    zeta_ung    = params["zeta_ung"]
    beta_in     = params["beta_in"]
    beta_out    = params["beta_out"]
    inlet_wedge = params["inlet_wedge"]
    exit_wedge  = params["exit_wedge"]
    N_blades    = params["N_blades"]
    throat      = params["throat"]
    Radius      = params["Radius"]

    # ---- strip units ----
    LE  = meter(LE_radius)
    TE  = meter(TE_radius)
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
    # Points 1..5 (your construction)
    # -------------------------
    beta_1 = bout - ew
    x_1 = Cax - TE * (1 + np.sin(beta_1))
    y_1 = TE * np.cos(beta_1)

    beta_2 = bout - ew + zeta
    x_2 = Cax - TE + (t + TE) * np.sin(beta_2)
    y_2 = pitch - (t + TE) * np.cos(beta_2)

    beta_3 = bin_ + iw
    x_3 = LE * (1 - np.sin(beta_3))
    y_3 = y_2 + ((x_2 - x_3)/(beta_2 - beta_3)) * np.log(np.cos(beta_2)/np.cos(beta_3))

    Chord_tan = y_3 - LE * np.cos(beta_3)
    stagger = np.arctan(Chord_tan / Cax)

    beta_4 = bin_ - iw
    x_4 = LE * (1 + np.sin(beta_4))
    y_4 = Chord_tan - LE * np.cos(beta_4)

    beta_5 = bout + ew
    x_5 = Cax - TE * (1 - np.sin(beta_5))
    y_5 = -TE * np.cos(beta_5)

    # -------------------------
    # Curve 2 -> 3
    # -------------------------
    x23 = np.linspace(x_2, x_3, n)
    beta_x = (beta_2 - beta_3)/(x_2 - x_3) * (x23 - x_3) + beta_3
    y23 = y_2 + ((x_2 - x_3)/(beta_2 - beta_3)) * np.log(np.cos(beta_2)/np.cos(beta_x))

    # -------------------------
    # Curve 4 -> 5 (cubic)
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
    # Curve 1 -> 2 (quadratic via value+value+slope at 2)
    # -------------------------
    m2 = np.tan(beta_2)
    A = np.array([
        [x_1**2, x_1, 1.0],
        [x_2**2, x_2, 1.0],
        [2.0*x_2, 1.0, 0.0],
    ], dtype=float)
    rhs = np.array([y_1, y_2, m2], dtype=float)
    a12, b12, c12 = np.linalg.solve(A, rhs)

    x12 = np.linspace(x_1, x_2, n)
    y12 = a12*x12**2 + b12*x12 + c12

    # -------------------------
    # LE circle arc 3 -> 4
    # -------------------------
    x0_le = x_3 + LE * np.sin(beta_3)
    y0_le = y_3 - LE * np.cos(beta_3)

    ang3 = np.arctan2(y_3 - y0_le, x_3 - x0_le)
    ang4 = np.arctan2(y_4 - y0_le, x_4 - x0_le)

    def angle_linspace_minor(a, b, n):
        d = (b - a) % (2*np.pi)
        if d > np.pi:
            d -= 2*np.pi
        return a + np.linspace(0.0, d, n)

    th_le = angle_linspace_minor(ang3, ang4, n)
    x_le_circ = x0_le + LE * np.cos(th_le)
    y_le_circ = y0_le + LE * np.sin(th_le)

    # -------------------------
    # TE arc 5 -> 1
    # If TE == 0: make it a single point at x_5,y_5 then x_1,y_1 (both should collapse)
    # -------------------------
    if TE > 0.0:
        x0_te = x_1 + TE * np.sin(beta_1)
        y0_te = y_1 - TE * np.cos(beta_1)

        ang5 = np.arctan2(y_5 - y0_te, x_5 - x0_te)
        ang1 = np.arctan2(y_1 - y0_te, x_1 - x0_te)

        th_te = angle_linspace_minor(ang5, ang1, n)
        x_te_circ = x0_te + TE * np.cos(th_te)
        y_te_circ = y0_te + TE * np.sin(th_te)
    else:
        x_te_circ = np.array([x_5, x_1], dtype=float)
        y_te_circ = np.array([y_5, y_1], dtype=float)

    # -------------------------
    # Concatenate closed profile (your ordering)
    # -------------------------
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

    # Ensure closure not strictly required but nice
    x_profile, y_profile = dedup_consecutive(x_profile, y_profile, tol=1e-12)

    extra = dict(
        pitch=pitch, Cax=Cax,
        x_le_circ=x_le_circ, y_le_circ=y_le_circ,
        x_te_circ=x_te_circ, y_te_circ=y_te_circ,
        points=dict(x1=x_1,y1=y_1,x2=x_2,y2=y_2,x3=x_3,y3=y_3,x4=x_4,y4=y_4,x5=x_5,y5=y_5),
        stagger_angle = stagger
    )
    return x_profile, y_profile, pitch, extra


# ==========================================================
# Hybrid passage boundary builder (the "perfect" version we agreed on)
# ==========================================================
def build_hybrid_passage_boundary(x_profile, y_profile, pitch, Cax, n=25, tol=1e-12):
    """
    Builds the AE523-style hybrid boundary:
      - periodic straight segments in inlet/outlet (top/bottom)
      - blade walls between LE and TE (SS and PS+pitch)
      - inlet/outlet planes
      - join segments at top (LE and TE) as needed
    Returns:
      segments dict: name -> (x,y)
      walls dict: SS, PS (LE->TE, in physical position)
      meta dict
    """
    (x_ss, y_ss), (x_ps, y_ps), info = split_profile_le_te(x_profile, y_profile, tol=tol)
    x_ss, y_ss = dedup_consecutive(x_ss, y_ss, tol=tol)
    x_ps, y_ps = dedup_consecutive(x_ps, y_ps, tol=tol)

    x_le = x_ss[0]
    x_te = x_ss[-1]

    # blade endpoints in "passage coordinates" (PS shifted by +pitch)
    LE_ss = (x_le, y_ss[0])
    LE_ps = (x_le, y_ps[0] + pitch)
    TE_ss = (x_te, y_ss[-1])
    TE_ps = (x_te, y_ps[-1] + pitch)

    # inlet/outlet planes
    x_inlet_plane  = x_le - 2.0 * Cax
    x_outlet_plane = x_te + 3.0 * Cax

    # define TRUE periodic y-levels by anchoring bottom to SS endpoints
    y_in_bot  = LE_ss[1]
    y_in_top  = y_in_bot + pitch

    y_out_bot = TE_ss[1]
    y_out_top = y_out_bot + pitch

    nseg = n

    # segments (ordered for looping later, but stored by name)
    INLET_x = np.full(nseg, x_inlet_plane)
    INLET_y = np.linspace(y_in_bot, y_in_top, nseg)

    PER_IN_TOP_x = np.linspace(x_inlet_plane, x_le, nseg)
    PER_IN_TOP_y = np.full(nseg, y_in_top)

    PER_IN_BOT_x = np.linspace(x_le, x_inlet_plane, nseg)
    PER_IN_BOT_y = np.full(nseg, y_in_bot)

    LE_JOIN_TOP_x = np.full(nseg, x_le)
    LE_JOIN_TOP_y = np.linspace(y_in_top, LE_ps[1], nseg)

    WALL_PS_x = x_ps
    WALL_PS_y = y_ps + pitch

    TE_JOIN_TOP_x = np.full(nseg, x_te)
    TE_JOIN_TOP_y = np.linspace(TE_ps[1], y_out_top, nseg)

    PER_OUT_TOP_x = np.linspace(x_te, x_outlet_plane, nseg)
    PER_OUT_TOP_y = np.full(nseg, y_out_top)

    OUTLET_x = np.full(nseg, x_outlet_plane)
    OUTLET_y = np.linspace(y_out_top, y_out_bot, nseg)

    PER_OUT_BOT_x = np.linspace(x_outlet_plane, x_te, nseg)
    PER_OUT_BOT_y = np.full(nseg, y_out_bot)

    WALL_SS_x = x_ss[::-1]
    WALL_SS_y = y_ss[::-1]

    # de-dup each segment
    def dd(x, y): return dedup_consecutive(x, y, tol=tol)

    segments = {
        "INLET":       dd(INLET_x, INLET_y),
        "PER_IN_TOP":  dd(PER_IN_TOP_x, PER_IN_TOP_y),
        "LE_JOIN_TOP": dd(LE_JOIN_TOP_x, LE_JOIN_TOP_y),
        "WALL_PS":     dd(WALL_PS_x, WALL_PS_y),
        "TE_JOIN_TOP": dd(TE_JOIN_TOP_x, TE_JOIN_TOP_y),
        "PER_OUT_TOP": dd(PER_OUT_TOP_x, PER_OUT_TOP_y),
        "OUTLET":      dd(OUTLET_x, OUTLET_y),
        "PER_OUT_BOT": dd(PER_OUT_BOT_x, PER_OUT_BOT_y),
        "WALL_SS":     dd(WALL_SS_x, WALL_SS_y),
        "PER_IN_BOT":  dd(PER_IN_BOT_x, PER_IN_BOT_y),
    }

    walls = {
        "SS": (x_ss, y_ss),               # LE->TE
        "PS": (x_ps, y_ps + pitch),       # LE->TE, shifted into passage
    }

    meta = dict(
        pitch=pitch,
        Cax=Cax,
        x_le=x_le, x_te=x_te,
        x_inlet_plane=x_inlet_plane,
        x_outlet_plane=x_outlet_plane,
        y_in_bot=y_in_bot, y_in_top=y_in_top,
        y_out_bot=y_out_bot, y_out_top=y_out_top,
        endpoints=dict(LE_ss=LE_ss, LE_ps=LE_ps, TE_ss=TE_ss, TE_ps=TE_ps),
        split_info=info,
    )

    # quick diagnostics you used before
    meta["diagnostics"] = dict(
        LE_x_mismatch=float(abs(x_ss[0] - x_ps[0])),
        TE_x_mismatch=float(abs(x_ss[-1] - x_ps[-1])),
        SS_min_dx=float(np.min(np.diff(x_ss))),
        PS_min_dx=float(np.min(np.diff(x_ps))),
        inlet_periodic_gap=float(y_in_top - y_in_bot),
        outlet_periodic_gap=float(y_out_top - y_out_bot),
    )

    return segments, walls, meta


# ==========================================================
# Preview plot
# ==========================================================
def plot_hybrid_geometry(segments, walls, meta, title="Hybrid cascade geometry", show=True):
    fig, ax = plt.subplots(figsize=(10, 4))

    # plot segments
    for name, (x, y) in segments.items():
        ax.plot(x, y, lw=2, label=name)

    # plot walls prominently
    xss, yss = walls["SS"]
    xps, yps = walls["PS"]
    ax.plot(xss, yss, lw=3, label="SS (LE->TE)")
    ax.plot(xps, yps, lw=3, label="PS+pitch (LE->TE)")

    # endpoints
    ep = meta["endpoints"]
    ax.scatter(
        [ep["LE_ss"][0], ep["LE_ps"][0], ep["TE_ss"][0], ep["TE_ps"][0]],
        [ep["LE_ss"][1], ep["LE_ps"][1], ep["TE_ss"][1], ep["TE_ps"][1]],
        s=40, zorder=5, label="LE/TE endpoints"
    )

    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax

# ==========================================================
# Quick demo: plot 2 blades separated by pitch
# ==========================================================
def plot_two_blades(x_profile, y_profile, pitch, title="Two blades (1 pitch apart)", show=True):
    fig, ax = plt.subplots(figsize=(8, 3.5))

    # Blade 0
    ax.plot(x_profile, y_profile, lw=2, label="Blade 0")

    # Blade +1 pitch (same shape shifted in y)
    ax.plot(x_profile, y_profile + pitch, lw=2, label="Blade +1 pitch")

    # Helpful reference lines
    ax.axhline(np.min(y_profile), ls="--", alpha=0.4)
    ax.axhline(np.min(y_profile) + pitch, ls="--", alpha=0.4)

    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    # Example parameters (edit to match your case)
    params = dict(
        LE_radius   = Q_(0.05, "in"),
        TE_radius   = Q_(0.025, "in"),
        Chord_ax    = Q_(1.5,   "in"),
        zeta_ung    = Q_(6.5,   "deg"),
        beta_in     = Q_(0,   "deg"),
        beta_out    = -Q_(70,   "deg"),
        inlet_wedge = Q_(15.0,   "deg"),
        exit_wedge  = Q_(3,    "deg"),
        N_blades    = 60,
        throat      = Q_(0.35,  "in"),
        Radius      = Q_(10,  "in"),
    )

    x_prof, y_prof, pitch, extra = build_pritchard_profile(params, n=60)
    if x_prof[0] != x_prof[-1] or y_prof[0] != y_prof[-1]:
        x = np.append(x_prof, x_prof[0])
        y = np.append(y_prof, y_prof[0])
    else:
        x = x_prof.copy()
        y = y_prof.copy()
    A = 0.0
    Cx = 0.0
    Cy = 0.0

    for i in range(len(x)-1):
        cross = x[i]*y[i+1] - x[i+1]*y[i]

        A += cross
        Cx += (x[i] + x[i+1]) * cross
        Cy += (y[i] + y[i+1]) * cross

    A *= 0.5
    Cx /= (6*A)
    Cy /= (6*A)
    Ix = 0.0
    Iy = 0.0
    Ixy = 0.0

    for i in range(len(x)-1):
        cross = x[i]*y[i+1] - x[i+1]*y[i]

        Ix  += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2) * cross
        Iy  += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2) * cross
        Ixy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i]) * cross

    Ix  /= 12
    Iy  /= 12
    Ixy /= 24
    
    Ix_c = Ix - A*Cy**2
    Iy_c = Iy - A*Cx**2
    Ixy_c = Ixy - A*Cx*Cy
    
    # Plot just 2 blades, pitch apart
    print("Stagger angle (degrees):", np.degrees(extra["stagger_angle"]))
    print("Area:", A)
    print("Centroid:", Cx, Cy)
    print("Ix:", Ix_c)
    print("Iy:", Iy_c)
    print("Ixy:", Ixy_c)
    print("x:", max(abs(x_prof - Cx)))
    print("y:", max(abs(y_prof - Cy)))
    plot_two_blades(x_prof, y_prof, pitch)
    

    # (Optional) also show your hybrid passage boundary, if you want
    # Cax = extra["Cax"]
    # segments, walls, meta = build_hybrid_passage_boundary(x_prof, y_prof, pitch, Cax, n=40)
    # plot_hybrid_geometry(segments, walls, meta, title="Hybrid geometry (sanity check)")