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

    if "stagger" in params:
        # Stagger given externally → pin tangential chord, use cubic for SS curve.
        stagger = rad(params["stagger"])
        Chord_tan = Cax * np.tan(stagger)
        y_3 = Chord_tan + LE * np.cos(beta_3)
    else:
        # Default: derive tangential chord from the log-cosine SS curve.
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
    if "stagger" in params:
        # Cubic (same form as PS curve): satisfies both endpoints and slopes.
        # Needed when y_3 is set from a fixed stagger rather than derived from the
        # log-cosine, so the log-cosine endpoint would not match y_3.
        d23 = ((np.tan(beta_2) + np.tan(beta_3)) / (x_2 - x_3)**2
               - 2*(y_2 - y_3) / (x_2 - x_3)**3)
        c23 = ((y_2 - y_3) / (x_2 - x_3)**2
               - np.tan(beta_3) / (x_2 - x_3)
               - d23 * (x_2 + 2*x_3))
        b23 = np.tan(beta_3) - 2*c23*x_3 - 3*d23*x_3**2
        a23 = y_3 - b23*x_3 - c23*x_3**2 - d23*x_3**3
        y23 = a23 + b23*x23 + c23*x23**2 + d23*x23**3
    else:
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
        te_center = (x0_te, y0_te)
    else:
        x_te_circ = np.array([x_5, x_1], dtype=float)
        y_te_circ = np.array([y_5, y_1], dtype=float)
        te_center = (0.5 * (x_5 + x_1), 0.5 * (y_5 + y_1))

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
        le_center=(x0_le, y0_le),
        te_center=te_center,
        x_le_circ=x_le_circ, y_le_circ=y_le_circ,
        x_te_circ=x_te_circ, y_te_circ=y_te_circ,
        points=dict(x1=x_1,y1=y_1,x2=x_2,y2=y_2,x3=x_3,y3=y_3,x4=x_4,y4=y_4,x5=x_5,y5=y_5),
        stagger_angle = stagger,
        # TE arc is concatenated last (x_te_circ[1:], P5→P1); strip this many points
        # from the end of the profile when comparing to reference data that end at
        # Point 1 with a sharp TE (no arc data tabulated).
        n_te_arc_pts  = len(x_te_circ) - 1,
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


# ==========================================================
# Smooth a closed blade profile via periodic cubic spline
# ==========================================================
def smooth_profile(x, y, n_out=400, smoothing=0.0):
    """
    Fit a periodic cubic spline to a closed blade profile and re-sample
    at n_out equally-spaced parameter values.

    Parameters
    ----------
    x, y      : array-like  closed profile coordinates (meters)
    n_out     : int         number of output points
    smoothing : float       spline smoothing factor (0 = interpolating)

    Returns
    -------
    x_sm, y_sm : ndarray  smoothed closed profile (n_out points)
    """
    from scipy.interpolate import splprep, splev

    x, y = np.asarray(x, float), np.asarray(y, float)

    # ensure closure
    if not (np.isclose(x[0], x[-1]) and np.isclose(y[0], y[-1])):
        x, y = np.append(x, x[0]), np.append(y, y[0])

    # strip exact duplicate consecutive points (splprep rejects them)
    ds = np.hypot(np.diff(x), np.diff(y))
    keep = np.r_[True, ds > 1e-14]
    x, y = x[keep], y[keep]

    tck, _ = splprep([x, y], s=smoothing, per=True, k=3)
    u = np.linspace(0, 1, n_out, endpoint=False)
    xs, ys = splev(u, tck)
    return np.array(xs, float), np.array(ys, float)


# ==========================================================
# One-sided RMS distance: candidate -> reference
# ==========================================================
def _profile_rms(x_ref, y_ref, x_cand, y_cand):
    """
    For each point on the candidate profile, find the nearest point on the
    reference profile. Returns the RMS of those minimum distances (meters).
    Both inputs should come from smooth_profile (uniform arc-length spacing).
    """
    xr = np.asarray(x_ref, float)
    yr = np.asarray(y_ref, float)
    xc = np.asarray(x_cand, float)
    yc = np.asarray(y_cand, float)
    # (N_cand, N_ref) squared-distance matrix — fine at ~400 pts each
    d2 = (xc[:, None] - xr[None, :])**2 + (yc[:, None] - yr[None, :])**2
    return float(np.sqrt(np.mean(np.min(d2, axis=1))))


# ==========================================================
# Convert tabulated (x_chord, y_lower, y_upper) blade data
# to chord-local closed profile [meters]
# ==========================================================
def raw_to_chord_local_m(raw):
    """
    Convert a blade coordinate table with columns
      [x_chord_in, y_lower_in, y_upper_in]
    to a closed profile in chord-local coordinates (meters), with the
    LE circle center at the origin.

    y_lower and y_upper are **absolute y-coordinates** in chord-local
    (not offsets from the chord line).  At the LE both equal the LE
    circle radius; at the TE both equal the TE circle radius.  The SS
    (upper surface) has larger y than the PS (lower surface) at every
    chord station.

    Both surfaces are shifted by -y_LE so the LE circle center lands at
    (x=0, y=0), matching the Pritchard chord-local convention.
    """
    raw = np.asarray(raw, float)
    xc  = raw[:, 0]   # chord position [in]
    yL  = raw[:, 1]   # PS absolute y-coordinate [in]  (positive, smaller)
    yU  = raw[:, 2]   # SS absolute y-coordinate [in]  (positive, larger)

    y_le = yL[0]   # = LE_radius; subtract to put LE center at y = 0

    # SS LE->TE, PS TE->LE; skip shared LE and TE endpoints on the PS return
    x_loc = np.concatenate([xc,                    xc[-2:0:-1]])
    y_loc = np.concatenate([yU - y_le,  (yL - y_le)[-2:0:-1]])
    x_loc = np.append(x_loc, x_loc[0])   # close
    y_loc = np.append(y_loc, y_loc[0])

    return x_loc * 0.0254, y_loc * 0.0254   # inches -> meters


# ==========================================================
# Fit metal blade angles to match a reference profile shape
# ==========================================================
def fit_blade_angles(
    x_ref, y_ref,
    fixed_params,
    beta_in_init       =  0.0,
    beta_out_init      = -65.0,
    beta_in_bounds     = (-20.0,  30.0),
    beta_out_bounds    = (-85.0, -40.0),
    inlet_wedge_init   = None,
    inlet_wedge_bounds = (2.0,  35.0),
    exit_wedge_init    = None,
    exit_wedge_bounds  = (1.0,  12.0),
    zeta_ung_init      = None,
    zeta_ung_bounds    = (3.0,  15.0),
    throat_init        = None,
    throat_bounds      = (0.5,  20.0),
    chord_ax_init      = None,
    chord_ax_bounds    = (1.0, 20.0),
    n_profile          = 80,
    tol                = 1e-7,
    verbose            = True,
):
    """
    Find Pritchard metal angles (beta_in, beta_out) whose profile best matches
    a reference blade given in **chord-local** coordinates (x along chord,
    y perpendicular, both in meters, closed polyline).

    The Pritchard profile is built in the axial/tangential frame then
    rotated by -stagger and translated to its LE center so that both
    profiles share the same chord-local origin (LE center at x=0, y=0).
    No spline smoothing is applied — the raw Pritchard polyline points are
    used directly, which is sufficient for Nelder-Mead convergence.

    The converged beta_in reveals built-in incidence: a positive value means
    the metal inlet angle is biased toward SS relative to the design flow angle.

    Parameters
    ----------
    x_ref, y_ref    : array-like  reference closed profile in chord-local
                                  coordinates (meters).  For tabulated blade
                                  data (x_chord, y_lower, y_upper) use
                                  raw_to_chord_local_m() to build this.
    fixed_params       : dict        pint Quantities for all Pritchard params
                                     except beta_in and beta_out:
                                       LE_radius, TE_radius, Chord_ax,
                                       zeta_ung, inlet_wedge, exit_wedge,
                                       N_blades, throat, Radius
    beta_in_init       : float  initial guess (deg, positive toward SS)
    beta_out_init      : float  initial guess (deg, negative toward TE)
    beta_in_bounds     : (lo, hi) search bounds in degrees
    beta_out_bounds    : (lo, hi) search bounds in degrees
    inlet_wedge_init   : float or None  if not None, also optimise inlet_wedge
                                        (overrides fixed_params["inlet_wedge"])
    inlet_wedge_bounds : (lo, hi) search bounds for inlet_wedge in degrees
    n_profile          : int    points per segment in build_pritchard_profile
    tol                : float  optimizer convergence tolerance (meters)
    verbose            : bool   print progress every 20 evaluations

    Returns
    -------
    dict with:
      beta_in_metal   Q_ (deg)   converged inlet metal angle
      beta_out_metal  Q_ (deg)   converged exit metal angle
      rms_mm          float      final RMS distance in chord-local frame (mm)
      x_fit, y_fit               best-fit Pritchard profile, chord-local (m)
      x_ref, y_ref               reference profile as passed in (m)
      pitch           float      pitch (m) at converged angles
      stagger_deg     float      converged Pritchard stagger angle (deg)
      extra           dict       from build_pritchard_profile
      result                     scipy OptimizeResult
    """
    from scipy.optimize import minimize

    x_ref = np.asarray(x_ref, float)
    y_ref = np.asarray(y_ref, float)

    def _pritchard_chord_local(b_in, b_out, iw_free=None, ew_free=None, zu_free=None, t_free=None, c_free=None):
        """Build Pritchard profile in chord-local frame (no smoothing)."""
        params = dict(fixed_params)
        params["beta_in"]  = Q_(b_in,  "deg")
        params["beta_out"] = Q_(b_out, "deg")
        if iw_free is not None:
            params["inlet_wedge"] = Q_(iw_free, "deg")
        if ew_free is not None:
            params["exit_wedge"] = Q_(ew_free, "deg")
        if zu_free is not None:
            params["zeta_ung"] = Q_(zu_free, "deg")
        if t_free is not None:
            params["throat"] = Q_(t_free, "mm")
        if c_free is not None:
            params["Chord_ax"] = Q_(c_free, "cm")
        xp, yp, pitch, extra = build_pritchard_profile(params, n=n_profile)
        le_cx, le_cy = extra["le_center"]
        te_cx, te_cy = extra["te_center"]
        # translate: put LE center at origin
        xp -= le_cx;  yp -= le_cy
        # rotate so that LE→TE chord vector aligns with +x axis
        # chord slopes downward (TE below LE in tangential), so chord_angle < 0
        # rotating by -chord_angle (CCW) brings it horizontal
        chord_angle = np.arctan2(te_cy - le_cy, te_cx - le_cx)
        rot = -chord_angle   # positive: CCW rotation
        c, s = np.cos(rot), np.sin(rot)
        x_cl = c * xp - s * yp
        y_cl = s * xp + c * yp
        # Shift so the LE stagnation tip (leftmost arc point = min x) aligns with
        # x=0, matching the reference table which starts at the LE arc tangency.
        x_cl -= np.min(x_cl)
        return x_cl, y_cl, pitch, extra

    # determine which geometry params are free variables
    fit_iw = inlet_wedge_init is not None
    fit_ew = exit_wedge_init  is not None
    fit_zu = zeta_ung_init    is not None
    fit_t  = throat_init      is not None
    fit_c  = chord_ax_init    is not None

    # index map: p = [b_in, b_out, (iw)?, (ew)?, (zu)?, (t)?, (c)?]
    _idx = 2
    idx_iw = idx_ew = idx_zu = idx_t = idx_c = None
    if fit_iw: idx_iw = _idx; _idx += 1
    if fit_ew: idx_ew = _idx; _idx += 1
    if fit_zu: idx_zu = _idx; _idx += 1
    if fit_t:  idx_t  = _idx; _idx += 1
    if fit_c:  idx_c  = _idx; _idx += 1

    _call = [0]

    def objective(p):
        b_in  = float(np.clip(p[0], *beta_in_bounds))
        b_out = float(np.clip(p[1], *beta_out_bounds))
        iw = float(np.clip(p[idx_iw], *inlet_wedge_bounds)) if fit_iw else None
        ew = float(np.clip(p[idx_ew], *exit_wedge_bounds))  if fit_ew else None
        zu = float(np.clip(p[idx_zu], *zeta_ung_bounds))    if fit_zu else None
        t  = float(np.clip(p[idx_t],  *throat_bounds))      if fit_t  else None
        c  = float(np.clip(p[idx_c],  *chord_ax_bounds))    if fit_c  else None
        try:
            xc, yc, _, _ = _pritchard_chord_local(b_in, b_out, iw, ew, zu, t, c)
        except Exception:
            return 1e6
        # The reference table starts at P3/P4 (LE tangency, no LE arc tabulated)
        # and ends at P1/P5 (TE tangency, no TE arc tabulated).
        # Profile layout: x12(n) + x23[1:](n-1) + x_le_circ[1:](n-1) + x45[1:](n-1) + x_te_circ[1:](n-1)
        # SS body P1→P3: first 2n-1 points; PS body P4→P5: points [3n-3 : 4n-3)
        _n = n_profile
        xc_cmp = np.concatenate([xc[:2*_n-1], xc[3*_n-3:4*_n-3]])
        yc_cmp = np.concatenate([yc[:2*_n-1], yc[3*_n-3:4*_n-3]])
        rms = _profile_rms(x_ref, y_ref, xc_cmp, yc_cmp)
        _call[0] += 1
        if verbose and _call[0] % 20 == 0:
            extras = ""
            if fit_iw: extras += f"  iw={iw:+6.2f}"
            if fit_ew: extras += f"  ew={ew:+6.2f}"
            if fit_zu: extras += f"  zu={zu:+6.2f}"
            if fit_t:  extras += f"  t={t:+6.3f}mm"
            if fit_c:  extras += f"  c={c:+6.3f}cm"
            print(f"  eval {_call[0]:4d}  "
                  f"b_in={b_in:+7.3f} deg  b_out={b_out:+7.3f} deg{extras}  "
                  f"RMS={rms*1e3:.4f} mm")
        return rms

    x0 = [beta_in_init, beta_out_init]
    if fit_iw: x0.append(float(inlet_wedge_init))
    if fit_ew: x0.append(float(exit_wedge_init))
    if fit_zu: x0.append(float(zeta_ung_init))
    if fit_t:  x0.append(float(throat_init))
    if fit_c:  x0.append(float(chord_ax_init))

    result = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options=dict(xatol=tol, fatol=tol * 1e-3, maxiter=5000, disp=False),
    )

    b_in_opt  = float(np.clip(result.x[0], *beta_in_bounds))
    b_out_opt = float(np.clip(result.x[1], *beta_out_bounds))
    iw_opt = float(np.clip(result.x[idx_iw], *inlet_wedge_bounds)) if fit_iw else None
    ew_opt = float(np.clip(result.x[idx_ew], *exit_wedge_bounds))  if fit_ew else None
    zu_opt = float(np.clip(result.x[idx_zu], *zeta_ung_bounds))    if fit_zu else None
    t_opt  = float(np.clip(result.x[idx_t],  *throat_bounds))      if fit_t  else None
    c_opt  = float(np.clip(result.x[idx_c],  *chord_ax_bounds))    if fit_c  else None

    x_fit, y_fit, pitch, extra = _pritchard_chord_local(b_in_opt, b_out_opt, iw_opt, ew_opt, zu_opt, t_opt, c_opt)
    # Same LE+TE arc exclusion as the objective (full profile kept for plotting)
    _n = n_profile
    x_cmp = np.concatenate([x_fit[:2*_n-1], x_fit[3*_n-3:4*_n-3]])
    y_cmp = np.concatenate([y_fit[:2*_n-1], y_fit[3*_n-3:4*_n-3]])
    rms_mm = _profile_rms(x_ref, y_ref, x_cmp, y_cmp) * 1e3
    stagger_deg = float(np.degrees(extra["stagger_angle"]))

    if verbose:
        print(f"\nConverged after {_call[0]} evaluations")
        print(f"  b_in    = {b_in_opt:+.4f} deg  (positive -> built-in incidence toward SS)")
        print(f"  b_out   = {b_out_opt:+.4f} deg")
        if fit_iw:
            print(f"  iw_fit  = {iw_opt:+.4f} deg  (inlet_wedge optimised)")
        if fit_ew:
            print(f"  ew_fit  = {ew_opt:+.4f} deg  (exit_wedge optimised)")
        if fit_zu:
            print(f"  zu_fit  = {zu_opt:+.4f} deg  (zeta_ung optimised)")
        if fit_t:
            print(f"  t_fit   = {t_opt:+.4f} mm   (throat optimised)")
        if fit_c:
            print(f"  c_fit   = {c_opt:+.4f} cm   (Chord_ax optimised)")
        print(f"  stagger = {stagger_deg:.4f} deg")
        print(f"  RMS     = {rms_mm:.4f} mm")

    out = dict(
        beta_in_metal  = Q_(b_in_opt,  "deg"),
        beta_out_metal = Q_(b_out_opt, "deg"),
        rms_mm         = rms_mm,
        x_fit          = x_fit,
        y_fit          = y_fit,
        x_ref          = x_ref,
        y_ref          = y_ref,
        pitch          = pitch,
        stagger_deg    = stagger_deg,
        extra          = extra,
        result         = result,
    )
    if fit_iw:
        out["inlet_wedge_fit"] = Q_(iw_opt, "deg")
    if fit_ew:
        out["exit_wedge_fit"] = Q_(ew_opt, "deg")
    if fit_zu:
        out["zeta_ung_fit"] = Q_(zu_opt, "deg")
    if fit_t:
        out["throat_fit"] = Q_(t_opt, "mm")
    if fit_c:
        out["chord_ax_fit"] = Q_(c_opt, "cm")
    return out


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