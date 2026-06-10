"""
Port of MATLAB blade.m — Method of Characteristics supersonic impulse blade solver.

Builds a symmetric impulse blade passage with:
    - straight inlet/outlet legs at the blade metal angle
    - lower constant-Mach turning arc (radius Rstar_l)
    - upper constant-Mach turning arc (radius Rstar_u)
    - characteristic-traced walls for the expansion/compression
      transitions between the straight legs and the circular arcs

Follows Boxer, Sterrett & Wlodarski (NACA RML52B06, 1952) — the approach
Linhardt & Silvern 1961 cite as the way to realize ψ_R beyond Eq. [13].

Translated verbatim from blade.m; variable names and logic preserved so
outputs can be diffed against the MATLAB.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve


# ═══════════════════════════════════════════════════════════════════════════════
#  ISENTROPIC HELPERS  (1:1 port)
# ═══════════════════════════════════════════════════════════════════════════════
def mstar(M: float, gamma: float) -> float:
    """M* — critical Mach number form."""
    return (((gamma + 1) / 2) * M**2 / (1 + ((gamma - 1) / 2) * M**2)) ** 0.5


def prandtl_meyer(M: float, gamma: float) -> float:
    """ν(M) Prandtl–Meyer function, radians."""
    gm1 = gamma - 1
    gp1 = gamma + 1
    return (np.sqrt(gp1 / gm1)
            * np.arctan(np.sqrt((gm1 / gp1) * (M**2 - 1)))
            - np.arctan(np.sqrt(M**2 - 1)))


def muki(Rstar: float, gamma: float) -> float:
    """Local Mach wave angle μ as a function of R* (arc radius)."""
    gp1 = gamma + 1
    gm1 = gamma - 1
    return -np.arcsin(np.sqrt((gp1 / 2) * Rstar - (gm1 / 2)))


def phiki_l(nu_i: float, nu_l: float, delta_nu: float, k: int) -> float:
    """Flow direction φ at station k (lower arc)."""
    return nu_i - nu_l - (k - 1) * delta_nu


def phiki_u(nu_i: float, nu_u: float, delta_nu: float, k: int) -> float:
    """Flow direction φ at station k (upper arc)."""
    return nu_u - nu_i - (k - 1) * delta_nu


def mach_slope(phi: float, phi1: float, mu: float, mu1: float) -> float:
    """Average Mach-line slope between two stations."""
    return np.tan((phi + phi1) / 2 + (mu + mu1) / 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  MACH TRANSITIONS  (1:1 port)
# ═══════════════════════════════════════════════════════════════════════════════
def lower_mach(Mstar_i: float, gamma: float) -> float:
    """Lower transition Mach (end of the lower arc)."""
    GP1 = gamma + 1
    GM1 = gamma - 1
    G_GP1 = gamma / GP1
    GM1_GP1 = GM1 / GP1
    sqrtGP1_GM1 = np.sqrt(GP1 / GM1)
    GM1_G = GM1 / gamma

    # Target M*_l — matches MATLAB formula verbatim
    Mstar_l = sqrtGP1_GM1 * (
        1 - (1 - GM1_GP1 * Mstar_i**2)
        * (1 + 0.5 * (G_GP1 * Mstar_i**2) / (1 - GM1_GP1 * Mstar_i**2)) ** GM1_G
    ) ** 0.5

    def Mstar_M(M):
        return (((gamma + 1) / 2) * M**2
                / (1 + ((gamma - 1) / 2) * M**2)) ** 0.5

    return float(fsolve(lambda M: Mstar_M(M) - Mstar_l, 1.5, full_output=False)[0])


def upper_mach(Mstar_i: float, gamma: float) -> float:
    """Upper transition Mach (start of the upper arc)."""
    GP1 = gamma + 1
    GM1 = gamma - 1
    G_GP1 = gamma / GP1
    GM1_GP1 = GM1 / GP1
    sqrtGP1_GM1 = np.sqrt(GP1 / GM1)
    GM1_G = GM1 / gamma

    def Mstar_u_func(Mstar_u):
        return sqrtGP1_GM1 * (
            1 - (1 - GM1_GP1 * Mstar_u**2)
            * (1 + 0.5 * (G_GP1 * Mstar_u**2) / (1 - GM1_GP1 * Mstar_u**2)) ** GM1_G
        ) ** 0.5

    Mstar_u = float(fsolve(lambda x: Mstar_u_func(x) - Mstar_i, 1.5, full_output=False)[0])

    def Mstar_M(M):
        return (((gamma + 1) / 2) * M**2
                / (1 + ((gamma - 1) / 2) * M**2)) ** 0.5

    return float(fsolve(lambda M: Mstar_M(M) - Mstar_u, 1.5, full_output=False)[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  R* SOLVERS  (1:1 port)
# ═══════════════════════════════════════════════════════════════════════════════
def solve4Rki_l(gamma: float, k: int, delta_nu: float, nu_i: float,
                Rstar_initial_guess: float) -> float:
    """Solve for R* at station k along the LOWER arc."""
    gp1 = gamma + 1
    gm1 = gamma - 1
    fRstarki = (2 * nu_i
                - (np.pi / 2) * (np.sqrt(gp1 / gm1) - 1)
                - 2 * (k - 1) * delta_nu)

    def residual(Rs):
        return (np.sqrt(gp1 / gm1) * np.arcsin((gm1 / (Rs**2)) - gamma)
                + np.arcsin(gp1 * Rs**2 - gamma)
                - fRstarki)

    return float(fsolve(residual, Rstar_initial_guess, full_output=False)[0])


def solve4Rki_u(gamma: float, k: int, delta_nu: float, nu_i: float,
                Rstar_initial_guess: float) -> float:
    """Solve for R* at station k along the UPPER arc (sign of delta_nu flipped)."""
    gp1 = gamma + 1
    gm1 = gamma - 1
    fRstarki = (2 * nu_i
                - (np.pi / 2) * (np.sqrt(gp1 / gm1) - 1)
                + 2 * (k - 1) * delta_nu)   # +, not −

    def residual(Rs):
        return (np.sqrt(gp1 / gm1) * np.arcsin((gm1 / (Rs**2)) - gamma)
                + np.arcsin(gp1 * Rs**2 - gamma)
                - fRstarki)

    return float(fsolve(residual, Rstar_initial_guess, full_output=False)[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN BLADE BUILDER  (1:1 port)
# ═══════════════════════════════════════════════════════════════════════════════
def blade(beta_i: float, gamma: float, M_i: float, chord: float,
          delta_nu_deg: float = 0.1) -> dict:
    """
    Build a symmetric MoC supersonic impulse blade passage.

    Parameters
    ----------
    beta_i : float
        Inlet metal angle — RADIANS, from the horizontal axis
        (axial direction in our convention).  Input per MATLAB.
    gamma : float
        Ratio of specific heats.
    M_i : float
        Inlet Mach number (= outlet Mach, by symmetry).
    chord : float
        Physical chord length [same units you want (x,y) in — MATLAB uses inches].
    delta_nu_deg : float
        Angular step for characteristic tracing, degrees.  Default 0.1° per MATLAB.

    Returns
    -------
    dict with keys:
        blade_distance : pitch  t  (center-to-center spacing)
        passage_width  : throat width of the passage
        middle_thickness : blade thickness at the mid-chord
        x, y : full contour coordinates (arrays, concatenated)
        x_segments, y_segments : dict of segment-labelled coords for plotting
        Rstar_l, Rstar_u : the two arc radii (scaled by r*)
        M_l, M_u : lower and upper transition Machs
        rstar : scale factor converting unit-chord geometry to physical
    """
    # ── 1:1 with MATLAB ──────────────────────────────────────────────────
    delta_nu = np.deg2rad(delta_nu_deg)

    M_l = lower_mach(mstar(M_i, gamma), gamma)
    M_u = upper_mach(mstar(M_i, gamma), gamma)

    nu_i = prandtl_meyer(M_i, gamma)
    nu_l = prandtl_meyer(M_l, gamma)
    nu_u = prandtl_meyer(M_u, gamma)

    Rstar_l = 1.0 / mstar(M_l, gamma)
    Rstar_u = 1.0 / mstar(M_u, gamma)

    alpha_l = beta_i - (nu_i - nu_l)
    alpha_u = beta_i - (nu_u - nu_i)

    # ── LOWER transition arc ─────────────────────────────────────────────
    k_max = int(np.ceil((nu_i - nu_l) / delta_nu))
    N_l = k_max + 1

    Rstarl = np.zeros(N_l); Rstarl[-1] = Rstar_l
    phi    = np.zeros(N_l)
    y_k    = np.zeros(N_l)
    x_k    = np.zeros(N_l)
    x_star_l = np.zeros(N_l)
    y_star_l = np.zeros(N_l); y_star_l[-1] = Rstar_l
    mu       = np.zeros(N_l)
    mu[-1]   = -np.arcsin(np.sqrt(((gamma + 1) / 2) * Rstar_l - ((gamma - 1) / 2)))
    mach_line         = np.zeros(N_l)
    Slope_Wall_Seg    = np.zeros(N_l)

    for k in range(k_max, 0, -1):            # k = k_max .. 1   (MATLAB 1-based)
        kpy = k - 1                           # Python 0-based index
        kp1 = k                               # MATLAB's k+1  ==  Python k

        Rstarl[kpy] = solve4Rki_l(gamma, k, delta_nu, nu_i, Rstar_l)
        phi[kpy]    = phiki_l(nu_i, nu_l, delta_nu, k)

        x_k[kpy] = -Rstarl[kpy] * np.sin(phi[kpy])
        y_k[kpy] =  Rstarl[kpy] * np.cos(phi[kpy])

        mu[kpy]          = muki(Rstarl[kpy], gamma)
        mach_line[kpy]   = mach_slope(phi[kpy], phi[kp1], mu[kpy], mu[kp1])
        Slope_Wall_Seg[kpy] = np.tan(phi[kp1])

        # Wall coords (intersection of Mach line and wall-slope line)
        ml = mach_line[kpy]
        sw = Slope_Wall_Seg[kpy]
        x_star_l[kpy] = (
            (y_star_l[kp1] - sw * x_star_l[kp1])
            - (y_k[kpy]   - ml * x_k[kpy])
        ) / (ml - sw)
        y_star_l[kpy] = (
            ml * (y_star_l[kp1] - sw * x_star_l[kp1])
            - sw * (y_k[kpy] - ml * x_k[kpy])
        ) / (ml - sw)

    # Rotate lower wall by alpha_l
    c_al = np.cos(alpha_l); s_al = np.sin(alpha_l)
    x_star_lower = x_star_l * c_al - y_star_l * s_al
    y_star_lower = x_star_l * s_al + y_star_l * c_al

    # Symmetric (exit-side) reflection.  np.flip() returns a VIEW, so copy()
    # is required — otherwise later in-place `*= rstar` on both lower and
    # lower_out will scale the underlying array twice.
    x_star_lower_out = -np.flip(x_star_lower).copy()
    y_star_lower_out =  np.flip(y_star_lower).copy()

    # ── UPPER transition arc ─────────────────────────────────────────────
    j_max = int(np.ceil((nu_u - nu_i) / delta_nu))
    N_u = j_max + 1

    Rstaru = np.zeros(N_u); Rstaru[-1] = Rstar_u
    phi_u  = np.zeros(N_u)
    y_ku   = np.zeros(N_u)
    x_ku   = np.zeros(N_u)
    x_star_u = np.zeros(N_u)
    y_star_u = np.zeros(N_u); y_star_u[-1] = Rstar_u
    mu_u     = np.zeros(N_u)
    mu_u[-1] = -np.arcsin(np.sqrt(((gamma + 1) / 2) * Rstar_u - ((gamma - 1) / 2)))
    mach_line_u          = np.zeros(N_u)
    Slope_Wall_Seg_u     = np.zeros(N_u)

    for j in range(j_max, 0, -1):
        jpy = j - 1
        jp1 = j

        Rstaru[jpy] = solve4Rki_u(gamma, j, delta_nu, nu_i, Rstar_u)
        phi_u[jpy]  = phiki_u(nu_i, nu_u, delta_nu, j)

        x_ku[jpy] = -Rstaru[jpy] * np.sin(phi_u[jpy])
        y_ku[jpy] =  Rstaru[jpy] * np.cos(phi_u[jpy])

        mu_u[jpy]        = muki(Rstaru[jpy], gamma)
        mach_line_u[jpy] = np.tan((phi_u[jpy] + phi_u[jp1]) / 2
                                  + (mu_u[jpy] + mu_u[jp1]) / 2)
        Slope_Wall_Seg_u[jpy] = np.tan(phi_u[jp1])

        ml = mach_line_u[jpy]
        sw = Slope_Wall_Seg_u[jpy]
        x_star_u[jpy] = (
            (y_star_u[jp1] - sw * x_star_u[jp1])
            - (y_ku[jpy]   - ml * x_ku[jpy])
        ) / (ml - sw)
        y_star_u[jpy] = (
            ml * (y_star_u[jp1] - sw * x_star_u[jp1])
            - sw * (y_ku[jpy] - ml * x_ku[jpy])
        ) / (ml - sw)

    c_au = np.cos(alpha_u); s_au = np.sin(alpha_u)
    x_star_upper = x_star_u * c_au - y_star_u * s_au
    y_star_upper = x_star_u * s_au + y_star_u * c_au

    x_star_upper_out = -np.flip(x_star_upper).copy()
    y_star_upper_out =  np.flip(y_star_upper).copy()

    # ── Constant-Mach arcs (lower and upper circles) ─────────────────────
    theta_lower = np.arange(np.pi/2 - alpha_l, alpha_l + np.pi/2 + 1e-12, 0.01)
    x_lower_circle = Rstar_l * np.cos(theta_lower)
    y_lower_circle = Rstar_l * np.sin(theta_lower)

    theta_upper = np.arange(np.pi/2 - alpha_u, alpha_u + np.pi/2 + 1e-12, 0.01)
    x_upper_circle = Rstar_u * np.cos(theta_upper)
    y_upper_circle = Rstar_u * np.sin(theta_upper)

    # ── Scale from unit chord to physical chord ──────────────────────────
    Cstar = x_star_lower_out[-1] - x_star_lower[0]
    rstar = chord / Cstar

    # ── Straight-edge section between the two wall traces ───────────────
    horiz = abs(x_star_lower[0] - x_star_upper[0])
    v     = np.tan(beta_i) * horiz
    # MATLAB sign flip (author's note: "I have no idea why")
    if np.rad2deg(beta_i) > 48:
        sign = -1.0
    else:
        sign = 1.0
    vert = v + sign * abs(y_star_upper[0] - y_star_lower[0])

    # Straight inlet leg (tangent to wall at beta_i)
    x_straight_in = np.arange(x_star_lower[0], x_star_upper[0] + delta_nu, delta_nu)
    b_shift       = y_star_lower[0] - np.tan(beta_i) * x_star_lower[0]
    y_straight_in = np.tan(beta_i) * x_straight_in + b_shift

    x_straight_out = -np.flip(x_straight_in).copy()
    y_straight_out =  np.flip(y_straight_in).copy()

    # Blade spacing (passage + middle thickness) — all in unit-chord units
    passage_width = (y_star_lower[-1] - y_star_upper[-1]) * rstar

    # ── Scale all geometry to physical chord ─────────────────────────────
    y_straight_in  *= rstar; x_straight_in  *= rstar
    y_straight_out *= rstar; x_straight_out *= rstar

    x_lower_circle *= rstar; y_lower_circle *= rstar
    x_upper_circle *= rstar
    y_upper_circle  = (y_upper_circle + vert) * rstar

    x_star_lower     *= rstar; y_star_lower     *= rstar
    x_star_lower_out *= rstar; y_star_lower_out *= rstar
    x_star_upper     *= rstar
    y_star_upper      = (y_star_upper + vert) * rstar
    x_star_upper_out *= rstar
    y_star_upper_out  = (y_star_upper_out + vert) * rstar

    middle_thickness = y_star_upper[-1] - y_star_lower[-1]
    blade_distance   = passage_width + middle_thickness

    # ── Assemble full contour (same order as MATLAB) ────────────────────
    x_segments = {
        "straight_in":     x_straight_in,
        "straight_out":    x_straight_out,
        "lower_circle":    x_lower_circle,
        "upper_circle":    x_upper_circle,
        "wall_lower_in":   x_star_lower,
        "wall_lower_out":  x_star_lower_out,
        "wall_upper_in":   x_star_upper,
        "wall_upper_out":  x_star_upper_out,
    }
    y_segments = {
        "straight_in":     y_straight_in,
        "straight_out":    y_straight_out,
        "lower_circle":    y_lower_circle,
        "upper_circle":    y_upper_circle,
        "wall_lower_in":   y_star_lower,
        "wall_lower_out":  y_star_lower_out,
        "wall_upper_in":   y_star_upper,
        "wall_upper_out":  y_star_upper_out,
    }

    x = np.concatenate(list(x_segments.values()))
    y = np.concatenate(list(y_segments.values()))

    return dict(
        blade_distance   = blade_distance,
        passage_width    = passage_width,
        middle_thickness = middle_thickness,
        x = x, y = y,
        x_segments = x_segments,
        y_segments = y_segments,
        Rstar_l = Rstar_l * rstar,
        Rstar_u = Rstar_u * rstar,
        M_l     = M_l,
        M_u     = M_u,
        rstar   = rstar,
        alpha_l_deg = np.rad2deg(alpha_l),
        alpha_u_deg = np.rad2deg(alpha_u),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # MATLAB-style inputs — pick something representative
    beta_i_deg = 65.0          # axial metal angle (impulse)
    gamma      = 1.20
    M_i        = 1.8           # inlet / outlet Mach
    chord      = 1.0           # any unit — output scales linearly

    b = blade(np.deg2rad(beta_i_deg), gamma, M_i, chord, delta_nu_deg=0.1)

    print(f"  β_i              = {beta_i_deg:.2f}°  (axial)")
    print(f"  γ                = {gamma:.3f}")
    print(f"  M_i              = {M_i:.3f}")
    print(f"  chord            = {chord:.4f}")
    print(f"  M_l  (lower arc) = {b['M_l']:.4f}")
    print(f"  M_u  (upper arc) = {b['M_u']:.4f}")
    print(f"  Rstar_l          = {b['Rstar_l']:.4f}")
    print(f"  Rstar_u          = {b['Rstar_u']:.4f}")
    print(f"  α_l              = {b['alpha_l_deg']:.3f}°")
    print(f"  α_u              = {b['alpha_u_deg']:.3f}°")
    print(f"  passage_width    = {b['passage_width']:.5f}")
    print(f"  middle_thickness = {b['middle_thickness']:.5f}")
    print(f"  blade_distance   = {b['blade_distance']:.5f}   (= pitch t)")
    print(f"  rstar            = {b['rstar']:.5f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {
        "straight_in":    "C0",
        "straight_out":   "C0",
        "lower_circle":   "C1",
        "upper_circle":   "C2",
        "wall_lower_in":  "C3",
        "wall_lower_out": "C3",
        "wall_upper_in":  "C4",
        "wall_upper_out": "C4",
    }
    for name, xs in b["x_segments"].items():
        ys = b["y_segments"][name]
        ax.plot(xs, ys, 'o-', ms=2, lw=1, color=colors.get(name, 'k'),
                label=name if not name.endswith("_out") else None)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'MoC impulse blade — β={beta_i_deg}°, γ={gamma}, M={M_i}, chord={chord}')
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig('moc_blade_test.png', dpi=150)
    plt.show()