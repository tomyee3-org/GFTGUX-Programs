"""
plot_bh.py
==========
Matplotlib visualisations for Black_hole_spacetime_visualizer.

One routine per mode:

    plot_embedding   Flamm's paraboloid, a 2-D cross-section and a 3-D
                      surface of revolution
    plot_tidal        radial/tangential tidal acceleration vs. radius, and
                      (if requested) a cross-mass survival comparison
    plot_infall       proper time, coordinate time, local speed and
                      redshift for a radially infalling test particle
    plot_horizons     apparent horizon vs. event horizon in a Vaidya
                      spacetime, with the shooting-method geodesic family

Each routine always displays the figure on screen; if the student also
supplies --outdir, it additionally writes a timestamped PNG into that
directory -- the screen display and the saved file are not alternatives,
both happen on the same run. Use --no_plot to skip the figure (and so
both the screen display and any PNG) entirely.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

import physics_bh as phys

# Colour scheme, kept broadly consistent with the other GFTGU programs.
C_HORIZON  = "#1c1c1c"
C_SURFACE  = "#1f77b4"
C_RADIAL   = "#d62728"      # stretching
C_TANG     = "#1f77b4"      # compressing
C_MARK     = "#9467bd"
C_REF      = "#8c564b"
C_ESCAPE   = "#2ca02c"
C_PLUNGE   = "#d62728"
C_AH       = "#ff7f0e"
C_EH       = "#1c1c1c"


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def _timestamp_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


def _finish(fig, outdir, prefix, dpi):
    """
    Display the figure on screen, and -- if `outdir` is given -- ALSO save a
    timestamped PNG there. --outdir is additive: it augments the on-screen
    display, it does not replace it. To skip the figure (and hence the
    screen display) entirely, use --no_plot, which keeps this function from
    being called at all.
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, _timestamp_name(prefix))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_bh] PNG saved -> {path}")
    print("[plot_bh] Displaying figure on screen ...")
    plt.show()
    plt.close(fig)


# ----------------------------------------------------------------------
# Mode: embed
# ----------------------------------------------------------------------
def plot_embedding(result, outdir=None, dpi=150, lw=1.6, figsize=(13.0, 6.2)):
    """2-D profile and 3-D surface of revolution of Flamm's paraboloid."""
    s = result["summary"]
    r = result["r"]
    z = result["z"]
    rs = s["rs_m"]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    fig.suptitle(
        f"Flamm's paraboloid — embedding of the equatorial Schwarzschild "
        fr"slice, $M={s['m_msun']:.3g}\,M_\odot$, $r_s={s['rs_km']:.3g}$ km",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: 2-D cross-section, both halves for a symmetric profile ---
    ax1.plot(r / rs, z / rs, color=C_SURFACE, lw=lw, label="z(r)  (upper sheet)")
    ax1.plot(r / rs, -z / rs, color=C_SURFACE, lw=lw, alpha=0.35, ls="--",
              label="mirror image (plotting choice only)")
    ax1.axvline(1.0, color=C_HORIZON, lw=1.1, ls=":", label=r"horizon, $r=r_s$")
    ax1.set_xlabel(r"$r / r_s$")
    ax1.set_ylabel(r"$z / r_s$")
    ax1.set_title("Radial profile")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax1.text(0.98, 0.97,
             "vertical tangent at $r=r_s$: the throat,\nnot a numerical artefact",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                       alpha=0.9))

    # --- Panel 2: 3-D surface of revolution -----------------------------
    n_phi = 140
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    R, PHI = np.meshgrid(r, phi)
    Z = np.tile(z, (n_phi, 1))
    X = R * np.cos(PHI)
    Y = R * np.sin(PHI)
    ax2.plot_surface(X / rs, Y / rs, Z / rs, cmap="viridis",
                     rstride=2, cstride=6, linewidth=0, antialiased=True,
                     alpha=0.92)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta),
             color=C_HORIZON, lw=1.8, zorder=10)
    ax2.set_xlabel(r"$x/r_s$")
    ax2.set_ylabel(r"$y/r_s$")
    ax2.set_zlabel(r"$z/r_s$")
    ax2.set_title("Surface of revolution")
    ax2.view_init(elev=28, azim=-60)

    note = "\n".join([
        f"plotted to {s['r_max_rs']:.3g} " r"$r_s$",
        f"{s['n_points']:,} radial points",
    ])
    ax1.text(0.02, 0.98, note, transform=ax1.transAxes, ha="left", va="top",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                   ec="gray", alpha=0.85))

    _finish(fig, outdir, f"bh_embed_{s['m_msun']:.2f}Msun", dpi)


# ----------------------------------------------------------------------
# Mode: tidal
# ----------------------------------------------------------------------
def plot_tidal(result, compare_rows, separation_m, limit_g,
              outdir=None, dpi=150, lw=1.7, figsize=(12.5, 9.2)):
    """Tidal acceleration vs. radius, and (optionally) a cross-mass panel."""
    s = result["summary"]
    r = result["r"]
    rs = s["rs_m"]
    a_r = result["a_radial"]
    a_t = result["a_tangential"]

    if compare_rows:
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
        ax1, ax2 = axes

    fig.suptitle(
        f"Tidal acceleration — "
        fr"$M={s['m_msun']:.3g}\,M_\odot$, test-rod separation "
        f"{s['separation_m']:.2g} m",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: acceleration in SI units, log-log ---------------------
    ax1.plot(r / rs, a_r, color=C_RADIAL, lw=lw, label="radial (stretching)")
    ax1.plot(r / rs, a_t, color=C_TANG, lw=lw, ls="--", label="tangential (compressing)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.axvline(1.0, color=C_HORIZON, lw=1.0, ls=":", label=r"horizon $r=r_s$")
    ax1.set_xlabel(r"$r/r_s$")
    ax1.set_ylabel(r"tidal acceleration [m s$^{-2}$]")
    ax1.set_title("Tidal acceleration vs. radius")
    ax1.grid(True, which="both", lw=0.3, alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # --- Panel 2: in units of g, with the survival threshold ------------
    ax2.plot(r / rs, a_r / phys.g0, color=C_RADIAL, lw=lw, label="radial (stretching)")
    ax2.plot(r / rs, a_t / phys.g0, color=C_TANG, lw=lw, ls="--",
             label="tangential (compressing)")
    ax2.axhline(limit_g, color=C_MARK, lw=1.1, ls="-.",
               label=fr"illustrative survival limit, {limit_g:g} $g$")
    ax2.axvline(1.0, color=C_HORIZON, lw=1.0, ls=":")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$r/r_s$")
    ax2.set_ylabel(r"tidal acceleration [$g$]")
    ax2.set_title("Same, in standard gravities")
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)

    if compare_rows:
        masses = np.array([row["m_msun"] for row in compare_rows])
        a_h_g = np.array([row["a_radial_horizon_g"] for row in compare_rows])
        r_ratio = np.array([row["r_crit_over_rs"] for row in compare_rows])

        # --- Panel 3: tidal accel. at the horizon vs. mass --------------
        ax3.plot(masses, a_h_g, color=C_RADIAL, lw=lw, marker="o", ms=5)
        ax3.axhline(limit_g, color=C_MARK, lw=1.1, ls="-.",
                   label=fr"{limit_g:g} $g$ limit")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.set_xlabel(r"$M/M_\odot$")
        ax3.set_ylabel(r"radial tidal accel. at $r_s$  [$g$]")
        ax3.set_title(r"Bigger holes are gentler at the horizon ($\propto M^{-2}$)")
        ax3.grid(True, which="both", lw=0.3, alpha=0.5)
        ax3.legend(fontsize=8, loc="upper right", framealpha=0.85)

        # --- Panel 4: survival radius / horizon radius vs. mass ---------
        ax4.plot(masses, r_ratio, color=C_MARK, lw=lw, marker="o", ms=5)
        ax4.axhline(1.0, color=C_HORIZON, lw=1.1, ls=":",
                   label=r"$r_{\rm crit}=r_s$")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.set_xlabel(r"$M/M_\odot$")
        ax4.set_ylabel(r"$r_{\rm crit}/r_s$")
        ax4.set_title("Where the survival limit is reached, relative to the horizon")
        ax4.grid(True, which="both", lw=0.3, alpha=0.5)
        ax4.text(0.03, 0.06,
                 "above the dotted line: torn apart before reaching the horizon\n"
                 "below it: crosses the horizon intact (by this criterion)",
                 transform=ax4.transAxes, ha="left", va="bottom", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                           alpha=0.9))
        ax4.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _finish(fig, outdir, f"bh_tidal_{s['m_msun']:.2f}Msun", dpi)


# ----------------------------------------------------------------------
# Mode: infall
# ----------------------------------------------------------------------
def plot_infall(result, outdir=None, dpi=150, lw=1.7, figsize=(12.5, 9.2)):
    """Four-panel view of a radial free fall from rest at r0."""
    s = result["summary"]
    tau_ms = result["tau"] * 1.0e3
    t_ms = result["t"] * 1.0e3
    r_rs = result["r"] / s["rs_m"]
    v = result["v_local"]
    z = result["redshift"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    fig.suptitle(
        f"Radial infall — "
        fr"$M={s['m_msun']:.3g}\,M_\odot$, released from rest at "
        fr"$r_0={s['r0_rs']:.3g}\,r_s$  ($E={s['E']:.4f}$)",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: r vs. proper time --------------------------------------
    ax1.plot(tau_ms, r_rs, color=C_SURFACE, lw=lw)
    ax1.axhline(1.0, color=C_HORIZON, lw=1.0, ls=":", label=r"horizon $r=r_s$")
    ax1.set_xlabel(r"proper time $\tau$ [ms]")
    ax1.set_ylabel(r"$r/r_s$")
    ax1.set_title("Radius vs. the infalling observer's own clock")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)
    ax1.text(0.03, 0.06,
             fr"reaches $r={s['r_stop_rs']:.6g}\,r_s$ at $\tau \approx "
             fr"{tau_ms[-1]:.4g}$ ms (run stopped there; the horizon"
             "\nitself is crossed at a slightly larger, finite $\\tau$, "
             "not computed by this run)",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=7.3,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                       alpha=0.9))

    # --- Panel 2: coordinate time vs. proper time -------------------------
    ax2.plot(tau_ms, t_ms, color=C_RADIAL, lw=lw)
    ax2.plot(tau_ms, tau_ms, color=C_REF, lw=1.0, ls="--",
             label=r"$t=\tau$ (Newtonian expectation)")
    ax2.set_xlabel(r"proper time $\tau$ [ms]")
    ax2.set_ylabel(r"Schwarzschild coordinate time $t$ [ms]")
    ax2.set_title("Coordinate time races ahead of proper time")
    ax2.grid(True, lw=0.3, alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # --- Panel 3: local infall speed --------------------------------------
    ax3.plot(r_rs, v, color=C_MARK, lw=lw)
    ax3.axhline(1.0, color=C_HORIZON, lw=1.0, ls=":", label=r"$v=c$")
    ax3.axvline(1.0, color=C_HORIZON, lw=1.0, ls=":")
    ax3.invert_xaxis()
    ax3.set_xlabel(r"$r/r_s$")
    ax3.set_ylabel(r"local infall speed $v/c$")
    ax3.set_title("Speed measured by the local static observers passed en route")
    ax3.grid(True, lw=0.3, alpha=0.5)
    ax3.legend(fontsize=8, loc="upper right", framealpha=0.85)

    # --- Panel 4: redshift factor ------------------------------------------
    ax4.plot(r_rs, z, color=C_TANG, lw=lw)
    ax4.set_yscale("log")
    ax4.axvline(1.0, color=C_HORIZON, lw=1.0, ls=":", label=r"horizon $r=r_s$")
    ax4.invert_xaxis()
    ax4.set_xlabel(r"$r/r_s$")
    ax4.set_ylabel(r"$\nu_{\rm obs}/\nu_{\rm emit}$")
    ax4.set_title("Light sent outward is increasingly redshifted, ever later")
    ax4.grid(True, which="both", lw=0.3, alpha=0.5)
    ax4.legend(fontsize=8, loc="upper right", framealpha=0.85)

    _finish(fig, outdir, f"bh_infall_{s['m_msun']:.2f}Msun", dpi)


# ----------------------------------------------------------------------
# Mode: horizons
# ----------------------------------------------------------------------
def plot_horizons(result, outdir=None, dpi=150, lw=2.0, figsize=(13.5, 5.6)):
    """Apparent vs. event horizon in the Vaidya spacetime, and the geodesic
    family used to locate the event horizon."""
    s = result["summary"]
    rs0 = s["rs0_m"]
    v = result["v"] / rs0
    r_ah = result["r_AH"] / rs0
    r_eh = result["r_EH"] / rs0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    fig.suptitle(
        f"Vaidya spacetime — accreting from "
        fr"${s['m0_msun']:.3g}$ to ${s['m1_msun']:.3g}\,M_\odot$ over "
        fr"${s['duration_rs0']:.3g}\,r_{{s0}}/c$ of advanced time",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: r_AH(v) and r_EH(v) -------------------------------------
    no_accretion = (s["m0_msun"] == s["m1_msun"])
    if not no_accretion:
        ax1.axvspan(s["v1_rs0"], s["v2_rs0"], color="#fdecd8", zorder=0,
                   label="accretion under way")
    ax1.plot(v, r_ah, color=C_AH, lw=lw, ls="--", label=r"apparent horizon $r_{AH}=2M(v)$")
    ax1.plot(v, r_eh, color=C_EH, lw=lw, label=r"event horizon $r_{EH}(v)$")
    if not no_accretion:
        ax1.axvline(s["v1_rs0"], color="gray", lw=0.8, ls=":")
        ax1.axvline(s["v2_rs0"], color="gray", lw=0.8, ls=":")
    ax1.set_xlabel(r"advanced time $v / r_{s0}$")
    ax1.set_ylabel(r"$r / r_{s0}$")
    ax1.set_title("Apparent vs. event horizon")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(fontsize=8, loc="upper left", framealpha=0.9)
    if no_accretion:
        ax1.text(0.98, 0.04,
                 "M0 = M1: no accretion,\nso $r_{AH}=r_{EH}=r_{s0}$\nfor the whole run",
                 transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.7,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                           alpha=0.9))
    else:
        ax1.text(0.98, 0.04,
                 "event horizon rises\nbefore accretion starts:\nit anticipates the future",
                 transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.7,
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                           alpha=0.9))

    # --- Panel 2: geodesic family -----------------------------------------
    for fam in result["family"]:
        col = C_ESCAPE if fam["escapes"] else C_PLUNGE
        ax2.plot(fam["v"] / rs0, fam["r"] / rs0, color=col, lw=1.1, alpha=0.85)
    ax2.plot(v, r_eh, color=C_EH, lw=lw, label="event-horizon generator")
    if not no_accretion:
        ax2.axvspan(s["v1_rs0"], s["v2_rs0"], color="#fdecd8", zorder=0)
    ax2.set_xlim(v.min(), v.max())
    # Escaping family members are integrated well past r_s1 (out to an
    # unambiguous-escape cutoff), so cap the axis at a fixed multiple of the
    # *final* horizon size rather than at the trajectories' own maxima.
    rs1_over_rs0 = s["rs1_m"] / rs0
    y_hi = max(2.2 * rs1_over_rs0, r_eh.max() * 1.15)
    ax2.set_ylim(0, y_hi)
    ax2.set_xlabel(r"advanced time $v / r_{s0}$")
    ax2.set_ylabel(r"$r / r_{s0}$")
    ax2.set_title("How the shooting method finds it")
    ax2.grid(True, lw=0.3, alpha=0.5)
    green_patch = plt.Line2D([0], [0], color=C_ESCAPE, lw=2, label="escapes to infinity")
    red_patch = plt.Line2D([0], [0], color=C_PLUNGE, lw=2, label="plunges to $r=0$")
    black_patch = plt.Line2D([0], [0], color=C_EH, lw=lw, label="event horizon")
    ax2.legend(handles=[green_patch, red_patch, black_patch], fontsize=8,
              loc="upper left", framealpha=0.9)

    note = "\n".join([
        fr"$r_{{\rm crit}}/r_{{s0}} = {s['r_crit_over_rs0']:.6f}$ at $v_{{\rm start}}$",
        fr"bisection residual $\approx {s['residual_rs0']:.2e}\,r_{{s0}}$",
    ])
    ax1.text(0.02, 0.34, note, transform=ax1.transAxes, ha="left", va="top",
             fontsize=7.7, bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                     ec="gray", alpha=0.85))

    _finish(fig, outdir, f"bh_horizons_{s['m0_msun']:.2f}to{s['m1_msun']:.2f}Msun", dpi)
