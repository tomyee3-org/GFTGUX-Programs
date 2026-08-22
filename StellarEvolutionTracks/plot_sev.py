"""
plot_sev.py
===========
Matplotlib visualisations for StellarEvolutionTracks.

One routine per mode:

    plot_track          single-star evolution, four panels
    plot_hr_diagram     HR diagram with several tracks and isochrones
    plot_wd_cooling     white-dwarf structure and cooling, four panels
    plot_ns_mass_radius neutron-star mass-radius relation, three panels

Each routine displays the figure.  When an output directory is supplied,
it also writes a timestamped PNG into that directory before displaying it.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import physics_sev as phys

# Colour scheme, kept consistent with the other GFTGU programs.
C_MS    = "#1f77b4"      # main sequence
C_SGB   = "#ff7f0e"      # subgiant branch / Hertzsprung gap
C_RGB   = "#d62728"      # red-giant branch
C_ZAMS  = "#7f7f7f"      # ZAMS reference line
C_ISO   = "#2ca02c"      # isochrones
C_MARK  = "#9467bd"      # special points
C_REF   = "#8c564b"      # analytic reference curves

PHASE_COLOR = {0: C_MS, 1: C_SGB, 2: C_RGB}
PHASE_NAME = {0: "main sequence",
              1: "subgiant / Hertzsprung gap",
              2: "red-giant branch"}


# ----------------------------------------------------------------------
# Output helpers
# ----------------------------------------------------------------------
def _timestamp_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


def _finish(fig, outdir, prefix, dpi):
    """Optionally save a timestamped PNG, then display the figure."""
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, _timestamp_name(prefix))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_sev] PNG saved -> {path}")
    print("[plot_sev] Displaying figure on screen ...")
    plt.show()
    plt.close(fig)


def _hr_axes(ax):
    """Standard HR-diagram axis conventions: hot and bright at upper left."""
    ax.invert_xaxis()
    ax.set_xlabel(r"$\log_{10}\,T_{\rm eff}$ [K]")
    ax.set_ylabel(r"$\log_{10}\,(L/L_\odot)$")
    ax.grid(True, lw=0.3, alpha=0.5)


def _draw_zams(ax, label=True):
    _, logT, logL = phys.zams_curve()
    ax.plot(logT, logL, color=C_ZAMS, lw=1.0, ls="--",
            label="zero-age main sequence" if label else None, zorder=1)


def _frame_hr(ax, logT, logL, pad_x=0.12, pad_y=0.45):
    """Limit an HR panel to the plotted data with a little breathing room."""
    lo_x, hi_x = float(np.min(logT)), float(np.max(logT))
    lo_y, hi_y = float(np.min(logL)), float(np.max(logL))
    span_x = max(hi_x - lo_x, 0.25)
    ax.set_xlim(hi_x + pad_x * span_x, lo_x - pad_x * span_x)
    ax.set_ylim(lo_y - pad_y, hi_y + pad_y)


def _draw_sun(ax):
    ax.plot(np.log10(phys.TEFF_SUN), 0.0, marker="o", ms=7,
            mfc="gold", mec="k", mew=0.7, ls="none", label="Sun today", zorder=6)


# ----------------------------------------------------------------------
# Mode: tracks
# ----------------------------------------------------------------------
def plot_track(result, outdir=None, dpi=150, lw=1.6, figsize=(12.5, 9.0)):
    """Four-panel view of a single evolutionary track."""
    s = result["summary"]
    t_gyr = result["t"] / phys.GYR
    L = result["L"]
    R = result["R"]
    T = result["Teff"]
    Xc = result["Xc"]
    Mc = result["Mcore"]
    ph = result["phase"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    fig.suptitle(
        f"Stellar evolution track — "
        fr"$M={s['m_msun']:.2f}\,M_\odot$,  $X={s['X']:.2f}$,  $Z={s['Z']:.3f}$,  "
        fr"{s['burning']} burning ($\nu={s['nu']:.0f}$), {s['opacity']} opacity",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: HR diagram ------------------------------------------
    _draw_zams(ax1)
    for code in (0, 1, 2):
        sel = ph == code
        if np.any(sel):
            ax1.plot(np.log10(T[sel]), np.log10(L[sel]),
                     color=PHASE_COLOR[code], lw=lw, label=PHASE_NAME[code])
    ax1.plot(np.log10(T[0]), np.log10(L[0]), marker="o", ms=6, color="k",
             ls="none", label="ZAMS", zorder=5)
    if s["reached_tams"]:
        i_tams = (int(np.argmax(t_gyr >= s["t_ms_gyr"])) if s["post_ms"]
                  else len(T) - 1)
        ax1.plot(np.log10(T[i_tams]), np.log10(L[i_tams]), marker="s", ms=6,
                 mfc="none", mec="k", ls="none", label="TAMS", zorder=5)
    else:
        # The track was stopped at t_max while still burning hydrogen, so
        # its last point is an integration stop, not a terminal-age main
        # sequence.  Labelling it TAMS would be a straightforward lie.
        ax1.plot(np.log10(T[-1]), np.log10(L[-1]), marker="x", ms=8, mew=1.6,
                 color="k", ls="none",
                 label=fr"stopped at $t_{{\rm max}}={s['t_stop_gyr']:.3g}\,$Gyr",
                 zorder=5)
    if s["post_ms"]:
        ax1.plot(np.log10(T[-1]), np.log10(L[-1]), marker="*", ms=13,
                 color=C_MARK, ls="none", label="He ignition", zorder=5)
    if abs(s["m_msun"] - 1.0) < 1e-9:
        _draw_sun(ax1)
    _hr_axes(ax1)
    _frame_hr(ax1, np.log10(T), np.log10(L))
    ax1.set_title("Hertzsprung–Russell diagram")
    ax1.legend(fontsize=7.5, loc="lower left", framealpha=0.8)

    # --- Panel 2: luminosity history ----------------------------------
    ax2.plot(t_gyr, L, color=C_MS, lw=lw)
    ax2.set_yscale("log")
    if s["reached_tams"]:
        ax2.axvline(s["t_ms_gyr"], color="k", lw=0.9, ls=":",
                    label=fr"TAMS  $t={s['t_ms_gyr']:.3g}\,$Gyr")
    else:
        ax2.axvline(s["t_stop_gyr"], color="k", lw=0.9, ls=":",
                    label=fr"integration stop  $t_{{\rm max}}"
                          fr"={s['t_stop_gyr']:.3g}\,$Gyr (still on the MS)")
    ax2.set_xlabel("Age [Gyr]")
    ax2.set_ylabel(r"$L/L_\odot$")
    ax2.set_title("Luminosity history")
    ax2.grid(True, lw=0.3, alpha=0.5)
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.8)

    # --- Panel 3: radius history --------------------------------------
    ax3.plot(t_gyr, R, color=C_RGB, lw=lw)
    ax3.set_yscale("log")
    ax3.axvline(s["t_ms_gyr"] if s["reached_tams"] else s["t_stop_gyr"],
                color="k", lw=0.9, ls=":")
    ax3.set_xlabel("Age [Gyr]")
    ax3.set_ylabel(r"$R/R_\odot$")
    ax3.set_title("Radius history")
    ax3.grid(True, lw=0.3, alpha=0.5)

    # --- Panel 4: composition and core mass ---------------------------
    ax4.plot(t_gyr, Xc, color=C_MS, lw=lw, label=r"central $X_c$")
    ax4.set_xlabel("Age [Gyr]")
    ax4.set_ylabel(r"central hydrogen fraction $X_c$")
    ax4.set_ylim(-0.02, max(0.05, s["X"] * 1.08))
    ax4.grid(True, lw=0.3, alpha=0.5)
    ax4b = ax4.twinx()
    ax4b.plot(t_gyr, Mc, color=C_RGB, lw=lw, ls="--",
              label=r"helium core $M_c$")
    ax4b.set_ylabel(r"$M_c/M_\odot$")
    ax4.set_title("Central hydrogen and helium-core growth")
    handles = ax4.get_lines() + ax4b.get_lines()
    ax4.legend(handles, [h.get_label() for h in handles],
               fontsize=8, loc="center right", framealpha=0.8)

    if s["reached_tams"]:
        note = [fr"$t_{{\rm MS}}={s['t_ms_gyr']:.3g}\,$Gyr"]
    else:
        note = [fr"$t_{{\rm MS}}>{s['t_stop_gyr']:.3g}\,$Gyr (not reached)"]
    if s["post_ms"]:
        note.append(fr"$t_{{\rm post}}={s['t_post_gyr']:.3g}\,$Gyr")
        note.append("track ends at He ignition")
    note.append(f"schematic remnant: {s['remnant_kind']}")
    note.append(fr"$M_{{\rm rem}}\approx{s['remnant_msun']:.2f}\,M_\odot$"
                " (not integrated)")
    ax1.text(0.02, 0.98, "\n".join(note), transform=ax1.transAxes,
             ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                       ec="gray", alpha=0.9))

    _finish(fig, outdir, f"sev_track_{s['m_msun']:.2f}Msun", dpi)


# ----------------------------------------------------------------------
# Mode: hr
# ----------------------------------------------------------------------
def plot_hr_diagram(result, outdir=None, dpi=150, lw=1.5, figsize=(10.0, 8.5)):
    """HR diagram with one track per mass and optional isochrones."""
    s = result["summary"]
    tracks = result["tracks"]
    isos = result["isochrones"]

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.suptitle(
        f"Hertzsprung–Russell diagram — {s['n_tracks']} evolutionary tracks, "
        fr"$X={s['X']:.2f}$, $Z={s['Z']:.3f}$",
        fontsize=12, fontweight="bold")

    _draw_zams(ax)

    cmap = plt.get_cmap("viridis")
    n = len(tracks)
    for i, tr in enumerate(tracks):
        col = cmap(i / max(n - 1, 1))
        m = tr["summary"]["m_msun"]
        logT = np.log10(tr["Teff"])
        logL = np.log10(tr["L"])
        ax.plot(logT, logL, color=col, lw=lw)
        ax.plot(logT[0], logL[0], marker="o", ms=4, color=col, ls="none")
        ax.annotate(f"{m:g}", xy=(logT[0], logL[0]),
                    xytext=(4, -9), textcoords="offset points",
                    fontsize=8, color=col, fontweight="bold")

    # Isochrones join stars of equal age, so the points are drawn in order
    # of increasing mass: up the main sequence, round the turn-off, and out
    # along the giant branch.
    iso_styles = ["-", "--", "-.", ":"]
    iso_colors = ["#2ca02c", "#e377c2", "#17becf", "#bcbd22"]
    for j, iso in enumerate(isos):
        pts = iso["points"]
        lt = np.array([q[1] for q in pts])
        ll = np.array([q[2] for q in pts])
        ax.plot(lt, ll, color=iso_colors[j % len(iso_colors)], lw=1.4,
                ls=iso_styles[j % len(iso_styles)], marker="d", ms=4,
                zorder=4,
                label=fr"isochrone  $t={iso['age_gyr']:g}\,$Gyr")

    _draw_sun(ax)
    _hr_axes(ax)
    all_T = np.concatenate([np.log10(tr["Teff"]) for tr in tracks])
    all_L = np.concatenate([np.log10(tr["L"]) for tr in tracks])
    _frame_hr(ax, all_T, all_L, pad_x=0.06, pad_y=0.35)
    truncated = [m for m, ok in zip(s["masses"], s["reached_tams"]) if not ok]
    subtitle = r"Masses annotated at the ZAMS, in $M_\odot$"
    subtitle += "  ·  schematic teaching tracks and isochrones"
    if truncated:
        subtitle += ("\nstill on the main sequence at "
                     fr"$t_{{\rm max}}={s['t_max_gyr']:g}\,$Gyr: "
                     + ", ".join(f"{m:g}" for m in truncated)
                     + r"$\,M_\odot$")
    ax.set_title(subtitle, fontsize=9.5)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.85)

    _finish(fig, outdir, "sev_hr", dpi)


# ----------------------------------------------------------------------
# Mode: wdcool
# ----------------------------------------------------------------------
def plot_wd_cooling(result, outdir=None, dpi=150, lw=1.6, figsize=(12.5, 9.0)):
    """White-dwarf mass-radius relation and Mestel cooling history."""
    s = result["summary"]
    t_gyr = result["t"] / phys.GYR
    L = result["L"]
    Teff = result["Teff"]
    Tc = result["Tc"]

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3, ax4 = axes.ravel()

    fig.suptitle(
        f"White-dwarf structure and cooling — "
        fr"$M={s['m_msun']:.3f}\,M_\odot$,  $\mu_e={s['mu_e']:.2f}$,  "
        fr"$A_{{\rm ion}}={s['A_ion']:.0f}$,  envelope $Z={s['Z_env']:g}$"
        "\nzero-temperature Chandrasekhar structure with classical Mestel "
        "cooling: ages are model ages, not modern cooling ages",
        fontsize=11.5, fontweight="bold")

    # --- Panel 1: mass-radius relation --------------------------------
    R_km = result["mr_R"] * phys.R_sun / 1.0e3
    ax1.plot(result["mr_M"], R_km, color=C_MS, lw=lw,
             label="Chandrasekhar structure")
    ax1.axvline(s["M_ch"], color=C_RGB, lw=1.0, ls="--",
                label=fr"$M_{{\rm Ch}}={s['M_ch']:.3f}\,M_\odot$")
    ax1.plot([s["m_msun"]], [s["R_km"]], marker="*", ms=15, color=C_MARK,
             ls="none", label="this white dwarf")
    ax1.axhline(6371.0, color=C_ZAMS, lw=0.9, ls=":", label="Earth radius")
    ax1.set_xlabel(r"$M/M_\odot$")
    ax1.set_ylabel("Radius [km]")
    ax1.set_yscale("log")
    ax1.set_title("Cold white-dwarf mass–radius relation")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(fontsize=8, loc="lower left", framealpha=0.85)

    # --- Panel 2: luminosity vs age, log-log --------------------------
    good = t_gyr > 0
    ax2.plot(t_gyr[good], L[good], color=C_MS, lw=lw, label=r"RK4 $L(t)$")
    if np.any(good):
        # Draw the asymptotic Mestel slope only over the late-time portion,
        # where it is supposed to apply.
        t_ref = t_gyr[good]
        L_ref = L[good]
        start = int(len(t_ref) * 0.45)
        anchor = int(len(t_ref) * 0.75)
        tt = t_ref[start:]
        ax2.plot(tt, L_ref[anchor] * (tt / t_ref[anchor]) ** (-1.4),
                 color=C_REF, lw=1.3, ls="--",
                 label=r"asymptotic Mestel slope $L\propto t^{-7/5}$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cooling age [Gyr]")
    ax2.set_ylabel(r"$L/L_\odot$")
    ax2.set_title("Cooling luminosity (Mestel model)")
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.85)

    # --- Panel 3: temperatures ----------------------------------------
    # Both curves share one logarithmic axis: the core is always some three
    # orders of magnitude hotter than the surface, and the gap widens as the
    # star cools.
    ax3.plot(t_gyr, Tc, color=C_MS, lw=lw, ls="--", label=r"core $T_c$")
    ax3.plot(t_gyr, Teff, color=C_RGB, lw=lw, label=r"surface $T_{\rm eff}$")
    ax3.set_xlabel("Cooling age [Gyr]")
    ax3.set_ylabel("Temperature [K]")
    ax3.set_yscale("log")
    ax3.grid(True, which="both", lw=0.3, alpha=0.5)
    ax3.set_title("Surface and core temperature")
    ax3.legend(fontsize=8.5, loc="upper right", framealpha=0.85)

    # --- Panel 4: cooling track on the HR diagram ---------------------
    _draw_zams(ax4)
    ax4.plot(np.log10(Teff), np.log10(L), color=C_MARK, lw=lw,
             label="white-dwarf cooling track")
    ax4.plot(np.log10(Teff[0]), np.log10(L[0]), marker="o", ms=6, color="k",
             ls="none", label=fr"$t=0$, $T_c={s['Tc0']:.1e}\,$K")
    ax4.plot(np.log10(Teff[-1]), np.log10(L[-1]), marker="s", ms=6,
             mfc="none", mec="k", ls="none",
             label=fr"$t={s['t_end_gyr']:.2f}\,$Gyr")
    _draw_sun(ax4)
    _hr_axes(ax4)
    _frame_hr(ax4,
              np.append(np.log10(Teff), np.log10(phys.TEFF_SUN)),
              np.append(np.log10(L), 0.0), pad_x=0.10, pad_y=0.5)
    ax4.set_title("Where white dwarfs sit in the HR diagram")
    ax4.legend(fontsize=8, loc="upper left", framealpha=0.85)

    note = "\n".join([
        fr"$R={s['R_km']:.0f}$ km $={s['R_rearth']:.2f}\,R_\oplus$",
        fr"$\rho_c={s['rho_c']:.2e}$ kg m$^{{-3}}$",
        fr"$\bar\rho={s['mean_density']:.2e}$ kg m$^{{-3}}$",
        fr"$g={s['surface_gravity']:.2e}$ m s$^{{-2}}$",
    ])
    ax1.text(0.98, 0.97, note, transform=ax1.transAxes, ha="right", va="top",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                                   ec="gray", alpha=0.9))

    _finish(fig, outdir, f"sev_wdcool_{s['m_msun']:.2f}Msun", dpi)


# ----------------------------------------------------------------------
# Mode: nsmr
# ----------------------------------------------------------------------
def plot_ns_mass_radius(result, outdir=None, dpi=150, lw=1.7,
                        figsize=(13.0, 4.8), m_observed=2.01):
    """
    Neutron-star mass-radius relation.

    The figure adapts to what has actually been computed.  Stability
    labels appear only when the sequence really turned over; the
    gravitational redshift, the horizon line and the Buchdahl bound are
    results of general relativity and are drawn only for a TOV sequence.
    For a Newtonian run the third panel becomes a self-consistency
    diagnostic instead.
    """
    s = result["summary"]
    rho = result["rho"]
    M = result["M"]
    R = result["R"]
    z = result["z"]
    i_max = result["i_max"]
    gr = bool(s["relativistic"])
    turned = bool(s["turning_point"])
    classify = gr and turned            # only then is "unstable" meaningful

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3 = axes

    eos_label = ("ideal degenerate neutron gas" if s["eos"] == "neutron"
                 else fr"toy stiff polytrope $\Gamma={s['gamma']:g}$, "
                      fr"$p_{{\rm nuc}}={s['p_nuc']:g}$")
    title = (f"Neutron-star models — {eos_label}, "
             + ("TOV (general relativistic)" if gr else "Newtonian gravity"))
    if not gr:
        title += ("\nNewtonian gravity is not a valid description of a "
                  "neutron star: this run is a controlled failure, for "
                  "comparison with the TOV result")
    fig.suptitle(title, fontsize=11.5, fontweight="bold")

    stable = np.arange(M.size) <= i_max
    good = np.isfinite(M) & np.isfinite(R)

    # --- Panel 1: mass-radius -----------------------------------------
    if classify:
        ax1.plot(R[good & stable], M[good & stable], color=C_MS, lw=lw,
                 label="stable branch")
        if np.any(good & ~stable):
            ax1.plot(R[good & ~stable], M[good & ~stable], color=C_ZAMS,
                     lw=1.1, ls="--", label="unstable branch")
        ax1.plot([R[i_max]], [M[i_max]], marker="*", ms=15, color=C_MARK,
                 ls="none",
                 label=fr"$M_{{\rm max}}={M[i_max]:.3f}\,M_\odot$")
    else:
        ax1.plot(R[good], M[good], color=C_MS, lw=lw,
                 label="computed sequence")
        ax1.plot([R[i_max]], [M[i_max]], marker="*", ms=15, color=C_MARK,
                 ls="none",
                 label=fr"largest sampled mass ${M[i_max]:.3f}\,M_\odot$")
    if m_observed is not None and m_observed > 0:
        ax1.axhline(m_observed, color=C_RGB, lw=1.0, ls="--",
                    label=fr"$\sim{m_observed:g}\,M_\odot$ observational"
                          " benchmark")

    r_grid = np.linspace(max(np.nanmin(R[good]) * 0.6, 1.0),
                         np.nanmax(R[good]) * 1.15, 200)
    if gr:
        m_bh = r_grid * 1.0e3 * phys.c**2 / (2.0 * phys.G * phys.M_sun)
        m_buch = 4.0 * r_grid * 1.0e3 * phys.c**2 / (9.0 * phys.G * phys.M_sun)
        ax1.plot(r_grid, m_bh, color="k", lw=0.9, ls=":",
                 label=r"$R=2GM/c^2$ (black hole)")
        ax1.plot(r_grid, m_buch, color="k", lw=0.9, ls="-.",
                 label=r"$R=9GM/4c^2$ (Buchdahl)")
    ax1.set_xlim(r_grid[0], r_grid[-1])
    ax1.set_ylim(0, max(np.nanmax(M[good]) * 1.35,
                        (m_observed or 0) * 1.25))
    ax1.set_xlabel("Radius [km]")
    ax1.set_ylabel(r"$M/M_\odot$")
    ax1.set_title("Mass–radius relation")
    ax1.grid(True, lw=0.3, alpha=0.5)
    ax1.legend(fontsize=7.0, loc="upper right", framealpha=0.92)

    # --- Panel 2: mass vs central density -----------------------------
    if classify:
        ax2.plot(rho[good & stable], M[good & stable], color=C_MS, lw=lw,
                 label="stable  $dM/d\\rho_c>0$")
        if np.any(good & ~stable):
            ax2.plot(rho[good & ~stable], M[good & ~stable], color=C_ZAMS,
                     lw=1.1, ls="--", label="unstable  $dM/d\\rho_c<0$")
        ax2.set_title("Turning point sets the maximum mass")
    else:
        ax2.plot(rho[good], M[good], color=C_MS, lw=lw,
                 label="computed sequence")
        ax2.set_title("No turning point in the sampled range"
                      if not turned else "Mass against central density")
    ax2.plot([rho[i_max]], [M[i_max]], marker="*", ms=15, color=C_MARK, ls="none")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"central density $\rho_c$ [kg m$^{-3}$]")
    ax2.set_ylabel(r"$M/M_\odot$")
    ax2.grid(True, which="both", lw=0.3, alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.85)

    # --- Panel 3: compactness, and redshift only if it means anything --
    ax3.plot(M[good], result["compact"][good], color=C_MS, lw=lw,
             label=r"compactness $GM/Rc^2$" if gr
             else r"$GM/Rc^2$ (consistency diagnostic)")
    ax3.set_xlabel(r"$M/M_\odot$")
    ax3.set_ylabel(r"$GM/Rc^2$")
    ax3.grid(True, lw=0.3, alpha=0.5)
    if gr:
        ax3.axhline(4.0 / 9.0, color="k", lw=0.9, ls="-.",
                    label=r"Buchdahl bound $4/9$")
        ax3b = ax3.twinx()
        ax3b.plot(M[good], z[good], color=C_RGB, lw=lw, ls="--",
                  label=r"surface redshift $z$")
        ax3b.set_ylabel(r"surface redshift $z$")
        ax3.set_title("Compactness and gravitational redshift")
        handles = ax3.get_lines() + ax3b.get_lines()
    else:
        ax3.axhline(0.1, color="k", lw=0.9, ls="-.",
                    label=r"$GM/Rc^2=0.1$: Newtonian gravity in trouble")
        ax3.set_title("Where the Newtonian description breaks down")
        handles = ax3.get_lines()
    ax3.legend(handles, [h.get_label() for h in handles],
               fontsize=7.5, loc="upper left", framealpha=0.85)

    lines = [fr"$R={s['R_at_Mmax']:.2f}$ km",
             fr"$\rho_c={s['rho_at_Mmax']:.2e}$ kg m$^{{-3}}$"]
    if gr:
        lines.append(fr"$z_{{\rm surf}}={s['z_at_Mmax']:.3f}$")
    lines.append(fr"$c_s/c={s['cs_over_c_max_branch']:.3f}$ peak"
                 + ("" if s["causal"] else "  (acausal!)"))
    if not turned:
        lines.append("no turning point found")
    ax2.text(0.03, 0.97, "\n".join(lines), transform=ax2.transAxes,
             ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                       ec="gray", alpha=0.9))

    _finish(fig, outdir, f"sev_nsmr_{s['eos']}", dpi)
