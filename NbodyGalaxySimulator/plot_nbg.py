"""
plot_nbg.py
===========
Matplotlib visualisations for NbodyGalaxySimulator.

One routine per mode:

    plot_cluster   star-cluster evaporation: projected positions, Lagrangian
                   radii, virial ratio, instantaneous-unbound count and
                   near-escape tail fraction, energy conservation
    plot_galaxy    cold-collapse galaxy formation: projected positions,
                   Lagrangian radii (the collapse-and-bounce signature),
                   virial ratio, energy conservation
    plot_chaos     sensitivity to initial conditions: projected positions of
                   both realizations, divergence versus time (log scale)
                   with the exponential-fit window marked, energy
                   conservation for both realizations

Each routine displays the figure.  When an output directory is supplied,
it also writes a timestamped PNG into that directory before displaying
it, together with a same-stem `.provenance.txt` sidecar recording the
rendering settings and (via driver_nbg.run()) the scientific run
parameters -- see _finish() for the exact contract, which follows
plot_sev.py's.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import physics_nbg as phys

C_A = "#1f77b4"
C_B = "#d62728"
C_FIT = "#2ca02c"
C_REF = "#7f7f7f"
C_WARN = "#ff7f0e"

FRACTION_COLORS = {
    0.1: "#08306b", 0.25: "#2171b5", 0.5: "#1f77b4",
    0.75: "#6baed6", 0.9: "#c6dbef",
}


# ----------------------------------------------------------------------
# Output helpers (mirrors plot_sev.py's _finish/_unique_stem contract)
# ----------------------------------------------------------------------
def _timestamp_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


def _unique_stem(outdir, name):
    stem, ext = os.path.splitext(name)

    def _taken(candidate_stem):
        png = os.path.join(outdir, candidate_stem + ext)
        sidecar = os.path.join(outdir, candidate_stem + ".provenance.txt")
        return os.path.exists(png) or os.path.exists(sidecar)

    if not _taken(stem):
        return os.path.join(outdir, stem + ext)
    n = 2
    while _taken(f"{stem}_{n}"):
        n += 1
    return os.path.join(outdir, f"{stem}_{n}{ext}")


def _finish(fig, outdir, prefix, dpi, lw=None, figsize=None, provenance=None):
    """
    Optionally save a timestamped PNG (with a provenance sidecar), then
    display the figure. See plot_sev.py's _finish() for the full
    rationale; the contract here is identical.
    """
    fig.text(0.995, 0.005,
              f"NbodyGalaxySimulator {phys.MODEL_VERSION} "
              f"(build {phys.BUILD_ID})",
              ha="right", va="bottom", fontsize=6, color="0.55")
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        path = _unique_stem(outdir, _timestamp_name(prefix))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_nbg] PNG saved -> {path}")
        stem, _ = os.path.splitext(path)
        sidecar = stem + ".provenance.txt"
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(f"Provenance for {os.path.basename(path)}\n")
            f.write("(rendering parameters are always recorded below; "
                    "scientific run parameters are recorded whenever the "
                    "caller supplied them)\n\n")
            f.write("rendering parameters:\n")
            f.write(f"    dpi = {dpi}\n")
            f.write(f"    lw = {lw}\n")
            if figsize is not None:
                f.write(f"    figsize_inches = {figsize[0]:g}, {figsize[1]:g}\n\n")
            else:
                f.write("    figsize_inches = unknown\n\n")
            if provenance:
                f.write("scientific run parameters:\n")
                f.write("\n".join(provenance) + "\n")
            else:
                f.write(
                    "scientific run parameters: not supplied to this call.  "
                    "This normally means plot_nbg.py was called directly as "
                    "a Python API rather than through the documented CLI.\n"
                )
    plt.show()
    plt.close(fig)


def _scatter_projection(ax, positions, color, label, alpha=0.75, s=8):
    """Scatter the x-y projection of a snapshot. Axis framing is the
    caller's job (see _finalize_scatter_axes) so that two populations
    sharing one Axes are framed once, consistently, rather than fighting
    each other's aspect/limit settings."""
    x = positions[:, 0] / phys.PC
    y = positions[:, 1] / phys.PC
    ax.scatter(x, y, s=s, c=color, alpha=alpha, label=label, linewidths=0)
    ax.set_xlabel("x  [pc]")
    ax.set_ylabel("y  [pc]")


def _finalize_scatter_axes(ax, position_sets, robust_zoom=False):
    """
    Set the final aspect ratio (and, if requested, a robust zoom) for an
    Axes carrying one or more _scatter_projection() calls.

    If ``robust_zoom`` is set, the axes are framed around the 95th-
    percentile radius of the COMBINED point cloud rather than its full
    extent. A violent collapse or a close encounter can eject a body to
    many times the system's own characteristic size (a real outcome, not
    an artifact); plotting every point but framing the axes on the bulk
    of the mass keeps that bulk visible instead of being collapsed into a
    speck by one distant outlier's axis range. Combining every population
    on this Axes before computing the limit (rather than letting each
    _scatter_projection call set its own) is what avoids two populations
    fighting over the Axes' aspect/limit settings.
    """
    all_xy = np.concatenate([p[:, :2] for p in position_sets], axis=0) / phys.PC
    r = np.hypot(all_xy[:, 0], all_xy[:, 1])
    zoomed = False
    if robust_zoom and r.size >= 8:
        r95 = np.percentile(r, 95)
        limit = 1.6 * max(r95, 1e-30)
        if limit < r.max():
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            zoomed = True
    # adjustable="datalim" (the usual choice for an equal-aspect scatter)
    # silently overrides any set_xlim/set_ylim just applied above by
    # resizing the DATA limits instead of the axes box; adjustable="box"
    # keeps the requested limits and resizes the box instead.
    ax.set_aspect("equal", adjustable="box" if zoomed else "datalim")


def _energy_panel(ax, t_myr, energy, label, color):
    e0 = energy[0]
    frac = (energy - e0) / e0 if e0 != 0.0 else np.zeros_like(energy)
    ax.plot(t_myr, frac, color=color, label=label)
    ax.axhline(0.0, color=C_REF, lw=0.8, ls=":")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"$(E(t)-E_0)/E_0$")


# ----------------------------------------------------------------------
# Cluster mode
# ----------------------------------------------------------------------
def plot_cluster(result, outdir=None, dpi=150, lw=1.6, provenance=None,
                  figsize=(13.0, 8.0)):
    s = result["summary"]
    t = result["t"] / phys.MYR
    lag = result["lagrangian_radii"] / phys.PC
    fractions = s["lagrangian_fractions"]

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    _scatter_projection(axes[0, 0], result["positions"][0], C_A, "initial")
    _finalize_scatter_axes(axes[0, 0], [result["positions"][0]])
    axes[0, 0].set_title(f"Initial positions (N={s['n_bodies']})")

    _scatter_projection(axes[0, 1], result["positions"][-1], C_B, "final")
    _finalize_scatter_axes(axes[0, 1], [result["positions"][-1]], robust_zoom=True)
    axes[0, 1].set_title(f"Final positions (t = {s['total_time_myr']:.3g} Myr)")

    ax = axes[0, 2]
    for j, f in enumerate(fractions):
        ax.plot(t, lag[:, j], label=f"{f:.0%}", lw=lw,
                color=FRACTION_COLORS.get(f, None))
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel("Lagrangian radius  [pc]")
    ax.set_title("Lagrangian radii")
    ax.legend(fontsize=7, title="mass fraction")

    ax = axes[1, 0]
    ax.plot(t, result["virial_ratio"], color=C_A, lw=lw)
    ax.axhline(1.0, color=C_REF, lw=0.8, ls=":")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"virial ratio  $2T/|W|$")
    ax.set_title("Virial ratio")

    ax = axes[1, 1]
    ax.plot(t, result["n_unbound"], color=C_WARN, lw=lw,
            label="instantaneously unbound")
    ax2 = ax.twinx()
    ax2.plot(t, np.asarray(result["high_velocity_fraction"]) * 100.0,
             color=C_A, lw=lw, ls="--", label="near-escape tail")
    ax.set_xlabel("time  [Myr]")
    # Labeled "instantaneously unbound", not "number escaped" -- this is
    # a snapshot-by-snapshot positive-specific-energy count, not a
    # cumulative, confirmed escape count (see identify_unbound()'s
    # docstring); it is not guaranteed to be monotonically increasing.
    ax.set_ylabel("instantaneously unbound", color=C_WARN)
    ax2.set_ylabel("near-escape tail  [%]", color=C_A)
    ax.set_title("Unbound-fraction diagnostics")

    _energy_panel(axes[1, 2], t, result["energy"], "total energy", C_A)
    axes[1, 2].set_title(f"Energy conservation (max drift "
                          f"{s['max_fractional_energy_drift']:.2%})")

    fig.suptitle("NbodyGalaxySimulator -- star-cluster evaporation "
                  f"(theta={s['theta']:.2f}, method={s['method']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, outdir, "cluster", dpi, lw=lw, figsize=figsize, provenance=provenance)


# ----------------------------------------------------------------------
# Galaxy mode
# ----------------------------------------------------------------------
def plot_galaxy(result, outdir=None, dpi=150, lw=1.6, provenance=None,
                 figsize=(13.0, 8.0)):
    s = result["summary"]
    t = result["t"] / phys.MYR
    lag = result["lagrangian_radii"] / phys.PC
    fractions = s["lagrangian_fractions"]
    i_collapse = int(np.argmin(np.abs(t - s["time_of_deepest_collapse_myr"])))

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    _scatter_projection(axes[0, 0], result["positions"][0], C_A, "initial")
    _finalize_scatter_axes(axes[0, 0], [result["positions"][0]])
    axes[0, 0].set_title(f"Initial positions (N={s['n_bodies']}, cold sphere)")

    _scatter_projection(axes[0, 1], result["positions"][i_collapse], C_WARN,
                         "deepest collapse")
    _finalize_scatter_axes(axes[0, 1], [result["positions"][i_collapse]])
    axes[0, 1].set_title(f"Deepest collapse (t = "
                          f"{s['time_of_deepest_collapse_myr']:.3g} Myr)")

    ax = axes[0, 2]
    for j, f in enumerate(fractions):
        ax.plot(t, lag[:, j], label=f"{f:.0%}", lw=lw,
                color=FRACTION_COLORS.get(f, None))
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel("Lagrangian radius  [pc]")
    ax.set_title("Lagrangian radii (collapse + bounce)")
    ax.legend(fontsize=7, title="mass fraction")

    _scatter_projection(axes[1, 0], result["positions"][-1], C_B, "final")
    _finalize_scatter_axes(axes[1, 0], [result["positions"][-1]], robust_zoom=True)
    # "quasi-equilibrium" describes the settled remnant this experiment
    # usually produces, not a guaranteed outcome for every parameter
    # choice -- label it that way only when the run's own final virial
    # ratio actually landed near scalar balance (2T/|W| = 1); otherwise
    # use a neutral title so the panel never claims a settled remnant it
    # did not observe.
    if abs(s["virial_ratio_final"] - 1.0) <= 0.3:
        axes[1, 0].set_title("Final (quasi-equilibrium) positions")
    else:
        axes[1, 0].set_title("Final positions")

    ax = axes[1, 1]
    ax.plot(t, result["virial_ratio"], color=C_A, lw=lw)
    ax.axhline(1.0, color=C_REF, lw=0.8, ls=":")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel(r"virial ratio  $2T/|W|$")
    ax.set_title("Virial ratio (violent relaxation)")

    _energy_panel(axes[1, 2], t, result["energy"], "total energy", C_A)
    axes[1, 2].set_title(f"Energy conservation (max drift "
                          f"{s['max_fractional_energy_drift']:.2%})")

    fig.suptitle("NbodyGalaxySimulator -- galaxy-formation cold collapse "
                  f"(theta={s['theta']:.2f}, method={s['method']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, outdir, "galaxy", dpi, lw=lw, figsize=figsize, provenance=provenance)


# ----------------------------------------------------------------------
# Chaos mode
# ----------------------------------------------------------------------
def plot_chaos(result, outdir=None, dpi=150, lw=1.6, provenance=None,
               figsize=(13.0, 8.0)):
    s = result["summary"]
    t = result["t"] / phys.MYR
    div = result["divergence"] / phys.PC

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    _scatter_projection(ax, result["positions_a"][-1], C_A, "realization A",
                         alpha=0.6)
    _scatter_projection(ax, result["positions_b"][-1], C_B, "realization B",
                         alpha=0.4)
    _finalize_scatter_axes(
        ax, [result["positions_a"][-1], result["positions_b"][-1]],
        robust_zoom=True,
    )
    ax.set_title(f"Final positions of both realizations (t = "
                 f"{s['total_time_myr']:.3g} Myr)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogy(t, np.maximum(div, 1e-300), color=C_A, lw=lw)
    lam = s["lyapunov_exponent_per_myr"]
    has_fit_overlay = False
    fit_lo = s.get("lyapunov_fit_start_index")
    fit_hi = s.get("lyapunov_fit_stop_index")
    if np.isfinite(lam) and lam > 0 and fit_lo is not None and fit_hi is not None:
        n_used = s["n_points_used_in_fit"]
        # Highlight the EXACT contiguous slice the estimator fit, not a
        # reconstructed amplitude-window mask -- the latter can include
        # points outside the single contiguous run actually used, which
        # would misrepresent what was fit whenever the divergence trace
        # re-enters the amplitude window after leaving it.
        ax.semilogy(t[fit_lo:fit_hi], div[fit_lo:fit_hi], color=C_FIT, lw=lw * 1.8,
                    label=f"fit window ({n_used} pts, "
                          f"$R^2$={s['lyapunov_fit_r_squared']:.4f})")
        has_fit_overlay = True
        ax.set_title(f"Divergence (Lyapunov time = "
                     f"{s['lyapunov_time_myr']:.3g} Myr = "
                     f"{s['lyapunov_time_over_t_cross']:.2g} $t_\\mathrm{{cross}}$)")
    else:
        ax.set_title("Divergence (no clean exponential window found)")
    ax.set_xlabel("time  [Myr]")
    ax.set_ylabel("RMS separation  [pc]")
    # The only labeled artist on this axis is the fit-window overlay
    # above, which is added only when a fit was found. Calling legend()
    # unconditionally would raise matplotlib's "No artists with labels
    # found to put in legend" UserWarning on every unsuccessful-fit run
    # whenever nothing is labeled -- an unexpected warning by this
    # project's own testing standard. Only call it when there is
    # something to show.
    if has_fit_overlay:
        ax.legend(fontsize=7)

    ax = axes[1, 0]
    _energy_panel(ax, t, result["energy_a"], "realization A", C_A)
    _energy_panel(ax, t, result["energy_b"], "realization B", C_B)
    ax.set_title(f"Energy conservation, both realizations (max drift "
                 f"{s['max_fractional_energy_drift']:.2%})")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    diff = (result["positions_a"][-1] - result["positions_b"][-1]) / phys.PC
    r = np.sqrt(np.sum(diff ** 2, axis=1))
    ax.hist(r, bins=min(30, max(5, result["masses"].size // 3)), color=C_A)
    ax.set_xlabel("final per-body separation  [pc]")
    ax.set_ylabel("number of bodies")
    ax.set_title("Final separation distribution")

    fig.suptitle("NbodyGalaxySimulator -- sensitivity to initial conditions "
                 f"(theta={s['theta']:.2f}, method={s['method']})")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _finish(fig, outdir, "chaos", dpi, lw=lw, figsize=figsize, provenance=provenance)


PLOTTERS = {"cluster": plot_cluster, "galaxy": plot_galaxy, "chaos": plot_chaos}


def plot_mode(mode, result, outdir=None, dpi=150, lw=1.6, provenance=None):
    if mode not in PLOTTERS:
        raise ValueError(f"mode must be one of {sorted(PLOTTERS)}; got {mode!r}.")
    return PLOTTERS[mode](result, outdir=outdir, dpi=dpi, lw=lw, provenance=provenance)
