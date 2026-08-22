"""
plot_photon.py — PhotonOrbit

Matplotlib visualization of photon trajectories around a Schwarzschild black hole.

The figure is always displayed on screen. When an output directory is
supplied via `outdir`, a timestamped PNG copy is *additionally* written
there before the interactive window is shown; --outdir augments the
on-screen display, it does not replace it.
"""

import math
import os
from datetime import datetime

import matplotlib.pyplot as plt


def _timestamp_name(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"


def _finish(fig, outdir, prefix, dpi):
    """Optionally save a timestamped PNG, then display the figure."""
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, _timestamp_name(prefix))
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_photon] PNG saved -> {path}")
    print("[plot_photon] Displaying figure on screen ...")
    plt.show()
    plt.close(fig)


def plot_photon_orbit(x_values, y_values, b, info, outdir=None, dpi=150, lw=1.5):
    """
    Plot a photon trajectory, event horizon, photon sphere, and diagnostics.

    Parameters
    ----------
    outdir : str or None
        If given, also save a timestamped PNG in this folder (created if
        necessary), in addition to showing the figure on screen.
    dpi : int
        Resolution of the saved PNG. Ignored when outdir is None.
    lw : float
        Line width, in points, used for the plotted trajectory.
    """
    if not x_values or len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must be nonempty arrays of equal length.")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for values in (x_values, y_values)
        for value in values
    ):
        raise ValueError("x_values and y_values must contain only finite real numbers.")
    if not isinstance(b, (int, float)) or isinstance(b, bool) or not math.isfinite(b):
        raise ValueError("b must be a finite real number.")
    if not isinstance(info, dict):
        raise ValueError("info must be a diagnostics dictionary.")

    required_info = {"r_s", "r_photon", "status", "closest_approach", "delta_phi"}
    missing = required_info.difference(info)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"info is missing required diagnostic field(s): {names}.")

    for key in ("r_s", "r_photon", "closest_approach", "delta_phi"):
        value = info[key]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"info['{key}'] must be a finite real number.")
    if info["r_s"] <= 0.0 or info["r_photon"] <= 0.0:
        raise ValueError("r_s and r_photon must be greater than zero.")
    if not isinstance(info["status"], str):
        raise ValueError("info['status'] must be a string.")

    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if not isinstance(lw, (int, float)) or isinstance(lw, bool) or not math.isfinite(lw) or lw <= 0.0:
        raise ValueError("lw must be a finite positive number.")

    r_s = info["r_s"]
    r_photon = info["r_photon"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", "box")
    ax.plot(x_values, y_values, lw=lw, label="Photon trajectory")

    horizon = plt.Circle((0.0, 0.0), r_s, alpha=0.3, label="Event horizon")
    ax.add_patch(horizon)
    sphere = plt.Circle(
        (0.0, 0.0), r_photon, alpha=0.6,
        fill=False, linestyle="--", label="Photon sphere",
    )
    ax.add_patch(sphere)

    ax.set_xlabel("x (same length units as GM/c^2)")
    ax.set_ylabel("y (same length units as GM/c^2)")
    ax.set_title(f"Photon orbit around a Schwarzschild black hole (b = {b:.4g})")
    summary = (
        f"status: {info['status']}\n"
        f"closest r: {info['closest_approach']:.5g}\n"
        f"Δφ: {info['delta_phi']:.5g} rad"
    )
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, va="bottom")
    ax.legend(loc="upper right")
    ax.grid(True)

    prefix = f"photon_b{b:.4g}_{info['status']}"
    _finish(fig, outdir, prefix, dpi)
