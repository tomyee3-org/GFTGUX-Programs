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


def _base_stem(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _reserve_unique_stem(outdir, prefix):
    """
    Reserve a filename stem in `outdir` for which neither "<stem>.png"
    nor "<stem>.provenance.txt" already exists, and return
    (stem, png_path, sidecar_path). The PNG path is left behind as an
    empty, exclusively-created placeholder that fig.savefig() then
    overwrites with real image data.

    Two runs with the same impact parameter and outcome status (for
    example the repeated d_lambda convergence runs of EXP-9) that both
    land in the same wall-clock second would otherwise collide on a plain
    timestamped name; appending "_2", "_3", ... resolves that.

    The PNG path itself is reserved atomically with
    os.open(..., O_CREAT|O_EXCL): a single syscall that either creates the
    file because no other process got there first, or fails with
    FileExistsError because one did, with no window in between where two
    concurrent PhotonOrbit processes could both observe the same candidate
    name as free. That guarantee covers the PNG stem only -- the sidecar
    path is checked with a plain existence test just above the atomic PNG
    reservation, not reserved by the same atomic call, so the PNG and its
    sidecar are not a single atomic unit; see _finish() below for what
    that means in practice.
    """
    base = _base_stem(prefix)
    n = None
    while True:
        stem = base if n is None else f"{base}_{n}"
        png_path = os.path.join(outdir, f"{stem}.png")
        sidecar_path = os.path.join(outdir, f"{stem}.provenance.txt")
        if not os.path.exists(sidecar_path):
            try:
                fd = os.open(png_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                pass
            else:
                return stem, png_path, sidecar_path
        n = 2 if n is None else n + 1


def _format_provenance(b, info, dpi, lw, GM_over_c2, r0, lambda_max, d_lambda):
    """
    Build the text written to a saved run's ``.provenance.txt`` sidecar.

    A saved PNG's on-figure annotation states only b, status, closest
    approach and delta_phi -- not GM_over_c2, r0, lambda_max, d_lambda,
    dpi, lw, or the program's version/build -- so the PNG alone cannot be
    used to independently reconstruct the run that produced it. Every
    value below is written with Python's repr() (the shortest string that
    round-trips back to the exact same float), not the console's
    human-readable .6g/.4g formatting, so copying a value out of this
    file and back into the command line reproduces the exact physics
    inputs, including numerically sensitive near-separatrix cases where a
    six-significant-digit rounding of b changes the outcome. The
    "Reproduce with:" command line includes the physics parameters and
    PhotonOrbit's own two rendering options (dpi, lw); it omits --outdir,
    since where to save a new copy is a per-invocation choice rather than
    part of what produced this run. This reproduces the same trajectory
    at the same resolution and line width, but NOT necessarily a byte-
    identical PNG: Matplotlib's own defaults (fonts, colors, backend,
    version) are not fixed or recorded here, and can vary between
    environments.
    """
    def line(name, value):
        if value is None:
            return f"    {name} = (not provided to plot_photon_orbit)"
        return f"    {name} = {value!r}"

    lines = [
        f"PhotonOrbit {info.get('model_version', '?')} "
        f"(build {info.get('build_id', '?')})",
        "",
        "Physics parameters (repr precision; safe to copy back as CLI flags):",
        line("GM_over_c2", GM_over_c2),
        line("r0", r0),
        line("b", b),
        line("lambda_max", lambda_max),
        line("d_lambda", d_lambda),
        "",
        "Rendering parameters:",
        f"    dpi = {dpi!r}",
        f"    lw  = {lw!r}",
        "",
        "Outcome:",
    ]
    for key in ("status", "closest_approach", "delta_phi", "lambda_final", "steps"):
        if key in info:
            lines.append(f"    {key} = {info[key]!r}")
    lines.append("")
    if None not in (GM_over_c2, r0, lambda_max, d_lambda):
        lines.append("Reproduce with:")
        lines.append(
            "    python main.py --GM_over_c2 {0!r} --r0 {1!r} --b {2!r} "
            "--lambda_max {3!r} --d_lambda {4!r} --lw {5!r} --dpi {6!r}".format(
                GM_over_c2, r0, b, lambda_max, d_lambda, lw, dpi
            )
        )
    else:
        lines.append(
            "(Reproduce-with command omitted: this figure was produced by a "
            "direct call to plot_photon_orbit() that did not supply "
            "GM_over_c2/r0/lambda_max/d_lambda.)"
        )
    return "\n".join(lines) + "\n"


def _finish(fig, outdir, prefix, dpi, provenance=None):
    """Optionally save a timestamped PNG (and its provenance sidecar),
    then display the figure.

    If the process is interrupted between _reserve_unique_stem()
    reserving the empty placeholder PNG and fig.savefig() completing (or
    between that and the sidecar write), the placeholder can be left
    behind as a zero-byte file, or a completed PNG can be left without
    its sidecar. This is a narrow, low-risk residue of an abnormal
    termination, not a collision or data-corruption issue -- a later run
    simply reserves a different stem -- and is not cleaned up
    automatically.
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        _, png_path, sidecar_path = _reserve_unique_stem(outdir, prefix)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        print(f"[plot_photon] PNG saved -> {png_path}")
        if provenance is not None:
            with open(sidecar_path, "w", encoding="utf-8") as sidecar:
                sidecar.write(provenance)
            print(f"[plot_photon] Provenance saved -> {sidecar_path}")
    print("[plot_photon] Displaying figure on screen ...")
    plt.show()
    plt.close(fig)


def plot_photon_orbit(x_values, y_values, b, info, outdir=None, dpi=150, lw=1.5,
                       GM_over_c2=None, r0=None, lambda_max=None, d_lambda=None):
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
    GM_over_c2, r0, lambda_max, d_lambda : float or None
        The physics parameters that produced x_values/y_values/info.
        Optional -- a direct caller that only has the trajectory and
        diagnostics can omit them -- but when given (as driver_photon.py
        always does) and outdir is also given, they are written losslessly
        into a ``.provenance.txt`` sidecar next to the saved PNG, so the
        run can be exactly reproduced later.
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
    provenance = _format_provenance(b, info, dpi, lw, GM_over_c2, r0, lambda_max, d_lambda)
    _finish(fig, outdir, prefix, dpi, provenance=provenance)
