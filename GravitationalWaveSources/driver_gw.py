"""
driver_gw.py
============
Orchestration layer for GravitationalWaveSources.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np
import physics_gw as phys
import plot_gw as viz

#: Conservative upper bound on --dpi. The figure is a fixed 12x9 inches, so
#: pixel area grows as dpi^2; an unbounded dpi lets a single typo (or a
#: deliberately adversarial value) request a multi-hundred-megabyte to
#: multi-gigabyte in-memory image before any useful error can be produced.
#: 600 dpi already exceeds anything these instructional plots need (a
#: 7200x5400 pixel PNG), so this is not expected to constrain any normal use.
MAX_DPI = 600


def _finite_number(name, value):
    """Return value as float after giving a consistent user-facing error.

    ``bool`` (and ``numpy.bool_``) are rejected explicitly even though
    ``float(True) == 1.0`` would otherwise convert silently -- see the
    matching guard in ``physics_gw._require_finite`` for the rationale.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a finite number; got {value!r} (a bool is not "
            "an accepted numeric value)."
        )
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # See the matching comment in physics_gw._require_finite: a Python
        # int too large to represent as a float is a value problem for this
        # caller, not a programming error.
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def version_info():
    """Return the program's model version and build identifier.

    Exposed at the driver layer (mirroring the physics-layer constants) so
    callers and tests can query provenance without reaching into
    ``physics_gw`` directly.
    """
    return {"model_version": phys.MODEL_VERSION, "build_id": phys.BUILD_ID}


def _validate_plot_inputs(t_before, t_after, lw, dpi):
    lw = _finite_number("lw", lw)
    dpi = _finite_number("dpi", dpi)
    if lw <= 0:
        raise ValueError("lw must be greater than zero.")
    if int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if dpi > MAX_DPI:
        raise ValueError(
            f"dpi must not exceed {MAX_DPI}; the figure is a fixed 12x9in, "
            f"so a larger value requests an excessively large PNG. Reduce "
            "--dpi."
        )

    normalized_zoom = []
    for name, value in (("t_before", t_before), ("t_after", t_after)):
        if value is None:
            normalized_zoom.append(None)
        else:
            value = _finite_number(name, value)
            if value < 0:
                raise ValueError(f"{name} must be None or a finite non-negative number.")
            normalized_zoom.append(value)

    return normalized_zoom[0], normalized_zoom[1], lw, int(dpi)


def _timestamp_fname(prefix="gw_inspiral", ext="csv"):
    # Microsecond resolution avoids two rapid saves in the same directory
    # (e.g. a scripted parameter sweep) silently colliding and overwriting
    # each other under the previous second-resolution timestamp.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"


def _write_csv(result, csvdir):
    """Save t, f, A, h as a plain tabular CSV (blank cells where the
    ringdown segment leaves f/A undefined -- see integrate_inspiral).

    This gives students a documented, no-programming-required route to the
    numerical arrays behind the plot -- e.g. for the chirp-mass-extraction
    exercise (estimate df/dt between two nearby rows, then invert with
    physics_gw.chirp_mass_from_fdot()) or for EXP-8's fixed-time RK4
    convergence comparison -- without requiring every student to write a
    Python import snippet.
    """
    os.makedirs(csvdir, exist_ok=True)
    fpath = os.path.join(csvdir, _timestamp_fname())
    with open(fpath, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "f_hz", "A", "h"])
        for t, f, A, h in zip(result["t"], result["f"], result["A"], result["h"]):
            writer.writerow([
                f"{float(v):.17g}" if math.isfinite(v) else ""
                for v in (t, f, A, h)
            ])
    print(f"[driver_gw] CSV saved -> {fpath}")
    return fpath


def run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
        dt=2e-4, f_start=20.0,
        outdir=None, csvdir=None,
        t_before=None, t_after=None, lw=0.4,
        include_ringdown=False,
        n_ringdown_tau=6, ringdown_pts=4000,
        dpi=150):
    """Run the full calculation, print diagnostics, and render the figure."""
    t_before, t_after, lw, dpi = _validate_plot_inputs(
        t_before, t_after, lw, dpi
    )

    print("[driver_gw] Integrating inspiral ...")
    result = phys.integrate_inspiral(
        m1_msun, m2_msun, d_mpc,
        dt=dt,
        f_start=f_start,
        include_ringdown=include_ringdown,
        n_ringdown_tau=n_ringdown_tau,
        ringdown_pts=ringdown_pts,
    )

    _print_summary(result["summary"])

    if csvdir is not None:
        _write_csv(result, csvdir)

    print("[driver_gw] Rendering figure ...")
    viz.plot_inspiral(
        result,
        outdir=outdir,
        t_before=t_before,
        t_after=t_after,
        lw=lw,
        dpi=dpi,
    )
    return result


def _print_summary(s):
    W = 62
    sep = "─" * W
    print(sep)
    print(
        f"  GravitationalWaveSources {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID}) — Run Summary"
    )
    print(sep)
    print(f"  Masses              : {s['m1_msun']:.3f} + {s['m2_msun']:.3f}  M☉")
    print(f"  Total mass          : {s['M_total_msun']:.3f}  M☉")
    print(f"  Chirp mass          : {s['Mc_msun']:.4f}  M☉")
    print(f"  Distance            : {s['d_mpc']:.1f}  Mpc")
    print(f"  Band start          : {s['f_start_hz']:.1f}  Hz")
    print(f"  ISCO cutoff         : {s['f_isco_hz']:.1f}  Hz")
    print(f"  Time to ISCO        : {s['T_band_s']:.3f}  s")
    print(f"  Strain scale at ISCO: {s['A_isco']:.3e}")
    print(f"  Inspiral samples    : {s['inspiral_steps']:,}")
    if s["include_ringdown"]:
        print("  Ringdown            : illustrative Schwarzschild QNM")
        print(f"  Final mass (toy est): {s['M_final_msun']:.3f}  M☉")
        print(f"  QNM frequency       : {s['f_qnm_hz']:.1f}  Hz")
        print(f"  QNM decay time      : {s['tau_qnm_ms']:.3f}  ms")
    else:
        print("  Ringdown            : not included")
    print(sep)
