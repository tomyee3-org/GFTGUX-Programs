"""
driver_gw.py
============
Orchestration layer for GravitationalWaveSources.
"""

import math
import numpy as np
import physics_gw as phys
import plot_gw as viz


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
    except (TypeError, ValueError) as exc:
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


def run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
        dt=2e-4, f_start=20.0,
        outdir=None,
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
