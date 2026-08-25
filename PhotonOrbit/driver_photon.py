"""
driver_photon.py — PhotonOrbit

Driver module for setting parameters and calling the physics and plot modules.
"""

import math

import physics_photon as phys
from plot_photon import plot_photon_orbit


def _validate_plot_inputs(dpi, lw):
    if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or not math.isfinite(dpi):
        raise ValueError("dpi must be a finite number.")
    if int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if not isinstance(lw, (int, float)) or isinstance(lw, bool) or not math.isfinite(lw):
        raise ValueError("lw must be a finite number.")
    if lw <= 0.0:
        raise ValueError("lw must be greater than zero.")
    return int(dpi), lw


def _print_summary(GM_over_c2, r0, b, lambda_max, info):
    W = 62
    sep = "-" * W
    print(sep)
    print(
        f"  PhotonOrbit {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID}) -- Run Summary"
    )
    print(sep)
    print(f"  GM_over_c2 (M)       : {GM_over_c2:.6g}")
    print(f"  Starting radius r0   : {r0:.6g}")
    print(f"  Impact parameter b   : {b:.6g}")
    print(f"  Event horizon r_s    : {info['r_s']:.6g}")
    print(f"  Photon sphere r_ph   : {info['r_photon']:.6g}")
    print(f"  Critical b (b_crit)  : {info['critical_b_infinity']:.6g}")
    print(f"  Status               : {info['status']}")
    print(f"  Closest approach     : {info['closest_approach']:.6g}")
    print(f"  Delta phi            : {info['delta_phi']:.6g} rad")
    print(f"  Affine parameter     : {info['lambda_final']:.6g}  (of {lambda_max:.6g} requested)")
    print(f"  RK4 steps taken      : {info['steps']:,}")
    print(sep)


def driver_photon_orbit(
    GM_over_c2=1.0,
    r0=20.0,
    b=5.0,
    lambda_max=200.0,
    d_lambda=0.01,
    outdir=None,
    dpi=150,
    lw=1.5,
):
    """
    Run one photon-orbit calculation, print a run summary, and display the
    result. If `outdir` is given, also save a timestamped PNG there, in
    addition to the on-screen display.
    """
    dpi, lw = _validate_plot_inputs(dpi, lw)

    x_vals, y_vals, info = phys.integrate_photon_orbit(
        GM_over_c2=GM_over_c2,
        r0=r0,
        b=b,
        lambda_max=lambda_max,
        d_lambda=d_lambda,
    )
    _print_summary(GM_over_c2, r0, b, lambda_max, info)
    plot_photon_orbit(x_vals, y_vals, b, info, outdir=outdir, dpi=dpi, lw=lw)
    return info
