"""
driver_bhs.py
=============
Glue between the command line and the physics / plot modules.

Each mode builds a figure once.  ``--interactive`` only means “open a
window”; there are no sliders.  Students change a run by passing flags
on the command line.  Headless tests pass ``show=False`` and an ``outdir``.
"""

import os

import physics_bhs as phys
import plot_bhs as plotting

MODES = ("rays", "pixels", "capture", "ring", "radii", "weak", "compare", "backlight")


def _show_wanted(show, interactive):
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        return False
    return bool(show) or bool(interactive)


def _print_summary(mode, extra_lines):
    width = 66
    sep = "-" * width
    print(sep)
    print(f"  BlackHoleShadow {phys.MODEL_VERSION} "
          f"(build {phys.BUILD_ID}) -- {mode}")
    print(sep)
    for line in extra_lines:
        print(f"  {line}")
    print(sep)


def run_rays(M=1.0, r_cam=None, b_in=5.0, b_out=6.0,
             lambda_max=200.0, d_lambda=0.02,
             outdir=None, dpi=140, show=True, interactive=False):
    """Beat 0: two PhotonOrbit-class rays, one captured and one escaped."""
    M = phys.validate_user_value("M", M, positive=True)
    r_cam = phys.R_CAM_DEFAULT if r_cam is None else r_cam
    r_cam = phys.validate_user_value("r_cam", r_cam, positive=True)
    info_in = phys.integrate_photon_orbit(M, r_cam, b_in, lambda_max, d_lambda)
    info_out = phys.integrate_photon_orbit(M, r_cam, b_out, lambda_max, d_lambda)
    _print_summary("rays", [
        f"M                      : {M:.6g}",
        f"r_cam                  : {r_cam:.6g}",
        f"b_in  (expect capture) : {b_in:.6g}  -> {info_in[2]['status']}",
        f"b_out (expect escape)  : {b_out:.6g}  -> {info_out[2]['status']}",
        f"b_crit                 : {phys.critical_impact_parameter(M):.6g}",
        f"Delta phi in / out     : {info_in[2]['delta_phi']:.6g} / "
        f"{info_out[2]['delta_phi']:.6g} rad",
    ])
    return plotting.plot_rays(
        info_in, info_out, M=M, r_cam=r_cam, b_in=b_in, b_out=b_out,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_pixels(M=1.0, r_cam=None, b_in=5.0, b_out=6.0,
               n_pix=21, fov_M=16.0,
               outdir=None, dpi=140, show=True, interactive=False):
    """Beat 1: the same two b values as two pixels in a coarse camera."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix if n_pix is not None else 21)
    # A coarse teaching grid; do not enforce the dense-map shadow contract.
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M)
    captured = phys.capture_map(bx, by, M)
    _print_summary("pixels", [
        f"M                      : {M:.6g}",
        f"b_in / b_out           : {b_in:.6g} / {b_out:.6g}",
        f"b_crit                 : {phys.critical_impact_parameter(M):.6g}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        f"fov                    : {fov_M:.6g} M",
    ])
    return plotting.plot_pixels(
        bx, by, captured, M=M, b_in=b_in, b_out=b_out,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_capture(M=1.0, n_pix=161, fov_M=16.0,
                outdir=None, dpi=140, show=True, interactive=False):
    """Beat 2: dense capture map.  Dark disk of radius b_crit, not r_s."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    phys.validate_user_value("fov_M", fov_M, positive=True)
    phys.require_resolved_shadow(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M)
    captured = phys.capture_map(bx, by, M)
    b_crit = phys.critical_impact_parameter(M)
    n_dark = int(captured.sum())
    _print_summary("capture", [
        f"M                      : {M:.6g}",
        f"b_crit                 : {b_crit:.6g}",
        f"r_s                    : {phys.event_horizon(M):.6g}",
        f"r_photon               : {phys.photon_sphere(M):.6g}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        f"captured pixels        : {n_dark} / {captured.size}",
    ])
    return plotting.plot_capture(
        bx, by, captured, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_ring(M=1.0, n_pix=161, fov_M=16.0,
             outdir=None, dpi=140, show=True, interactive=False):
    """Beat 3: capture map plus the high-winding photon-ring highlighter."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    phys.require_resolved_shadow(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M)
    captured = phys.capture_map(bx, by, M)
    ring = phys.photon_ring_mask(bx, by, M)
    _print_summary("ring", [
        f"M                      : {M:.6g}",
        f"b_crit                 : {phys.critical_impact_parameter(M):.6g}",
        f"photon-ring pixels     : {int(ring.sum())}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
    ])
    return plotting.plot_ring(
        bx, by, captured, ring, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_radii(M=1.0, n_pix=161, fov_M=16.0,
              outdir=None, dpi=140, show=True, interactive=False):
    """Beat 4: three circles, three meanings — r_s, r_photon, b_crit."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    phys.require_resolved_shadow(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M)
    captured = phys.capture_map(bx, by, M)
    _print_summary("radii", [
        f"M                      : {M:.6g}",
        f"r_s / M                : {phys.event_horizon(M) / M:.6g}",
        f"r_photon / M           : {phys.photon_sphere(M) / M:.6g}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        "These are three different circles.  Only b_crit is the shadow rim.",
    ])
    return plotting.plot_radii(
        bx, by, captured, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_weak(M=1.0, outdir=None, dpi=140, show=True, interactive=False):
    """Beat 5: exact deflection versus 4M/b."""
    M = phys.validate_user_value("M", M, positive=True)
    bs, exact, weak = phys.deflection_curve(M)
    # A couple of checkpoints for the console.
    b_far = float(bs[-1])
    b_near = float(bs[0])
    _print_summary("weak", [
        f"M                      : {M:.6g}",
        f"b_crit                 : {phys.critical_impact_parameter(M):.6g}",
        f"hat alpha (b={b_far:.3g} M) exact/weak : "
        f"{exact[-1]:.5g} / {weak[-1]:.5g} rad",
        f"hat alpha (b={b_near:.3g} M) exact/weak : "
        f"{exact[0]:.5g} / {weak[0]:.5g} rad",
        "The weak-field formula is the large-b limit of the same rays.",
    ])
    return plotting.plot_weak(
        bs, exact, weak, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_compare(M=1.0, logM=12.0,
                outdir=None, dpi=140, show=True, interactive=False):
    """Beat 6: galaxy Einstein radius versus this hole's b_crit."""
    M = phys.validate_user_value("M", M, positive=True)
    logM = phys.validate_user_value("logM", logM)
    numbers = phys.compare_rings(log10_m_galaxy=logM)
    _print_summary("compare", [
        f"galaxy log10(M/M_sun)  : {logM:.6g}",
        f"theta_E                : {numbers['theta_e_arcsec']:.4g} arcsec",
        f"R_E / (GM/c^2)_galaxy  : {numbers['r_e_over_M']:.6g}",
        f"b_crit / M (this hole) : {numbers['b_crit_over_M']:.6g}",
        f"r_photon / M           : {numbers['r_photon_over_M']:.6g}",
        "These are not the same ring.",
    ])
    return plotting.plot_compare(
        numbers, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_backlight(M=1.0, n_pix=161, fov_M=16.0,
                  r_hot=None, inclination=60.0,
                  outdir=None, dpi=140, show=True, interactive=False):
    """Beat 7: thin equatorial backlight.  Ring on a dark disk, no Kerr."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    phys.require_resolved_shadow(M, fov_M, n_pix)
    r_hot = phys.HOT_RING_R_DEFAULT * M if r_hot is None else r_hot
    r_hot = phys.validate_user_value("r_hot", r_hot, positive=True)
    inclination = phys.validate_user_value("inclination", inclination)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M)
    image, b_hot = phys.backlight_image(
        bx, by, M, r_hot=r_hot, inclination_deg=inclination,
    )
    _print_summary("backlight", [
        f"M                      : {M:.6g}",
        f"r_hot                  : {r_hot:.6g}",
        f"b_hot (primary image)  : {b_hot:.6g}",
        f"inclination            : {inclination:.6g} deg  (display shading)",
        "False colour is a display scale.  The shadow is not a rainbow.",
    ])
    return plotting.plot_backlight(
        bx, by, image, M=M, r_hot=r_hot, b_hot=b_hot,
        inclination=inclination,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


RUNNERS = {
    "rays": run_rays,
    "pixels": run_pixels,
    "capture": run_capture,
    "ring": run_ring,
    "radii": run_radii,
    "weak": run_weak,
    "compare": run_compare,
    "backlight": run_backlight,
}


def run(mode, **kwargs):
    if mode not in RUNNERS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    return RUNNERS[mode](**kwargs)
