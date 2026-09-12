"""
driver_bhs.py
=============
Glue between the command line and the physics / plot modules.

Each mode builds a figure once.  ``--interactive`` only means “open a
window”; there are no sliders.  Students change a run by passing flags
on the command line.  Headless tests pass ``show=False`` and an ``outdir``.

Length arguments arriving here are *multiples of M* (r_cam/M, b/M,
fov/M, r_hot/M).  Absolute lengths are formed only at the physics
boundary.
"""

import os

import physics_bhs as phys
import plot_bhs as plotting

MODES = ("rays", "pixels", "capture", "ring", "transfer", "image",
         "radii", "weak", "compare")


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


def _ratios_to_abs(M, r_cam_over_M, b_in_over_M, b_out_over_M):
    M = phys.validate_user_value("M", M, positive=True)
    if r_cam_over_M is None:
        r_cam_over_M = phys.R_CAM_DEFAULT
    r_cam_over_M = phys.validate_user_value("r_cam", r_cam_over_M, positive=True)
    b_in_over_M = phys.validate_user_value("b_in", b_in_over_M, min_value=0.0)
    b_out_over_M = phys.validate_user_value("b_out", b_out_over_M, min_value=0.0)
    r_cam = phys.to_absolute_length(r_cam_over_M, M, "r_cam")
    b_in = phys.to_absolute_length(b_in_over_M, M, "b_in")
    b_out = phys.to_absolute_length(b_out_over_M, M, "b_out")
    phys.require_camera_outside_photon_sphere(r_cam, M)
    return M, r_cam_over_M, r_cam, b_in_over_M, b_in, b_out_over_M, b_out


def run_rays(M=1.0, r_cam=None, b_in=5.0, b_out=6.0,
             lambda_max=200.0, d_lambda=0.02,
             outdir=None, dpi=140, show=True, interactive=False):
    """Beat 0: two PhotonOrbit-class rays."""
    (M, r_cam_over_M, r_cam_abs, b_in_over_M, b_in_abs,
     b_out_over_M, b_out_abs) = _ratios_to_abs(M, r_cam, b_in, b_out)
    lam_over_M = phys.validate_user_value("lambda_max", lambda_max, positive=True)
    dlam_over_M = phys.validate_user_value("d_lambda", d_lambda, positive=True)
    lam_abs = phys.to_absolute_length(lam_over_M, M, "lambda_max")
    dlam_abs = phys.to_absolute_length(dlam_over_M, M, "d_lambda")
    info_in = phys.integrate_photon_orbit(M, r_cam_abs, b_in_abs,
                                          lam_abs, dlam_abs)
    info_out = phys.integrate_photon_orbit(M, r_cam_abs, b_out_abs,
                                           lam_abs, dlam_abs)
    _print_summary("rays", [
        f"M                      : {M:.6g}",
        f"r_cam                  : {r_cam_over_M:.6g} M  ({r_cam_abs:.6g})",
        f"b_in / M               : {b_in_over_M:.6g}  -> {info_in[2]['status']}",
        f"b_out / M              : {b_out_over_M:.6g}  -> {info_out[2]['status']}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        f"lambda_max / M         : {lam_over_M:.6g}",
        f"d_lambda / M           : {dlam_over_M:.6g}",
        f"Delta phi in / out     : {info_in[2]['delta_phi']:.6g} / "
        f"{info_out[2]['delta_phi']:.6g} rad",
        "Delta phi is accumulated azimuth to a finite stopping surface,",
        "not the asymptotic deflection hat{alpha}.",
    ])
    return plotting.plot_rays(
        info_in, info_out, M=M, r_cam=r_cam_abs,
        b_in=b_in_abs, b_out=b_out_abs,
        lambda_max=lam_abs, d_lambda=dlam_abs,
        lambda_max_over_M=lam_over_M, d_lambda_over_M=dlam_over_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_pixels(M=1.0, b_in=5.0, b_out=6.0,
               n_pix=None, fov_M=16.0, n_pix_requested=None,
               outdir=None, dpi=140, show=True, interactive=False):
    """Beat 1: two impact parameters on a coarse camera grid."""
    M = phys.validate_user_value("M", M, positive=True)
    if n_pix is None:
        n_pix = phys.PIXELS_N_PIX_DEFAULT
    n_pix = phys.odd_n_pix(n_pix)
    if n_pix > phys.PIXELS_N_PIX_MAX:
        n_pix = phys.odd_n_pix(phys.PIXELS_N_PIX_MAX)
    clamped = (n_pix_requested is not None
               and int(n_pix_requested) > phys.PIXELS_N_PIX_MAX)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M, M)
    captured = phys.capture_map(bx, by, M)
    b_in_abs = phys.to_absolute_length(b_in, M, "b_in")
    b_out_abs = phys.to_absolute_length(b_out, M, "b_out")
    lines = [
        f"M                      : {M:.6g}",
        f"b_in / M               : {b_in:.6g}  -> "
        f"{'captured' if phys.is_captured(b_in_abs, M) else 'escaped'}",
        f"b_out / M              : {b_out:.6g}  -> "
        f"{'captured' if phys.is_captured(b_out_abs, M) else 'escaped'}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        f"fov                    : {fov_M:.6g} M",
    ]
    if clamped:
        lines.append(
            f"note                   : pixels mode caps --n_pix at "
            f"{phys.PIXELS_N_PIX_MAX} so the coarse camera stays coarse "
            f"(requested {int(n_pix_requested)})."
        )
    _print_summary("pixels", lines)
    return plotting.plot_pixels(
        bx, by, captured, M=M, b_in=b_in_abs, b_out=b_out_abs,
        n_pix_requested=n_pix_requested, fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_capture(M=1.0, n_pix=161, fov_M=16.0,
                outdir=None, dpi=140, show=True, interactive=False):
    """Beat 2: geometric capture map.  Dark disk of radius b_crit."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    phys.validate_user_value("fov", fov_M, positive=True)
    fov_M, need, _ = phys.require_shadow_in_frame(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M, M)
    captured = phys.capture_map(bx, by, M)
    b_crit = phys.critical_impact_parameter(M)
    _print_summary("capture", [
        f"M                      : {M:.6g}",
        f"b_crit / M             : {b_crit / M:.6g}",
        f"r_s / M                : {phys.event_horizon(M) / M:.6g}",
        f"r_photon / M           : {phys.photon_sphere(M) / M:.6g}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        f"fov                    : {fov_M:.6g} M  (min to contain shadow {need:.4g} M)",
        f"captured pixels        : {int(captured.sum())} / {captured.size}",
        "Geometric capture map.  A distant spherical screen is assumed",
        "when this dark disk is called a shadow.",
    ])
    return plotting.plot_capture(
        bx, by, captured, M=M, fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_ring(M=1.0, n_pix=161, fov_M=16.0,
             outdir=None, dpi=140, show=True, interactive=False):
    """Beat 3: capture map plus a high-winding-ray overlay."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    b_lo, b_hi = phys.high_winding_b_window(M)
    extra = b_hi / M
    fov_M, need, _ = phys.require_shadow_in_frame(M, fov_M, n_pix,
                                                 extra_radius_over_M=extra)
    phys.require_resolved_high_winding(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M, M)
    captured = phys.capture_map(bx, by, M)
    ring = phys.photon_ring_mask(bx, by, M)
    _print_summary("ring", [
        f"M                      : {M:.6g}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        f"high-winding window    : ({b_lo / M:.6g}, {b_hi / M:.6g}] M",
        f"overlay pixels         : {int(ring.sum())}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        "Overlay = strongly wound escapers, not a photon-ring profile.",
    ])
    return plotting.plot_ring(
        bx, by, captured, ring, M=M, b_lo=b_lo, b_hi=b_hi, fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_radii(M=1.0, n_pix=161, fov_M=16.0,
              outdir=None, dpi=140, show=True, interactive=False):
    """Beat 6: coordinate radii versus the image-plane critical curve."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    fov_M, need, _ = phys.require_shadow_in_frame(M, fov_M, n_pix)
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M, M)
    captured = phys.capture_map(bx, by, M)
    _print_summary("radii", [
        f"M                      : {M:.6g}",
        f"r_s / M                : {phys.event_horizon(M) / M:.6g}",
        f"r_photon / M           : {phys.photon_sphere(M) / M:.6g}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        "Left: Schwarzschild coordinate-radius cross-section.",
        "Right: image-plane critical curve (the camera's capture rim).",
    ])
    return plotting.plot_radii(
        bx, by, captured, M=M, fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_weak(M=1.0, outdir=None, dpi=140, show=True, interactive=False):
    """Beat 7: exact deflection versus 4M/b."""
    M = phys.validate_user_value("M", M, positive=True)
    bs, exact, weak = phys.deflection_curve(M)
    _print_summary("weak", [
        f"M                      : {M:.6g}",
        f"b_crit / M             : {phys.critical_impact_parameter(M) / M:.6g}",
        f"hat alpha (b={bs[-1] / M:.3g} M) exact/weak : "
        f"{exact[-1]:.5g} / {weak[-1]:.5g} rad",
        f"hat alpha (b={bs[0] / M:.3g} M) exact/weak : "
        f"{exact[0]:.5g} / {weak[0]:.5g} rad",
        "The weak-field formula is the large-b limit of the same rays.",
        "The divergence as b -> b_crit is strong deflection, not a ring.",
    ])
    return plotting.plot_weak(
        bs, exact, weak, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_compare(M=1.0, logM=12.0,
                outdir=None, dpi=140, show=True, interactive=False):
    """Beat 8: galaxy Einstein radius versus this hole's critical curve."""
    M = phys.validate_user_value("M", M, positive=True)
    logM = phys.validate_user_value(
        "logM", logM, min_value=phys.LOGM_MIN, max_value=phys.LOGM_MAX,
    )
    numbers = phys.compare_rings(log10_m_galaxy=logM)
    _print_summary("compare", [
        f"galaxy log10(M/M_sun)  : {logM:.6g}",
        f"theta_E                : {numbers['theta_e_arcsec']:.4g} arcsec",
        f"R_E / (GM/c^2)_galaxy  : {numbers['r_e_over_M']:.6g}",
        f"b_crit / M (this hole) : {numbers['b_crit_over_M']:.6g}",
        f"r_photon / M           : {numbers['r_photon_over_M']:.6g}",
        "Einstein critical curve and Schwarzschild critical curve",
        "are not the same ring.",
        f"(--M = {M:.6g} is unused here; the comparison is dimensionless.)",
    ])
    return plotting.plot_compare(
        numbers, M=M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_transfer(M=1.0, n_pix=161, fov_M=16.0, r_hot=None,
                 outdir=None, dpi=140, show=True, interactive=False):
    """Beat 4: face-on transfer functions r_m(b)."""
    M = phys.validate_user_value("M", M, positive=True)
    if r_hot is None:
        r_hot_over_M = phys.HOT_RING_R_DEFAULT
    else:
        r_hot_over_M = phys.validate_user_value("r_hot", r_hot, positive=True)
    r_hot_abs = phys.to_absolute_length(r_hot_over_M, M, "r_hot")
    b_max_over_M = max(0.5 * fov_M, 1.6 * r_hot_over_M, 8.0)
    bs, table, counts, b_crit = phys.transfer_curves(
        M, b_max_over_M=b_max_over_M, max_m=3,
    )
    n_r3 = 0
    if table.shape[1] > 2:
        n_r3 = int(sum(1 for v in table[:, 2] if v == v))
    peaks = phys.source_crossing_impacts(M, r_hot_abs, max_m=3)
    _print_summary("transfer", [
        f"M                      : {M:.6g}",
        f"r_hot                  : {r_hot_over_M:.6g} M",
        f"I_em width             : {phys.HOT_RING_WIDTH_DEFAULT:.6g} M",
        f"b_crit / M             : {b_crit / M:.6g}",
        f"adaptive b samples     : {bs.size}",
        f"finite r_3 samples     : {n_r3}",
        f"b(r_m=r_hot)/M         : "
        + ", ".join("—" if p is None else f"{p / M:.5g}" for p in peaks),
        "Static face-on emitters.  No Doppler.  m>=4 omitted here.",
    ])
    return plotting.plot_transfer(
        bs, table, counts, M=M, r_hot_over_M=r_hot_over_M,
        width_over_M=phys.HOT_RING_WIDTH_DEFAULT, peaks=peaks,
        fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


def run_image(M=1.0, n_pix=161, fov_M=16.0, r_hot=None,
              outdir=None, dpi=140, show=True, interactive=False):
    """Beat 5: direct, +lensing, +photon-ring contributions."""
    M = phys.validate_user_value("M", M, positive=True)
    n_pix = phys.odd_n_pix(n_pix)
    if r_hot is None:
        r_hot_over_M = phys.HOT_RING_R_DEFAULT
    else:
        r_hot_over_M = phys.validate_user_value("r_hot", r_hot, positive=True)
    r_hot_abs = phys.to_absolute_length(r_hot_over_M, M, "r_hot")
    extra = max(r_hot_over_M * 1.4, phys.critical_impact_parameter(M) / M * 1.4)
    fov_M, need, _ = phys.require_shadow_in_frame(
        M, fov_M, n_pix, extra_radius_over_M=extra,
    )
    bx, by, n_pix, fov_M = phys.make_impact_grid(n_pix, fov_M, M)
    img1, img2, img3, sample, parts, b_hot = phys.disk_image_components(
        bx, by, M, r_hot=r_hot_abs,
    )
    photon_inc = float((img3 - img2).max())
    _print_summary("image", [
        f"M                      : {M:.6g}",
        f"r_hot                  : {r_hot_over_M:.6g} M",
        f"I_em width             : {phys.HOT_RING_WIDTH_DEFAULT:.6g} M",
        f"b_hot (periapsis) / M  : {b_hot / M:.6g}",
        f"n_pix x n_pix          : {n_pix} x {n_pix}",
        f"fov                    : {fov_M:.6g} M",
        f"adaptive I(b) samples  : {sample.size}",
        f"max m>=3 increment     : {photon_inc:.4g}",
        "I_obs = sum_{m=1..4} g^4 I_em;  m>=5 omitted.",
        "g=sqrt(1-2M/r) for static emitters (no orbital Doppler).",
        "Third panel = photon-ring/subring crossings m=3 and m=4.",
        "Captured means the future endpoint is the horizon; the ray",
        "may still cross the disk on the way in.",
    ])
    return plotting.plot_image(
        bx, by, img1, img2, img3, M=M, r_hot=r_hot_abs, b_hot=b_hot,
        fov_over_M=fov_M,
        outdir=outdir, dpi=dpi, show=_show_wanted(show, interactive),
    )


RUNNERS = {
    "rays": run_rays,
    "pixels": run_pixels,
    "capture": run_capture,
    "ring": run_ring,
    "transfer": run_transfer,
    "image": run_image,
    "radii": run_radii,
    "weak": run_weak,
    "compare": run_compare,
}


def run(mode, **kwargs):
    if mode not in RUNNERS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    return RUNNERS[mode](**kwargs)
