"""
driver_gl.py
============
Glue between the command line and the physics / plot modules.

Each mode builds a figure once.  ``--interactive`` only means “open a
window”; there are no sliders.  Students change a run by passing
``--beta_x`` / ``--beta_y`` on the command line.  Headless tests pass
``show=False`` and an ``outdir``.
"""

import math
import os

import numpy as np

import physics_gl as phys
import plot_gl as plotting

MODES = ("rays", "point", "blob", "critical", "shear", "arcs", "kappa")


def _show_wanted(show, interactive):
    if os.environ.get("MPLBACKEND", "").lower() == "agg":
        return False
    return bool(show) or bool(interactive)


def _resolved_grid(theta_e, fov_arcsec, n_pix, extras, scale_arcsec):
    fov_arcsec = phys.adapted_fov_arcsec(theta_e, fov_arcsec, extras=extras)
    n_pix = phys.adapted_n_pix(n_pix, fov_arcsec, scale_arcsec)
    theta_x, theta_y = phys.make_grid(n_pix=n_pix, fov_arcsec=fov_arcsec)
    return theta_x, theta_y, fov_arcsec, n_pix


def _point_mass_grid_args(theta_e, beta_x, beta_y, source_scale, contain_radius):
    """FOV extras and the smaller image-plane scale for a point-mass run."""
    beta = math.hypot(beta_x, beta_y)
    plus, minus = phys.point_mass_image_radii(beta, theta_e)
    far_p, far_m = phys.point_mass_mapped_radii(beta, theta_e, contain_radius)
    s_plus = phys.point_mass_image_plane_scale(plus, theta_e, source_scale)
    s_minus = phys.point_mass_image_plane_scale(minus, theta_e, source_scale)
    extras = (beta_x, beta_y, plus, minus, far_p, far_m, contain_radius)
    return extras, min(s_plus, s_minus)


def run_rays(beta_arcsec=0.35, outdir=None, dpi=140,
             show=True, interactive=False):
    # Forward thin-lens rays.  Heights are angles times schematic
    # distances D_l = D_ls = 1, D_s = 2, so the y-axis is in radians.
    # beta_arcsec is signed: positive is above the axis.
    theta_e = phys.default_point_mass_theta_e(12.0)
    on = phys.forward_point_mass_bundle(0.0, theta_e)
    beta = phys.arcsec_to_rad(beta_arcsec)
    off = phys.forward_point_mass_bundle(beta, theta_e)
    return plotting.plot_rays(on, off, outdir=outdir, dpi=dpi,
                              show=_show_wanted(show, interactive))


def run_point(log10_m=12.0, beta_x_arcsec=0.50, beta_y_arcsec=0.00,
              n_pix=181, fov_arcsec=6.0, outdir=None, dpi=140,
              show=True, interactive=False):
    theta_e = phys.default_point_mass_theta_e(log10_m)
    beta_x = phys.arcsec_to_rad(beta_x_arcsec)
    beta_y = phys.arcsec_to_rad(beta_y_arcsec)
    sigma = phys.arcsec_to_rad(phys.COMPACT_SOURCE_SIGMA_ARCSEC)
    extras, scale = _point_mass_grid_args(
        theta_e, beta_x, beta_y, sigma, 3.0 * sigma,
    )
    theta_x, theta_y, fov_arcsec, n_pix = _resolved_grid(
        theta_e, fov_arcsec, n_pix,
        extras=extras,
        scale_arcsec=phys.COMPACT_SOURCE_SIGMA_ARCSEC,
    )
    phys.require_resolved_point_mass_images(
        theta_e, math.hypot(beta_x, beta_y), sigma, fov_arcsec, n_pix,
    )
    image = phys.render_point_mass_source(theta_x, theta_y, theta_e,
                                          beta_x, beta_y, sigma)
    title = (rf"Point-mass lens (false colour)  |  "
             rf"$\beta=({beta_x_arcsec:.2f},{beta_y_arcsec:.2f})''$")
    return plotting.plot_point(image, theta_x, theta_y, theta_e,
                               beta_x, beta_y, outdir=outdir, dpi=dpi,
                               show=_show_wanted(show, interactive),
                               title=title,
                               extra_prov={
                                   "fov_arcsec": f"{fov_arcsec:.4f}",
                                   "n_pix": str(n_pix),
                                   "sigma_src_arcsec":
                                       f"{phys.COMPACT_SOURCE_SIGMA_ARCSEC:.4f}",
                               })


def run_blob(log10_m=12.0, beta_x_arcsec=0.15, beta_y_arcsec=0.05,
             r_eff_arcsec=0.25, q=0.60, phi_deg=30.0,
             n_pix=181, fov_arcsec=6.0, outdir=None, dpi=140,
             show=True, interactive=False):
    theta_e = phys.default_point_mass_theta_e(log10_m)
    beta_x = phys.arcsec_to_rad(beta_x_arcsec)
    beta_y = phys.arcsec_to_rad(beta_y_arcsec)
    r_eff = phys.arcsec_to_rad(r_eff_arcsec)
    extras, scale = _point_mass_grid_args(
        theta_e, beta_x, beta_y, r_eff * q, r_eff,
    )
    theta_x, theta_y, fov_arcsec, n_pix = _resolved_grid(
        theta_e, fov_arcsec, n_pix,
        extras=extras,
        scale_arcsec=max(r_eff_arcsec * q, 0.05),
    )
    image = phys.render_point_mass_extended(
        theta_x, theta_y, theta_e,
        beta_x, beta_y, r_eff, q, np.deg2rad(phi_deg),
    )
    return plotting.plot_point(
        image, theta_x, theta_y, theta_e,
        beta_x,
        beta_y,
        outdir=outdir, dpi=dpi,
        show=_show_wanted(show, interactive),
        title="Extended source on a point-mass lens (false colour)",
        mode="blob",
        extra_prov={
            "r_eff_arcsec": f"{r_eff_arcsec:.4f}",
            "q": f"{q:.4f}",
            "phi_deg": f"{phi_deg:.2f}",
            "n_pix": str(n_pix),
        },
    )


def run_critical(log10_m=12.0, beta_x_arcsec=0.00, beta_y_arcsec=0.00,
                 n_pix=181, fov_arcsec=6.0,
                 outdir=None, dpi=140, show=True, interactive=False):
    theta_e = phys.default_point_mass_theta_e(log10_m)
    beta_x = phys.arcsec_to_rad(beta_x_arcsec)
    beta_y = phys.arcsec_to_rad(beta_y_arcsec)
    sigma = phys.arcsec_to_rad(phys.COMPACT_SOURCE_SIGMA_ARCSEC)
    extras, scale = _point_mass_grid_args(
        theta_e, beta_x, beta_y, sigma, 3.0 * sigma,
    )
    theta_x, theta_y, fov_arcsec, n_pix = _resolved_grid(
        theta_e, fov_arcsec, n_pix,
        extras=extras,
        scale_arcsec=phys.COMPACT_SOURCE_SIGMA_ARCSEC,
    )
    phys.require_resolved_point_mass_images(
        theta_e, math.hypot(beta_x, beta_y), sigma, fov_arcsec, n_pix,
    )
    image = phys.render_point_mass_source(theta_x, theta_y, theta_e,
                                          beta_x, beta_y, sigma)
    det_a = phys.jacobian_det_point_mass(theta_x, theta_y, theta_e)
    # Side view is a vertical slice.  Height is beta_y; beta_x is
    # perpendicular to that page.
    bundle = phys.forward_point_mass_bundle(beta_y, theta_e)
    return plotting.plot_critical(image, det_a, theta_x, theta_y, theta_e,
                                  beta_x, beta_y, bundle,
                                  outdir=outdir, dpi=dpi,
                                  show=_show_wanted(show, interactive))


def run_shear(sigma_v_kms=300.0, gamma=0.25,
              beta_x_arcsec=0.05, beta_y_arcsec=0.03,
              outdir=None, dpi=140, show=True, interactive=False):
    theta_e = phys.default_sis_theta_e(sigma_v_kms)
    gamma = float(gamma)
    crit_x, crit_y = phys.critical_curve_sis_shear(theta_e, gamma)
    cau_x, cau_y = phys.caustic_from_critical(crit_x, crit_y, theta_e, gamma)
    images = phys.images_sis_shear(phys.arcsec_to_rad(beta_x_arcsec),
                                   phys.arcsec_to_rad(beta_y_arcsec),
                                   theta_e, gamma)
    return plotting.plot_shear(
        crit_x, crit_y, cau_x, cau_y, images,
        phys.arcsec_to_rad(beta_x_arcsec),
        phys.arcsec_to_rad(beta_y_arcsec),
        theta_e, gamma=gamma, outdir=outdir, dpi=dpi,
        show=_show_wanted(show, interactive),
    )


def run_arcs(sigma_v_kms=300.0, gamma=0.25,
             beta_x_arcsec=0.18, beta_y_arcsec=0.00,
             r_eff_arcsec=0.25, q=0.60, phi_deg=30.0,
             n_pix=181, fov_arcsec=6.0,
             outdir=None, dpi=140, show=True, interactive=False):
    theta_e = phys.default_sis_theta_e(sigma_v_kms)
    beta_x = phys.arcsec_to_rad(beta_x_arcsec)
    beta_y = phys.arcsec_to_rad(beta_y_arcsec)
    r_eff = phys.arcsec_to_rad(r_eff_arcsec)
    r_crit_max = theta_e / (1.0 - abs(gamma))
    theta_x, theta_y, fov_arcsec, n_pix = _resolved_grid(
        theta_e, fov_arcsec, n_pix,
        extras=(r_crit_max + r_eff,
                math.hypot(beta_x, beta_y) + r_eff,
                r_eff),
        scale_arcsec=r_eff_arcsec * q,
    )
    image = phys.render_sis_shear_extended(
        theta_x, theta_y, theta_e, gamma,
        beta_x, beta_y, r_eff, q, np.deg2rad(phi_deg),
    )
    crit_x, crit_y = phys.critical_curve_sis_shear(theta_e, gamma)
    return plotting.plot_arcs(image, crit_x, crit_y, theta_x, theta_y,
                              outdir=outdir, dpi=dpi,
                              show=_show_wanted(show, interactive),
                              extra_prov={
                                  "gamma": f"{float(gamma):.6f}",
                                  "sigma_v_kms": f"{sigma_v_kms:.4f}",
                                  "theta_e_arcsec":
                                      f"{float(phys.rad_to_arcsec(theta_e)):.6f}",
                                  "beta_x_arcsec": f"{beta_x_arcsec:.6f}",
                                  "beta_y_arcsec": f"{beta_y_arcsec:.6f}",
                                  "r_eff_arcsec": f"{r_eff_arcsec:.4f}",
                                  "q": f"{q:.4f}",
                                  "phi_deg": f"{phi_deg:.2f}",
                              })


def run_kappa(sigma_v_kms=300.0, n_pix=181, fov_arcsec=6.0,
              outdir=None, dpi=140, show=True, interactive=False):
    theta_e = phys.default_sis_theta_e(sigma_v_kms)
    te_as = float(phys.rad_to_arcsec(theta_e))
    theta_x, theta_y, fov_arcsec, n_pix = _resolved_grid(
        theta_e, fov_arcsec, n_pix,
        extras=(),
        scale_arcsec=max(te_as / 8.0, 0.05),
    )
    kappa = phys.convergence_sis(theta_x, theta_y, theta_e)
    return plotting.plot_kappa(kappa, theta_x, theta_y, theta_e,
                               outdir=outdir, dpi=dpi,
                               show=_show_wanted(show, interactive),
                               extra_prov={"sigma_v_kms": f"{sigma_v_kms:.4f}"})


RUNNERS = {
    "rays": run_rays,
    "point": run_point,
    "blob": run_blob,
    "critical": run_critical,
    "shear": run_shear,
    "arcs": run_arcs,
    "kappa": run_kappa,
}


def run(mode, **kwargs):
    if mode not in RUNNERS:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    return RUNNERS[mode](**kwargs)
