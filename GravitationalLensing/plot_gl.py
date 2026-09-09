"""
plot_gl.py
==========
Matplotlib figures for GravitationalLensing.

One drawing routine per mode.  Each routine returns the figure.  When an
output directory is supplied it also writes a timestamped PNG and a
same-stem `.provenance.txt` sidecar.  There are no sliders; source
position is a command-line argument.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import physics_gl as phys

C_RING = "#00bcd4"
C_CRIT = "#c1121f"
C_CAUS = "#0077b6"
C_SRC = "#6a4c93"


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


def _finish(fig, outdir, prefix, dpi, provenance=None):
    footer = (f"GravitationalLensing {phys.MODEL_VERSION} "
              f"(build {phys.BUILD_ID})")
    fig.text(0.99, 0.01, footer, ha="right", va="bottom",
             fontsize=7, color="0.35")
    saved = None
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        path = _unique_stem(outdir, _timestamp_name(prefix))
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        sidecar = os.path.splitext(path)[0] + ".provenance.txt"
        lines = [
            "GravitationalLensing provenance",
            f"    model_version = {phys.MODEL_VERSION}",
            f"    build_id = {phys.BUILD_ID}",
            f"    prefix = {prefix}",
            f"    dpi = {dpi}",
        ]
        if provenance:
            for key, value in provenance.items():
                lines.append(f"    {key} = {value}")
        with open(sidecar, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        saved = path
    return saved


def _extent_arcsec(theta_x, theta_y):
    return [
        float(phys.rad_to_arcsec(theta_x.min())),
        float(phys.rad_to_arcsec(theta_x.max())),
        float(phys.rad_to_arcsec(theta_y.min())),
        float(phys.rad_to_arcsec(theta_y.max())),
    ]


def _draw_forward_bundle(ax, bundle, title):
    """One panel: forward rays from a single source.  No observer artwork."""
    axis_color = "#888888"
    d_ls = bundle["d_ls"]
    d_s = bundle["d_s"]
    y_src = bundle["y_src"]
    theta_e = bundle["theta_e"]

    ax.axhline(0.0, color=axis_color, lw=0.8, zorder=1)
    ax.axvline(d_ls, color=axis_color, lw=0.8, zorder=1)

    for ray in bundle["misses"]:
        pts = ray["points"]
        ax.plot(pts[:, 0], pts[:, 1], color="#0000ff", lw=1.15, zorder=2)
    for ray in bundle["hits"]:
        pts = ray["points"]
        ax.plot(pts[:, 0], pts[:, 1], color="#e10600", lw=1.35, zorder=3)

    ax.plot(0.0, y_src, marker="*", color="#6a4c93", ms=11, zorder=5)
    ax.plot(d_ls, 0.0, marker="o", color="k", ms=6, zorder=5)
    ax.plot(d_s, 0.0, marker="s", color="k", ms=6, zorder=5)
    ax.text(d_ls, 0.0, "  lens", fontsize=8, va="bottom", color="0.25")
    ax.text(d_s, 0.0, "  observer", fontsize=8, va="bottom", color="0.25")

    y_pad = 3.4 * theta_e * bundle["d_l"]
    ax.set_xlim(-0.02 * d_s, 1.02 * d_s)
    ax.set_ylim(-y_pad, y_pad)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("height (schematic)")


def plot_rays(bundle_on, bundle_off, outdir=None, dpi=140, show=True):
    """Two-panel forward-ray figure with the thin-lens deflection."""
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), sharex=True)
    _draw_forward_bundle(
        axes[0], bundle_on,
        "On axis: only the Einstein cone hits the observer",
    )
    if abs(bundle_off["beta"]) < 1.0e-16 * max(bundle_off["theta_e"], 1.0e-16):
        off_title = "On axis: the two red rays are a slice of the Einstein cone"
    else:
        off_title = "Off axis: two rays (two images) reach the observer"
    _draw_forward_bundle(axes[1], bundle_off, off_title)
    axes[1].set_xlabel("distance along the line of sight (schematic)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "rays", dpi,
                    provenance={
                        "mode": "rays",
                        "style": "forward-thin-lens",
                        "theta_e_rad": f"{bundle_on['theta_e']:.8e}",
                        "beta_off_rad": f"{bundle_off['beta']:.8e}",
                    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_point(image, theta_x, theta_y, theta_e, beta_x, beta_y,
               outdir=None, dpi=140, show=True, title=None, mode="point",
               extra_prov=None):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    im = ax.imshow(image, origin="lower", cmap="inferno",
                   extent=_extent_arcsec(theta_x, theta_y))
    te = float(phys.rad_to_arcsec(theta_e))
    ax.add_patch(Circle((0.0, 0.0), te, fill=False, color=C_RING, lw=1.4,
                        ls="--", label=rf"$\theta_E={te:.2f}''$"))
    ax.plot(phys.rad_to_arcsec(beta_x), phys.rad_to_arcsec(beta_y),
            marker="+", color="white", ms=10, mew=1.6,
            label="source (unlensed)")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_x$ [arcsec]")
    ax.set_ylabel(r"$\theta_y$ [arcsec]")
    ax.set_title(title or "Point-mass lens (false colour)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="intensity (false colour)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    prov = {
        "mode": mode,
        "theta_e_arcsec": f"{te:.6f}",
        "beta_x_arcsec": f"{float(phys.rad_to_arcsec(beta_x)):.6f}",
        "beta_y_arcsec": f"{float(phys.rad_to_arcsec(beta_y)):.6f}",
        "n_pix": str(image.shape[0]),
        "fov_arcsec": f"{(_extent_arcsec(theta_x, theta_y)[1]
                          - _extent_arcsec(theta_x, theta_y)[0]):.4f}",
    }
    if extra_prov:
        prov.update(extra_prov)
    saved = _finish(fig, outdir, mode, dpi, provenance=prov)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_critical(image, det_a, theta_x, theta_y, theta_e,
                  beta_x, beta_y, bundle, outdir=None, dpi=140, show=True,
                  extra_prov=None):
    """Image plane, source plane (caustic = a point), and side-view rays."""
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.4))
    ext = _extent_arcsec(theta_x, theta_y)
    te = float(phys.rad_to_arcsec(theta_e))

    ax = axes[0]
    ax.imshow(image, origin="lower", cmap="inferno", extent=ext)
    ax.contour(phys.rad_to_arcsec(theta_x), phys.rad_to_arcsec(theta_y),
               det_a, levels=[0.0], colors=C_RING, linewidths=1.6)
    ax.set_aspect("equal")
    ax.set_title("Image plane (false colour)\ncritical curve = Einstein ring")
    ax.set_xlabel(r"$\theta_x$ [arcsec]")
    ax.set_ylabel(r"$\theta_y$ [arcsec]")

    ax = axes[1]
    ax.set_aspect("equal")
    bx_a = float(phys.rad_to_arcsec(beta_x))
    by_a = float(phys.rad_to_arcsec(beta_y))
    span = max(1.4 * te, 1.3 * abs(bx_a), 1.3 * abs(by_a), 0.5)
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.plot(0.0, 0.0, marker="*", color=C_CAUS, ms=16, zorder=5)
    ax.plot(bx_a, by_a, marker="+", color=C_SRC, ms=12, mew=1.6, zorder=6)
    ax.set_title("Source plane\ncaustic = one point (symbol enlarged)")
    ax.set_xlabel(r"$\beta_x$ [arcsec]")
    ax.set_ylabel(r"$\beta_y$ [arcsec]")
    ax.axhline(0.0, color="0.85", lw=0.7)
    ax.axvline(0.0, color="0.85", lw=0.7)

    ax = axes[2]
    d_s = bundle["d_s"]
    d_ls = bundle["d_ls"]
    theta_e_b = bundle["theta_e"]
    ax.axhline(0.0, color="#888888", lw=0.8, zorder=1)
    ax.axvline(d_ls, color="#888888", lw=0.8, zorder=1)
    for ray in bundle.get("misses", []):
        pts = ray["points"]
        ax.plot(pts[:, 0], pts[:, 1], color="#0000ff", lw=1.05, zorder=2)
    for ray in bundle.get("hits", []):
        pts = ray["points"]
        ax.plot(pts[:, 0], pts[:, 1], color="#e10600", lw=1.25, zorder=3)
    ax.plot(0.0, bundle["y_src"], marker="*", color=C_SRC, ms=10, zorder=5)
    ax.plot(d_ls, 0.0, marker="o", color="k", ms=6, zorder=5)
    ax.plot(d_s, 0.0, marker="s", color="k", ms=6, zorder=5)
    y_pad = max(3.4 * theta_e_b * bundle["d_l"], 1.3 * abs(bundle["y_src"]))
    ax.set_xlim(-0.02 * d_s, 1.02 * d_s)
    ax.set_ylim(-y_pad, y_pad)
    ax.set_box_aspect(0.72)
    ax.set_title("Side view\nheight is " + r"$\beta_y$")
    ax.set_xlabel("line of sight")
    ax.set_ylabel("height (schematic)")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "critical", dpi,
                    provenance={
                        "mode": "critical",
                        "theta_e_arcsec": f"{te:.6f}",
                        "beta_x_arcsec": f"{float(phys.rad_to_arcsec(beta_x)):.6f}",
                        "beta_y_arcsec": f"{float(phys.rad_to_arcsec(beta_y)):.6f}",
                        "n_pix": str(image.shape[0]),
                        "fov_arcsec": f"{(ext[1] - ext[0]):.4f}",
                        "sigma_src_arcsec":
                            f"{phys.COMPACT_SOURCE_SIGMA_ARCSEC:.4f}",
                        **(extra_prov or {}),
                    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_shear(crit_x, crit_y, cau_x, cau_y, images, beta_x, beta_y, theta_e,
               gamma=0.25, outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0))
    te = float(phys.rad_to_arcsec(theta_e))
    ring = phys.is_einstein_ring_case(beta_x, beta_y, gamma, theta_e)
    pseudo = float(phys.rad_to_arcsec(phys.pseudo_caustic_radius_sis_shear(theta_e)))

    ax = axes[0]
    ax.plot(phys.rad_to_arcsec(crit_x), phys.rad_to_arcsec(crit_y),
            color=C_CRIT, lw=2.0, label="critical curve")
    ax.plot(0.0, 0.0, "k.", ms=5)
    if ring:
        ax.add_patch(Circle((0.0, 0.0), te, fill=False, color=C_RING, lw=2.0,
                            label="Einstein ring"))
        title = "Image plane  (Einstein ring)"
    else:
        colors = ["#1d3557", "#e63939", "#2a9d8f", "#f4a261", "#9b5de5"]
        for k, (ix, iy) in enumerate(images):
            ax.plot(phys.rad_to_arcsec(ix), phys.rad_to_arcsec(iy), "o",
                    color=colors[k % len(colors)], ms=8, zorder=5)
            ax.text(phys.rad_to_arcsec(ix) + 0.04, phys.rad_to_arcsec(iy) + 0.04,
                    str(k + 1), color=colors[k % len(colors)], fontsize=8)
        title = f"Image plane  ({len(images)} image(s))"
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(r"$\theta_x$ [arcsec]")
    ax.set_ylabel(r"$\theta_y$ [arcsec]")
    ax.legend(loc="upper right", fontsize=8)
    img_span = [te]
    if images:
        img_span.extend(abs(phys.rad_to_arcsec(v)) for ix, iy in images for v in (ix, iy))
    img_span.extend(abs(phys.rad_to_arcsec(v)) for v in list(crit_x) + list(crit_y))
    lim = max(img_span) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax = axes[1]
    ax.plot(phys.rad_to_arcsec(cau_x), phys.rad_to_arcsec(cau_y),
            color=C_CAUS, lw=2.0, label="tangential caustic")
    ax.add_patch(Circle((0.0, 0.0), pseudo, fill=False, color="0.45",
                        lw=1.2, ls="--", label="pseudo-caustic"))
    ax.plot(phys.rad_to_arcsec(beta_x), phys.rad_to_arcsec(beta_y),
            marker="*", color=C_SRC, ms=16, markeredgecolor="k",
            markeredgewidth=0.4, zorder=6, label="source")
    ax.set_aspect("equal")
    ax.set_title("Source plane")
    ax.set_xlabel(r"$\beta_x$ [arcsec]")
    ax.set_ylabel(r"$\beta_y$ [arcsec]")
    ax.legend(loc="upper right", fontsize=8)
    slim = max(pseudo,
               float(np.max(np.abs(phys.rad_to_arcsec(cau_x)))),
               float(np.max(np.abs(phys.rad_to_arcsec(cau_y)))),
               abs(float(phys.rad_to_arcsec(beta_x))),
               abs(float(phys.rad_to_arcsec(beta_y))),
               0.3) * 1.35
    ax.set_xlim(-slim, slim)
    ax.set_ylim(-slim, slim)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "shear", dpi,
                    provenance={
                        "mode": "shear",
                        "n_images": "ring" if ring else str(len(images)),
                        "theta_e_arcsec": f"{te:.6f}",
                        "gamma": f"{float(gamma):.6f}",
                        "beta_x_arcsec": f"{float(phys.rad_to_arcsec(beta_x)):.6f}",
                        "beta_y_arcsec": f"{float(phys.rad_to_arcsec(beta_y)):.6f}",
                    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_arcs(image, crit_x, crit_y, theta_x, theta_y, outdir=None,
              dpi=140, show=True,
              title="Extended source on an SIS + shear lens (false colour)",
              extra_prov=None):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    im = ax.imshow(image, origin="lower", cmap="inferno",
                   extent=_extent_arcsec(theta_x, theta_y))
    ax.plot(phys.rad_to_arcsec(crit_x), phys.rad_to_arcsec(crit_y),
            color=C_RING, lw=1.2, alpha=0.85)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(r"$\theta_x$ [arcsec]")
    ax.set_ylabel(r"$\theta_y$ [arcsec]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="intensity (false colour)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "arcs", dpi,
                    provenance={
                        "mode": "arcs",
                        "n_pix": str(image.shape[0]),
                        "fov_arcsec": f"{_extent_arcsec(theta_x, theta_y)[1]
                                         - _extent_arcsec(theta_x, theta_y)[0]:.4f}",
                        **(extra_prov or {}),
                    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_kappa(kappa, theta_x, theta_y, theta_e, outdir=None, dpi=140, show=True,
               extra_prov=None):
    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    positive = np.maximum(kappa, np.nanmax(kappa) * 1.0e-6)
    im = ax.imshow(positive, origin="lower", cmap="viridis",
                   extent=_extent_arcsec(theta_x, theta_y),
                   norm=LogNorm(vmin=positive.min(), vmax=positive.max()))
    te = float(phys.rad_to_arcsec(theta_e))
    ax.add_patch(Circle((0.0, 0.0), te, fill=False, color=C_CRIT, lw=2.0,
                        label=rf"$\kappa=1/2$  (critical curve, $\theta_E={te:.2f}''$)"))
    ax.set_aspect("equal")
    ax.set_title(r"SIS convergence $\kappa$ (false colour)")
    ax.set_xlabel(r"$\theta_x$ [arcsec]")
    ax.set_ylabel(r"$\theta_y$ [arcsec]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\kappa$")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "kappa", dpi,
                    provenance={
                        "mode": "kappa",
                        "theta_e_arcsec": f"{te:.6f}",
                        "n_pix": str(kappa.shape[0]),
                        "fov_arcsec": f"{_extent_arcsec(theta_x, theta_y)[1]
                                         - _extent_arcsec(theta_x, theta_y)[0]:.4f}",
                        **(extra_prov or {}),
                    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved
