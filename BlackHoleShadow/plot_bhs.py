"""
plot_bhs.py
===========
Matplotlib figures for BlackHoleShadow.

One drawing routine per mode.  Each routine returns the figure.  When an
output directory is supplied it also writes a timestamped PNG and a
same-stem ``.provenance.txt`` sidecar.  There are no sliders.
"""

import math
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import physics_bhs as phys

C_RS = "#888888"
C_PH = "#e8a838"
C_BC = "#00bcd4"
C_CAP = "#1a1a1a"
C_ESC = "#d9c48b"
C_RING = "#ff6b3d"


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
    footer = (f"BlackHoleShadow {phys.MODEL_VERSION} "
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
            "BlackHoleShadow provenance",
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


def _extent(bx, by):
    return [float(bx.min()), float(bx.max()),
            float(by.min()), float(by.max())]


def _draw_circles(ax, M, show_rs=True, show_ph=True, show_bc=True):
    r_s = phys.event_horizon(M)
    r_ph = phys.photon_sphere(M)
    b_crit = phys.critical_impact_parameter(M)
    if show_rs:
        ax.add_patch(Circle((0.0, 0.0), r_s, fill=False, color=C_RS,
                            lw=1.0, ls=":", label=rf"$r_s=2M$"))
    if show_ph:
        ax.add_patch(Circle((0.0, 0.0), r_ph, fill=False, color=C_PH,
                            lw=1.2, ls="--", label=rf"$r_{{\rm ph}}=3M$"))
    if show_bc:
        ax.add_patch(Circle((0.0, 0.0), b_crit, fill=False, color=C_BC,
                            lw=1.4, ls="-", label=rf"$b_{{\rm crit}}=3\sqrt{{3}}M$"))


def plot_rays(traj_in, traj_out, M=1.0, r_cam=40.0, b_in=5.0, b_out=6.0,
              outdir=None, dpi=140, show=True):
    xs_in, ys_in, info_in = traj_in
    xs_out, ys_out, info_out = traj_out
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    r_s = phys.event_horizon(M)
    r_ph = phys.photon_sphere(M)
    hole = Circle((0.0, 0.0), r_s, facecolor="k", edgecolor="k", zorder=5)
    ax.add_patch(hole)
    ax.add_patch(Circle((0.0, 0.0), r_ph, fill=False, color=C_PH,
                        lw=1.2, ls="--", label="photon sphere"))
    ax.plot(xs_in, ys_in, color="#c1121f", lw=1.6,
            label=rf"$b={b_in:g}$ ({info_in['status']})")
    ax.plot(xs_out, ys_out, color="#0077b6", lw=1.6,
            label=rf"$b={b_out:g}$ ({info_out['status']})")
    ax.set_aspect("equal")
    span = max(r_cam, 2.0 * r_s)
    ax.set_xlim(-1.05 * span, 1.05 * span)
    ax.set_ylim(-1.05 * span, 1.05 * span)
    ax.set_xlabel(r"$x$ [$M$]")
    ax.set_ylabel(r"$y$ [$M$]")
    ax.set_title("Beat 0 · Two Schwarzschild null geodesics")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "rays", dpi, provenance={
        "mode": "rays",
        "M": repr(M),
        "r_cam": repr(r_cam),
        "b_in": repr(b_in),
        "b_out": repr(b_out),
        "status_in": info_in["status"],
        "status_out": info_out["status"],
        "delta_phi_in": repr(info_in["delta_phi"]),
        "delta_phi_out": repr(info_out["delta_phi"]),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_pixels(bx, by, captured, M=1.0, b_in=5.0, b_out=6.0,
                outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
    ax = axes[0]
    # Paint the coarse camera as a classified grid.
    image = np.where(captured, 0.0, 1.0)
    ax.imshow(image, origin="lower", cmap="gray", extent=_extent(bx, by),
              vmin=0.0, vmax=1.0, interpolation="nearest")
    _draw_circles(ax, M, show_rs=False, show_ph=False, show_bc=True)
    ax.plot([b_in], [0.0], marker="o", color="#c1121f", ms=8, zorder=5,
            label=rf"$b={b_in:g}$ pixel")
    ax.plot([b_out], [0.0], marker="o", color="#0077b6", ms=8, zorder=5,
            label=rf"$b={b_out:g}$ pixel")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_title("A ray becomes a pixel")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.set_axis_off()
    b_crit = phys.critical_impact_parameter(M)
    text = (
        "Beat 1\n\n"
        f"The left panel is a coarse camera.\n"
        f"Each cell is one direction on the sky,\n"
        f"labelled by the impact parameter $(b_x,b_y)$.\n\n"
        f"$b={b_in:g}$ lies inside $b_\\mathrm{{crit}}={b_crit:.4g} M$\n"
        f"and is captured — a dark pixel.\n\n"
        f"$b={b_out:g}$ lies outside $b_\\mathrm{{crit}}$ and escapes\n"
        f"— a light pixel.\n\n"
        "PhotonOrbit already classified these two rays.\n"
        "The new object is the camera map."
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11,
            family="serif", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "pixels", dpi, provenance={
        "mode": "pixels",
        "M": repr(M),
        "b_in": repr(b_in),
        "b_out": repr(b_out),
        "n_pix": str(bx.shape[0]),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_capture(bx, by, captured, M=1.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    image = np.where(captured, 0.05, 0.92)
    ax.imshow(image, origin="lower", cmap="gray", extent=_extent(bx, by),
              vmin=0.0, vmax=1.0)
    _draw_circles(ax, M)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_title("Beat 2 · Geometric shadow (no backlight)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "capture", dpi, provenance={
        "mode": "capture",
        "M": repr(M),
        "b_crit": repr(phys.critical_impact_parameter(M)),
        "n_pix": str(bx.shape[0]),
        "fov_M": f"{float(bx.max() - bx.min()):.6f}",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_ring(bx, by, captured, ring, M=1.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    rgb = np.zeros(captured.shape + (3,))
    rgb[..., :] = 0.92
    rgb[captured] = (0.05, 0.05, 0.05)
    rgb[ring] = (1.0, 0.42, 0.18)
    ax.imshow(rgb, origin="lower", extent=_extent(bx, by))
    _draw_circles(ax, M, show_rs=False)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_title("Beat 3 · Photon ring on the shadow rim")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "ring", dpi, provenance={
        "mode": "ring",
        "M": repr(M),
        "n_ring_pixels": str(int(np.count_nonzero(ring))),
        "n_pix": str(bx.shape[0]),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_radii(bx, by, captured, M=1.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    image = np.where(captured, 0.05, 0.92)
    ax.imshow(image, origin="lower", cmap="gray", extent=_extent(bx, by),
              vmin=0.0, vmax=1.0)
    _draw_circles(ax, M)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_title("Beat 4 · Three circles, three meanings")
    ax.legend(loc="upper right", fontsize=8, title="drawn in the image plane")
    note = (
        r"$r_s$ is the horizon in the spacetime diagram;"
        "\n"
        r"$r_{\rm ph}$ is where a photon can orbit;"
        "\n"
        r"$b_{\rm crit}$ is the shadow rim a camera records."
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
            va="bottom", color="0.15",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.4))
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "radii", dpi, provenance={
        "mode": "radii",
        "M": repr(M),
        "r_s_over_M": "2",
        "r_photon_over_M": "3",
        "b_crit_over_M": repr(3.0 * math.sqrt(3.0)),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_weak(bs, exact, weak, M=1.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.plot(bs / M, exact, color="#c1121f", lw=1.8, label=r"exact $\hat\alpha(b)$")
    ax.plot(bs / M, weak, color="#0077b6", lw=1.6, ls="--",
            label=r"weak field $4M/b$")
    b_crit = phys.critical_impact_parameter(M)
    ax.axvline(b_crit / M, color=C_BC, ls=":", lw=1.2,
               label=r"$b_{\rm crit}$")
    ax.set_xlabel(r"$b/M$")
    ax.set_ylabel(r"deflection $\hat\alpha$ [rad]")
    ax.set_title("Beat 5 · Large-$b$ deflection recovers $4GM/(c^2 b)$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(bs.min() / M, bs.max() / M)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    rel = abs(exact[-1] - weak[-1]) / max(weak[-1], 1.0e-16)
    saved = _finish(fig, outdir, "weak", dpi, provenance={
        "mode": "weak",
        "M": repr(M),
        "relative_error_at_bmax": f"{rel:.6e}",
        "b_max_over_M": f"{float(bs[-1] / M):.6f}",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_compare(numbers, M=1.0, outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

    ax = axes[0]
    # Unit-M hole: three radii in units of M.
    ax.add_patch(Circle((0.0, 0.0), numbers["r_s_over_M"], fill=True,
                        facecolor="k", edgecolor="k"))
    ax.add_patch(Circle((0.0, 0.0), numbers["r_photon_over_M"], fill=False,
                        color=C_PH, lw=1.4, ls="--",
                        label=r"$r_{\rm ph}=3M$"))
    ax.add_patch(Circle((0.0, 0.0), numbers["b_crit_over_M"], fill=False,
                        color=C_BC, lw=1.6,
                        label=r"$b_{\rm crit}=3\sqrt{3}M$"))
    ax.set_aspect("equal")
    lim = 1.3 * numbers["b_crit_over_M"]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$x$ [$GM/c^2$ of this hole]")
    ax.set_ylabel(r"$y$ [$GM/c^2$ of this hole]")
    ax.set_title("Schwarzschild hole: photon ring")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.set_axis_off()
    te = numbers["theta_e_arcsec"]
    re_over_m = numbers["r_e_over_M"]
    text = (
        "Beat 6 · Do not call these the same ring\n\n"
        f"A transparent $10^{{{numbers['log10_m_galaxy']:.0f}}} M_\\odot$ "
        "galaxy lens at the\n"
        "GravitationalLensing cartoon distances "
        f"($D_l=1\\,\\mathrm{{Gpc}}$, $D_s=2\\,\\mathrm{{Gpc}}$)\n"
        f"has $\\theta_E = {te:.3f}''$.\n\n"
        "The physical Einstein radius at the lens, in units of\n"
        "that galaxy's own $GM/c^2$, is\n"
        f"    $R_E / M \\approx {re_over_m:.3e}$.\n\n"
        "The photon-sphere radius of a Schwarzschild hole is $3M$.\n"
        "The shadow rim is $b_\\mathrm{crit} \\approx 5.196\\,M$.\n\n"
        "One ring is a weak-field critical curve of a transparent\n"
        "mass.  The other is the unstable photon orbit of the\n"
        "vacuum Schwarzschild metric.  GravitationalLensing's\n"
        "thin-lens $\\alpha = \\theta_E^2\\,\\theta/|\\theta|^2$ cannot\n"
        "produce the second one."
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11,
            family="serif", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "compare", dpi, provenance={
        "mode": "compare",
        "theta_e_arcsec": f"{te:.6f}",
        "r_e_over_M": f"{re_over_m:.6e}",
        "b_crit_over_M": f"{numbers['b_crit_over_M']:.6f}",
        "log10_m_galaxy": f"{numbers['log10_m_galaxy']:.6f}",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_backlight(bx, by, image, M=1.0, r_hot=6.0, b_hot=None,
                   inclination=60.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    vmax = max(float(np.max(image)), 1.0e-12)
    ax.imshow(image, origin="lower", cmap="inferno", extent=_extent(bx, by),
              vmin=0.0, vmax=vmax)
    _draw_circles(ax, M, show_rs=False, show_ph=True, show_bc=True)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$ [$M$]")
    ax.set_ylabel(r"$b_y$ [$M$]")
    ax.set_title("Beat 7 · Thin equatorial backlight (false colour)")
    ax.legend(loc="upper right", fontsize=8)
    note = (
        f"hot ring at $r={r_hot:g} M$, "
        f"$i={inclination:g}^\\circ$ (display shading only)"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8,
            color="white", va="bottom")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "backlight", dpi, provenance={
        "mode": "backlight",
        "M": repr(M),
        "r_hot": repr(r_hot),
        "b_hot": "None" if b_hot is None else repr(b_hot),
        "inclination_deg": repr(inclination),
        "n_pix": str(bx.shape[0]),
        "false_colour": "inferno display scale; not a spectrum",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved
