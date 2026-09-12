"""
plot_bhs.py
===========
Matplotlib figures for BlackHoleShadow.

One drawing routine per mode.  Each routine returns the figure.  When an
output directory is supplied it also writes a timestamped PNG and a
same-stem ``.provenance.txt`` sidecar.  There are no sliders.

Image-plane axes are always labelled in units of M.  Incoming ``bx, by``
arrays are absolute lengths; they are divided by M for display.
"""

import math
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

import physics_bhs as phys

C_RS = "#9a9a9a"
C_PH = "#c9a227"
C_BC = "#0097a7"
C_WIND = "#ff6b3d"


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


def _extent_over_M(bx, by, M):
    ext = phys.pixel_edge_extent(bx, by)
    return [e / M for e in ext]


def _draw_bcrit(ax, M, lw=2.0):
    b_crit_over_M = phys.critical_impact_parameter(M) / M
    ax.add_patch(Circle((0.0, 0.0), b_crit_over_M, fill=False, color=C_BC,
                        lw=lw, ls="-", zorder=4,
                        label=r"$b_{\rm crit}=3\sqrt{3}\,M$  (critical curve)"))


def plot_rays(traj_in, traj_out, M=1.0, r_cam=40.0, b_in=5.0, b_out=6.0,
              lambda_max=200.0, d_lambda=0.02,
              lambda_max_over_M=None, d_lambda_over_M=None,
              outdir=None, dpi=140, show=True):
    xs_in, ys_in, info_in = traj_in
    xs_out, ys_out, info_out = traj_out
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    r_s_o = phys.event_horizon(M) / M
    r_ph_o = phys.photon_sphere(M) / M
    ax.add_patch(Circle((0.0, 0.0), r_s_o, facecolor="k", edgecolor="k",
                        zorder=5, label=r"$r_s=2M$"))
    ax.add_patch(Circle((0.0, 0.0), r_ph_o, fill=False, color=C_PH,
                        lw=1.4, ls="--", label=r"photon sphere $r=3M$"))
    ax.plot(np.asarray(xs_in) / M, np.asarray(ys_in) / M, color="#c1121f",
            lw=1.6, label=rf"$b={b_in / M:g}\,M$ ({info_in['status']})")
    ax.plot(np.asarray(xs_out) / M, np.asarray(ys_out) / M, color="#0077b6",
            lw=1.6, label=rf"$b={b_out / M:g}\,M$ ({info_out['status']})")
    ax.set_aspect("equal")
    span = max(r_cam / M, 2.0 * r_s_o)
    ax.set_xlim(-1.05 * span, 1.05 * span)
    ax.set_ylim(-1.05 * span, 1.05 * span)
    ax.set_xlabel(r"$x/M$")
    ax.set_ylabel(r"$y/M$")
    ax.set_title("Beat 0 · Two Schwarzschild null geodesics")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "rays", dpi, provenance={
        "mode": "rays",
        "M": repr(M),
        "r_cam": repr(r_cam),
        "r_cam_over_M": repr(r_cam / M),
        "b_in": repr(b_in),
        "b_out": repr(b_out),
        "lambda_max": repr(lambda_max),
        "d_lambda": repr(d_lambda),
        "lambda_max_over_M": repr(lambda_max / M if lambda_max_over_M is None else lambda_max_over_M),
        "d_lambda_over_M": repr(d_lambda / M if d_lambda_over_M is None else d_lambda_over_M),
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
                n_pix_requested=None, fov_over_M=16.0,
                outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0))
    ax = axes[0]
    image = np.where(captured, 0.0, 1.0)
    ax.imshow(image, origin="lower", cmap="gray",
              extent=_extent_over_M(bx, by, M),
              vmin=0.0, vmax=1.0, interpolation="nearest")
    _draw_bcrit(ax, M, lw=1.4)
    status_in = "captured" if phys.is_captured(b_in, M) else "escaped"
    status_out = "captured" if phys.is_captured(b_out, M) else "escaped"
    axis = np.asarray(bx[0, :], dtype=float)
    def _snap(b_abs):
        j = int(np.argmin(np.abs(axis - b_abs)))
        return float(axis[j]), j
    cin, jin = _snap(b_in)
    cout, jout = _snap(b_out)
    ax.plot([cin / M], [0.0], marker="s", color="#c1121f", ms=9, zorder=5,
            label=rf"cell $b={cin / M:g}\,M$ ({status_in})")
    ax.plot([cout / M], [0.0], marker="s", color="#0077b6", ms=9, zorder=5,
            label=rf"cell $b={cout / M:g}\,M$ ({status_out})")
    ax.plot([b_in / M], [0.0], marker="o", color="#c1121f", ms=4, zorder=6)
    ax.plot([b_out / M], [0.0], marker="o", color="#0077b6", ms=4, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x/M$")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("A ray becomes a direction on the sky")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.set_axis_off()
    b_crit = phys.critical_impact_parameter(M)
    text = (
        "Beat 1\n\n"
        "The left panel is a coarse camera.\n"
        "Each cell is one conserved impact\n"
        "parameter $(b_x,b_y)$.\n\n"
        f"requested $b={b_in / M:g}\\,M$ sits in the\n"
        f"cell centred at ${cin / M:g}\\,M$ ({status_in})\n"
        f"relative to $b_\\mathrm{{crit}}={b_crit / M:.4g}\\,M$.\n\n"
        f"requested $b={b_out / M:g}\\,M$ sits in the\n"
        f"cell centred at ${cout / M:g}\\,M$ ({status_out}).\n\n"
        "PhotonOrbit already classified single rays.\n"
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
        "n_pix_requested": "None" if n_pix_requested is None else str(n_pix_requested),
        "fov_over_M": repr(fov_over_M),
        "status_in": status_in,
        "status_out": status_out,
        "cell_center_in": repr(cin),
        "cell_center_out": repr(cout),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_capture(bx, by, captured, M=1.0, fov_over_M=16.0,
                 outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    image = np.where(captured, 0.05, 0.92)
    ax.imshow(image, origin="lower", cmap="gray",
              extent=_extent_over_M(bx, by, M), vmin=0.0, vmax=1.0)
    _draw_bcrit(ax, M)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x/M$")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("Beat 2 · Geometric capture map")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "capture", dpi, provenance={
        "mode": "capture",
        "M": repr(M),
        "b_crit": repr(phys.critical_impact_parameter(M)),
        "n_pix": str(bx.shape[0]),
        "fov_over_M": repr(fov_over_M),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_ring(bx, by, captured, ring, M=1.0, b_lo=None, b_hi=None,
              fov_over_M=16.0, outdir=None, dpi=140, show=True):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    rgb = np.zeros(captured.shape + (3,))
    rgb[..., :] = 0.92
    rgb[captured] = (0.05, 0.05, 0.05)
    rgb[ring] = (1.0, 0.42, 0.18)
    ax.imshow(rgb, origin="lower", extent=_extent_over_M(bx, by, M))
    _draw_bcrit(ax, M, lw=1.5)
    if b_lo is not None and b_hi is not None and b_hi > b_lo:
        # Continuous overlay so a coarse grid still shows the window.
        wedge = Wedge((0.0, 0.0), b_hi / M, 0.0, 360.0, width=(b_hi - b_lo) / M,
                      facecolor=C_WIND, alpha=0.18, edgecolor=C_WIND, lw=0.8,
                      label="high-winding window")
        ax.add_patch(wedge)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x/M$")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("Beat 3 · High-winding overlay on the critical curve")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "ring", dpi, provenance={
        "mode": "ring",
        "M": repr(M),
        "n_overlay_pixels": str(int(np.count_nonzero(ring))),
        "n_pix": str(bx.shape[0]),
        "fov_over_M": repr(fov_over_M),
        "b_lo": "None" if b_lo is None else repr(b_lo),
        "b_hi": "None" if b_hi is None else repr(b_hi),
        "r_window_over_M": repr(phys.PHOTON_RING_R_WINDOW),
        "delta_phi_min": repr(phys.WINDING_DELTA_PHI),
        "note": "pedagogical high-winding overlay, not a photon-ring profile",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_radii(bx, by, captured, M=1.0, fov_over_M=16.0,
               outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4))

    ax = axes[0]
    ax.add_patch(Circle((0.0, 0.0), 2.0, facecolor="k", edgecolor="k",
                        label=r"$r_s=2M$  (horizon)"))
    ax.add_patch(Circle((0.0, 0.0), 3.0, fill=False, color=C_PH, lw=1.6, ls="--",
                        label=r"$r_{\rm ph}=3M$  (photon sphere)"))
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel(r"coordinate $x/M$")
    ax.set_ylabel(r"coordinate $y/M$")
    ax.set_title("Schwarzschild coordinate-radius cross-section")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    image = np.where(captured, 0.05, 0.92)
    ax.imshow(image, origin="lower", cmap="gray",
              extent=_extent_over_M(bx, by, M), vmin=0.0, vmax=1.0)
    _draw_bcrit(ax, M, lw=2.2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x/M$")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("Image plane (critical curve of a distant camera)")
    ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Beat 6 · Coordinate radii are not image-plane radii",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    saved = _finish(fig, outdir, "radii", dpi, provenance={
        "mode": "radii",
        "M": repr(M),
        "r_s_over_M": "2",
        "r_photon_over_M": "3",
        "b_crit_over_M": repr(3.0 * math.sqrt(3.0)),
        "n_pix": str(bx.shape[0]),
        "fov_over_M": repr(fov_over_M),
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
    ax.set_title(r"Beat 7 · Large-$b$ deflection recovers $4M/b$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(bs.min() / M, bs.max() / M)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    rel = abs(exact[-1] - weak[-1]) / max(weak[-1], 1.0e-16)
    saved = _finish(fig, outdir, "weak", dpi, provenance={
        "mode": "weak",
        "M": repr(M),
        "relative_error_at_bmax": f"{rel:.6e}",
        "b_max_over_M": f"{float(bs[-1] / M):.6f}",
        "n_samples": str(len(bs)),
        "b_min_factor": "1.02",
        "b_max_factor": "8.0",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_compare(numbers, M=1.0, outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

    ax = axes[0]
    ax.add_patch(Circle((0.0, 0.0), numbers["r_s_over_M"], fill=True,
                        facecolor="k", edgecolor="k",
                        label=r"$r_s=2M$"))
    ax.add_patch(Circle((0.0, 0.0), numbers["r_photon_over_M"], fill=False,
                        color=C_PH, lw=1.4, ls="--",
                        label=r"$r_{\rm ph}=3M$"))
    ax.add_patch(Circle((0.0, 0.0), numbers["b_crit_over_M"], fill=False,
                        color=C_BC, lw=2.0,
                        label=r"$b_{\rm crit}=3\sqrt{3}M$"))
    ax.set_aspect("equal")
    lim = 1.3 * numbers["b_crit_over_M"]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$b_x/M$  (image plane)")
    ax.set_ylabel(r"$b_y/M$")
    ax.set_title("Schwarzschild hole: critical curve")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.set_axis_off()
    te = numbers["theta_e_arcsec"]
    re_over_m = numbers["r_e_over_M"]
    text = (
        "Beat 8 · Do not call these the same ring\n\n"
        f"A transparent $10^{{{numbers['log10_m_galaxy']:.0f}}} M_\\odot$ "
        "galaxy lens at the\n"
        "GravitationalLensing cartoon distances "
        r"($D_l=1\,\mathrm{Gpc}$, $D_s=2\,\mathrm{Gpc}$)"
        "\n"
        f"has $\\theta_E = {te:.3f}''$.\n\n"
        "The physical Einstein radius at the lens, in units of\n"
        "that galaxy's own $GM/c^2$, is\n"
        f"    $R_E / M \\approx {re_over_m:.3e}$.\n\n"
        "The photon-sphere radius of a Schwarzschild hole is $3M$.\n"
        "The image-plane critical curve is "
        r"$b_{\mathrm{crit}} \approx 5.196\,M$."
        "\n\n"
        "One is a weak-field critical curve of a transparent mass.\n"
        "The other is the capture boundary of the vacuum\n"
        "Schwarzschild metric.  A thin-lens "
        r"$\alpha = \theta_E^2\,\theta/|\theta|^2$"
        "\ncannot produce the second one."
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", fontsize=11,
            family="serif", transform=ax.transAxes)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    saved = _finish(fig, outdir, "compare", dpi, provenance={
        "mode": "compare",
        "program_M": repr(M),
        "theta_e_arcsec": f"{te:.6f}",
        "r_e_over_M": f"{re_over_m:.6e}",
        "b_crit_over_M": f"{numbers['b_crit_over_M']:.6f}",
        "log10_m_galaxy": f"{numbers['log10_m_galaxy']:.6f}",
        "d_l_m": repr(numbers["d_l_m"]),
        "d_s_m": repr(numbers["d_s_m"]),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_transfer(bs, table, counts, M=1.0, r_hot_over_M=6.0,
                  width_over_M=0.45, peaks=None, fov_over_M=16.0,
                  outdir=None, dpi=140, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))
    ax = axes[0]
    colors = ("#0077b6", "#c9a227", "#c1121f")
    labels = (r"$r_1/M$ (direct)", r"$r_2/M$ (lensing)",
              r"$r_3/M$ (photon ring / subring)")
    for m in range(table.shape[1]):
        y = table[:, m]
        ax.plot(bs / M, y, color=colors[m % 3], lw=1.6, label=labels[m])
    ax.axvline(phys.critical_impact_parameter(M) / M, color=C_BC, ls=":",
               label=r"$b_{\rm crit}$")
    ax.axhline(3.0, color=C_PH, ls="--", lw=1.0, label=r"$r_{\rm ph}=3M$")
    ax.axhline(r_hot_over_M, color="0.4", ls="-.", lw=1.0,
               label=rf"$r_{{\rm hot}}={r_hot_over_M:g}M$")
    if peaks:
        for m, bp in enumerate(peaks):
            if bp is None:
                continue
            ax.plot([bp / M], [r_hot_over_M], marker="o", color=colors[m % 3],
                    ms=7, zorder=5)
    ax.set_xlabel(r"$b/M$")
    ax.set_ylabel(r"crossing radius $r_m/M$")
    ax.set_title("Transfer functions (adaptive $b$ grid)")
    ax.legend(loc="upper right", fontsize=7)
    y_top = max(12.0, r_hot_over_M + 4.0)
    ax.set_ylim(2.0, y_top)

    ax = axes[1]
    r = np.linspace(2.01, max(12.0, r_hot_over_M + 4.0), 200)
    iem = np.array([
        phys.emitted_intensity(float(rv) * M, M, r_hot=r_hot_over_M * M,
                               width=width_over_M * M)
        for rv in r
    ])
    ax.plot(r, iem, color="#7b2d00", lw=1.8)
    ax.axvline(r_hot_over_M, color="0.4", ls="-.", lw=1.0)
    ax.set_xlabel(r"$r/M$")
    ax.set_ylabel(r"$I_{\mathrm{em}}(r)$  (static emitters)")
    ax.set_title(rf"Source: Gaussian annulus, width ${width_over_M:g}M$")
    fig.suptitle("Beat 4 · Source profile and face-on transfer functions",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    saved = _finish(fig, outdir, "transfer", dpi, provenance={
        "mode": "transfer",
        "M": repr(M),
        "r_hot_over_M": repr(r_hot_over_M),
        "width_over_M": repr(width_over_M),
        "fov_over_M": repr(fov_over_M),
        "n_samples": str(len(bs)),
        "max_m": str(table.shape[1]),
        "n_finite_r3": str(int(np.isfinite(table[:, 2]).sum()) if table.shape[1] > 2 else 0),
        "sampling": "adaptive log cluster around b_crit",
        "emitters": "static; no orbital Doppler",
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved


def plot_image(bx, by, img1, img2, img3, M=1.0, r_hot=6.0, peaks=None,
               fov_over_M=16.0, sample=None, parts=None,
               outdir=None, dpi=140, show=True):
    fig = plt.figure(figsize=(12.6, 6.4))
    axes = [fig.add_subplot(2, 3, i) for i in range(1, 4)]
    ax_r = fig.add_subplot(2, 1, 2)
    extent = _extent_over_M(bx, by, M)
    vmax = max(float(np.max(img1)), float(np.max(img2)),
               float(np.max(img3)), 1.0e-12)
    titles = (
        "direct (m=1)",
        "+ lensing (m<=2)",
        "+ photon-ring/subring (m=3,4)",
    )
    for ax, img, title in zip(axes, (img1, img2, img3), titles):
        ax.imshow(img, origin="lower", cmap="inferno",
                  extent=extent, vmin=0.0, vmax=vmax)
        _draw_bcrit(ax, M, lw=1.1)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$b_x/M$")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel(r"$b_y/M$")
    if sample is not None and parts is not None:
        b_over = np.asarray(sample) / M
        if parts.shape[1] >= 2:
            ax_r.plot(b_over, parts[:, 0], color="#0077b6", lw=1.4, label="m=1")
            ax_r.plot(b_over, parts[:, 1], color="#c9a227", lw=1.4, label="m=2")
        if parts.shape[1] >= 4:
            ax_r.plot(b_over, parts[:, 2] + parts[:, 3], color="#c1121f",
                      lw=1.6, label="m=3+4")
        elif parts.shape[1] >= 3:
            ax_r.plot(b_over, parts[:, 2], color="#c1121f", lw=1.6, label="m=3")
        ax_r.axvline(phys.critical_impact_parameter(M) / M, color=C_BC, ls=":")
        x_lo = phys.critical_impact_parameter(M) / M - 0.2
        x_hi = phys.critical_impact_parameter(M) / M + 2.4
        if peaks:
            finite = [p / M for p in peaks if p is not None]
            if finite:
                x_lo = min(x_lo, min(finite) - 0.4)
                x_hi = max(x_hi, max(finite) + 0.6)
                for p in finite:
                    ax_r.axvline(p, color="0.7", ls="--", lw=0.7)
        ax_r.set_xlim(x_lo, x_hi)
        ax_r.set_xlabel(r"$b/M$")
        ax_r.set_ylabel(r"$I_m(b)$  (pointwise)")
        ax_r.set_title(
            "Radial I(b) from the source-aware table. "
            "Narrow peaks may miss a pixel centre."
        )
        ax_r.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        r"Beat 5 · Point-sampled face-on image  "
        r"$I_{\mathrm{obs}}\approx\sum_{m=1}^{4} g^4 I_{\mathrm{em}}$"
        r"  (static emitters; m>=5 omitted)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    saved = _finish(fig, outdir, "image", dpi, provenance={
        "mode": "image",
        "M": repr(M),
        "r_hot": repr(r_hot),
        "r_hot_over_M": repr(r_hot / M),
        "b_m_over_M": repr(
            None if peaks is None else
            [None if p is None else p / M for p in peaks]
        ),
        "width_over_M": repr(phys.HOT_RING_WIDTH_DEFAULT),
        "n_pix": str(bx.shape[0]),
        "n_samples": "None" if sample is None else str(len(sample)),
        "fov_over_M": repr(fov_over_M),
        "I_obs": "sum_{m=1..4} g^4 I_em; m>=5 omitted",
        "pixel_contract": "point sample at pixel centre from adaptive I(b)",
        "false_colour": "inferno display scale; not a spectrum",
        "emitters": "static; no orbital Doppler",
        "sampling": "adaptive log cluster around b_crit; no peak stamp",
        "max_m": str(phys.MAX_IMAGE_M),
    })
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, saved
