"""
plot_gw.py
==========
Three-panel Matplotlib visualization for GravitationalWaveSources.
"""

import os
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

C_WAVE = "#1f77b4"
C_ENV  = "#ff7f0e"
C_RD   = "#d62728"
C_FREQ = "#2ca02c"
C_ISCO = "#9467bd"


def _xlim(t_isco, t_before, t_after, t_min, t_max):
    lo = (t_isco - t_before) if t_before is not None else t_min
    hi = (t_isco + t_after) if t_after is not None else t_max
    lo = max(lo, t_min)
    hi = min(hi, t_max)
    if hi <= lo:
        raise ValueError("The requested zoom interval contains no plotted data.")
    return lo, hi


def _sci_y(ax):
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def _timestamp_fname(prefix="gw_inspiral"):
    # Microsecond resolution makes two rapid saves in the same directory
    # (e.g. a scripted parameter sweep) collide far less often than under
    # the previous second-resolution timestamp, but does not make a
    # collision impossible (a coarser OS clock, two calls landing in the
    # same microsecond, or a clock adjustment can still repeat a value).
    # _reserve_unique_path() below is what actually guarantees no silent
    # overwrite; this function only picks the first candidate name.
    #
    # Audit3 (Gemini): UTC, matching driver_gw's CSV filenames and its
    # export_timestamp_utc metadata field -- so PNG and CSV filenames from
    # the same run sort and compare directly without a timezone conversion.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.png"


def _reserve_unique_path(directory, prefix="gw_inspiral", ext="png", max_attempts=1000):
    """Atomically claim a not-yet-existing "prefix_timestamp[_n].ext" path.

    See driver_gw._reserve_unique_path for the full rationale (Audit2
    Copilot A2-3, Audit3 Codex P2-1/Copilot A3-6): a timestamp collision
    only reduces, rather than eliminates, the chance of two writes picking
    the same filename, so this claims the path atomically with
    O_CREAT|O_EXCL and retries with a "_1", "_2", ... suffix on collision.
    This only reserves the name, though -- callers must not write image
    content directly into the returned path by reopening it; write to a
    temporary path and publish with os.replace() instead (see
    _publish_or_cleanup() below), or a failed savefig() leaves a zero-byte
    or partial PNG permanently occupying this filename.
    """
    base = _timestamp_fname(prefix=prefix)
    stem = base[: -(len(ext) + 1)]
    for attempt in range(max_attempts):
        candidate = f"{stem}.{ext}" if attempt == 0 else f"{stem}_{attempt}.{ext}"
        path = os.path.join(directory, candidate)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise RuntimeError(
        f"Could not reserve a unique output filename in {directory!r} "
        f"after {max_attempts} attempts."
    )


def _publish_or_cleanup(write_fn, fpath):
    """Run write_fn(tmp_path) to build the artifact at a private temporary
    path beside fpath, then atomically publish it to fpath -- or, on any
    failure, remove every trace and re-raise. See driver_gw._publish_or_
    cleanup for the full rationale (Audit3 Codex P2-1 / Copilot A3-3);
    duplicated here rather than imported, per this project's existing
    convention of not sharing code across the driver_gw/plot_gw split.
    """
    tmp_path = fpath + ".tmp"
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, fpath)
    except BaseException:
        for stray in (tmp_path, fpath):
            try:
                os.remove(stray)
            except OSError:
                pass
        raise
    return fpath


def plot_inspiral(result, outdir=None,
                  t_before=None, t_after=None,
                  lw=0.4, dpi=150, figsize=(12, 9)):
    """Render waveform, inspiral amplitude scale, and instantaneous frequency."""
    t = result["t"]
    h = result["h"]
    A = result["A"]
    f = result["f"]
    t_isco = result["t_isco"]
    f_isco = result["f_isco_hz"]
    s = result["summary"]

    insp = np.isfinite(f)
    rd = ~insp
    valid_A = np.isfinite(A)

    x_lo, x_hi = _xlim(t_isco, t_before, t_after, t[0], t[-1])

    # Validate/create the output directory before building the figure, not
    # only before saving into it -- a bad --outdir (e.g. an existing
    # regular file) is then rejected before any Matplotlib work happens at
    # all. driver_gw.run() additionally validates/creates this same
    # directory (and --csvdir's) even earlier, before either requested
    # artifact is written, when this function is reached through run();
    # this call keeps plot_inspiral() safe to call directly too.
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    try:
        _render(fig, axes, result, t_isco, f_isco, s, insp, rd, valid_A,
                x_lo, x_hi, lw)

        # The figure is always shown on screen. When --outdir is given, a
        # timestamped PNG is *additionally* saved to that folder; saving
        # happens before plt.show() so the file is written even if the
        # user closes the interactive window without waiting, or the run
        # is otherwise interrupted after the window appears. The save is
        # atomic (write to a temp path, then os.replace() into place --
        # see _publish_or_cleanup()): a failed savefig() leaves no PNG
        # behind rather than a zero-byte or partial one (Audit3 Codex
        # P2-1 / Copilot A3-3).
        if outdir is not None:
            fpath = _reserve_unique_path(outdir)
            # format="png" is passed explicitly because the temporary path
            # _publish_or_cleanup() writes to ends in ".tmp", not ".png" --
            # savefig() otherwise infers the output format from the
            # filename extension and would reject ".tmp" as unrecognized.
            _publish_or_cleanup(
                lambda tmp_path: fig.savefig(
                    tmp_path, dpi=dpi, bbox_inches="tight", format="png"
                ),
                fpath,
            )
            print(f"[plot_gw] PNG saved -> {fpath}")

        print("[plot_gw] Displaying figure on screen ...")
        plt.show()
    finally:
        # Audit3 (Codex P2-1, item 5): this must run even if savefig() or
        # anything else above raised, so a plotting exception can never
        # leak a live figure into pyplot's registry.
        plt.close(fig)


def _render(fig, axes, result, t_isco, f_isco, s, insp, rd, valid_A,
            x_lo, x_hi, lw):
    """Draw all three panels onto an already-created figure/axes."""
    ax1, ax2, ax3 = axes
    t, h, A, f = result["t"], result["h"], result["A"], result["f"]

    fig.suptitle(
        f"Compact-binary inspiral — "
        fr"$m_1={s['m1_msun']:.2f}\,M_\odot$,  "
        fr"$m_2={s['m2_msun']:.2f}\,M_\odot$,  "
        fr"$\mathcal{{M}}={s['Mc_msun']:.3f}\,M_\odot$,  "
        fr"$d={s['d_mpc']:.0f}\,\mathrm{{Mpc}}$",
        fontsize=11, fontweight="bold"
    )

    cutoff_kw = dict(color="k", lw=0.8, ls=":",
                     label=fr"ISCO cutoff  $t={t_isco:.3f}\,\mathrm{{s}}$")

    ann_lines = [fr"$f_{{\rm ISCO}}={f_isco:.1f}\,\mathrm{{Hz}}$"]
    if s["include_ringdown"]:
        ann_lines.extend([
            fr"$f_{{\rm QNM}}={s['f_qnm_hz']:.1f}\,\mathrm{{Hz}}$",
            fr"$\tau_{{\rm QNM}}={s['tau_qnm_ms']:.3f}\,\mathrm{{ms}}$",
            "illustrative ringdown --",
            "not a physical merger",
        ])
    ann_text = "\n".join(ann_lines)

    ax1.plot(t[insp], h[insp], color=C_WAVE, lw=lw, label="Inspiral strain scale $h(t)$")
    if np.any(rd):
        ax1.plot(t[rd], h[rd], color=C_RD, lw=lw, label="Illustrative Schwarzschild QNM")
    ax1.plot(t[valid_A], A[valid_A], color=C_ENV, lw=0.9, ls="--", label=r"Envelope $\pm A(t)$")
    ax1.plot(t[valid_A], -A[valid_A], color=C_ENV, lw=0.9, ls="--")
    ax1.axvline(t_isco, **cutoff_kw)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Strain scale $h(t)$ (dimensionless)")
    ax1.set_title("Illustrative gravitational-wave strain")
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    ax1.text(0.99, 0.97, ann_text, transform=ax1.transAxes,
             ha="right", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))
    _sci_y(ax1)
    ax1.grid(True, lw=0.3, alpha=0.5)

    ax2.plot(t[valid_A], A[valid_A], color=C_ENV, lw=0.9, label="Inspiral amplitude scale $A(t)$")
    ax2.axvline(t_isco, **cutoff_kw)
    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylabel("Amplitude $A(t)$ (dimensionless)")
    ax2.set_title("Leading-order inspiral amplitude scale")
    ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    _sci_y(ax2)
    ax2.grid(True, lw=0.3, alpha=0.5)

    ax3.plot(t[insp], f[insp], color=C_FREQ, lw=0.9, label="GW frequency $f(t)$")
    ax3.axhline(f_isco, color=C_ISCO, lw=0.9, ls="--",
                label=fr"$f_{{\rm ISCO}}={f_isco:.1f}\,\mathrm{{Hz}}$")
    ax3.axvline(t_isco, **cutoff_kw)
    ax3.set_xlim(x_lo, x_hi)
    ax3.set_xlabel("Time $t$ [s]")
    ax3.set_ylabel("GW frequency [Hz]")
    ax3.set_title("Instantaneous inspiral frequency — the chirp")
    ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    ax3.grid(True, lw=0.3, alpha=0.5)
