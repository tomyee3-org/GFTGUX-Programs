"""
plot_gw.py
==========
Three-panel Matplotlib visualization for GravitationalWaveSources.
"""

import os
import numpy as np
from datetime import datetime
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.png"


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

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    ax1, ax2, ax3 = axes

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
        ])
    ann_text = "\n".join(ann_lines)

    ax1.plot(t[insp], h[insp], color=C_WAVE, lw=lw, label="Inspiral strain scale $h(t)$")
    if np.any(rd):
        ax1.plot(t[rd], h[rd], color=C_RD, lw=lw, label="Illustrative Schwarzschild QNM")
    ax1.plot(t[valid_A], A[valid_A], color=C_ENV, lw=0.9, ls="--", label=r"Envelope $\pm A(t)$")
    ax1.plot(t[valid_A], -A[valid_A], color=C_ENV, lw=0.9, ls="--")
    ax1.axvline(t_isco, **cutoff_kw)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Strain scale $h(t)$")
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
    ax2.set_ylabel("Amplitude $A(t)$")
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

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, _timestamp_fname())
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"[plot_gw] PNG saved -> {fpath}")
    else:
        print("[plot_gw] Displaying figure on screen ...")
        plt.show()

    plt.close(fig)
