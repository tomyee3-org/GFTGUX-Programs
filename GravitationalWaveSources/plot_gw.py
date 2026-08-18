"""
plot_gw.py
==========
Three-panel Matplotlib figure for GravitationalWaveSources:

  Panel 1 (top)    : GW strain waveform h(t)  — inspiral + ringdown
  Panel 2 (middle) : Strain amplitude envelope A(t)
  Panel 3 (bottom) : Instantaneous GW frequency f(t)

Output
------
* Default  → interactive screen display (plt.show())
* Optional → PNG saved to a user-specified directory with a
             timestamp in the filename (YYYYMMDD_HHmm resolution).
"""

import os
import numpy as np
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Palette ───────────────────────────────────────────────────────────────────
C_WAVE = "#1f77b4"   # blue   – inspiral waveform
C_ENV  = "#ff7f0e"   # orange – amplitude envelope
C_RD   = "#d62728"   # red    – ringdown
C_FREQ = "#2ca02c"   # green  – frequency
C_ISCO = "#9467bd"   # purple – ISCO level


# ── Helpers ───────────────────────────────────────────────────────────────────

def _xlim(t_merger, t_before, t_after, t_min, t_max):
    lo = (t_merger - t_before) if t_before is not None else t_min
    hi = (t_merger + t_after)  if t_after  is not None else t_max
    return max(lo, t_min), min(hi, t_max)


def _sci_y(ax):
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))


def _timestamp_fname(prefix="gw_inspiral"):
    """Return  prefix_YYYYMMDD_HHmm.png  using the current local time."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{prefix}_{ts}.png"


# ── Main plotting function ────────────────────────────────────────────────────

def plot_inspiral(result,
                  outdir=None,
                  t_before=None, t_after=None,
                  lw=0.4, dpi=150, figsize=(12, 9)):
    """
    Render the three-panel inspiral figure.

    Parameters
    ----------
    result   : dict from physics_gw.integrate_inspiral()
    outdir   : directory for PNG output.
               None  → display on screen only (plt.show()).
               str   → save  <outdir>/gw_inspiral_YYYYMMDD_HHmm.png
    t_before : seconds before merger to show  (None → full inspiral)
    t_after  : seconds after  merger to show  (None → full ringdown)
    lw       : waveform trace line width [points]
    dpi      : PNG resolution
    figsize  : (width, height) inches
    """

    t = result["t"]
    h = result["h"]
    A = result["A"]
    f = result["f"]
    t_merger = result["t_merger"]
    f_isco = result["f_isco_hz"]
    s = result["summary"]

    insp = t <= t_merger
    rd = t > t_merger
    valid = insp & np.isfinite(A)

    x_lo, x_hi = _xlim(t_merger, t_before, t_after, t[0], t[-1])

    fig, axes = plt.subplots(3, 1, figsize=figsize,
                             constrained_layout=True)
    ax1, ax2, ax3 = axes

    suptitle = (
        f"Binary Inspiral — "
        f"$m_1={s['m1_msun']:.2f}\\,M_\\odot$,  "
        f"$m_2={s['m2_msun']:.2f}\\,M_\\odot$,  "
        f"$\\mathcal{{M}}_c={s['Mc_msun']:.3f}\\,M_\\odot$,  "
        f"$d={s['d_mpc']:.0f}\\,\\mathrm{{Mpc}}$"
    )
    fig.suptitle(suptitle, fontsize=11, fontweight="bold")

    merger_kw = dict(color="k", lw=0.8, ls=":",
                     label=f"Merger  $t={t_merger:.3f}\\,\\mathrm{{s}}$")

    ann_text = (
        f"$f_{{\\rm ISCO}}={f_isco:.1f}\\,\\mathrm{{Hz}}$\n"
        f"$f_{{\\rm QNM}}={s['f_qnm_hz']:.1f}\\,\\mathrm{{Hz}}$\n"
        f"$\\tau_{{\\rm QNM}}={s['tau_qnm_ms']:.3f}\\,\\mathrm{{ms}}$"
    )
    ann_kw = dict(transform=ax1.transAxes, ha="right", va="top",
                  fontsize=8,
                  bbox=dict(boxstyle="round,pad=0.3",
                            fc="lightyellow", ec="gray", alpha=0.9))

    # ════════════════════════════════════════════════════
    # Panel 1 : Strain waveform h(t)
    # ════════════════════════════════════════════════════
    ax1.plot(t[insp], h[insp], color=C_WAVE, lw=lw,
             label="Inspiral strain $h(t)$")
    ax1.plot(t[rd],   h[rd],   color=C_RD,   lw=lw,
             label="Ringdown (QNM)")
    ax1.plot(t[valid],  A[valid], color=C_ENV, lw=0.9, ls="--",
             label="Envelope $\\pm A(t)$")
    ax1.plot(t[valid], -A[valid], color=C_ENV, lw=0.9, ls="--")
    ax1.axvline(t_merger, **merger_kw)

    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Strain $h(t)$", fontsize=10)
    ax1.set_title("What LIGO Measures — GW Strain Waveform", fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    ax1.text(0.99, 0.97, ann_text, **ann_kw)
    _sci_y(ax1)
    ax1.grid(True, lw=0.3, alpha=0.5)

    # ════════════════════════════════════════════════════
    # Panel 2 : Strain amplitude A(t)
    # ════════════════════════════════════════════════════
    ax2.plot(t[valid], A[valid], color=C_ENV, lw=0.9,
             label="Amplitude $A(t)$")
    ax2.axvline(t_merger, **merger_kw)

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylabel("Amplitude $A(t)$", fontsize=10)
    ax2.set_title("Strain Amplitude Envelope", fontsize=10)
    ax2.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    _sci_y(ax2)
    ax2.grid(True, lw=0.3, alpha=0.5)

    # ════════════════════════════════════════════════════
    # Panel 3 : GW frequency f(t)
    # ════════════════════════════════════════════════════
    ax3.plot(t[insp], f[insp], color=C_FREQ, lw=0.9,
             label="GW frequency $f(t)$")
    ax3.axhline(f_isco, color=C_ISCO, lw=0.9, ls="--",
                label=f"$f_{{\\rm ISCO}}={f_isco:.1f}\\,\\mathrm{{Hz}}$")
    ax3.axvline(t_merger, **merger_kw)

    ax3.set_xlim(x_lo, x_hi)
    ax3.set_xlabel("Time $t$  [s]", fontsize=10)
    ax3.set_ylabel("GW Frequency [Hz]", fontsize=10)
    ax3.set_title("Instantaneous GW Frequency — Chirp", fontsize=10)
    ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.75)
    ax3.grid(True, lw=0.3, alpha=0.5)

    # ── Output ────────────────────────────────────────────────────────────────
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fname = _timestamp_fname("gw_inspiral")
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"[plot_gw] PNG saved → {fpath}")
    else:
        print("[plot_gw] Displaying figure on screen …")
        plt.show()

    plt.close(fig)
