"""
plot_cosmo.py
=============
Matplotlib visualization layer for Cosmology_expansion_simulator.
"""

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

C_M = "#1f77b4"     # matter
C_R = "#ff7f0e"     # radiation
C_K = "#7f7f7f"     # curvature
C_DE = "#9467bd"    # dark energy
C_A = "#2ca02c"     # scale factor / age
C_H = "#d62728"     # Hubble parameter
C_Q = "#17becf"     # deceleration parameter
C_SUM = "#333333"   # sum-check line
C_REF = "#b5811a"   # reference / benchmark lines


def _timestamp_fname(prefix):
    # Microsecond resolution avoids silently overwriting a previous figure
    # when two plotting calls in the same run land in the same second.
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.png"


def _save_and_show(fig, outdir, prefix, dpi):
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        fpath = os.path.join(outdir, _timestamp_fname(prefix))
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        print(f"[plot_cosmo] PNG saved -> {fpath}")
    print("[plot_cosmo] Displaying figure on screen ...")
    plt.show()
    plt.close(fig)


# ======================================================================
# Mode: evolve
# ======================================================================
def _big_rip_continuation(result):
    """Plot-only future of a certified constant-w phantom expanding branch.

    No infinity is integrated by RK4 and no returned trajectory is modified.
    Put m=-3(1+w)/2>0 and x=(a_end/a)**m. Then the remaining cosmic time is
    a regular integral over 0<=x<=1, with x=0 representing a=infinity:

      dt/(-dx) = a_end**(-m)/(m H0 sqrt(Ode)) /
        sqrt(1 + cm*x**(2+3/m) + cr*x**(2+4/m) + ck*x**(2+2/m)).

    ci=(Oi/Ode)*a_end**(-ni-2m), ni=3,4,2. Thus matter, radiation and
    curvature are retained, unlike the summary's DE-only time estimate.
    The absolute time inherits the existing trajectory's numerical error.
    Only an a_max stop at/after today is extended; a genuine time stop,
    recollapse, unresolved fate or evolving CPL equation of state is not.
    """
    s = result["summary"]
    if (s.get("fate_status") != "big_rip" or s.get("stop_reason") != "a_max"
            or s["wa"] != 0.0 or s["w0"] >= -1.0 or s["omega_de"] <= 0.0):
        return None
    a_end, t_end = float(result["a"][-1]), float(result["t_gyr"][-1])
    if not np.isfinite(a_end) or a_end < 1.0 or not np.isfinite(t_end):
        return None
    m = -1.5 * (1.0 + s["w0"])
    log_a = np.log(a_end)
    # Show at least six further decades when possible, approaching within
    # ~0.1% of the remaining lifetime for ordinary phantom parameters.
    # Stay within float64 and the engine's |ln rho_DE/rho_DE0|<300 policy.
    log_span = min(max(6.0 * np.log(10.0), np.log(1000.0) / m),
                   680.0 - log_a, 149.0 / m - log_a)
    if log_span <= 0.0:
        return None
    x = np.exp(-m * np.linspace(0.0, log_span, 241))
    powers = 2.0 + np.array([3.0, 4.0, 2.0]) / m
    densities = np.array([s["omega_m"], s["omega_r"], s["omega_k"]])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        coeff = np.sign(densities) * np.exp(
            np.log(np.abs(densities)) - np.log(s["omega_de"])
            - (np.array([3.0, 4.0, 2.0]) + 2.0 * m) * log_a)
        time_scale = (s["H0_inv_gyr"] * np.exp(-m * log_a)
                      / (m * np.sqrt(s["omega_de"])))
    if not np.all(np.isfinite(coeff)) or not np.isfinite(time_scale):
        return None

    # Integrate on short log-spaced intervals; include the final interval
    # down to x=0 for the finite asymptote, never for a plotted data point.
    edges = np.r_[x, 0.0]
    lo, hi = edges[1:], edges[:-1]

    def quadrature(order):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        xx = lo[:, None] + 0.5 * (hi - lo)[:, None] * (nodes + 1.0)
        terms = coeff[:, None, None] * xx[None, :, :] ** powers[:, None, None]
        factor = 1.0 + np.sum(terms, axis=0)
        if np.any(~np.isfinite(factor)) or np.any(factor <= 0.0):
            return None
        return time_scale * 0.5 * (hi - lo) * (1.0 / np.sqrt(factor) @ weights)

    intervals = quadrature(64)
    check = quadrature(32)
    if intervals is None or check is None:
        return None
    # An extreme near-loitering model may need adaptive quadrature. Keep
    # its ordinary plot rather than draw an uncertified extrapolation.
    if (not np.all(np.isfinite(intervals)) or
            not np.allclose(intervals, check, rtol=1e-7, atol=0.0)):
        return None
    remaining = float(np.sum(intervals))
    t_rip = t_end + remaining
    if not np.isfinite(t_rip) or t_rip <= t_end:
        return None  # remaining lifetime is below this clock's resolution
    tt = t_end + np.r_[0.0, np.cumsum(intervals[:-1])]
    aa = np.exp(log_a - np.log(x) / m)
    terms = coeff[:, None] * x[None, :] ** powers[:, None]
    factor = 1.0 + np.sum(terms, axis=0)
    hh = s["H0_kms_mpc"] * np.sqrt(s["omega_de"] * factor) * np.exp(
        m * np.log(aa))
    qq = 0.5 * (terms[0] + 2.0 * terms[1] + 1.0 + 3.0 * s["w0"]) / factor
    # Joining at the actual last sample avoids a one-ULP gap in the curve.
    aa[0] = a_end
    hh[0] = result["H_kms_mpc"][-1]
    qq[0] = result["q"][-1]
    if not np.all(np.isfinite(hh)):
        return None
    visible = tt < t_rip
    if np.count_nonzero(visible) < 2:
        return None
    return dict(t_gyr=tt[visible], a=aa[visible], H_kms_mpc=hh[visible],
                q=qq[visible], t_rip_gyr=t_rip)


def plot_evolve(result, outdir=None, dpi=150, lw=1.6, figsize=(13, 10),
                extend_big_rip=True):
    s = result["summary"]
    t, a, H, q = result["t_gyr"], result["a"], result["H_kms_mpc"], result["q"]
    Om, Or, Ok, Ode = result["Om"], result["Or"], result["Ok"], result["Ode"]
    future = _big_rip_continuation(result) if extend_big_rip else None

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    (ax1, ax2), (ax3, ax4) = axes

    title = (fr"$\Omega_m$={s['omega_m']:.3f}, $\Omega_r$={s['omega_r']:.2g}, "
             fr"$\Omega_k$={s['omega_k']:.3f}, $\Omega_{{DE}}$={s['omega_de']:.3f}, "
             fr"$w_0$={s['w0']:.2f}, $w_a$={s['wa']:.2f}, $H_0$={s['H0_kms_mpc']:.1f} km/s/Mpc")
    fig.suptitle("FLRW expansion history\n" + title, fontsize=10.5)

    # --- Panel 1: a(t) ---
    ax1.plot(t, a, color=C_A, lw=lw, label="$a(t)$")
    ax1.set_yscale("log")
    ax1.axhline(1.0, color="k", lw=0.6, ls=":")
    if s["age_today_gyr"] is not None:
        ax1.axvline(s["age_today_gyr"], color="k", lw=0.8, ls="--",
                    label=fr"today, $t_0$={s['age_today_gyr']:.3g} Gyr")
    if s["turnaround"] is not None:
        ta = s["turnaround"]
        ax1.axvline(ta["t_turn_gyr"], color="crimson", lw=0.9, ls="-.",
                    label=fr"turnaround $t$={ta['t_turn_gyr']:.3g} Gyr")
    ax1.set_xlabel("Cosmic time $t$ [Gyr]")
    ax1.set_ylabel("Scale factor $a$")
    ax1.set_title("Scale-factor evolution")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, lw=0.3, alpha=0.5, which="both")

    # --- Panel 2: H(t) ---
    ax2.plot(t, H, color=C_H, lw=lw, label="$H(t)$")
    has_negative_H = np.any(H < 0.0)
    if has_negative_H:
        # H < 0 on the mirrored contracting branch (H = a_dot/a, and
        # a_dot < 0 there); a plain log axis cannot show that sign
        # change at all, so use a symmetric-log axis instead, with a
        # small linear region around H=0 where the sign flips.
        finite_abs = np.abs(H[np.isfinite(H) & (H != 0.0)])
        linthresh = float(np.min(finite_abs)) if finite_abs.size else 1.0e-3
        ax2.set_yscale("symlog", linthresh=max(linthresh, 1.0e-12))
        ax2.axhline(0.0, color="k", lw=0.5, ls="-")
        # Automatic symlog decades pack a dozen +/- labels into the tiny
        # central region for a collapsing run. Choose signed anchors from
        # the actual H range; fixed 10^16 ticks would expand the axis and
        # hide smaller-scale models. Keep zero and one intermediate pair.
        highest_decade = float(np.floor(np.log10(np.max(finite_abs))))
        lowest_decade = float(np.floor(np.log10(max(linthresh, 1e-12))))
        outer = 10.0 ** highest_decade
        if highest_decade - lowest_decade >= 3:
            # Symlog's default formatter labels integer powers of ten;
            # half-decade positions remain unlabeled even if ticked.
            middle = 10.0 ** float(np.floor((highest_decade + lowest_decade) / 2.0))
            ax2.set_yticks([-outer, -middle, 0.0, middle, outer])
        else:
            ax2.set_yticks([-outer, 0.0, outer])
        if s["turnaround"] is not None:
            ax2.axvline(s["turnaround"]["t_turn_gyr"], color="crimson",
                        lw=0.9, ls="-.",
                        label="turnaround ($H=0$)")
    else:
        ax2.set_yscale("log")
    ax2.axhline(s["H0_kms_mpc"], color="k", lw=0.6, ls=":", label="$H_0$")
    if s["age_today_gyr"] is not None:
        ax2.axvline(s["age_today_gyr"], color="k", lw=0.8, ls="--")
    ax2.set_xlabel("Cosmic time $t$ [Gyr]")
    ax2.set_ylabel("$H(t)$ [km s$^{-1}$ Mpc$^{-1}$]" +
                   ("  (negative = contracting)" if has_negative_H else ""))
    ax2.set_title("Hubble parameter history")
    ax2.legend(fontsize=8)
    ax2.grid(True, lw=0.3, alpha=0.5, which="both")

    # --- Panel 3: Omega_i(a) ---
    ax3.plot(a, Om, color=C_M, lw=lw, label=r"$\Omega_m(a)$")
    if s["omega_r"] > 0:
        ax3.plot(a, Or, color=C_R, lw=lw, label=r"$\Omega_r(a)$")
    if abs(s["omega_k"]) > 1e-12:
        ax3.plot(a, Ok, color=C_K, lw=lw, label=r"$\Omega_k(a)$")
    if s["omega_de"] != 0:
        ax3.plot(a, Ode, color=C_DE, lw=lw, label=r"$\Omega_{DE}(a)$")
    ax3.plot(a, Om + Or + Ok + Ode, color=C_SUM, lw=0.8, ls=":",
             label=r"sum (=1 identically, except NaN at a turnaround)")
    if s["a_eq_rm"] is not None:
        ax3.axvline(s["a_eq_rm"], color=C_R, lw=0.8, ls="--", alpha=0.7)
    # Every detected matter-DE-equality crossing is marked, not just the
    # first (Codex Audit 8 P1-2C): a non-monotonic CPL history can have
    # more than one, and a plot showing only the first would visually
    # imply there is only one to see.
    for c in (s.get("a_eq_mde_crossings") or []):
        ax3.axvline(c["a"], color=C_DE, lw=0.8, ls="--", alpha=0.7)
    ax3.axvline(1.0, color="k", lw=0.6, ls=":")
    ax3.set_xscale("log")
    ax3.set_xlabel("Scale factor $a$")
    ax3.set_ylabel(r"$\Omega_i(a)$")
    ax3.set_title("Fractional energy-density content")
    # Omega_i(a) is only masked to NaN where E(a)^2<=0 exactly (see
    # physics_cosmo.omega_fractions); a point that is genuinely on the
    # expanding branch with a small but strictly positive E(a)^2 -- a
    # loitering model's near-double-root dip -- now correctly plots its
    # large but finite Omega values instead of being hidden as NaN
    # (Codex Audit 6 P2-1). Left unclamped, a single such spike (which
    # can be many orders of magnitude) would flatten the rest of this
    # panel to an unreadable line. Clip the DISPLAYED y-range only --
    # the underlying data (and the CSV export) are untouched -- to keep
    # the ordinary Omega_i in [0, 1] regime readable; a callout notes
    # when a spike has been clipped from view.
    y_lo, y_hi = -0.3, 1.3
    clipped = False
    for series in (Om, Or, Ok, Ode):
        finite = series[np.isfinite(series)]
        if finite.size and (finite.min() < y_lo or finite.max() > y_hi):
            clipped = True
    ax3.set_ylim(y_lo, y_hi)
    if clipped:
        ax3.text(0.02, 0.02,
                  "some Omega_i(a) excursions clipped from view\n"
                  "(full values remain in the returned data/CSV)",
                  transform=ax3.transAxes, fontsize=7, color="0.4",
                  va="bottom", ha="left")
    ax3.legend(fontsize=8)
    ax3.grid(True, lw=0.3, alpha=0.5, which="both")

    # --- Panel 4: q(t) ---
    ax4.plot(t, q, color=C_Q, lw=lw, label="$q(t)$")
    ax4.axhline(0.0, color="k", lw=0.7, ls="--")
    if s["age_today_gyr"] is not None:
        ax4.axvline(s["age_today_gyr"], color="k", lw=0.8, ls="--")
    ax4.set_xlabel("Cosmic time $t$ [Gyr]")
    ax4.set_ylabel("Deceleration parameter $q(t)$")
    ax4.set_title("Deceleration $\\to$ acceleration (from the 2nd Friedmann eq.)")
    ax4.legend(fontsize=8)
    ax4.grid(True, lw=0.3, alpha=0.5)

    ann = [f"age today = {s['age_today_gyr']:.4g} Gyr" if s["age_today_gyr"] else "age today = n/a"]
    if s["z_accel"] is not None:
        ann.append(f"accel. transition at z={s['z_accel']:.3g}")
    if s["turnaround"] is not None:
        ann.append(f"recollapses at a={s['turnaround']['a_turn']:.3g}")
    if future is not None:
        tr = future["t_rip_gyr"]
        for ax, values, color in ((ax1, future["a"], C_A),
                                  (ax2, future["H_kms_mpc"], C_H),
                                  (ax4, future["q"], C_Q)):
            ax.plot(future["t_gyr"], values, color=color, lw=lw, ls="--",
                    label="future continuation (plot only)")
            ax.axvline(tr, color="crimson", lw=1.0, ls=":",
                       label="Big Rip" if ax is ax1 else None)
            span = tr - float(t[0])
            ax.set_xlim(float(t[0]) - 0.03 * span, tr + 0.05 * span)
        ax1.plot(t[-1], a[-1], "o", color=C_A, ms=4)
        ax1.annotate(r"$a\to\infty$", xy=(tr, 0.99),
                     xytext=(0.76, 0.80), xycoords=("data", "axes fraction"),
                     textcoords="axes fraction", ha="center", fontsize=10,
                     arrowprops=dict(arrowstyle="->", color=C_A))
        ax1.legend(fontsize=7, loc="lower right")
        ax2.legend(fontsize=7, loc="upper right")
        ax4.legend(fontsize=7, loc="upper right")
        clock = "t" if s["age_today_gyr"] is not None else "run time"
        ann.append(f"Big Rip (full model): {clock} ~ {tr:.4g} Gyr")
        fig.supxlabel(
            f"Solid curves: numerical run to a={a[-1]:g} (dot). "
            "Dashed: same-model future, plotted only.\n"
            "Dotted red line: finite-time limit as a tends to infinity; "
            "not a simulated point.", fontsize=8)
        if s["age_today_gyr"] is None:
            for ax in (ax1, ax2, ax4):
                ax.set_xlabel(r"Time since $a_i$ [Gyr]")
        print(f"[plot_cosmo] Big Rip (full Friedmann continuation): "
              f"{clock} ~ {tr:.6g} Gyr; "
              "dashed future is plot-only, not additional CSV data.")
    elif s["big_rip_gyr"] is not None:
        ann.append(f"DE-only Big Rip estimate: t ~ {s['big_rip_gyr']:.3g} Gyr")
    ax1.text(0.03, 0.97, "\n".join(ann), transform=ax1.transAxes, ha="left", va="top",
             fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow",
                                      ec="gray", alpha=0.9))

    _save_and_show(fig, outdir, "cosmo_evolve", dpi)


# ======================================================================
# Mode: compare
# ======================================================================
def plot_compare(names, results, outdir=None, dpi=150, lw=1.4, figsize=(13, 10)):
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
    (ax1, ax2), (ax3, ax4) = axes
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 2)))

    ages, age_labels, age_colors, missing_ages = [], [], [], []
    for name, r, col in zip(names, results, colors):
        s = r["summary"]
        label = f"{name}"
        ax1.plot(r["t_gyr"], r["a"], color=col, lw=lw, label=label)
        if s["age_today_gyr"] is not None:
            ax1.plot(s["age_today_gyr"], 1.0, "o", color=col, ms=5)
        ax2.plot(r["t_gyr"], r["q"], color=col, lw=lw, label=label)
        ax4.plot(r["a"], r["Ode"], color=col, lw=lw, label=label)

        ages.append(s["age_today_gyr"] if s["age_today_gyr"] is not None else 0.0)
        age_labels.append(name)
        age_colors.append(col)
        if s["age_today_gyr"] is None:
            if s.get("past_status") == "past_eternal":
                reason = "past-eternal\n(no finite age)"
            elif s.get("elapsed_ai_to_today_gyr") is None:
                reason = "$a=1$ not reached"
            else:
                reason = "age unavailable"
            missing_ages.append((len(ages) - 1, reason))

    ax1.axhline(1.0, color="k", lw=0.5, ls=":")
    ax1.set_yscale("log")
    ax1.set_xlabel("Cosmic time $t$ [Gyr]")
    ax1.set_ylabel("Scale factor $a$")
    ax1.set_title("Scale-factor histories (dot = present, $a=1$)")
    ax1.legend(fontsize=7.5)
    ax1.grid(True, lw=0.3, alpha=0.5, which="both")

    ax2.axhline(0.0, color="k", lw=0.6, ls="--")
    ax2.set_xlabel("Cosmic time $t$ [Gyr]")
    ax2.set_ylabel("Deceleration parameter $q(t)$")
    ax2.set_title("Expansion-history comparison: $q(t)$")
    # q formally diverges at a recollapsing model's turnaround (H -> 0
    # while the densities stay finite); clip the axis rather than let one
    # curve's divergence swamp the others, which are the point of the plot.
    ax2.set_ylim(-1.6, 3.0)
    ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(True, lw=0.3, alpha=0.5)

    ax3.bar(age_labels, ages, color=age_colors)
    for i, reason in missing_ages:
        # A cross and a reason replace a numeric age; zero is only the
        # plotting baseline and must not look like a measured zero age.
        ax3.plot(i, 0, "x", color=age_colors[i], ms=7, clip_on=False)
        ax3.text(i, 0.04, reason, transform=ax3.get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=7)
    ax3.set_ylabel("Age today [Gyr]")
    ax3.set_title("Age of the universe by cosmology")
    ax3.tick_params(axis="x", labelrotation=30)
    ax3.grid(True, lw=0.3, alpha=0.5, axis="y")

    ax4.set_xscale("log")
    ax4.set_xlabel("Scale factor $a$")
    ax4.set_ylabel(r"$\Omega_{DE}(a)$")
    ax4.set_title("Dark-energy fraction of the total density")
    ax4.legend(fontsize=7.5)
    ax4.grid(True, lw=0.3, alpha=0.5, which="both")

    fig.suptitle("Comparing cosmologies", fontsize=11)
    _save_and_show(fig, outdir, "cosmo_compare", dpi)


# ======================================================================
# Mode: age scan
# ======================================================================
_LABELS = {
    "omega_m": r"$\Omega_{m0}$",
    "omega_de": r"$\Omega_{DE,0}$",
    "w0": r"$w_0$",
    "H0": r"$H_0$ [km/s/Mpc]",
}


def plot_age_scan(scan, outdir=None, dpi=150, lw=1.8, figsize=(12, 5.5)):
    param = scan["scan_param"]
    x = scan["values"]
    xl = _LABELS.get(param, param)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax1.plot(x, scan["ages"], color=C_A, lw=lw, marker="o", ms=3)
    ax1.axhline(scan["age_ref_gyr"], color=C_REF, lw=1.0, ls="--",
                label=f"reference age = {scan['age_ref_gyr']:g} Gyr")
    rc_before = scan.get("recollapsed_before_today", scan["recollapsed"])
    if np.any(rc_before):
        ax1.plot(x[rc_before], np.zeros(np.sum(rc_before)), "x", color="crimson",
                 label="recollapses before $a=1$ (no age defined)")
    rc_future = scan["recollapsed"] & ~rc_before
    if np.any(rc_future):
        ax1.plot(x[rc_future], scan["ages"][rc_future], "^", color="darkorange",
                 ms=6, mfc="none", label="reaches $a=1$, recollapses later")
    ax1.set_xlabel(xl)
    ax1.set_ylabel("Age of the universe today [Gyr]")
    ax1.set_title("Age vs. cosmological parameter")
    ax1.legend(fontsize=8)
    ax1.grid(True, lw=0.3, alpha=0.5)

    ax2.plot(x, scan["h0t0"], color=C_H, lw=lw, marker="o", ms=3)
    ax2.axhline(2.0 / 3.0, color="gray", lw=0.8, ls=":", label=r"$H_0 t_0=2/3$ (EdS)")
    ax2.axhline(1.0, color="gray", lw=0.8, ls="-.", label=r"$H_0 t_0=1$ (Milne)")
    ax2.set_xlabel(xl)
    ax2.set_ylabel(r"$H_0\,t_0$ (dimensionless)")
    ax2.set_title(r"$H_0 t_0$ vs. cosmological parameter")
    ax2.legend(fontsize=8)
    ax2.grid(True, lw=0.3, alpha=0.5)

    fig.suptitle(f"Age of the universe: scanning {xl}", fontsize=11)
    _save_and_show(fig, outdir, "cosmo_age_scan", dpi)
