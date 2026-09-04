"""
driver_nbg.py
=============
Orchestration layer for NbodyGalaxySimulator.

The driver validates the presentation-level inputs (output paths, figure
settings), dispatches to the requested physics_nbg.run_* simulation,
prints a run summary, writes an optional CSV diagnostics file, and hands
the result to the plotting layer.  All of the physics -- initial
conditions, the leapfrog engine, the Barnes-Hut and direct force laws,
and the diagnostics computed from the resulting trajectories -- lives in
physics_nbg.py; nothing here recomputes or duplicates it.
"""

import csv
import math
import os
import warnings
from datetime import datetime

import numpy as np

import physics_nbg as phys

# plot_nbg.py (and, through it, matplotlib.pyplot) is imported lazily,
# inside run_mode() itself, only on the code path that actually draws a
# figure -- NOT unconditionally at module load. A prior version imported
# it here unconditionally, which meant Matplotlib had to be installed and
# importable even for a plot-free, CSV-only run (`--no_plot --csvdir`):
# NumPy alone is genuinely sufficient for that case, and this program
# should not demand a plotting library it never calls into.

MODES = ("cluster", "galaxy", "chaos")

W = 68
SEP = "-" * W


# ======================================================================
# Validation helpers (style matches driver_sev.py)
# ======================================================================
def _finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_output(outdir, csvdir, dpi, lw):
    dpi = _finite("dpi", dpi)
    lw = _finite("lw", lw)
    if int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if not (10 <= dpi <= 1200):
        raise ValueError("dpi must lie between 10 and 1200.")
    if lw <= 0:
        raise ValueError("lw must be greater than zero.")
    for name, path in (("outdir", outdir), ("csvdir", csvdir)):
        if path is None:
            continue
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{name} must be a non-empty directory path.")
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(f"{name} = {path!r} exists but is not a directory.")
    return int(dpi), lw


# ======================================================================
# CSV output
# ======================================================================
# _provenance() below is passed result["summary"] -- the run's SUMMARY
# dict -- not the raw CLI keyword arguments, so every name here must be
# an actual summary key (e.g. "n_relax_requested", not "n_relax"; the
# summary dicts do not store the raw argument names verbatim). A
# mismatched or omitted name here would silently write "None" into the
# CSV provenance comments and the printed run header instead of failing
# loudly, so every key below must be an actual key of each run_*()
# summary dict.
# softening_explicit is listed alongside softening_pc so the comment
# records whether the value was chosen by the student or computed by
# athanassoula_softening() -- a bare resolved number cannot be told apart from
# an explicit override otherwise, defeating the reproducibility contract.
PARAMS_BY_MODE = {
    "cluster": ("n_bodies", "total_mass_msun", "scale_radius_pc",
                "n_relax_requested", "steps_per_crossing", "target_snapshots",
                "softening_pc", "softening_explicit", "theta", "method",
                "seed"),
    "galaxy":  ("n_bodies", "total_mass_msun", "radius_pc", "virial_ratio_init",
                "n_freefall_requested", "steps_per_freefall", "target_snapshots",
                "softening_pc", "softening_explicit", "theta",
                "method", "seed"),
    "chaos":   ("n_bodies", "total_mass_msun", "scale_radius_pc",
                "relative_perturbation", "n_cross_requested", "steps_per_crossing",
                "target_snapshots", "softening_pc", "softening_explicit",
                "theta", "method", "seed", "perturbation_seed"),
}


def _python_version():
    import sys
    return "{}.{}.{}".format(*sys.version_info[:3])


def _module_version(module):
    return getattr(module, "__version__", "unknown")


def _matplotlib_version():
    try:
        import matplotlib
        return matplotlib.__version__
    except ImportError:
        return "not installed"


def _provenance(mode, kw):
    """
    Comment lines recording exactly how a data file was produced.

    Only the parameters that the selected mode actually uses are listed,
    so a CSV file can never suggest that an irrelevant option had an
    effect on the numbers beside it.

    physics_nbg.BUILD_ID silently falls back to the string "unknown" if
    the core source files cannot be located or decoded at import time
    (e.g. some frozen or zipped distributions) -- a deliberately
    nonfatal fallback so the program still runs. But a provenance record
    that silently degrades is a provenance record a reader can't trust
    without checking by hand, so every time provenance is actually
    written out (here, the one chokepoint both the CSV and PNG-sidecar
    paths call), a concise warning is raised whenever BUILD_ID could not
    be resolved, while the run itself still completes.
    """
    if phys.BUILD_ID == "unknown":
        warnings.warn(
            "BUILD_ID could not be computed (core source files not found "
            "or not decodable next to physics_nbg.py); provenance below "
            "records BUILD_ID as 'unknown' and cannot be used to verify "
            "which source revision produced this output.",
            RuntimeWarning, stacklevel=2,
        )
    lines = [
        f"NbodyGalaxySimulator version {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID})",
        f"mode = {mode}",
        f"run at {datetime.now().isoformat(timespec='seconds')}",
        f"environment: Python {_python_version()}, NumPy {_module_version(np)}"
        f", Matplotlib {_matplotlib_version()}",
        "parameters actually used by this mode:",
    ]
    for name in PARAMS_BY_MODE[mode]:
        value = kw.get(name)
        if name == "softening_pc":
            # kw is the run's SUMMARY dict, which always stores the
            # resolved numeric softening actually used (never None) --
            # softening_explicit (recorded separately, right below this
            # line) is what distinguishes a student-supplied override
            # from the default computed by athanassoula_softening() from
            # n_bodies and the scale radius (Athanassoula et al. 2000
            # optimal-softening scaling; see that function's docstring).
            suffix = "" if kw.get("softening_explicit") else \
                " (default: computed by athanassoula_softening(), the " \
                "Athanassoula et al. 2000 optimal-softening scaling)"
            lines.append(f"    {name} = {value}{suffix}")
            continue
        lines.append(f"    {name} = {value}")
    lines.append("options belonging to the other modes were not used")
    return lines


def _write_csv(csvdir, prefix, header, rows, comments=()):
    os.makedirs(csvdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(csvdir, f"{prefix}_{stamp}.csv")
    if os.path.exists(path):
        # Two runs within the same second would otherwise silently
        # overwrite each other's output with no warning.
        n = 2
        while os.path.exists(
                candidate := os.path.join(csvdir, f"{prefix}_{stamp}_{n}.csv")):
            n += 1
        path = candidate
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[driver_nbg] CSV saved -> {path}")
    return path


# virial_work_J is the scalar virial-theorem quantity Wvir =
# sum_i r_i.F_i (virial_force_term()), not the potential energy U
# (potential_energy()) -- the two coincide only as softening -> 0. It is
# recorded alongside potential_J so that virial_ratio = 2*kinetic_J /
# abs(virial_work_J) is independently checkable from the CSV.
CLUSTER_HEADER = ["t_Myr", "r10_pc", "r25_pc", "r50_pc", "r75_pc", "r90_pc",
                   "virial_ratio", "n_unbound", "high_velocity_fraction",
                   "kinetic_J", "potential_J", "virial_work_J", "energy_J"]
# GALAXY_HEADER is built to match _galaxy_rows() exactly, column for
# column, rather than derived from CLUSTER_HEADER by slicing -- galaxy
# mode has no escaper tracking, so it must not carry "n_unbound" or
# "high_velocity_fraction" columns that _galaxy_rows() never populates.
GALAXY_HEADER = ["t_Myr", "r10_pc", "r25_pc", "r50_pc", "r75_pc", "r90_pc",
                  "virial_ratio", "kinetic_J", "potential_J", "virial_work_J",
                  "energy_J"]
CHAOS_HEADER = ["t_Myr", "divergence_pc", "energy_a_J", "energy_b_J"]


def _cluster_rows(result):
    lag = result["lagrangian_radii"]
    rows = []
    for i in range(result["t"].size):
        rows.append([
            f"{result['t'][i] / phys.MYR:.8g}",
            *[f"{lag[i, j] / phys.PC:.6g}" for j in range(lag.shape[1])],
            f"{result['virial_ratio'][i]:.6g}",
            int(result["n_unbound"][i]),
            f"{result['high_velocity_fraction'][i]:.6g}",
            f"{result['kinetic'][i]:.6e}",
            f"{result['potential'][i]:.6e}",
            f"{result['virial_work'][i]:.6e}",
            f"{result['energy'][i]:.6e}",
        ])
    return rows


def _galaxy_rows(result):
    lag = result["lagrangian_radii"]
    rows = []
    for i in range(result["t"].size):
        rows.append([
            f"{result['t'][i] / phys.MYR:.8g}",
            *[f"{lag[i, j] / phys.PC:.6g}" for j in range(lag.shape[1])],
            f"{result['virial_ratio'][i]:.6g}",
            f"{result['kinetic'][i]:.6e}",
            f"{result['potential'][i]:.6e}",
            f"{result['virial_work'][i]:.6e}",
            f"{result['energy'][i]:.6e}",
        ])
    return rows


def _chaos_rows(result):
    rows = []
    for i in range(result["t"].size):
        rows.append([
            f"{result['t'][i] / phys.MYR:.8g}",
            f"{result['divergence'][i] / phys.PC:.6g}",
            f"{result['energy_a'][i]:.6e}",
            f"{result['energy_b'][i]:.6e}",
        ])
    return rows


# ======================================================================
# Summary printers
# ======================================================================
def _head(title):
    print(SEP)
    print(
        f"  NbodyGalaxySimulator {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID}) -- {title}"
    )
    print(SEP)


def _print_wrapped(text, indent="        "):
    line = indent
    for word in text.strip().split():
        if len(line) + len(word) + 1 > W:
            print(line)
            line = indent + word
        else:
            line = f"{line} {word}" if line.strip() else line + word
    if line.strip():
        print(line)


def _print_warnings(s):
    notes = s.get("warnings") or []
    if not notes:
        return
    print("  NOTES ON THIS RUN")
    for note in notes:
        _print_wrapped(note)
    print(SEP)


def _print_cluster_summary(s):
    _head("star-cluster relaxation")
    print(f"  Bodies              : {s['n_bodies']}  (each {s['m_body_msun']:.3f} Msun)")
    print(f"  Total mass          : {s['total_mass_msun']:.4g}  Msun")
    print(f"  Plummer scale a     : {s['scale_radius_pc']:.4g}  pc")
    print(f"  Softening           : {s['softening_pc']:.4g}  pc")
    print(f"  Force method        : {s['method']}  (theta = {s['theta']:.2f})")
    print(SEP)
    print(f"  Timestep            : {s['dt_myr']:.5g}  Myr  "
          f"({s['n_steps']:,} steps, {s['n_snapshots']} snapshots)")
    print(f"  Crossing time t0    : {s['t_cross0_myr']:.4g}  Myr")
    print(f"  Relaxation time t0  : {s['t_relax0_myr']:.4g}  Myr")
    print(f"  Total time run      : {s['total_time_myr']:.4g}  Myr "
          f"({s['n_relax_requested']:.3g} x t_relax)")
    print(SEP)
    print(f"  Half-mass radius    : {s['r50_initial_pc']:.4g} -> "
          f"{s['r50_final_pc']:.4g}  pc")
    print(f"  Virial ratio 2T/|W| : {s['virial_ratio_initial']:.4f} -> "
          f"{s['virial_ratio_final']:.4f}")
    print(f"  Unbound (instant.)  : {s['n_unbound_initial']} -> "
          f"{s['n_unbound_final']}  of {s['n_bodies']} "
          f"({s['unbound_fraction_final']:.2%})  [instantaneous, not a "
          "cumulative escape count -- see the Help file]")
    print(f"  Near-escape tail    : {s['high_velocity_fraction_initial']:.2%} -> "
          f"{s['high_velocity_fraction_final']:.2%}  "
          "(fraction above 90% of local escape speed)")
    print(f"  Max sampled energy drift: {s['max_fractional_energy_drift']:.3%}")
    print(SEP)
    if s["n_unbound_final"] == 0:
        # Checked against the ACTUAL parameters used, not just the
        # outcome: "expected at default settings" is only true when this
        # run actually used the default softening and n_relax -- a run
        # with explicit, nondefault softening/n_relax that still shows
        # zero instantaneous escapers is a genuinely different situation
        # (those settings did not produce the effect either, which is
        # worth saying plainly, not attributing to "the defaults").
        used_default_settings = (
            not s["softening_explicit"] and s["n_relax_requested"] == 5.0
        )
        print("  No body was instantaneously unbound at the end of this run.")
        if used_default_settings:
            print("  This is expected at the default softening and run length:")
            print("  two-body evaporation is a slow process (order 10^2 relaxation")
            print("  times for an isolated cluster), and force softening chosen")
            print("  for accuracy also suppresses the hard encounters that")
            print("  physically drive it.  The near-escape tail above is often")
            print("  ALSO zero throughout a default run for the same reason, not")
            print("  only near the very end -- a flat Lagrangian-radius/virial-")
            print("  ratio history at these defaults is not by itself evidence of")
            print("  strong two-body relaxation. See the Help file for the")
            print("  reduced-softening exercise that lets these diagnostics")
            print("  actually respond within a practical run.")
        else:
            print("  This run used explicit, nondefault softening and/or run")
            print("  length, so this is NOT automatically explained by the")
            print("  default settings' known suppression of two-body relaxation")
            print("  -- these particular settings simply did not produce a")
            print("  measurable escape signal either. See the Help file's")
            print("  reduced-softening exercise for parameter choices measured")
            print("  to actually respond within a practical run.")
    print(SEP)
    _print_warnings(s)


def _print_galaxy_summary(s):
    _head("galaxy-formation cold collapse")
    print(f"  Bodies              : {s['n_bodies']}  (each {s['m_body_msun']:.3g} Msun)")
    print(f"  Total mass          : {s['total_mass_msun']:.4g}  Msun")
    print(f"  Initial radius      : {s['radius_pc']:.4g}  pc")
    print(f"  Initial virial ratio: {s['virial_ratio_init']:.3f}  "
          "(0 = perfectly cold)")
    print(f"  Softening           : {s['softening_pc']:.4g}  pc")
    print(f"  Force method        : {s['method']}  (theta = {s['theta']:.2f})")
    print(SEP)
    print(f"  Timestep            : {s['dt_myr']:.5g}  Myr  "
          f"({s['n_steps']:,} steps, {s['n_snapshots']} snapshots)")
    print(f"  Free-fall time t_ff : {s['t_freefall_myr']:.4g}  Myr")
    print(f"  Total time run      : {s['total_time_myr']:.4g}  Myr "
          f"({s['n_freefall_requested']:.3g} x t_ff)")
    print(SEP)
    print(f"  Half-mass radius    : {s['r50_initial_pc']:.4g} pc (initial) -> "
          f"{s['r50_minimum_pc']:.4g} pc (deepest collapse, at "
          f"{s['time_of_deepest_collapse_myr']:.3g} Myr) -> "
          f"{s['r50_final_pc']:.4g} pc (final)")
    print(f"  Virial ratio 2T/|W| : {s['virial_ratio_initial']:.4f} -> "
          f"{s['virial_ratio_final']:.4f}  "
          f"(at deepest collapse: {s['virial_ratio_at_deepest_collapse']:.4f})")
    print(f"  Max sampled energy drift: {s['max_fractional_energy_drift']:.3%}")
    print(SEP)
    # Conditional on what THIS run's own inputs and outputs actually were,
    # not a fixed narrative printed regardless of them: a run that did not
    # start cold did not exhibit the classic cold-collapse-into-quasi-
    # equilibrium scenario, and saying so unconditionally would misdescribe
    # it.
    #
    # The quasi-equilibrium CLAIM itself (as opposed to the purely factual
    # "reached a sampled minimum, then expanded" statement above) is
    # decided ONLY by whether the run's own LATE-TIME window (the final
    # late_window_fraction of its elapsed time, by time -- not by snapshot
    # count) has actually settled: bounded fractional variation in the
    # half-mass radius, a small secular trend in it, and a bounded range in
    # the virial ratio over that same window -- never from a single final
    # value, a single earlier sampled minimum, or the sample count used to
    # store the run's diagnostics. Two runs with bit-for-bit identical
    # integrated trajectories that differ only in target_snapshots reach
    # the same verdict here, because the late window is defined by ELAPSED
    # TIME and requires a minimum number of stored samples in it before any
    # claim is made at all (see _late_time_window_stats()'s docstring) --
    # a run whose diagnostics were stored too sparsely to populate that
    # window is reported as such, not silently classified either way.
    started_cold = s["virial_ratio_initial"] <= 0.3
    late_ok = s["late_window_has_enough_snapshots"]
    late_settled = s["late_window_is_settled"]
    if s["r50_minimum_pc"] > 0.0:
        rebound_fraction = (
            (s["r50_final_pc"] - s["r50_minimum_pc"]) / s["r50_minimum_pc"]
        )
    else:
        rebound_fraction = 0.0

    if not started_cold:
        print("  This run did not start close to cold (Q_initial above 0.3), so")
        print("  it is not an instance of the classic cold-collapse-into-")
        print("  quasi-equilibrium scenario -- interpret its virial ratio history")
        print("  on its own terms, not against that specific benchmark.")
    elif not late_ok:
        print(f"  This run started close to cold (Q_initial near 0) and its half-mass")
        print(f"  radius fell to a SAMPLED minimum of {s['r50_minimum_pc']:.4g} pc at "
              f"{s['time_of_deepest_collapse_myr']:.3g} Myr, ending "
              f"{rebound_fraction:+.1%} away from that minimum -- but only "
              f"{s['late_window_n_snapshots']} snapshot(s) fall within the final "
              f"{s['late_window_fraction']:.0%} of the run's elapsed time, too few to")
        print("  assess whether the late-time half-mass radius and virial ratio have")
        print("  actually settled. Store more snapshots (raise target_snapshots) or")
        print("  run longer (raise n_freefall) before drawing any quasi-equilibrium")
        print("  conclusion; the sampled minimum above is itself sampling-dependent")
        print("  and may understate the true collapse.")
    elif late_settled:
        print(f"  This run started close to cold (Q_initial near 0). Over the final")
        print(f"  {s['late_window_fraction']:.0%} of its elapsed time "
              f"({s['late_window_n_snapshots']} stored snapshots from "
              f"{s['late_window_start_myr']:.3g} Myr onward), the half-mass radius")
        print(f"  varied by only {s['late_r50_fractional_range']:.1%} of its own mean "
              f"value and the virial ratio's range was {s['late_virial_ratio_range']:.3f}")
        print("  -- both within this program's documented 'modest' bounds. This is")
        print("  consistent with the classic cold-collapse scenario having settled")
        print("  into a quasi-equilibrium remnant through 'violent relaxation'")
        print("  (Lynden-Bell 1967): the sphere collapses, overshoots, and rebounds,")
        print("  then oscillates around a roughly steady state rather than")
        print("  continuing to expand, contract, or swing in Q. This is evidence")
        print("  CONSISTENT WITH approaching a virialized remnant, not proof of")
        print("  genuine phase-space (Boltzmann-equation) stationarity -- see the")
        print("  Help file's Domain of Validity section.")
    else:
        print(f"  This run started close to cold (Q_initial near 0) and reached a")
        print(f"  sampled half-mass-radius minimum of {s['r50_minimum_pc']:.4g} pc at "
              f"{s['time_of_deepest_collapse_myr']:.3g} Myr, but its final "
              f"{s['late_window_fraction']:.0%}")
        print(f"  of elapsed time ({s['late_window_n_snapshots']} stored snapshots) "
              f"shows a half-mass-radius range of {s['late_r50_fractional_range']:.1%} "
              "of its own mean value")
        print(f"  (linear trend {s['late_r50_linear_slope_pc_per_myr']:+.4g} pc/Myr) "
              f"and a virial-ratio range of {s['late_virial_ratio_range']:.3f}.")
        print("  This is too much drift and/or oscillation to call this run settled:")
        print("  it has NOT yet shown a genuine, sustained quasi-equilibrium remnant --")
        print("  it may still be mid-collapse, mid-rebound, or ringing through a")
        print("  later contraction. Run longer (raise n_freefall) before concluding")
        print("  otherwise; a single final virial ratio near 1 does not by itself")
        print("  establish settling (see above).")
    print("  Q near 1 alone is necessary but not sufficient for genuine")
    print("  dynamical equilibrium; see the Help file's Domain of Validity")
    print("  section before drawing physical conclusions from Q alone.")
    print(SEP)
    _print_warnings(s)


def _print_chaos_summary(s):
    _head("sensitivity to initial conditions (chaos)")
    print(f"  Bodies              : {s['n_bodies']}  (Plummer sphere, "
          f"a = {s['scale_radius_pc']:.4g} pc)")
    print(f"  Initial separation  : {s['initial_divergence_pc']:.3e}  pc  "
          f"(relative perturbation {s['relative_perturbation']:.1e})")
    print(f"  Softening           : {s['softening_pc']:.4g}  pc")
    print(f"  Force method        : {s['method']}  (theta = {s['theta']:.2f})")
    print(SEP)
    print(f"  Timestep            : {s['dt_myr']:.5g}  Myr  "
          f"({s['n_steps']:,} steps, {s['n_snapshots']} snapshots)")
    print(f"  Crossing time t0    : {s['t_cross0_myr']:.4g}  Myr")
    print(f"  Total time run      : {s['total_time_myr']:.4g}  Myr "
          f"({s['n_cross_requested']:.3g} x t_cross)")
    print(SEP)
    print(f"  Separation grew     : {s['initial_divergence_pc']:.3e} -> "
          f"{s['final_divergence_pc']:.4g}  pc  "
          f"(peak {s['max_divergence_pc']:.4g} pc)")
    if math.isfinite(s["lyapunov_exponent_per_myr"]) and s["lyapunov_exponent_per_myr"] > 0:
        # The qualifier sits on the SAME line as the number, not only in a
        # trailing footnote several lines below -- a reader who stops at
        # the number itself (as in a quick terminal scan, or a value
        # copied out of context) still sees that this is a finite-time
        # estimate from an automatically chosen window, not a certified,
        # held-out-validated Lyapunov exponent (see the Help file's
        # Domain of Validity section).
        print(f"  Lyapunov exponent*  : {s['lyapunov_exponent_per_myr']:.4g}  1/Myr  "
              "(finite-time growth-rate ESTIMATE, not a certified value)")
        print(f"  Lyapunov time*      : {s['lyapunov_time_myr']:.4g}  Myr  "
              f"({s['lyapunov_time_over_t_cross']:.3g} x crossing times; "
              "i.e. the e-folding time of that same estimate)")
        print(f"  Fit used            : {s['n_points_used_in_fit']} of "
              f"{s['n_snapshots']} snapshots  (R^2 = "
              f"{s['lyapunov_fit_r_squared']:.5f}, spanning "
              f"{s['lyapunov_fit_window_efolds']:.2f} e-folds, curvature t = "
              f"{s['lyapunov_fit_curvature_t_statistic']:.2f})")
        print("  * a heuristic finite-time fit to the measured divergence over an")
        print("  automatically chosen window, not a formal chaos test and not")
        print("  validated outside the specific model/parameter regime this")
        print("  program explores -- see the HTML notes.")
    else:
        print("  No clean exponential-growth window was found in this run; "
              "see the notes below.")
    print(f"  Max sampled energy drift : {s['max_fractional_energy_drift']:.3%}")
    print(SEP)
    _print_warnings(s)


# ======================================================================
# Public entry point
# ======================================================================
def run(mode="cluster",
        n_bodies=None, total_mass_msun=None, scale_radius_pc=None,
        radius_pc=None, virial_ratio_init=None,
        n_relax=None, n_freefall=None, n_cross=None,
        steps_per_crossing=None, steps_per_freefall=None,
        relative_perturbation=None,
        softening_pc=None, theta=None, method=None,
        target_snapshots=None, seed=None, perturbation_seed=None,
        outdir=None, csvdir=None, no_plot=False, dpi=150, lw=1.6):
    """
    Validate presentation-level inputs, run the requested mode, print a
    summary, optionally write a CSV, and hand the result to plot_nbg.py.

    Only the keyword arguments relevant to ``mode`` need be supplied;
    unset ones fall back to physics_nbg.run_{mode}'s own defaults (passed
    through as None here and simply omitted from the call below).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}.")
    dpi, lw = _validate_output(outdir, csvdir, dpi, lw)
    # A bare bool() call would silently coerce ANY value -- including the
    # string "False", which is truthy -- into True with no error, which
    # is exactly the kind of input mistake a type check exists to catch
    # for a boolean flag.
    if not isinstance(no_plot, bool):
        raise TypeError(f"no_plot must be a bool; got {type(no_plot).__name__}.")
    # --outdir controls only the figure that no_plot skips, so accepting
    # no_plot=True with outdir set but csvdir unset let a run "succeed"
    # while producing no artifact at all (a confirmed CLI run printed
    # "skipping figure generation despite --outdir being set" and left
    # the directory empty). csvdir is the only artifact no_plot leaves
    # available, so it is what is now required.
    if no_plot and csvdir is None:
        raise ValueError("no_plot requires --csvdir: --outdir alone controls "
                          "only the plot that no_plot skips, so a run with "
                          "no_plot and no csvdir would produce no output "
                          "at all.")

    def _kw(**pairs):
        return {k: v for k, v in pairs.items() if v is not None}

    if mode == "cluster":
        kwargs = _kw(n_bodies=n_bodies, total_mass_msun=total_mass_msun,
                     scale_radius_pc=scale_radius_pc, n_relax=n_relax,
                     steps_per_crossing=steps_per_crossing,
                     softening_pc=softening_pc, theta=theta, method=method,
                     target_snapshots=target_snapshots, seed=seed)
        result = phys.run_cluster(**kwargs)
        _print_cluster_summary(result["summary"])
        header, rows, prefix = CLUSTER_HEADER, _cluster_rows(result), "cluster"
    elif mode == "galaxy":
        kwargs = _kw(n_bodies=n_bodies, total_mass_msun=total_mass_msun,
                     radius_pc=radius_pc, virial_ratio_init=virial_ratio_init,
                     n_freefall=n_freefall, steps_per_freefall=steps_per_freefall,
                     softening_pc=softening_pc, theta=theta, method=method,
                     target_snapshots=target_snapshots, seed=seed)
        result = phys.run_galaxy(**kwargs)
        _print_galaxy_summary(result["summary"])
        header, rows, prefix = GALAXY_HEADER, _galaxy_rows(result), "galaxy"
    else:
        kwargs = _kw(n_bodies=n_bodies, total_mass_msun=total_mass_msun,
                     scale_radius_pc=scale_radius_pc,
                     relative_perturbation=relative_perturbation,
                     n_cross=n_cross, steps_per_crossing=steps_per_crossing,
                     softening_pc=softening_pc, theta=theta, method=method,
                     target_snapshots=target_snapshots, seed=seed,
                     perturbation_seed=perturbation_seed)
        result = phys.run_chaos(**kwargs)
        _print_chaos_summary(result["summary"])
        header, rows, prefix = CHAOS_HEADER, _chaos_rows(result), "chaos"

    if csvdir is not None:
        _write_csv(csvdir, prefix, header, rows,
                   comments=_provenance(mode, result["summary"]))

    if not no_plot:
        import plot_nbg as viz  # see the module-level comment on this import
        viz.plot_mode(mode, result, outdir=outdir, dpi=dpi, lw=lw,
                       provenance=_provenance(mode, result["summary"]))
    elif outdir is not None:
        print("[driver_nbg] no_plot=True: skipping figure generation "
              "despite --outdir being set.")

    return result
