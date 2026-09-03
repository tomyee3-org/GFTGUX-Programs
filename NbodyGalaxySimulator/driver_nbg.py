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
import plot_nbg as viz

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
# loudly, so every key below is checked against each run_*() summary
# dict's actual keys by
# TestCsvOutput.test_provenance_lines_match_actual_summary_values.
# softening_explicit is listed alongside softening_pc so the comment
# records whether the value was chosen by the student or computed by
# dehnen_softening() -- a bare resolved number cannot be told apart from
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
            # from the default computed by dehnen_softening() from
            # n_bodies and the scale radius (Athanassoula et al. 2000
            # optimal-softening scaling; see that function's docstring).
            suffix = "" if kw.get("softening_explicit") else \
                " (default: computed by dehnen_softening(), the " \
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
# "high_velocity_fraction" columns that _galaxy_rows() never populates;
# see TestCsvOutput.test_galaxy_csv_header_matches_row_length.
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
    _head("star-cluster evaporation")
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
        print("  No body was instantaneously unbound at the end of this run.  This")
        print("  is expected at the default softening and run length: two-body")
        print("  evaporation is a slow process (order 10^2 relaxation times for an")
        print("  isolated cluster), and force softening chosen for accuracy also")
        print("  suppresses the hard encounters that physically drive it.  Watch")
        print("  the near-escape tail and half-mass radius above for the same")
        print("  process at an earlier stage; see the Help file for the reduced-")
        print("  softening exercise that produces instantaneously-unbound bodies")
        print("  within a practical run.")
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
    print("  A perfectly cold sphere collapses, overshoots, and rebounds into a")
    print("  quasi-equilibrium remnant through 'violent relaxation' (Lynden-Bell")
    print("  1967).  The virial ratio above typically settles into a modest")
    print("  oscillation close to Q = 1 rather than converging to it exactly --")
    print("  watch for that oscillation, not a single settled number, as the")
    print("  signature of a properly virialized remnant.  A run ending far from")
    print("  Q = 1 (well below or above) more often means the run has not yet")
    print("  had time to virialize, or dt/softening need tightening, than it")
    print("  means genuine incomplete relaxation -- see the Help file's Domain")
    print("  of Validity section before drawing physical conclusions from Q")
    print("  alone.")
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
        print(f"  Lyapunov exponent*  : {s['lyapunov_exponent_per_myr']:.4g}  1/Myr")
        print(f"  Lyapunov time*      : {s['lyapunov_time_myr']:.4g}  Myr  "
              f"({s['lyapunov_time_over_t_cross']:.3g} x crossing times)")
        print(f"  Fit used            : {s['n_points_used_in_fit']} of "
              f"{s['n_snapshots']} snapshots  (R^2 = "
              f"{s['lyapunov_fit_r_squared']:.5f}, "
              f"{s['lyapunov_fit_residual_sign_changes']} residual sign changes)")
        print("  * heuristic finite-time fit to the measured divergence, not "
              "a formal chaos test -- see the HTML notes.")
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
        viz.plot_mode(mode, result, outdir=outdir, dpi=dpi, lw=lw,
                       provenance=_provenance(mode, result["summary"]))
    elif outdir is not None:
        print("[driver_nbg] no_plot=True: skipping figure generation "
              "despite --outdir being set.")

    return result
