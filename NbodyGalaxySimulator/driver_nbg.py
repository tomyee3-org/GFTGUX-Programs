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
PARAMS_BY_MODE = {
    "cluster": ("n_bodies", "total_mass_msun", "scale_radius_pc", "n_relax",
                "steps_per_crossing", "softening_pc", "theta", "method",
                "seed"),
    "galaxy":  ("n_bodies", "total_mass_msun", "radius_pc", "virial_ratio_init",
                "n_freefall", "steps_per_freefall", "softening_pc", "theta",
                "method", "seed"),
    "chaos":   ("n_bodies", "total_mass_msun", "scale_radius_pc",
                "relative_perturbation", "n_cross", "steps_per_crossing",
                "softening_pc", "theta", "method", "seed", "perturbation_seed"),
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
    """
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
        if name == "softening_pc" and value is None:
            value = "computed from Dehnen (2001) optimal-softening scaling (default)"
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


CLUSTER_HEADER = ["t_Myr", "r10_pc", "r25_pc", "r50_pc", "r75_pc", "r90_pc",
                   "virial_ratio", "n_escaped", "high_velocity_fraction",
                   "kinetic_J", "potential_J", "energy_J"]
GALAXY_HEADER = CLUSTER_HEADER[:-3] + ["kinetic_J", "potential_J", "energy_J"]
CHAOS_HEADER = ["t_Myr", "divergence_pc", "energy_a_J", "energy_b_J"]


def _cluster_rows(result):
    lag = result["lagrangian_radii"]
    rows = []
    for i in range(result["t"].size):
        rows.append([
            f"{result['t'][i] / phys.MYR:.8g}",
            *[f"{lag[i, j] / phys.PC:.6g}" for j in range(lag.shape[1])],
            f"{result['virial_ratio'][i]:.6g}",
            int(result["n_escaped"][i]),
            f"{result['high_velocity_fraction'][i]:.6g}",
            f"{result['kinetic'][i]:.6e}",
            f"{result['potential'][i]:.6e}",
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
    print(f"  Escaped (formal)    : {s['n_escaped_initial']} -> "
          f"{s['n_escaped_final']}  of {s['n_bodies']} "
          f"({s['evaporated_fraction_final']:.2%})")
    print(f"  Near-escape tail    : {s['high_velocity_fraction_initial']:.2%} -> "
          f"{s['high_velocity_fraction_final']:.2%}  "
          "(fraction above 90% of local escape speed)")
    print(f"  Max energy drift    : {s['max_fractional_energy_drift']:.3%}")
    print(SEP)
    if s["n_escaped_final"] == 0:
        print("  No body formally escaped in this run.  This is expected at the")
        print("  default softening and run length: two-body evaporation is a slow")
        print("  process (order 10^2 relaxation times for an isolated cluster),")
        print("  and force softening chosen for accuracy also suppresses the hard")
        print("  encounters that physically drive it.  Watch the near-escape tail")
        print("  and half-mass radius above for the same process at an earlier")
        print("  stage; see the Help file for the reduced-softening exercise that")
        print("  produces genuine escapers within a practical run.")
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
          f"{s['virial_ratio_final']:.4f}")
    print(f"  Max energy drift    : {s['max_fractional_energy_drift']:.3%}")
    print(SEP)
    print("  A perfectly cold sphere collapses, overshoots, and rebounds into a")
    print("  quasi-equilibrium remnant through 'violent relaxation' (Lynden-Bell")
    print("  1967).  Classic numerical experiments (e.g. van Albada 1982) find")
    print("  that this relaxation is generally INCOMPLETE: the final virial ratio")
    print("  commonly settles below 1 with an extended, non-Maxwellian halo,")
    print("  rather than reaching a clean Q = 1 equilibrium -- watch for that")
    print("  same signature above rather than expecting Q -> 1.")
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
        print(f"  Lyapunov exponent   : {s['lyapunov_exponent_per_myr']:.4g}  1/Myr")
        print(f"  Lyapunov time       : {s['lyapunov_time_myr']:.4g}  Myr  "
              f"({s['lyapunov_time_over_t_cross']:.3g} x crossing times)")
        print(f"  Fit used            : {s['n_points_used_in_fit']} of "
              f"{s['n_snapshots']} snapshots")
    else:
        print("  No clean exponential-growth window was found in this run; "
              "see the notes below.")
    print(f"  Max energy drift    : {s['max_fractional_energy_drift']:.3%}")
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
    no_plot = bool(no_plot)
    if no_plot and outdir is None and csvdir is None:
        raise ValueError("no_plot requires --csvdir or --outdir; otherwise "
                          "this run would produce no output at all.")

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
