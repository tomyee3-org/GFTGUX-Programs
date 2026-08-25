"""
driver_bh.py
============
Orchestration layer for Black_hole_spacetime_visualizer.

The driver validates the presentation-level inputs, dispatches to the
requested physics calculation, prints a run summary, writes optional CSV
data files, and hands the result to the plotting layer.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np

import physics_bh as phys
import plot_bh as viz

MODES = phys.MODES

W = 70
SEP = "-" * W


# ======================================================================
# Validation helpers (same conventions as driver_sev.py)
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


def _parse_float_list(name, text, lo=None, hi=None, max_items=None):
    """Parse a comma-separated list of numbers supplied on the command line."""
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [piece for piece in str(text).replace(";", ",").split(",")
                 if piece.strip()]
    if not items:
        raise ValueError(f"{name} did not contain any numbers.")
    values = []
    for piece in items:
        try:
            v = float(piece)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} contains {piece!r}, which is not a number."
            ) from exc
        if not math.isfinite(v):
            raise ValueError(f"{name} contains a non-finite value.")
        if lo is not None and v < lo:
            raise ValueError(f"{name} contains {v:g}, which is below {lo:g}.")
        if hi is not None and v > hi:
            raise ValueError(f"{name} contains {v:g}, which is above {hi:g}.")
        values.append(v)
    if max_items is not None and len(values) > max_items:
        raise ValueError(f"{name} may contain at most {max_items} entries.")
    return values


# ======================================================================
# CSV output
# ======================================================================
PARAMS_BY_MODE = {
    "embed":    ("M", "r_max_rs", "n_r"),
    "tidal":    ("M", "r_min_rs", "r_max_rs", "n_r", "separation", "limit_g",
                 "compare_masses"),
    "infall":   ("M", "r0_rs", "n_points", "r_stop_rs", "step_frac"),
    "horizons": ("M0", "M1", "v1_rs0", "duration_rs0", "v_start_margin_rs0",
                 "v_end_margin_rs0", "n_steps", "bisect_iters", "n_family"),
}


def _provenance(mode, kw):
    """Comment lines recording exactly how a data file was produced."""
    lines = [
        f"Black_hole_spacetime_visualizer version "
        f"{phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID})",
        f"mode = {mode}",
        f"run at {datetime.now().isoformat(timespec='seconds')}",
        "parameters actually used by this mode:",
    ]
    for name in PARAMS_BY_MODE[mode]:
        lines.append(f"    {name} = {kw.get(name)}")
    lines.append("options belonging to the other modes were not used")
    return lines


def _write_csv(csvdir, prefix, header, rows, comments=()):
    os.makedirs(csvdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(csvdir, f"{prefix}_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[driver_bh] CSV saved -> {path}")
    return path


# ======================================================================
# Summary printers
# ======================================================================
def _head(title):
    print(SEP)
    print(
        f"  Black_hole_spacetime_visualizer "
        f"{phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID}) — {title}")
    print(SEP)


def _print_note_lines(notes):
    """Word-wrap and print a list of note strings, no header or separator."""
    for note in notes:
        text = note.strip()
        line = "        "
        for word in text.split():
            if len(line) + len(word) + 1 > W:
                print(line)
                line = "        " + word
            else:
                line = f"{line} {word}" if line.strip() else line + word
        if line.strip():
            print(line)


def _print_warnings(s):
    notes = s.get("warnings") or []
    if not notes:
        return
    print("  NOTES ON THIS RUN")
    _print_note_lines(notes)
    print(SEP)


def _print_embed_summary(s):
    _head("Flamm's paraboloid -- embedding diagram")
    print(f"  Mass                : {s['m_msun']:.4g}  Msun")
    print(f"  Schwarzschild radius: {s['rs_km']:.4g}  km")
    print(f"  Plotted out to      : {s['r_max_rs']:.3g}  r_s")
    print(f"  Points              : {s['n_points']:,}")
    print(SEP)
    print(f"  {s['throat_note']}")
    print(SEP)
    _print_warnings(s)


def _print_tidal_summary(s):
    _head("tidal (geodesic-deviation) acceleration vs. radius")
    print(f"  Mass                : {s['m_msun']:.4g}  Msun")
    print(f"  Schwarzschild radius: {s['rs_km']:.4g}  km")
    print(f"  Test separation     : {s['separation_m']:.3g}  m")
    print(f"  Range plotted       : {s['r_min_rs']:.3g} - {s['r_max_rs']:.3g}  r_s"
          f"  ({s['n_points']:,} points)")
    print(SEP)
    print(f"  Radial (stretching) tidal accel. at the horizon:")
    print(f"      {s['a_radial_horizon']:.4e}  m/s^2  = {s['a_radial_horizon_g']:.4e}  g")
    print(f"  Tangential (compressing) tidal accel. at the horizon:")
    print(f"      {s['a_tangential_horizon']:.4e}  m/s^2  = {s['a_tangential_horizon_g']:.4e}  g")
    print(SEP)
    _print_warnings(s)


def _print_tidal_compare(rows, separation_m, limit_g):
    print(f"  Comparison across masses (separation = {separation_m:.3g} m,"
          f" survival limit = {limit_g:.3g} g)")
    print(f"  {'M [Msun]':>12}  {'r_s [km]':>12}  {'a_r(r_s) [g]':>14}"
          f"  {'r_crit/r_s':>11}  {'survives?':>10}")
    for row in rows:
        verdict = "yes" if row["survives_horizon"] else "no"
        flag = "  *" if row.get("warnings") else ""
        print(f"  {row['m_msun']:12.4g}  {row['rs_km']:12.4g}"
              f"  {row['a_radial_horizon_g']:14.4e}"
              f"  {row['r_crit_over_rs']:11.4g}  {verdict:>10}{flag}")
    if any(row.get("warnings") for row in rows):
        print("  * outside the astrophysically known black-hole mass range:")
        all_notes = [note for row in rows for note in row.get("warnings", [])]
        _print_note_lines(all_notes)
    print(SEP)


def _print_infall_summary(s):
    _head("radial infall -- proper time, coordinate time and redshift")
    print(f"  Mass                : {s['m_msun']:.4g}  Msun")
    print(f"  Schwarzschild radius: {s['rs_km']:.4g}  km")
    print(f"  Released from rest at r0 = {s['r0_rs']:.4g}  r_s"
          f"   (E = {s['E']:.6f})")
    print(f"  Integration stopped at r = {s['r_stop_rs']:.5g}  r_s")
    print(SEP)
    print(f"  Proper time elapsed (falling observer's own clock):")
    print(f"      tau = {s['tau_total_ms']:.8g}  ms")
    print(f"  Schwarzschild coordinate time elapsed (distant observer's clock):")
    print(f"      t   = {s['t_total_ms']:.8g}  ms   (t/tau = {s['t_total_ms']/s['tau_total_ms']:.4g})")
    print("      (printed to 8 figures so EXP-11's 1e-5-level closed-form"
          " comparison can be read off directly; the CSV file carries the"
          " same precision internally either way)")
    print(f"  Local infall speed at the last recorded point : {s['v_local_final']:.6f}  c")
    print(f"  Redshift factor (nu_obs/nu_emit) at that point: {s['redshift_final']:.4e}")
    print(SEP)
    print("  t and the redshift factor are both heading toward, respectively,")
    print("  infinity and zero as r -> r_s; tau is not.  A distant observer")
    print("  never sees the infall finish, only ever more redshifted and delayed.")
    print(SEP)
    _print_warnings(s)


def _print_horizons_summary(s):
    _head("Vaidya spacetime -- event horizon vs. apparent horizon")
    print(f"  Initial mass M0     : {s['m0_msun']:.4g}  Msun"
          f"   (r_s0 = {s['rs0_km']:.4g} km)")
    print(f"  Final mass M1       : {s['m1_msun']:.4g}  Msun"
          f"   (r_s1 = {s['rs1_km']:.4g} km)")
    print(f"  Accretion window    : v = {s['v1_rs0']:.3g}  to  {s['v2_rs0']:.3g}"
          f"   r_s0  (duration {s['duration_rs0']:.3g} r_s0)")
    print(f"  v range integrated  : {s['v_start_rs0']:.3g}  to  {s['v_end_rs0']:.3g}  r_s0")
    print(f"  r_s0/c              : {s['light_crossing_time_rs0_s']*1e6:.4g}  microseconds")
    print(SEP)
    print(f"  Event-horizon generator located at r/r_s0 = "
          f"{s['r_crit_over_rs0']:.8f} at v_start")
    print(f"  Bisection bracket width : {s['residual_rs0']:.3e}  r_s0"
          f"   ({s['bisect_iters']} iterations)")
    print("      (this is only the bisection's own precision; it is not a")
    print("       measure of the separate, generally larger error from the")
    print("       finite --v_start_margin_rs0 -- see the help file and EXP-10)")
    print(SEP)
    no_accretion = (s["m0_msun"] == s["m1_msun"])
    if no_accretion:
        print("  Apparent horizon r_AH(v) = 2M(v): local, reads the mass function")
        print("  alone.  Event horizon r_EH(v): global, found by shooting outgoing")
        print("  light rays to the far future.  Here M0 = M1, so the hole never")
        print("  accretes: the mass function is constant, and the apparent and")
        print("  event horizons coincide at r = r_s0 for the whole run, as they")
        print("  always do in a static (unchanging) spacetime.")
    else:
        print("  Apparent horizon r_AH(v) = 2M(v): local, reads the mass function")
        print("  alone.  Event horizon r_EH(v): global, found by shooting outgoing")
        print("  light rays to the far future.  r_EH >= r_AH always; the two")
        print("  coincide only once the mass has stopped growing (v > v2), and the")
        print("  event horizon starts rising measurably *before* the accretion")
        print("  begins (v < v1) -- it anticipates mass that has not arrived yet.")
    print(SEP)
    _print_warnings(s)


# ======================================================================
# Mode runners
# ======================================================================
def _run_embed(kw, outdir, csvdir, dpi, lw):
    result = phys.embedding_profile(m_msun=kw["M"], r_max_rs=kw["r_max_rs"],
                                    n_r=kw["n_r"])
    _print_embed_summary(result["summary"])
    if csvdir is not None:
        s = result["summary"]
        rows = [[f"{r/s['rs_m']:.6f}", f"{r:.6g}", f"{z:.6g}", f"{d:.6g}"]
                for r, z, d in zip(result["r"], result["z"],
                                    result["proper_radial_distance"])]
        _write_csv(csvdir, f"bh_embed_{s['m_msun']:.2f}Msun",
                   ["r_over_rs", "r_m", "z_m", "proper_radial_distance_m"],
                   rows,
                   comments=["Flamm's paraboloid embedding profile"]
                   + _provenance("embed", kw))
    if not kw["no_plot"]:
        viz.plot_embedding(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_tidal(kw, outdir, csvdir, dpi, lw):
    # limit_g is validated here, unconditionally, because plot_tidal draws
    # the illustrative survival-limit line on every tidal plot -- not only
    # when --compare_masses is supplied and the physics layer's own
    # validation (inside compare_tidal_across_masses) actually runs.
    kw["limit_g"] = _finite("limit_g", kw["limit_g"])
    if kw["limit_g"] <= 0.0:
        raise ValueError(f"limit_g must be greater than zero; got {kw['limit_g']:g}.")

    result = phys.tidal_profile(m_msun=kw["M"], r_min_rs=kw["r_min_rs"],
                                r_max_rs=kw["r_max_rs"], n_r=kw["n_r"],
                                separation_m=kw["separation"])
    _print_tidal_summary(result["summary"])

    compare_rows = None
    compare_masses = _parse_float_list("compare_masses", kw["compare_masses"],
                                       lo=1.0e-6, max_items=12)
    if compare_masses:
        compare_rows = phys.compare_tidal_across_masses(
            compare_masses, separation_m=kw["separation"], limit_g=kw["limit_g"])
        _print_tidal_compare(compare_rows, kw["separation"], kw["limit_g"])

    if csvdir is not None:
        s = result["summary"]
        rows = [[f"{r/s['rs_m']:.6f}", f"{r:.6g}", f"{ar:.6e}", f"{ar/phys.g0:.6e}",
                 f"{at:.6e}", f"{at/phys.g0:.6e}"]
                for r, ar, at in zip(result["r"], result["a_radial"],
                                     result["a_tangential"])]
        _write_csv(csvdir, f"bh_tidal_{s['m_msun']:.2f}Msun",
                   ["r_over_rs", "r_m", "a_radial_m_s2", "a_radial_g",
                    "a_tangential_m_s2", "a_tangential_g"], rows,
                   comments=["radial (stretching) and tangential (compressing) "
                             "tidal acceleration vs. radius"]
                   + _provenance("tidal", kw))
        if compare_rows:
            crows = [[f"{r['m_msun']:.6g}", f"{r['rs_km']:.6g}",
                      f"{r['a_radial_horizon_g']:.6e}", f"{r['r_crit_over_rs']:.6g}",
                      "yes" if r["survives_horizon"] else "no"]
                     for r in compare_rows]
            _write_csv(csvdir, "bh_tidal_compare",
                       ["M_Msun", "rs_km", "a_radial_horizon_g",
                        "r_crit_over_rs", "survives_horizon"], crows,
                       comments=[f"tidal comparison across masses, separation = "
                                 f"{kw['separation']:g} m, limit = {kw['limit_g']:g} g"]
                       + _provenance("tidal", kw))
    if not kw["no_plot"]:
        viz.plot_tidal(result, compare_rows, kw["separation"], kw["limit_g"],
                       outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_infall(kw, outdir, csvdir, dpi, lw):
    result = phys.infall_radial(m_msun=kw["M"], r0_rs=kw["r0_rs"],
                                n_points=kw["n_points"], r_stop_rs=kw["r_stop_rs"],
                                step_frac=kw["step_frac"])
    _print_infall_summary(result["summary"])
    if csvdir is not None:
        s = result["summary"]
        rows = [[f"{tau*1e3:.6g}", f"{t*1e3:.6g}", f"{r/s['rs_m']:.6f}",
                 f"{v:.6f}", f"{z:.6e}", f"{d:.6f}"]
                for tau, t, r, v, z, d in zip(
                    result["tau"], result["t"], result["r"],
                    result["v_local"], result["redshift"], result["dtau_dt"])]
        _write_csv(csvdir, f"bh_infall_{s['m_msun']:.2f}Msun",
                   ["tau_ms", "t_coord_ms", "r_over_rs", "v_local_over_c",
                    "redshift_ratio", "dtau_dt"], rows,
                   comments=["radial free fall from rest: proper time, "
                             "coordinate time, local speed and redshift"]
                   + _provenance("infall", kw))
    if not kw["no_plot"]:
        viz.plot_infall(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


def _run_horizons(kw, outdir, csvdir, dpi, lw):
    result = phys.vaidya_horizons(
        m0_msun=kw["M0"], m1_msun=kw["M1"], v1_rs0=kw["v1_rs0"],
        duration_rs0=kw["duration_rs0"],
        v_start_margin_rs0=kw["v_start_margin_rs0"],
        v_end_margin_rs0=kw["v_end_margin_rs0"],
        n_steps=kw["n_steps"], bisect_iters=kw["bisect_iters"],
        n_family=kw["n_family"],
    )
    _print_horizons_summary(result["summary"])
    if csvdir is not None:
        s = result["summary"]
        rows = [[f"{v/s['rs0_m']:.6f}", f"{v:.6g}",
                 f"{rah/s['rs0_m']:.6f}", f"{reh/s['rs0_m']:.6f}",
                 f"{rah:.6g}", f"{reh:.6g}"]
                for v, rah, reh in zip(result["v"], result["r_AH"], result["r_EH"])]
        _write_csv(csvdir, f"bh_horizons_{s['m0_msun']:.2f}to{s['m1_msun']:.2f}Msun",
                   ["v_over_rs0", "v_m", "r_AH_over_rs0", "r_EH_over_rs0",
                    "r_AH_m", "r_EH_m"], rows,
                   comments=["apparent horizon r_AH(v) = 2M(v) vs. the "
                             "shooting-method event horizon r_EH(v)"]
                   + _provenance("horizons", kw)
                   + [f"event-horizon bisection residual = "
                      f"{s['residual_rs0']:.3e} r_s0"])
        frows = []
        for j, fam in enumerate(result["family"]):
            for v, r in zip(fam["v"], fam["r"]):
                frows.append([j, f"{fam['r_i_over_rs0']:.6f}",
                              "yes" if fam["escapes"] else "no",
                              f"{v/s['rs0_m']:.6f}", f"{r/s['rs0_m']:.6f}"])
        _write_csv(csvdir, "bh_horizons_family",
                   ["trajectory_id", "r_i_over_rs0", "escapes",
                    "v_over_rs0", "r_over_rs0"], frows,
                   comments=["family of outgoing null geodesics bracketing "
                             "the event-horizon generator"]
                   + _provenance("horizons", kw))
    if not kw["no_plot"]:
        viz.plot_horizons(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


# ======================================================================
# Public entry point
# ======================================================================
def run(mode="embed",
        # --- embed ---
        M=10.0, r_max_rs=8.0, n_r=400,
        # --- tidal ---
        r_min_rs=1.01, separation=1.8, limit_g=100.0, compare_masses=None,
        # --- infall ---
        r0_rs=6.0, n_points=4000, r_stop_rs=1.0005, step_frac=0.02,
        # --- horizons ---
        M0=5.0, M1=10.0, v1_rs0=5.0, duration_rs0=10.0,
        v_start_margin_rs0=25.0, v_end_margin_rs0=15.0,
        n_steps=6000, bisect_iters=60, n_family=9,
        # --- output ---
        outdir=None, csvdir=None, dpi=150, lw=1.6, no_plot=False):
    """
    Run one Black_hole_spacetime_visualizer calculation.

    mode selects the calculation:
        "embed"     Flamm's paraboloid embedding diagram
        "tidal"     radial and tangential tidal acceleration vs. radius
        "infall"    radial free fall: proper time, coordinate time, redshift
        "horizons"  Vaidya apparent horizon vs. event horizon
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}.")

    dpi, lw = _validate_output(outdir, csvdir, dpi, lw)
    # --no_plot skips the figure (screen display AND any PNG) entirely,
    # exactly as documented in main.py's own --no_plot --help text and in
    # the help file's Parameters and Algorithm sections ("no screen
    # display and no PNG, regardless of --outdir"). An --outdir supplied
    # alongside --no_plot is therefore simply unused, not an error --
    # main.py's own help text says so explicitly ("regardless of
    # --outdir"), so rejecting the combination here would silently
    # contradict the documented behaviour, e.g. every time a student left
    # --outdir set in a shell loop while adding --no_plot for a sweep.
    if no_plot and csvdir is None:
        raise ValueError(
            "no_plot was requested but no csvdir was given, so the run would "
            "produce no output at all."
        )

    kw = dict(
        M=M, r_max_rs=r_max_rs, n_r=n_r,
        r_min_rs=r_min_rs, separation=separation, limit_g=limit_g,
        compare_masses=compare_masses,
        r0_rs=r0_rs, n_points=n_points, r_stop_rs=r_stop_rs, step_frac=step_frac,
        M0=M0, M1=M1, v1_rs0=v1_rs0, duration_rs0=duration_rs0,
        v_start_margin_rs0=v_start_margin_rs0, v_end_margin_rs0=v_end_margin_rs0,
        n_steps=n_steps, bisect_iters=bisect_iters, n_family=n_family,
        no_plot=no_plot,
    )

    print(f"[driver_bh] mode = {mode}")
    runner = {"embed": _run_embed, "tidal": _run_tidal,
              "infall": _run_infall, "horizons": _run_horizons}[mode]
    return runner(kw, outdir, csvdir, dpi, lw)
