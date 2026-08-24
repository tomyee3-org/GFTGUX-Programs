"""
driver_cosmo.py
================
Orchestration layer for Cosmology_expansion_simulator.

The driver validates presentation-level inputs, dispatches to the
requested calculation, prints a run summary, writes optional CSV data
files, and hands the result to the plotting layer.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np

import physics_cosmo as phys
import plot_cosmo as viz

MODES = ("evolve", "compare", "age")
SCAN_PARAMS = ("omega_m", "omega_de", "w0", "H0")

W = 68
SEP = "-" * W


# ======================================================================
# Validation helpers
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


def _parse_preset_list(text):
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = [piece.strip() for piece in str(text).split(",") if piece.strip()]
    if not items:
        raise ValueError("presets did not contain any names.")
    if len(items) > 12:
        raise ValueError("presets may contain at most 12 entries.")
    known = set(phys.PRESETS) | {"custom"}
    for name in items:
        if name not in known:
            raise ValueError(
                f"{name!r} is not a known preset. Choices are: "
                f"{', '.join(sorted(known))}."
            )
    return items


# ======================================================================
# CSV output
# ======================================================================
def _write_csv(csvdir, prefix, header, rows, comments=()):
    os.makedirs(csvdir, exist_ok=True)
    # Microsecond resolution avoids silently overwriting a previous file
    # when two csv-writing calls in the same run (or two quick successive
    # runs) land in the same wall-clock second.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(csvdir, f"{prefix}_{stamp}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[driver_cosmo] CSV saved -> {path}")
    return path


def _provenance(mode, kw, results=None):
    """Build the '# '-prefixed comment lines written above a CSV's header.

    `kw` is the input parameter dictionary (always included). `results`,
    when given, is a dict of derived/output values (e.g. stop_reason,
    the derived Omega_k0, age_today_gyr, turnaround) that get their own
    labeled section -- these are the run's actual OUTPUTS, not inputs,
    and are kept visually distinct from the parameter list above them.
    """
    lines = [
        f"Cosmology_expansion_simulator version {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID})",
        f"mode = {mode}",
        f"run at {datetime.now().isoformat(timespec='seconds')}",
        "parameters used by this run:",
    ]
    for name, value in kw.items():
        lines.append(f"    {name} = {value}")
    if results:
        lines.append("derived results for this run:")
        for name, value in results.items():
            lines.append(f"    {name} = {value}")
    return lines


def _crossing_provenance_fields(prefix, crossings):
    """
    Serialize a milestone crossings list (a_eq_mde_crossings or
    a_accel_crossings) as numbered, machine-readable provenance fields
    (Codex Audit 8 P1-2C): the underlying data already existed in the
    summary dict, but never reached the CSV a script would actually
    parse -- only the first-crossing scalar columns did. Returns {} for
    0 or 1 crossings (the existing scalar columns already say
    everything); {prefix_crossing_count, prefix_crossing_1_a,
    prefix_crossing_1_z, prefix_crossing_1_direction, ...} otherwise.
    """
    if not crossings or len(crossings) < 2:
        return {}
    fields = {f"{prefix}_crossing_count": len(crossings)}
    for i, c in enumerate(crossings, start=1):
        fields[f"{prefix}_crossing_{i}_a"] = c["a"]
        fields[f"{prefix}_crossing_{i}_z"] = c["z"]
        fields[f"{prefix}_crossing_{i}_direction"] = c["direction"]
    return fields


EVOLVE_HEADER = ["t_Gyr", "a", "z", "H_km_s_Mpc", "Omega_m", "Omega_r",
                 "Omega_k", "Omega_DE", "q", "w_DE"]


def _evolve_rows(result):
    # t_Gyr and a are printed to 17 significant digits (round-trip
    # double precision) rather than 8: at late contraction, several
    # distinct rows near the mirrored endpoint can differ only in the
    # 9th-or-later digit of t_Gyr, and an 8-digit format made them look
    # like duplicate timestamps. The remaining columns are derived,
    # lower-precision-by-construction quantities, so 10 significant
    # digits is ample there.
    rows = []
    n = result["a"].size
    for i in range(n):
        rows.append([
            f"{result['t_gyr'][i]:.17g}",
            f"{result['a'][i]:.17g}",
            f"{result['z'][i]:.10g}",
            f"{result['H_kms_mpc'][i]:.10g}",
            f"{result['Om'][i]:.10g}",
            f"{result['Or'][i]:.10g}",
            f"{result['Ok'][i]:.10g}",
            f"{result['Ode'][i]:.10g}",
            f"{result['q'][i]:.10g}",
            f"{result['w_de'][i]:.10g}",
        ])
    return rows


# ======================================================================
# Summary printers
# ======================================================================
def _head(title):
    print(SEP)
    print(f"  Cosmology_expansion_simulator {phys.MODEL_VERSION} - {title}")
    print(SEP)


def _fmt(x, spec="{:.4g}"):
    if x is None:
        return "n/a"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "n/a"
    except TypeError:
        pass
    return spec.format(x)


def _print_warnings(warnings_list):
    if warnings_list:
        print("  Notes on this run:")
        for w in warnings_list:
            print(f"    * {w}")


def _extra_crossings_suffix(crossings):
    """
    "" if there is 0 or 1 crossing (the scalar field printed alongside
    this already says everything); otherwise a compact listing of every
    ADDITIONAL crossing, so a non-monotonic CPL history's full picture
    is visible in the ordinary terminal summary, not only in the
    summary dict's a_eq_mde_crossings/a_accel_crossings list fields that
    a plain terminal user would otherwise never see (Codex Audit 8
    P1-2C: the underlying data already existed, but never reached
    student-facing output).
    """
    if not crossings or len(crossings) < 2:
        return ""
    extra = ", ".join(
        f"a={c['a']:.4g} ({c['direction']})" for c in crossings[1:]
    )
    return f"  [{len(crossings)} crossings total; also: {extra}]"


def _print_evolve_summary(s):
    _head("evolve")
    print(f"  H0                    : {s['H0_kms_mpc']:.3f}  km/s/Mpc "
          f"(Hubble time 1/H0 = {s['H0_inv_gyr']:.4f} Gyr)")
    print(f"  Omega_m0, Omega_r0    : {s['omega_m']:.5f}, {s['omega_r']:.5g}")
    print(f"  Omega_k0 (derived)    : {s['omega_k']:.5f}")
    print(f"  Omega_DE0             : {s['omega_de']:.5f}   (w0={s['w0']:.3f}, wa={s['wa']:.3f})")
    print(f"  Early-time regime     : {s['early_regime']}  (a_i={s['a_i']:.2g})")
    print(SEP)
    if s.get("past_status") == "past_eternal":
        print("  Age today (a=1)       : UNDEFINED -- this model is "
              "PAST-ETERNAL, not a finite-age Big Bang (see warnings)")
        print(f"  Elapsed a_i -> today  : {_fmt(s.get('elapsed_ai_to_today_gyr'))}"
              "  Gyr  (a_i-dependent bookkeeping only, NOT a physical age)")
    else:
        print(f"  Age today (a=1)       : {_fmt(s['age_today_gyr'])}  Gyr")
    print(f"  q0 (deceleration)     : {_fmt(s['q0'])}")
    print(f"  Matter-radiation eq.  : a_eq={_fmt(s['a_eq_rm'])}, z_eq={_fmt(s['z_eq_rm'])}")
    print(f"  Matter-DE equality    : a_eq={_fmt(s['a_eq_mde'])}, z_eq={_fmt(s['z_eq_mde'])}"
          + _extra_crossings_suffix(s.get("a_eq_mde_crossings")))
    print(f"  Decel/accel transition: a={_fmt(s['a_accel'])}, z={_fmt(s['z_accel'])}"
          + _extra_crossings_suffix(s.get("a_accel_crossings")))
    if s["turnaround"] is not None:
        ta = s["turnaround"]
        print(f"  TURNAROUND (recollapse): a_turn={ta['a_turn']:.5g}, "
              f"t_turn={ta['t_turn_gyr']:.4g} Gyr")
        if s["total_lifetime_gyr"] is not None:
            print(f"  Estimated total lifetime (Big Bang to Big Crunch): "
                  f"{s['total_lifetime_gyr']:.4g} Gyr (the factor of two is "
                  "exact by time-reversal symmetry; t_turn itself is a "
                  "numerical estimate)")
    if s["big_rip_gyr"] is not None:
        print(f"  Estimated Big Rip time: {s['big_rip_gyr']:.4g} Gyr "
              f"(i.e. {s['big_rip_gyr'] - (s['age_today_gyr'] or 0):.4g} Gyr from today)")
    elif s.get("big_rip_remaining_gyr") is not None:
        print(f"  Estimated time to Big Rip (from today): "
              f"{s['big_rip_remaining_gyr']:.4g} Gyr (no absolute Big-Rip "
              "time is reported since age_today_gyr is undefined -- see "
              "past_status)")
    if s.get("fate_status") == "future_recollapse":
        print(f"  Fate (beyond this run): future recollapse near "
              f"a~{_fmt(s.get('future_turnaround_a'))}")
    elif s.get("fate_status") == "unresolved":
        print("  Fate (beyond this run): unresolved (see warnings)")
    print(f"  Forward-loop iterations: {s['n_forward_iterations']:,}")
    print(f"  Output samples        : {s['n_output_samples']:,}")
    print(f"  Run stopped because   : {s.get('stop_reason') or 'n/a'} "
          "(a_max reached / t_max reached / genuine turnaround)")
    _print_warnings(s["warnings"])
    print(SEP)


def _run_evolve(kw, outdir, csvdir, dpi, lw):
    result = phys.integrate_evolution(
        H0=kw["H0"], omega_m=kw["omega_m"], omega_r=kw["omega_r"],
        omega_de=kw["omega_de"], w0=kw["w0"], wa=kw["wa"],
        a_i=kw["a_i"], a_max=kw["a_max"], t_max_gyr=kw["t_max"],
        step_frac=kw["step_frac"], continue_collapse=kw["continue_collapse"],
    )
    _print_evolve_summary(result["summary"])

    if csvdir is not None:
        s = result["summary"]
        ta = s.get("turnaround")
        prov = _provenance("evolve", {
            k: kw[k] for k in ("H0", "omega_m", "omega_r", "omega_de", "w0",
                               "wa", "a_i", "a_max", "t_max", "step_frac",
                               "continue_collapse")
        }, results={
            "stop_reason": s.get("stop_reason"),
            # omega_de as GIVEN on the command line can be None (flat,
            # unspecified); this is the actual numeric value the model
            # used, resolved by the closure relation -- distinct from
            # the raw input echoed in the parameters section above, and
            # what a script parsing this CSV programmatically actually
            # needs.
            "omega_de0_resolved": s.get("omega_de"),
            "omega_k0_derived": s.get("omega_k"),
            "age_today_gyr": s.get("age_today_gyr"),
            # past_status distinguishes a genuine finite-age Big Bang
            # from a past-eternal model, for which age_today_gyr above
            # is deliberately None; elapsed_ai_to_today_gyr recovers the
            # raw (a_i-dependent, non-physical-age) elapsed-time value
            # in that case (Codex Audit 8 P0-1).
            "past_status": s.get("past_status"),
            "elapsed_ai_to_today_gyr": s.get("elapsed_ai_to_today_gyr"),
            # Serialized as separate scalar keys, not a nested Python
            # dict string, so a machine reading this header does not
            # need to parse Python literal syntax to recover them.
            "a_turn": ta["a_turn"] if ta is not None else None,
            "t_turn_gyr": ta["t_turn_gyr"] if ta is not None else None,
            "total_lifetime_gyr": s.get("total_lifetime_gyr"),
            "big_rip_gyr": s.get("big_rip_gyr"),
            "big_rip_remaining_gyr": s.get("big_rip_remaining_gyr"),
            "fate_status": s.get("fate_status"),
            "future_turnaround_a": s.get("future_turnaround_a"),
            "fate_search_limit_a": s.get("fate_search_limit_a"),
            "n_forward_iterations": s.get("n_forward_iterations"),
            "n_output_samples": s.get("n_output_samples"),
            **_crossing_provenance_fields("a_eq_mde", s.get("a_eq_mde_crossings")),
            **_crossing_provenance_fields("a_accel", s.get("a_accel_crossings")),
        })
        _write_csv(csvdir, "cosmo_evolve", EVOLVE_HEADER,
                  _evolve_rows(result), comments=prov)

    if not kw["no_plot"]:
        viz.plot_evolve(result, outdir=outdir, dpi=dpi, lw=lw)
    return result


# ======================================================================
# Mode: compare
# ======================================================================
def _resolve_preset(name, kw):
    if name == "custom":
        p = dict(H0=kw["H0"], omega_m=kw["omega_m"], omega_r=kw["omega_r"],
                 omega_de=kw["omega_de"], w0=kw["w0"], wa=kw["wa"],
                 label="Custom (command-line parameters)")
    else:
        p = dict(phys.PRESETS[name])
    return p


def _cell(x, width, prec=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        text = "n/a"
    else:
        text = f"{x:.{prec}g}"
    return f"{text:>{width}}"


def _fate_note(s):
    """
    Short human-readable fate label for --compare mode's summary table
    and CSV, built from the structured fate_status field (see
    physics_cosmo.integrate_evolution) rather than re-deriving it from
    turnaround/big_rip_gyr alone -- the latter left the
    diagnosed-but-out-of-range "future_recollapse" case (a phantom model
    certified to recollapse beyond this run's --a_max/--t_max) with a
    blank note even though its fate WAS known, and left "unresolved"
    indistinguishable from "no fate question applies here at all"
    (Codex Audit 6 P1-2 / Copilot Audit 6 P2-2).
    """
    # past_status is checked first (Codex Audit 8 P0-1): a past-eternal
    # model's missing age_today_gyr is a DIFFERENT, more fundamental
    # fact than any fate_status value below, and must not be reported
    # with a blank or misleading note.
    if s.get("past_status") == "past_eternal":
        return "past-eternal (no finite age)"
    status = s.get("fate_status")
    if status == "recollapse":
        return "recollapses"
    if status == "big_rip":
        return "Big Rip ahead"
    if status == "future_recollapse":
        return "future recollapse"
    if status == "unresolved":
        return "fate unresolved"
    return ""


def _print_compare_summary(names, results):
    _head("compare")
    # note is left-aligned in a wide, fixed field with an explicit
    # leading space before it, rather than right-justified flush
    # against z_accel: "past-eternal (no finite age)" and "future
    # recollapse" are both longer than the old 14-character column, and
    # a right-justified "n/a" + note previously ran together with no
    # separator at all (Codex Audit 8 P2-3, e.g. "n/afuture recollapse").
    hdr = (f"  {'name':<14}{'age_Gyr':>10}{'H0*t0':>9}{'q0':>9}"
           f"{'z_accel':>10}  {'note':<28}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, r in zip(names, results):
        s = r["summary"]
        h0t0 = (s["age_today_gyr"] / s["H0_inv_gyr"]) if s["age_today_gyr"] else None
        note = _fate_note(s)
        print(f"  {name:<14}{_cell(s['age_today_gyr'], 10)}"
              f"{_cell(h0t0, 9)}{_cell(s['q0'], 9, 3)}"
              f"{_cell(s['z_accel'], 10)}  {note:<28}")
    print(SEP)


def _run_compare(kw, outdir, csvdir, dpi, lw):
    names = _parse_preset_list(kw["presets"])
    results = []
    for name in names:
        p = _resolve_preset(name, kw)
        r = phys.integrate_evolution(
            H0=p["H0"], omega_m=p["omega_m"], omega_r=p["omega_r"],
            omega_de=p["omega_de"], w0=p["w0"], wa=p["wa"],
            a_i=kw["a_i"], a_max=kw["a_max"], t_max_gyr=kw["t_max"],
            step_frac=kw["step_frac"],
            continue_collapse=kw["continue_collapse"],
        )
        r["label"] = p["label"]
        results.append(r)

    _print_compare_summary(names, results)

    if csvdir is not None:
        prov = _provenance("compare", {"presets": ",".join(names),
                                       "a_i": kw["a_i"], "a_max": kw["a_max"],
                                       "t_max": kw["t_max"],
                                       "step_frac": kw["step_frac"],
                                       "continue_collapse": kw["continue_collapse"]})
        # Every structured fate field physics_cosmo.py now produces is
        # written here under its own machine-readable name (never as a
        # human phrase under a generic "note" column -- Codex Audit 7
        # P1-4), plus the raw turnaround/big-rip values a downstream
        # reader needs to interpret fate_status without re-deriving
        # them: a "future_recollapse" row is otherwise unable to say
        # WHERE that certified turnaround is, and an "unresolved" row
        # cannot reveal its own search horizon.
        def _opt(x, fmt="{:.6g}"):
            return "" if x is None else fmt.format(x)

        age_rows = []
        for name, r in zip(names, results):
            s = r["summary"]
            h0t0 = (s["age_today_gyr"] / s["H0_inv_gyr"]) if s["age_today_gyr"] else None
            ta = s.get("turnaround")
            age_rows.append([
                name, r["label"], f"{s['H0_kms_mpc']:.4f}", f"{s['omega_m']:.6g}",
                f"{s['omega_r']:.6g}", f"{s['omega_k']:.6g}", f"{s['omega_de']:.6g}",
                f"{s['w0']:.4f}", f"{s['wa']:.4f}",
                _opt(s["age_today_gyr"]), _opt(h0t0), _opt(s["q0"]),
                _opt(s["z_accel"]),
                s.get("stop_reason") or "",
                s.get("past_status") or "",
                _opt(s.get("elapsed_ai_to_today_gyr")),
                s.get("fate_status") or "",
                _opt(ta["a_turn"] if ta is not None else None),
                _opt(ta["t_turn_gyr"] if ta is not None else None),
                _opt(s.get("total_lifetime_gyr")),
                _opt(s.get("big_rip_gyr")),
                _opt(s.get("big_rip_remaining_gyr")),
                _opt(s.get("future_turnaround_a")),
                _opt(s.get("fate_search_limit_a")),
                _fate_note(s),
            ])
        _write_csv(csvdir, "cosmo_compare_ages",
                  ["preset", "label", "H0", "omega_m", "omega_r", "omega_k",
                   "omega_de", "w0", "wa", "age_today_Gyr", "H0_t0", "q0",
                   "z_accel", "stop_reason", "past_status",
                   "elapsed_ai_to_today_Gyr", "fate_status", "a_turn",
                   "t_turn_gyr", "total_lifetime_gyr", "big_rip_gyr",
                   "big_rip_remaining_gyr",
                   "future_turnaround_a", "fate_search_limit_a",
                   "fate_note"],
                  age_rows, comments=prov)

        curve_rows = []
        for name, r in zip(names, results):
            for i in range(r["a"].size):
                curve_rows.append([
                    name, f"{r['t_gyr'][i]:.17g}", f"{r['a'][i]:.17g}",
                    f"{r['H_kms_mpc'][i]:.10g}", f"{r['q'][i]:.10g}",
                ])
        _write_csv(csvdir, "cosmo_compare_curves",
                  ["preset", "t_Gyr", "a", "H_km_s_Mpc", "q"],
                  curve_rows, comments=prov)

    if not kw["no_plot"]:
        viz.plot_compare(names, results, outdir=outdir, dpi=dpi, lw=lw)
    return dict(names=names, results=results)


# ======================================================================
# Mode: age (parameter scan)
# ======================================================================
_AGE_SCAN_A_MAX = 5.0  # a finite look-ahead horizon, large enough to catch
                        # recollapse in the exercises this program documents
                        # (a_max=1.05, used previously, could never see a
                        # turnaround at all: even the mildest recollapsing
                        # cases in these exercises have a_turn well above
                        # 1.05) -- but an arbitrary, user-chosen scan range
                        # can always place a genuine a_turn beyond ANY fixed
                        # horizon. This is documented as exactly that: a
                        # finite-horizon convention (see recollapses_by_a5
                        # below and the help file), not a claim that every
                        # possible recollapse is caught.


def _run_age(kw, outdir, csvdir, dpi, lw):
    scan_param = kw["scan_param"]
    lo, hi, n = kw["scan_lo"], kw["scan_hi"], kw["scan_n"]
    values = np.linspace(lo, hi, n)

    ages = np.full(n, np.nan)
    h0t0 = np.full(n, np.nan)
    z_accel = np.full(n, np.nan)
    recollapsed = np.zeros(n, dtype=bool)
    recollapsed_before_today = np.zeros(n, dtype=bool)
    past_eternal = np.zeros(n, dtype=bool)
    failures = {}

    base_H0, base_om, base_or = kw["H0"], kw["omega_m"], kw["omega_r"]
    base_ode, base_w0, base_wa = kw["omega_de"], kw["w0"], kw["wa"]
    base_ode_resolved, _ = phys.resolve_omega_de(base_om, base_or, base_ode)

    for i, val in enumerate(values):
        H0, om, orr, ode, w0, wa = base_H0, base_om, base_or, base_ode_resolved, base_w0, base_wa
        if scan_param == "omega_m":
            om = val
            ode = (1.0 - om - base_or) if kw["force_flat"] else base_ode_resolved
        elif scan_param == "omega_de":
            ode = val
            om = (1.0 - ode - base_or) if kw["force_flat"] else base_om
        elif scan_param == "w0":
            w0 = val
        elif scan_param == "H0":
            H0 = val
        try:
            r = phys.integrate_evolution(H0=H0, omega_m=om, omega_r=orr, omega_de=ode,
                                         w0=w0, wa=wa, a_i=kw["a_i"],
                                         a_max=_AGE_SCAN_A_MAX,
                                         step_frac=kw["step_frac"])
        except (ValueError, RuntimeError) as exc:
            failures[i] = str(exc)
            continue
        s = r["summary"]
        # A past-eternal point (Codex Audit 8 P0-1) genuinely completed
        # and, in general, DID reach a=1 -- age_today_gyr is None there
        # by deliberate design, not because the run failed to get that
        # far, so it must not be lumped in with the "never reached a=1"
        # failure note below (checked via elapsed_ai_to_today_gyr, which
        # stays defined for a past-eternal point whenever a=1 was
        # actually reached, unlike age_today_gyr itself).
        if s.get("past_status") == "past_eternal":
            past_eternal[i] = True
        elif s["age_today_gyr"] is not None:
            ages[i] = s["age_today_gyr"]
            h0t0[i] = s["age_today_gyr"] / s["H0_inv_gyr"]
        elif s["turnaround"] is None and s.get("elapsed_ai_to_today_gyr") is None:
            failures[i] = (
                f"run completed but never reached a=1 within a_max="
                f"{_AGE_SCAN_A_MAX:g}"
            )
        if s["z_accel"] is not None:
            z_accel[i] = s["z_accel"]
        recollapsed[i] = s["turnaround"] is not None
        if s["turnaround"] is not None and s["turnaround"]["a_turn"] < 1.0:
            recollapsed_before_today[i] = True

    if len(failures) == n:
        example = next(iter(failures.values()))
        raise RuntimeError(
            f"All {n} points in this {scan_param} scan failed; every run "
            f"raised or produced no result (first failure: {example}). "
            "Check that the scan range is physically sensible (e.g. with "
            "force_flat, omega_m values near or above 1 push omega_de "
            "negative)."
        )

    _head("age")
    print(f"  Scanning {scan_param} from {lo:g} to {hi:g} ({n} points), "
          f"force_flat={kw['force_flat']}")
    finite = np.isfinite(ages)
    if np.any(finite):
        i_max = np.nanargmax(ages)
        i_min = np.nanargmin(ages)
        print(f"  Oldest age in scan : {ages[i_max]:.4g} Gyr at {scan_param}={values[i_max]:.4g}")
        print(f"  Youngest age in scan: {ages[i_min]:.4g} Gyr at {scan_param}={values[i_min]:.4g}")
    if np.any(recollapsed_before_today):
        print(f"  {int(recollapsed_before_today.sum())} of {n} scan points "
              "recollapse BEFORE reaching their present size (a=1): no "
              "age-today is defined for these.")
    n_future_recollapse = int(recollapsed.sum() - recollapsed_before_today.sum())
    if n_future_recollapse:
        print(f"  {n_future_recollapse} of {n} scan points reach a=1 fine "
              f"but recollapse at some a_turn > 1 found within this scan's "
              f"a_max={_AGE_SCAN_A_MAX:g} look-ahead horizon; a model with "
              "a_turn beyond that horizon would be reported as non-"
              "recollapsing here.")
    if failures:
        print(f"  {len(failures)} of {n} scan points failed and are "
              "omitted (see the 'note' column in the CSV, if requested).")
    if np.any(past_eternal):
        print(f"  {int(past_eternal.sum())} of {n} scan points are "
              "PAST-ETERNAL (no finite Big-Bang age; see past_status in "
              "--mode evolve for that point) -- reported as no age here, "
              "not as a failure.")
    print(SEP)

    if csvdir is not None:
        prov = _provenance("age", {k: kw[k] for k in
                                   ("scan_param", "scan_lo", "scan_hi", "scan_n",
                                    "force_flat", "H0", "omega_m", "omega_r",
                                    "omega_de", "w0", "wa", "a_i", "step_frac",
                                    "age_ref_gyr")})
        rows = []
        for i in range(n):
            note = failures.get(i, "")
            rows.append([
                f"{values[i]:.6g}",
                "" if not np.isfinite(ages[i]) else f"{ages[i]:.6g}",
                "" if not np.isfinite(h0t0[i]) else f"{h0t0[i]:.6g}",
                "" if not np.isfinite(z_accel[i]) else f"{z_accel[i]:.6g}",
                "yes" if recollapsed[i] else "no",
                "yes" if recollapsed_before_today[i] else "no",
                "yes" if past_eternal[i] else "no",
                note,
            ])
        _write_csv(csvdir, f"cosmo_age_scan_{scan_param}",
                  [scan_param, "age_today_Gyr", "H0_t0", "z_accel",
                   f"recollapses_by_a{_AGE_SCAN_A_MAX:g}",
                   "recollapses_before_a1", "past_eternal", "note"],
                  rows, comments=[
                      f"recollapses_by_a{_AGE_SCAN_A_MAX:g}: 'yes' means a "
                      f"turning point was found at or before a={_AGE_SCAN_A_MAX:g} "
                      "in this scan's finite look-ahead integration; a model "
                      "that recollapses only beyond that horizon is reported "
                      "'no' here, not confirmed non-recollapsing for all time.",
                  ] + list(prov))

    scan_result = dict(scan_param=scan_param, values=values, ages=ages, h0t0=h0t0,
                       z_accel=z_accel, recollapsed=recollapsed,
                       recollapsed_before_today=recollapsed_before_today,
                       past_eternal=past_eternal,
                       age_ref_gyr=kw["age_ref_gyr"])
    if not kw["no_plot"]:
        viz.plot_age_scan(scan_result, outdir=outdir, dpi=dpi, lw=lw)
    return scan_result


# ======================================================================
# Public entry point
# ======================================================================
def run(mode="evolve",
        H0=70.0, omega_m=0.30, omega_r=9.24e-5, omega_de=None, w0=-1.0, wa=0.0,
        a_i=1.0e-8, a_max=5.0, t_max=None, step_frac=0.005, continue_collapse=False,
        presets="EdS,lambdaCDM,closed,phantom",
        scan_param="omega_m", scan_lo=0.05, scan_hi=1.5, scan_n=40,
        force_flat=True, age_ref_gyr=13.0,
        outdir=None, csvdir=None, dpi=150, lw=1.6, no_plot=False):
    """
    Run one Cosmology_expansion_simulator calculation.

    mode selects the calculation:
        "evolve"   integrate a(t), H(t), the density parameters, and q(t)
                   for a single cosmology
        "compare"  overlay several named or custom cosmologies and
                   tabulate their ages
        "age"      scan the age of the universe (and related epochs)
                   across one cosmological parameter
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}.")

    dpi, lw = _validate_output(outdir, csvdir, dpi, lw)
    if no_plot and outdir is not None:
        raise ValueError("no_plot and outdir cannot both be requested.")
    if no_plot and csvdir is None:
        raise ValueError(
            "no_plot was requested but no csvdir was given, so the run "
            "would produce no output at all."
        )
    # scan_param/scan_lo/scan_hi/scan_n/age_ref_gyr are meaningful only for
    # mode="age"; validating them unconditionally here would fail an
    # evolve or compare run over an irrelevant, mode-specific default (or
    # a value the caller never intended to use). That validation now
    # happens in _run_age, immediately below, where these values are
    # actually consumed.
    if mode == "age":
        if scan_param not in SCAN_PARAMS:
            raise ValueError(f"scan_param must be one of {SCAN_PARAMS}; got {scan_param!r}.")
        scan_lo = _finite("scan_lo", scan_lo)
        scan_hi = _finite("scan_hi", scan_hi)
        if scan_hi <= scan_lo:
            raise ValueError("scan_hi must be greater than scan_lo.")
        if not isinstance(scan_n, int) or not (3 <= scan_n <= 500):
            raise ValueError("scan_n must be an integer between 3 and 500.")
        age_ref_gyr = _finite("age_ref_gyr", age_ref_gyr)
        if age_ref_gyr <= 0:
            raise ValueError("age_ref_gyr must be greater than zero.")
    else:
        _AGE_ONLY_DEFAULTS = dict(scan_param="omega_m", scan_lo=0.05,
                                   scan_hi=1.5, scan_n=40, force_flat=True,
                                   age_ref_gyr=13.0)
        _ignored = {
            "scan_param": scan_param, "scan_lo": scan_lo, "scan_hi": scan_hi,
            "scan_n": scan_n, "force_flat": force_flat,
            "age_ref_gyr": age_ref_gyr,
        }
        changed = [k for k, v in _ignored.items() if v != _AGE_ONLY_DEFAULTS[k]]
        if changed:
            print(f"[driver_cosmo] note: {', '.join(changed)} only affect "
                  f"mode='age' and will be ignored for mode={mode!r}.")

    kw = dict(
        H0=H0, omega_m=omega_m, omega_r=omega_r, omega_de=omega_de,
        w0=w0, wa=wa, a_i=a_i, a_max=a_max, t_max=t_max,
        step_frac=step_frac, continue_collapse=continue_collapse,
        presets=presets,
        scan_param=scan_param, scan_lo=scan_lo, scan_hi=scan_hi, scan_n=scan_n,
        force_flat=force_flat, age_ref_gyr=age_ref_gyr,
        no_plot=no_plot,
    )

    print(f"[driver_cosmo] mode = {mode}")
    runner = {"evolve": _run_evolve, "compare": _run_compare, "age": _run_age}[mode]
    return runner(kw, outdir, csvdir, dpi, lw)
