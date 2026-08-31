"""
driver_gw.py
============
Orchestration layer for GravitationalWaveSources.
"""

import csv
import math
import os
from datetime import datetime, timezone

import numpy as np
import physics_gw as phys
import plot_gw as viz

#: Conservative upper bound on --dpi. The figure is a fixed 12x9 inches, so
#: pixel area grows as dpi^2; an unbounded dpi lets a single typo (or a
#: deliberately adversarial value) request a multi-hundred-megabyte to
#: multi-gigabyte in-memory image before any useful error can be produced.
#: 600 dpi already exceeds anything these instructional plots need (a
#: 7200x5400 pixel PNG), so this is not expected to constrain any normal use.
MAX_DPI = 600


def _finite_number(name, value):
    """Return value as float after giving a consistent user-facing error.

    ``bool`` (and ``numpy.bool_``) are rejected explicitly even though
    ``float(True) == 1.0`` would otherwise convert silently -- see the
    matching guard in ``physics_gw._require_finite`` for the rationale.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a finite number; got {value!r} (a bool is not "
            "an accepted numeric value)."
        )
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # See the matching comment in physics_gw._require_finite: a Python
        # int too large to represent as a float is a value problem for this
        # caller, not a programming error.
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def version_info():
    """Return the program's model version and build identifier.

    Exposed at the driver layer (mirroring the physics-layer constants) so
    callers and tests can query provenance without reaching into
    ``physics_gw`` directly.
    """
    return {"model_version": phys.MODEL_VERSION, "build_id": phys.BUILD_ID}


def _validate_plot_inputs(t_before, t_after, lw, dpi):
    lw = _finite_number("lw", lw)
    dpi = _finite_number("dpi", dpi)
    if lw <= 0:
        raise ValueError("lw must be greater than zero.")
    if int(dpi) != dpi or dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if dpi > MAX_DPI:
        raise ValueError(
            f"dpi must not exceed {MAX_DPI}; the figure is a fixed 12x9in, "
            f"so a larger value requests an excessively large PNG. Reduce "
            "--dpi."
        )

    normalized_zoom = []
    for name, value in (("t_before", t_before), ("t_after", t_after)):
        if value is None:
            normalized_zoom.append(None)
        else:
            value = _finite_number(name, value)
            if value < 0:
                raise ValueError(f"{name} must be None or a finite non-negative number.")
            normalized_zoom.append(value)

    return normalized_zoom[0], normalized_zoom[1], lw, int(dpi)


def _timestamp_fname(prefix="gw_inspiral", ext="csv"):
    # Microsecond resolution makes two rapid saves in the same directory
    # (e.g. a scripted parameter sweep) collide far less often than under
    # the previous second-resolution timestamp, but does not make a
    # collision impossible (a coarser OS clock, two calls landing in the
    # same microsecond, or a clock adjustment can still repeat a value).
    # _reserve_unique_path() below is what actually guarantees no silent
    # overwrite; this function only picks the first candidate name.
    #
    # Audit3 (Gemini): this uses UTC (not local time) so a filename's
    # embedded timestamp can be compared directly against the UTC
    # export_timestamp_utc field inside the CSV's own metadata block,
    # without a reader needing to know or convert the local timezone this
    # program happened to run in.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"


def _reserve_unique_path(directory, prefix, ext, max_attempts=1000):
    """Atomically claim a not-yet-existing "prefix_timestamp[_n].ext" path.

    Audit2 (Copilot A2-3) correctly points out that a microsecond-resolution
    timestamp only reduces, rather than eliminates, the chance that two
    writes pick the same filename (a coarser effective clock resolution on
    some platforms, two calls landing in the same microsecond, concurrent
    processes, or a clock adjustment can still repeat a value) -- and an
    ordinary open(path, "w") would then silently overwrite the earlier
    file. This claims the path with O_CREAT|O_EXCL, which fails atomically
    if the path already exists, and retries with a "_1", "_2", ... suffix
    until an unclaimed name is found.

    This only reserves the *name*: it guarantees no two callers walk away
    believing they each own the same final path, but the reserved file
    itself is empty. Audit3 (Codex P2-1 / Copilot A3-6) correctly pointed
    out that an earlier version of this project both reserved the name AND
    treated that as the file to write content into later by reopening its
    path -- which is a real gap (a failure between reservation and the
    later open/write leaves an empty file with no cleanup) and also made
    the "no TOCTOU race" claim overstated (the guarantee covers claiming
    the name, not a later reopen of that same path). Callers must not
    write content directly into the path this function returns; write into
    a temporary path derived from it instead, and atomically replace it
    into place only once the content write has fully succeeded -- see
    _write_csv() and plot_gw.plot_inspiral() for that pattern, and
    _publish_or_cleanup() below for the shared cleanup half of it.
    """
    base = _timestamp_fname(prefix=prefix, ext=ext)
    stem = base[: -(len(ext) + 1)]  # strip the trailing ".<ext>"
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
    failure, remove every trace (the temp file and the empty reservation
    placeholder at fpath) and re-raise.

    Audit3 (Codex P2-1; Copilot A3-2/A3-3): before this, both _write_csv()
    and plot_gw.plot_inspiral() reopened the already-reserved fpath
    directly and wrote content into it in place, so a failure partway
    through (a mid-write exception, a disk-full condition, a patched
    writer/savefig raising) left a truncated or zero-byte file permanently
    occupying that filename -- indistinguishable, by its presence alone,
    from a genuinely completed export. Writing to a same-directory
    temporary file first and renaming it into place with os.replace() only
    after the write fully succeeds means fpath is never observably partial:
    from the outside, it is either absent or complete. os.replace() is
    atomic on the same filesystem and here always overwrites only the
    empty placeholder this run itself created via _reserve_unique_path(),
    never a concurrent writer's file (each reservation is independently
    unique).
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


def _write_csv(result, csvdir):
    """Save a self-documenting CSV of t, f, A, h, phase (blank cells where
    the ringdown segment leaves f/A undefined -- see integrate_inspiral).

    This gives students a documented, no-programming-required route to the
    numerical arrays behind the plot -- e.g. for the chirp-mass-extraction
    exercise (estimate df/dt between two nearby rows, then invert with
    physics_gw.chirp_mass_from_fdot()) or for EXP-8's fixed-time RK4
    convergence comparison -- without requiring every student to write a
    Python import snippet.

    Audit2 (Codex P1-1 / Copilot A2-5) found that the original export --
    a bare "t_s,f_hz,A,h" table with only a timestamped filename -- could
    not be traced back to the executable revision or run parameters that
    produced it once separated from the terminal output, which defeats the
    point of EXP-8's multi-dt comparison (four files that differ only by
    --dt, with nothing in the files themselves saying which is which). A
    commented metadata block (mirroring the convention Gemini/Codex point
    to in StellarEvolutionTracks) is written before the header row, naming
    MODEL_VERSION, BUILD_ID, a UTC export timestamp, and every input
    parameter that affects the numerical waveform this run produced (Audit2
    Copilot A2-6 additionally asked for the phase array, which is now a
    fifth data column). This records masses, distance, dt, f_start, and the
    ringdown settings -- not plotting/output-only options such as
    --t_before, --lw, or --dpi, which do not change the numerical arrays
    themselves (Audit3 Copilot A3-4).

    The metadata field is named export_timestamp_utc, not run_timestamp_utc
    (Audit3 Copilot A3-5): it is captured here, when the export actually
    happens, which for a long integration can be measurably later than when
    the run itself began -- the field is named for what it actually
    measures rather than implying it marks run start.
    """
    os.makedirs(csvdir, exist_ok=True)
    fpath = _reserve_unique_path(csvdir, "gw_inspiral", "csv")
    s = result["summary"]
    include_rd = s["include_ringdown"]
    export_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _fmt(value):
        return f"{float(value):.17g}" if value is not None else "n/a"

    meta_lines = [
        "# GravitationalWaveSources CSV export",
        f"# model_version: {s['model_version']}",
        f"# build_id: {s['build_id']}",
        f"# export_timestamp_utc: {export_timestamp}",
        f"# m1_msun: {_fmt(s['m1_msun'])}",
        f"# m2_msun: {_fmt(s['m2_msun'])}",
        f"# d_mpc: {_fmt(s['d_mpc'])}",
        f"# dt_s: {_fmt(s['dt_s'])}",
        f"# f_start_hz: {_fmt(s['f_start_hz'])}",
        f"# include_ringdown: {include_rd}",
        f"# n_ringdown_tau: {s['n_ringdown_tau'] if include_rd else 'n/a'}",
        f"# ringdown_pts: {s['ringdown_pts'] if include_rd else 'n/a'}",
        f"# f_qnm_hz: {_fmt(s['f_qnm_hz']) if include_rd else 'n/a'}",
        f"# tau_qnm_ms: {_fmt(s['tau_qnm_ms']) if include_rd else 'n/a'}",
        (f"# ringdown_model: illustrative Schwarzschild QNM "
         f"(95% of total mass retained, zero remnant spin) -- not a "
         f"physical merger" if include_rd else "# ringdown_model: n/a"),
        "# columns: t_s,f_hz,A,h,phase_rad -- f_hz/A are blank on ringdown-only rows",
    ]

    def _do_write(tmp_path):
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            for line in meta_lines:
                handle.write(line + "\n")
            writer = csv.writer(handle)
            writer.writerow(["t_s", "f_hz", "A", "h", "phase_rad"])
            for t, f, A, h, phase in zip(
                result["t"], result["f"], result["A"], result["h"], result["phase"]
            ):
                writer.writerow([
                    f"{float(v):.17g}" if math.isfinite(v) else ""
                    for v in (t, f, A, h, phase)
                ])
            handle.flush()
            os.fsync(handle.fileno())

    _publish_or_cleanup(_do_write, fpath)
    print(f"[driver_gw] CSV saved -> {fpath}")
    return fpath


def run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
        dt=2e-4, f_start=20.0,
        outdir=None, csvdir=None,
        t_before=None, t_after=None, lw=0.4,
        include_ringdown=False,
        n_ringdown_tau=6, ringdown_pts=4000,
        dpi=150):
    """Run the full calculation, print diagnostics, and render the figure."""
    t_before, t_after, lw, dpi = _validate_plot_inputs(
        t_before, t_after, lw, dpi
    )

    print("[driver_gw] Integrating inspiral ...")
    result = phys.integrate_inspiral(
        m1_msun, m2_msun, d_mpc,
        dt=dt,
        f_start=f_start,
        include_ringdown=include_ringdown,
        n_ringdown_tau=n_ringdown_tau,
        ringdown_pts=ringdown_pts,
    )

    _print_summary(result["summary"])

    # Validate the zoom window against the actual computed data range
    # before writing any requested output file. Audit2 (Codex P3-5) found
    # that a request whose zoom window ends up empty (e.g. --t_before 0
    # --t_after 0) still left a complete CSV behind, because the CSV was
    # written before plot_inspiral()'s own data-dependent window check --
    # so a failed run's exit code did not match what was left on disk. This
    # duplicates a cheap check (plot_inspiral() below still performs its
    # own, so calling plot_inspiral() directly without going through run()
    # is equally safe), buying a clean failure here before either output
    # file is written.
    viz._xlim(result["t_isco"], t_before, t_after, result["t"][0], result["t"][-1])

    # Validate/create both requested output directories up front, before
    # writing either artifact. Audit3 (Codex P2-1) found that an invalid
    # --outdir (e.g. a path that already exists as a regular file, so
    # os.makedirs() raises FileExistsError) was previously only discovered
    # inside plot_inspiral(), by which point a --csvdir export had already
    # completed and been left on disk -- a plotting-only mistake caused an
    # unrelated, already-successful CSV export to look like a failed run.
    # Checking both directories here, before either write begins, prevents
    # that specific ordering problem outright rather than cleaning up after
    # it. (plot_inspiral() still performs its own os.makedirs(outdir, ...)
    # too, so it remains safe to call directly without going through run().)
    if csvdir is not None:
        os.makedirs(csvdir, exist_ok=True)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Transaction contract (Audit3 Codex P2-1, item 4): each requested
    # artifact is independently atomic -- _write_csv()/plot_inspiral() each
    # either produce their complete file or leave none behind (see
    # _publish_or_cleanup() above and its plot_gw counterpart). This is
    # NOT a cross-artifact all-or-nothing guarantee: if both --csvdir and
    # --outdir are requested and the CSV export succeeds but the PNG save
    # then fails, the completed CSV is kept, not retroactively deleted.
    # Rolling back a genuinely successful, independent export because a
    # later, unrelated artifact failed would discard correct data the user
    # may still want, for no real safety benefit -- so a run that requests
    # multiple artifacts can complete some and fail others; it is only
    # each individual artifact that is guaranteed never to be partial.
    if csvdir is not None:
        _write_csv(result, csvdir)

    print("[driver_gw] Rendering figure ...")
    viz.plot_inspiral(
        result,
        outdir=outdir,
        t_before=t_before,
        t_after=t_after,
        lw=lw,
        dpi=dpi,
    )
    return result


def _print_summary(s):
    W = 62
    sep = "─" * W
    print(sep)
    print(
        f"  GravitationalWaveSources {phys.MODEL_VERSION} "
        f"(build {phys.BUILD_ID}) — Run Summary"
    )
    print(sep)
    print(f"  Masses              : {s['m1_msun']:.3f} + {s['m2_msun']:.3f}  M☉")
    print(f"  Total mass          : {s['M_total_msun']:.3f}  M☉")
    print(f"  Chirp mass          : {s['Mc_msun']:.4f}  M☉")
    print(f"  Distance            : {s['d_mpc']:.1f}  Mpc")
    print(f"  Band start          : {s['f_start_hz']:.1f}  Hz")
    print(f"  ISCO cutoff         : {s['f_isco_hz']:.1f}  Hz")
    print(f"  Time to ISCO        : {s['T_band_s']:.3f}  s")
    print(f"  Strain scale at ISCO: {s['A_isco']:.3e}")
    print(f"  Inspiral samples    : {s['inspiral_steps']:,}")
    if s["include_ringdown"]:
        print("  Ringdown            : illustrative Schwarzschild QNM")
        print(f"  Final mass (toy est): {s['M_final_msun']:.3f}  M☉")
        print(f"  QNM frequency       : {s['f_qnm_hz']:.1f}  Hz")
        print(f"  QNM decay time      : {s['tau_qnm_ms']:.3f}  ms")
    else:
        print("  Ringdown            : not included")
    print(sep)
