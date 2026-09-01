"""
driver_gw.py
============
Orchestration layer for GravitationalWaveSources.
"""

import csv
import math
import os
import stat
import tempfile
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


def _timestamp_fname(now, prefix="gw_inspiral", ext="csv"):
    """Format an already-captured aware UTC datetime as a
    "prefix_YYYYMMDD_HHMMSS_ffffff.ext" filename.

    Audit4 (Codex P3-1): this now takes the instant to format as an
    explicit argument rather than calling datetime.now() itself, so a
    caller that also needs that same instant elsewhere (e.g. _write_csv()'s
    export_timestamp_utc metadata field) can capture it exactly once and
    reuse it for both the filename and the metadata, instead of two
    independent datetime.now() calls that can straddle a second or date
    boundary and never quite agree.

    Microsecond resolution makes two rapid saves in the same directory
    (e.g. a scripted parameter sweep) collide far less often than under
    the previous second-resolution timestamp, but does not make a
    collision impossible (a coarser OS clock, two calls landing in the
    same microsecond, or a clock adjustment can still repeat a value).
    _publish_atomically() below is what actually guarantees no silent
    overwrite; this function only formats the first candidate name.

    Audit3 (Gemini): this uses UTC (not local time) so a filename's
    embedded timestamp can be compared directly against the UTC
    export_timestamp_utc field inside the CSV's own metadata block,
    without a reader needing to know or convert the local timezone this
    program happened to run in.
    """
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.{ext}"


def _create_private_temp_file(directory, ext):
    """Securely create an empty, randomly-named temporary file in
    `directory` to build one artifact's content into before publication.

    Audit4 (Codex P2-1): the previous design wrote content to a
    caller-predictable "<final-name>.tmp" path. That path is guessable in
    advance (it is derived from the same timestamp as the eventual public
    name) and was opened with an ordinary open(path, "w"), which follows a
    pre-existing symlink at that path rather than refusing to -- a
    pre-planted symlink at the guessed .tmp name could redirect this
    program's write to overwrite an unrelated file. tempfile.mkstemp()
    instead draws its filename from a wide random namespace unrelated to
    any published name and creates it with O_CREAT|O_EXCL (refusing to
    follow an existing path/symlink), so nothing else on the system can
    anticipate or pre-plant a symlink at this name.

    Audit5 (Codex P2-1): the returned file descriptor `fd` is kept open
    (mkstemp already created it) and returned alongside tmp_path, instead
    of being closed here. The earlier Audit4 version closed fd immediately
    and had every caller reopen tmp_path by name to write content -- but
    that reopen is itself a second, path-based operation, separated in
    time from the fd-based creation above, and nothing stops something
    else from deleting tmp_path and creating a symlink in its place during
    that gap (e.g. a shared/world-writable output directory, or another
    process racing this one). Reopening by name would then silently follow
    that symlink. Keeping the original fd open and writing through it
    (os.fdopen(fd, ..., closefd=False), see _write_csv/_do_write and the
    plot_gw counterpart) means the bytes always land in the exact inode
    mkstemp created, never in whatever tmp_path happens to resolve to by
    the time writing starts. _verify_temp_identity() below adds a second,
    independent check of this immediately before publication.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=".gw_tmp_", suffix=f".{ext}", dir=directory)
    return fd, tmp_path


def _default_output_file_mode():
    """Return the permission mode an ordinary open(path, "w") would have
    produced for a new file, given the process's current umask.

    Audit5 (Codex P3-1): tempfile.mkstemp() always creates its file with
    mode 0o600 (owner read/write only), regardless of umask -- that is
    mkstemp's own documented, deliberately conservative default, chosen
    because a temp file's name is normally private to the process that
    created it. Once this program links that same file out under a public,
    predictable name for a student to open (see _publish_atomically), a
    mode of 0o600 is a behavior change from the pre-Audit4 design (which
    used a plain open(path, "w") and therefore got the ordinary
    umask-controlled mode, typically 0o644) -- and an unwelcome one, since
    it silently makes exported CSVs/PNGs unreadable by anyone but the user
    who ran the program, on a multi-user machine or a shared directory.
    This computes what that ordinary open() would have produced, so the
    published file's permissions can be normalized to match (see
    _publish_or_cleanup below) rather than inheriting mkstemp's stricter
    default. Reading the umask requires briefly setting it (os.umask() has
    no read-only form) and then immediately restoring the original value;
    this is not atomic with respect to other threads in the same process
    that also touch the umask, but CPython's umask is process-wide and
    this program does not otherwise set it, so in practice this executes
    without any other umask change interleaved.
    """
    saved = os.umask(0)
    os.umask(saved)
    return 0o666 & ~saved


def _verify_temp_identity(fd, tmp_path):
    """Confirm tmp_path still refers to the exact same regular file that
    `fd` was opened against, immediately before that path is used to
    publish the file under a public name.

    Audit5 (Codex P2-1): _create_private_temp_file() creates tmp_path
    securely and this program always writes through `fd`, not by
    reopening tmp_path -- but _publish_atomically() below still uses
    tmp_path (a name, not a descriptor) as the *source* of os.link(), since
    os.link() has no fd-based form in the standard library usable here.
    Between tmp_path's creation and that os.link() call, something else
    with access to `directory` could in principle delete tmp_path and
    create a new file or a symlink at that same name (e.g. to a sensitive
    file elsewhere) -- os.link() would then publish that substituted
    target's content under this program's output name instead of the
    content this program actually wrote.

    This narrows that window rather than eliminating it: os.lstat(tmp_path)
    (which does NOT follow a symlink) is compared against os.fstat(fd)
    (the descriptor this program has held open since creation) by device
    and inode number. A mismatch, or tmp_path having become a symlink at
    all, means the name no longer identifies the file this program wrote,
    and publication is refused. This still leaves a narrow race between
    this check and the os.link() call immediately after it (a true
    TOCTOU-free guarantee would require an operation that both verifies
    identity and links atomically by fd, which POSIX does not offer
    through Python's standard library) -- so this is a defense-in-depth
    check, not an airtight one, and the atomicity/security claims this
    program makes are scoped to ordinary, non-adversarial output
    directories (a student's own working directory), not to a directory
    under an untrusted party's control.
    """
    try:
        entry = os.lstat(tmp_path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot verify temporary file {tmp_path!r} before publication: {exc}."
        ) from exc
    if stat.S_ISLNK(entry.st_mode):
        raise RuntimeError(
            f"Refusing to publish: {tmp_path!r} has been replaced with a "
            "symlink since it was created."
        )
    fd_entry = os.fstat(fd)
    if (entry.st_dev, entry.st_ino) != (fd_entry.st_dev, fd_entry.st_ino):
        raise RuntimeError(
            f"Refusing to publish: {tmp_path!r} no longer refers to the "
            "file this program created."
        )


def _fsync_directory(directory):
    """Best-effort fsync of a directory's own metadata after a publish, so
    the newly published directory entry is more likely to survive a crash
    that happens shortly afterward.

    This is deliberately best-effort, not required for correctness: the
    absent-or-complete guarantee in _publish_atomically() below holds
    regardless of whether this succeeds. Not every platform supports
    opening a directory this way (notably Windows, where os.open() on a
    directory raises), so failure here is silently ignored rather than
    treated as an error -- it only affects durability across a crash
    landing in the narrow window immediately after publication, which this
    project does not claim to guarantee in the first place (see the
    docstring on _publish_atomically for exactly what is and is not
    promised).
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _publish_atomically(fd, tmp_path, directory, prefix, ext, now, max_attempts=1000):
    """Publish the already-fully-written file at tmp_path under a fresh
    "prefix_timestamp[_n].ext" public name in `directory`, without ever
    creating that public name before the content is complete.

    Audit4 (Codex P2-1): the previous design (see _reserve_unique_path in
    earlier revisions) created the *final* public filename itself, empty,
    as a reservation placeholder before any content existed, then wrote
    content elsewhere and used os.replace() to overwrite that placeholder.
    Between reservation and replace, the public path was visibly present
    with 0 bytes -- observable by a directory watcher, file browser, or a
    student opening the file mid-write -- and a process killed in that
    window left exactly that misleading zero-byte file behind permanently.
    This function never creates the public path until it is already the
    finished file: os.link() creates a second name (this one) pointing at
    tmp_path's already-complete content, and fails atomically with
    FileExistsError if the candidate name is already taken rather than
    silently replacing it -- so a collision is detected and retried with a
    "_1", "_2", ... suffix at the actual publish operation, not merely at
    an earlier placeholder claim. At every instant an external observer
    could inspect `directory`, the public name is therefore either
    completely absent or the complete, correctly named file -- never a
    partial or empty one. os.link() requires tmp_path and the destination
    to be on the same filesystem, which holds here because
    _create_private_temp_file() always creates tmp_path inside this same
    `directory`.

    Audit4 (Copilot A4-1): as with any atomicity claim built on a single
    filesystem operation, this holds specifically on filesystems that
    implement POSIX hard-link/rename semantics as the OS documents them
    (the ordinary case for local Linux/macOS/Windows-NTFS filesystems this
    program is expected to run on) -- not as a claim that spans every
    filesystem a Python program could conceivably be pointed at (e.g. some
    network or FAT-family filesystems weaken these guarantees).

    This guarantees atomicity/visibility, not full crash durability: an
    fsync of file content is the caller's responsibility (see _write_csv's
    explicit fsync and the note on the plot_gw counterpart), and
    _fsync_directory() above is only a best-effort attempt to make the new
    directory entry itself durable, not a guaranteed one.

    Audit5 (Codex P2-1): now also takes the open `fd` from
    _create_private_temp_file() and calls _verify_temp_identity(fd,
    tmp_path) immediately before the first os.link() attempt, so a
    tmp_path that has been deleted-and-recreated or replaced with a
    symlink since creation is caught here rather than silently linked in.
    See _verify_temp_identity's docstring for exactly what this does and
    does not guarantee.
    """
    _verify_temp_identity(fd, tmp_path)
    base = _timestamp_fname(now, prefix=prefix, ext=ext)
    stem = base[: -(len(ext) + 1)]  # strip the trailing ".<ext>"
    for attempt in range(max_attempts):
        candidate = f"{stem}.{ext}" if attempt == 0 else f"{stem}_{attempt}.{ext}"
        path = os.path.join(directory, candidate)
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            continue
        break
    else:
        raise RuntimeError(
            f"Could not publish a unique output filename in {directory!r} "
            f"after {max_attempts} attempts."
        )
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    _fsync_directory(directory)
    return path


def _publish_or_cleanup(write_fn, directory, prefix, ext, now):
    """Build one artifact's content by calling write_fn(fd, tmp_path),
    where tmp_path is a securely-created private temporary file inside
    `directory` and fd is the still-open descriptor mkstemp created it
    with (see _create_private_temp_file), then publish it under a fresh
    "prefix_timestamp[_n].ext" public name using the single already-
    captured `now` instant (see _publish_atomically) -- or, on any failure
    from write_fn or publication, remove the temporary file and re-raise
    without ever having created a public name.

    Audit3 (Codex P2-1; Copilot A3-2/A3-3) originally raised the failure-
    cleanup requirement this implements; Audit4 (Codex P2-1) is the reason
    the underlying reservation/publish mechanism changed; Audit5 (Codex
    P2-1) is why write_fn now receives the open fd instead of just a path
    -- see _create_private_temp_file's and _publish_atomically's
    docstrings for what changed and why.

    write_fn is expected to write through fd (e.g. via
    os.fdopen(fd, mode, ..., closefd=False)) and must NOT close fd itself
    -- this function always closes it exactly once, in the finally block
    below, regardless of how write_fn or publication finishes. Audit5
    (Codex P3-1): on the success path, before publication, the file's
    permissions are normalized from mkstemp's fixed 0o600 to the ordinary
    umask-controlled mode a plain open(path, "w") would have produced (see
    _default_output_file_mode) -- using os.fchmod(fd, ...), which acts on
    the descriptor rather than the path, so this cannot be redirected by a
    symlink substituted at tmp_path the way a path-based os.chmod() could
    be. os.fchmod is POSIX-only (absent on Windows), so it is skipped
    there; Windows' own ACL-based permission model does not have an
    equivalent umask-driven "default mode" for this to normalize toward.
    """
    fd, tmp_path = _create_private_temp_file(directory, ext)
    try:
        write_fn(fd, tmp_path)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, _default_output_file_mode())
        return _publish_atomically(fd, tmp_path, directory, prefix, ext, now)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


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
    (Audit3 Copilot A3-5): it is captured once, right here, before this
    export's write begins -- which for a long integration can be
    measurably earlier than when the CSV write actually finishes on disk,
    so this is an export-*initiation* timestamp, not a completion one
    (Audit4 Codex P3-1 -- an earlier response's wording overstated this as
    an "export-completion timestamp", which was never accurate). That same
    captured instant is also reused, via _publish_or_cleanup()/
    _publish_atomically(), to build this file's own published filename, so
    the filename's embedded timestamp and this metadata field always agree
    exactly (Audit4 Codex P3-1) rather than coming from two independent
    datetime.now() calls that could straddle a second or date boundary.
    """
    os.makedirs(csvdir, exist_ok=True)
    now = datetime.now(timezone.utc)
    s = result["summary"]
    include_rd = s["include_ringdown"]
    export_timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

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

    def _do_write(fd, tmp_path):
        # Audit5 (Codex P2-1): write through the already-open fd (via
        # os.fdopen(..., closefd=False)) instead of reopening tmp_path by
        # name -- see _create_private_temp_file's docstring for why a
        # path-based reopen here would reintroduce the race this fixes.
        # closefd=False means the `with` block's close() at the end closes
        # only this Python-level file object, not the underlying fd, which
        # _publish_or_cleanup() owns and closes itself.
        with os.fdopen(fd, "w", newline="", encoding="utf-8", closefd=False) as handle:
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
            os.fsync(fd)

    fpath = _publish_or_cleanup(_do_write, csvdir, "gw_inspiral", "csv", now)
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
