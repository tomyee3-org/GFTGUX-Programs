"""
plot_gw.py
==========
Three-panel Matplotlib visualization for GravitationalWaveSources.
"""

import os
import stat
import tempfile
import numpy as np
from datetime import datetime, timezone
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


def _timestamp_fname(now, prefix="gw_inspiral"):
    """Format an already-captured aware UTC datetime as a
    "prefix_YYYYMMDD_HHMMSS_ffffff.png" filename.

    Audit4 (Codex P3-1), plot_gw-layer counterpart to driver_gw's fix: this
    now takes the instant to format as an explicit argument rather than
    calling datetime.now() itself, so the one instant captured by the
    caller is reused consistently rather than re-derived.

    Microsecond resolution makes two rapid saves in the same directory
    (e.g. a scripted parameter sweep) collide far less often than under
    the previous second-resolution timestamp, but does not make a
    collision impossible (a coarser OS clock, two calls landing in the
    same microsecond, or a clock adjustment can still repeat a value).
    _publish_atomically() below is what actually guarantees no silent
    overwrite; this function only formats the first candidate name.

    Audit3 (Gemini): UTC, matching driver_gw's CSV filenames and its
    export_timestamp_utc metadata field -- so PNG and CSV filenames from
    the same run sort and compare directly without a timezone conversion.
    """
    ts = now.strftime("%Y%m%d_%H%M%S_%f")
    return f"{prefix}_{ts}.png"


def _create_private_temp_file(directory, ext="png"):
    """Securely create an empty, randomly-named temporary file in
    `directory` to render the PNG into before publication.

    See driver_gw._create_private_temp_file for the full rationale (Audit4
    Codex P2-1): a caller-predictable "<final-name>.tmp" path is guessable
    in advance and, opened with an ordinary open()/savefig() call, follows
    a pre-existing symlink at that path rather than refusing to.
    tempfile.mkstemp() draws its name from a wide random namespace and
    creates it with O_CREAT|O_EXCL, so nothing else on the system can
    anticipate or pre-plant a symlink at this name; duplicated here rather
    than imported, per this project's existing convention of not sharing
    code across the driver_gw/plot_gw split.

    Audit5 (Codex P2-1): the returned fd is kept open and returned
    alongside tmp_path, instead of being closed here -- see
    driver_gw._create_private_temp_file's docstring for the full
    rationale (a path-based reopen of tmp_path, after fd is closed, is a
    second operation separated in time from this one, and can be raced).
    """
    fd, tmp_path = tempfile.mkstemp(prefix=".gw_tmp_", suffix=f".{ext}", dir=directory)
    return fd, tmp_path


def _default_output_file_mode():
    """Return the permission mode an ordinary open(path, "w") would have
    produced, given the process's current umask.

    See driver_gw._default_output_file_mode for the full rationale (Audit5
    Codex P3-1); duplicated here rather than imported, per this project's
    existing convention of not sharing code across the driver_gw/plot_gw
    split.
    """
    saved = os.umask(0)
    os.umask(saved)
    return 0o666 & ~saved


def _verify_temp_identity(fd, tmp_path):
    """Confirm tmp_path still refers to the exact same regular file that
    `fd` was opened against, immediately before that path is used to
    publish the file under a public name.

    See driver_gw._verify_temp_identity for the full rationale (Audit5
    Codex P2-1) and for exactly what this does and does not guarantee;
    duplicated here rather than imported, per this project's existing
    convention of not sharing code across the driver_gw/plot_gw split.
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
    """Best-effort fsync of a directory's own metadata after a publish.
    See driver_gw._fsync_directory for the full rationale; duplicated here
    rather than imported, per this project's existing convention.
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


def _publish_atomically(fd, tmp_path, directory, prefix, now, ext="png", max_attempts=1000):
    """Publish the already-fully-written file at tmp_path under a fresh
    "prefix_timestamp[_n].png" public name in `directory`, without ever
    creating that public name before the content is complete.

    See driver_gw._publish_atomically for the full rationale (Audit4 Codex
    P2-1): the previous design reserved the *final* filename itself, empty,
    before any content existed, so the public path was visibly a zero-byte
    placeholder for the entire duration of a savefig() call and a process
    killed mid-save left exactly that misleading file behind. os.link()
    here creates the public name only once tmp_path already holds the
    complete PNG, and fails atomically (retried with a "_1", "_2", ...
    suffix) rather than silently overwriting a same-named collision.

    Audit5 (Codex P2-1): now also takes the open `fd` and calls
    _verify_temp_identity(fd, tmp_path) immediately before the first
    os.link() attempt -- see driver_gw._publish_atomically's Audit5 note.
    """
    _verify_temp_identity(fd, tmp_path)
    base = _timestamp_fname(now, prefix=prefix)
    stem = base[: -(len(ext) + 1)]
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


def _publish_or_cleanup(write_fn, directory, prefix, now):
    """Build the PNG by calling write_fn(fd, tmp_path), where tmp_path is a
    securely-created private temporary file inside `directory` and fd is
    the still-open descriptor mkstemp created it with, then publish it
    under a fresh "prefix_timestamp[_n].png" public name using the single
    already-captured `now` instant -- or, on any failure, remove the
    temporary file and re-raise without ever having created a public name.
    See driver_gw._publish_or_cleanup for the full rationale (including
    the Audit5 Codex P2-1/P3-1 changes to this signature); duplicated here
    rather than imported, per this project's existing convention of not
    sharing code across the driver_gw/plot_gw split.

    write_fn is expected to write through fd (e.g. via a file object
    wrapped with closefd=False) and must NOT close fd itself -- this
    function always closes it exactly once, in the finally block below.
    """
    fd, tmp_path = _create_private_temp_file(directory)
    try:
        write_fn(fd, tmp_path)
        if hasattr(os, "fchmod"):
            os.fchmod(fd, _default_output_file_mode())
        return _publish_atomically(fd, tmp_path, directory, prefix, now)
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

    # Validate/create the output directory before building the figure, not
    # only before saving into it -- a bad --outdir (e.g. an existing
    # regular file) is then rejected before any Matplotlib work happens at
    # all. driver_gw.run() additionally validates/creates this same
    # directory (and --csvdir's) even earlier, before either requested
    # artifact is written, when this function is reached through run();
    # this call keeps plot_inspiral() safe to call directly too.
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    try:
        _render(fig, axes, result, t_isco, f_isco, s, insp, rd, valid_A,
                x_lo, x_hi, lw)

        # The figure is always shown on screen. When --outdir is given, a
        # timestamped PNG is *additionally* saved to that folder; saving
        # happens before plt.show() so the file is written even if the
        # user closes the interactive window without waiting, or the run
        # is otherwise interrupted after the window appears. The save is
        # published atomically (write to a securely-created private temp
        # file, then os.link() it into place under a name that did not
        # already publicly exist -- see _publish_atomically()): a failed
        # savefig() leaves no PNG behind rather than a zero-byte or partial
        # one (Audit3 Codex P2-1 / Copilot A3-3; Audit4 Codex P2-1 replaced
        # the reservation/rename mechanism itself -- see that docstring).
        if outdir is not None:
            def _do_save(fd, tmp_path):
                # Audit5 (Codex P2-1): save through the already-open fd
                # (via os.fdopen(..., closefd=False)) rather than passing
                # tmp_path to savefig() by name -- see
                # driver_gw._create_private_temp_file's docstring for why
                # a path-based reopen here would reintroduce the race this
                # fixes. format="png" stays explicit: a file object's
                # .name is not a real ".png" path Matplotlib can sniff an
                # extension from.
                with os.fdopen(fd, "wb", closefd=False) as handle:
                    fig.savefig(handle, dpi=dpi, bbox_inches="tight", format="png")
                    # Audit4 (Codex P2-1, item 4): fsync the PNG's own file
                    # content before it is published -- Matplotlib's
                    # savefig() does not fsync on its own, and
                    # _publish_atomically() only best-effort-fsyncs the
                    # *directory* entry, not file data.
                    handle.flush()
                    os.fsync(fd)

            now = datetime.now(timezone.utc)
            fpath = _publish_or_cleanup(_do_save, outdir, "gw_inspiral", now)
            print(f"[plot_gw] PNG saved -> {fpath}")

        print("[plot_gw] Displaying figure on screen ...")
        plt.show()
    finally:
        # Audit3 (Codex P2-1, item 5): this must run even if savefig() or
        # anything else above raised, so a plotting exception can never
        # leak a live figure into pyplot's registry.
        plt.close(fig)


def _render(fig, axes, result, t_isco, f_isco, s, insp, rd, valid_A,
            x_lo, x_hi, lw):
    """Draw all three panels onto an already-created figure/axes."""
    ax1, ax2, ax3 = axes
    t, h, A, f = result["t"], result["h"], result["A"], result["f"]

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
            "illustrative ringdown --",
            "not a physical merger",
        ])
    ann_text = "\n".join(ann_lines)

    ax1.plot(t[insp], h[insp], color=C_WAVE, lw=lw, label="Inspiral strain scale $h(t)$")
    if np.any(rd):
        ax1.plot(t[rd], h[rd], color=C_RD, lw=lw, label="Illustrative Schwarzschild QNM")
    ax1.plot(t[valid_A], A[valid_A], color=C_ENV, lw=0.9, ls="--", label=r"Envelope $\pm A(t)$")
    ax1.plot(t[valid_A], -A[valid_A], color=C_ENV, lw=0.9, ls="--")
    ax1.axvline(t_isco, **cutoff_kw)
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylabel("Strain scale $h(t)$ (dimensionless)")
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
    ax2.set_ylabel("Amplitude $A(t)$ (dimensionless)")
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
