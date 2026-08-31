"""
physics_gw.py
=============
Core physics engine for the GravitationalWaveSources program.

Computes the leading-order quadrupole inspiral of a quasi-circular compact
binary up to the Schwarzschild ISCO frequency.  An optional, explicitly
illustrative Schwarzschild quasi-normal-mode (QNM) ringdown can be appended.

SI units are used internally; user-facing masses and distances are in solar
masses and megaparsecs.
"""

import math
import sys
import numpy as np

MODEL_VERSION = "1.2.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_gw.py",
    "driver_gw.py",
    "main.py",
    "plot_gw.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same declared
    version.  The hash is independent of LF versus CRLF line endings and
    frames each file with its name and length so file-boundary changes cannot
    collide with an unchanged concatenated byte stream.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located or decoded, as can happen in some
    frozen or zipped distributions.
    """
    import hashlib
    import os

    try:
        here = os.path.dirname(os.path.abspath(__file__))
        digest = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as source:
                content = source.read().encode("utf-8")
            digest.update(name.encode("utf-8"))
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        return digest.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# Physical constants
G = 6.674_30e-11       # m^3 kg^-1 s^-2
c = 2.997_924_58e8     # m s^-1
M_sun = 1.988_92e30    # kg
MPC_M = 3.085_677_581_49e22

MAX_INSPIRAL_STEPS = 5_000_000
MIN_INSPIRAL_STEPS = 50
MAX_RINGDOWN_POINTS = 500_000

#: Minimum samples per QNM oscillation cycle the stored ringdown must
#: provide. 2 (bare Nyquist) is a necessary but visually inadequate
#: standard for a teaching plot: a value just above 2 samples/cycle still
#: aliases badly on screen, and --rd_pts=2 stores exactly one ringdown
#: point (after the seam point is dropped -- see integrate_inspiral),
#: which cannot be rendered as a visible line at all. 8 is a conservative
#: value that keeps the default settings (n_ringdown_tau=6, ringdown_pts=
#: 4000) comfortably valid while still catching grossly under-sampled
#: requests before they silently produce a misleading or invisible plot.
MIN_RINGDOWN_SAMPLES_PER_CYCLE = 8

#: Natural-log bounds on a representable positive double, used by
#: chirp_mass_from_fdot() to detect an out-of-range implied result while
#: still working entirely in the log domain (see that function for why).
#: _LOG_FLOAT_MIN uses the smallest *normal* double (not the smallest
#: subnormal) so a "successful" inference is never a reduced-precision
#: subnormal result -- a chirp mass that tiny is not physically meaningful
#: at this program's leading-order pedagogical level regardless.
_LOG_FLOAT_MAX = math.log(sys.float_info.max)
_LOG_FLOAT_MIN = math.log(sys.float_info.min)


def _require_finite(name, value):
    """Return value as float after giving a consistent user-facing error.

    ``bool`` (and ``numpy.bool_``) are rejected explicitly even though
    ``float(True) == 1.0`` would otherwise convert silently: a Boolean flag
    is never a meaningful mass, distance, timestep, or frequency, and
    accepting one without complaint would let a programmatic caller (e.g. a
    notebook) pass a stray flag into a physical quantity unnoticed.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{name} must be a finite number; got {value!r} (a bool is not "
            "an accepted numeric value)."
        )
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError happens for a Python int too large to represent as a
        # float (e.g. a CLI argument like --n_tau with hundreds of digits);
        # it is a value problem for this caller, not a programming error, so
        # it is normalized into the same ValueError as every other bad input.
        raise ValueError(f"{name} must be a finite number; got {value!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return value


def chirp_mass(m1_kg, m2_kg):
    """Return chirp mass in kg for two positive component masses.

    This is the *forward* calculation: chirp mass computed directly from
    already-known component masses. See chirp_mass_from_fdot() for the
    *observational* inverse -- inferring chirp mass from a measured
    frequency and its rate of change, the calculation actually used to
    estimate chirp mass from a detected chirp.
    """
    return (m1_kg * m2_kg) ** 0.6 / (m1_kg + m2_kg) ** 0.2


def chirp_mass_from_fdot(f_hz, fdot_hz_per_s):
    """Infer chirp mass in kg from an observed GW frequency and df/dt.

    Inverts the same leading-order quadrupole relation used by dfdt():

        df/dt = (96/5) pi^(8/3) (G Mc/c^3)^(5/3) f^(11/3)
        =>  Mc = (c^3/G) [ (5/96) pi^(-8/3) f^(-11/3) (df/dt) ]^(3/5)

    This is the leading-order version of the relation used observationally
    to estimate a binary's chirp mass directly from its measured frequency
    sweep (e.g. in the GW150914 discovery analysis), as distinct from
    chirp_mass(), which requires already knowing the two component masses.
    Both f_hz and fdot_hz_per_s must be strictly positive: a gravitational-
    wave frequency is positive by definition, and an inspiraling binary's
    GW frequency always increases (df/dt > 0) under this leading-order
    model.
    """
    f_hz = _require_finite("f_hz", f_hz)
    fdot_hz_per_s = _require_finite("fdot_hz_per_s", fdot_hz_per_s)
    if f_hz <= 0:
        raise ValueError("f_hz must be greater than zero.")
    if fdot_hz_per_s <= 0:
        raise ValueError(
            "fdot_hz_per_s must be greater than zero (an inspiraling "
            "binary's GW frequency always increases under this model)."
        )
    # Evaluate in the log domain rather than forming f**(-11/3) and
    # fdot**1 as intermediate powers/products directly. Audit2 (Codex P2-1)
    # found that the direct form can individually overflow (OverflowError)
    # or underflow to exactly 0.0 for f or fdot near the extreme ends of
    # the representable float range, even when the true, mathematically
    # implied chirp mass is a normal, representable, nonzero number -- or,
    # in the opposite direction, is genuinely outside any representable
    # range and should be reported as such rather than silently returned
    # as 0.0. Working in log space keeps every intermediate value in a
    # moderate numeric range (natural logs of doubles span roughly
    # -745..+709, never overflowing/underflowing on their own), so the
    # only overflow/underflow decision left is the single explicit
    # boundary check on the final log-domain result below.
    log_Mc_kg = (
        (3.0 / 5.0) * math.log(5.0 / 96.0)
        - (8.0 / 5.0) * math.log(math.pi)
        + 3.0 * math.log(c)
        - math.log(G)
        - (11.0 / 5.0) * math.log(f_hz)
        + (3.0 / 5.0) * math.log(fdot_hz_per_s)
    )
    if log_Mc_kg > _LOG_FLOAT_MAX:
        raise ValueError(
            f"f_hz={f_hz:g} and fdot_hz_per_s={fdot_hz_per_s:g} imply a "
            "chirp mass too large to represent as a finite number; check "
            "the units and magnitude of the inputs."
        )
    if log_Mc_kg < _LOG_FLOAT_MIN:
        raise ValueError(
            f"f_hz={f_hz:g} and fdot_hz_per_s={fdot_hz_per_s:g} imply a "
            "chirp mass too small to represent as a positive finite "
            "number; check the units and magnitude of the inputs."
        )
    Mc_kg = math.exp(log_Mc_kg)
    if not (math.isfinite(Mc_kg) and Mc_kg > 0):
        raise ValueError(
            f"f_hz={f_hz:g} and fdot_hz_per_s={fdot_hz_per_s:g} did not "
            "produce a positive finite chirp mass."
        )
    return Mc_kg


def dfdt(f_hz, Mc_kg):
    """Leading-order quadrupole derivative df/dt for GW frequency f."""
    Mc_geom = G * Mc_kg / c**3
    return ((96.0 / 5.0) * np.pi**(8.0 / 3.0)
            * Mc_geom**(5.0 / 3.0) * f_hz**(11.0 / 3.0))


def strain_amplitude(f_hz, Mc_kg, d_m):
    """
    Newtonian quadrupole strain-amplitude scale.

    This is the commonly used face-on amplitude scale

        A = 4 (G Mc)^(5/3) (pi f)^(2/3) / (c^4 d),

    not a sky-and-polarisation-averaged detector response.
    """
    return ((4.0 * G * Mc_kg / (c**2 * d_m))
            * (np.pi * G * Mc_kg * f_hz / c**3) ** (2.0 / 3.0))


def f_isco(M_total_kg):
    """GW frequency at the Schwarzschild ISCO, r = 6 GM/c^2."""
    return c**3 / (6.0**1.5 * np.pi * G * M_total_kg)


def qnm_params(M_final_kg):
    """
    Return (f_qnm [Hz], tau_qnm [s]) for the fundamental l=2
    Schwarzschild gravitational QNM, using M omega = 0.3737 - 0.0890 i.
    """
    scale = G * M_final_kg / c**3
    return 0.3737 / (2.0 * np.pi * scale), scale / 0.0890


def inspiral_time(f_start_hz, f_end_hz, Mc_kg):
    """Analytic leading-order time required to chirp from f_start to f_end."""
    Mc_geom = G * Mc_kg / c**3
    coeff = 5.0 / (256.0 * np.pi**(8.0 / 3.0) * Mc_geom**(5.0 / 3.0))
    return coeff * (f_start_hz**(-8.0 / 3.0) - f_end_hz**(-8.0 / 3.0))


def _rk4_frequency_phase_step(f, phase, dt, Mc):
    """One coupled RK4 step for frequency and GW phase."""
    k1_f = dfdt(f, Mc)
    k1_p = 2.0 * np.pi * f

    f2 = f + 0.5 * dt * k1_f
    k2_f = dfdt(f2, Mc)
    k2_p = 2.0 * np.pi * f2

    f3 = f + 0.5 * dt * k2_f
    k3_f = dfdt(f3, Mc)
    k3_p = 2.0 * np.pi * f3

    f4 = f + dt * k3_f
    k4_f = dfdt(f4, Mc)
    k4_p = 2.0 * np.pi * f4

    f_next = f + (dt / 6.0) * (k1_f + 2*k2_f + 2*k3_f + k4_f)
    phase_next = phase + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    return f_next, phase_next


def integrate_inspiral(m1_msun, m2_msun, d_mpc,
                       dt=2e-4, f_start=20.0,
                       include_ringdown=False,
                       n_ringdown_tau=6, ringdown_pts=4000):
    """
    Integrate a leading-order quasi-circular inspiral to the Schwarzschild
    ISCO cutoff. Optionally append a deliberately simplified Schwarzschild
    QNM ringdown.

    The ringdown is a pedagogical extension, not a physical merger model.
    In particular it should not be interpreted as the generic post-merger
    signal of a binary-neutron-star system.

    API note on integer-valued parameters: ``include_ringdown`` must be a
    real ``bool`` or ``numpy.bool_`` (not ``0``/``1`` or another duck-typed
    boolean-like object -- see _require_finite's bool guard, which applies
    to every other argument here). ``n_ringdown_tau``, ``ringdown_pts``,
    and (at the driver layer) ``--dpi`` are conceptually integer counts,
    but this function accepts any integer-*valued* float for them (e.g.
    ``4000.0``) as a convenience for callers who already have
    floating-point arithmetic results on hand; each is validated with
    ``int(value) == value`` and rejected otherwise, then converted to
    ``int`` before use. ``n_ringdown_tau`` and ``ringdown_pts`` are only
    inspected (and therefore can only be rejected) when
    ``include_ringdown`` is ``True``; when ringdown is disabled they have
    no effect on the calculation and are not validated at all, so a
    caller carrying an unused/invalid ringdown configuration alongside a
    pure-inspiral request is not broken by it.
    """
    # Validate scalar inputs before unit conversion.
    m1_msun = _require_finite("m1_msun", m1_msun)
    m2_msun = _require_finite("m2_msun", m2_msun)
    d_mpc = _require_finite("d_mpc", d_mpc)
    dt = _require_finite("dt", dt)
    f_start = _require_finite("f_start", f_start)

    if m1_msun <= 0 or m2_msun <= 0:
        raise ValueError("Component masses must both be greater than zero.")
    if d_mpc <= 0:
        raise ValueError("Luminosity distance d_mpc must be greater than zero.")
    if dt <= 0:
        raise ValueError("Integration timestep dt must be greater than zero.")
    if f_start <= 0:
        raise ValueError("Starting GW frequency f_start must be greater than zero.")
    if not isinstance(include_ringdown, (bool, np.bool_)):
        raise ValueError("include_ringdown must be True or False.")

    # n_ringdown_tau/ringdown_pts are meaningless when ringdown is not
    # requested, so they are validated (and can only reject the call) when
    # include_ringdown is True -- see the API note above (Audit2/Copilot A2-4).
    if include_ringdown:
        n_ringdown_tau = _require_finite("n_ringdown_tau", n_ringdown_tau)
        ringdown_pts = _require_finite("ringdown_pts", ringdown_pts)
        if int(n_ringdown_tau) != n_ringdown_tau or n_ringdown_tau <= 0:
            raise ValueError("n_ringdown_tau must be a positive integer.")
        if int(ringdown_pts) != ringdown_pts or ringdown_pts < 2:
            raise ValueError("ringdown_pts must be an integer of at least 2.")
        if ringdown_pts > MAX_RINGDOWN_POINTS:
            raise ValueError(
                f"ringdown_pts must not exceed {MAX_RINGDOWN_POINTS:,}; "
                "reduce --rd_pts."
            )
        n_ringdown_tau = int(n_ringdown_tau)
        ringdown_pts = int(ringdown_pts)

    # Extreme-but-technically-finite inputs (e.g. dt or f_start at the
    # smallest positive subnormal float) can make the derived arithmetic
    # below overflow even though each individual input passed
    # _require_finite on its own. Convert that into the same clean
    # ValueError as every other out-of-range request instead of letting an
    # OverflowError/ZeroDivisionError escape as a bare traceback.
    try:
        m1 = m1_msun * M_sun
        m2 = m2_msun * M_sun
        M_total = m1 + m2
        Mc = chirp_mass(m1, m2)
        d = d_mpc * MPC_M
        f_isco_hz = f_isco(M_total)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            "The requested component masses and/or distance are outside "
            "the range this leading-order model can compute; use more "
            "moderate values."
        ) from exc
    if not (math.isfinite(M_total) and math.isfinite(Mc) and math.isfinite(d)
            and math.isfinite(f_isco_hz) and f_isco_hz > 0):
        raise ValueError(
            "The requested component masses and/or distance produced a "
            "non-finite derived quantity; use more moderate values."
        )

    # Validate the optional ringdown's coupled sampling requirement here,
    # as soon as M_total (and hence the toy remnant mass and QNM
    # parameters) is known -- deliberately *before* estimating or running
    # the inspiral integration below. Audit2 (Codex P2-3 / Copilot A2-2)
    # found that checking this only after the full inspiral had already
    # been integrated and copied into NumPy arrays let an invalid ringdown
    # request (e.g. the default BNS case with --rd_pts 2) burn several
    # seconds and ~200 MiB before being rejected, even though nothing
    # about this check depends on the inspiral waveform itself.
    f_qnm = np.nan
    tau_qnm = np.nan
    M_final = np.nan
    if include_ringdown:
        # Deliberate toy assumption: 5% of total mass radiated and zero remnant spin.
        M_final = 0.95 * M_total
        try:
            f_qnm, tau_qnm = qnm_params(M_final)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError(
                "The requested component masses do not yield computable "
                "ringdown QNM parameters; use more moderate values."
            ) from exc
        if not (math.isfinite(f_qnm) and math.isfinite(tau_qnm)
                and f_qnm > 0 and tau_qnm > 0):
            raise ValueError(
                "The requested component masses produced non-finite "
                "ringdown QNM parameters; use more moderate values."
            )

        # A stored ringdown that cannot resolve the QNM oscillation itself
        # is worse than useless for a teaching plot: at exactly the Nyquist
        # limit (2 samples/cycle) the waveform aliases, and at --rd_pts=2 it
        # collapses to a single stored point (see the t_rd[1:] slice below)
        # that cannot be rendered as a visible line at all, even though the
        # legend would still claim a ringdown curve is present. Require a
        # visually adequate sampling rate up front instead of silently
        # producing a misleading or empty-looking plot.
        #
        # An astronomically large --n_tau (e.g. 1e308) makes this product
        # overflow to float('inf') -- float multiplication overflows
        # silently to inf rather than raising, unlike float()/** -- so
        # isfinite() is checked explicitly before math.ceil(), which does
        # raise OverflowError on an infinite input (Audit2 Codex P2-2).
        try:
            duration = n_ringdown_tau * tau_qnm
            cycles = duration * f_qnm
            raw_min_pts = MIN_RINGDOWN_SAMPLES_PER_CYCLE * cycles
        except OverflowError:
            raw_min_pts = math.inf
        if not math.isfinite(raw_min_pts):
            raise ValueError(
                f"n_ringdown_tau={n_ringdown_tau:.4g} is too large for this "
                "leading-order ringdown model to compute a required sample "
                "count; reduce --n_tau."
            )
        min_pts_needed = math.ceil(raw_min_pts) + 1
        if ringdown_pts < min_pts_needed:
            raise ValueError(
                f"ringdown_pts={ringdown_pts} is too small to resolve the "
                f"QNM oscillation (f_QNM={f_qnm:.4g} Hz) over "
                f"n_ringdown_tau={n_ringdown_tau} decay times; at least "
                f"{min_pts_needed} points are needed for "
                f"{MIN_RINGDOWN_SAMPLES_PER_CYCLE} samples per cycle. "
                "Increase --rd_pts or reduce --n_tau."
            )

    if f_start >= f_isco_hz:
        raise ValueError(
            f"f_start={f_start:g} Hz must be below the Schwarzschild ISCO "
            f"frequency ({f_isco_hz:.3g} Hz) for these masses."
        )

    try:
        T_est = inspiral_time(f_start, f_isco_hz, Mc)
        estimated_steps = math.ceil(T_est / dt)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            f"f_start={f_start:g} Hz and/or dt={dt:g} s are outside the "
            "range this leading-order model can compute; use more "
            "moderate values."
        ) from exc
    if not math.isfinite(T_est) or T_est <= 0:
        raise ValueError(
            f"f_start={f_start:g} Hz produced a non-finite or non-positive "
            "estimated inspiral time; use a larger f_start."
        )

    # A sampled waveform cannot represent the highest inspiral frequency if
    # the timestep violates the Nyquist criterion at the ISCO cutoff.  This is
    # only a necessary resolution condition; students should still check
    # convergence with smaller timesteps when late-time phase matters.
    nyquist_dt = 1.0 / (2.0 * f_isco_hz)
    if dt >= nyquist_dt:
        raise ValueError(
            f"dt={dt:g} s is too large to sample the waveform at the ISCO "
            f"frequency ({f_isco_hz:.3g} Hz); use dt < {nyquist_dt:.3g} s "
            "and verify convergence with smaller timesteps."
        )

    if estimated_steps < MIN_INSPIRAL_STEPS:
        raise ValueError(
            f"dt is too large to resolve this inspiral (only about "
            f"{estimated_steps} steps); reduce --dt."
        )
    if estimated_steps > MAX_INSPIRAL_STEPS:
        raise ValueError(
            f"The requested run would require about {estimated_steps:,} inspiral "
            f"steps, exceeding the safety limit of {MAX_INSPIRAL_STEPS:,}. "
            "Increase dt or f_start."
        )

    t_list = []
    f_list = []
    A_list = []
    h_list = []
    phase_list = []

    t = 0.0
    f = float(f_start)
    phase = 0.0

    # Store the initial point and then advance with coupled RK4.
    while True:
        A = strain_amplitude(f, Mc, d)
        t_list.append(t)
        f_list.append(f)
        A_list.append(A)
        h_list.append(A * np.cos(phase))
        phase_list.append(phase)

        if f >= f_isco_hz:
            break

        f_next, phase_next = _rk4_frequency_phase_step(f, phase, dt, Mc)
        if not (np.isfinite(f_next) and np.isfinite(phase_next)):
            raise RuntimeError(
                "Numerical integration produced a non-finite state; "
                "try a smaller --dt."
            )
        if f_next <= f:
            raise RuntimeError(
                "Numerical integration failed to increase GW frequency; "
                "try a smaller --dt."
            )

        if f_next >= f_isco_hz:
            # Interpolate the final partial step to end exactly at the ISCO cutoff.
            frac = (f_isco_hz - f) / (f_next - f)
            t += frac * dt
            phase += frac * (phase_next - phase)
            f = f_isco_hz
        else:
            t += dt
            f = f_next
            phase = phase_next

        if len(t_list) > MAX_INSPIRAL_STEPS:
            raise RuntimeError("Inspiral exceeded the internal step safety limit.")

    t_isco = t
    phase_isco = phase
    A_peak = strain_amplitude(f_isco_hz, Mc, d)

    t_arr = np.asarray(t_list, dtype=float)
    h_arr = np.asarray(h_list, dtype=float)
    A_arr = np.asarray(A_list, dtype=float)
    f_arr = np.asarray(f_list, dtype=float)
    phase_arr = np.asarray(phase_list, dtype=float)

    # Optional illustrative Schwarzschild ringdown. f_qnm/tau_qnm/M_final and
    # the coupled sampling requirement were already computed and validated
    # above, before this inspiral was integrated -- this block only builds
    # the ringdown arrays using those already-known-good values.
    if include_ringdown:
        duration = n_ringdown_tau * tau_qnm
        t_rd = np.linspace(0.0, duration, ringdown_pts)
        phase_rd = phase_isco + 2.0 * np.pi * f_qnm * t_rd
        h_rd = A_peak * np.cos(phase_rd) * np.exp(-t_rd / tau_qnm)

        # Skip t_rd[0] because the inspiral array already contains the ISCO point.
        t_arr = np.concatenate([t_arr, t_isco + t_rd[1:]])
        h_arr = np.concatenate([h_arr, h_rd[1:]])
        A_arr = np.concatenate([A_arr, np.full(ringdown_pts - 1, np.nan)])
        f_arr = np.concatenate([f_arr, np.full(ringdown_pts - 1, np.nan)])
        phase_arr = np.concatenate([phase_arr, phase_rd[1:]])

    summary = dict(
        m1_msun=m1_msun,
        m2_msun=m2_msun,
        Mc_msun=Mc / M_sun,
        M_total_msun=M_total / M_sun,
        d_mpc=d_mpc,
        dt_s=dt,
        f_start_hz=f_start,
        f_isco_hz=f_isco_hz,
        t_isco_s=t_isco,
        T_band_s=t_isco,
        A_isco=A_peak,
        include_ringdown=bool(include_ringdown),
        n_ringdown_tau=(n_ringdown_tau if include_ringdown else None),
        ringdown_pts=(ringdown_pts if include_ringdown else None),
        M_final_msun=(M_final / M_sun if include_ringdown else np.nan),
        f_qnm_hz=f_qnm,
        tau_qnm_ms=(tau_qnm * 1e3 if include_ringdown else np.nan),
        estimated_newtonian_time_s=T_est,
        inspiral_steps=len(t_list),
        model_version=MODEL_VERSION,
        build_id=BUILD_ID,
    )

    return dict(
        t=t_arr,
        h=h_arr,
        A=A_arr,
        f=f_arr,
        phase=phase_arr,
        t_isco=t_isco,
        f_isco_hz=f_isco_hz,
        Mc_msun=Mc / M_sun,
        summary=summary,
    )
