"""
physics_cosmo.py
=================
Core physics engine for Cosmology_expansion_simulator.

This module integrates the Friedmann equation for a homogeneous, isotropic
(FLRW) universe filled with matter, radiation, spatial curvature, and a
dark-energy fluid with a Chevallier-Polarski-Linder (CPL) equation of state
w(a) = w0 + wa(1-a). Three distinct senses of "exact" matter here, and they
should not be conflated: the governing equations are exact consequences of
General Relativity once the FLRW symmetry and the energy content are
specified; the CPL ansatz for w(a) is not exact but a phenomenological
placeholder for physics nobody yet understands (the nature of dark
energy); and this module's numerical output (integration, root-finding,
quadrature) is not exact either, though its error is small and
well-characterized away from event boundaries (see main.py and the help
file for details).

Because every quantity is expressed through the dimensionless density
parameters Omega_i = rho_i / rho_crit and the Hubble constant H0, neither G
nor c ever needs to appear explicitly. They are absorbed into the
definitions of H0 and rho_crit = 3 H0^2 / (8 pi G).

Units: internally, time is tracked in Gyr and H0 is converted from
km/s/Mpc to Gyr^-1 once, at the start of a calculation. All results are
returned in the same student-facing units.
"""

import math
import numpy as np

MODEL_VERSION = "1.1.0"

# ----------------------------------------------------------------------
# Constants (SI, used only for the km/s/Mpc <-> Gyr^-1 conversion)
# ----------------------------------------------------------------------
MPC_M = 3.085_677_581_49e22      # metres per megaparsec. 1 pc = 648000/pi AU
                                  # is exact by IAU 2015 convention, but pi is
                                  # irrational, so this finite decimal is a
                                  # (double-precision-exact-enough) rounded
                                  # approximation to that exact definition,
                                  # not itself an exact value.
YEAR = 3.155_760_0e7             # seconds per Julian year (365.25 days), the
GYR_S = 1.0e9 * YEAR             # standard astronomical year used for Gyr
KMS_MPC_PER_INV_GYR = MPC_M / (1000.0 * GYR_S)
#   H [km/s/Mpc] = H [Gyr^-1] * KMS_MPC_PER_INV_GYR

MAX_STEPS = 2_000_000
MAX_BISECTIONS = 100

RADIATION_W = 1.0 / 3.0
MATTER_W = 0.0


class _Overshoot(Exception):
    """Internal signal: E(a)^2 went negative -- a's expanding branch has
    reached (or would cross) a turning point where H = 0."""


# ----------------------------------------------------------------------
# Small validation helpers (style shared with the rest of GFTGUX)
# ----------------------------------------------------------------------
def _finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


# ----------------------------------------------------------------------
# Cosmological parameter bookkeeping
# ----------------------------------------------------------------------
def resolve_omega_de(omega_m, omega_r, omega_de):
    """
    Resolve the dark-energy density today and the curvature density today.

    If omega_de is None, the model is forced flat: omega_de is set to
    1 - omega_m - omega_r and omega_k = 0 identically. If omega_de is
    given explicitly, omega_k = 1 - omega_m - omega_r - omega_de is
    whatever that arithmetic implies -- positive (open), negative
    (closed), or zero (flat). This is not a fit: the closure relation
    Omega_m + Omega_r + Omega_k + Omega_DE = 1 is the algebraic
    definition of Omega_k, not an independent physical assumption.
    """
    omega_m = _finite("omega_m", omega_m)
    omega_r = _finite("omega_r", omega_r)
    if omega_m < 0:
        raise ValueError("omega_m must be non-negative.")
    if omega_r < 0:
        raise ValueError("omega_r must be non-negative.")
    if omega_de is None:
        omega_de = 1.0 - omega_m - omega_r
        omega_k = 0.0
    else:
        omega_de = _finite("omega_de", omega_de)
        omega_k = 1.0 - omega_m - omega_r - omega_de
    return omega_de, omega_k


def H0_per_Gyr(H0_kms_mpc):
    H0_kms_mpc = _finite("H0", H0_kms_mpc)
    if H0_kms_mpc <= 0:
        raise ValueError("H0 must be greater than zero.")
    return H0_kms_mpc / KMS_MPC_PER_INV_GYR


# ----------------------------------------------------------------------
# Dark-energy equation of state and density evolution (CPL)
# ----------------------------------------------------------------------
def w_de(a, w0, wa):
    """Instantaneous dark-energy equation-of-state parameter w(a) = w0 + wa(1-a)."""
    return w0 + wa * (1.0 - a)


_LOG_G_LIMIT = 300.0  # A conservative POLICY threshold, not a hardware
                       # representability limit: exp(+-300) is finite in
                       # float64 (float64 overflows only above ln|value|
                       # of about 709.78). |ln g(a)| > 300 already means
                       # a dark-energy density ratio of at least ~1e130
                       # relative to today -- far outside any physically
                       # sane parameter combination -- so it is rejected
                       # as a domain error well before it could reach the
                       # true float64 overflow boundary, rather than
                       # being allowed to silently overflow to inf.


def de_density_shape(a, w0, wa):
    """
    rho_DE(a) / rho_DE(1), the exact solution of the fluid continuity
    equation d(ln rho)/d(ln a) = -3(1+w(a)) for the CPL ansatz w(a) =
    w0 + wa(1-a). This is exact GIVEN that ansatz; the ansatz itself is
    a phenomenological placeholder for unknown physics (tag: FIT).

    Evaluated in log space (ln g(a) = -3(1+w0+wa) ln(a) - 3 wa(1-a))
    rather than as np.power(...)*np.exp(...) directly: for large |wa|
    or a wide a-range the direct form can silently overflow to inf (or
    underflow to 0) in floating point, which downstream root-finding
    code could otherwise mistake for a genuine physical event. Instead,
    an implausibly large |ln g(a)| is rejected outright with a clear
    error. This is a deliberate, conservative POLICY threshold
    (|ln g(a)| > 300, i.e. a dark-energy density ratio beyond ~1e130
    relative to today) chosen well inside float64's actual overflow
    boundary (~e^709.78) -- the parameter combination is not literally
    unrepresentable, it is simply far outside any physically sane
    regime this program is meant to model.
    """
    a = np.asarray(a, dtype=float)
    if np.any(a <= 0.0):
        raise ValueError("de_density_shape: a must be positive.")
    with np.errstate(over="ignore", invalid="ignore"):
        log_g = -3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (1.0 - a)
    bad = ~np.isfinite(log_g) | (np.abs(log_g) > _LOG_G_LIMIT)
    if np.any(bad):
        a_bad = float(np.asarray(a)[bad].flat[0])
        raise ValueError(
            "The dark-energy density ratio g(a) = rho_DE(a)/rho_DE(1) is "
            f"not representable in finite arithmetic near a={a_bad:.6g} "
            f"for w0={w0:.4g}, wa={wa:.4g}. This combination of the CPL "
            "parameters and the requested a-range is too extreme for "
            "this program's numerics; choose milder w0/wa or a narrower "
            "a_i-to-a_max range."
        )
    return np.exp(log_g)


# ----------------------------------------------------------------------
# The Friedmann equation, E(a)^2 = H(a)^2 / H0^2
# ----------------------------------------------------------------------
def E2(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """
    Dimensionless Friedmann equation (first Friedmann equation), exact
    for a homogeneous isotropic universe given this energy content:

        E(a)^2 = Om a^-3 + Or a^-4 + Ok a^-2 + Ode * g(a)

    where g(a) is the CPL dark-energy density shape above. Works on
    scalars or numpy arrays.
    """
    a = np.asarray(a, dtype=float)
    val = omega_m * a ** -3 + omega_r * a ** -4 + omega_k * a ** -2
    if omega_de != 0.0:
        val = val + omega_de * de_density_shape(a, w0, wa)
    return val


_E2_FLOOR = 1.0e-12  # below this, H is numerically zero and Omega_i(a) formally diverges


def omega_fractions(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """
    Fractional contribution of each component to the total energy
    density at scale factor a: Omega_i(a) = rho_i(a) / rho_crit(a).
    By construction Om(a)+Or(a)+Ok(a)+Ode(a) = 1 identically -- this is
    an algebraic identity, not something to fit or calibrate.

    Exactly at a turnaround, H(a) = 0 while the individual densities
    stay finite, so every Omega_i formally diverges there; that single
    point is reported as NaN rather than +/-inf.
    """
    a = np.asarray(a, dtype=float)
    e2 = E2(a, omega_m, omega_r, omega_k, omega_de, w0, wa)
    safe = np.where(e2 > _E2_FLOOR, e2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        om = omega_m * a ** -3 / safe
        orr = omega_r * a ** -4 / safe
        ok = omega_k * a ** -2 / safe
        ode = (omega_de * de_density_shape(a, w0, wa) / safe
               if omega_de != 0.0 else np.zeros_like(a))
    return om, orr, ok, ode


def deceleration_q(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """
    Deceleration parameter q(a) = -(a a_ddot)/a_dot^2, from the SECOND
    Friedmann equation (the acceleration equation). Only genuine stress
    -energy components enter this equation -- spatial curvature is a
    geometric bookkeeping term in the first Friedmann equation, not a
    fluid, and correctly plays no direct role here:

        q(a) = 1/2 * sum_i Omega_i(a) (1 + 3 w_i(a)),   i = matter, radiation, DE

    q > 0 is decelerating, q < 0 is accelerating. Sanity checks: matter
    only gives q=1/2 (matter domination); radiation only gives q=1;
    a cosmological constant alone gives q=-1 (de Sitter).
    """
    om, orr, _ok, ode = omega_fractions(a, omega_m, omega_r, omega_k, omega_de, w0, wa)
    wde = w_de(a, w0, wa)
    return 0.5 * om * (1.0 + 3.0 * MATTER_W) \
        + 0.5 * orr * (1.0 + 3.0 * RADIATION_W) \
        + 0.5 * ode * (1.0 + 3.0 * wde)


def redshift(a):
    return 1.0 / np.asarray(a, dtype=float) - 1.0


# ----------------------------------------------------------------------
# Early-time analytic offset (sets t=0 at the true Big Bang, a=0)
# ----------------------------------------------------------------------
def early_time_offset_Gyr(a_i, H0_pgyr, omega_m, omega_r, omega_k,
                           omega_de=0.0, w0=-1.0, wa=0.0):
    """
    Analytic cosmic time elapsed between the Big Bang (a=0) and a=a_i,
    used to anchor the numerical integration to a genuine t=0 at a=0
    rather than starting the clock arbitrarily at a=a_i.

    As a -> 0, E(a)^2 is a sum of power-law terms a^{-3(1+w_i)}:
    radiation (w=1/3, a^-4), matter (w=0, a^-3), curvature (w=-1/3,
    a^-2), and -- for an unusual choice of the CPL parameters -- dark
    energy itself, since g(a) -> a^{-3(1+w0+wa)} exp(-3 wa) as a -> 0,
    i.e. an effective constant equation of state w_eff = w0 + wa there.
    Whichever term is numerically LARGEST at a_i (a magnitude
    comparison of finite positive numbers, so it is safe even when the
    underlying Omega's differ by many orders of magnitude) is treated
    as the sole driver of the expansion at a_i, and the single-
    component Friedmann equation is solved exactly for that one term
    (tag: DERIVED, exact given single-component domination):

        E(a)^2 ~ Omega_eff a^{-3(1+w)}   (w > -1)
        =>  t(a) = a^{1.5(1+w)} / [1.5(1+w) H0 sqrt(Omega_eff)]

    which reproduces the textbook radiation (w=1/3), matter (w=0), and
    Milne-curvature (w=-1/3) results as special cases.

    Returns (t_i_Gyr, regime, dominant_magnitude). dominant_magnitude
    is the value of the winning term at a_i; dividing it by the true
    E(a_i)^2 (done by the caller) gives the diagnostic dominance
    fraction: if that fraction is not close to 1, a_i was not chosen
    deep enough for the analytic offset to be trustworthy.
    """
    terms = []
    if omega_r > 0.0:
        terms.append(("radiation-dominated", RADIATION_W, omega_r,
                       omega_r * a_i ** -4))
    if omega_m > 0.0:
        terms.append(("matter-dominated", MATTER_W, omega_m,
                       omega_m * a_i ** -3))
    if omega_k > 0.0:
        terms.append(("curvature-dominated (Milne)", -1.0 / 3.0, omega_k,
                       omega_k * a_i ** -2))
    if omega_de != 0.0:
        w_eff = w0 + wa
        if w_eff > -1.0:
            omega_eff = omega_de * math.exp(-3.0 * wa)
            if omega_eff > 0.0:
                terms.append((
                    "dark-energy-dominated at early times (unusual CPL "
                    "choice)", w_eff, omega_eff,
                    omega_eff * a_i ** (-3.0 * (1.0 + w_eff))))

    if not terms:
        # No matter, radiation, positive curvature, or early-time-
        # dominant dark energy at all: there is no standard early-time
        # limit. Fall back to t=0 at a=a_i itself and flag this clearly
        # rather than pretending to know the offset.
        return 0.0, ("undetermined (no matter, radiation, open curvature, "
                      "or early-time-dominant dark energy)"), None

    regime, w, omega_eff, magnitude = max(terms, key=lambda term: term[3])
    exponent = 1.5 * (1.0 + w)
    t_i = a_i ** exponent / (exponent * H0_pgyr * math.sqrt(omega_eff))
    return t_i, regime, magnitude


# ----------------------------------------------------------------------
# RK4 integration of da/dt = a H(a), with turnaround (recollapse) handling
# ----------------------------------------------------------------------
_TURNAROUND_TRIGGER_REL = 0.1  # see the proactive check in integrate_evolution


def _dE2_da(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """Analytic derivative of E(a)^2, used only to flag proximity to a
    turning point before RK4 gets there (see integrate_evolution). This
    is a diagnostic heuristic, not a value fed to the integrator itself,
    but it is still checked for finiteness explicitly (rather than
    relying only on de_density_shape's own domain check) because the
    pure-matter/radiation/curvature power-law terms below can in
    principle overflow at extremely small a even when omega_de is 0."""
    val = -3.0 * omega_m * a ** -4 - 4.0 * omega_r * a ** -5 - 2.0 * omega_k * a ** -3
    if omega_de != 0.0:
        g = de_density_shape(a, w0, wa)
        dlng_da = -3.0 * (1.0 + w0 + wa) / a + 3.0 * wa
        val += omega_de * g * dlng_da
    val = float(val)
    if not math.isfinite(val):
        raise ValueError(
            f"dE(a)^2/da is not finite at a={a:.6g} for these parameters; "
            "this model is not numerically representable there."
        )
    return val


def _rate(a, H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa):
    e2 = float(omega_m * a ** -3 + omega_r * a ** -4 + omega_k * a ** -2)
    if omega_de != 0.0:
        e2 += omega_de * float(de_density_shape(a, w0, wa))
    if not math.isfinite(e2):
        # NaN/inf must never be mistaken for "found a turning point": a
        # turning point is E(a)^2 < 0, a specific, finite, physically
        # meaningful condition, not the absence of a valid number.
        raise ValueError(
            f"E(a)^2 is not finite at a={a:.6g} for these parameters; "
            "this model is not numerically representable there."
        )
    if e2 < 0.0:
        raise _Overshoot()
    return a * H0_pgyr * math.sqrt(e2)


def _rk4_step(a, dt, H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa):
    args = (H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa)
    k1 = _rate(a, *args)
    k2 = _rate(a + 0.5 * dt * k1, *args)
    k3 = _rate(a + 0.5 * dt * k2, *args)
    k4 = _rate(a + dt * k3, *args)
    return a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _bisect_dt_for_a(a, target_a, dt_hi, args, n_iter=MAX_BISECTIONS):
    """
    Solve for the sub-step dt in [0, dt_hi] such that one RK4 step from
    a lands exactly on target_a, by bisection on dt itself (the RK4
    map a -> _rk4_step(a, dt, ...) is monotonically increasing in dt on
    an ordinary expanding-branch step, since da/dt = a H(a) > 0
    throughout). This replaces linear interpolation between the
    bracketing grid points -- which caps the effective accuracy of a
    checkpoint time at 2nd order regardless of RK4's 4th-order local
    accuracy -- with a sub-step that inherits RK4's full order.

    The caller must guarantee that dt_hi actually brackets target_a
    (i.e. _rk4_step(a, dt_hi, ...) >= target_a >= a): this is checked
    explicitly rather than assumed, because an unbracketed bisection can
    silently return a plausible-looking but wrong time (a real defect
    found by third-round review, in the near-turnaround checkpoint path
    that used to call this with an unverified dt_hi).
    """
    try:
        a_hi_check = _rk4_step(a, dt_hi, *args)
    except _Overshoot:
        a_hi_check = float("nan")
    if not math.isfinite(a_hi_check) or a_hi_check < target_a or target_a < a:
        raise ValueError(
            f"_bisect_dt_for_a: dt_hi={dt_hi:.6g} does not bracket "
            f"target_a={target_a:.6g} from a={a:.6g} (reaches "
            f"{a_hi_check!r} instead); this is an internal consistency "
            "error, not a user input problem."
        )
    lo, hi = 0.0, dt_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        a_mid = _rk4_step(a, mid, *args)
        if a_mid < target_a:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _bisect_root_a(a_lo, a_hi, args, n_iter=MAX_BISECTIONS):
    """
    Bisect for the root of E(a)^2=0 between a_lo and a_hi, given
    (checked by the caller) that E(a_lo)^2 >= 0 and E(a_hi)^2 < 0, both
    finite -- i.e. a genuine, already-bracketed sign change. Purely
    algebraic and independent of the RK4 trajectory; this matters
    because da/dt ~ sqrt(a_turn - a) has an infinite derivative at a
    turning point, so RK4 (or any fixed-order Taylor-series method)
    loses its formal order of accuracy in the last step before one,
    while bisecting the exact algebraic condition E(a)^2 = 0 does not.

    The bracket is always taken from within the caller's own requested
    a-range (never searched for by open-ended extrapolation), which
    keeps every evaluation inside the region the user actually asked
    about and avoids the extreme, easily-overflowing values of a that
    an unbounded search could otherwise wander into.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args

    def e2(aa):
        return float(E2(aa, omega_m, omega_r, omega_k, omega_de, w0, wa))

    lo, hi = a_lo, a_hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        val = e2(mid)
        if not math.isfinite(val):
            raise ValueError(
                f"E(a)^2 became non-finite at a={mid:.6g} while "
                "bisecting for a turning point; this parameter "
                "combination is too extreme for this program's "
                "numerics."
            )
        if val >= 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _first_root_ahead(a_lo, a_max, args, n_scan=2000):
    """
    Scan [a_lo, a_max] on a dense grid for the FIRST root of E(a)^2=0
    strictly ahead of a_lo, returning a bracketing pair (a_scan_lo,
    a_scan_hi) around it, or None if E(a)^2 stays non-negative and
    finite everywhere on the grid (including at a_max itself).

    This replaces a bug found by third-round review: checking only the
    SIGN OF E(a_max)^2 cannot detect a turning point that occurs and
    then reverses before a_max is reached (E(a)^2 can have more than
    one positive root -- e.g. a closed universe with matter and a small
    positive cosmological constant can have a forbidden interval
    bounded by two positive roots, with E(a_max)^2 positive again
    beyond it). The actual expanding-branch solution must stop at the
    FIRST root it reaches, not be waved through because some later
    point happens to test non-negative. A dense grid scan, like the one
    already used elsewhere in this module for equality/transition
    epochs, is the simplest robust way to find that first root; as with
    those searches, a turning point confined to an interval narrower
    than the grid spacing could in principle be missed, which is an
    inherent (and documented) limitation of a bounded-resolution scan,
    not of the underlying physics.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args
    grid = np.linspace(a_lo, a_max, n_scan)
    vals = np.asarray(E2(grid, omega_m, omega_r, omega_k, omega_de, w0, wa),
                       dtype=float)
    bad = ~np.isfinite(vals)
    if np.any(bad):
        idx = int(np.argmax(bad))
        raise ValueError(
            f"E(a)^2 is not finite at a={grid[idx]:.6g}; this parameter "
            "combination is too extreme for this program's numerics."
        )
    neg = vals < 0.0
    if not np.any(neg):
        return None
    idx = int(np.argmax(neg))  # first negative grid point; idx >= 1
    return float(grid[idx - 1]), float(grid[idx])


def _quad_time_between(a_lo, a_hi, args, n_nodes=60):
    """
    Cosmic time elapsed between two ORDINARY (non-singular) scale
    factors, t = Integral_{a_lo}^{a_hi} da / [a H0 E(a)], by plain
    Gauss-Legendre quadrature. Valid whenever E(a)^2 is positive and
    finite throughout [a_lo, a_hi] -- i.e. no turning point in between,
    which the caller must have already established (e.g. via
    _first_root_ahead returning None over that range). Unlike a single
    enlarged RK4 step, this is accurate and safe regardless of how
    close a_hi sits to a genuine turning point elsewhere outside
    [a_lo, a_hi], since it uses no ODE stepping at all.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args
    if a_hi <= a_lo:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    aa = 0.5 * (a_hi - a_lo) * nodes + 0.5 * (a_hi + a_lo)
    wq = 0.5 * (a_hi - a_lo) * weights
    e2 = np.asarray(E2(aa, omega_m, omega_r, omega_k, omega_de, w0, wa),
                     dtype=float)
    if not np.all(np.isfinite(e2)) or np.any(e2 <= 0.0):
        raise ValueError(
            "E(a)^2 is non-finite or non-positive while integrating the "
            f"cosmic time between a={a_lo:.6g} and a={a_hi:.6g}; this "
            "indicates an unexpected turning point in that range."
        )
    integrand = 1.0 / (aa * H0_pgyr * np.sqrt(e2))
    return float(np.sum(wq * integrand))


def _advance_to_boundary(a, t, dt, args, a_max, t_max_gyr):
    """
    Called when the current step has been flagged as approaching a
    turning point (or has already overshot one). Determines which of
    three mutually exclusive events actually comes first -- reaching
    a_max, reaching t_max, or a genuine turning point -- and returns
    (t_next, a_next, event, time_to). a_max and t_max, when given, are
    treated as hard stopping bounds: a turning point beyond either of
    them is never reported, exactly as an ordinary (non-turning-point)
    step would never be allowed to overshoot them.

    time_to is a callable, time_to(target_a) -> elapsed cosmic time
    from the current a to any target_a in [a, a_next], used by the
    caller to place checkpoints (such as a=1) that fall in this
    interval. Everything here is computed by quadrature, never by a
    single enlarged RK4 step: near a turning point da/dt ~
    sqrt(a_turn-a) has an infinite derivative, and a full-stride RK4
    step aimed at a target close to that singularity is exactly what
    caused the release-blocking failures found by third-round review
    (RuntimeError near/at an exact turnaround, silently wrong
    checkpoint times, and leaked internal exceptions near t_max).
    """
    def _bisect_for_time(time_to_fn, target_dt, lo, hi):
        # Solve time_to_fn(target_a) == target_dt for target_a in
        # [lo, hi], by bisection on the (monotonically increasing in
        # target_a) quadrature-based time function itself -- never by
        # RK4-stepping toward a target that may sit close to a
        # singularity.
        for _ in range(MAX_BISECTIONS):
            mid = 0.5 * (lo + hi)
            if time_to_fn(mid) < target_dt:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # Search a bit BEYOND a_max, not just up to it. This matters because
    # E(a)^2 can come arbitrarily close to zero at a_max without ever
    # testing negative there in floating point (e.g. a_max chosen to sit
    # exactly at, or a hair below, a genuine turning point: E(a_max)^2
    # rounds to 0.0 or a tiny positive residual either way). A scan that
    # only looked as far as a_max would then wrongly conclude "no
    # turning point ahead" and hand the endpoint to ordinary
    # (non-singularity-aware) quadrature, whose accuracy degrades the
    # closer its endpoint sits to a real turning point -- exactly the
    # release-blocking failure found by third-round review at and near
    # an exact a_max/a_turn tie. Extending the search past a_max, by a
    # margin tied to how far away a_max already is, reliably converts
    # any such near-boundary coincidence into an ordinary "root found"
    # case below, handled by the same singularity-removing quadrature
    # used for an interior turning point -- which remains fully accurate
    # for a target arbitrarily far from, at, or infinitesimally below
    # the anchor root, so there is no accuracy cost to preferring it
    # whenever any root is found nearby. It does not change how an
    # interior root (strictly before a_max, e.g. the two-positive-root
    # case) is found, since that is already detected by this same scan
    # without needing to look past a_max at all.
    search_margin = max(0.5 * (a_max - a), 0.1 * a_max, 1.0e-9)
    bracket = _first_root_ahead(a, a_max + search_margin, args)

    if bracket is None:
        # No turning point even in this generously extended search:
        # a_max sits comfortably away from any recollapse, so ordinary
        # quadrature to a_max is safe and accurate. a_max stops the run
        # unless --t_max is given and would be reached first (the
        # turning-point heuristic that routed us into this function can
        # fire well before a_max even when no genuine turning point
        # exists there at all, so t_max must still be checked here, not
        # assumed already handled by the ordinary per-step loop).
        time_to = lambda ta: _quad_time_between(a, ta, args)  # noqa: E731
        t_to_amax = time_to(a_max)
        if t_max_gyr is not None and t + t_to_amax > t_max_gyr:
            a_next = _bisect_for_time(time_to, t_max_gyr - t, a, a_max)
            return t_max_gyr, a_next, "t_max", time_to
        return t + t_to_amax, a_max, "a_max", time_to

    a_lo, a_hi = bracket
    a_turn = _bisect_root_a(a_lo, a_hi, args)
    t_to_turn = _quad_time_to_turnaround(a, a_turn, args)

    def time_to(target_a):
        # Anchored at the same a_turn via the existing singularity-
        # removing substitution, which stays accurate for a target
        # arbitrarily close to a_turn -- unlike RK4 stepping there.
        if target_a >= a_turn:
            return t_to_turn
        return t_to_turn - _quad_time_to_turnaround(target_a, a_turn, args)

    if a_max < a_turn:
        t_to_amax = time_to(a_max)
        if t_max_gyr is not None and t + t_to_amax > t_max_gyr:
            a_next = _bisect_for_time(time_to, t_max_gyr - t, a, a_max)
            return t_max_gyr, a_next, "t_max", time_to
        return t + t_to_amax, a_max, "a_max", time_to

    t_turn = t + t_to_turn
    if t_max_gyr is not None and t_turn > t_max_gyr:
        # The turning point is real and within a_max, but occurs later
        # than the requested t_max: t_max stops the run first, still on
        # the smooth pre-turnaround branch.
        a_next = _bisect_for_time(time_to, t_max_gyr - t, a, a_turn)
        return t_max_gyr, a_next, "t_max", time_to

    return t_turn, a_turn, "turnaround", time_to


def _quad_time_to_turnaround(a_lo, a_turn, args, n_nodes=60):
    """
    Cosmic time elapsed from (a_lo, with E(a_lo)^2 > 0) to the turning
    point a_turn where E(a_turn)^2 = 0:

        t_turn - t_lo = Integral_{a_lo}^{a_turn} da / [a H0 E(a)]

    The integrand has an integrable 1/sqrt(a_turn - a) singularity at
    the upper limit (E(a) -> 0 linearly in (a_turn - a) near a simple
    turning point, so 1/E(a) ~ 1/sqrt(a_turn - a)). Substituting
    a = a_turn - u^2 (da = -2 u du) turns it into a smooth, bounded
    function of u on [0, u0] with u0 = sqrt(a_turn - a_lo), which a
    fixed-order Gauss-Legendre rule integrates to near machine
    precision since Gauss-Legendre nodes never land exactly on the
    endpoints u=0 or u=u0.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args
    u0 = math.sqrt(max(a_turn - a_lo, 0.0))
    if u0 <= 0.0:
        return 0.0
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    u = 0.5 * u0 * (nodes + 1.0)
    wq = 0.5 * u0 * weights
    aa = a_turn - u ** 2
    e2 = E2(aa, omega_m, omega_r, omega_k, omega_de, w0, wa)
    if not np.all(np.isfinite(e2)):
        raise ValueError(
            "E(a)^2 is non-finite while integrating the singularity-"
            "removed quadrature for the turnaround time; this parameter "
            "combination is too extreme for this program's numerics."
        )
    # Every quadrature node has aa < a_turn strictly (Gauss-Legendre
    # nodes never land on the u=0 endpoint), so a substantively negative
    # E2 here -- as opposed to round-off noise right next to the root --
    # means E(a)^2 is not single-signed on (a_lo, a_turn) and a_turn is
    # not the FIRST root the trajectory would actually reach. That is a
    # genuine inconsistency, not something to paper over with a floor.
    if np.any(e2 < -1.0e-9):
        raise ValueError(
            "E(a)^2 is negative before the located turning point "
            f"(min = {float(e2.min()):.6g}); this indicates the model "
            "has more than one turning point in this range, which this "
            "program's quadrature does not support."
        )
    # Floor only round-off-scale non-positive noise, to a small POSITIVE
    # value rather than exactly 0.0 -- the integrand divides by sqrt(e2),
    # so an exact zero would produce a spurious inf/nan at that one
    # quadrature node even though the true value is a well-behaved small
    # positive number (aa never exactly equals a_turn; see above).
    e2 = np.where(e2 <= 0.0, 1.0e-14, e2)
    integrand = 2.0 * u / (aa * H0_pgyr * np.sqrt(e2))
    return float(np.sum(wq * integrand))


def integrate_evolution(H0=70.0, omega_m=0.30, omega_r=9.24e-5, omega_de=None,
                        w0=-1.0, wa=0.0,
                        a_i=1.0e-8, a_max=5.0, t_max_gyr=None,
                        step_frac=0.005, continue_collapse=False):
    """
    Integrate the Friedmann equation da/dt = a H(a) forward from a=a_i to
    a=a_max (or until t_max_gyr, or until the universe recollapses,
    whichever comes first), returning the full history plus a summary of
    derived milestones.

    step_frac sets the Hubble-time-scaled timestep dt = step_frac / H(a),
    i.e. the approximate fractional change in ln(a) per RK4 step. This
    is a physically motivated variable step size, not a true adaptive
    (embedded local-error-estimating) method: it has no independent
    error control, so accuracy should be checked by re-running at a
    smaller step_frac and confirming the results have converged.

    A turning point (H=0, where an expanding branch would recollapse)
    cannot be stepped through by RK4 at all, because da/dt ~
    sqrt(a_turn - a) has an infinite derivative there. When a step
    would push E(a)^2 negative, the turnaround scale factor a_turn is
    instead found by direct bisection on the algebraic condition
    E(a_turn)^2 = 0 (independent of the RK4 trajectory), and the
    cosmic time t_turn is found by Gauss-Legendre quadrature of the
    exact integral t = Integral da / [a H0 E(a)] after a substitution
    that removes its endpoint singularity -- see _advance_to_boundary,
    _bisect_root_a, and _quad_time_to_turnaround below. a_max and t_max,
    when given, are hard bounds: a genuine turning point beyond either
    of them is never reported -- the run instead stops cleanly at
    whichever of a_max, t_max, or the turning point comes first.
    """
    H0 = _finite("H0", H0)
    a_i = _finite("a_i", a_i)
    a_max = _finite("a_max", a_max)
    step_frac = _finite("step_frac", step_frac)
    omega_m = _finite("omega_m", omega_m)
    omega_r = _finite("omega_r", omega_r)
    if omega_m < 0.0:
        raise ValueError("omega_m must be non-negative.")
    if omega_r < 0.0:
        raise ValueError("omega_r must be non-negative.")
    if not (0.0 < a_i < 1.0):
        raise ValueError("a_i must lie strictly between 0 and 1.")
    if a_max <= a_i:
        raise ValueError("a_max must be greater than a_i.")
    if not (1.0e-6 <= step_frac <= 0.5):
        raise ValueError("step_frac must lie in [1e-6, 0.5].")
    if t_max_gyr is not None:
        t_max_gyr = _finite("t_max_gyr", t_max_gyr)
        if t_max_gyr <= 0:
            raise ValueError("t_max_gyr must be greater than zero.")

    est_steps = math.log(a_max / a_i) / step_frac
    if est_steps > MAX_STEPS:
        raise ValueError(
            f"This combination of a_i={a_i:.3g}, a_max={a_max:.3g}, and "
            f"step_frac={step_frac:.3g} would need on the order of "
            f"{est_steps:,.0f} RK4 steps, above the internal safety "
            f"limit of {MAX_STEPS:,}. Increase step_frac, raise a_i, or "
            "lower a_max."
        )

    omega_de, omega_k = resolve_omega_de(omega_m, omega_r, omega_de)
    w0 = _finite("w0", w0)
    wa = _finite("wa", wa)

    H0_pgyr = H0_per_Gyr(H0)

    e2_i = float(E2(a_i, omega_m, omega_r, omega_k, omega_de, w0, wa))
    if e2_i <= 0.0:
        raise ValueError(
            "E(a)^2 <= 0 at a_i: this parameter combination has no "
            "expanding branch there. Increase a_i or change the "
            "density parameters."
        )

    t_i, early_regime, dominance = early_time_offset_Gyr(
        a_i, H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa)

    if t_max_gyr is not None and t_max_gyr <= t_i:
        raise ValueError(
            f"t_max_gyr ({t_max_gyr:.6g} Gyr) must be greater than the "
            f"analytic Big-Bang time offset at a_i ({t_i:.6g} Gyr); "
            "choose a larger t_max_gyr or a smaller a_i."
        )

    warnings = []
    if dominance is not None:
        # Compare the winning term against the SUM OF ABSOLUTE VALUES of
        # every component at a_i, not against the signed total E(a_i)^2.
        # A signed ratio is cancellation-unsafe: e.g. a negative
        # omega_de component can partially cancel a positive matter
        # term, shrinking E(a_i)^2 itself and making
        # dominance/E(a_i)^2 exceed 1 even though the "dominant" term is
        # nowhere near the sole driver of the early-time expansion. The
        # sum of absolute values is never smaller than the true total,
        # so this check can only under-, never over-, state how
        # dominant the assumed term actually is.
        abs_total = (abs(omega_m) * a_i ** -3 + abs(omega_r) * a_i ** -4
                     + abs(omega_k) * a_i ** -2)
        if omega_de != 0.0:
            abs_total += abs(omega_de) * float(de_density_shape(a_i, w0, wa))
        frac_dominant = dominance / abs_total if abs_total > 0.0 else 0.0
        if frac_dominant < 0.999:
            warnings.append(
                f"a_i = {a_i:.3g} is not deep enough into the {early_regime} "
                f"regime for the analytic Big-Bang time offset to be "
                f"asymptotically accurate (the assumed-dominant term "
                f"supplies only {frac_dominant * 100:.2f}% of the total "
                f"early-time density budget); reduce --a_i for a cleaner "
                "age estimate."
            )
    else:
        warnings.append(
            "No matter, radiation, or open curvature in this model: the "
            "analytic Big-Bang time offset could not be computed, and the "
            "clock was started at t=0 at a=a_i instead of at the true a=0 "
            "singularity. Ages quoted below are lower bounds."
        )

    checkpoints = sorted(c for c in (1.0,) if a_i < c < a_max)
    checkpoint_at_a_max = (a_max == 1.0)
    checkpoint_times = {}

    t_list = [t_i]
    a_list = [a_i]
    t = t_i
    a = a_i
    turnaround = None
    stop_reason = None
    steps = 0

    while a < a_max:
        steps += 1
        if steps > MAX_STEPS:
            raise RuntimeError(
                "Integration exceeded the internal step safety limit; "
                "increase --step_frac (a larger value takes fewer, "
                "bigger steps) or reduce --a_max."
            )
        e2_now = float(E2(a, omega_m, omega_r, omega_k, omega_de, w0, wa))
        args = (H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa)

        # Proactively hand off to the exact algebraic/quadrature method
        # BEFORE RK4 takes a step anywhere near a turning point: as H(a)
        # falls toward zero there, the "Hubble-time-scaled" dt = step_frac
        # / H(a) grows large exactly where da/dt ~ sqrt(a_turn - a) is
        # least smooth, which otherwise lets RK4 accumulate its worst
        # error in precisely the last few steps. A cheap linear (Newton)
        # estimate of the distance to the nearest root of E(a)^2, dE2_da
        # < 0, flags this well before it matters: in every ordinary
        # single-component-dominated regime (matter, radiation, open
        # curvature, or a non-recollapsing dark-energy history) this
        # estimated distance is a fixed, large fraction of a itself, so
        # the check below only fires close to a genuine turning point.
        dE2 = _dE2_da(a, omega_m, omega_r, omega_k, omega_de, w0, wa)
        near_turnaround = False
        if dE2 < 0.0:
            delta_a_est = e2_now / (-dE2)
            if delta_a_est < _TURNAROUND_TRIGGER_REL * a:
                near_turnaround = True

        H_now = H0_pgyr * math.sqrt(e2_now)
        dt = step_frac / H_now
        if t_max_gyr is not None and t + dt > t_max_gyr:
            dt = t_max_gyr - t

        overshoot = False
        if not near_turnaround:
            try:
                a_next = _rk4_step(a, dt, *args)
                if not math.isfinite(a_next) or a_next <= 0.0:
                    overshoot = True
                elif float(E2(a_next, *args[1:])) < 0.0:
                    overshoot = True
            except _Overshoot:
                overshoot = True

        if near_turnaround or overshoot:
            # Determine which of THREE mutually exclusive events actually
            # comes first -- reaching a_max, reaching t_max, or a genuine
            # turning point -- treating a_max/t_max as hard bounds that a
            # turning point can never override (this is the fix for the
            # bug where a requested --a_max/--t_max was silently ignored
            # once the trajectory neared a genuine recollapse). The
            # turning point itself, when it IS what stops the run, is
            # still located algebraically (independent of the RK4
            # trajectory, which cannot resolve the sqrt-singularity in
            # da/dt right at a turning point) and its time found by
            # singularity-removing quadrature. The proactive branch above
            # should catch this in ordinary use; the reactive E(a)^2 < 0
            # check above remains as a safety net for very coarse
            # step_frac choices that could jump past the proactive
            # trigger zone in a single stride.
            t_next, a_next, event, time_to = _advance_to_boundary(
                a, t, dt, args, a_max, t_max_gyr)
            for c in list(checkpoints):
                if a < c <= a_next:
                    # Use the same quadrature this function already used
                    # to resolve the boundary itself, NOT an RK4 sub-step
                    # bisection: a fixed small dt from before the
                    # turning-point heuristic fired is not guaranteed to
                    # even reach a checkpoint like a=1 in one stride here,
                    # and bisecting an unverified bracket was a real bug
                    # (found by third-round review) that silently returned
                    # a plausible-looking but wrong checkpoint time.
                    t_c = t + time_to(c)
                    checkpoint_times[c] = t_c
                    t_list.append(t_c)
                    a_list.append(c)
                    checkpoints.remove(c)
            t_list.append(t_next)
            a_list.append(a_next)
            t, a = t_next, a_next
            stop_reason = event
            if event == "turnaround":
                turnaround = dict(a_turn=a_next, t_turn_gyr=t_next)
            elif event == "a_max" and checkpoint_at_a_max:
                checkpoint_times.setdefault(1.0, t_next)
            break

        t_next = t + dt
        for c in list(checkpoints):
            if a < c <= a_next:
                t_c = t + _bisect_dt_for_a(a, c, dt, args)
                checkpoint_times[c] = t_c
                t_list.append(t_c)
                a_list.append(c)
                checkpoints.remove(c)

        if a_next >= a_max:
            t_final = t + _bisect_dt_for_a(a, a_max, dt, args)
            t_list.append(t_final)
            a_list.append(a_max)
            t, a = t_final, a_max
            if checkpoint_at_a_max:
                checkpoint_times.setdefault(1.0, t_final)
            stop_reason = "a_max"
            break
        if t_max_gyr is not None and t_next >= t_max_gyr:
            t_list.append(t_next)
            a_list.append(a_next)
            t, a = t_next, a_next
            stop_reason = "t_max"
            break

        t, a = t_next, a_next
        t_list.append(t)
        a_list.append(a)
    else:
        stop_reason = "a_max"

    a_arr = np.asarray(a_list, dtype=float)
    t_arr = np.asarray(t_list, dtype=float)
    n_forward = a_arr.size

    total_lifetime_gyr = None
    if turnaround is not None and continue_collapse:
        # Exact time-reversal symmetry of the Friedmann equation: E(a)^2
        # depends on a alone, never on the sign of a_dot, so the
        # contracting branch is the mirror image of the expanding one
        # about the turnaround. No re-integration is needed.
        t_turn = turnaround["t_turn_gyr"]
        t_mirror = 2.0 * t_turn - t_arr[::-1][1:]
        a_mirror = a_arr[::-1][1:]
        t_arr = np.concatenate([t_arr, t_mirror])
        a_arr = np.concatenate([a_arr, a_mirror])
        total_lifetime_gyr = 2.0 * t_turn

    # H = a_dot / a is positive on the expanding branch and negative on
    # the mirrored contracting branch; E(a)^2 = (H/H0)^2 cannot carry
    # that sign on its own, so it must be reattached explicitly here.
    h_sign = np.ones_like(a_arr)
    if turnaround is not None and continue_collapse:
        h_sign[n_forward:] = -1.0

    z_arr = redshift(a_arr)
    e2_arr = E2(a_arr, omega_m, omega_r, omega_k, omega_de, w0, wa)
    # Every 'a' in a_arr is either an ordinary point on the integrated
    # trajectory (E(a)^2 > 0 there, guaranteed by the RK4/overshoot
    # machinery above) or the algebraically-located turning point itself
    # (E(a)^2 = 0 exactly, up to floating-point round-off). So the only
    # negative values E2 should ever produce here are round-off-scale
    # noise at a known root -- never a substantively negative value that
    # would indicate a real bug. Flooring is restricted to that
    # round-off band; anything larger or non-finite is a genuine
    # inconsistency and must be raised, not silently hidden.
    if not np.all(np.isfinite(e2_arr)):
        raise ValueError(
            "E(a)^2 is non-finite at one or more points on the computed "
            "trajectory; this indicates a numerical inconsistency and "
            "the result cannot be trusted."
        )
    _round_off_floor = -1.0e-9
    if np.any(e2_arr < _round_off_floor):
        raise ValueError(
            f"E(a)^2 is substantively negative (min = {e2_arr.min():.6g}) "
            "at a point that should lie on the expanding branch; this "
            "indicates a numerical inconsistency and the result cannot "
            "be trusted."
        )
    e2_arr = np.clip(e2_arr, 0.0, None)
    H_pgyr_arr = h_sign * H0_pgyr * np.sqrt(e2_arr)
    H_kms_mpc_arr = H_pgyr_arr * KMS_MPC_PER_INV_GYR
    om_arr, or_arr, ok_arr, ode_arr = omega_fractions(
        a_arr, omega_m, omega_r, omega_k, omega_de, w0, wa)
    q_arr = deceleration_q(a_arr, omega_m, omega_r, omega_k, omega_de, w0, wa)
    wde_arr = w_de(a_arr, w0, wa)

    age_today_gyr = checkpoint_times.get(1.0)
    if age_today_gyr is None and a_max <= 1.0 and turnaround is None:
        warnings.append(
            "a_max <= 1: the run stopped before reaching the present "
            "scale factor a=1, so no age-today is available."
        )
    if age_today_gyr is None and a_max > 1.0 and turnaround is None \
            and stop_reason == "t_max":
        warnings.append(
            "--t_max stopped this run before it reached a=1, so no "
            "age-today is available even though a_max > 1; increase "
            "--t_max or drop it if you need the present-day epoch."
        )

    if wa != 0.0 and a_max > 1.0:
        warnings.append(
            "wa != 0: the w0-only phantom/quintessence/'no acceleration' "
            "classification printed above describes the dark-energy "
            "component only AT a=1 (today). With wa != 0, w(a) evolves, "
            "so this model can cross between those regimes at other "
            "epochs -- check w_de(a) at the a of interest, not just w0. "
            "CPL is an observational parametrization centered near the "
            "measured redshift range; results for a>1 are extrapolations, "
            "and w(a) = w0 + wa(1-a) diverges without bound as a -> "
            "infinity, so this model's far-future behavior should not be "
            "over-interpreted."
        )
    if turnaround is not None and turnaround["a_turn"] < 1.0:
        warnings.append(
            "This model recollapses at a_turn = "
            f"{turnaround['a_turn']:.4g} < 1: it would already have "
            "recollapsed before reaching its present size. It is not a "
            "candidate for our own universe, but it is a legitimate "
            "closed-FRW toy model."
        )

    a_eq_rm = None
    if omega_m > 0.0 and omega_r > 0.0:
        a_eq_rm = omega_r / omega_m

    # Search only out to the scale factor the run actually reached (a),
    # not the requested a_max: if the run stopped early at a genuine
    # turnaround or at --t_max, a milestone reported between the reached
    # 'a' and a_max would describe a point the trajectory never got to.
    a_reached = min(a, a_max)
    a_eq_mde = _find_sign_change(
        lambda aa: (omega_fractions(aa, omega_m, omega_r, omega_k, omega_de, w0, wa)[0]
                    - omega_fractions(aa, omega_m, omega_r, omega_k, omega_de, w0, wa)[3]),
        1.0e-6, a_reached)

    a_accel = _find_sign_change(
        lambda aa: deceleration_q(aa, omega_m, omega_r, omega_k, omega_de, w0, wa),
        1.0e-6, a_reached)

    big_rip_gyr = None
    if wa == 0.0 and w0 < -1.0 and omega_de > 0.0 and turnaround is None and age_today_gyr is not None:
        m = -1.5 * (1.0 + w0)
        big_rip_gyr = age_today_gyr + 1.0 / (m * H0_pgyr * math.sqrt(omega_de))
        warnings.append(
            "w0 < -1 (phantom dark energy): the DE density grows without "
            "bound, driving a's expansion rate to infinity at a finite "
            "future time (a 'Big Rip'). The quoted t_rip is an analytic "
            "estimate that neglects the (by then negligible) matter and "
            "radiation contributions and assumes wa=0."
        )

    if abs(omega_k) > 0.05:
        warnings.append(
            f"|Omega_k0| = {abs(omega_k):.3g} is far outside the "
            "observational bound (|Omega_k0| below about 0.001 from CMB "
            "and BAO data); this is a legitimate but strongly non-flat "
            "toy universe, not a candidate for the real one."
        )
    if omega_de != 0.0 and (1.0 + 3.0 * w0) >= 0.0:
        warnings.append(
            "1 + 3*w0 >= 0 for this dark-energy component: it does not "
            "accelerate the expansion on its own, so 'dark energy' is a "
            "misnomer for this parameter choice; it behaves like an "
            "additional decelerating fluid."
        )

    summary = dict(
        H0_kms_mpc=H0, omega_m=omega_m, omega_r=omega_r, omega_k=omega_k,
        omega_de=omega_de, w0=w0, wa=wa, a_i=a_i, a_max=a_max,
        step_frac=step_frac, early_regime=early_regime,
        age_today_gyr=age_today_gyr,
        H0_inv_gyr=1.0 / H0_pgyr,
        a_eq_rm=a_eq_rm, z_eq_rm=(redshift(a_eq_rm) if a_eq_rm else None),
        a_eq_mde=a_eq_mde, z_eq_mde=(redshift(a_eq_mde) if a_eq_mde else None),
        a_accel=a_accel, z_accel=(redshift(a_accel) if a_accel else None),
        q0=float(deceleration_q(1.0, omega_m, omega_r, omega_k, omega_de, w0, wa)),
        turnaround=turnaround,
        stop_reason=stop_reason,
        total_lifetime_gyr=total_lifetime_gyr,
        big_rip_gyr=big_rip_gyr,
        n_steps=len(a_arr),
        model_version=MODEL_VERSION,
        warnings=warnings,
    )

    return dict(
        t_gyr=t_arr, a=a_arr, z=z_arr, H_kms_mpc=H_kms_mpc_arr,
        Om=om_arr, Or=or_arr, Ok=ok_arr, Ode=ode_arr,
        q=q_arr, w_de=wde_arr,
        summary=summary,
    )


def _find_sign_change(func, a_lo, a_hi, n_scan=800):
    """
    Generic bisection root finder used for the matter/dark-energy
    equality epoch and the deceleration/acceleration transition. Scans
    a log-spaced grid for the first sign change of func(a), then
    bisects it to high precision. Returns None if no crossing is found.
    """
    a_lo = max(a_lo, 1.0e-8)
    if a_hi <= a_lo:
        return None
    grid = np.geomspace(a_lo, a_hi, n_scan)
    vals = np.array([func(x) for x in grid])
    finite = np.isfinite(vals)
    if not np.any(finite):
        return None
    sign = np.sign(vals)
    idx = np.where(np.diff(sign[finite]) != 0)[0]
    if idx.size == 0:
        return None
    grid_f = grid[finite]
    i = idx[0]
    lo, hi = grid_f[i], grid_f[i + 1]
    f_lo = func(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ----------------------------------------------------------------------
# Named preset cosmologies for --mode compare
# ----------------------------------------------------------------------
PRESETS = {
    "EdS": dict(
        H0=70.0, omega_m=1.0, omega_r=0.0, omega_de=0.0, w0=-1.0, wa=0.0,
        label="Einstein-de Sitter (flat, matter only)"),
    "radiation": dict(
        H0=70.0, omega_m=0.0, omega_r=1.0, omega_de=0.0, w0=-1.0, wa=0.0,
        label="Radiation only (flat)"),
    "milne": dict(
        H0=70.0, omega_m=0.0, omega_r=0.0, omega_de=0.0, w0=-1.0, wa=0.0,
        label="Milne (empty, pure curvature) -- exact a=H0 t"),
    "lambdaCDM": dict(
        H0=67.4, omega_m=0.315, omega_r=9.24e-5, omega_de=1.0 - 0.315 - 9.24e-5,
        w0=-1.0, wa=0.0, label="Flat LCDM, Planck-2018-like concordance"),
    "open": dict(
        H0=70.0, omega_m=0.3, omega_r=9.24e-5, omega_de=0.0, w0=-1.0, wa=0.0,
        label="Open: matter + curvature only, no dark energy"),
    "closed": dict(
        H0=70.0, omega_m=1.5, omega_r=9.24e-5, omega_de=0.0, w0=-1.0, wa=0.0,
        label="Closed matter-dominated: recollapses at a_turn=3"),
    "phantom": dict(
        H0=67.4, omega_m=0.315, omega_r=9.24e-5,
        omega_de=1.0 - 0.315 - 9.24e-5,
        w0=-1.2, wa=0.0, label="Phantom dark energy, w0=-1.2 (Big Rip)"),
    "evolving_de": dict(
        H0=67.4, omega_m=0.315, omega_r=9.24e-5,
        omega_de=1.0 - 0.315 - 9.24e-5,
        w0=-0.8, wa=0.3,
        label="Evolving CPL dark energy, w0=-0.8, wa=0.3 -- NOT "
              "quintessence: w(a) crosses -1 at a=5/3 and turns "
              "phantom thereafter"),
    # Illustrative only -- NOT a fit to any survey's data or covariance.
    # w0 > -1 and wa < 0 is merely the SIGN quadrant that recent DESI DR2
    # BAO analyses (arXiv:2503.14738, combined with CMB and supernova
    # data) favor over a pure cosmological constant, at a few-sigma
    # significance that depends on the supernova compilation used; the
    # round numbers below are chosen only to sit inside that quadrant
    # and are not DESI's quoted best-fit values, which the reader should
    # get from the primary source if an actual comparison is needed.
    "desi_like": dict(
        H0=67.4, omega_m=0.315, omega_r=9.24e-5,
        omega_de=1.0 - 0.315 - 9.24e-5,
        w0=-0.7, wa=-1.0,
        label="Illustrative only: w0=-0.7, wa=-1.0, in the same w0>-1, "
              "wa<0 quadrant DESI DR2 (2025) favors over LCDM -- NOT a "
              "fit to DESI's data"),
}
