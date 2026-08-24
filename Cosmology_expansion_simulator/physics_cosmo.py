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
quadrature) is not exact either. Its accuracy has been validated on the
benchmark cases listed in the test suite; users applying it to other
models should check convergence themselves, e.g. by re-running at a
smaller step_frac (see main.py and the help file for details).

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

MODEL_VERSION = "1.3.0"

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
            f"outside this program's supported policy range near "
            f"a={a_bad:.6g} for w0={w0:.4g}, wa={wa:.4g} (a deliberately "
            "conservative cutoff, well inside where float64 itself would "
            "actually overflow -- see the |ln g(a)| policy threshold "
            "above). This combination of the CPL parameters and the "
            "requested a-range is too extreme for this program's "
            "numerics; choose milder w0/wa or a narrower a_i-to-a_max "
            "range."
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


def omega_fractions(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """
    Fractional contribution of each component to the total energy
    density at scale factor a: Omega_i(a) = rho_i(a) / rho_crit(a).
    By construction Om(a)+Or(a)+Ok(a)+Ode(a) = 1 identically -- this is
    an algebraic identity, not something to fit or calibrate.

    Exactly at a turnaround, H(a) = 0 while the individual densities
    stay finite, so every Omega_i formally diverges there; that single
    point (E(a)^2 <= 0, up to the round-off floor already applied
    upstream) is reported as NaN rather than +/-inf. An earlier version
    of this function additionally treated any E(a)^2 below a fixed
    absolute floor (1e-12) as undefined, which is WRONG for a point
    that is genuinely on the expanding branch with a small but positive
    E(a)^2 -- e.g. a loitering model's near-double-root dip, where
    E(a)^2 can be many orders of magnitude smaller than 1 without ever
    reaching zero. Such a point has large but perfectly finite,
    physically meaningful Omega_i values (the near-divergence IS the
    physics there), and masking it to NaN discarded real data under a
    label ("only NaN at a turnaround") that was not actually true (see
    Codex Audit 6 P2-1). The only value masked to NaN now is one where
    E(a)^2 itself is non-positive.
    """
    a = np.asarray(a, dtype=float)
    e2 = E2(a, omega_m, omega_r, omega_k, omega_de, w0, wa)
    safe = np.where(e2 > 0.0, e2, np.nan)
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


def _cancellation_safe_H_scale(a, omega_m, omega_r, omega_k, omega_de, w0, wa):
    """
    A reference expansion-rate scale, in the same units as H0_pgyr *
    sqrt(E(a)^2), that stays well-behaved even where the SIGNED E(a)^2
    happens to be anomalously small purely from near-cancellation among
    its terms -- e.g. a "loitering" model where matter/curvature and a
    small positive dark-energy term nearly balance, so E(a)^2 dips very
    close to zero and rebounds without ever actually crossing it. Built
    from the SUM OF ABSOLUTE VALUES of each component's contribution
    (the same cancellation-safety idea already used by the early-time
    dominance diagnostic elsewhere in this module), so it is never
    small unless every individual component's density is itself small
    -- a genuinely different, unrelated regime.
    """
    abs_e2 = (abs(omega_m) * a ** -3 + abs(omega_r) * a ** -4
              + abs(omega_k) * a ** -2)
    if omega_de != 0.0:
        abs_e2 += abs(omega_de) * float(de_density_shape(a, w0, wa))
    return math.sqrt(max(abs_e2, 1.0e-300))


def _safe_dt(a, args, step_frac):
    """
    A Hubble-time-scaled step size, dt = step_frac / H_scale(a), using
    the cancellation-safe reference scale above rather than the true
    (signed) H(a) itself. Using the signed H(a) can give a
    pathologically LARGE dt exactly where a "loitering" universe's
    E(a)^2 dips close to zero without a genuine turning point there:
    the ordinary dt = step_frac / H(a) would then take one enormous
    RK4 stride across the entire dip, discarding trajectory resolution
    through it, once the proactive near-turnaround check correctly
    identifies such a dip as a false alarm and hands integration back
    to ordinary RK4 stepping. This keeps that step size sane by
    construction, everywhere.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args
    H_scale = H0_pgyr * _cancellation_safe_H_scale(
        a, omega_m, omega_r, omega_k, omega_de, w0, wa)
    return step_frac / H_scale


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
    a lands on target_a to within numerical (bisection) tolerance, by
    bisection on dt itself (the RK4 map a -> _rk4_step(a, dt, ...) is
    monotonically increasing in dt on an ordinary expanding-branch
    step, since da/dt = a H(a) > 0 throughout). This replaces linear
    interpolation between the bracketing grid points -- which caps the
    effective accuracy of a checkpoint time at 2nd order regardless of
    RK4's 4th-order local accuracy -- with a sub-step that inherits
    RK4's full order.

    The caller must guarantee that dt_hi actually brackets target_a
    (i.e. _rk4_step(a, dt_hi, ...) >= target_a >= a): this is checked
    explicitly rather than assumed, because an unbracketed bisection can
    silently return a plausible-looking but wrong time.
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


def _de_extremum_a(n, wa):
    """The scale factor at which g(a) = de_density_shape(a, w0, wa) has
    its one possible interior extremum for a>0 (dg/da=0 there), or None
    if wa=0 (g is then a pure power law, monotonic, with no interior
    extremum) or if that point is non-positive. dg/da = g(a)*(n/a+3wa),
    and n/a+3wa is itself strictly monotonic in a (its own derivative,
    -n/a^2, is single-signed), so it can cross zero at most once for
    a>0 -- meaning g(a) is provably unimodal (monotonic, or a single
    hump) there, never more wiggly than that, regardless of wa."""
    if wa == 0.0:
        return None
    a_crit = -n / (3.0 * wa)
    return a_crit if a_crit > 0.0 else None


def _e2_derivative_lipschitz_bound(a_lo, a_hi, omega_m, omega_r, omega_k,
                                    omega_de, w0, wa):
    """
    An analytic upper bound L, in EXACT real arithmetic, on
    |dE(a)^2/da| for every a in [a_lo,a_hi], derived in closed form --
    not by sampling E(a)^2 or its derivative on a grid, which cannot
    rule out a narrow forbidden interval between two nearby roots
    falling entirely between sample points, however fine the grid.
    Given such an L, E(a)^2 cannot change by more than L*(a_hi-a_lo)
    anywhere in the interval relative to either endpoint (mean value
    theorem), so if min(E2(a_lo), E2(a_hi)) - L*(a_hi-a_lo) > 0, E(a)^2
    is positive throughout [a_lo,a_hi] in exact arithmetic. This
    function's float64 evaluation of that exact-arithmetic expression
    is then conservatively inflated (see _CERT_L_INFLATE_REL at the
    call site) to absorb its own ordinary (non-outward-rounded)
    rounding error -- a documented, conservative ENGINEERING margin,
    not a formal directed-rounding or interval-arithmetic guarantee.
    The combination is validated on the adversarial and randomized
    cases in the test suite, which is evidence of correctness on those
    cases, not a proof for every possible input.

    The matter, radiation, and curvature terms of dE2/da are each a
    single power of a (e.g. -3*Om*a^-4), and a pure power function has
    no interior extrema for a>0 -- its magnitude is maximized at one of
    the interval's two endpoints, so bounding each term needs no
    sampling at all. The CPL dark-energy term is the product of g(a)
    (which _de_extremum_a shows is unimodal, so its own maximum over
    [a_lo,a_hi] is at an endpoint or at its single interior extremum,
    all checkable directly) and a strictly monotonic factor (n/a+3wa,
    likewise maximized at an endpoint); the product of their two
    separately-bounded maxima is a valid, if not perfectly tight,
    bound on the term's own maximum magnitude.
    """
    L = 0.0
    if omega_m != 0.0:
        L += 3.0 * abs(omega_m) * max(a_lo ** -4, a_hi ** -4)
    if omega_r != 0.0:
        L += 4.0 * abs(omega_r) * max(a_lo ** -5, a_hi ** -5)
    if omega_k != 0.0:
        L += 2.0 * abs(omega_k) * max(a_lo ** -3, a_hi ** -3)
    if omega_de != 0.0:
        n = -3.0 * (1.0 + w0 + wa)
        h_lo = n / a_lo + 3.0 * wa
        h_hi = n / a_hi + 3.0 * wa
        a_candidates = [a_lo, a_hi]
        a_crit = _de_extremum_a(n, wa)
        if a_crit is not None and a_lo < a_crit < a_hi:
            a_candidates.append(a_crit)
        g_max = max(float(de_density_shape(aa, w0, wa)) for aa in a_candidates)
        L += abs(omega_de) * g_max * max(abs(h_lo), abs(h_hi))
    return L


_CERT_MAX_DEPTH = 60          # a range this many bisections deep is well
                               # below any float64-representable width
_CERT_MAX_EVALUATIONS = 200_000  # generous hard cap; see _first_root_ahead
_CERT_L_INFLATE_REL = 1.0e-9  # conservative RELATIVE inflation applied to
                               # the analytic Lipschitz bound before it is
                               # used, so that ordinary float64 rounding
                               # in computing L itself (which is evaluated
                               # with plain arithmetic, not outward-
                               # rounded interval arithmetic) cannot make
                               # a rounded-down L understate the true
                               # bound. This is a generous, documented
                               # margin, not a formal directed-rounding
                               # proof -- see _first_root_ahead.
_CERT_POSITIVE_MARGIN_REL = 1.0e-9  # a sub-interval is accepted as
                               # certified positive only when its
                               # computed lower bound B exceeds this
                               # SMALL POSITIVE fraction of the
                               # interval's own E(a)^2 scale -- never a
                               # negative or zero threshold. Requiring a
                               # positive margin (rather than merely
                               # B > 0, let alone B > a negative slack)
                               # is what keeps this conservative under
                               # ordinary floating-point rounding.


class _IndeterminateRootSearch(RuntimeError):
    """
    Raised when the certified search can neither find a root nor
    certify E(a)^2 positive throughout some sub-interval, whether
    because its evaluation budget was exhausted or because the
    sub-interval has been bisected down to float64's own subdivision
    limit without resolving either way. This is a deliberate, LOUD
    failure: an interval this search could not resolve is never
    silently reported as root-free. It subclasses RuntimeError so
    existing callers that catch RuntimeError still see it, while
    letting tests and callers that need to distinguish it do so by
    name.
    """


def _first_root_ahead(a_lo, a_max, args):
    """
    Find the FIRST root of E(a)^2=0 strictly ahead of a_lo, within
    [a_lo, a_max], returning a bracketing pair; return None once
    E(a)^2 is certified positive and finite throughout the whole
    range; or raise _IndeterminateRootSearch if neither can be
    established. This is a genuinely three-way (tri-state) result --
    root found, certified root-free, or unresolved -- and no caller in
    this module ever treats "unresolved" as "root-free": the two are
    kept strictly distinct, and an indeterminate result propagates
    upward as a loud failure rather than a silent "no root ahead."

    An earlier version of this scanned a fixed number of evenly spaced
    grid points, which cannot detect a turning point (or, worse, a
    forbidden interval bounded by two very close-together roots) that
    happens to fall entirely between two adjacent sample points --
    however fine the grid, two positive roots of E(a)^2 can always be
    tuned closer together than its spacing. Checking only the sign of
    E(a_max)^2 has the same blind spot for the same reason.

    This replaces that scan with recursive bisection guarded by an
    analytic per-interval bound (_e2_derivative_lipschitz_bound): a
    sub-interval is only accepted as root-free when a closed-form
    bound on how far E(a)^2 could possibly move across it -- derived
    from its exact analytic derivative, never from sampling -- shows
    it cannot reach zero anywhere inside, WITH a strictly positive
    safety margin (never a negative or zero one) to absorb ordinary
    floating-point rounding in computing that bound itself. Wherever
    that is not immediately established, the interval is bisected and
    the left half is fully resolved before the right half is even
    examined (so the FIRST root is always what gets returned),
    recursing as deep as necessary -- all the way to float64's own
    subdivision limit if that is genuinely what it takes, at which
    point an unresolved interval raises rather than being guessed
    safe. This is a conservative analytic-bound search validated by
    adversarial and randomized tests (see the test suite), not a
    formal interval-arithmetic proof: the bound L is computed with
    ordinary (not outward-rounded) float64 arithmetic, inflated by a
    documented conservative margin rather than a directed-rounding
    guarantee.

    Raises _IndeterminateRootSearch if a well-behaved certificate
    cannot be reached within a generous evaluation budget, or if the
    search reaches float64's own subdivision limit without resolving
    either way -- both are loud, honest failures for a pathological
    parameter combination, never a silent guess. Raises ValueError if
    E(a)^2 itself is ever found non-finite.
    """
    H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa = args
    budget = [_CERT_MAX_EVALUATIONS]

    def e2(aa):
        return float(E2(aa, omega_m, omega_r, omega_k, omega_de, w0, wa))

    def scan(lo, hi, e2_lo, e2_hi, depth):
        if not (math.isfinite(e2_lo) and math.isfinite(e2_hi)):
            raise ValueError(
                f"E(a)^2 is not finite between a={lo:.6g} and a={hi:.6g}; "
                "this parameter combination is too extreme for this "
                "program's numerics."
            )
        # By construction, e2_lo passed in here is always already known
        # non-negative (the top-level caller established it; internally,
        # e2_lo is always either that value or a previously-computed
        # midpoint that this same check already passed). So a directly
        # OBSERVED negative e2_hi is immediate, sufficient proof of a
        # sign change in [lo, hi] -- return that bracket right away
        # rather than continuing to recurse for a "prove positive
        # everywhere" certificate that a value we've already measured
        # negative can never satisfy. Without this short-circuit, once
        # a genuine (or near-degenerate) dip straddles a bisection
        # midpoint closely enough, BOTH halves can keep failing to
        # certify at every subsequent level, recursing exponentially in
        # depth for no benefit -- this keeps refinement linear instead.
        if e2_hi < 0.0:
            return lo, hi
        L_raw = _e2_derivative_lipschitz_bound(lo, hi, omega_m, omega_r,
                                                omega_k, omega_de, w0, wa)
        L = L_raw * (1.0 + _CERT_L_INFLATE_REL) + 1.0e-300
        width = hi - lo
        scale = max(abs(e2_lo), abs(e2_hi), 1.0e-300)
        B = min(e2_lo, e2_hi) - L * width
        if B > _CERT_POSITIVE_MARGIN_REL * scale:
            return None  # certified: E(a)^2 > 0 everywhere in [lo, hi]
        budget[0] -= 1
        if budget[0] <= 0:
            raise _IndeterminateRootSearch(
                f"Could not certify that E(a)^2 stays positive, nor find "
                f"a root, between a={lo:.6g} and a={hi:.6g} within this "
                "program's search budget (minimum E(a)^2 observed at "
                f"either endpoint so far: {min(e2_lo, e2_hi):.6g}); this "
                "model's parameter combination may sit pathologically "
                "close to a hidden recollapse. Try a narrower --a_max or "
                "different parameters."
            )
        mid = 0.5 * (lo + hi)
        if depth >= _CERT_MAX_DEPTH or mid <= lo or mid >= hi:
            # This sub-interval cannot be certified positive AND has
            # been bisected down to (or past) float64's own inability
            # to represent a strictly-between midpoint at all -- mid<=lo
            # or mid>=hi means the interval is already narrower than
            # one ULP at this scale, so no further bisection is even
            # possible in this implementation's float64 arithmetic.
            # This is an honest, loud failure, not a "safe" guess: the
            # absence of an observed negative endpoint is not proof
            # that no negative interior exists, and a method with more
            # precision (interval arithmetic, arbitrary-precision, or
            # exact polynomial analysis for the constant-w case) could
            # in principle resolve an interval this narrow that this
            # float64 implementation cannot.
            raise _IndeterminateRootSearch(
                f"Could not certify that E(a)^2 stays positive, nor find "
                f"a root, between a={lo:.6g} and a={hi:.6g}; this "
                "interval is already as narrow as this program's float64 "
                "arithmetic can subdivide (minimum E(a)^2 observed at "
                f"either endpoint: {min(e2_lo, e2_hi):.6g}). This is an "
                "honest limitation of this specific implementation, not "
                "evidence that no root exists here."
            )
        e2_mid = e2(mid)
        left = scan(lo, mid, e2_lo, e2_mid, depth + 1)
        if left is not None:
            return left
        return scan(mid, hi, e2_mid, e2_hi, depth + 1)

    # The certificate above is tightest over a NARROW dynamic range: its
    # Lipschitz bound uses the largest |dE2/da| anywhere in the interval,
    # and that can be many orders of magnitude larger at the small-a end
    # of a wide request (e.g. a_lo near a turning point but a_max many
    # decades beyond it) than anywhere else in the range, which would
    # otherwise force needless deep recursion across huge stretches of
    # ordinary, uneventful territory just to re-derive that they are
    # fine. A moderate geometric (log-spaced) outer partition keeps each
    # cell's own dynamic range modest -- and, unlike the old fixed
    # linear grid, this partition only has to be coarse enough that each
    # cell's *certificate* converges quickly, never fine enough to
    # directly resolve a root itself: that job still belongs entirely to
    # the certified recursion above, which is designed so that a root's
    # detection does not depend on how coarsely the outer cells are cut
    # -- see _e2_derivative_lipschitz_bound's docstring for the precise
    # sense (exact-arithmetic analytic bound, conservatively inflated in
    # float64) in which that recursion's coverage is supported.
    n_coarse = 300
    if a_max > 2.0 * a_lo:
        edges = np.geomspace(a_lo, a_max, n_coarse + 1)
    else:
        edges = np.linspace(a_lo, a_max, n_coarse + 1)
    e2_prev = e2(float(edges[0]))
    for i in range(n_coarse):
        lo_i, hi_i = float(edges[i]), float(edges[i + 1])
        e2_hi_i = e2(hi_i)
        bracket = scan(lo_i, hi_i, e2_prev, e2_hi_i, 0)
        if bracket is not None:
            return bracket
        e2_prev = e2_hi_i
    return None


def _dominant_leading_term_exponent_and_sign(omega_m, omega_r, omega_k,
                                              omega_de, w0, wa):
    """
    As a -> 0+, E(a)^2 = Om*a^-3 + Or*a^-4 + Ok*a^-2 + Ode*g(a), where
    the CPL dark-energy shape g(a) behaves as a^n_de with n_de =
    -3*(1+w0+wa) in that limit: ln g(a) = -3(1+w0+wa)*ln(a) - 3*wa*(1-a),
    and the second term tends to the FINITE constant -3*wa as a->0, so
    it only contributes a bounded multiplicative factor exp(-3*wa) and
    never changes which power of a dominates. Whichever single term has
    the most negative exponent (steepest divergence) therefore fixes
    the a->0 SIGN of E(a)^2 itself -- a purely algebraic fact about the
    model's parameters, independent of a_i and of the integrated
    trajectory. This is a fundamentally different, wider question than
    early_time_offset_Gyr's "which term dominates AT a_i" check just
    below: that one compares magnitudes at a specific a_i and can be
    satisfied even when a different term is the true a->0 asymptote;
    this one is exact in the strict limit (see Codex Audit 6 P0-1).

    Returns (p_min, sign, tied):
      p_min  -- the most negative exponent among components with a
                nonzero coefficient (None if every coefficient is
                exactly zero).
      sign   -- +1.0, -1.0, or 0.0: the sign of the SUM of coefficients
                of every component sharing that most-negative exponent.
                0.0 only in the knife-edge case where that sum is
                exactly zero (an exact leading-order cancellation,
                genuinely indeterminate from this term alone).
      tied   -- True if more than one component shares p_min.
    """
    terms = []
    if omega_m != 0.0:
        terms.append((-3.0, omega_m))
    if omega_r != 0.0:
        terms.append((-4.0, omega_r))
    if omega_k != 0.0:
        terms.append((-2.0, omega_k))
    if omega_de != 0.0:
        n_de = -3.0 * (1.0 + w0 + wa)
        terms.append((n_de, omega_de))
    if not terms:
        return None, 0.0, False
    p_min = min(p for p, _c in terms)
    coeffs_at_min = [c for p, c in terms if p == p_min]
    tied = len(coeffs_at_min) > 1
    s = sum(coeffs_at_min)
    sign = 1.0 if s > 0.0 else (-1.0 if s < 0.0 else 0.0)
    return p_min, sign, tied


def _big_bang_connectivity_anchor(a_i, omega_m, omega_r, omega_k, omega_de,
                                   w0, wa, expected_sign):
    """
    Find a small anchor scale factor a_anchor << a_i at which the
    NUMERICALLY evaluated sign of E(a)^2 agrees with the analytically
    predicted a->0 limiting sign (expected_sign, from
    _dominant_leading_term_exponent_and_sign), by shrinking a_anchor
    geometrically (by factors of 1e-3) until they agree or a hard
    iteration limit / de_density_shape's own conservative policy floor
    is hit. This anchor then serves as a verified non-negative starting
    point for _first_root_ahead's certified search up to a_i, which is
    what actually certifies (or refutes) that a_i sits on a branch
    continuously connected back toward a=0, with no hidden forbidden
    interval in between.

    Raises _IndeterminateRootSearch if no anchor small enough to agree
    with the predicted sign can be reached -- an honest failure, never
    a silent guess. Shrinking a_anchor further after de_density_shape's
    own |ln g(a)| policy floor is hit cannot help (it only makes
    |ln g(a)| larger, not smaller), so that case stops immediately
    rather than looping to the iteration limit.
    """
    a_anchor = a_i * 1.0e-3
    for _ in range(80):
        if a_anchor <= 0.0 or not math.isfinite(a_anchor):
            break
        try:
            e2_anchor = float(E2(a_anchor, omega_m, omega_r, omega_k,
                                  omega_de, w0, wa))
        except ValueError:
            break
        if math.isfinite(e2_anchor) and e2_anchor != 0.0:
            observed_sign = 1.0 if e2_anchor > 0.0 else -1.0
            if observed_sign == expected_sign:
                return a_anchor, e2_anchor
        a_anchor *= 1.0e-3
    raise _IndeterminateRootSearch(
        "Could not find an anchor scale factor close enough to a=0 whose "
        "numerically evaluated E(a)^2 sign agrees with the analytically "
        "predicted Big-Bang-limit sign, needed to certify that a_i is "
        "connected to the true Big Bang along a continuous expanding "
        "branch. This is an honest limitation of this program's "
        "float64 search, not evidence that no such branch exists."
    )


_LEGGAUSS_CACHE = {}


def _leggauss_cached(n_nodes):
    """np.polynomial.legendre.leggauss(n) performs an eigenvalue
    decomposition and is not cheap; both quadrature routines below call
    it with the same fixed n_nodes on every invocation, and the new
    interior-sample generation across a turning-point handoff
    (_advance_to_boundary) can call one of them thousands of times for
    a single run. The nodes/weights depend only on n_nodes, never on
    the physics parameters, so they are computed once per n_nodes and
    reused -- a pure performance optimization with no effect on the
    numerical result."""
    cached = _LEGGAUSS_CACHE.get(n_nodes)
    if cached is None:
        cached = np.polynomial.legendre.leggauss(n_nodes)
        _LEGGAUSS_CACHE[n_nodes] = cached
    return cached


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
    nodes, weights = _leggauss_cached(n_nodes)
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


_BRACKET_NOT_YET_KNOWN = object()  # sentinel distinct from a real bracket
                                    # (a pair) or a confirmed "no root"
                                    # result (None), so a caller can pass
                                    # its own already-computed None (no
                                    # root found) without it being
                                    # mistaken for "not yet computed".
_INDETERMINATE = object()  # sentinel a caller can use to mark a
                            # _first_root_ahead search that raised
                            # _IndeterminateRootSearch, distinct from
                            # both a real bracket and a certified "no
                            # root" None -- used where a caller wants to
                            # handle "could not resolve" as its own
                            # third outcome rather than letting the
                            # exception propagate (e.g. the Big Rip
                            # look-ahead check, which degrades to a
                            # neutral "not certified" statement instead
                            # of crashing an otherwise-valid run).


_BOUNDARY_INTERIOR_SAMPLES = 120  # interior (t, a) points generated across
                               # a genuine turning-point handoff, evenly
                               # spaced in cosmic time (found by bisecting
                               # the monotonic time-to-turnaround
                               # quadrature, not in the singularity-
                               # removing variable u=sqrt(a_turn-a) that
                               # quadrature itself uses internally) -- see
                               # _advance_to_boundary. This keeps the
                               # returned trajectory from silently
                               # jumping straight from the handoff point
                               # to the resolved endpoint in one segment
                               # that can span a large fraction of the
                               # run's remaining history.


def _advance_to_boundary(a, t, dt, args, a_max, t_max_gyr,
                          bracket=_BRACKET_NOT_YET_KNOWN):
    """
    Called when the current step has been flagged as approaching a
    turning point (or has already overshot one). Determines which of
    three mutually exclusive events actually comes first -- reaching
    a_max, reaching t_max, or a genuine turning point -- and returns
    (t_next, a_next, event, time_to, interior_samples). a_max and
    t_max, when given, are treated as hard stopping bounds: EVERY
    returned point, including a_next itself, satisfies a_next <= a_max
    and (when given) t_next <= t_max_gyr -- a turning point beyond
    either of them is never reported, exactly as an ordinary
    (non-turning-point) step would never be allowed to overshoot them.
    This is an unconditional invariant, not a labeling convenience: it
    is enforced by comparing a_max and a_turn (and t_max_gyr and
    t_turn) with plain, tolerance-free "<"/">" on the literal values
    involved, never a floating-point-noise tolerance that could let a
    reported endpoint exceed a bound the user actually specified (a
    tolerance-based reclassification tried in an earlier revision, to
    make a coincidental a_max/a_turn tie read as "turnaround," could
    return a_next slightly past a_max -- a strictly worse defect than
    the label-flip it was meant to smooth over, since it broke the
    hard-bound contract this function exists to guarantee. A user who
    deliberately sets a_max a hair below a genuine turning point gets
    exactly what they asked for: stop_reason='a_max' at that literal
    a_max, not a silently substituted a_turn).

    time_to is a callable, time_to(target_a) -> elapsed cosmic time
    from the current a to any target_a in [a, a_next], used by the
    caller to place checkpoints (such as a=1) that fall in this
    interval. Everything here is computed by quadrature, never by a
    single enlarged RK4 step: near a turning point da/dt ~
    sqrt(a_turn-a) has an infinite derivative, and a full-stride RK4
    step aimed at a target close to that singularity cannot be trusted
    (RuntimeError near/at an exact turnaround, silently wrong
    checkpoint times, or a leaked internal exception near t_max).

    interior_samples is a list of (t_i, a_i) pairs strictly between the
    starting (a, t) and the returned (a_next, t_next), evenly spaced in
    cosmic time (via bisection on the monotonic time-to-turnaround
    quadrature -- see the inline comment in the nested interior_samples
    construction below for why time, not the quadrature's own internal
    u=sqrt(a_turn-a) substitution variable, is the right spacing choice)
    when a genuine turning point anchors this handoff (empty when it
    resolves to an
    ordinary, non-singular a_max/t_max boundary with no turning point
    nearby at all, since there the quadrature used for the single
    remaining segment is already smooth and accurate throughout, with
    no singularity to resolve finely). Without these, the interval
    between the point where turning-point handling engaged and the
    actual stopping point -- which can span a large fraction of the
    run's remaining time, since the proactive trigger can fire well
    before the singularity itself -- would be represented by only its
    two endpoints, silently discarding the trajectory's shape across
    everything in between even though the reported endpoint time
    itself is accurate.

    bracket, if given (as a bracket pair, or None meaning "confirmed no
    root"), is a pre-computed _first_root_ahead(...) result the caller
    has already determined -- e.g. a proactive probe the main loop
    performed to decide whether to divert here at all -- so this
    function can skip a redundant repeat of that same certified
    search. Leaving it at its default performs the search itself,
    exactly as before.
    """
    def _bisect_for_time(time_to_fn, target_dt, lo, hi, n_iter=MAX_BISECTIONS):
        # Solve time_to_fn(target_a) == target_dt for target_a in
        # [lo, hi], by bisection on the (monotonically increasing in
        # target_a) quadrature-based time function itself -- never by
        # RK4-stepping toward a target that may sit close to a
        # singularity.
        for _ in range(n_iter):
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
    # closer its endpoint sits to a real turning point. Extending the
    # search past a_max, by a margin tied to how far away a_max already
    # is, reliably converts any such near-boundary coincidence into an
    # ordinary "root found" case below, handled by the same
    # singularity-removing quadrature used for an interior turning
    # point -- which remains fully accurate for a target arbitrarily
    # far from, at, or infinitesimally below the anchor root, so there
    # is no accuracy cost to preferring it whenever any root is found
    # nearby. It does not change how an interior root (strictly before
    # a_max, e.g. the two-positive-root case) is found, since that is
    # already detected by this same scan without needing to look past
    # a_max at all.
    if bracket is _BRACKET_NOT_YET_KNOWN:
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
            return t_max_gyr, a_next, "t_max", time_to, []
        return t + t_to_amax, a_max, "a_max", time_to, []

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

    def interior_samples(a_stop):
        # Evenly spaced points in COSMIC TIME between the current (t,
        # a) and (t_stop, a_stop) -- exclusive of both ends, which the
        # caller supplies -- each located by bisecting the same
        # monotonic time_to quadrature used for the endpoint itself,
        # never by RK4-stepping toward a target near the singularity.
        # An earlier version of this sampled evenly in the
        # singularity-removing variable u=sqrt(a_turn-a) instead, which
        # gives good resolution close to a_turn (where the quadrature's
        # integrand is smooth and bounded in u, so steps there are also
        # roughly uniform in time) but not necessarily far from it: the
        # proactive handoff that reaches this function can fire well
        # before the singularity, leaving a stretch where dt/du is not
        # constant and a large time gap could still appear. Bisecting
        # directly on time removes that assumption and bounds the
        # largest gap in COSMIC TIME across the ENTIRE handoff, which
        # is what most affects how the returned trajectory looks in a
        # plot or CSV. A reduced iteration count (well beyond what
        # placing a cosmetic interior sample needs, though far fewer
        # than the full precision the actual event boundary requires)
        # keeps the added cost modest.
        t_stop = t + time_to(a_stop)
        if t_stop <= t:
            return []
        ts = np.linspace(t, t_stop, _BOUNDARY_INTERIOR_SAMPLES + 2)[1:-1]
        pts = []
        lo_a = a
        for t_target in ts:
            a_i = _bisect_for_time(time_to, t_target - t, lo_a, a_stop,
                                    n_iter=40)
            pts.append((t_target, a_i))
            lo_a = a_i  # target_a is monotonically increasing in t, so
                        # each subsequent bisection can start from here
        return pts

    # a_max and a_turn (and, below, t_max_gyr and t_turn) are compared
    # with plain, tolerance-free "<"/">": a_max is a HARD bound on the
    # returned trajectory (see the docstring above), so a requested
    # a_max even a single ULP below the true a_turn must still return
    # exactly that a_max, not a substituted a_turn that would exceed
    # it. A user who wants "turnaround" reported instead can simply
    # request an a_max at or beyond a_turn.
    if a_max < a_turn:
        t_to_amax = time_to(a_max)
        if t_max_gyr is not None and t + t_to_amax > t_max_gyr:
            a_next = _bisect_for_time(time_to, t_max_gyr - t, a, a_max)
            return t_max_gyr, a_next, "t_max", time_to, interior_samples(a_next)
        return t + t_to_amax, a_max, "a_max", time_to, interior_samples(a_max)

    t_turn = t + t_to_turn
    if t_max_gyr is not None and t_turn > t_max_gyr:
        # The turning point is real and within a_max, but occurs later
        # than the requested t_max: t_max stops the run first, still on
        # the smooth pre-turnaround branch. Compared with plain "<"/">"
        # for the same hard-bound reason as a_max above -- t_max is
        # never overridden by a near-tie with t_turn.
        a_next = _bisect_for_time(time_to, t_max_gyr - t, a, a_turn)
        return t_max_gyr, a_next, "t_max", time_to, interior_samples(a_next)

    return t_turn, a_turn, "turnaround", time_to, interior_samples(a_turn)


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
    nodes, weights = _leggauss_cached(n_nodes)
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


def _bisect_time_to_turnaround_for_a(delta_t, a_turn, args, a_lo_bound,
                                      n_iter=MAX_BISECTIONS):
    """
    Invert _quad_time_to_turnaround: given a target time-to-turnaround
    delta_t (assumed to lie in [0, _quad_time_to_turnaround(a_lo_bound,
    a_turn, args)] -- the caller checks this), find the scale factor
    a_source in [a_lo_bound, a_turn] such that
    _quad_time_to_turnaround(a_source, a_turn, args) == delta_t.

    _quad_time_to_turnaround is exactly monotonically DEcreasing in its
    a_lo argument (a scale factor closer to a_turn always has less time
    left to reach it), so this is a well-posed inversion by bisection on
    that function's OWN OUTPUT -- reusing the same accurate,
    singularity-aware quadrature already used to go forward from a
    given a to a_turn, rather than needing any new physics or a separate
    "mirror quadrature." This is what resolves the exact contracting-
    branch endpoint at a requested t_max (see Copilot Audit 6 P0-1 /
    Codex Audit 6 P1-1, which independently flagged the same defect: an
    earlier version merely filtered the mirrored samples to t <= t_max
    without inserting the actual state at t_max).
    """
    lo, hi = a_lo_bound, a_turn
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if _quad_time_to_turnaround(mid, a_turn, args) > delta_t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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

    # E(a_i)^2 > 0 alone does NOT mean a_i lies on a branch continuously
    # connected back to the true Big Bang at a=0: there can be a hidden
    # forbidden interval (E(a)^2 < 0 between two nearby roots) strictly
    # between a=0 and a_i, or E(a)^2 itself can tend to a NEGATIVE value
    # as a->0 for this parameter combination even though it happens to
    # be positive at a_i. Both are certified against here, reusing the
    # same certified search machinery _first_root_ahead already uses to
    # look for a FUTURE turning point, pointed instead at the interval
    # behind the starting point (see Codex Audit 6 P0-1).
    args0 = (H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa)
    p_min, expected_sign, _tied = _dominant_leading_term_exponent_and_sign(
        omega_m, omega_r, omega_k, omega_de, w0, wa)
    if p_min is not None:
        if expected_sign == 0.0:
            raise ValueError(
                "This parameter combination has an exact cancellation "
                "among the components that jointly dominate as a->0 "
                "(their coefficients sum to zero at the leading power "
                "of a), so the sign of E(a)^2 in the Big-Bang limit "
                "cannot be determined from the leading term alone, and "
                "whether a_i is connected to a genuine Big Bang cannot "
                "be certified. Adjust the density parameters slightly "
                "to break the tie."
            )
        if expected_sign < 0.0:
            raise ValueError(
                f"E(a)^2 tends to a negative value as a->0 for this "
                "parameter combination (the component that dominates "
                "in that limit has a negative coefficient), even though "
                f"E(a_i={a_i:.3g})^2 > 0: a_i sits on an isolated "
                "expanding branch with no continuous path back to the "
                "true Big Bang at a=0. This model has no valid history "
                "before some earlier turning point; increase a_i past "
                "that point, or change the density parameters."
            )
        # _IndeterminateRootSearch is deliberately let through UNCHANGED
        # here (not wrapped in a ValueError) -- it already carries a
        # clear message, and it is the same tri-state, loud-failure
        # search used elsewhere in this module (e.g. for a future
        # turnaround), which callers and tests already distinguish by
        # exception type rather than by parsing a wrapped message.
        a_anchor, _e2_anchor = _big_bang_connectivity_anchor(
            a_i, omega_m, omega_r, omega_k, omega_de, w0, wa,
            expected_sign)
        bracket = _first_root_ahead(a_anchor, a_i, args0)
        if bracket is not None:
            lo_b, hi_b = bracket
            raise ValueError(
                "This parameter combination has a hidden forbidden "
                f"interval (E(a)^2 < 0 somewhere between a={lo_b:.6g} "
                f"and a={hi_b:.6g}) strictly between the true Big Bang "
                f"(a=0) and a_i={a_i:.3g}: E(a_i)^2 > 0 does not by "
                "itself mean a_i is on a branch continuously connected "
                "back to a=0. Increase a_i to sit past this forbidden "
                "interval, or change the density parameters."
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
    # Once a proactive probe has CERTIFIED that no root of E(a)^2=0
    # exists anywhere between the current a and a_max, that remains
    # true for the rest of the run: the certified root-free interval
    # only shrinks as a advances (it stays a subset of [a, a_max]), so
    # there is no need to repeat that same expensive certified search
    # on every subsequent step through a "loitering" dip. See its use
    # below, in the near_turnaround handling.
    certified_safe_to_a_max = False

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
        near_turnaround = False
        if not certified_safe_to_a_max:
            dE2 = _dE2_da(a, omega_m, omega_r, omega_k, omega_de, w0, wa)
            if dE2 < 0.0:
                delta_a_est = e2_now / (-dE2)
                if delta_a_est < _TURNAROUND_TRIGGER_REL * a:
                    near_turnaround = True

        # A proactive trigger can be a FALSE ALARM: a "loitering"
        # cosmology where E(a)^2 dips close to zero and rebounds
        # without ever crossing it. Probe for a genuine root right
        # here, before committing to the boundary-handling branch
        # below, rather than letting _advance_to_boundary discover "no
        # root" on its own and fast-forward straight to a_max by
        # quadrature -- that shortcut is only valid when no ordinary
        # RK4 stepping remains to be done in between, which is false
        # here: a confirmed false alarm still has ordinary trajectory
        # between here and a_max that must be resolved and recorded
        # step by step, not skipped.
        root_bracket = _BRACKET_NOT_YET_KNOWN
        if near_turnaround:
            search_margin = max(0.5 * (a_max - a), 0.1 * a_max, 1.0e-9)
            root_bracket = _first_root_ahead(a, a_max + search_margin, args)
            if root_bracket is None:
                certified_safe_to_a_max = True
                near_turnaround = False

        if certified_safe_to_a_max:
            # Use the cancellation-safe reference scale rather than the
            # true (signed) H(a): near the bottom of a confirmed
            # loitering dip, E(a)^2 can be anomalously small purely
            # from near-cancellation among its terms, which would
            # otherwise make dt = step_frac / H(a) grow pathologically
            # large and take one enormous RK4 stride across the dip,
            # discarding trajectory resolution through it. This is at
            # least as small as the ordinary step size everywhere else
            # (never larger), so it costs nothing once past the dip.
            dt = _safe_dt(a, args, step_frac)
        else:
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
            # trigger zone in a single stride. root_bracket is passed
            # through so a genuine root already found by the probe above
            # is not searched for a second time; a reactive-only
            # overshoot (root_bracket left at its "not yet known"
            # sentinel here) still gets a fresh search of its own.
            t_next, a_next, event, time_to, interior = _advance_to_boundary(
                a, t, dt, args, a_max, t_max_gyr, bracket=root_bracket)
            # interior holds any fine-grained (t, a) samples generated
            # across a genuine turning-point handoff (see
            # _advance_to_boundary): without these, this segment would
            # be represented only by its two endpoints, even though it
            # can span a large fraction of the run's remaining time.
            # Merge them with any checkpoint that also falls in this
            # same interval, all in increasing-a order, before the
            # final endpoint itself.
            merged_points = list(interior)
            for c in list(checkpoints):
                if a < c <= a_next:
                    # Use the same quadrature this function already used
                    # to resolve the boundary itself, NOT an RK4 sub-step
                    # bisection: a fixed small dt from before the
                    # turning-point heuristic fired is not guaranteed to
                    # even reach a checkpoint like a=1 in one stride here,
                    # and bisecting an unverified bracket can silently
                    # return a plausible-looking but wrong checkpoint time.
                    t_c = t + time_to(c)
                    checkpoint_times[c] = t_c
                    merged_points.append((t_c, c))
                    checkpoints.remove(c)
            merged_points.sort(key=lambda pt: pt[1])
            for t_pt, a_pt in merged_points:
                t_list.append(t_pt)
                a_list.append(a_pt)
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
    if turnaround is not None:
        # Exact time-reversal symmetry of the Friedmann equation: E(a)^2
        # depends on a alone, never on the sign of a_dot, so the total
        # lifetime of a recollapsing model (twice the time to
        # turnaround) is known exactly the instant a genuine turnaround
        # is found -- independent of whether the mirrored contracting-
        # branch ARRAY data below is also generated. An earlier version
        # left total_lifetime_gyr as None whenever --continue_collapse
        # was not requested, silently withholding a fundamental physical
        # property of the model from the run summary even though it was
        # already known exactly (Gemini Audit 6).
        t_turn = turnaround["t_turn_gyr"]
        total_lifetime_gyr = 2.0 * t_turn

        if continue_collapse:
            # The contracting branch is the mirror image of the
            # expanding one about the turnaround. No re-integration is
            # needed.
            t_mirror = 2.0 * t_turn - t_arr[::-1][1:]
            a_mirror = a_arr[::-1][1:]
            # t_max_gyr, when given, is a hard bound on the RETURNED
            # trajectory on either branch -- mirroring must not silently
            # extend the arrays past a time limit the forward
            # integration itself would have respected. This matters
            # whenever the turnaround occurs before t_max_gyr (so the
            # forward pass legitimately reports "turnaround," not
            # "t_max") but t_max_gyr itself falls strictly on the
            # CONTRACTING branch, i.e. t_turn < t_max_gyr < 2*t_turn.
            if t_max_gyr is not None:
                keep = t_mirror <= t_max_gyr
                if not np.all(keep):
                    t_mirror = t_mirror[keep]
                    a_mirror = a_mirror[keep]
                    # Merely dropping every mirrored sample past t_max
                    # (as an earlier version did) silently returned a
                    # trajectory ending strictly BEFORE t_max while still
                    # claiming stop_reason="t_max" (Copilot Audit 6 P0-1
                    # / Codex Audit 6 P1-1, independently). Resolve the
                    # EXACT contracting-branch state at t_max_gyr
                    # instead: by the same time-reversal symmetry used
                    # to build t_mirror above, the contracting state at
                    # t_max_gyr is the mirror image of the EXPANDING
                    # state at t_source = 2*t_turn - t_max_gyr, whose
                    # scale factor is found by inverting the existing,
                    # accurate _quad_time_to_turnaround quadrature (never
                    # by RK4-stepping near the singularity) -- no new
                    # physics or a separate "mirror quadrature" needed.
                    a_turn_val = turnaround["a_turn"]
                    delta_t = t_max_gyr - t_turn
                    a_lo_bound = float(a_arr[0])
                    max_delta_t = _quad_time_to_turnaround(
                        a_lo_bound, a_turn_val, args0)
                    if delta_t < 0.0 or delta_t > max_delta_t:
                        raise ValueError(
                            f"--t_max ({t_max_gyr:.6g} Gyr) falls on the "
                            "contracting branch but its mirrored source "
                            f"point precedes a_i={a_lo_bound:.3g}, "
                            "outside the domain of this run's computed "
                            "trajectory; choose a smaller --t_max or a "
                            "smaller --a_i."
                        )
                    a_source = _bisect_time_to_turnaround_for_a(
                        delta_t, a_turn_val, args0, a_lo_bound)
                    t_mirror = np.concatenate([t_mirror, [t_max_gyr]])
                    a_mirror = np.concatenate([a_mirror, [a_source]])
                    # The returned trajectory was truncated by t_max on
                    # the contracting branch, even though the forward
                    # pass reached a genuine turnaround -- the RETURNED
                    # history stops at t_max, short of the full mirrored
                    # lifetime recorded above in total_lifetime_gyr.
                    stop_reason = "t_max"
            t_arr = np.concatenate([t_arr, t_mirror])
            a_arr = np.concatenate([a_arr, a_mirror])

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

    # Search only out to the scale factor the run actually reached (a),
    # not the requested a_max: if the run stopped early at a genuine
    # turnaround or at --t_max, a milestone reported between the reached
    # 'a' and a_max would describe a point the trajectory never got to.
    # a_eq_rm = omega_r/omega_m is a pure algebraic identity, computed
    # independently of the integrated trajectory -- but it still
    # describes a scale factor, and the SAME reachability rule that
    # already gated a_eq_mde/a_accel below (both found by scanning only
    # up to a_reached) must also apply here: reporting a_eq_rm beyond
    # a_reached would claim a milestone the model's own history never
    # got to, e.g. a model that recollapses at a~1 has no meaningful
    # "matter-radiation equality at a=100" (see Codex Audit 6 P1-3).
    a_reached = min(a, a_max)
    a_eq_rm = None
    if omega_m > 0.0 and omega_r > 0.0:
        _a_eq_rm_candidate = omega_r / omega_m
        if _a_eq_rm_candidate <= a_reached:
            a_eq_rm = _a_eq_rm_candidate

    a_eq_mde = _find_sign_change(
        lambda aa: (omega_fractions(aa, omega_m, omega_r, omega_k, omega_de, w0, wa)[0]
                    - omega_fractions(aa, omega_m, omega_r, omega_k, omega_de, w0, wa)[3]),
        1.0e-6, a_reached)

    a_accel = _find_sign_change(
        lambda aa: deceleration_q(aa, omega_m, omega_r, omega_k, omega_de, w0, wa),
        1.0e-6, a_reached)

    big_rip_gyr = None
    # fate_status/future_turnaround_a/fate_search_limit_a are structured,
    # machine-readable fields distinguishing the model's ultimate
    # PHYSICAL FATE from stop_reason (why the RETURNED trajectory
    # stopped -- e.g. it can stop at t_max while fate_status is still
    # "recollapse"). Earlier this fate distinction lived only in the
    # human-readable warnings strings above, which neither --compare
    # mode's summary table nor a CSV consumer could parse (see Copilot
    # Audit 6 P2-2 / Codex Audit 6 P1-2). Values: "recollapse" (a
    # turnaround was found within the requested run), "big_rip",
    # "future_recollapse" (certified to recollapse beyond the requested
    # a_max/t_max, so never shown in this run's own trajectory), or
    # "unresolved" (the phantom-fate search itself could not certify
    # either outcome); None when none of these fate questions apply
    # (e.g. w0 >= -1, or omega_de <= 0).
    fate_status = "recollapse" if turnaround is not None else None
    future_turnaround_a = None
    fate_search_limit_a = None
    if wa == 0.0 and w0 < -1.0 and omega_de > 0.0 and turnaround is None and age_today_gyr is not None:
        # turnaround is None here means only that the REQUESTED run (up
        # to a_max/t_max) did not encounter a turning point before
        # stopping -- NOT that no root of E(a)^2=0 exists at all beyond
        # that point. A closed (or otherwise negative-curvature-
        # dominated) phantom model can still recollapse at some a
        # beyond the requested boundary, before the growing phantom
        # term ever takes over; reporting a Big Rip for such a model
        # would state a fate its expanding branch can never actually
        # reach. This is certified, not assumed: the phantom term
        # omega_de*a^p (p=-3(1+w0)>0 since w0<-1 and wa=0, so g(a) is a
        # pure power law) grows without bound, while every OTHER
        # component's magnitude is non-increasing for a>=1 -- so a
        # cheap, closed-form upper bound on all of them gives a finite
        # scale a_dom beyond which the phantom term is guaranteed to
        # dominate, and hence E(a)^2>0, for every larger a as well. The
        # same certified search used for ordinary turnaround detection
        # then only has to resolve the finite stretch from here to
        # a_dom.
        #
        # Only NEGATIVE curvature can ever drive E(a)^2 negative for
        # a>=1: matter and radiation have non-negative coefficients and
        # their a^-3/a^-4 contributions are non-increasing there, so
        # they can never overcome a growing phantom term. An earlier
        # version summed in abs(omega_m)+abs(omega_r)+abs(omega_k) as
        # "hazards" here, which was not merely numerically unsafe but
        # substantively wrong: including non-negative components that
        # can never cause a future recollapse. The dominance-scale
        # exponent below is 1/(p_phantom+2), not 1/p_phantom: the "+2"
        # covers curvature's own a^-2 falloff directly (rather than
        # bounding it by its value at a=1 the way the old formula
        # effectively did), and it stays bounded (<=0.5) even as
        # w0->-1+ (p_phantom->0+) -- unlike 1/p_phantom, which diverges
        # in exactly that limit and could overflow float64 when used to
        # raise a power directly (see Codex Audit 6 P0-2's w0=-1.000001
        # reproducer). Everything here is worked out, and evaluated, in
        # log space so no intermediate power large enough to overflow
        # is ever actually formed.
        args_full = (H0_pgyr, omega_m, omega_r, omega_k, omega_de, w0, wa)
        a_probe = max(a, 1.0)
        p_phantom = -3.0 * (1.0 + w0)
        c_other = -omega_k if omega_k < 0.0 else 0.0
        a_dom_cap = max(1.0e12, 1.0e6 * a_probe)
        log_a_probe = math.log(a_probe)
        log_a_dom_cap = math.log(a_dom_cap)
        exponent = 1.0 / (p_phantom + 2.0)
        if c_other <= 0.0:
            dominance_certain = True
            a_dom = a_probe
        else:
            log_a_dom_raw = (math.log(c_other) - math.log(omega_de)) * exponent
            dominance_certain = log_a_dom_raw <= log_a_dom_cap
            log_a_dom = min(max(log_a_dom_raw, log_a_probe), log_a_dom_cap)
            a_dom = math.exp(log_a_dom)

        try:
            future_bracket = _first_root_ahead(a_probe, a_dom, args_full)
        except _IndeterminateRootSearch:
            future_bracket = _INDETERMINATE

        fate_search_limit_a = a_dom
        if future_bracket is None and dominance_certain:
            m = -1.5 * (1.0 + w0)
            big_rip_gyr = age_today_gyr + 1.0 / (m * H0_pgyr * math.sqrt(omega_de))
            fate_status = "big_rip"
            warnings.append(
                "w0 < -1 (phantom dark energy): checked, out to the "
                "scale factor beyond which the phantom term is "
                "guaranteed to permanently dominate every other "
                "component, that the expanding branch does not "
                "recollapse first. The DE density grows without bound, "
                "driving a's expansion rate to infinity at a finite "
                "future time (a 'Big Rip'). The quoted t_rip is an "
                "analytic estimate that neglects the (by then "
                "negligible) matter and radiation contributions and "
                "assumes wa=0."
            )
        elif isinstance(future_bracket, tuple):
            a_lo_f, a_hi_f = future_bracket
            a_turn_future = _bisect_root_a(a_lo_f, a_hi_f, args_full)
            fate_status = "future_recollapse"
            future_turnaround_a = a_turn_future
            warnings.append(
                "w0 < -1 (phantom dark energy), but this model's "
                f"expanding branch recollapses at a_turn~{a_turn_future:.6g} "
                "-- beyond the requested --a_max/--t_max, so not shown "
                "in this run's trajectory -- before it ever reaches the "
                "phantom-dominated regime. No Big Rip occurs for this "
                "model; increase --a_max or --t_max to see the "
                "recollapse directly."
            )
        else:
            fate_status = "unresolved"
            warnings.append(
                "w0 < -1 (phantom dark energy), but this program could "
                "not certify, within a practical search range, whether "
                "the expanding branch recollapses before entering the "
                "phantom-dominated regime. No Big Rip time is reported "
                "for this model."
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
        # Structured fate fields -- see the comment where fate_status is
        # first initialized, above, for the full description of each
        # value. These let --compare mode and CSV consumers report the
        # model's ultimate physical fate without parsing prose warnings.
        fate_status=fate_status,
        future_turnaround_a=future_turnaround_a,
        fate_search_limit_a=fate_search_limit_a,
        # These are two different counts, kept distinct rather than
        # conflated under one ambiguous "n_steps": n_forward_iterations
        # is the actual RK4 loop counter (how many times the ODE was
        # advanced) -- renamed from n_integration_steps (Codex Audit 6
        # P2-4) because "integration steps" read as though it might
        # also cover the quadrature-based turning-point/handoff work
        # elsewhere in this function, which it never did -- while
        # n_output_samples is the length of the returned arrays, which
        # also includes inserted checkpoints (e.g. a=1) and, with a
        # genuine turning-point handoff, the fine-grained interior
        # samples generated across it, and, with --continue_collapse,
        # the mirrored contracting-branch points that were never
        # independently integrated at all.
        n_forward_iterations=steps,
        n_output_samples=len(a_arr),
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
