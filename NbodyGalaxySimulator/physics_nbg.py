"""
physics_nbg.py
===============
Core physics engine for NbodyGalaxySimulator.

This module owns every governing equation used by the program:

  gravity     Newtonian point-mass gravity with Plummer softening, evaluated
              either by direct O(N^2) pairwise summation or by a Barnes-Hut
              (1986) octree approximation, O(N log N).

  integrator  A kick-drift-kick (leapfrog) time-stepping loop.  Leapfrog is
              symplectic: it does not conserve energy exactly step to step,
              but the error stays bounded for as long as the timestep
              resolves the local dynamical time, instead of drifting
              secularly the way a non-symplectic scheme would.

  initial conditions
              Plummer (1911) sphere sampling by the standard inversion/
              rejection method of Aarseth, Henon & Wielen (1974), used for
              the star-cluster mode; and a uniform sphere with a
              user-set initial virial ratio, used for the cold-collapse
              galaxy-formation mode.

  diagnostics energy and virial-ratio bookkeeping, Lagrangian radii,
              two-body relaxation and dynamical timescales, an
              instantaneous-unboundedness criterion, and the position-space
              divergence measure used by the chaos-sensitivity mode.

SI units are used internally throughout.  User-facing quantities are solar
masses, parsecs, km/s and megayears; main.py and driver_nbg.py convert at
the boundary, exactly as physics_sev.py converts to and from solar units.

This is a deliberately transparent teaching model, in the same spirit as
StellarEvolutionTracks: every simplification is stated here and in the
Help file. In particular, *Multiple* (the direct-summation few-body
program elsewhere in this project) is assumed as a prerequisite: this
program does not re-derive Newtonian two-body motion or re-teach what a
gravitational acceleration is.  It starts from the point where Multiple's
O(N^2), unsoftened, single-orbit-accurate approach stops scaling, and
teaches the two ideas that let a simulation reach hundreds or thousands of
bodies: force softening (so close encounters no longer need an
astronomically small timestep) and hierarchical (tree) force
approximation (so cost grows as N log N rather than N^2).
"""

from fractions import Fraction
import math
import warnings

import numpy as np

MODEL_VERSION = "1.0.0"


#: The exact source files this build identifier covers: only the four core
#: program modules listed here change this value -- a change to any other
#: file in the project (documentation, a sample-output file, or anything
#: else outside this list) leaves it unchanged.  Exposed so callers can
#: determine precisely what BUILD_ID covers without duplicating this list.
BUILD_ID_COVERS = (
    "physics_nbg.py",
    "driver_nbg.py",
    "main.py",
    "plot_nbg.py",
)


def _compute_build_id():
    """Return a short identifier derived from the core source files.

    MODEL_VERSION records the program's declared release version.  BUILD_ID
    additionally distinguishes source revisions that retain the same
    declared version.  The hash is independent of LF versus CRLF line
    endings and frames each file with its name and length so file-boundary
    changes cannot collide with an unchanged concatenated byte stream.

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


# ----------------------------------------------------------------------
# Physical constants (SI, CODATA 2022 / IAU nominal values)
# ----------------------------------------------------------------------
G = 6.674_30e-11               # m^3 kg^-1 s^-2  (CODATA 2022)

#: IAU 2015 Resolution B3 nominal solar mass parameter, exact by definition.
#: GM_sun is known from solar-system dynamics far more precisely than G or
#: M_sun individually, so M_sun is derived from it and from the same G used
#: everywhere else in this module, rather than hard-coding an independently
#: rounded kilogram value that could drift out of consistency with G.
GM_SUN_NOMINAL = 1.327_124_4e20    # m^3 s^-2
M_sun = GM_SUN_NOMINAL / G          # kg

#: IAU-defined astronomical unit (exact) and the parsec derived from it
#: (1 pc is the distance at which 1 AU subtends 1 arcsecond).
AU = 1.495_978_707e11           # m (IAU 2012, exact)
PC = AU * (648_000.0 / math.pi)  # m  (exact, given the AU above)
KPC = 1.0e3 * PC

YEAR = 365.25 * 86400.0         # s  (Julian year)
MYR = 1.0e6 * YEAR              # s
KM = 1.0e3                      # m

# ----------------------------------------------------------------------
# Safety limits
# ----------------------------------------------------------------------
MIN_BODIES = 3
MAX_BODIES = 5_000
MIN_STEPS = 1
MAX_STEPS = 200_000
MAX_SNAPSHOTS = 4_000
# MAX_BODIES and MAX_SNAPSHOTS are each individually bounded, but their
# PRODUCT also needs a bound: the stored positions and velocities
# histories are each (n_snapshots, n_bodies, 3) float64 arrays, so
# MAX_BODIES=5000 combined with MAX_SNAPSHOTS=4000 could allocate about
# 5000*4000*3*8*2 bytes (position + velocity) = 960 MB for a single run,
# which is not a practical memory-safety guard for a teaching tool on its
# own. This caps the product directly; see integrate_nbody().
MAX_BODY_SNAPSHOT_PRODUCT = 2_000_000
MIN_THETA = 0.0
MAX_THETA = 2.0
#: Above this particle count, the O(N^2) direct force method is accepted
#: but the driver prints a runtime-cost warning rather than silently
#: letting a run take an unexpectedly long time.
DIRECT_METHOD_WARN_BODIES = 800
#: Octree recursion is capped, not to bound cost (a well-separated
#: distribution never approaches this), but to guarantee termination when
#: two or more bodies coincide (or nearly coincide) in position, which
#: would otherwise force the recursive splitting-in-half of the bounding
#: box forever, since no finite depth of subdivision separates two
#: identical points into different octants.
MAX_TREE_DEPTH = 42
#: Guards the two rejection-sampling loops in plummer_sphere() against an
#: unbounded loop. Both loops have a per-draw acceptance probability
#: bounded well away from zero for any physically sensible parameter
#: choice, so this limit is a defensive backstop, not a value runs are
#: expected to approach.
MAX_REJECTION_ROUNDS = 10_000


# ======================================================================
# Small validation helpers (style matches physics_sev.py)
# ======================================================================
def _require_finite(name, value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number; got {value!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return value


def _require_positive(name, value):
    value = _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero; got {value:g}.")
    return value


def _require_nonnegative(name, value):
    value = _require_finite(name, value)
    if value < 0.0:
        raise ValueError(f"{name} must not be negative; got {value:g}.")
    return value


def _require_bool(name, value):
    """
    Reject anything that is not literally True/False before it is used in a
    truthiness test.  Without this, a caller who passes a non-bool
    "truthy" value to a boolean-like direct-API parameter (a non-empty
    string such as "False", or 0/1) is silently reinterpreted by Python's
    ordinary truthiness rules instead of getting a clear error -- the CLI
    itself never produces anything but a real bool here, but the physics
    layer is a reusable API and a caller bypassing the CLI can pass
    anything.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False; got {value!r}.")
    return value


def _require_int(name, value, lo=None, hi=None):
    value = _require_finite(name, value)
    if int(value) != value:
        raise ValueError(f"{name} must be an integer; got {value:g}.")
    value = int(value)
    if lo is not None and value < lo:
        raise ValueError(f"{name} must be at least {lo}; got {value}.")
    if hi is not None and value > hi:
        raise ValueError(f"{name} must not exceed {hi:,}; got {value:,}.")
    return value


def _require_method(method):
    if method not in ("tree", "direct"):
        raise ValueError(f"method must be 'tree' or 'direct'; got {method!r}.")
    return method


def _require_theta(theta):
    """
    Validates theta up front, at the same point as every other run-mode
    argument, before any initial-condition or timescale setup work runs
    -- rather than relying solely on compute_accelerations_tree()'s own
    [MIN_THETA, MAX_THETA] range check, which would not reject an
    out-of-range theta until deep inside the first leapfrog step.
    """
    theta = _require_finite("theta", theta)
    if not (MIN_THETA <= theta <= MAX_THETA):
        raise ValueError(
            f"theta must lie in [{MIN_THETA:g}, {MAX_THETA:g}]; got {theta:g}."
        )
    return theta


def _as_finite_array(values, name, shape=None):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array


def _require_snapshot(values, name, n_bodies=None):
    """
    Validate a single (n_bodies, 3) position/velocity snapshot array.

    Every public helper that takes one physical snapshot (as opposed to a
    whole (n_snapshot, n_bodies, 3) history -- position_space_divergence()
    is the one function that intentionally supports both, and validates
    its own shape separately) must reject anything that is not exactly
    two-dimensional with a trailing axis of 3, not merely check the
    leading axis against ``masses``. Without this, a caller could pass a
    2-D array meant to hold N one-component or two-component "vectors"
    (e.g. an (N, 2) array) and have it silently broadcast or index against
    the wrong axis, producing a normal-looking but physically meaningless
    number instead of an error.
    """
    array = _as_finite_array(values, name)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (n_bodies, 3); got {array.shape}."
        )
    if n_bodies is not None and array.shape[0] != n_bodies:
        raise ValueError(
            f"{name} must have one row per body ({n_bodies}); got shape "
            f"{array.shape}."
        )
    return array


def _require_masses(masses, n_bodies=None, allow_nonpositive=False):
    """
    Validate a masses array: exactly one-dimensional, non-empty, and (by
    default) strictly positive in every entry. Every public helper that
    takes a ``masses`` argument shares this one check, so a two-
    dimensional masses array (which would otherwise silently broadcast
    into a wrong-shaped intermediate result) or a bare scalar mass raises
    a clear ValueError here rather than a confusing failure deep inside
    whichever helper happened to be called. ``allow_nonpositive`` exists
    for callers that need shape/dtype validation on a masses-like array
    whose entries are not required to be strictly positive masses.
    """
    array = _as_finite_array(masses, "masses")
    if array.ndim != 1:
        raise ValueError(f"masses must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError("masses must contain at least one body; got an empty array.")
    if n_bodies is not None and array.shape[0] != n_bodies:
        raise ValueError(
            f"masses must have one entry per body ({n_bodies}); got shape "
            f"{array.shape}."
        )
    if not allow_nonpositive and np.any(array <= 0.0):
        raise ValueError("all masses must be strictly positive.")
    return array


def _validate_state(positions, velocities, masses):
    """Validate a full N-body state; return (positions, velocities, masses)."""
    masses = _require_masses(masses)
    n = masses.size
    if n < MIN_BODIES:
        raise ValueError(f"at least {MIN_BODIES} bodies are required; got {n}.")
    if n > MAX_BODIES:
        raise ValueError(f"at most {MAX_BODIES:,} bodies are supported; got {n:,}.")
    positions = _require_snapshot(positions, "positions", n_bodies=n)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=n)
    return positions, velocities, masses


# ======================================================================
# Scale-safe extreme-value helpers
#
# Every function below that reduces to a Euclidean separation, an
# inverse-square/inverse-cube force law, a total mass, or a squared
# speed can be handed individually finite inputs whose NAIVE evaluation
# order (square every component, then sum; sum every mass; square a
# velocity) forms an intermediate that overflows float64's range even
# though the true, fully-simplified mathematical answer is itself
# comfortably representable (occasionally as a subnormal). Such an
# overflow is not caught by an isfinite() check on the final result
# unless the intermediate itself is checked and repaired at the point
# it actually occurs; left unchecked, it silently becomes a finite-
# looking WRONG answer (typically 0.0, from inf**-1.5 or inf**-1)
# instead of a clear error or the true value.
#
# The shared strategy for the separation itself: factor out the
# largest-magnitude component before squaring (the same trick np.hypot
# uses internally for two arguments), so every squared ratio actually
# formed is of order 1 or smaller. A function whose formula only ever
# DIVIDES by that safe separation (potential energy, specific potential,
# lagrangian radii) needs nothing further -- dividing one finite value
# by another cannot overflow, only underflow toward a (legitimate)
# zero. A function whose formula needs an inverse SQUARE or CUBE of the
# separation (gravitational acceleration) additionally needs the two
# divisions and the unit-vector formed sequentially, with G folded in
# before the first division rather than at the very end: some
# representable results would otherwise underflow to exactly 0.0 in the
# un-scaled coefficient before G ever gets a chance to rescale them back
# into range.
#
# That reordering trick has a limit, though: it only rescues a product
# whose factors are combined SEQUENTIALLY (a*b then *c), one rounding at
# a time. There are cases where every individual pairwise reordering
# still fails because one PARTIAL product underflows to
# exactly 0.0 (or overflows to inf) before the remaining factor(s) can
# rescale it back into range -- e.g. 0.5*mass alone underflowing to 0
# for a mass at the representable-denormal floor, even though the fully
# combined 0.5*mass*v*v is an ordinary, representable number once the
# enormous v*v is folded in. _scaled_product() below is the general
# fix: it factors every argument into a mantissa in [0.5, 1) times a
# power of two (frexp), multiplies the mantissas and sums the exponents
# WITHOUT ever rounding a partial product to a real float64 in between,
# and reconstructs the final float (ldexp) only once, at the very end.
# This is RANGE-SAFE (immune to the intermediate overflow/underflow
# this section exists to fix) whenever the true mathematical product is
# itself finite and representable -- but it is not thereby EXACT: each
# running mantissa multiplication is still an ordinary, rounded float64
# operation, so a handful of ulp of accumulated rounding error relative
# to a single correctly-rounded result is expected, not eliminated (see
# _scaled_product()'s own docstring). It does not, and cannot, rescue a
# product whose true value is genuinely non-representable. The same
# caveat applies to every other "scale-safe" helper in this section:
# each is safe against the specific overflow/underflow/cancellation
# hazard it targets, not a guarantee of a faithfully-rounded result.
# ======================================================================
def _scaled_product(*factors):
    """
    Multiply the given (mutually broadcastable) float arrays/scalars
    together, WITHOUT ever materializing an intermediate partial product
    that could itself overflow to inf or underflow to 0 before the
    remaining factors are applied -- even though the true, fully combined
    mathematical product would have been representable in float64.

    Uses np.frexp to factor each argument into (mantissa in [0.5, 1) or
    0.0, integer exponent), accumulates the mantissa product and the
    exponent sum symbolically across all factors (renormalizing the
    running mantissa via frexp after each multiplication, so it never
    itself needs a wide dynamic range), and reconstructs the final float64
    with one np.ldexp call. Handles zero and negative factors correctly
    (frexp(0.0) = (0.0, 0); sign rides along in the mantissa).

    This is NOT a single correctly-rounded (0.5-ulp) result for the fully
    combined product across many factors: each running "mantissa =
    mantissa * fm" step is itself an ordinary float64 multiplication, so
    it rounds at every step just like any other chain of float64
    multiplications, and can accumulate a few ulp of error over many
    factors (typically no more than a handful of ulp for the small
    factor counts this module actually calls it with) relative to what
    an exact/extended-precision computation rounded once at the very end
    would give. What IS guaranteed is the overflow/underflow safety this
    helper exists for: the exponent is tracked
    separately as an exact integer with no intermediate range limit, so
    the only place this computation itself can overflow or underflow is
    the single final np.ldexp reconstruction -- exactly when the true
    mathematical product is not representable in float64, which is the
    correct, unavoidable case to report as such (e.g. via an isfinite()
    postcondition at the call site) -- not any intermediate step along
    the way, unlike a naive left-to-right product of the same factors.
    """
    shape = np.broadcast_shapes(*(np.shape(f) for f in factors))
    mantissa = np.ones(shape, dtype=float)
    exponent = np.zeros(shape, dtype=np.int64)
    for factor in factors:
        f = np.broadcast_to(np.asarray(factor, dtype=float), shape)
        fm, fe = np.frexp(f)
        mantissa = mantissa * fm
        # Renormalize immediately: two mantissas in [0.5, 1) multiply to
        # something in [0.25, 1), still perfectly safe, but repeating
        # this over MANY factors would otherwise let the running
        # mantissa itself drift toward underflow long before the true
        # product does.
        mantissa, me = np.frexp(mantissa)
        exponent = exponent + fe.astype(np.int64) + me.astype(np.int64)
    # over='ignore' silences NumPy's RuntimeWarning for the one case
    # where this reconstruction CAN legitimately overflow: the true
    # mathematical product itself is not representable. That is exactly
    # the case every call site already checks for via an isfinite()
    # postcondition immediately afterward, so it must not also raise a
    # warnings-as-errors exception here.
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(mantissa, exponent)


def _scaled_product_over(divisor, *factors, power=1):
    """
    Like _scaled_product(), but for ``factors`` divided by
    ``divisor ** power`` -- e.g. ``_scaled_product_over(s, G, m, dx,
    power=3)`` computes ``G * m * dx / s**3`` -- WITHOUT ever forming
    the standalone reciprocal ``1.0 / divisor`` (or ``divisor **
    power``) as its own float64 value first.

    That standalone reciprocal can itself be non-representable even
    when the true, fully-combined result is comfortably representable:
    a divisor that is a subnormal near the smallest representable
    positive float64 (~4.9e-324) has a true reciprocal near 2e+323, far
    past float64's ~1.8e+308 ceiling, so computing "1.0 / divisor" and
    folding it into _scaled_product() as an ordinary factor would
    overflow to +inf before the other factors ever get a chance to
    bring the combined magnitude back into range -- and multiplying
    that spurious +inf by a component that happens to be exactly 0.0
    (as when one body sits at the coordinate origin) produces a
    spurious nan instead of the true, finite, representable answer.

    Decomposing the divisor via frexp and negating (and scaling by
    ``power``) its exponent keeps the reciprocal's mantissa in the safe
    range (1, 2**power] throughout, exactly like every other factor in
    _scaled_product() stays in a safe mantissa range -- only the single
    final ldexp can overflow/underflow, and only when the true
    mathematical result itself is not representable.
    """
    shape = np.broadcast_shapes(np.shape(divisor), *(np.shape(f) for f in factors))
    mantissa = np.ones(shape, dtype=float)
    exponent = np.zeros(shape, dtype=np.int64)
    for factor in factors:
        f = np.broadcast_to(np.asarray(factor, dtype=float), shape)
        fm, fe = np.frexp(f)
        mantissa = mantissa * fm
        mantissa, me = np.frexp(mantissa)
        exponent = exponent + fe.astype(np.int64) + me.astype(np.int64)
    d = np.broadcast_to(np.asarray(divisor, dtype=float), shape)
    dm, de = np.frexp(d)
    # 1/divisor**power = (1/dm)**power * 2**(-de*power); dm in [0.5, 1)
    # keeps 1/dm in (1, 2], so raising it to a small integer power stays
    # comfortably representable no matter how extreme divisor itself is.
    rm = (1.0 / dm) ** power
    mantissa = mantissa * rm
    mantissa, me = np.frexp(mantissa)
    exponent = exponent - de.astype(np.int64) * power + me.astype(np.int64)
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(mantissa, exponent)


def _scale_safe_vector_norm(vectors):
    """
    Vectorized, scale-safe Euclidean norm along the last axis:
    sqrt(sum(vectors**2, axis=-1)), computed by factoring out each
    vector's own largest-magnitude component before squaring (see the
    module note above) so a representable norm is never lost to an
    intermediate component-squared overflow.
    """
    vectors = np.asarray(vectors, dtype=float)
    scale = np.max(np.abs(vectors), axis=-1)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    normalized = vectors / safe_scale[..., None]
    r = np.sqrt(np.sum(normalized * normalized, axis=-1)) * safe_scale
    return np.where(scale == 0.0, 0.0, r)


def _scale_safe_rms(values, axis=-1):
    """
    Vectorized, scale-safe root-mean-square: sqrt(mean(values**2,
    axis=axis)), computed by factoring out the largest magnitude along
    ``axis`` before squaring, so an RMS that is itself representable is
    never lost to an intermediate values**2 overflow (see
    position_space_divergence(), whose per-body distances can each be
    of an order whose SQUARE alone overflows float64 while the RMS
    across bodies does not).
    """
    values = np.abs(np.asarray(values, dtype=float))
    scale = np.max(values, axis=axis, keepdims=True)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    normalized = values / safe_scale
    rms = np.sqrt(np.mean(normalized * normalized, axis=axis)) * np.squeeze(
        safe_scale, axis=axis
    )
    return np.where(np.squeeze(scale, axis=axis) == 0.0, 0.0, rms)


def _scale_safe_sum(values, axis, keepdims=False):
    """
    Sum ``values`` along ``axis``, robust to a PARTIAL sum along the way
    overflowing to +-inf even when the true, fully combined sum is
    itself representable -- e.g. terms [+1e308, +1e308, -1e308] along
    one axis: the true sum, 1e308, is comfortably representable, but a
    naive sequential or pairwise reduction overflows the moment it adds
    the first two +1e308 terms together, before the compensating
    -1e308 term ever gets a chance to bring the running total back into
    range (np.sum's own reduction order is not something a caller
    controls). This is the aggregate-scale sibling of _scaled_product():
    that function fuses several individually-safe FACTORS of one term
    without ever rounding a partial product; this one fuses many
    individually-safe TERMS of one sum without ever forming a partial
    sum that could itself already be non-representable.

    Rescales every term by the largest-magnitude term along ``axis``
    first (bounding every scaled term in [-1, 1], so the running total
    during the reduction stays of order (size along axis) regardless of
    the terms' own absolute scale), sums the rescaled terms with
    Neumaier's (1974) improved Kahan compensated-summation algorithm
    (not a plain np.sum), then rescales back up in the single final
    multiplication -- the only place the true result's own magnitude
    can overflow, and only when the true sum itself is not
    representable.

    The rescale step alone, without Neumaier compensation, is
    range-safe but NOT cancellation-safe: for terms like [+1e308,
    +1e308, -1e308, -1e308, +1], rescaling by the largest magnitude
    (1e308) makes the true residual (1/1e308, about 1e-308) fall at the
    very bottom of a plain np.sum's rounding error, so np.sum silently
    keeps or drops it depending on the ORDER the terms happen to arrive
    in -- the same physical configuration, permuted, could then report
    two different centers of mass. Neumaier summation tracks a running
    compensation term (the low-order bits an ordinary running sum would
    otherwise discard) alongside the running total, so a small but
    genuine contribution like this is preserved regardless of
    summation order, making the result permutation-invariant to within
    ordinary floating-point rounding rather than losing real
    cancellation residue outright -- PROVIDED every term survives the
    rescale division in the first place. A term whose magnitude, though
    itself perfectly finite and nonzero, sits more than about 1074
    binary exponents below the largest term along this axis (e.g.
    1e-100 next to a 1e308 scale: 1e-100/1e308 = 1e-408, below the
    smallest representable subnormal, ~4.9e-324) is rounded away to
    exactly 0.0 by the rescale division itself, before Neumaier
    compensation ever runs on it -- compensation corrects ordinary
    rounding error introduced by the ADDITIONS, it cannot resurrect a
    term already annihilated upstream of them. Every reduction slice
    where this occurs is detected below and recomputed exactly instead
    (see the exact fallback), so this failure mode never reaches the
    returned result; every ordinary slice, where no term underflows the
    rescale, still uses the fast vectorized path above unchanged.
    """
    values = np.asarray(values, dtype=float)
    axis = axis % values.ndim
    scale = np.max(np.abs(values), axis=axis, keepdims=True)
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    scaled = values / safe_scale

    # A term that is itself finite and nonzero but rounds to exactly 0.0
    # once divided by this axis's own largest-magnitude term (see the
    # docstring above) must not be silently treated as if it had
    # genuinely been zero -- flag every reduction slice containing one so
    # it can be recomputed exactly below, instead of returning whatever
    # the fast path below happens to produce for it.
    underflowed_term = (scaled == 0.0) & (values != 0.0)
    needs_exact = np.any(underflowed_term, axis=axis)

    # Neumaier compensated summation along ``axis``, vectorized over every
    # other axis at once: move the reduction axis to the front, then walk
    # it one term at a time, maintaining a running total and a running
    # compensation array (both shaped like every OTHER axis combined).
    moved = np.moveaxis(scaled, axis, 0)
    total = np.zeros(moved.shape[1:], dtype=float)
    comp = np.zeros(moved.shape[1:], dtype=float)
    for k in range(moved.shape[0]):
        term = moved[k]
        new_total = total + term
        # Whichever of the running total or the new term has the larger
        # magnitude determines which one absorbs the rounding error of
        # this step's addition -- Neumaier's correction, generalizing
        # Kahan summation to remain accurate even when a single term
        # outweighs the running total (which plain Kahan summation does
        # not handle correctly).
        total_ge = np.abs(total) >= np.abs(term)
        comp = comp + np.where(
            total_ge, (total - new_total) + term, (term - new_total) + total
        )
        total = new_total
    scaled_sum = total + comp

    scale_squeezed = np.squeeze(scale, axis=axis)
    with np.errstate(over="ignore", invalid="ignore"):
        result = scaled_sum * scale_squeezed
    result = np.array(result, dtype=float)

    if np.any(needs_exact):
        # Exact fallback, applied only to the (rare, extreme-magnitude-
        # spread) reduction slices flagged above: sum the ORIGINAL,
        # unscaled terms as Python Fraction objects, which represent
        # every finite float64 value exactly and add without any
        # intermediate rounding or range limit at all, then convert the
        # single exact total back to float64 only once, at the very end
        # -- the only place this path can overflow, and only when the
        # true, fully-combined sum genuinely is not representable. See
        # _exact_axis_sum().
        moved_values = np.moveaxis(values, axis, 0)
        flat_values = moved_values.reshape(moved_values.shape[0], -1)
        flat_needs_exact = np.asarray(needs_exact).reshape(-1)
        flat_result = result.reshape(-1).copy()
        for flat_idx in np.nonzero(flat_needs_exact)[0]:
            flat_result[flat_idx] = _exact_axis_sum(flat_values[:, flat_idx])
        result = flat_result.reshape(result.shape)

    if keepdims:
        result = np.expand_dims(result, axis=axis)
    return result


def _exact_axis_sum(terms):
    """
    Exact sum of ``terms`` (an iterable of float64 values), via Python's
    arbitrary-precision Fraction. Used only as _scale_safe_sum()'s
    fallback for the rare reduction slice where its own rescale-by-
    largest-magnitude step would otherwise round a genuine, nonzero term
    to exactly 0.0 before Neumaier compensation ever sees it (see
    _scale_safe_sum's own docstring).

    Fraction represents every finite float64 value exactly (as a ratio of
    two arbitrary-precision integers) and sums without any intermediate
    rounding or range limit at all -- unlike a float64 accumulator, no
    partial total can ever overflow, underflow, or lose a small term to
    cancellation, no matter how extreme the individual terms' magnitudes
    are. Overflow is therefore possible only in the single final
    conversion back to float64, exactly when the true, fully-combined sum
    genuinely is not representable; that case is reported as a signed
    infinity here (matching how every other overflow in this module is
    reported) rather than letting a raw OverflowError escape from inside
    a reduction helper.
    """
    total = Fraction()
    for term in terms:
        total += Fraction(float(term))
    try:
        return float(total)
    except OverflowError:
        return math.inf if total > 0 else -math.inf


def _safe_distance_scalar(dx, dy, dz, eps2, eps):
    """
    Scale-safe sqrt(dx*dx + dy*dy + dz*dz + eps2) as plain Python floats.

    ``eps`` must equal sqrt(eps2) (the softening length, or 0.0 for an
    unsoftened separation); it is taken as a parameter, rather than
    recomputed here, because every caller already needs it once per
    call rather than once per pair.
    """
    scale = max(abs(dx), abs(dy), abs(dz), eps)
    if scale == 0.0:
        return 0.0
    rx, ry, rz, re = dx / scale, dy / scale, dz / scale, eps / scale
    return scale * math.sqrt(rx * rx + ry * ry + rz * rz + re * re)


def _safe_softened_r2_over_denom_scalar(dx, dy, dz, eps2, eps):
    """
    Scale-safe replacement for r2 / (r2 + eps2) ** 1.5, where
    r2 = dx**2 + dy**2 + dz**2 (the per-pair factor in
    virial_force_term()'s Wvir sum), used as a fallback once the naive
    r2 and/or (r2 + eps2) ** 1.5 has already overflowed to +inf -- which
    would otherwise silently drop this pair's contribution (or raise
    outright) even though the true ratio is comfortably representable:
    for r >> eps, r2/(r2+eps2)^1.5 -> 1/r, which stays small and
    well-behaved however large r itself is (e.g. a two-body
    configuration separated by ~1e160 with unit-scale masses, whose
    true Wvir is approximately -G, would otherwise be rejected outright
    by the naive formula above).

    Computes the raw (un-softened) separation r first via
    _safe_distance_scalar() (which never overflows, even when r itself
    is too large for r*r to be representable), THEN normalizes both r
    and eps by their own max before ever squaring either -- so r2 is
    never formed as a standalone value that could itself already have
    overflowed. ``eps`` must equal sqrt(eps2), as in
    _safe_distance_scalar().
    """
    r = _safe_distance_scalar(dx, dy, dz, 0.0, 0.0)
    s = max(r, eps)
    if s == 0.0:
        return 0.0
    x = (r / s) ** 2
    e = (eps / s) ** 2
    bracket = (x + e) ** 1.5
    if not (math.isfinite(bracket) and bracket > 0.0):
        raise ValueError(
            "a softened virial-force-term denominator is not "
            "representable even after scale-safe evaluation; check that "
            "positions and softening are physically reasonable."
        )
    ratio = (x / bracket) / s
    if not math.isfinite(ratio):
        raise ValueError(
            "a softened virial-force-term ratio overflowed even after "
            "scale-safe evaluation; check that positions and softening "
            "are physically reasonable."
        )
    return ratio


def _safe_pairwise_acceleration_term(dx, dy, dz, eps2, eps, mass):
    """
    Scale-safe replacement for

        G * mass * (dx, dy, dz) / (dx**2 + dy**2 + dz**2 + eps2) ** 1.5

    used as a fallback whenever the naive squared-separation above has
    either already overflowed to +inf, or stayed finite but produced a
    force coefficient that itself overflows or silently underflows to
    0 -- both would otherwise silently zero out or corrupt this pair's
    entire force contribution. Returns the (ax, ay, az) contribution,
    already scaled by G. Raises ValueError if the true softened
    acceleration is not itself representable as a finite float64 value.

    Every factor (G, mass, each displacement component, and the
    division by s**3) is combined in ONE fused _scaled_product_over()
    call per axis, rather than sequential divisions: a mass at or near
    the smallest representable positive float64 makes even G*mass
    ALONE underflow to exactly 0.0 before any division by s gets a
    chance to rescale it back into range, so no order of SEQUENTIAL
    two-factor-at-a-time multiplication/division can save every
    representable case -- only combining every factor before any
    intermediate is rounded can. _scaled_product_over() additionally
    avoids ever forming the standalone reciprocal 1/s (or 1/s**3) as
    its own float64 value, which matters when s itself is extremely
    small: that reciprocal can be non-representable even when the
    fully combined force is not (see _scaled_product_over()'s
    docstring).
    """
    s = _safe_distance_scalar(dx, dy, dz, eps2, eps)
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError(
            "a softened pairwise separation is not representable as a "
            "finite float64 distance; check that positions and "
            "softening are physically reasonable."
        )
    ax = float(_scaled_product_over(s, G, mass, dx, power=3))
    ay = float(_scaled_product_over(s, G, mass, dy, power=3))
    az = float(_scaled_product_over(s, G, mass, dz, power=3))
    if not (math.isfinite(ax) and math.isfinite(ay) and math.isfinite(az)):
        raise ValueError(
            "a pairwise gravitational force term overflowed even after "
            "scale-safe, fused evaluation; check that masses and "
            "positions are physically reasonable."
        )
    return ax, ay, az


# ======================================================================
# Softened Newtonian gravity
# ======================================================================
def athanassoula_softening(n_bodies, scale_radius):
    """
    Approximately optimal Plummer-softening length for an N-body
    realization of a smooth density profile with characteristic radius
    ``scale_radius``.

        eps_opt = 0.98 * a * N^(-0.26)

    This is the empirical scaling of Athanassoula, Fady, Lambert & Bosma
    (2000, MNRAS 314, 475), "Optimal softening for force calculations in
    collisionless N-body simulations" (building on the mean-square-force-
    error framework of Athanassoula et al. 1998), fit against simulated
    Plummer-model realizations. It is used here as a general order-of-
    magnitude default for any of this program's roughly-Plummer-like or
    roughly-uniform mass distributions, not only the exact Plummer
    profile it was fit to; treat it as a well-motivated starting point to
    experiment around, not a precise result for every initial condition
    this program can generate. The function is named after the citation
    above.
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=1)
    scale_radius = _require_positive("scale_radius", scale_radius)
    eps = 0.98 * scale_radius * n_bodies ** (-0.26)
    # For a sufficiently small (but individually finite and positive)
    # scale_radius, the product above can underflow all the way to
    # exactly 0.0 in float64 -- silently violating this function's own
    # documented positive-output contract instead of signaling that no
    # positive softening length can represent the true (infinitesimally
    # small) result. Rejected explicitly here rather than returned.
    if not (math.isfinite(eps) and eps > 0.0):
        raise ValueError(
            "the optimal softening length for n_bodies="
            f"{n_bodies} and scale_radius={scale_radius:g} underflows to "
            "zero in float64 and cannot be returned as a positive "
            "softening length; use a larger scale_radius (or fewer "
            "bodies)."
        )
    return eps


def compute_accelerations_direct(positions, masses, softening):
    """
    Direct O(N^2) softened Newtonian accelerations.

        a_i = G * sum_{j != i} m_j (x_j - x_i) / (|x_j - x_i|^2 + eps^2)^(3/2)

    The softening length eps regularizes the force at zero separation
    (unlike Multiple's exact point-mass law, which raises an error on
    coincident bodies): eps represents the finite extent smeared over each
    simulation particle, which typically stands in for many real stars or
    a patch of dark matter, not a literal point.

    Vectorized one source-loop at a time (O(N) memory per iteration rather
    than O(N^2)), so this remains usable up to MAX_BODIES without an
    excessive memory footprint; it is still O(N^2) in time; see
    DIRECT_METHOD_WARN_BODIES.
    """
    masses = _require_masses(masses)
    n = masses.size
    positions = _require_snapshot(positions, "positions", n_bodies=n)
    softening = _require_positive("softening", softening)
    # For a sufficiently large (but individually
    # finite) softening_pc, softening*softening can silently overflow to
    # inf. Unchecked, that makes every (r^2 + eps^2)^(-3/2) term evaluate
    # to exactly 0.0 -- finite, not nan or inf, so it would NOT be caught
    # by a downstream np.isfinite() postcondition check on the resulting
    # accelerations -- silently zeroing out all gravity while still
    # reporting a "successful" run. Caught explicitly here instead, at
    # the one place the overflow actually originates.
    eps2 = _require_positive("softening**2", softening * softening)
    # eps is the ORIGINAL softening length, not recomputed via
    # math.sqrt(eps2) -- squaring then unsquaring a softening length
    # whose square falls deep in the denormal range loses precision
    # that using the already-available original value avoids entirely.
    eps = softening
    # Halved copies of the softening length/its square, used ONLY by the
    # scale-safe fallback below (never by the fast path): halving any
    # finite float64 value can never overflow (it only decrements the
    # exponent), so these are always finite even when eps itself is at
    # the extreme end of the representable range.
    eps_half = 0.5 * eps
    eps2_half = eps_half * eps_half
    # Halved copies of every position, for the same reason: 0.5*x_j -
    # 0.5*x_i is always finite even for opposite-sign extreme-magnitude
    # coordinates whose true difference x_j - x_i would overflow (see the
    # fallback loop below). Precomputed once outside the body loop rather
    # than per-pair.
    positions_half = 0.5 * positions

    acc = np.zeros_like(positions)
    for i in range(n):
        # Opposite-sign extreme-magnitude coordinates (e.g. -1e308 and
        # +1e308) make this subtraction itself overflow to +-inf for the
        # affected rows even though the eventual softened force is
        # comfortably representable; that is expected and handled below
        # (via needs_fallback / the halved-displacement fallback), not a
        # bug to surface as a warning-turned-exception.
        with np.errstate(over="ignore"):
            d = positions - positions[i]           # (n, 3), points i -> all
        r2 = np.einsum("ij,ij->i", d, d) + eps2
        # A sufficiently large (but individually finite) separation can
        # overflow r2 to inf. Left unhandled, that would make that one
        # source's inv_r3 evaluate to exactly 0.0 -- finite, not nan/inf,
        # so a downstream np.isfinite(acc) postcondition would NOT catch
        # it if other sources for body i stayed normal -- silently
        # discarding a source's force even when the true softened force
        # from that source is itself comfortably representable (or a
        # genuine subnormal). Recomputed exactly for these rows below via
        # a scale-safe, sequential-division force law instead of being
        # silently dropped; see _safe_pairwise_acceleration_term.
        overflowed = ~np.isfinite(r2)
        # r2 is otherwise finite here, but a sufficiently small (though
        # still positive and finite) separation and/or softening can
        # make r2**(-1.5) ITSELF overflow
        # to inf even though r2 stayed perfectly representable -- a
        # near-coincident pair whose true softened force is comfortably
        # representable can still hit this, not only a genuinely
        # non-representable one. Detected here (not left to the final
        # isfinite(acc) postcondition) so it routes to the same
        # scale-safe fallback as an overflowed r2, rather than silently
        # corrupting this row's sum via an inf/0 or inf*0=nan term
        # mixed in with otherwise-normal contributions from other
        # sources.
        with np.errstate(over="ignore"):
            inv_r3 = r2 ** (-1.5)
        needs_fallback = overflowed | ~np.isfinite(inv_r3)
        # masses[j] and inv_r3[j] can each be individually finite and
        # nonzero yet still multiply to exactly 0.0 -- the true product
        # underflows below the smallest representable subnormal
        # (~4.9e-324) -- while the FULLY combined force (which still has
        # G and a potentially enormous displacement d left to fold in)
        # is comfortably representable. That silently reads exactly like
        # a legitimate zero mass unless caught here, before this row's
        # per-source coefficient is formed; routed to the same fused,
        # order-independent fallback as an overflowed r2/inv_r3 rather
        # than dropped.
        with np.errstate(over="ignore"):
            coeff = masses * inv_r3
        underflowed = (coeff == 0.0) & (masses != 0.0) & (inv_r3 != 0.0)
        # coeff itself can overflow to +inf even though r2 and inv_r3
        # were each individually finite -- an enormous mass times an
        # enormous inv_r3 (e.g. a very close, very massive source) whose
        # product cannot be represented, while the FULLY combined force
        # (which still has the displacement and the compensating G left
        # to fold in) is comfortably representable. Left unchecked, this
        # would either corrupt row_sum with a spurious +-inf term or, if
        # it happens to cancel, be silently wrong; routed to the same
        # scale-safe fallback as every other case here instead.
        overflowed_coeff = ~np.isfinite(coeff)
        needs_fallback = needs_fallback | underflowed | overflowed_coeff
        coeff[i] = 0.0                              # exclude self term
        if np.any(needs_fallback):
            coeff[needs_fallback] = 0.0
            # d itself can contain +-inf at exactly these rows (an
            # opposite-sign extreme-magnitude coordinate pair whose raw
            # x_j - x_i overflows even though the eventual softened force
            # is representable). Zeroing coeff alone is not enough to
            # exclude such a row from the vectorized sum below: 0.0 * inf
            # is nan, which would silently contaminate every other body's
            # otherwise-normal contribution to this row's total. d is
            # therefore zeroed at exactly the same rows; those rows'
            # actual contribution is added back below via the halved-
            # displacement fallback, which never needs the raw (possibly
            # non-representable) d.
            d = np.where(needs_fallback[:, None], 0.0, d)
        # G is folded into the per-source coefficient (coeff_g = G *
        # coeff) BEFORE it ever multiplies the displacement, not applied
        # to the completed sum afterward: forming coeff[:, None] * d
        # first and applying G only at the end leaves an unscaled
        # intermediate that can overflow even when every individual
        # source's fully-G-scaled contribution -- and the true total --
        # are both comfortably representable (e.g. ~20 sources whose
        # mass*inv_r3 coefficient is individually just under float64's
        # ceiling: G*coeff*d per source is a mundane ~1e296, but the sum
        # of the UNSCALED coeff*d terms alone already exceeds ~1.8e308).
        # Multiplying by G itself can never newly overflow here (|G| << 1
        # only ever shrinks a magnitude that was already finite, and every
        # entry of coeff is already finite at this point -- the overflowed/
        # underflowed entries were zeroed above), so this step never raises
        # a RuntimeWarning and needs no errstate guard of its own; the
        # np.errstate below covers the one multiply-then-sum that actually
        # can overflow.
        coeff_g = G * coeff
        # The ordinary vectorized reduction is tried first: for the
        # overwhelming majority of calls (no term anywhere near float64's
        # range limit) this is both correct and far cheaper than the
        # scale-safe fallback below, which pays a per-source Python-level
        # loop to stay correct at extreme magnitudes this common case
        # never approaches.
        with np.errstate(over="ignore", invalid="ignore"):
            row_sum = np.sum(coeff_g[:, None] * d, axis=0)
        if not np.all(np.isfinite(row_sum)):
            # Rare: even after folding G in per-source, an individual
            # coeff_g*d term, or a partial sum across several same-sign
            # sources, can still overflow before an opposite-sign source
            # brings the running total back into the representable range
            # the true total occupies. Recomputed via a fused per-term
            # product (coeff_g and d combined without ever rounding a
            # partial product) and a cancellation-/range-safe reduction
            # across sources -- see _scaled_product()/_scale_safe_sum()'s
            # own docstrings.
            terms = _scaled_product(coeff_g[:, None], d)
            row_sum = _scale_safe_sum(terms, axis=0)
        if np.any(needs_fallback):
            for j in np.nonzero(needs_fallback)[0]:
                if j == i:
                    continue
                # Displacement is recomputed here from HALVED copies of
                # the two positions (0.5*x_j - 0.5*x_i), not from d[j]
                # (which may be non-finite for exactly this row -- see
                # above). Halving each position first can never overflow
                # (it only decrements the exponent), and 0.5*x_j - 0.5*x_i
                # is always finite even when x_j - x_i itself is not, for
                # any two individually-finite float64 values. The result
                # fed into _safe_pairwise_acceleration_term (with eps and
                # eps2 halved to match) is therefore exactly 4x the true
                # acceleration -- s_half = s/2, so
                # G*mass*d_half/s_half**3 = G*mass*(d/2)/(s/2)**3 =
                # 4*G*mass*d/s**3 -- and dividing by 4 below recovers the
                # true, representable result even in cases where the raw
                # displacement itself could never have been formed.
                dx_half = positions_half[j, 0] - positions_half[i, 0]
                dy_half = positions_half[j, 1] - positions_half[i, 1]
                dz_half = positions_half[j, 2] - positions_half[i, 2]
                fx_h, fy_h, fz_h = _safe_pairwise_acceleration_term(
                    dx_half, dy_half, dz_half, eps2_half, eps_half, masses[j]
                )
                row_sum = row_sum + np.array([fx_h, fy_h, fz_h]) / 4.0
        acc[i] = row_sum

    if not np.all(np.isfinite(acc)):
        raise ValueError(
            "the direct-summation acceleration overflowed; check that "
            "positions and masses are physically reasonable."
        )
    return acc


class _OctreeNode:
    """
    One node of a Barnes-Hut octree. Internal to this module.

    Geometry (center, half_size, com) is stored as plain Python float
    tuples rather than NumPy arrays.  NumPy's per-call dispatch overhead,
    negligible when it amortizes over a large array, dominates when it is
    paid millions of times for arithmetic on 3-element vectors -- exactly
    the access pattern of a per-body tree walk -- so the walk below is
    plain-Python-float arithmetic throughout; see compute_accelerations_tree
    for the resulting cost, and the Help file's Algorithm section for the
    measured consequence (a pure-Python object tree's wall-clock crossover
    with vectorized direct summation sits far higher than the O(N log N)
    versus O(N^2) operation-count argument alone would suggest).
    """

    __slots__ = ("cx", "cy", "cz", "half_size", "mass", "comx", "comy",
                 "comz", "is_leaf", "indices", "children")


def _build_octree(positions, masses, indices, center, half_size, depth,
                   bucket_report):
    node = _OctreeNode()
    node.cx, node.cy, node.cz = center
    node.half_size = half_size
    sub_pos = positions[indices]
    sub_mass = masses[indices]
    # Both the total mass and the mass-weighted center of mass are
    # computed the same scale-safe way as center_of_mass() (see its own
    # docstring): normalizing by the largest mass before summing avoids
    # overflowing total_mass for many extreme-but-individually-finite
    # masses, and combining each mass/max_mass/position product in one
    # fused _scaled_product_over() call (rather than materializing
    # sub_mass[:, None] * sub_pos directly, or dividing by max_mass as
    # its own separate step) avoids overflowing -- or, for a max_mass
    # near the smallest representable positive float64, spuriously
    # producing nan from 0 times an intermediate +inf -- for a single
    # extreme mass-position pair even when the true center of mass is
    # representable.
    max_mass = float(np.max(sub_mass))
    weight_sum = float(np.sum(sub_mass / max_mass))
    total_mass = max_mass * weight_sum
    node.mass = total_mass
    numerator = _scaled_product_over(max_mass, sub_mass[:, None], sub_pos)
    # _scale_safe_sum, not a plain np.sum, for the same reason
    # center_of_mass() below needs it: the per-axis reduction across
    # bodies can overflow at a partial sum even when the true,
    # fully-combined numerator total is representable (e.g. two large
    # same-sign terms whose partial sum overflows before a third,
    # compensating opposite-sign term is added in).
    com = _scale_safe_sum(numerator, axis=0) / weight_sum
    node.comx, node.comy, node.comz = float(com[0]), float(com[1]), float(com[2])

    if depth >= MAX_TREE_DEPTH and len(indices) > 1:
        # This branch is reached only when MAX_TREE_DEPTH levels of octant
        # subdivision still could not separate every body into its own
        # cell -- i.e. two or more bodies are so close together (often
        # exactly or near-exactly coincident) that floating-point-precision
        # subdivision has run out of resolution, forcing this multi-body
        # "bucket" leaf. This is a warning, not an error: _node_acceleration()
        # still evaluates every body in a leaf INDIVIDUALLY (looping over
        # node.indices and summing each one's own softened pairwise force,
        # exactly as compute_accelerations_direct() would for the same
        # bodies -- see below), so no force detail is actually lost; what
        # is lost is the octree's usual O(log N) traversal benefit for
        # these particular bodies, and the warning exists so that a
        # genuine severe coordinate overlap does not otherwise pass with
        # no visible signal to a student debugging an extreme-parameter
        # run.
        #
        # Rather than calling warnings.warn() directly at every bucket
        # node -- which, for a dense cluster
        # with many separate small clumps of near-coincident bodies,
        # would fire once per clump, on every single integration step's
        # tree rebuild, and (since the message text embeds len(indices),
        # defeating Python's own default once-per-identical-message
        # warning dedup whenever the count differs between clumps or
        # steps) could genuinely spam stdout and slow the run -- this
        # only RECORDS the occurrence in ``bucket_report`` here.
        # build_octree() below issues at most ONE consolidated warning
        # per tree build (i.e. at most once per integration step),
        # summarizing every bucket node from that build together.
        bucket_report.append(len(indices))

    if len(indices) <= 1 or depth >= MAX_TREE_DEPTH:
        node.is_leaf = True
        node.indices = tuple(indices.tolist())
        node.children = None
        return node

    node.is_leaf = False
    node.indices = ()
    node.children = [None] * 8
    rel = sub_pos - np.asarray(center)
    # Octant bit pattern: bit0=x>=0, bit1=y>=0, bit2=z>=0.
    octant = ((rel[:, 0] >= 0.0).astype(np.int64)
              | ((rel[:, 1] >= 0.0).astype(np.int64) << 1)
              | ((rel[:, 2] >= 0.0).astype(np.int64) << 2))
    child_half = 0.5 * half_size
    for k in range(8):
        sub_idx = indices[octant == k]
        if sub_idx.size == 0:
            continue
        child_center = (
            center[0] + (child_half if (k & 1) else -child_half),
            center[1] + (child_half if (k & 2) else -child_half),
            center[2] + (child_half if (k & 4) else -child_half),
        )
        node.children[k] = _build_octree(
            positions, masses, sub_idx, child_center, child_half, depth + 1,
            bucket_report,
        )
    return node


def build_octree(positions, masses):
    """
    Build a Barnes-Hut octree over the given bodies and return its root.

    The bounding cube is centered on the particle-set's own geometric
    midpoint and sized to just enclose every body, with a small margin
    (0.1%) so that a body exactly on the computed boundary is still
    strictly inside its cell.  If every body occupies (numerically) the
    same point, the box half-size is floored to a small positive number
    rather than left at zero, so the tree remains well-defined; the
    physical force in that degenerate case is finite and small because
    the softened numerator (x_j - x_i) also vanishes between coincident
    bodies (see compute_accelerations_direct).
    """
    masses = _require_masses(masses)
    n = masses.size
    positions = _require_snapshot(positions, "positions", n_bodies=n)
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    # Halve lo and hi SEPARATELY before combining, both for the midpoint
    # and for the half-extent, rather than forming lo+hi or hi-lo as a
    # standalone value first: for same-sign extreme coordinates (e.g.
    # lo=8e307, hi=1e308) lo+hi can overflow even though the true
    # midpoint is representable, and for opposite-sign extreme
    # coordinates (e.g. lo=-1e308, hi=1e308) hi-lo can overflow even
    # though the true half-extent (1e308) is representable. This is a
    # coordinate-only special case of the same "no order of sequential
    # combination saves every representable case" issue documented at
    # _scaled_product()'s module note.
    center = tuple((0.5 * lo + 0.5 * hi).tolist())
    half_size = float(np.max(0.5 * hi - 0.5 * lo)) * 1.001
    if not (np.all(np.isfinite(center)) and math.isfinite(half_size)):
        raise ValueError(
            "the octree bounding cube overflowed; check that positions "
            "are physically reasonable."
        )
    if not (half_size > 0.0):
        half_size = 1.0
    bucket_report = []
    root = _build_octree(positions, masses, np.arange(n), center, half_size,
                          0, bucket_report)
    if bucket_report:
        # ONE consolidated warning per tree build (i.e. per integration
        # step, not per bucket node/clump within it) -- see
        # _build_octree()'s matching comment. Summarizes every bucket
        # node from this build: how many separate nodes needed
        # bucketing, how many bodies in total they contained, and the
        # single LARGEST clump's own size -- the total alone cannot
        # distinguish one pathological clump of, say, 100 bodies from
        # ten independent 10-body clumps, which have very different
        # implications for a student debugging a run (one severe
        # coordinate overlap versus many small, likely-benign ones).
        n_buckets = len(bucket_report)
        n_bucketed_bodies = sum(bucket_report)
        max_clump_size = max(bucket_report)
        warnings.warn(
            f"build_octree: {n_bucketed_bodies} bodies (in {n_buckets} "
            f"separate 'bucket' node{'s' if n_buckets != 1 else ''}, "
            f"largest {max_clump_size} bodies) could "
            f"not be separated after {MAX_TREE_DEPTH} levels of octree "
            "subdivision (they are at or extremely near the same "
            "position); the tree walk still evaluates each of their "
            "forces individually and exactly (the same softened pairwise "
            "force direct summation would give), not as a combined "
            "monopole, so no force accuracy is lost -- only the usual "
            "octree traversal speedup for these particular bodies.",
            RuntimeWarning,
            stacklevel=2,
        )
    return root


def _fast_pairwise_coeff(mass, r2):
    """
    Fast-path scalar force coefficient ``G * mass * r2 ** -1.5`` used by
    the tree walk's leaf and monopole terms, returning ``math.nan``
    whenever the result cannot be trusted as the fully combined,
    representable value -- the caller then routes to the fused,
    scale-safe fallback (_safe_pairwise_acceleration_term) instead of
    silently zeroing out or corrupting this pair's/node's contribution.
    Three distinct failure modes are all folded into the same nan
    signal: r2 itself overflowing to +inf; r2 ** -1.5 overflowing
    (raised by C's pow() as a raw OverflowError rather than saturating
    to inf); and mass * r2 ** -1.5 underflowing to exactly 0.0 even
    though mass and r2 ** -1.5 are each individually finite and
    nonzero. That last case is not hypothetical: a mass and an inv_r3
    that are each comfortably representable on their own can still
    multiply (in either order, including with G folded in first) to
    something below the smallest representable subnormal
    (~4.9e-324) -- while the FULLY combined force, which still has the
    displacement components left to fold in, is itself representable.
    """
    if not math.isfinite(r2):
        return math.nan
    try:
        inv_r3 = r2 ** -1.5
    except OverflowError:
        return math.nan
    if not math.isfinite(inv_r3):
        return math.nan
    coeff = G * mass * inv_r3
    if coeff == 0.0 and mass != 0.0 and inv_r3 != 0.0:
        return math.nan
    return coeff


def _node_acceleration(root, i, pos_list, mass_list, theta, eps2, softening):
    """
    Iterative (explicit-stack) tree walk for one body.

    Written with an explicit stack of nodes, rather than recursion, and
    with plain Python float arithmetic on 3-tuples rather than NumPy
    3-vectors, purely for wall-clock speed: see _OctreeNode's docstring.
    The algorithm is exactly the recursive Barnes-Hut walk described in
    compute_accelerations_tree's docstring; only the implementation is
    optimized.

    ``softening`` (the ORIGINAL, un-squared softening length) is taken as
    its own parameter, separate from ``eps2``, rather than recovered via
    math.sqrt(eps2): squaring then unsquaring a softening length whose
    square falls deep in the denormal range loses precision that using
    the already-available original value avoids entirely.
    """
    xi, yi, zi = pos_list[i]
    eps = softening
    # Halved copies of softening and body i's own position, used ONLY by
    # the scale-safe fallback branches below: halving any finite float64
    # value can never overflow (it only decrements the exponent), so
    # these stay finite even at the extreme end of the representable
    # range. See compute_accelerations_direct's matching comment -- the
    # tree walk needs the identical halving trick for the identical
    # reason (an opposite-sign extreme-magnitude pair whose raw
    # difference overflows even though the final softened force is
    # representable).
    eps_half = 0.5 * eps
    eps2_half = eps_half * eps_half
    xi_half, yi_half, zi_half = 0.5 * xi, 0.5 * yi, 0.5 * zi
    ax = ay = az = 0.0
    stack = [root]
    theta2 = theta * theta
    while stack:
        node = stack.pop()
        if node is None or node.mass == 0.0:
            continue
        if node.is_leaf:
            for j in node.indices:
                if j == i:
                    continue
                xj, yj, zj = pos_list[j]
                # xj - xi (etc.) can silently overflow to +-inf here for
                # an opposite-sign extreme-magnitude pair; unlike NumPy,
                # plain Python float arithmetic does not warn when this
                # happens, so no errstate suppression is needed, but dx/
                # dy/dz themselves must never be used below once that
                # happens -- only r2's resulting non-finiteness (routing
                # to the halved-displacement fallback) is relied upon.
                dx, dy, dz = xj - xi, yj - yi, zj - zi
                r2 = dx * dx + dy * dy + dz * dz + eps2
                # G is applied inside _fast_pairwise_coeff so that the
                # fast and scale-safe fallback paths share the same
                # units; see _safe_pairwise_acceleration_term's docstring
                # for why the fallback needs G folded in before its
                # divisions. _fast_pairwise_coeff returns nan (routing to
                # that fallback) for every case where the fast, un-fused
                # G*mass*r2**-1.5 computation cannot be trusted: r2
                # overflowing, r2**-1.5 overflowing, or mass*r2**-1.5
                # underflowing to exactly 0.0 despite both factors being
                # individually finite and nonzero.
                coeff = _fast_pairwise_coeff(mass_list[j], r2)
                if math.isfinite(coeff):
                    ax += coeff * dx
                    ay += coeff * dy
                    az += coeff * dz
                else:
                    # Recomputed from HALVED copies of both positions,
                    # not from dx/dy/dz (which may themselves be
                    # non-finite -- see above); see
                    # compute_accelerations_direct's matching fallback
                    # for the full derivation of the factor of 4.
                    dx_half = 0.5 * xj - xi_half
                    dy_half = 0.5 * yj - yi_half
                    dz_half = 0.5 * zj - zi_half
                    fx_h, fy_h, fz_h = _safe_pairwise_acceleration_term(
                        dx_half, dy_half, dz_half, eps2_half, eps_half,
                        mass_list[j],
                    )
                    ax += fx_h / 4.0
                    ay += fy_h / 4.0
                    az += fz_h / 4.0
            continue

        # node.comx - xi (etc.) can silently overflow to +-inf for an
        # opposite-sign extreme-magnitude pair, exactly as in the leaf
        # loop above; dx/dy/dz are then only used for dist2/the fast
        # path below, never for the fallback (which recomputes from
        # halved copies instead -- see below).
        dx, dy, dz = node.comx - xi, node.comy - yi, node.comz - zi
        dist2 = dx * dx + dy * dy + dz * dz
        size = 2.0 * node.half_size
        # Self-force containment: an internal node containing body i's own
        # position must never be accepted as a monopole, however small its
        # opening angle appears -- accepting it would fold body i's own
        # mass and position into the node's mass/center-of-mass and apply
        # a spurious self-force (in the direct-summation path, the
        # equivalent i==j term is excluded explicitly by index instead;
        # see compute_accelerations_direct()). Node bounds are exact
        # octree-construction cubes (see _build_octree), so a simple
        # axis-aligned containment test against this node's own GEOMETRIC
        # center (cx, cy, cz -- NOT its mass-weighted comx/comy/comz,
        # which need not lie inside the cube at all) and half_size is
        # exact, not approximate; a contained body forces descent into the
        # children regardless of theta -- an adversarial configuration
        # that relies on this (a body placed so a naive opening-angle-only
        # check would otherwise accept a node containing it) would
        # otherwise produce a large, spurious relative acceleration error
        # at theta in [0.7, 1.0].
        contains_i = (abs(node.cx - xi) <= node.half_size
                      and abs(node.cy - yi) <= node.half_size
                      and abs(node.cz - zi) <= node.half_size)
        # dist2 == 0 forces recursion rather than division by zero: it can
        # only occur when this body's own position coincides exactly with
        # the node's mass-weighted center, which is not a legitimate case
        # for the monopole approximation regardless of theta.
        if dist2 > 0.0 and not contains_i and (size * size) < theta2 * dist2:
            r2 = dist2 + eps2
            # Same fast/safe split as the leaf-node loop above, via the
            # same helper -- here for a monopole's total mass and
            # separation rather than a single body's.
            coeff = _fast_pairwise_coeff(node.mass, r2)
            if math.isfinite(coeff):
                ax += coeff * dx
                ay += coeff * dy
                az += coeff * dz
            else:
                # Recomputed from HALVED copies of both positions, not
                # from dx/dy/dz (which may themselves be non-finite --
                # see above); see compute_accelerations_direct's matching
                # fallback for the full derivation of the factor of 4.
                dx_half = 0.5 * node.comx - xi_half
                dy_half = 0.5 * node.comy - yi_half
                dz_half = 0.5 * node.comz - zi_half
                fx_h, fy_h, fz_h = _safe_pairwise_acceleration_term(
                    dx_half, dy_half, dz_half, eps2_half, eps_half,
                    node.mass,
                )
                ax += fx_h / 4.0
                ay += fy_h / 4.0
                az += fz_h / 4.0
        else:
            stack.extend(node.children)
    return ax, ay, az


def compute_accelerations_tree(positions, masses, theta, softening):
    """
    Barnes-Hut tree accelerations, O(N log N) for a well-distributed set of
    bodies (Barnes & Hut 1986, Nature 324, 446).

    For each body, the octree is walked from the root; an internal node is
    treated as a single point mass at its center of mass whenever its
    angular size as seen from the body, ``size / distance``, is smaller
    than the opening angle ``theta``; otherwise the walk recurses into that
    node's children.  theta = 0 forces a full descent to the leaves for
    every body (equivalent to, but slower than, direct summation) and is
    useful as a correctness check; theta of order 0.5-0.8 is the usual
    working range, trading force accuracy for speed.
    """
    # _require_masses()/_require_snapshot(), the same shared validators
    # compute_accelerations_direct() already uses, not a separate
    # _as_finite_array() + ad hoc positivity/shape
    # check -- a bare scalar mass, a mismatched masses length, or a
    # wrong-shaped positions array now gets the identical error wording
    # regardless of which force method (direct or tree) was requested,
    # rather than two different validation contracts for what is meant
    # to be an interchangeable choice (see compute_accelerations()).
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    theta = _require_theta(theta)
    softening = _require_positive("softening", softening)
    # For a sufficiently large (but individually
    # finite) softening_pc, softening*softening can silently overflow to
    # inf. Unchecked, that makes every (r^2 + eps^2)^(-3/2) term evaluate
    # to exactly 0.0 -- finite, not nan or inf, so it would NOT be caught
    # by a downstream np.isfinite() postcondition check on the resulting
    # accelerations -- silently zeroing out all gravity while still
    # reporting a "successful" run. Caught explicitly here instead, at
    # the one place the overflow actually originates.
    eps2 = _require_positive("softening**2", softening * softening)

    root = build_octree(positions, masses)
    n = masses.size
    pos_list = positions.tolist()
    mass_list = masses.tolist()
    acc = np.empty_like(positions)
    for i in range(n):
        acc[i] = _node_acceleration(root, i, pos_list, mass_list, theta, eps2, softening)

    if not np.all(np.isfinite(acc)):
        raise ValueError(
            "the tree-summation acceleration overflowed; check that "
            "positions and masses are physically reasonable."
        )
    return acc


def compute_accelerations(positions, masses, softening, method="tree", theta=0.5):
    """Dispatch to the tree or direct force evaluator."""
    method = _require_method(method)
    if method == "direct":
        return compute_accelerations_direct(positions, masses, softening)
    return compute_accelerations_tree(positions, masses, theta, softening)


# ======================================================================
# Energy, momentum and virial diagnostics
#
# Two distinct scalars are computed from the pairwise separations here, and
# must not be conflated: potential_energy() (U) is the right quantity for
# total-energy bookkeeping (E = T + U, conserved by the integrator up to
# numerical error), but it is NOT the quantity the scalar virial theorem
# uses once gravity is softened. virial_force_term() (Wvir) is that
# quantity; see its docstring. Use potential_energy() for energy
# conservation and virial_force_term() for virial-ratio/virial-balance
# diagnostics -- never the other way around.
# ======================================================================
def kinetic_energy(velocities, masses):
    masses = _require_masses(masses)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=masses.size)
    # 0.5 * mass * v_component * v_component is combined in ONE fused
    # _scaled_product() call per component, rather than computed as
    # (0.5*mass*v_component)*v_component or any other sequential
    # two-factor-at-a-time order: a mass at or near the smallest
    # representable positive float64 makes even 0.5*mass ALONE underflow
    # to exactly 0.0 before any velocity factor gets a chance to rescale
    # it back into range (e.g. velocities [[1e160,0,0],[0,0,0]], masses
    # [nextafter(0,1), 1] would otherwise silently return 0.0 instead of
    # the true, representable 2.470e-4 J), so no SEQUENTIAL reordering
    # can rescue every representable case -- only combining every factor
    # before any intermediate is rounded can; see _scaled_product()'s
    # own docstring. This does not make kinetic
    # energy immune to overflow -- a genuinely non-representable result
    # (e.g. a non-negligible mass at v of order 1e200) still overflows,
    # and is reported below as a clear ValueError rather than a silent
    # inf.
    with np.errstate(over="ignore"):
        per_component = _scaled_product(0.5, masses[:, None], velocities, velocities)
        per_body = np.sum(per_component, axis=1)
    if not np.all(np.isfinite(per_body)):
        raise ValueError(
            "kinetic energy overflowed for at least one body; check that "
            "velocities and masses are physically reasonable."
        )
    total = float(np.sum(per_body))
    if not math.isfinite(total):
        raise ValueError("kinetic energy overflowed.")
    return total


def _com_relative_kinetic_energy(velocities, masses):
    """
    COM-relative ("internal") kinetic energy,
    sum_i (1/2) m_i |v_i - v_COM|^2, used wherever the virial-ratio
    numerator specifically needs internal (not lab-frame) kinetic energy
    -- see integrate_nbody()'s _record()/_record_dense() closures.

    Never materializes the raw difference v_i - v_COM: two individually
    finite, opposite-sign velocities can subtract to something that
    overflows float64 even when the resulting COM-relative kinetic
    energy is itself perfectly representable (a small enough body mass
    keeps the internal energy finite even when the lab-frame velocities
    involved are each near float64's own ceiling). Instead, forms
    h_i = 0.5*v_i - 0.5*v_COM per component -- finite for ANY two
    individually finite float64 endpoints, since halving each one first
    bounds the magnitude of their difference by float64's own largest
    representable value, exactly the halving trick this module's
    pairwise-force fallbacks already use for the analogous displacement
    hazard -- and uses v_i - v_COM = 2*h_i, so
    (1/2)*m_i*|v_i - v_COM|^2 = (1/2)*m_i*(2*h_i)^2 = 2*m_i*h_i^2,
    fusing that factor of 2 into the same per-component _scaled_product()
    call kinetic_energy() itself uses (with 0.5 in place of 2.0 there),
    rather than ever reconstructing v_i - v_COM (= 2*h_i) as its own
    float64 value.
    """
    masses = _require_masses(masses)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=masses.size)
    vel_com = center_of_mass_velocity(velocities, masses)
    h = 0.5 * velocities - 0.5 * vel_com
    with np.errstate(over="ignore"):
        per_component = _scaled_product(2.0, masses[:, None], h, h)
        per_body = np.sum(per_component, axis=1)
    if not np.all(np.isfinite(per_body)):
        raise ValueError(
            "COM-relative kinetic energy overflowed for at least one "
            "body; check that velocities and masses are physically "
            "reasonable."
        )
    total = float(np.sum(per_body))
    if not math.isfinite(total):
        raise ValueError("COM-relative kinetic energy overflowed.")
    return total


def potential_energy(positions, masses, softening):
    """
    Total softened potential energy, U = -sum_{i<j} G m_i m_j / sqrt(r_ij^2 + eps^2).

    Direct O(N^2) summation. Used only for periodic diagnostics (not once
    per integration step), so its cost is independent of which force
    method (tree or direct) drives the actual time integration.

    This is the correct quantity for total-energy bookkeeping (E = T + U).
    It is NOT, by itself, the virial-theorem quantity once softening is
    nonzero -- see virial_force_term().
    """
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    softening = _require_positive("softening", softening)
    # For a sufficiently large (but individually
    # finite) softening_pc, softening*softening can silently overflow to
    # inf. Unchecked, that makes every (r^2 + eps^2)^(-3/2) term evaluate
    # to exactly 0.0 -- finite, not nan or inf, so it would NOT be caught
    # by a downstream np.isfinite() postcondition check on the resulting
    # accelerations -- silently zeroing out all gravity while still
    # reporting a "successful" run. Caught explicitly here instead, at
    # the one place the overflow actually originates.
    eps2 = _require_positive("softening**2", softening * softening)
    # eps is the ORIGINAL softening length, not recomputed via
    # math.sqrt(eps2) -- squaring then unsquaring a softening length
    # whose square falls deep in the denormal range loses precision
    # that using the already-available original value avoids entirely.
    eps = softening
    n = masses.size
    u = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        r2 = np.einsum("ij,ij->i", d, d) + eps2
        # r2 can overflow to +inf for a sufficiently large (but
        # individually finite) separation, which would otherwise make
        # that pair's contribution to r silently become +inf too --
        # and dividing by +inf below discards a real, and possibly
        # dominant, potential-energy contribution as exactly 0.0 rather
        # than the true (representable) value.
        overflowed = ~np.isfinite(r2)
        with np.errstate(over="ignore"):
            r = np.sqrt(r2)
        if np.any(overflowed):
            for k in np.nonzero(overflowed)[0]:
                r[k] = _safe_distance_scalar(d[k, 0], d[k, 1], d[k, 2], eps2, eps)
        if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
            raise ValueError(
                "a pair separation in the potential-energy sum is not "
                "representable as a finite, positive float64 distance; "
                "check that positions and softening are physically "
                "reasonable."
            )
        # Each pair's full G*m_i*m_j/r contribution is combined in ONE
        # fused _scaled_product_over() call, rather than computed as
        # G*m_i*(m_j/r): a sufficiently large m_j and small r can make
        # m_j/r ALONE overflow to +inf even though the fully combined
        # term is comfortably representable once a sufficiently small
        # compensating m_i is folded in. _scaled_product_over() also
        # avoids ever forming the standalone reciprocal 1/r, which
        # matters when r itself is an extremely small (but still
        # finite and positive) separation: that reciprocal alone can
        # already exceed float64's range even though the fully
        # combined energy term does not.
        pair_terms = _scaled_product_over(r, G, masses[i], masses[i + 1:])
        if not np.all(np.isfinite(pair_terms)):
            raise ValueError(
                "potential energy overflowed for at least one pair; "
                "check that positions and masses are physically "
                "reasonable."
            )
        u -= np.sum(pair_terms)
    if not math.isfinite(u):
        raise ValueError("potential energy overflowed.")
    return float(u)


def virial_force_term(positions, masses, softening):
    """
    The scalar virial-theorem quantity for Plummer-softened gravity,

        Wvir = sum_i r_i . F_i
             = -sum_{i<j} G m_i m_j r_ij^2 / (r_ij^2 + eps^2)^(3/2),

    where F_i is the total softened force on body i.  This is NOT the same
    number as potential_energy() (U) once eps > 0: the softened force is not
    homogeneous of degree -1 in separation, so U (which pairs with 1/r) is
    not the quantity the virial theorem actually constrains.  As eps -> 0,
    Wvir -> U exactly, since the r_ij^2/(r_ij^2+eps^2) factor -> 1.

    Using 2T/|U| (potential_energy) as a scalar-virial-balance diagnostic
    for the softened equations actually being integrated is a real, and
    non-negligible, error: for eps/a of order this program's own default
    softening (roughly 0.2-0.25 for the default cluster N), the two
    ratios differ by several percent to tens of percent (measured:
    2T/|U| = 1.298 vs 2T/|Wvir| = 1.380 for one N=200 Plummer realization
    at the default softening). virial_ratio() itself is agnostic to which
    W is supplied (it is simply 2T/|W|); every CALLER that wants a
    scalar-virial-balance diagnostic for the softened force must pass
    virial_force_term(...), not potential_energy(...), and every caller
    that wants total energy must use potential_energy(...) instead. See
    run_cluster()/run_galaxy() and the Help file's Governing Equations
    section.

    NOTE ON WHAT THIS DIAGNOSTIC DOES AND DOES NOT ESTABLISH: a value of
    2T/|Wvir| close to 1 shows only that the system satisfies the scalar
    virial theorem AT THAT INSTANT -- a single global constraint on the
    two scalars T and Wvir. It is not evidence, by itself, that the
    system is in a genuine dynamical (phase-space) equilibrium, i.e. a
    stationary solution of the collisionless Boltzmann equation; a system
    can be constructed to have 2T/|Wvir| = 1 at t=0 by a single global
    velocity rescale while still being far from phase-space stationarity,
    and will generally evolve away from strict scalar balance as it
    relaxes even while approaching a more nearly stationary state. See
    plummer_sphere(), uniform_sphere() and run_cluster() for where this
    distinction matters to how their results should be described.
    """
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    softening = _require_positive("softening", softening)
    # For a sufficiently large (but individually
    # finite) softening_pc, softening*softening can silently overflow to
    # inf. Unchecked, that makes every (r^2 + eps^2)^(-3/2) term evaluate
    # to exactly 0.0 -- finite, not nan or inf, so it would NOT be caught
    # by a downstream np.isfinite() postcondition check on the resulting
    # accelerations -- silently zeroing out all gravity while still
    # reporting a "successful" run. Caught explicitly here instead, at
    # the one place the overflow actually originates.
    eps2 = _require_positive("softening**2", softening * softening)
    eps = softening
    n = masses.size
    w = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        with np.errstate(over="ignore"):
            r2 = np.einsum("ij,ij->i", d, d)
        # r2 itself can overflow to +inf for a sufficiently large (but
        # individually finite) separation -- not only (r2+eps2)**1.5,
        # which can additionally overflow even when r2 alone stays
        # finite. Both are routed to the same scale-safe per-pair
        # fallback below rather than raised outright: for sufficiently
        # extreme (but each individually finite) position magnitudes,
        # the true r2/denom ratio -- and hence the true, fully combined
        # G*m_i*m_j*r2/denom term -- is comfortably representable (e.g.
        # the same two-body large-separation configuration described
        # above, whose true Wvir is approximately -G, would otherwise be
        # rejected outright for exactly this reason).
        with np.errstate(over="ignore"):
            denom = (r2 + eps2) ** 1.5
        overflowed = ~np.isfinite(r2) | ~np.isfinite(denom)
        ratio = np.where(overflowed, 1.0, r2 / np.where(overflowed, 1.0, denom))
        if np.any(overflowed):
            for k in np.nonzero(overflowed)[0]:
                ratio[k] = _safe_softened_r2_over_denom_scalar(
                    d[k, 0], d[k, 1], d[k, 2], eps2, eps
                )
        if not np.all(np.isfinite(ratio)):
            raise ValueError(
                "virial force term overflowed (a pair separation was too "
                "large for the softened force-law denominator to remain "
                "finite); check that positions are physically reasonable."
            )
        # Each pair's full G*m_i*m_j*ratio contribution is combined in
        # ONE fused _scaled_product() call, for the same reason as
        # potential_energy() above: masses[i+1:]*ratio alone can
        # overflow or underflow before G and the compensating mass get a
        # chance to rescale it back into range.
        pair_terms = _scaled_product(G, masses[i], masses[i + 1:], ratio)
        if not np.all(np.isfinite(pair_terms)):
            raise ValueError(
                "virial force term overflowed for at least one pair; "
                "check that positions and masses are physically "
                "reasonable."
            )
        w -= np.sum(pair_terms)
    if not math.isfinite(w):
        raise ValueError("virial force term overflowed.")
    return float(w)


def total_energy(positions, velocities, masses, softening):
    return kinetic_energy(velocities, masses) + potential_energy(
        positions, masses, softening
    )


def virial_ratio(kinetic, work_term):
    """2T / |W|, given the kinetic energy T and a virial-theorem work term
    W. 1.0 marks exact instantaneous scalar virial balance (2T = |W|),
    not necessarily a genuine dynamical equilibrium -- see
    virial_force_term()'s docstring for that distinction. For
    Plummer-softened gravity,
    W must be virial_force_term(...), NOT potential_energy(...) -- the two
    differ once softening is nonzero; see virial_force_term()'s docstring.
    Returns nan if work_term is exactly zero (only possible for a
    vanishing or fully unbound system)."""
    if work_term == 0.0:
        return float("nan")
    # Divide before multiplying by 2, not after: virial_ratio(1e308,
    # -1e308) has a true value of exactly 2 (kinetic and |work_term| are
    # equal), but 2.0*kinetic overflows to inf before the division ever
    # runs, silently returning inf instead of 2.
    # kinetic/abs(work_term) is bounded near 1 here and can only
    # overflow when the true ratio itself is not representable.
    return 2.0 * (kinetic / abs(work_term))


def center_of_mass(positions, masses):
    """
    Mass-weighted mean position, sum_i m_i x_i / sum_i m_i.

    ``masses`` must have one strictly positive entry per body, the same
    contract every other physical helper in this module enforces (a
    negative individual mass is not a meaningful input for a *mass*-
    weighted mean, even when it happens not to make the total mass
    non-positive; see _require_masses()).
    """
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    # sum(masses) can overflow to +inf for individually finite masses
    # (e.g. three bodies of 1e308 kg each) even though the true
    # mass-weighted mean position is comfortably representable -- and
    # dividing by that spurious +inf would silently produce [0, 0, 0]
    # instead. Normalizing every mass by the largest one first keeps
    # every weight in (0, 1] and their sum safely of order N, regardless
    # of the masses' absolute scale; the normalization cancels exactly
    # in the final ratio.
    max_mass = float(np.max(masses))
    weights = masses / max_mass
    weight_sum = np.sum(weights)
    # The NUMERATOR sum_i weight_i * position_i is combined as ONE fused
    # _scaled_product_over() call across (masses, positions, divided by
    # max_mass), rather than materializing ``weights`` (= masses/max_mass)
    # as its own rounded array first: for a mass ratio wide enough (e.g.
    # masses [1e308, 1e-100]), an individual weight_i can underflow to
    # exactly 0.0 on its own even though that same body's
    # weight_i * position_i is comfortably representable once a
    # sufficiently large compensating position is folded in (max_mass
    # cancels exactly between this numerator and weight_sum below, so
    # weight_sum itself can safely keep using the separately materialized
    # ``weights`` -- losing a term there only when it is already many
    # orders of magnitude smaller than weight_sum's own rounding error,
    # which changes nothing). _scaled_product_over() additionally avoids
    # ever forming the standalone reciprocal 1/max_mass, which matters
    # when max_mass itself is extremely small (see its own docstring).
    numerator = _scaled_product_over(max_mass, masses[:, None], positions)
    # _scale_safe_sum, not a plain np.sum, over the per-body axis: a
    # partial sum across bodies can overflow even when the true,
    # cancellation-involving total is representable (e.g. positions
    # [+1e308, +1e308, -1e308] with equal masses -- the true COM,
    # 1e308/3, is comfortably representable, but naive reduction
    # overflows at the first two same-sign terms before the third,
    # compensating term is folded in).
    com = _scale_safe_sum(numerator, axis=0) / weight_sum
    if not np.all(np.isfinite(com)):
        raise ValueError(
            "center of mass overflowed; check that positions and masses "
            "are physically reasonable."
        )
    return com


def center_of_mass_velocity(velocities, masses):
    """Mass-weighted mean velocity; see center_of_mass()'s docstring for
    the masses contract this shares."""
    masses = _require_masses(masses)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=masses.size)
    # See center_of_mass() for why the masses are normalized by their
    # maximum before summing, and why the weighted-sum numerator is
    # combined as one fused _scaled_product_over() call rather than
    # materializing the per-body weights on their own first.
    max_mass = float(np.max(masses))
    weights = masses / max_mass
    weight_sum = np.sum(weights)
    numerator = _scaled_product_over(max_mass, masses[:, None], velocities)
    # See center_of_mass() for why this reduction must be
    # _scale_safe_sum rather than a plain np.sum.
    com_v = _scale_safe_sum(numerator, axis=0) / weight_sum
    if not np.all(np.isfinite(com_v)):
        raise ValueError(
            "center-of-mass velocity overflowed; check that velocities "
            "and masses are physically reasonable."
        )
    return com_v


def recenter(positions, velocities, masses):
    """Return (positions, velocities) shifted to the center-of-mass frame."""
    com_x = center_of_mass(positions, masses)
    com_v = center_of_mass_velocity(velocities, masses)
    return positions - com_x, velocities - com_v


def lagrangian_radii(positions, masses, fractions, center=None):
    """
    Radii enclosing the requested cumulative mass fractions, measured from
    ``center`` (default: the mass-weighted center of the given positions).

    Returns a dict {fraction: radius}. Fractions must lie in (0, 1].
    """
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    fractions = [_require_finite("fraction", f) for f in fractions]
    for f in fractions:
        if not (0.0 < f <= 1.0):
            raise ValueError(f"lagrangian-radius fractions must lie in (0, 1]; got {f:g}.")
    if center is None:
        center = center_of_mass(positions, masses)
    else:
        center = _as_finite_array(center, "center", shape=(3,))

    d = positions - center
    r2 = np.einsum("ij,ij->i", d, d)
    # r2 can overflow to +inf for sufficiently large (but individually
    # finite) positions, which would otherwise make that body's radius
    # silently become +inf even though its true Euclidean distance from
    # ``center`` is itself comfortably representable (squaring before
    # taking the square root loses range that the direct distance never
    # needed). Recomputed exactly for these rows via a scale-safe
    # distance instead.
    overflowed = ~np.isfinite(r2)
    with np.errstate(over="ignore"):
        r = np.sqrt(r2)
    if np.any(overflowed):
        for k in np.nonzero(overflowed)[0]:
            r[k] = _safe_distance_scalar(d[k, 0], d[k, 1], d[k, 2], 0.0, 0.0)
    if not np.all(np.isfinite(r)):
        raise ValueError(
            "a lagrangian-radius distance overflowed; check that "
            "positions are physically reasonable."
        )
    order = np.argsort(r)
    r_sorted = r[order]
    # Summing masses via a raw np.cumsum() can
    # overflow to inf for sufficiently large (but individually finite and
    # representable) masses -- center_of_mass() already guards against the
    # analogous sum(masses) overflow by normalizing by the largest mass
    # first; cum_frac is a RATIO of cumulative sums, so scaling every mass
    # by the same positive constant beforehand leaves it mathematically
    # unchanged while keeping every partial sum representable.
    weights = masses / np.max(masses)
    cum_weight = np.cumsum(weights[order])
    cum_frac = cum_weight / cum_weight[-1]

    out = {}
    for f in fractions:
        idx = int(np.searchsorted(cum_frac, f))
        idx = min(idx, r_sorted.size - 1)
        out[f] = float(r_sorted[idx])
    return out


def half_mass_radius(positions, masses, center=None):
    return lagrangian_radii(positions, masses, [0.5], center=center)[0.5]


def _phi_and_speed2(positions, velocities, masses, softening):
    """Shared O(N^2) core of specific_energies() and high_velocity_fraction()."""
    masses = _require_masses(masses)
    positions = _require_snapshot(positions, "positions", n_bodies=masses.size)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=masses.size)
    softening = _require_positive("softening", softening)
    # For a sufficiently large (but individually
    # finite) softening_pc, softening*softening can silently overflow to
    # inf. Unchecked, that makes every (r^2 + eps^2)^(-3/2) term evaluate
    # to exactly 0.0 -- finite, not nan or inf, so it would NOT be caught
    # by a downstream np.isfinite() postcondition check on the resulting
    # accelerations -- silently zeroing out all gravity while still
    # reporting a "successful" run. Caught explicitly here instead, at
    # the one place the overflow actually originates.
    eps2 = _require_positive("softening**2", softening * softening)
    # eps is the ORIGINAL softening length, not recomputed via
    # math.sqrt(eps2) -- squaring then unsquaring a softening length
    # whose square falls deep in the denormal range loses precision
    # that using the already-available original value avoids entirely.
    eps = softening
    n = masses.size

    pos_cm, vel_cm = recenter(positions, velocities, masses)
    phi = np.zeros(n)
    for i in range(n):
        d = pos_cm - pos_cm[i]
        r2 = np.einsum("ij,ij->i", d, d) + eps2
        # Same overflow hazard as potential_energy(): r2 can overflow to
        # +inf for a large (but individually finite) separation, which
        # would otherwise silently drop a body's real contribution to
        # this specific potential (dividing by the resulting +inf gives
        # 0.0) even though the true 1/r contribution is representable.
        # Only the distance needs recomputing (see potential_energy's
        # comment on why 1/r itself cannot overflow once r is finite).
        overflowed = ~np.isfinite(r2)
        with np.errstate(over="ignore"):
            r = np.sqrt(r2)
        if np.any(overflowed):
            for k in np.nonzero(overflowed)[0]:
                if k == i:
                    continue
                r[k] = _safe_distance_scalar(d[k, 0], d[k, 1], d[k, 2], eps2, eps)
        r[i] = np.inf                       # exclude self term
        # Each body's G*m_j/r contribution is combined in ONE fused
        # _scaled_product_over() call, not computed as G*(masses/r): a
        # sufficiently large source mass and small r can make masses/r
        # ALONE overflow to +inf even though the fully combined G*m_j/r
        # term is comfortably representable once G (~6.674e-11) is
        # folded in -- e.g. masses of 1e308 kg at r=0.1 m makes
        # masses/r ~1e309 overflow, while G*masses/r ~6.67e298 does not.
        # This mirrors potential_energy()'s own pair_terms fusion above.
        with np.errstate(over="ignore"):
            phi[i] = -float(np.sum(_scaled_product_over(r, G, masses)))
        if not math.isfinite(phi[i]):
            raise ValueError(
                "specific potential energy overflowed for at least one "
                "body; check that positions and masses are physically "
                "reasonable."
            )
    # This returns (1/2)|v|^2 -- HALF the squared speed, in the
    # center-of-mass frame -- not the raw |v|^2. That distinction
    # matters: both of this
    # function's callers (specific_energies's (1/2)v^2 + Phi, and
    # high_velocity_fraction's bound/fast comparisons, reformulated below
    # to use half_speed2 directly rather than 0.5*speed2/speed2) only
    # ever need the HALVED quantity, but summing the raw per-component
    # squares first (vx^2+vy^2+vz^2) before applying the 1/2 factor can
    # overflow even when the true, halved result is representable: e.g.
    # v=(1e154,1e154,0) has vx^2+vy^2 = 2e308, which exceeds float64's
    # ~1.8e308 ceiling and overflows, while (1/2)(vx^2+vy^2) = 1e308 is
    # comfortably representable. Applying the 1/2 factor to each squared
    # COMPONENT before summing (via the same fused _scaled_product()
    # pattern kinetic_energy() already uses), rather than to their sum
    # afterward, keeps every individual term -- and therefore the
    # per-body total -- in range whenever the true halved speed-squared
    # itself is representable, even though the unhalved sum is not.
    with np.errstate(over="ignore"):
        half_speed2 = np.sum(_scaled_product(0.5, vel_cm, vel_cm), axis=1)
    if not np.all(np.isfinite(half_speed2)):
        raise ValueError(
            "squared speed overflowed for at least one body; check that "
            "velocities are physically reasonable."
        )
    return phi, half_speed2


def specific_energies(positions, velocities, masses, softening):
    """
    Per-body specific (per unit mass) mechanical energy in the center-of-
    mass frame, e_i = (1/2) v_i^2 + Phi_i, where Phi_i is the softened
    potential at body i due to every OTHER body. A positive e_i marks a
    body that is INSTANTANEOUSLY unbound from the rest of the system
    given its current position and velocity (see identify_unbound).

    O(N^2); intended for periodic diagnostics, not every integration step.
    """
    phi, half_speed2 = _phi_and_speed2(positions, velocities, masses, softening)
    with np.errstate(over="ignore"):
        energies = half_speed2 + phi
    if not np.all(np.isfinite(energies)):
        raise ValueError(
            "specific energy overflowed for at least one body; check "
            "that positions, velocities and masses are physically "
            "reasonable."
        )
    return energies


def high_velocity_fraction(positions, velocities, masses, softening, threshold=0.9):
    """
    Fraction of bodies whose speed already exceeds ``threshold`` times
    their own LOCAL escape speed, sqrt(2 |Phi_i|), without yet being
    formally unbound (speed < escape speed, i.e. e_i < 0 still).

    Two-body relaxation evaporates a cluster by slowly repopulating the
    high-velocity tail of the (approximately, not exactly, Maxwellian)
    speed distribution until individual stars cross their local escape
    speed. With the force softening this program uses to keep the
    tree/direct force evaluation numerically well-behaved (see Domain of
    Validity in the Help file), that final crossing is suppressed --
    and, at default parameters, this near-escape fraction is typically
    ALSO flat at zero for the whole run, not merely a smaller but still
    growing signal. This fraction is offered as a finer-grained, more
    sensitive diagnostic of relaxation than waiting for
    identify_unbound() to register a positive-energy body -- not a
    claim that it tracks a precise physical distribution, or that it
    grows smoothly or monotonically over a run: at finite N it is
    itself a count divided by N, so it only takes values in steps of
    1/N and, like n_unbound, can rise and fall from one snapshot to the
    next as the potential evolves. Its advantage over identify_unbound()
    is a lower detection threshold (a body approaching its escape speed
    rather than only one that has already crossed it), not continuity.

    ``threshold`` must be strictly positive but is not required to be
    below 1: a value at or above 1 is not unsafe -- it always returns
    0.0, because the ``bound`` mask used here already excludes anything
    at or past the local escape speed, so nothing can simultaneously be
    bound and at or above threshold=1 times that speed. That is a
    documented degenerate case for an unusual argument, not an error, so
    it is not rejected; a threshold intended to mean "near escape but
    still bound" should be chosen in (0, 1).
    """
    threshold = _require_positive("threshold", threshold)
    phi, half_speed2 = _phi_and_speed2(positions, velocities, masses, softening)
    # threshold>=1 is documented above as always returning 0.0 -- the
    # "bound" mask already excludes anything at or past the local escape
    # speed, so nothing can simultaneously be bound and at or above
    # threshold=1 times that speed. Returned here directly, immediately
    # AFTER the shared validation/computation above (so a malformed
    # positions/velocities/masses argument is still caught) but BEFORE
    # ever computing threshold**2: that squaring itself overflows for a
    # large but individually valid finite threshold like 1e308, even
    # though the documented answer for every threshold>=1 is the same
    # simple 0.0 regardless of phi/half_speed2.
    if threshold >= 1.0:
        return 0.0
    # Both comparisons below are stated in terms of HALF the escape
    # speed squared and half the speed squared, never the raw (unhalved)
    # quantities -- v_esc2 = 2*|phi| can itself overflow for a large but
    # individually representable phi even when |phi| (its own half)
    # does not, and speed2 = 2*half_speed2 has the identical hazard
    # _phi_and_speed2's own docstring now explains. Since both sides of
    # "speed2 >= threshold**2 * v_esc2" would be scaled by the same
    # factor of 2, the comparison's truth value is unaffected by
    # working in half-quantities throughout, so nothing is lost by
    # never forming the unhalved values at all.
    half_v_esc2 = np.abs(phi)
    bound = (half_speed2 + phi) < 0.0
    fast = half_speed2 >= (threshold ** 2) * half_v_esc2
    return float(np.mean(fast & bound))


def identify_unbound(positions, velocities, masses, softening):
    """
    Boolean mask: True where a body's specific energy is INSTANTANEOUSLY
    positive (formally unbound given its current position and velocity),
    evaluated in the system's center-of-mass frame.

    A positive specific energy is the physically correct criterion for a
    body that WILL eventually escape to infinity in a static potential
    that falls to zero there, regardless of its instantaneous radial
    velocity sign -- a body can be energetically unbound while
    momentarily moving inward (e.g. on the incoming branch of a
    hyperbolic-like encounter), so no outward-motion requirement is
    added here. But in an N-body system the potential is NOT static -- it evolves as
    the other bodies move -- so a body flagged here can later return to
    negative energy as the potential changes; this count is therefore
    NOT necessarily monotonically increasing over a run, and a body
    counted here is "instantaneously unbound," not a confirmed,
    permanent escaper -- the count can rise and fall over the course of a
    run rather than only accumulating. Downstream summary fields
    (n_unbound, unbound_fraction_final) and documentation are named
    accordingly.
    """
    return specific_energies(positions, velocities, masses, softening) > 0.0


def _scale_safe_ratio_mantissa_exponent(numerator, *denominator_factors):
    """
    Shared core of _scale_safe_scalar_ratio() and _scale_safe_sqrt_ratio():
    decompose ``numerator / (denominator_factors[0] * denominator_factors[1]
    * ...)`` into (mantissa in [0.5, 1) or its negative, integer exponent)
    via math.frexp, WITHOUT ever reconstructing the ratio as a standalone
    float64 (that reconstruction is exactly the step each caller wants to
    defer or skip -- see their own docstrings for why).
    """
    nm, ne = math.frexp(numerator)
    dm, de = 1.0, 0
    for factor in denominator_factors:
        fm, fe = math.frexp(factor)
        dm *= fm
        dm, me = math.frexp(dm)
        de += fe + me
    ratio_m = nm / dm
    ratio_m, me = math.frexp(ratio_m)
    ratio_e = ne - de + me
    return ratio_m, ratio_e


def _scale_safe_scalar_ratio(numerator, *denominator_factors):
    """
    Scale-safe ``numerator / (denominator_factors[0] * denominator_factors[1] * ...)``
    for plain Python floats, computed by frexp-decomposing every factor
    (numerator included) into a mantissa in [0.5, 1) times a power of
    two and combining the exponents symbolically, so the denominator's
    product is never formed as its own standalone float64 value.

    That standalone product can underflow to exactly 0.0 (turning an
    ordinary division into a raw ZeroDivisionError) or overflow to +inf
    even when the true, fully combined ratio is itself comfortably
    representable -- e.g. G * total_mass_kg underflows for a mass at
    the representable-denormal floor, even though r / (G * total_mass_kg)
    is not. Only the single final ldexp can overflow/underflow, and only
    when the true mathematical ratio itself is not representable.

    NOTE: this reconstructs the ratio itself as a standalone float64. If
    the caller's real interest is sqrt(ratio) rather than the ratio
    itself (e.g. crossing_time()), use _scale_safe_sqrt_ratio() instead --
    the ratio can be genuinely non-representable (too large or too small
    for float64's exponent range) even when its square root is not, so
    reconstructing the plain ratio first can fail a case the caller did
    not actually need to succeed.
    """
    ratio_m, ratio_e = _scale_safe_ratio_mantissa_exponent(numerator, *denominator_factors)
    # math.ldexp raises a raw OverflowError (rather than saturating to
    # inf the way multiplication does) when the reconstructed magnitude
    # is too large to represent -- exactly the "true ratio is not
    # representable" case every call site already checks for via an
    # isfinite()/_require_positive() postcondition, so it is converted
    # to a signed inf here instead of propagating as an uncaught
    # exception.
    try:
        return math.ldexp(ratio_m, ratio_e)
    except OverflowError:
        return math.inf if ratio_m > 0.0 else -math.inf


def _scale_safe_sqrt_ratio(numerator, *denominator_factors):
    """
    Scale-safe ``sqrt(numerator / (denominator_factors[0] *
    denominator_factors[1] * ...))`` for plain Python floats -- the
    square root is taken IN EXPONENT SPACE, on the (mantissa, exponent)
    decomposition from _scale_safe_ratio_mantissa_exponent(), before the
    ratio is ever reconstructed as a standalone float64.

    This matters because the plain ratio can be non-representable (its
    magnitude can exceed float64's ~1.8e308 ceiling, or underflow below
    its ~4.9e-324 floor) even when the SQUARE ROOT of that same ratio is
    comfortably representable -- taking a square root roughly halves a
    value's exponent, so a ratio whose true magnitude is, say, 1e620 is
    itself far outside float64's range, but its square root, about
    1e310, might still be only modestly out of range, or (for a less
    extreme example) a ratio of about 3e333 is not representable at all,
    while its square root, about 5.5e166, is. crossing_time() needs
    exactly this: t_cross = r * sqrt(r / (G*M)), and a total_mass_kg near
    the smallest representable positive float64 makes the intermediate
    ratio r/(G*M) itself overflow even though t_cross is finite.

    Reconstructing sqrt(mantissa * 2**exponent) requires care when
    ``exponent`` is odd, since 2**(exponent/2) is then not itself an
    integer power of two: this rewrites mantissa * 2**exponent as
    (mantissa * 2) * 2**(exponent - 1) first whenever exponent is odd,
    so the exponent handed to the square root is always even and the
    corresponding mantissa (in [0.5, 2)) stays comfortably within an
    ordinary, safe range for math.sqrt.
    """
    ratio_m, ratio_e = _scale_safe_ratio_mantissa_exponent(numerator, *denominator_factors)
    if ratio_m < 0.0:
        raise ValueError(
            "_scale_safe_sqrt_ratio's ratio is negative; its square root "
            "is not a real number."
        )
    if ratio_e % 2 != 0:
        ratio_m *= 2.0
        ratio_e -= 1
    sqrt_m = math.sqrt(ratio_m)
    sqrt_e = ratio_e // 2
    try:
        return math.ldexp(sqrt_m, sqrt_e)
    except OverflowError:
        return math.inf


# ======================================================================
# Characteristic timescales
# ======================================================================
def crossing_time(half_mass_radius_m, total_mass_kg):
    """
    Dynamical (crossing) time, t_cross = sqrt(r_h^3 / (G M)).

    An order-of-unity estimate of the time for a body to cross the system,
    used here as the natural unit for choosing an integration timestep and
    for expressing the relaxation time below.
    """
    half_mass_radius_m = _require_positive("half_mass_radius_m", half_mass_radius_m)
    total_mass_kg = _require_positive("total_mass_kg", total_mass_kg)
    # t_cross = sqrt(r^3 / (G M)) is computed as r * sqrt(r / (G M)), NOT
    # via r_cubed = r*r*r followed by a division and one final sqrt
    # (e.g. crossing_time(1e130, 1e30) has a true value of about
    # 1.224e185, comfortably representable, but
    # r_cubed = (1e130)**3 = 1e390 overflows float64 long before the
    # division or the sqrt ever run, discarding a representable result --
    # and cubing via ** can also raise a raw, uncaught OverflowError for
    # a large radius, since Python's float.__pow__ goes through C's
    # pow(), which signals ERANGE as OverflowError rather than silently
    # saturating to inf the way plain multiplication does).
    #
    # A plain r/(G*M) ratio -- even computed via the exponent-aware
    # _scale_safe_scalar_ratio() -- can STILL be non-representable on
    # its own (e.g. r=1, M at the smallest
    # representable positive float64: the true ratio is about 3.06e333,
    # which does not fit in float64) even though sqrt(r/(G*M)) (about
    # 5.53e166 here) and the final t_cross ARE representable. The fix is
    # to take the square root IN EXPONENT SPACE via
    # _scale_safe_sqrt_ratio(), which never reconstructs the plain ratio
    # as a standalone float64 at all, only its square root -- and only
    # THAT single reconstruction can overflow/underflow, exactly when
    # sqrt(r/(G*M)) itself is not representable. The final multiplication
    # by r uses _scaled_product(), not a plain ``*``, for the same reason
    # every other fused pairwise/scalar formula in this module does: r
    # and sqrt_ratio are each already normal-range values at this point,
    # but a genuinely extreme r (this function's own contract allows any
    # positive finite radius) should not reintroduce a standalone-
    # multiplication overflow right at the last step.
    sqrt_ratio = _scale_safe_sqrt_ratio(half_mass_radius_m, G, total_mass_kg)
    sqrt_ratio = _require_positive(
        "sqrt(half_mass_radius_m / (G * total_mass_kg))", sqrt_ratio
    )
    t_cross = float(_scaled_product(half_mass_radius_m, sqrt_ratio))
    if not math.isfinite(t_cross):
        raise ValueError(
            "crossing time overflowed; check that half_mass_radius_m and "
            "total_mass_kg are physically reasonable."
        )
    return t_cross


def free_fall_time(mean_density_kg_m3):
    """Free-fall time of a uniform sphere, t_ff = sqrt(3 pi / (32 G rho))."""
    mean_density_kg_m3 = _require_positive("mean_density_kg_m3", mean_density_kg_m3)
    # 32 * G * mean_density_kg_m3 can underflow toward zero for a
    # sufficiently small (but positive and finite) density, which would
    # otherwise make the single combined ratio 3*pi/(32*G*rho) overflow
    # to +inf even though the true free-fall time is comfortably
    # representable. Taking the two square roots separately (of the
    # density-independent constant, and of 1/rho) and dividing keeps
    # both intermediates in range.
    t = math.sqrt(3.0 * math.pi / (32.0 * G)) / math.sqrt(mean_density_kg_m3)
    if not math.isfinite(t):
        raise ValueError(
            "free-fall time overflowed; check that mean_density_kg_m3 is "
            "physically reasonable."
        )
    return t


def relaxation_time(n_bodies, half_mass_radius_m, total_mass_kg):
    """
    Two-body relaxation timescale,

        t_relax  ~=  (N / (8 ln N)) * t_cross,   t_cross = sqrt(r_h^3/(G M))

    following Binney & Tremaine, "Galactic Dynamics" (2nd ed., eq. 1.38).
    The Coulomb logarithm is approximated here as ln(N); real N-body
    codes commonly use ln(0.4 N) or ln(N/2) instead, which changes the
    result by an order-unity factor, not a scaling. Treat this as a
    NOMINAL, UNSOFTENED order-of-magnitude estimate: the closed form
    above assumes point-mass two-body encounters at arbitrarily close
    range, exactly what this program's force softening (see
    athanassoula_softening()) exists to suppress. It correctly captures how
    relaxation accelerates as N shrinks (this NOMINAL, unsoftened estimate
    is short enough, for a 200-body cluster, to fall within a run length a
    student can actually complete, unlike a 200,000-body one), but it is
    not a prediction of how fast THIS program's own softened,
    discrete realization will actually relax -- the true softened rate is
    suppressed below this nominal value, sometimes to the point of no
    measurable relaxation at all within a practical run length at default
    softening (see run_cluster()'s docstring and Suggested Experiment
    EXP-11 for the measured consequence).
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=2)
    t_cross = crossing_time(half_mass_radius_m, total_mass_kg)
    return (n_bodies / (8.0 * math.log(n_bodies))) * t_cross


# ======================================================================
# Initial conditions
# ======================================================================
def _isotropic_directions(n, rng):
    cos_theta = 1.0 - 2.0 * rng.random(n)
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta ** 2, 0.0, 1.0))
    phi = 2.0 * math.pi * rng.random(n)
    return np.column_stack((
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta,
    ))


def plummer_sphere(n_bodies, total_mass_msun, scale_radius_pc, softening=None,
                    max_radius_factor=50.0, seed=None):
    """
    Sample an isotropic, isolated Plummer (1911) sphere whose smooth,
    unsoftened continuum distribution function is a genuine dynamical
    (phase-space) equilibrium, using the closed-form inversion for
    positions and the rejection method for velocities of Aarseth, Henon &
    Wielen (1974, A&A 37, 183), as also given in Hut & Makino's online
    "Moving Stars Around" text and in Binney & Tremaine's "Galactic
    Dynamics".

    Density profile:  rho(r) = (3M / 4 pi a^3) (1 + r^2/a^2)^(-5/2)
    Enclosed mass:     M(r)/M = r^3 / (r^2 + a^2)^(3/2) = x1  =>
                       r = a (x1^(-2/3) - 1)^(-1/2)

    IMPORTANT SCOPE OF THIS EQUILIBRIUM CLAIM: the velocity rejection
    sampler below draws speeds from the exact Plummer distribution
    function, which is a genuine dynamical equilibrium for the SMOOTH,
    unsoftened Plummer potential Phi(r) = -GM/sqrt(r^2+a^2) only. This
    program integrates a discrete, Plummer-SOFTENED N-body realization
    instead (pairwise force ~ 1/(r^2+eps^2)), whose actual scalar-virial
    quantity Wvir (virial_force_term()) differs from the idealized
    continuum energy the DF was matched to -- both because N is finite
    and because eps > 0. Left uncorrected, this produces a systematic
    energy-SCALE mismatch (not a shape mismatch: relative particle-to-
    particle speeds are still drawn correctly) that shows up as apparent
    evolution during the first few crossing times of a run, easily
    mistaken for genuine two-body-relaxation-driven evolution when it is
    actually just the discrete system readjusting away from a mismatched
    initial condition. The mitigation applied here: after sampling,
    rescale ALL velocities by one overall constant (so the DF's shape is
    unchanged) to put the actual discrete, softened realization into
    exact INSTANTANEOUS SCALAR VIRIAL BALANCE, 2T/|Wvir| = 1, under the
    force law it will actually be integrated with. This is a necessary
    but not sufficient condition for genuine dynamical equilibrium: it
    fixes the single global energy-scale mismatch, but does not by
    itself establish that the discrete, softened realization sits at a
    stationary point of the collisionless Boltzmann equation, and some
    readjustment transient on the order of a few crossing times can
    still remain (see run_cluster()'s docstring for how this program
    tries to separate that initial readjustment from genuine two-body
    relaxation).

    Velocities are drawn from the exact isotropic Plummer distribution
    function f(E) ~ (-E)^(7/2) by rejection sampling q = v/v_esc(r)
    against g(q) = q^2 (1 - q^2)^(7/2), whose maximum is at
    q = 1/sqrt(4.5) approx 0.4714 (from dg/dq = 0 => 1 - 4.5 q^2 = 0),
    with true maximum value g(0.4714) approx 0.09221. The envelope
    constant 0.1 used below is comfortably above this true maximum
    (0.1 > 0.09221). The overall per-attempt acceptance probability is
    integral_0^1 g(q) dq / 0.1 approx 0.04295 / 0.1 approx 43%.

    The Plummer profile's mass distribution formally extends to infinite
    radius, which is impractical to realize with a finite N-body sample
    (one unlucky, very distant particle would dominate the octree's
    bounding box every step). Following standard N-body practice (e.g.
    NEMO's mkplummer), this sampler discards and redraws any body whose
    radius exceeds ``max_radius_factor`` scale radii; the fraction of
    mass this excludes is returned in the diagnostics for transparency.

    All bodies are given equal mass, total_mass_msun / n_bodies. Returns
    a dict with SI-unit arrays 'positions', 'velocities', 'masses'
    (shape (n,3), (n,3), (n,)) and a 'diagnostics' dict.
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    total_mass_msun = _require_positive("total_mass_msun", total_mass_msun)
    scale_radius_pc = _require_positive("scale_radius_pc", scale_radius_pc)
    max_radius_factor = _require_positive("max_radius_factor", max_radius_factor)

    a = scale_radius_pc * PC
    softening = softening if softening is not None else athanassoula_softening(n_bodies, a)
    softening = _require_positive("softening", softening)
    total_mass = total_mass_msun * M_sun
    m_body = total_mass / n_bodies
    r_max = max_radius_factor * a

    rng = np.random.default_rng(seed)

    r = np.empty(n_bodies)
    n_redrawn = 0
    filled = 0
    rounds = 0
    while filled < n_bodies:
        rounds += 1
        if rounds > MAX_REJECTION_ROUNDS:
            raise RuntimeError(
                f"max_radius_factor = {max_radius_factor:g} accepts too few "
                "Plummer-sphere draws to fill the requested body count; use "
                "a larger max_radius_factor (10-100 is typical)."
            )
        need = n_bodies - filled
        x1 = rng.random(need)
        x1 = np.clip(x1, 1e-300, None)          # x1 == 0.0 has negligible
                                                 # probability but would
                                                 # otherwise raise x1**(-2/3)
                                                 # to a RuntimeWarning inf
        with np.errstate(divide="ignore"):
            candidate = a / np.sqrt(np.maximum(x1 ** (-2.0 / 3.0) - 1.0, 1e-300))
        keep = candidate[candidate <= r_max]    # <= need values, by construction
        r[filled:filled + keep.size] = keep
        n_redrawn += (need - keep.size)
        filled += keep.size
    directions = _isotropic_directions(n_bodies, rng)
    positions = r[:, None] * directions

    v_esc = np.sqrt(2.0 * G * total_mass / np.sqrt(r ** 2 + a ** 2))
    speed_frac = np.empty(n_bodies)
    for i in range(n_bodies):
        for _attempt in range(MAX_REJECTION_ROUNDS):
            q = rng.random()
            y = 0.1 * rng.random()
            if y <= q * q * (1.0 - q * q) ** 3.5:
                speed_frac[i] = q
                break
        else:
            raise RuntimeError(
                "the Plummer velocity rejection sampler failed to converge; "
                "this should not happen for any valid input and indicates a "
                "bug rather than an unlucky draw."
            )
    velocities = (speed_frac * v_esc)[:, None] * _isotropic_directions(n_bodies, rng)

    masses = np.full(n_bodies, m_body)
    # Recenter BEFORE the scalar-virial-balance rescale below: scaling a
    # zero-mean velocity set by one overall constant preserves zero mean
    # exactly, so doing this first makes the rescale exact rather than
    # only approximately so (see uniform_sphere() for the N=3 worst case
    # this order fixes).
    positions, velocities = recenter(positions, velocities, masses)

    t_sampled = kinetic_energy(velocities, masses)
    wvir_sampled = virial_force_term(positions, masses, softening)
    if t_sampled > 0.0 and wvir_sampled != 0.0:
        target_t = 0.5 * abs(wvir_sampled)
        velocities = velocities * math.sqrt(target_t / t_sampled)
    # A uniform rescale by one overall constant cannot change the mean, so
    # this is a no-op up to floating-point roundoff; kept for defensiveness.
    positions, velocities = recenter(positions, velocities, masses)
    virial_ratio_initial = virial_ratio(
        kinetic_energy(velocities, masses),
        virial_force_term(positions, masses, softening),
    )

    diagnostics = dict(
        n_bodies=n_bodies, total_mass_msun=total_mass_msun,
        scale_radius_pc=scale_radius_pc, m_body_msun=total_mass_msun / n_bodies,
        max_radius_factor=max_radius_factor,
        fraction_redrawn=n_redrawn / (n_redrawn + n_bodies),
        softening=softening,
        virial_ratio_before_correction=(
            virial_ratio(t_sampled, wvir_sampled) if wvir_sampled != 0.0
            else float("nan")
        ),
        virial_ratio_initial=virial_ratio_initial,
        model="Plummer (1911)",
    )
    return dict(positions=positions, velocities=velocities, masses=masses,
                diagnostics=diagnostics)


def uniform_sphere(n_bodies, total_mass_msun, radius_pc, virial_ratio_init=0.0,
                    softening=None, seed=None):
    """
    Sample bodies uniformly through the volume of a sphere of given
    radius, with an initial virial ratio Q0 = 2 T0 / |Wvir0| set exactly
    by rescaling randomly drawn velocities by one overall constant
    (rather than trusting the realization's own sampling noise to hit
    Q0), where Wvir0 = virial_force_term(positions, masses, softening) is
    computed from the ACTUAL discrete, Plummer-softened realization that
    will be integrated -- not the idealized continuum, unsoftened
    self-energy W0 = -3 G M^2 / (5 R), which differs from Wvir0 once
    softening is nonzero (see virial_force_term()'s docstring).

    Two design choices, documented together because they touch the same
    few lines:

    1. Convention: Q0 is defined with the SAME 2T/|W| normalization as
       the virial_ratio() output diagnostic (scalar balance = 1), not a
       T/|W0| convention (balance = 0.5) -- using two different
       conventions for the same phrase "virial ratio" would make the
       virial_ratio_init argument and the virial_ratio_initial summary
       field disagree by close to a factor of 2 for the same physical
       state.
    2. Energy scale: this uses the actual discrete, softened Wvir0
       (virial_force_term) rather than the idealized continuum,
       unsoftened self-energy W0 = -3 G M^2 / (5 R) as the balance
       reference, for the same reason given in plummer_sphere()'s
       docstring -- for any nonzero softening or finite N, W0 != Wvir0,
       so scaling to Q0 = 1 against W0 would not actually put the
       realization into scalar virial balance under the force law it is
       integrated with.

    Q0 = 0 (the default) is a perfectly cold start: every body begins at
    rest, and the sphere free-falls before violently relaxing into a
    quasi-equilibrium remnant -- the classic toy model of collisionless
    "violent relaxation" (Lynden-Bell 1967; the numerical experiment goes
    back to van Albada 1982, MNRAS 201, 939). Q0 = 1 puts the actual
    discrete, softened realization into exact instantaneous scalar
    virial balance (2T/|Wvir| = 1) -- not, by itself, a claim that it is
    in genuine dynamical equilibrium (see virial_force_term()'s
    docstring); values in between give a "warm" collapse with a milder
    initial transient.

    Returns the same dict contract as plummer_sphere().
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    total_mass_msun = _require_positive("total_mass_msun", total_mass_msun)
    radius_pc = _require_positive("radius_pc", radius_pc)
    virial_ratio_init = _require_nonnegative("virial_ratio_init", virial_ratio_init)
    if virial_ratio_init > 10.0:
        raise ValueError(
            f"virial_ratio_init = {virial_ratio_init:g} is unreasonably large "
            "for a bound initial condition (super-virial systems above a few "
            "unbind almost immediately); use a value below about 10."
        )

    r_sphere = radius_pc * PC
    softening = softening if softening is not None else athanassoula_softening(n_bodies, r_sphere)
    softening = _require_positive("softening", softening)
    total_mass = total_mass_msun * M_sun
    m_body = total_mass / n_bodies

    rng = np.random.default_rng(seed)
    u = rng.random(n_bodies)
    r = r_sphere * u ** (1.0 / 3.0)
    positions = r[:, None] * _isotropic_directions(n_bodies, rng)
    masses = np.full(n_bodies, m_body)
    w0 = -3.0 * G * total_mass ** 2 / (5.0 * r_sphere)  # idealized reference only

    if virial_ratio_init <= 0.0:
        velocities = np.zeros_like(positions)
        positions, velocities = recenter(positions, velocities, masses)
    else:
        raw_v = rng.normal(size=(n_bodies, 3))
        # Recenter (subtract the mass-weighted mean position AND velocity)
        # BEFORE rescaling to the target kinetic energy, not after: scaling
        # an already-zero-mean velocity set by one overall constant keeps
        # it exactly zero-mean, so this order makes the rescale hit the
        # target T exactly for any N. Scaling first and recentering
        # afterward instead would let the recentering step perturb T away
        # from target -- negligible at large N, but severe at small N
        # (measured final-T/target-T ratio at N=3 ranging 0.148-0.983
        # across 50 seeds under that order, versus 0.979-1.000 at N=300).
        positions, raw_v = recenter(positions, raw_v, masses)
        wvir0 = virial_force_term(positions, masses, softening)
        target_t = 0.5 * virial_ratio_init * abs(wvir0)
        t_raw = kinetic_energy(raw_v, masses)
        velocities = raw_v * math.sqrt(target_t / t_raw) if t_raw > 0.0 else raw_v
        # A uniform rescale by one overall constant cannot change the
        # mean, so this is a no-op up to floating-point roundoff; kept for
        # defensiveness.
        positions, velocities = recenter(positions, velocities, masses)

    virial_ratio_initial = virial_ratio(
        kinetic_energy(velocities, masses),
        virial_force_term(positions, masses, softening),
    )

    diagnostics = dict(
        n_bodies=n_bodies, total_mass_msun=total_mass_msun,
        radius_pc=radius_pc, m_body_msun=total_mass_msun / n_bodies,
        virial_ratio_init=virial_ratio_init,
        virial_ratio_initial=virial_ratio_initial,
        softening=softening,
        analytic_potential_energy=w0,
        model="uniform sphere (cold/warm collapse)",
    )
    return dict(positions=positions, velocities=velocities, masses=masses,
                diagnostics=diagnostics)


# ======================================================================
# Leapfrog time integration (the shared simulation engine)
# ======================================================================
def leapfrog_step(positions, velocities, masses, dt, softening,
                   method="tree", theta=0.5, accel=None):
    """
    Advance one kick-drift-kick leapfrog step.

        v_(1/2) = v0 + a(x0) dt/2
        x1      = x0 + v_(1/2) dt
        v1      = v_(1/2) + a(x1) dt/2

    Leapfrog is a second-order symplectic integrator WHEN the kick comes
    from a conservative force (a gradient of one potential, with a
    symmetric acceleration Jacobian) -- which holds exactly for
    method="direct". The Barnes-Hut tree force (method="tree") is
    target-dependent and not pair-symmetric, and its approximation
    changes discontinuously as bodies cross cell boundaries between
    steps, so it has not been demonstrated to be the gradient of any
    single approximate Hamiltonian; the broad "symplectic, bounded
    nonsecular energy error" guarantee is therefore only established
    here for method="direct", not asserted for method="tree" merely
    because the same KDK update formula is used with either force.

    ``accel``, if given, is the already-computed acceleration at the
    input state (a(x0)), avoiding a redundant force evaluation when the
    caller is stepping in a loop and already has it from the previous
    step's final kick; if omitted it is computed here. A caller-supplied
    accel is validated (shape and finiteness) rather than trusted, so a
    bad value from a misbehaving caller fails at the point of the bad
    input instead of silently corrupting the whole subsequent
    integration.

    Returns (positions_new, velocities_new, accel_new).

    dt must be strictly positive, not merely nonzero. Every documented
    mode of this program represents forward-in-time evolution, and an
    accepted negative dt would run silently "backward" with no warning
    or documentation -- a forward-simulation API accepting an
    undocumented reversed-time mode by accident is a trap for a direct
    caller, not a supported feature. Reversed-time integration is not
    offered here; a caller who genuinely wants it can
    negate the returned velocities and re-run forward, which is
    mathematically equivalent for this reversible integrator.
    """
    # Validate the full input state up front, via the same shape/dtype
    # checks every other public per-snapshot helper uses (but not the
    # MIN_BODIES/MAX_BODIES range, which is a whole-simulation policy
    # from _validate_state()/run_*() -- this lower-level stepping
    # function is also exercised directly, e.g. with a 2-body state, so
    # it must not impose that stricter range), rather than relying on
    # compute_accelerations() to catch a bad positions/velocities/masses
    # only when accel is omitted. When a caller supplies accel (the whole
    # point of this parameter, for a caller stepping in a loop),
    # compute_accelerations() is never called here at all, so a wrong-
    # shaped velocities array (silently broadcasting against a correctly-
    # shaped accel) or a plain list (raising a raw AttributeError deep in
    # the arithmetic below) would otherwise go uncaught until some later,
    # more confusing failure.
    masses = _require_masses(masses)
    n = masses.size
    positions = _require_snapshot(positions, "positions", n_bodies=n)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=n)
    dt = _require_positive("dt", dt)
    if accel is None:
        accel = compute_accelerations(positions, masses, softening, method, theta)
    else:
        accel = _as_finite_array(accel, "accel")
        if accel.shape != positions.shape:
            raise ValueError(
                f"accel must have the same shape as positions; got "
                f"{accel.shape} vs {np.shape(positions)}."
            )
    v_half = velocities + 0.5 * dt * accel
    positions_new = positions + dt * v_half
    accel_new = compute_accelerations(positions_new, masses, softening, method, theta)
    velocities_new = v_half + 0.5 * dt * accel_new
    return positions_new, velocities_new, accel_new


def integrate_nbody(positions, velocities, masses, dt, n_steps, softening,
                     method="tree", theta=0.5, snapshot_stride=1,
                     track_dense=False):
    """
    Run a fixed-step leapfrog N-body integration and return snapshots.

    Snapshots (including the initial state) are recorded every
    ``snapshot_stride`` steps. Energy, momentum, and the virial work term
    are evaluated only at snapshot times (always by direct summation,
    independent of which force method drives the stepping), so their cost
    does not scale with n_steps for a large stride. Angular momentum is
    NOT evaluated or returned here -- it is formally zero for these
    initial conditions and not a diagnostic this program currently
    tracks (see Experiment 16 in the Help file, which invites a student
    to add it).

    Returns a dict with:
        t              (n_snap,)      seconds, snapshot times
        positions      (n_snap, N, 3) meters
        velocities     (n_snap, N, 3) m/s
        kinetic        (n_snap,)      J, LAB-FRAME kinetic energy (per the
                                       fixed body masses); kinetic+potential
                                       is the conserved total ("energy")
        kinetic_com    (n_snap,)      J, INTERNAL (center-of-mass-relative)
                                       kinetic energy -- the numerator the
                                       officially reported virial_ratio
                                       (see run_cluster()/run_galaxy()) is
                                       actually built from, not kinetic
        potential      (n_snap,)      J
        virial_work    (n_snap,)      J  (see virial_force_term)
        energy         (n_snap,)      J  (kinetic + potential)
        momentum       (n_snap, 3)    kg m/s
        n_steps_taken  int

    If ``track_dense`` is True, two ADDITIONAL arrays of length
    n_steps + 1 (one entry per integration step, entirely independent of
    ``snapshot_stride``/``target_snapshots``) are also returned:
        r50_dense              (n_steps+1,) meters, half-mass radius
        virial_ratio_dense     (n_steps+1,) 2T/|Wvir| computed from a
                                CHEAP per-step proxy, not the snapshot-time
                                virial_work above (see below)
    This exists so that any classification decision derived from a run's
    late-time behavior (see run_galaxy()'s late-time-window settling
    criterion) can be computed from the run's actual full time resolution
    rather than from however many snapshots happened to be STORED for
    plotting/CSV output -- eliminating snapshot-density dependence by
    construction rather than trying to make a sparse-sample statistic
    robust after the fact.

    The dense virial ratio uses a cheap per-step proxy for Wvir,
        Wvir_proxy = sum_i m_i * (r_i . a_i),
    reusing the per-body accelerations the leapfrog step already computes
    for the dynamics themselves (whichever force method -- tree or direct
    -- actually drives the stepping), rather than calling
    virial_force_term() (which always recomputes exact pairwise forces by
    direct summation, deliberately, as a high-accuracy diagnostic
    reference -- see its own docstring) at every single step, which would
    reintroduce an O(N^2) cost per step regardless of stride, exactly what
    snapshot_stride exists to avoid. r_i in Wvir_proxy is measured from the
    INSTANTANEOUS center of mass (not raw positions): for exact, antisymmetric
    pairwise forces (method="direct") the net force sums to zero and the
    COM-subtraction is a no-op, so Wvir_proxy is mathematically IDENTICAL to
    virial_force_term()'s sum_i r_i . F_i there (verified to agree with
    virial_force_term() to machine precision under method="direct"); under
    method="tree", Barnes-Hut's monopole-approximated forces are NOT exactly
    antisymmetric in general, so the net force does not sum to exactly zero,
    and using raw (non-COM-relative) positions would make Wvir_proxy silently
    translation-dependent. Measuring from the instantaneous COM keeps it
    translation-invariant under both methods; it still inherits that step's
    tree-approximation force error under method="tree", exactly as the
    dynamics themselves already do. The dense kinetic-energy term that
    virial_ratio_dense is formed from likewise uses velocities relative to
    the instantaneous center-of-mass velocity, not raw velocities, so a
    uniform (Galilean-boost) velocity added to every body -- which would
    otherwise add boost-dependent bulk kinetic energy unrelated to the
    system's internal virial state -- leaves both r50_dense and
    virial_ratio_dense unchanged. This is a self-consistent classification
    proxy, not a replacement for the officially reported, always-exact
    snapshot series: kinetic and energy remain the lab-frame totals the
    integrator actually conserves, while the official virial_ratio that
    run_cluster()/run_galaxy() plot and write to CSV is formed from
    kinetic_com (internal, COM-relative) and virial_work, i.e.
    2*kinetic_com/|virial_work| -- not from kinetic.

    dt must be strictly positive; see leapfrog_step's docstring for why a
    negative (reversed-time) dt is rejected rather than silently
    supported.
    """
    positions, velocities, masses = _validate_state(positions, velocities, masses)
    dt = _require_positive("dt", dt)
    n_steps = _require_int("n_steps", n_steps, lo=MIN_STEPS, hi=MAX_STEPS)
    softening = _require_positive("softening", softening)
    method = _require_method(method)
    snapshot_stride = _require_int("snapshot_stride", snapshot_stride, lo=1)

    # Snapshot indices: step 0, every snapshot_stride-th step, and always
    # the true final step (n_steps), even when n_steps is not an exact
    # multiple of snapshot_stride -- otherwise the reported final energy,
    # Lagrangian radii, etc. would silently belong to an earlier step than
    # the one the caller asked to run to.
    snap_steps = list(range(0, n_steps, snapshot_stride))
    if snap_steps[-1] != n_steps:
        snap_steps.append(n_steps)
    n_snapshots = len(snap_steps)
    if n_snapshots > MAX_SNAPSHOTS:
        raise ValueError(
            f"n_steps / snapshot_stride = {n_snapshots:,} snapshots exceeds "
            f"the limit of {MAX_SNAPSHOTS:,}; raise snapshot_stride or lower "
            "n_steps."
        )
    n_bodies = masses.size
    body_snapshot_product = n_bodies * n_snapshots
    if body_snapshot_product > MAX_BODY_SNAPSHOT_PRODUCT:
        raise ValueError(
            f"n_bodies * n_snapshots = {n_bodies:,} * {n_snapshots:,} = "
            f"{body_snapshot_product:,} exceeds the combined limit of "
            f"{MAX_BODY_SNAPSHOT_PRODUCT:,} (positions and velocities are "
            "each stored as a full (n_snapshots, n_bodies, 3) history); "
            "raise snapshot_stride or lower n_bodies/n_steps."
        )

    t_hist = np.empty(n_snapshots)
    pos_hist = np.empty((n_snapshots, n_bodies, 3))
    vel_hist = np.empty((n_snapshots, n_bodies, 3))
    kin_hist = np.empty(n_snapshots)
    kin_com_hist = np.empty(n_snapshots)
    pot_hist = np.empty(n_snapshots)
    vir_hist = np.empty(n_snapshots)
    mom_hist = np.empty((n_snapshots, 3))
    if track_dense:
        r50_dense = np.empty(n_steps + 1)
        qvir_dense = np.empty(n_steps + 1)

    def _record(k, t, pos, vel):
        t_hist[k] = t
        pos_hist[k] = pos
        vel_hist[k] = vel
        kin_hist[k] = kinetic_energy(vel, masses)
        # A SEPARATE COM-relative (internal) kinetic energy, used only
        # as the virial-ratio numerator (see
        # _virial_track's callers in run_cluster/run_galaxy below) -- not
        # a replacement for kin_hist/"kinetic" above, which stays the raw
        # lab-frame kinetic energy so that "energy" (=kinetic+potential)
        # remains the quantity the integrator actually conserves and
        # kinetic_J + potential_J == energy_J stays exactly true in the
        # CSV regardless of the frame velocities happen to be reported
        # in. The scalar virial theorem 2T = -Wvir concerns only the
        # system's INTERNAL kinetic energy: a uniform bulk boost added to
        # every velocity changes kin_hist (lab-frame KE gains bulk
        # translational energy having nothing to do with internal virial
        # balance) but leaves the true internal dynamics, and therefore
        # the correct virial ratio, unchanged. Without this correction, a
        # boosted-but-otherwise-identical run reports a wildly different
        # (and physically meaningless) official virial_ratio even though
        # the system's internal state is identical -- see
        # _record_dense's matching comment below, which already applies
        # the identical correction to the dense proxy. Uses
        # _com_relative_kinetic_energy() rather than
        # kinetic_energy(vel - center_of_mass_velocity(vel, masses),
        # masses): materializing that raw difference can itself overflow
        # for two individually finite, opposite-sign velocities even
        # when the resulting internal kinetic energy is representable
        # (a small enough body mass keeps it finite either way) -- see
        # _com_relative_kinetic_energy()'s own docstring.
        kin_com_hist[k] = _com_relative_kinetic_energy(vel, masses)
        pot_hist[k] = potential_energy(pos, masses, softening)
        # Wvir (virial_force_term), not U (potential_energy), is the correct
        # denominator for the virial ratio once softening is nonzero -- see
        # the module note above potential_energy() and virial_force_term()'s
        # own docstring.
        vir_hist[k] = virial_force_term(pos, masses, softening)
        mom_hist[k] = np.sum(masses[:, None] * vel, axis=0)

    def _record_dense(idx, pos, vel, accel):
        # Cheap, O(N) proxy -- see integrate_nbody()'s own docstring for why
        # this is used instead of virial_force_term() at every step.
        r50_dense[idx] = half_mass_radius(pos, masses)
        # Both r and v are measured relative to the INSTANTANEOUS center of
        # mass (position and velocity), not used raw, before this proxy is
        # formed. Two independent reasons:
        #  - Under method="tree" (theta > 0) the per-body accelerations come
        #    from a Barnes-Hut monopole approximation whose implied pairwise
        #    forces are not exactly antisymmetric, so sum_i m_i * a_i (the
        #    net "force") is not exactly zero the way it is for exact/direct
        #    forces -- and sum_i m_i * (r_i . a_i) computed from RAW
        #    positions would then silently pick up a translation-dependent
        #    offset of (constant shift) . sum_i m_i * a_i.
        #  - The kinetic term is not Galilean-boost-invariant when computed
        #    from raw (lab-frame) velocities: a uniform velocity added to
        #    every body adds bulk translational kinetic energy that has
        #    nothing to do with the system's internal virial state, which
        #    would bias this classifier proxy toward whatever frame the
        #    caller happened to report velocities in.
        # Subtracting the instantaneous COM position and velocity removes
        # both dependences regardless of method, at no extra asymptotic
        # cost (center_of_mass/center_of_mass_velocity are each O(N)); all
        # of this program's own initial-condition generators already
        # recenter to a zero COM velocity.
        #
        # The position term below is formed from HALVED position/COM
        # values, not from a materialized pos - center_of_mass(pos,
        # masses) difference: two individually finite, opposite-sign
        # positions can subtract to something that overflows float64
        # even when the resulting COM-relative position -- and the
        # complete proxy term that uses it -- is itself representable.
        # h = 0.5*pos - 0.5*com is finite for ANY two individually finite
        # float64 endpoints (halving each one first bounds the magnitude
        # of their difference by float64's own largest representable
        # value), and pos_rel = 2*h, so mass*pos_rel*accel =
        # 2*mass*h*accel -- fusing that factor of 2 into the same
        # per-(body, component) _scaled_product() call below rather than
        # ever reconstructing pos_rel (= 2*h) as its own float64 value.
        # kinetic_energy's own frame issue -- computed from raw
        # (lab-frame) velocities, a uniform velocity boost changes it
        # even though it says nothing about internal virial balance --
        # is fixed the analogous way, via _com_relative_kinetic_energy()
        # (see its own docstring for why materializing vel -
        # center_of_mass_velocity(vel, masses) directly has the same
        # overflow hazard as pos_rel above).
        h_pos = 0.5 * pos - 0.5 * center_of_mass(pos, masses)
        kinetic = _com_relative_kinetic_energy(vel, masses)
        # The two-stage reduction (components, then bodies) uses
        # _scale_safe_sum() rather than np.sum(), for the same partial-
        # sum-overflow reason center_of_mass() and _build_octree()
        # already need it.
        terms = _scaled_product(2.0, masses[:, None], h_pos, accel)
        per_body = _scale_safe_sum(terms, axis=1)
        wvir_proxy = float(_scale_safe_sum(per_body, axis=0))
        if not math.isfinite(wvir_proxy):
            raise ValueError(
                "the dense virial-proxy sum overflowed even after "
                "scale-safe, fused evaluation; check that positions, "
                "masses and softening are physically reasonable."
            )
        qvir_dense[idx] = virial_ratio(kinetic, wvir_proxy)

    # build_octree() (called once per force evaluation under method="tree")
    # already consolidates every bucket LEAF within a single build into one
    # warning (see its own docstring/comment), but a multi-step integration
    # still calls it once per leapfrog step, so the SAME warning shape can
    # still repeat once per step -- exactly the run-level flood the
    # per-build consolidation alone does not address. _bucket_warning_state
    # and _rate_limited() below re-emit that one specific warning shape at
    # most once per integrate_nbody() run (the first occurrence, so a
    # caller filtering warnings as errors or with "always" still sees and
    # can act on it) and tally any further occurrences into a single
    # end-of-run summary instead of repeating it on every subsequent step.
    # Every OTHER warning (any other message or category, from anywhere
    # inside fn()) is re-emitted completely unaffected -- this never
    # silently absorbs a warning a caller's filter depends on seeing; only
    # the specific build_octree bucket message is rate-limited, and only
    # inside this function's own force-evaluation calls.
    _bucket_warning_state = {"count": 0}

    def _rate_limited(fn):
        if method != "tree":
            return fn()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()
        for w in caught:
            if (issubclass(w.category, RuntimeWarning)
                    and str(w.message).startswith("build_octree:")):
                _bucket_warning_state["count"] += 1
                if _bucket_warning_state["count"] == 1:
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return result

    pos, vel = positions.copy(), velocities.copy()
    accel = _rate_limited(lambda: compute_accelerations(pos, masses, softening, method, theta))
    snap_set = set(snap_steps)
    k = 0
    if 0 in snap_set:
        _record(k, 0.0, pos, vel)
        k += 1
    if track_dense:
        _record_dense(0, pos, vel, accel)

    t = 0.0
    for step in range(1, n_steps + 1):
        pos, vel, accel = _rate_limited(lambda: leapfrog_step(
            pos, vel, masses, dt, softening, method, theta, accel=accel
        ))
        t += dt
        if step in snap_set:
            _record(k, t, pos, vel)
            k += 1
        if track_dense:
            _record_dense(step, pos, vel, accel)

    if _bucket_warning_state["count"] > 1:
        extra = _bucket_warning_state["count"] - 1
        warnings.warn(
            f"build_octree: the coincident-body 'bucket' condition warned "
            f"about above recurred on {extra} additional force evaluation"
            f"{'s' if extra != 1 else ''} during this integrate_nbody() "
            "run (only the first occurrence was reported individually; "
            "every occurrence's own force evaluation remains fully "
            "accurate -- see build_octree's warning above for what the "
            "condition means and why no force accuracy is lost).",
            RuntimeWarning,
        )

    result = dict(
        t=t_hist, positions=pos_hist, velocities=vel_hist,
        kinetic=kin_hist, kinetic_com=kin_com_hist,
        potential=pot_hist, virial_work=vir_hist,
        energy=kin_hist + pot_hist,
        momentum=mom_hist, n_steps_taken=n_steps, masses=masses,
        softening=softening, method=method, theta=theta, dt=dt,
    )
    if track_dense:
        result["r50_dense"] = r50_dense
        result["virial_ratio_dense"] = qvir_dense
    return result


# ======================================================================
# Chaos / sensitivity-to-initial-conditions diagnostics
# ======================================================================
#: How far the ACTUAL realized RMS displacement (after floating-point
#: addition to the position array) is allowed to drift, as a fraction of
#: the requested target RMS, before perturb_positions() raises rather than
#: silently returning a degraded perturbation. A requested
#: relative_perturbation at the extreme small end -- roughly 1e-16 to
#: 1e-17, depending on the specific position values and random offsets
#: involved -- approaches float64's own rounding resolution at the scale
#: of the affected coordinates, so the realized RMS can drift measurably
#: from the requested target through ordinary rounding alone, with no bug
#: involved; this tolerance is loose enough to accept that
#: rounding-scale drift while still catching a genuinely degraded
#: perturbation (for example, one that silently realizes as far smaller
#: than requested, or as zero). This boundary is inherently seed-dependent
#: (the specific random offset draw interacts with each position
#: coordinate's own floating-point resolution), so a given
#: relative_perturbation near this range may realize acceptably for one
#: seed and not another.
_PERTURBATION_REPRESENTABILITY_TOLERANCE = 0.10


def perturb_positions(positions, relative_perturbation, masses=None, seed=None):
    """
    Return a copy of ``positions`` with every body's position displaced by
    an independent, isotropic random offset, such that the RMS per-body
    displacement magnitude -- sqrt(mean_i |offset_i|^2) -- equals
    ``relative_perturbation`` times the RMS radius of the configuration
    (about its own centroid).

    The random offsets are drawn isotropically (an independent standard
    Gaussian per Cartesian component, so their directions are uniform on
    the sphere in the large-N limit) and, if ``masses`` is given,
    recentered to remove their own mass-weighted mean before being added,
    so the perturbation does not impart a net center-of-mass shift to the
    perturbed copy relative to the original -- otherwise the subsequent
    divergence measurement would partly be tracking a spurious coherent
    translation rather than the system's internal chaotic divergence
    (see position_space_divergence()). The whole offset array (after any
    such recentering) is then rescaled by a single overall constant so
    that ITS OWN realized RMS vector magnitude equals
    ``relative_perturbation`` times the RMS radius of the configuration
    EXACTLY, not merely in expectation over the random draw -- this
    matters because the recentering step for the masses= path changes the
    realized RMS relative to the raw, uncentered draw by an amount that
    depends on N and the mass distribution, which a fixed per-component
    sigma alone cannot correct for.

    That exact-rescale guarantee is about the offset array in isolation.
    The value actually returned is ``positions + offset``, and floating-
    point addition can round away part or all of a sufficiently small
    offset once it is added to a much larger position coordinate (a
    ``relative_perturbation`` pushed toward the position array's own
    double-precision resolution, exactly what EXP-12 asks students to
    explore, can do this). This function therefore RE-MEASURES the
    achieved RMS from ``(positions + offset) - positions`` -- the actual
    representable displacement -- rather than trusting the pre-addition
    offset array, and raises a clear, actionable ValueError before
    returning if that representable displacement has drifted more than
    ``_PERTURBATION_REPRESENTABILITY_TOLERANCE`` (a fixed fraction, see
    below) from the requested target, rather than silently returning an
    almost-unperturbed (or exactly unperturbed) copy that a caller such as
    run_chaos() would otherwise integrate and only fail on much later,
    deep inside estimate_lyapunov_exponent(), with no indication that the
    real cause was an unrepresentable perturbation request.

    This follows the classic gravitational N-body sensitivity experiment
    (Miller 1964, ApJ 140, 250): two realizations that start indistinguishably
    close together, evolved under exactly the same equations of motion and
    force method, separate exponentially -- the N-body problem is chaotic
    even though the underlying dynamics are entirely deterministic.
    """
    # _require_snapshot(), not the generic finite-array check, so that a
    # wrong-shaped positions array (an (N, 2) array of "vectors" that
    # aren't 3-component, or a 1-D array with no per-body axis at all --
    # which np.einsum's "ij,ij->i" reduction below would otherwise raise
    # a raw, unhelpful AxisError for) is rejected here with a clear
    # message instead.
    positions = _require_snapshot(positions, "positions")
    if positions.shape[0] == 0:
        raise ValueError(
            "positions must contain at least one body; got an empty array."
        )
    relative_perturbation = _require_positive(
        "relative_perturbation", relative_perturbation
    )
    # Every RMS/centroid reduction below goes through this module's own
    # scale-safe helpers
    # (_scale_safe_sum/_scale_safe_vector_norm/_scale_safe_rms), not raw
    # mean/sum/**2/sqrt -- this function makes exactly the same
    # extreme-but-representable-scale promises the rest of the module
    # does (chaos mode's positions can be at pc-to-kpc-in-meters scale,
    # and EXP-12 explicitly asks students to push relative_perturbation
    # toward representability limits), so it must not silently overflow
    # a partial sum-of-squares or a naive centroid sum that the fully
    # combined, representable result never would.
    # Each body's position is divided by n BEFORE the scale-safe sum, not
    # after: _scale_safe_sum() rescues a sum whose true, fully-combined
    # value is itself representable, but for many extreme-but-
    # individually-finite coordinates of the same sign (e.g. n bodies
    # all near +1e308) the raw SUM can genuinely exceed float64's range
    # even though the MEAN (the actual quantity needed here) does not --
    # no rescaling after the fact can recover a true sum that was never
    # representable to begin with. Dividing first makes the quantity
    # actually being summed the mean's own per-body contribution, whose
    # true total is the representable mean.
    centroid = _scale_safe_sum(positions / positions.shape[0], axis=0)
    # _scale_safe_vector_norm() first (per-body distance from centroid,
    # safe against a single component's square overflowing), then
    # _scale_safe_rms() over those already-reduced per-body distances
    # (safe against the cross-body mean-of-squares overflowing) --
    # together exactly reproducing sqrt(mean(sum((positions-centroid)**2,
    # axis=1))) without ever squaring a raw, unscaled coordinate.
    rms_radius = float(_scale_safe_rms(_scale_safe_vector_norm(positions - centroid), axis=0))
    if rms_radius == 0.0:
        rms_radius = 1.0
    rng = np.random.default_rng(seed)
    offset = rng.normal(size=positions.shape)
    if masses is not None:
        masses = _require_masses(masses, n_bodies=positions.shape[0])
        offset = offset - center_of_mass(offset, masses)
    achieved_rms = float(_scale_safe_rms(_scale_safe_vector_norm(offset), axis=0))
    if achieved_rms == 0.0:
        raise RuntimeError(
            "the random offset draw was degenerate (all-zero, including "
            "after center-of-mass removal); this should not happen for "
            "any valid input and indicates an unlucky draw rather than a "
            "supported degenerate case -- retry with a different seed."
        )
    target_rms = relative_perturbation * rms_radius
    if not math.isfinite(target_rms):
        raise ValueError(
            "the target RMS displacement (relative_perturbation * "
            "rms_radius) overflowed; check that relative_perturbation "
            "and positions are physically reasonable."
        )
    scale_factor = target_rms / achieved_rms
    if not math.isfinite(scale_factor):
        raise ValueError(
            "the perturbation scaling factor (target_rms / achieved_rms) "
            "overflowed or is not representable; check that "
            "relative_perturbation and positions are physically "
            "reasonable."
        )
    offset = offset * scale_factor

    new_positions = positions + offset
    realized_displacement = new_positions - positions
    realized_rms = float(_scale_safe_rms(_scale_safe_vector_norm(realized_displacement), axis=0))
    relative_error = (
        abs(realized_rms - target_rms) / target_rms if target_rms > 0.0 else float("inf")
    )
    if realized_rms == 0.0 or relative_error > _PERTURBATION_REPRESENTABILITY_TOLERANCE:
        raise ValueError(
            "relative_perturbation is too small to survive floating-point "
            f"addition at this position scale: the requested RMS "
            f"displacement is {target_rms:.6e} (position units), but "
            f"positions + offset only realized {realized_rms:.6e} once "
            "rounded to double precision -- more than "
            f"{_PERTURBATION_REPRESENTABILITY_TOLERANCE:.0%} off target "
            "(realized_rms == 0 means it vanished completely). Use a "
            "larger relative_perturbation; this is expected once the "
            "requested displacement approaches the position array's own "
            "double-precision resolution (typically relative_perturbation "
            "below roughly 1e-16 to 1e-17 for pc-to-kpc-scale coordinates)."
        )
    return new_positions


def position_space_divergence(positions_a, positions_b, masses=None):
    """
    RMS per-body position separation (meters) between two realizations of
    the same system, sqrt( mean_i |x_a,i - x_b,i|^2 ).

    This measures separation in POSITION space only (velocities are not
    involved at all); a genuine phase-space distance would need a
    velocity term too (with some, inherently arbitrary, choice of
    relative weighting between position and velocity units). If
    ``masses`` is given, each realization is recentered to its own
    mass-weighted center of mass before differencing, removing any
    coherent center-of-mass translation between the two realizations
    (e.g. from perturb_positions() displacing the system's own centroid,
    or from the tree method's imperfect momentum conservation letting the
    two realizations' centroids drift apart) from what should be a
    measurement of internal, relative chaotic divergence.
    """
    positions_a = _as_finite_array(positions_a, "positions_a")
    positions_b = _as_finite_array(positions_b, "positions_b")
    if positions_a.shape != positions_b.shape:
        raise ValueError("positions_a and positions_b must have the same shape.")
    if positions_a.ndim < 2 or positions_a.shape[-1] != 3:
        raise ValueError(
            "positions_a/positions_b must have shape (..., n_bodies, 3); "
            f"got {positions_a.shape}."
        )
    if positions_a.shape[-2] == 0:
        raise ValueError(
            "positions_a/positions_b must contain at least one body; got "
            f"shape {positions_a.shape} (an empty body axis would otherwise "
            "silently produce nan from a 0/0 mean, with a RuntimeWarning, "
            "rather than this explicit error)."
        )
    if masses is not None:
        # Recenter per snapshot without assuming positions_a/positions_b
        # are a single (n_bodies, 3) snapshot rather than a whole
        # (n_snapshots, n_bodies, 3) history -- center_of_mass() only
        # handles the former, so the mass-weighted mean over the
        # body axis (second-to-last) is computed directly here, with
        # keepdims so it broadcasts against either shape.
        masses = _as_finite_array(masses, "masses")
        if masses.ndim != 1 or masses.size != positions_a.shape[-2]:
            raise ValueError(
                "masses must be a 1-D array with one entry per body, "
                f"matching positions_a.shape[-2] = {positions_a.shape[-2]}; "
                f"got shape {masses.shape}."
            )
        if np.any(masses <= 0.0):
            raise ValueError("all masses must be strictly positive.")
        # See center_of_mass() for why masses are normalized by their
        # maximum before summing (avoids overflowing mass_sum), and why
        # the weighted-sum numerator is a single fused
        # _scaled_product_over() call across (masses, positions, divided
        # by max_mass) rather than a separately materialized per-body
        # weight array -- an extreme-mass-ratio underflow applies here
        # exactly as it does in center_of_mass().
        max_mass = float(np.max(masses))
        weight_sum = np.sum(masses / max_mass)
        # _scale_safe_sum, not a plain np.sum, over the per-body axis --
        # see center_of_mass()'s docstring for why a partial sum across
        # bodies can overflow even when the true, cancellation-involving
        # COM total is representable.
        com_a = _scale_safe_sum(
            _scaled_product_over(max_mass, masses[:, None], positions_a),
            axis=-2, keepdims=True,
        ) / weight_sum
        com_b = _scale_safe_sum(
            _scaled_product_over(max_mass, masses[:, None], positions_b),
            axis=-2, keepdims=True,
        ) / weight_sum
        positions_a = positions_a - com_a
        positions_b = positions_b - com_b
    # The per-body distance and the RMS across bodies are each computed
    # in a scale-safe way (see _scale_safe_vector_norm/_scale_safe_rms)
    # rather than via a direct sum-of-squares-then-sqrt: for a per-body
    # separation of order 1e200, squaring a component alone overflows
    # float64 even though the true RMS separation is comfortably
    # representable.
    per_body_distance = _scale_safe_vector_norm(positions_a - positions_b)
    return _scale_safe_rms(per_body_distance, axis=-1)


def estimate_lyapunov_exponent(t, divergence, min_points=5, min_r_squared=0.90,
                                min_window_efolds=math.log(10.0),
                                max_curvature_t_statistic=10.0,
                                curvature_n_bins=20):
    """
    Estimate an exponential growth rate lambda from a divergence time
    series by an ordinary least-squares fit of ln(divergence) against t,
    restricted to a single CONTIGUOUS window where growth is plausibly
    exponential, and only accepted if that window spans enough dynamic
    range, fits well enough in log space, and does not show excessive
    quadratic curvature, to trust as a genuine exponential-growth
    measurement rather than a short, coincidentally log-linear-looking
    stretch of some other curve.

    This is a heuristic gate, not a hypothesis test with a rigorously
    calibrated p-value: it is tuned to discriminate genuine (possibly
    noisy) exponential growth from several specific, measured families of
    smooth non-exponential curves (see check 6 below for exactly which
    ones, and which one it is NOT reliably able to reject), but a
    sufficiently pathological adversarial input could still slip through,
    and a genuinely chaotic but short or unusually smooth divergence
    trace can still fail every check and be reported as "no fit" even
    though the underlying dynamics are chaotic -- a known limitation of
    this heuristic, not a bug (see check 6 below for the specific shape
    this does and does not catch, and the measured rate at which it is
    fooled by an ordinary logistic curve).

    1. Amplitude window: only points with divergence in [3*d0, 0.5*d_max]
       (where d0 is the initial divergence and d_max the run's maximum)
       are considered, since points below 3*d0 are dominated by the
       arbitrary initial offset and points above 0.5*d_max are typically
       already saturating toward the system size rather than still
       growing exponentially.
    2. Contiguity: only the single LONGEST contiguous (in time index) run
       of in-window points is used, not a scattered subset -- this alone
       rejects an oscillating series, which cannot sustain a long
       contiguous run through a monotonically-defined amplitude window.
    3. Minimum size: at least ``min_points`` points (default 5) must fall
       in that run, since fewer are too few to assess fit quality at all.
    4. Minimum dynamic range: the window's own log-amplitude span,
       max(ln divergence) - min(ln divergence) over the window, must
       reach at least ``min_window_efolds`` (default ln(10) approx 2.303,
       i.e. the window must cover at least one order of magnitude of
       growth). This is a standard requirement in estimating exponential
       (Lyapunov) growth rates from finite data (see the logistic
       discussion above for why it is needed at all). Unlike a
       significance test, this is a plain ratio in log space, not a
       statistic whose apparent strength grows with the sample size, so
       it does not develop the same false-rejection problem at large
       n_used that a pure significance test does. For this program's own
       dynamics: a representative real chaos-mode run's selected window
       typically spans roughly 16-17 e-folds, comfortably above the
       default threshold; a saturating (logistic-like) curve's early,
       still-plausibly-exponential-looking phase spans only about 2.1
       e-folds, well below it; and a modest but genuine exponential can
       still span about 3.2, comfortably above it.
    5. Whole-window fit quality: the OLS fit of ln(divergence) against t
       over that window must reach R^2 >= ``min_r_squared`` (default
       0.90) -- a basic sanity floor that rejects a window with no clear
       overall log-linear trend at all (e.g. divergence dominated by
       noise with no net growth, or a strongly curved trend such as
       linear or quadratic growth in divergence itself rather than in its
       logarithm).
    6. Curvature significance: a SEPARATE quadratic-in-t regression,
       ln(divergence) = c0 + c1*t + c2*t^2, is fit and the window is
       rejected if the quadratic coefficient's OLS t statistic exceeds
       ``max_curvature_t_statistic`` (default 10.0) in magnitude -- this
       catches a genuinely different growth LAW throughout the window
       (e.g. a stretched/compressed exponential, d(t) = exp(A*(t/T)^p)
       for p far from 1), which check 4 does not, since such a curve can
       accumulate more than min_window_efolds of dynamic range while
       still clearing R^2 >= min_r_squared.

       The fit is NOT done on the window's raw points. They are first
       averaged into ``curvature_n_bins`` (default 20) equal-sized,
       time-ordered groups (or ``n_used`` bins, whichever is fewer, for a
       short window), and the quadratic regression (t centered and
       scaled for numerical conditioning) is fit to those bin means
       instead. This binning step exists because an OLS t statistic's
       magnitude grows with its number of independent data points, but
       the raw points of a divergence trace are NOT independent draws --
       they are dense, serially-correlated samples of one smooth
       underlying curve -- so fitting the raw points made this
       statistic's apparent strength grow mechanically with how densely
       a run happened to be recorded (``target_snapshots``), flagging an
       UNCHANGED physical trajectory as more "curved" purely because more
       of it was stored, with no genuine shape information behind the
       increase. Averaging down to a fixed number of representative bins
       removes that spurious sample-count dependence (the real physical
       curve's shape converges to the same ~20 bin means regardless of
       how many raw points were stored), while a curve with genuine
       systematic curvature survives bin-averaging just as well as, or
       better than, it survives a raw-point fit, since binning
       averages down within-bin noise without averaging away a
       consistent trend.

       A numerically degenerate quadratic fit (essentially zero residual
       variance) is treated as showing no significant curvature rather
       than raising a division-by-zero, since a fit that close to perfect
       is, if anything, stronger evidence of a clean exponential, not
       weaker.

       Behavior of this binned statistic (curvature_n_bins=20): a
       genuinely chaotic N-body divergence trace (a representative
       N=40-body configuration at default parameters) typically scores
       well under 10 in magnitude; a family of stretched/compressed-
       exponential curves, d(t) = exp(A*(t/T)^p) for p appreciably
       different from 1 with a few percent of multiplicative noise,
       typically scores at least about 12 in magnitude -- a comfortable,
       non-overlapping margin on both sides of the default threshold. A
       noisy quadratic growth curve, d(t) proportional to (1+t^2), is
       reliably rejected by this binned design.

       KNOWN REMAINING LIMITATION: a logistic (saturating) curve is, by
       construction, asymptotically exponential during its early growth,
       so no shape statistic computed purely within a narrow amplitude
       window -- binned or not -- can in principle always distinguish a
       logistic's early-growth window from a true exponential; only
       requiring more dynamic range than a logistic's early phase can
       sustain (check 4) can do that in general, and check 4 only rejects
       a logistic whose selected window fails to reach min_window_efolds.
       A sufficiently tall logistic curve, d(t) = (1 + 99/(1+exp(-(t-5))))
       *(1 + a few percent noise), whose selected window happens to clear
       that dynamic-range floor (roughly 69% of realizations of such a
       curve do) can still show too little quadratic curvature, within
       that specific narrow window, for this check to catch (about 92%
       of the time, once that floor is cleared) -- i.e. once this
       logistic's window clears the dynamic-range gate, the curvature
       check catches it only rarely. Overall, roughly 64% of realizations
       of this specific logistic family are accepted by the whole gate
       end to end. This is a real, non-negligible gap, not a rare edge
       case -- see NbodyGalaxySimulator.html's Known Model Artefacts
       section for the same caveat stated for students.

    Returns a dict with 'lyapunov_exponent' (1/s), 'lyapunov_time' (s,
    = 1/lambda), 'n_points_used', 'window_log_amplitude_span' (the
    log-amplitude span measured in check 4, or nan if rejected before
    that check runs), 'r_squared' (whole-window linear fit quality, check
    5, or nan if rejected before it runs), 'curvature_t_statistic' (the
    signed t statistic from check 6, or nan if rejected before it runs),
    and 'fit_start_index' / 'fit_stop_index' (the half-open [start, stop)
    slice of the input arrays actually used for the fit, or None if no
    window passed every gate). All numeric fields are nan (and
    lyapunov_time is left nan, not inf) when no window passes every gate.
    """
    t = _as_finite_array(t, "t")
    divergence = _as_finite_array(divergence, "divergence")
    if t.ndim != 1 or divergence.ndim != 1:
        raise ValueError(
            f"t and divergence must both be one-dimensional; got shapes "
            f"{t.shape} and {divergence.shape}."
        )
    if t.shape != divergence.shape:
        raise ValueError("t and divergence must have the same shape.")
    if np.any(divergence < 0.0):
        raise ValueError("divergence must be non-negative.")
    if t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing with at least two points.")
    min_points = _require_int("min_points", min_points, lo=4)
    min_r_squared = _require_finite("min_r_squared", min_r_squared)
    if not (0.0 <= min_r_squared <= 1.0):
        raise ValueError("min_r_squared must be in [0, 1].")
    min_window_efolds = _require_positive("min_window_efolds", min_window_efolds)
    max_curvature_t_statistic = _require_positive(
        "max_curvature_t_statistic", max_curvature_t_statistic
    )
    curvature_n_bins = _require_int("curvature_n_bins", curvature_n_bins, lo=4)

    def _rejected(n_used, window_efolds=float("nan"), r_squared=float("nan"),
                  curvature_t=float("nan")):
        return dict(lyapunov_exponent=float("nan"), lyapunov_time=float("nan"),
                    n_points_used=n_used, window_log_amplitude_span=window_efolds,
                    r_squared=r_squared, curvature_t_statistic=curvature_t,
                    fit_start_index=None, fit_stop_index=None)

    d0 = divergence[0]
    if d0 <= 0.0:
        raise ValueError("divergence[0] must be positive to define a growth window.")
    d_max = float(divergence.max())
    in_window = (divergence >= 3.0 * d0) & (divergence <= 0.5 * d_max) & (divergence > 0.0)

    # Longest contiguous (in time index) run of in-window points.
    runs = []
    start = None
    for i, flag in enumerate(in_window):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, in_window.size))
    if not runs:
        return _rejected(0)
    lo, hi = max(runs, key=lambda r: r[1] - r[0])
    n_used = hi - lo
    if n_used < min_points:
        return _rejected(n_used)

    tw, dw = t[lo:hi], divergence[lo:hi]
    logd = np.log(dw)

    window_efolds = float(np.max(logd) - np.min(logd))
    if window_efolds < min_window_efolds:
        return _rejected(n_used, window_efolds=window_efolds)

    slope, intercept = np.polyfit(tw, logd, 1)
    lin_fit = slope * tw + intercept
    lin_resid = logd - lin_fit
    ss_res = float(np.sum(lin_resid ** 2))
    ss_tot = float(np.sum((logd - np.mean(logd)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if r_squared < min_r_squared:
        return _rejected(n_used, window_efolds=window_efolds, r_squared=r_squared)

    # Curvature significance (check 6, see docstring): a quadratic-in-t
    # regression fit over BINNED window means, gating the fit alongside
    # check 4 (minimum dynamic range) rather than replacing it -- each
    # catches a negative-control family the other does not.
    #
    # Binning first: the raw
    # window points tw/logd are dense, serially-correlated samples of one
    # smooth underlying curve, not independent draws, so an OLS t
    # statistic fit directly to them grows mechanically with n_used --
    # i.e. with how densely a run happened to be recorded
    # (target_snapshots) -- even for an UNCHANGED physical trajectory.
    # Averaging the window down to a fixed number of time-ordered,
    # equal-sized bins before fitting removes that spurious sample-count
    # dependence (see check 6 in the docstring above for why raw,
    # serially-correlated points inflate this statistic and binning
    # removes that inflation without losing genuine curvature).
    n_bins = min(n_used, curvature_n_bins)
    bin_edges = np.array_split(np.arange(n_used), n_bins)
    t_bin = np.array([tw[idx].mean() for idx in bin_edges])
    logd_bin = np.array([logd[idx].mean() for idx in bin_edges])

    # t is centered and scaled before building the design matrix purely
    # for numerical conditioning -- this project's real time arrays are in
    # raw seconds (order 1e14-1e15), and fitting t and t^2 directly at
    # that scale makes the design matrix so ill-conditioned that
    # ``numpy.linalg.lstsq`` silently returns a rank-deficient (and
    # therefore meaningless) result. Centering/scaling t by an invertible
    # affine map leaves the quadratic term's t statistic mathematically
    # unchanged (it is invariant under such a reparametrization) while
    # keeping the fit numerically well-posed.
    t_center = float(np.mean(t_bin))
    t_scale = float(np.std(t_bin))
    if t_scale <= 0.0 or not math.isfinite(t_scale):
        t_scale = 1.0
    t_bin_scaled = (t_bin - t_center) / t_scale
    design = np.column_stack([np.ones(n_bins), t_bin_scaled, t_bin_scaled ** 2])
    quad_coef, _, quad_rank, _ = np.linalg.lstsq(design, logd_bin, rcond=None)
    quad_resid = logd_bin - design @ quad_coef
    quad_dof = n_bins - 3
    quad_rss = float(np.sum(quad_resid ** 2))
    curvature_t = 0.0
    if quad_rank == 3 and quad_dof > 0 and quad_rss > 0.0:
        sigma2 = quad_rss / quad_dof
        try:
            xtx_inv = np.linalg.inv(design.T @ design)
        except np.linalg.LinAlgError:
            xtx_inv = None
        if xtx_inv is not None:
            se_c2 = math.sqrt(sigma2 * xtx_inv[2, 2])
            if se_c2 > 0.0 and math.isfinite(se_c2):
                curvature_t = float(quad_coef[2] / se_c2)
    # A numerically degenerate or as-good-as-exact quadratic fit (rss <= 0,
    # or a singular design matrix) leaves curvature_t at its 0.0 default,
    # which is correctly treated as "no significant curvature detected"
    # below (a fit that close to perfect is, if anything, stronger evidence
    # of a clean exponential, not weaker).
    if abs(curvature_t) > max_curvature_t_statistic:
        return _rejected(n_used, window_efolds=window_efolds, r_squared=r_squared,
                          curvature_t=curvature_t)

    if slope <= 0.0:
        return _rejected(n_used, window_efolds=window_efolds, r_squared=r_squared,
                          curvature_t=curvature_t)

    lam = float(slope)
    return dict(lyapunov_exponent=lam, lyapunov_time=1.0 / lam,
                n_points_used=n_used, window_log_amplitude_span=window_efolds,
                r_squared=r_squared, curvature_t_statistic=curvature_t,
                fit_start_index=int(lo), fit_stop_index=int(hi))


# ======================================================================
# Top-level per-mode simulations
#
# These functions are the "run this mode" analogue of integrate_track()
# and build_hr_grid() in physics_sev.py: they build initial conditions,
# choose numerically sound defaults for the timestep and run length from
# the physical parameters themselves (rather than asking the student to
# already know a sensible dt), run the shared leapfrog engine above, and
# reduce the resulting snapshots to the diagnostics each mode cares about.
# driver_nbg.py calls exactly one of these per run; it does not touch the
# integrator directly.
# ======================================================================
LAGRANGIAN_FRACTIONS_DEFAULT = (0.1, 0.25, 0.5, 0.75, 0.9)


def _pick_stride(n_steps, target_snapshots):
    """
    Ceiling division keeps the realized snapshot count close to
    ``target_snapshots`` across the full range of n_steps, while still
    always storing at least the initial and final steps -- plain floor
    division would let the actual count run to nearly DOUBLE
    target_snapshots whenever n_steps fell just under
    2 * target_snapshots (e.g. n_steps=199, target_snapshots=100 gives
    floor(199 / 100) = 1, i.e. close to 200 snapshots instead of ~100).
    """
    target_snapshots = max(1, target_snapshots)
    return max(1, -(-n_steps // target_snapshots))  # ceil(n_steps / target)


def _lagrangian_track(pos_hist, masses, fractions):
    """(n_snap, n_frac) array of Lagrangian radii through a snapshot history."""
    n_snap = pos_hist.shape[0]
    out = np.empty((n_snap, len(fractions)))
    for k in range(n_snap):
        radii = lagrangian_radii(pos_hist[k], masses, fractions)
        out[k] = [radii[f] for f in fractions]
    return out


def _virial_track(kinetic, virial_work):
    """
    virial_work must be the Wvir series from virial_force_term(), not the
    potential-energy series -- see virial_ratio()'s docstring.
    """
    return np.array([virial_ratio(k, w) for k, w in zip(kinetic, virial_work)])


def _energy_drift(energy):
    """
    Maximum fractional total-energy drift relative to the initial value,
    evaluated only at the SAMPLED snapshots kept in ``energy`` (spaced by
    the run's snapshot stride), not at every integration step. A genuine
    energy excursion that occurs and partly relaxes between two kept
    snapshots would not be reflected here; this is a lower bound on the
    true max drift over the full integration, not an exact one. Reported
    to the user as "max SAMPLED energy drift" for this reason.
    """
    e0 = energy[0]
    if e0 == 0.0:
        return float("nan")
    return float(np.max(np.abs((energy - e0) / e0)))


LATE_WINDOW_FRACTION = 0.20
#: Minimum number of DENSE (every-integration-step, not stored-snapshot)
#: diagnostic samples required inside the late-time window; named for
#: what it actually counts (see _late_time_window_stats's own
#: min_samples parameter) rather than the stored/CSV snapshot cadence
#: target_snapshots controls, which is unrelated to this threshold.
LATE_WINDOW_MIN_DENSE_SAMPLES = 5
LATE_WINDOW_R50_RANGE_THRESHOLD = 0.30
LATE_WINDOW_Q_RANGE_THRESHOLD = 0.60
#: Bound on abs(r50_relative_drift) (a best-fit linear trend over the late
#: window, as a fraction of the window-mean r50) -- catches a MONOTONIC
#: expansion/contraction trend that a pure range check can miss (a straight
#: line has a large range-to-mean ratio too, but "range" alone does not
#: distinguish a one-way trend from noisy oscillation around a plateau;
#: this threshold specifically targets the trend itself). Calibrated
#: below a synthetic monotonic-expansion counterexample's fitted drift
#: of 0.2456 (which must fail) and above the drift
#: actually measured across several seeded default-scale
#: (n_bodies=300, n_freefall=8) run_galaxy() calls, which stayed in
#: 0.02-0.16.
LATE_WINDOW_DRIFT_THRESHOLD = 0.20
#: Bound on abs(mean(Q) - 1) over the late window -- the scalar virial
#: condition is Q = 1, not merely "Q holds still" (a synthetic
#: Q=5-constant counterexample has virial_ratio_range=0 yet is
#: nowhere near virialized). Calibrated above the late-window mean-Q
#: deviation actually measured across several seeded default-scale
#: (n_bodies=300, n_freefall=8) run_galaxy() calls (0.04-0.12) and well
#: below both the Q=5 synthetic case (4.0) and the real still-collapsing
#: N=20/n_freefall=0.75 counterexample (0.57-0.59).
LATE_WINDOW_Q_CENTER_TOLERANCE = 0.25
#: Minimum late-window-mean-r50 / global-minimum-r50 ratio required to
#: call a run's collapse "materially rebounded" rather than "still sitting
#: at its deepest point." Calibrated far below the ratios actually
#: measured for genuinely relaxed default-scale runs (2.7-8.9x) and just
#: above the ratio for a run that has not yet turned around (the real
#: N=20/n_freefall=0.75 counterexample sits at 1.06x).
LATE_WINDOW_REBOUND_RATIO = 1.10
#: Maximum global-minimum-r50 / initial-r50 ratio still allowed to call
#: the run's global minimum a genuine COLLAPSE rather than merely "the
#: smallest value seen so far," which a purely monotonically EXPANDING
#: r50 series also satisfies at its very first sample (global_min_idx=0,
#: technically "before" any window that starts later). Requiring the
#: minimum to fall at least this far below the INITIAL r50 (not merely
#: to occur at some index > 0) additionally rejects a series that dips by
#: a negligible amount and immediately resumes expanding -- collapse
#: implies a MATERIAL contraction, not an arbitrarily small one. 0.90
#: (a required >=10% contraction) sits comfortably below the contraction
#: measured in genuine seeded galaxy collapses (typically shrinking to a
#: small fraction of the initial radius) and rejects a synthetic series
#: that only ever expands (10 -> 12.8 pc, ratio 1.0, i.e. no contraction
#: at all).
LATE_WINDOW_MIN_COLLAPSE_CONTRACTION = 0.90


def _late_time_window_stats(t, r50_series, virial_series,
                             window_fraction=LATE_WINDOW_FRACTION,
                             min_samples=LATE_WINDOW_MIN_DENSE_SAMPLES):
    """
    Summarize the LAST ``window_fraction`` (by elapsed time, not sample
    count) of a run's own half-mass-radius and virial-ratio history, as
    the basis for judging whether a run's late-time behavior is actually
    settled.

    ``t``, ``r50_series`` and ``virial_series`` are expected to be the
    DENSE, every-integration-step diagnostic arrays (integrate_nbody()'s
    ``track_dense=True`` output; see its docstring), not the sparse,
    ``target_snapshots``-dependent snapshot series -- so that every
    quantity computed here, including the two range statistics that
    predate this function, is by construction independent of how many
    snapshots happened to be STORED for plotting/CSV output (two runs
    with bit-for-bit identical integrated positions and velocities,
    differing only in target_snapshots, are guaranteed the same
    late-time verdict, because they are built from the same dense
    arrays regardless of target_snapshots).

    A settled verdict requires ALL of the following:
      1. A genuine collapse before the window: the global minimum of
         r50 over the ENTIRE run (not just the window) must (a) occur
         strictly before the window starts -- a run whose smallest
         radius is still its most recent one has not turned around yet,
         regardless of how flat its own tail looks -- AND (b) occur at
         an index strictly after the FIRST sample, at a value at least
         LATE_WINDOW_MIN_COLLAPSE_CONTRACTION below the initial r50.
         Without (b), a purely monotonically EXPANDING r50 series
         satisfies a bare "minimum occurs before the window" test
         trivially, at its own first sample (global_min_idx=0 is
         "before" any later window) -- that is the smallest value SEEN
         SO FAR, not a collapse; genuine collapse requires the radius to
         have first contracted materially below where it started, not
         merely to have never yet been smaller than it is right now.
      2. A material rebound: the window-mean r50 must exceed that global
         minimum by at least LATE_WINDOW_REBOUND_RATIO -- otherwise a
         minimum reached one sample before the window, followed by
         negligible recovery, would satisfy (1) without having actually
         rebounded.
      3. The window-mean virial ratio Q must be centered near 1 (within
         LATE_WINDOW_Q_CENTER_TOLERANCE) -- a quiet series is not
         evidence of virial equilibrium unless it is quiet AROUND Q=1;
         the scalar virial condition is Q=1, not "Q roughly constant."
      4. A bounded secular trend: abs(r50_relative_drift) (see below)
         must stay within LATE_WINDOW_DRIFT_THRESHOLD -- catches a
         monotonic expansion/contraction that a pure range check can
         miss, since a straight line has a large range too.
      5. The existing range bounds: r50_fractional_range and
         virial_ratio_range must each stay within their documented
         "modest" thresholds.
      6. At least min_samples dense points fall in the window.

    Returns a dict with:
      n_samples                 -- dense points falling in the late window.
      has_enough_samples        -- n_samples >= min_samples; when False,
                                    every other field except this one and
                                    window_start_myr/window_fraction is
                                    nan/False, and no equilibrium claim
                                    should be made.
      window_start_myr          -- start time of the late window.
      global_min_r50_myr        -- time at which r50 reaches its smallest
                                    value over the WHOLE run (not just the
                                    window).
      collapse_before_window    -- True iff that global minimum occurs
                                    strictly before the window starts,
                                    strictly after the run's first
                                    sample, and at least
                                    LATE_WINDOW_MIN_COLLAPSE_CONTRACTION
                                    below the initial r50 -- i.e. a
                                    genuine, material contraction, not
                                    merely "the smallest value seen so
                                    far" (which a monotonically
                                    expanding series also satisfies,
                                    trivially, at its own first sample).
      r50_rebound_ratio         -- window-mean r50 divided by the global
                                    minimum r50; how far the run has
                                    recovered from its deepest collapse.
      r50_fractional_range      -- (max-min)/mean of r50 over the window;
                                    large values indicate the half-mass
                                    radius is still swinging by an amount
                                    comparable to its own size, not merely
                                    fluctuating around a settled value.
      r50_relative_drift        -- best-fit linear slope of r50 over the
                                    window, expressed as a FRACTION of the
                                    window-mean r50 gained or lost over
                                    the full window duration; large values
                                    indicate a sustained secular trend
                                    (still expanding or still contracting)
                                    rather than a plateau.
      virial_ratio_range        -- max-min of the virial ratio Q over the
                                    window; large values indicate Q is
                                    still oscillating violently rather
                                    than settling into a modest band.
      virial_ratio_mean         -- mean of Q over the window.
      virial_ratio_mean_deviation -- abs(virial_ratio_mean - 1).
      is_settled                -- True only when every criterion (1)-(6)
                                    above holds -- i.e. the minimum bar
                                    for describing a run as consistent
                                    with a genuine collapse followed by a
                                    sustained, virialized quasi-
                                    equilibrium remnant, not proof of
                                    genuine phase-space stationarity (see
                                    virial_force_term()'s docstring on
                                    what a virial-balance diagnostic does
                                    and does not establish).
    """
    t = np.asarray(t, dtype=float)
    r50_series = np.asarray(r50_series, dtype=float)
    virial_series = np.asarray(virial_series, dtype=float)
    total_time = float(t[-1] - t[0])
    window_start = t[-1] - window_fraction * total_time if total_time > 0.0 else t[0]
    mask = t >= window_start
    n_samples = int(np.sum(mask))
    out = dict(
        n_samples=n_samples,
        has_enough_samples=n_samples >= min_samples,
        window_start_myr=float(window_start / MYR),
        window_fraction=window_fraction,
        global_min_r50_myr=float("nan"),
        collapse_before_window=False,
        r50_rebound_ratio=float("nan"),
        r50_fractional_range=float("nan"),
        r50_relative_drift=float("nan"),
        r50_linear_slope_pc_per_myr=float("nan"),
        virial_ratio_range=float("nan"),
        virial_ratio_mean=float("nan"),
        virial_ratio_mean_deviation=float("nan"),
        is_settled=False,
    )
    if not out["has_enough_samples"]:
        return out

    t_win = t[mask]
    r50_win = r50_series[mask]
    q_win = virial_series[mask]
    window_start_idx = int(np.argmax(mask))  # first True index (mask is a suffix)

    if np.all(np.isfinite(r50_series)) and r50_series.size > 0:
        global_min_idx = int(np.argmin(r50_series))
        global_min_r50 = float(r50_series[global_min_idx])
        r50_initial = float(r50_series[0])
        out["global_min_r50_myr"] = float(t[global_min_idx] / MYR)
        # Three conditions, all required: the minimum occurs before the
        # window (a run whose smallest radius is still its most recent
        # one has not turned around yet); it occurs strictly after the
        # very first sample (index 0 trivially satisfies "before the
        # window" for ANY series, collapsing or not); and it sits
        # materially below the initial r50 (not merely a negligible dip
        # immediately followed by more expansion). See
        # LATE_WINDOW_MIN_COLLAPSE_CONTRACTION's own comment for why (b)
        # and (c) are both needed to reject a purely, monotonically
        # expanding r50 series.
        out["collapse_before_window"] = bool(
            0 < global_min_idx < window_start_idx
            and r50_initial > 0.0
            and global_min_r50 <= LATE_WINDOW_MIN_COLLAPSE_CONTRACTION * r50_initial
        )
        r50_win_mean = float(np.mean(r50_win))
        if global_min_r50 > 0.0:
            out["r50_rebound_ratio"] = float(r50_win_mean / global_min_r50)

    r50_mean = float(np.mean(r50_win))
    if r50_mean > 0.0 and np.all(np.isfinite(r50_win)):
        out["r50_fractional_range"] = float(
            (np.max(r50_win) - np.min(r50_win)) / r50_mean
        )
        # A degree-1 least-squares fit needs at least two distinct times;
        # min_samples (>= 5) already guarantees that whenever
        # has_enough_samples is True.
        slope_per_s, _intercept = np.polyfit(t_win, r50_win, 1)
        out["r50_linear_slope_pc_per_myr"] = float(slope_per_s * MYR / PC)
        out["r50_relative_drift"] = float(slope_per_s * total_time * window_fraction / r50_mean)
    if np.all(np.isfinite(q_win)):
        out["virial_ratio_range"] = float(np.max(q_win) - np.min(q_win))
        out["virial_ratio_mean"] = float(np.mean(q_win))
        out["virial_ratio_mean_deviation"] = float(abs(out["virial_ratio_mean"] - 1.0))

    out["is_settled"] = bool(
        out["collapse_before_window"]
        and math.isfinite(out["r50_rebound_ratio"])
        and out["r50_rebound_ratio"] >= LATE_WINDOW_REBOUND_RATIO
        and math.isfinite(out["virial_ratio_mean_deviation"])
        and out["virial_ratio_mean_deviation"] <= LATE_WINDOW_Q_CENTER_TOLERANCE
        and math.isfinite(out["r50_relative_drift"])
        and abs(out["r50_relative_drift"]) <= LATE_WINDOW_DRIFT_THRESHOLD
        and math.isfinite(out["r50_fractional_range"])
        and out["r50_fractional_range"] <= LATE_WINDOW_R50_RANGE_THRESHOLD
        and math.isfinite(out["virial_ratio_range"])
        and out["virial_ratio_range"] <= LATE_WINDOW_Q_RANGE_THRESHOLD
    )
    return out


def run_cluster(n_bodies=200, total_mass_msun=1.0e3, scale_radius_pc=1.0,
                 n_relax=5.0, steps_per_crossing=60, softening_pc=None,
                 theta=0.5, method="tree", target_snapshots=150, seed=None,
                 lagrangian_fractions=LAGRANGIAN_FRACTIONS_DEFAULT):
    """
    Star-cluster mode: a Plummer sphere evolved under mutual gravity for a
    chosen multiple of its own nominal two-body relaxation time.

    The run length is set as a multiple, ``n_relax``, of the cluster's own
    initial two-body relaxation time (relaxation_time()), and the
    timestep as a fraction, 1/steps_per_crossing, of its initial crossing
    time (crossing_time()) -- both computed from the requested N, mass and
    scale radius, so a run is numerically well resolved without the
    student having to already know what a sound dt looks like. Because
    t_relax ~ (N / 8 ln N) t_cross, a modest N (order 100-500, the range
    this teaching tool targets) reaches that many relaxation times within
    a run length a laptop can actually complete; a realistic star cluster
    (N ~ 10^5-10^6) would not, which is itself one of the lessons here.
    NOTE: relaxation_time() is a NOMINAL, unsoftened order-of-magnitude
    estimate (see its own docstring) -- with this program's default
    softening, the true relaxation rate of the actual, softened system
    being integrated is suppressed well below that nominal estimate, so
    ``n_relax`` nominal relaxation times of run length does not promise
    that many relaxation times' worth of ACTUAL relaxation have occurred.

    Two different diagnostics track the same underlying process at two
    different sensitivities, and BOTH are frequently flat or exactly zero
    for the ENTIRE default-parameter run, not only briefly: ``n_unbound``
    (identify_unbound) counts bodies that are INSTANTANEOUSLY unbound
    (positive specific energy) right now -- not a confirmed, permanent
    escape count: a body counted here can return to negative energy at a
    later snapshot as the system's own potential evolves, so n_unbound is
    not guaranteed to be monotonically increasing over a run. It commonly
    stays at zero for the whole default run length AND the default
    softening, because that softening is chosen to minimize the force
    error against the smooth mass distribution, which necessarily also
    damps the close, hard two-body encounters that physically drive
    relaxation and evaporation in the first place.
    ``high_velocity_fraction`` (the fraction of bodies already above 90%
    of their local escape speed) is intended as a more sensitive leading
    indicator of the same process, but at default parameters it too is
    frequently zero at every snapshot of a run, for the same
    softening-suppression reason -- it is not something a default run is
    guaranteed to show growing. Seeing either diagnostic move away from
    zero within a practical run generally requires lowering
    ``softening_pc`` well below its default AND raising
    ``steps_per_crossing`` to compensate (a smaller softening length
    demands a smaller timestep) -- an explicit exercise in the Help file,
    not a change made silently here. A short default-parameter run
    therefore mainly demonstrates the ABSENCE of strong two-body
    relaxation at this program's default softening, not its presence;
    treat any Lagrangian-radius or density-profile change over such a run
    as a candidate for readjustment or sampling noise (see the caveat
    below) until a lowered-softening run shows the diagnostics above
    actually respond.

    CAVEAT ON THE EARLY PART OF A RUN: this mode's initial condition
    (plummer_sphere) is put into exact instantaneous scalar virial
    balance for the actual discrete, softened realization, but that is a
    single global energy-scale correction, not a guarantee that the
    discrete realization sits at a true dynamical (phase-space)
    equilibrium -- see plummer_sphere()'s and virial_force_term()'s
    docstrings. A modest readjustment transient on the order of a few
    crossing times can therefore appear at the start of a run for
    reasons unrelated to two-body relaxation; radius or density-profile
    evolution should not be attributed to relaxation without checking
    that it persists well beyond this initial window (n_relax is
    expressed in units of the much longer relaxation time specifically
    so that a full run comfortably outlasts it).
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    n_relax = _require_positive("n_relax", n_relax)
    steps_per_crossing = _require_int("steps_per_crossing", steps_per_crossing, lo=4)
    theta = _require_theta(theta)
    method = _require_method(method)
    target_snapshots = _require_int("target_snapshots", target_snapshots, lo=2)

    # Softening is computed here, before the initial conditions, because
    # plummer_sphere() needs it to rescale velocities to the ACTUAL
    # discrete, softened realization's scalar virial balance, not an
    # idealized unsoftened one -- athanassoula_softening() depends only on
    # n_bodies and scale_radius_pc, not on the sampled realization, so
    # this ordering changes nothing about what softening value is chosen.
    softening = (softening_pc * PC if softening_pc is not None
                 else athanassoula_softening(n_bodies, scale_radius_pc * PC))
    softening = _require_positive("softening", softening)
    softening_explicit = softening_pc is not None

    ic = plummer_sphere(n_bodies, total_mass_msun, scale_radius_pc,
                         softening=softening, seed=seed)
    pos0, vel0, masses = ic["positions"], ic["velocities"], ic["masses"]
    total_mass_kg = float(masses.sum())

    r50_0 = half_mass_radius(pos0, masses)
    t_cross0 = crossing_time(r50_0, total_mass_kg)
    t_relax0 = relaxation_time(n_bodies, r50_0, total_mass_kg)

    dt = t_cross0 / steps_per_crossing
    n_steps = _require_int(
        "computed n_steps", round(n_relax * t_relax0 / dt), lo=MIN_STEPS, hi=MAX_STEPS
    )
    stride = _pick_stride(n_steps, target_snapshots)

    warnings = []
    if method == "direct" and n_bodies > DIRECT_METHOD_WARN_BODIES:
        warnings.append(
            f"method='direct' with {n_bodies} bodies costs O(N^2) per step; "
            f"'tree' is recommended above {DIRECT_METHOD_WARN_BODIES} bodies."
        )
    if softening > 2.0 * scale_radius_pc * PC:
        warnings.append(
            "softening exceeds twice the Plummer scale radius: internal "
            "structure below that scale is smoothed away, which will "
            "suppress two-body relaxation itself, not just close encounters."
        )

    sim = integrate_nbody(pos0, vel0, masses, dt, n_steps, softening,
                           method=method, theta=theta, snapshot_stride=stride)

    lagrangian = _lagrangian_track(sim["positions"], masses, lagrangian_fractions)
    # The OFFICIAL virial ratio (reported, plotted, and written to the
    # CSV) is computed from the COM-relative ("internal") kinetic
    # energy series, not the raw
    # lab-frame one -- see integrate_nbody()'s _record() for why. A
    # uniform velocity boost applied to every body no longer changes
    # this virial ratio, matching the Help file's description of it as
    # an indicator of internal contraction/expansion/balance.
    virial = _virial_track(sim["kinetic_com"], sim["virial_work"])
    n_unbound = np.array([
        int(np.sum(identify_unbound(sim["positions"][k], sim["velocities"][k],
                                     masses, softening)))
        for k in range(sim["t"].size)
    ])
    fast_fraction = np.array([
        high_velocity_fraction(sim["positions"][k], sim["velocities"][k],
                                masses, softening)
        for k in range(sim["t"].size)
    ])
    energy_drift = _energy_drift(sim["energy"])
    if energy_drift > 0.02:
        warnings.append(
            f"total energy drifted by {energy_drift:.1%} over the run; "
            "reduce dt (raise steps_per_crossing) for a more trustworthy "
            "integration."
        )

    summary = dict(
        n_bodies=n_bodies, total_mass_msun=total_mass_msun,
        scale_radius_pc=scale_radius_pc, m_body_msun=total_mass_msun / n_bodies,
        softening_pc=softening / PC, softening_explicit=softening_explicit,
        theta=theta, method=method,
        steps_per_crossing=steps_per_crossing, target_snapshots=target_snapshots,
        dt_myr=dt / MYR, n_steps=n_steps, n_snapshots=sim["t"].size,
        t_cross0_myr=t_cross0 / MYR, t_relax0_myr=t_relax0 / MYR,
        n_relax_requested=n_relax, total_time_myr=sim["t"][-1] / MYR,
        r50_initial_pc=r50_0 / PC, r50_final_pc=lagrangian[-1, list(lagrangian_fractions).index(0.5)] / PC
        if 0.5 in lagrangian_fractions else float("nan"),
        n_unbound_initial=int(n_unbound[0]), n_unbound_final=int(n_unbound[-1]),
        unbound_fraction_final=float(n_unbound[-1] / n_bodies),
        high_velocity_fraction_initial=float(fast_fraction[0]),
        high_velocity_fraction_final=float(fast_fraction[-1]),
        virial_ratio_initial=float(virial[0]), virial_ratio_final=float(virial[-1]),
        max_fractional_energy_drift=energy_drift,
        lagrangian_fractions=tuple(lagrangian_fractions),
        seed=seed, warnings=warnings,
        model_version=MODEL_VERSION, build_id=BUILD_ID,
    )
    return dict(kind="cluster", t=sim["t"], positions=sim["positions"],
                velocities=sim["velocities"], masses=masses,
                lagrangian_radii=lagrangian, virial_ratio=virial,
                n_unbound=n_unbound, high_velocity_fraction=fast_fraction,
                energy=sim["energy"],
                kinetic=sim["kinetic"], kinetic_com=sim["kinetic_com"],
                potential=sim["potential"],
                virial_work=sim["virial_work"],
                momentum=sim["momentum"], softening=softening,
                summary=summary)


def run_galaxy(n_bodies=300, total_mass_msun=1.0e6, radius_pc=200.0,
                virial_ratio_init=0.0, n_freefall=8.0, steps_per_freefall=80,
                softening_pc=None, theta=0.5, method="tree",
                target_snapshots=150, seed=None,
                lagrangian_fractions=LAGRANGIAN_FRACTIONS_DEFAULT):
    """
    Galaxy-formation mode: a uniform sphere released from a chosen initial
    virial ratio (0 = perfectly cold) and followed through gravitational
    collapse and "violent relaxation" (Lynden-Bell 1967) into a
    quasi-equilibrium remnant, following the classic cold-collapse
    numerical experiment (e.g. van Albada 1982).

    Each simulation particle here stands in for a large aggregate of
    stars (or dark matter), not one star: to keep N within reach of a
    pure-Python teaching code, this is a scaled-down protogalactic
    fragment (order 10^6 solar masses within order 100 pc), not a
    realistic galaxy (order 10^10-10^12 solar masses across kpc to
    100 kpc). With only a few hundred particles, a monopole tree
    approximation, fixed softening, and the resulting finite-N shot
    noise, this run is a qualitative analogue of collisionless
    mean-field collapse, not a physically equivalent scaled copy of
    one: the two-body relaxation time (see cluster mode and Physical
    Background) scales strongly with N, so an N of order 10^2-10^3
    here and a real galaxy's 10^10-10^11 stars are not simply the
    same collisionless system viewed at a different display scale.
    Treat this mode as illustrating the qualitative collisionless
    cold-collapse mechanism, not as reproducing a real galaxy's
    numbers -- check any feature of interest for N/softening/timestep
    convergence before drawing a physical conclusion from it (see the
    Help file's galaxy-convergence exercise).

    The run length is a multiple, ``n_freefall``, of the sphere's own
    initial free-fall time (free_fall_time()); the timestep is a
    fraction, 1/steps_per_freefall, of that same free-fall time.
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    n_freefall = _require_positive("n_freefall", n_freefall)
    steps_per_freefall = _require_int("steps_per_freefall", steps_per_freefall, lo=4)
    theta = _require_theta(theta)
    method = _require_method(method)
    target_snapshots = _require_int("target_snapshots", target_snapshots, lo=2)

    r_sphere = radius_pc * PC
    # Computed before the initial conditions, as in run_cluster(): see the
    # comment there -- athanassoula_softening() does not depend on the sampled
    # realization, only on n_bodies and radius_pc.
    softening = (softening_pc * PC if softening_pc is not None
                 else athanassoula_softening(n_bodies, r_sphere))
    softening = _require_positive("softening", softening)
    softening_explicit = softening_pc is not None

    ic = uniform_sphere(n_bodies, total_mass_msun, radius_pc,
                         virial_ratio_init=virial_ratio_init,
                         softening=softening, seed=seed)
    pos0, vel0, masses = ic["positions"], ic["velocities"], ic["masses"]
    total_mass_kg = float(masses.sum())

    # See crossing_time() for why r_sphere**3 is
    # computed by hand rather than via **, and why the result is then
    # explicitly validated finite rather than trusted.
    r_sphere_cubed = r_sphere * r_sphere * r_sphere
    mean_density = total_mass_kg / (4.0 / 3.0 * math.pi * r_sphere_cubed)
    mean_density = _require_positive("mean_density", mean_density)
    t_ff = free_fall_time(mean_density)

    dt = t_ff / steps_per_freefall
    n_steps = _require_int(
        "computed n_steps", round(n_freefall * t_ff / dt), lo=MIN_STEPS, hi=MAX_STEPS
    )
    stride = _pick_stride(n_steps, target_snapshots)

    warnings = []
    if method == "direct" and n_bodies > DIRECT_METHOD_WARN_BODIES:
        warnings.append(
            f"method='direct' with {n_bodies} bodies costs O(N^2) per step; "
            f"'tree' is recommended above {DIRECT_METHOD_WARN_BODIES} bodies."
        )

    sim = integrate_nbody(pos0, vel0, masses, dt, n_steps, softening,
                           method=method, theta=theta, snapshot_stride=stride,
                           track_dense=True)

    lagrangian = _lagrangian_track(sim["positions"], masses, lagrangian_fractions)
    # See run_cluster()'s matching comment -- the official virial ratio
    # uses the COM-relative kinetic series.
    virial = _virial_track(sim["kinetic_com"], sim["virial_work"])
    energy_drift = _energy_drift(sim["energy"])
    if energy_drift > 0.02:
        warnings.append(
            f"total energy drifted by {energy_drift:.1%} over the run; "
            "reduce dt (raise steps_per_freefall) for a more trustworthy "
            "integration, especially through the sharp central density "
            "spike at first collapse."
        )
    r50_series = lagrangian[:, list(lagrangian_fractions).index(0.5)] \
        if 0.5 in lagrangian_fractions else None
    collapse_idx = int(np.argmin(r50_series)) if r50_series is not None else None
    r50_min_myr = float(sim["t"][collapse_idx] / MYR) \
        if collapse_idx is not None else float("nan")
    virial_at_collapse = float(virial[collapse_idx]) \
        if collapse_idx is not None else float("nan")
    # The settling verdict is computed from the DENSE, every-step r50/Q
    # series (t_dense/r50_dense/virial_ratio_dense), not from the sparse,
    # target_snapshots-dependent r50_series/virial above -- otherwise two
    # bit-for-bit identical trajectories that differ only in how many
    # snapshots were kept for plotting could reach different settling
    # verdicts. r50_series/virial above remain the
    # OFFICIAL reported initial/final/deepest-collapse numbers, which are
    # exact at every stride by construction (snapshot 0 and the true final
    # step are always kept -- see integrate_nbody()).
    t_dense = np.arange(n_steps + 1, dtype=float) * dt
    late_window = _late_time_window_stats(
        t_dense, sim["r50_dense"], sim["virial_ratio_dense"]
    )

    summary = dict(
        n_bodies=n_bodies, total_mass_msun=total_mass_msun, radius_pc=radius_pc,
        m_body_msun=total_mass_msun / n_bodies,
        virial_ratio_init=virial_ratio_init,
        softening_pc=softening / PC, softening_explicit=softening_explicit,
        theta=theta, method=method,
        steps_per_freefall=steps_per_freefall, target_snapshots=target_snapshots,
        dt_myr=dt / MYR, n_steps=n_steps, n_snapshots=sim["t"].size,
        t_freefall_myr=t_ff / MYR, n_freefall_requested=n_freefall,
        total_time_myr=sim["t"][-1] / MYR,
        r50_initial_pc=(r50_series[0] / PC) if r50_series is not None else float("nan"),
        r50_final_pc=(r50_series[-1] / PC) if r50_series is not None else float("nan"),
        r50_minimum_pc=(float(r50_series.min()) / PC) if r50_series is not None else float("nan"),
        time_of_deepest_collapse_myr=r50_min_myr,
        virial_ratio_initial=float(virial[0]), virial_ratio_final=float(virial[-1]),
        virial_ratio_at_deepest_collapse=virial_at_collapse,
        late_window_fraction=late_window["window_fraction"],
        late_window_start_myr=late_window["window_start_myr"],
        # Named "dense_samples", not "snapshots": these count points in
        # the DENSE, every-integration-step diagnostic series
        # _late_time_window_stats() actually operates on, which is
        # entirely independent of target_snapshots (the separate,
        # coarser stored-for-plotting/CSV sample count) -- see
        # _late_time_window_stats()'s own docstring.
        late_window_n_dense_samples=late_window["n_samples"],
        late_window_has_enough_dense_samples=late_window["has_enough_samples"],
        late_window_collapse_before_window=late_window["collapse_before_window"],
        late_r50_rebound_ratio=late_window["r50_rebound_ratio"],
        late_r50_fractional_range=late_window["r50_fractional_range"],
        late_r50_relative_drift=late_window["r50_relative_drift"],
        late_r50_linear_slope_pc_per_myr=late_window["r50_linear_slope_pc_per_myr"],
        late_virial_ratio_range=late_window["virial_ratio_range"],
        late_virial_ratio_mean=late_window["virial_ratio_mean"],
        late_virial_ratio_mean_deviation=late_window["virial_ratio_mean_deviation"],
        late_window_is_settled=late_window["is_settled"],
        max_fractional_energy_drift=energy_drift,
        lagrangian_fractions=tuple(lagrangian_fractions),
        seed=seed, warnings=warnings,
        model_version=MODEL_VERSION, build_id=BUILD_ID,
    )
    return dict(kind="galaxy", t=sim["t"], positions=sim["positions"],
                velocities=sim["velocities"], masses=masses,
                lagrangian_radii=lagrangian, virial_ratio=virial,
                energy=sim["energy"], kinetic=sim["kinetic"],
                kinetic_com=sim["kinetic_com"],
                potential=sim["potential"], virial_work=sim["virial_work"],
                momentum=sim["momentum"],
                softening=softening, summary=summary)


def run_chaos(n_bodies=40, total_mass_msun=1.0e3, scale_radius_pc=1.0,
              relative_perturbation=1.0e-8, n_cross=120.0,
              steps_per_crossing=50, softening_pc=None, theta=0.5,
              method="direct", target_snapshots=200, seed=None,
              perturbation_seed=None):
    """
    Chaos mode: two Plummer-sphere realizations that start indistinguishably
    close together -- identical initial conditions except for an
    isotropic random position offset of relative size
    ``relative_perturbation`` applied to every body -- are integrated with
    IDENTICAL code, parameters and timestep, and their growing separation
    is tracked (perturb_positions, position_space_divergence). See Miller
    (1964) for the original demonstration that gravitational N-body
    motion is chaotic in this sense.

    ``method`` defaults to "direct", not "tree", for this mode
    specifically (the other two modes keep "tree" as their default). The
    Barnes-Hut tree is pair-asymmetric and rebuilt independently for the
    two realizations each step, so nearly-coincident bodies that fall on
    opposite sides of a cell boundary in one realization but not the
    other pick up a small, discontinuous, algorithmic force difference
    on top of the genuine physical divergence being measured -- this
    inflates or corrupts the very quantity chaos mode exists to measure.
    A controlled N=40 comparison at fixed initial conditions found the
    tree method both LESS accurate (larger energy drift) and SLOWER than
    direct summation at this mode's default N, so there is no accuracy/
    speed tradeoff being given up by defaulting to direct here; method
    remains an explicit, overridable argument for students who want to
    see the tree method's extra, non-physical contribution to divergence
    for themselves.

    The run length is a multiple, ``n_cross``, of the shared initial
    crossing time; a large multiple is needed here (order 10-100
    crossing times, versus order 1-10 for the other two modes) because
    the whole point is to watch a sub-microscopic initial difference
    grow, over many crossing times, until it is comparable to the size
    of the system itself.

    The reported Lyapunov exponent/time (see estimate_lyapunov_exponent)
    is a heuristic finite-time fit to the measured divergence, not a
    formal chaos indicator with known statistical properties; it is
    reported as nan, with an explanatory warning, whenever no window of
    the divergence trace passes this program's fit-quality gates, which
    happens for a non-negligible fraction of runs even when the
    underlying dynamics are genuinely chaotic (see its own docstring for
    the known limitation this implies).
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    n_cross = _require_positive("n_cross", n_cross)
    steps_per_crossing = _require_int("steps_per_crossing", steps_per_crossing, lo=4)
    theta = _require_theta(theta)
    method = _require_method(method)
    target_snapshots = _require_int("target_snapshots", target_snapshots, lo=2)
    relative_perturbation = _require_positive(
        "relative_perturbation", relative_perturbation
    )

    # Softening computed before the initial conditions -- see run_cluster():
    # plummer_sphere() needs it to rescale to the actual discrete,
    # softened realization's scalar virial balance.
    softening = (softening_pc * PC if softening_pc is not None
                 else athanassoula_softening(n_bodies, scale_radius_pc * PC))
    softening = _require_positive("softening", softening)
    softening_explicit = softening_pc is not None

    ic = plummer_sphere(n_bodies, total_mass_msun, scale_radius_pc,
                         softening=softening, seed=seed)
    pos_a, vel0, masses = ic["positions"], ic["velocities"], ic["masses"]
    total_mass_kg = float(masses.sum())
    # masses= removes any net center-of-mass shift the random perturbation
    # itself would otherwise impart to realization B relative to A: without
    # it, a coherent COM translation between the two realizations is folded
    # into the measured divergence, which is supposed to isolate their
    # internal, relative chaotic divergence.
    pos_b = perturb_positions(pos_a, relative_perturbation, masses=masses,
                               seed=perturbation_seed)

    r50_0 = half_mass_radius(pos_a, masses)
    t_cross0 = crossing_time(r50_0, total_mass_kg)
    dt = t_cross0 / steps_per_crossing
    n_steps = _require_int(
        "computed n_steps", round(n_cross * t_cross0 / dt), lo=MIN_STEPS, hi=MAX_STEPS
    )
    stride = _pick_stride(n_steps, target_snapshots)

    warnings = []
    if method == "direct" and n_bodies > DIRECT_METHOD_WARN_BODIES:
        warnings.append(
            f"method='direct' with {n_bodies} bodies costs O(N^2) per step; "
            f"'tree' is recommended above {DIRECT_METHOD_WARN_BODIES} bodies."
        )

    sim_a = integrate_nbody(pos_a, vel0, masses, dt, n_steps, softening,
                             method=method, theta=theta, snapshot_stride=stride)
    sim_b = integrate_nbody(pos_b, vel0, masses, dt, n_steps, softening,
                             method=method, theta=theta, snapshot_stride=stride)

    divergence = position_space_divergence(sim_a["positions"], sim_b["positions"],
                                            masses=masses)
    lyap = estimate_lyapunov_exponent(sim_a["t"], divergence)
    energy_drift_a = _energy_drift(sim_a["energy"])
    energy_drift_b = _energy_drift(sim_b["energy"])
    if max(energy_drift_a, energy_drift_b) > 0.02:
        warnings.append(
            f"total energy drifted by up to "
            f"{max(energy_drift_a, energy_drift_b):.1%} over the run; reduce "
            "dt (raise steps_per_crossing) so the divergence measured is "
            "physical growth, not integration error."
        )
    if not math.isfinite(lyap["lyapunov_exponent"]):
        warnings.append(
            "no single contiguous stretch of the measured divergence passed "
            "this program's exponential-growth-quality heuristic (a window "
            "spanning at least one order of magnitude of growth, ln(10) "
            "approx 2.303 in log-amplitude, with a whole-window r_squared "
            ">= 0.90, and without excessive quadratic curvature in log space) "
            "-- this is a heuristic finite-time fit, not a formal "
            "chaos test, and it can and does miss some genuinely chaotic "
            "runs; the divergence may still be too small, may have already "
            "saturated at the system size, or may simply not have grown "
            "over enough dynamic range yet for this heuristic to trust it. "
            "Raise n_cross or relative_perturbation and re-run, or inspect "
            "the plotted divergence curve directly -- a real (if noisy) "
            "exponential stretch may still be visible even when this "
            "automated check does not confirm it."
        )

    summary = dict(
        n_bodies=n_bodies, total_mass_msun=total_mass_msun,
        scale_radius_pc=scale_radius_pc, relative_perturbation=relative_perturbation,
        softening_pc=softening / PC, softening_explicit=softening_explicit,
        theta=theta, method=method,
        steps_per_crossing=steps_per_crossing, target_snapshots=target_snapshots,
        dt_myr=dt / MYR, n_steps=n_steps, n_snapshots=sim_a["t"].size,
        t_cross0_myr=t_cross0 / MYR, n_cross_requested=n_cross,
        total_time_myr=sim_a["t"][-1] / MYR,
        initial_divergence_pc=float(divergence[0]) / PC,
        final_divergence_pc=float(divergence[-1]) / PC,
        max_divergence_pc=float(divergence.max()) / PC,
        lyapunov_exponent_per_myr=(lyap["lyapunov_exponent"] * MYR
                                    if math.isfinite(lyap["lyapunov_exponent"])
                                    else float("nan")),
        lyapunov_time_myr=(lyap["lyapunov_time"] / MYR
                            if math.isfinite(lyap["lyapunov_time"]) else float("nan")),
        lyapunov_time_over_t_cross=(lyap["lyapunov_time"] / t_cross0
                                     if math.isfinite(lyap["lyapunov_time"]) else float("nan")),
        lyapunov_fit_r_squared=lyap["r_squared"],
        lyapunov_fit_window_efolds=lyap["window_log_amplitude_span"],
        lyapunov_fit_curvature_t_statistic=lyap["curvature_t_statistic"],
        n_points_used_in_fit=lyap["n_points_used"],
        lyapunov_fit_start_index=lyap["fit_start_index"],
        lyapunov_fit_stop_index=lyap["fit_stop_index"],
        max_fractional_energy_drift=max(energy_drift_a, energy_drift_b),
        seed=seed, perturbation_seed=perturbation_seed, warnings=warnings,
        model_version=MODEL_VERSION, build_id=BUILD_ID,
    )
    return dict(kind="chaos", t=sim_a["t"],
                positions_a=sim_a["positions"], positions_b=sim_b["positions"],
                velocities_a=sim_a["velocities"], velocities_b=sim_b["velocities"],
                masses=masses, divergence=divergence,
                energy_a=sim_a["energy"], energy_b=sim_b["energy"],
                softening=softening, summary=summary)
