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
    return 0.98 * scale_radius * n_bodies ** (-0.26)


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

    acc = np.zeros_like(positions)
    for i in range(n):
        d = positions - positions[i]              # (n, 3), points i -> all
        r2 = np.einsum("ij,ij->i", d, d) + eps2
        # Same silent-corruption risk as eps2
        # above, from the position-separation side instead -- a
        # sufficiently large (but individually finite) separation can
        # overflow r2 to inf, making that one source's inv_r3 exactly
        # 0.0 rather than nan/inf, which the isfinite(acc) postcondition
        # below would not catch if other sources for body i stayed
        # normal (their finite contributions would still sum to
        # something finite, just silently missing this one source's
        # force). Checked per-source here instead.
        if not np.all(np.isfinite(r2)):
            raise ValueError(
                "the direct-summation force denominator overflowed; "
                "check that positions are physically reasonable."
            )
        # r2 is finite here (just checked above), but for an extremely
        # small (though still positive and finite) softening, r2**(-1.5)
        # can itself overflow to inf for a near-coincident pair -- a
        # genuine overflow, but one the isfinite(acc) postcondition below
        # already catches; over='ignore' only silences numpy's redundant
        # RuntimeWarning for that same, already-handled case.
        with np.errstate(over="ignore"):
            inv_r3 = r2 ** (-1.5)
        inv_r3[i] = 0.0                            # exclude self term
        acc[i] = G * np.sum((masses * inv_r3)[:, None] * d, axis=0)

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


def _build_octree(positions, masses, indices, center, half_size, depth):
    node = _OctreeNode()
    node.cx, node.cy, node.cz = center
    node.half_size = half_size
    sub_pos = positions[indices]
    sub_mass = masses[indices]
    total_mass = float(sub_mass.sum())
    node.mass = total_mass
    com = (sub_mass[:, None] * sub_pos).sum(axis=0) / total_mass
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
        warnings.warn(
            f"build_octree: {len(indices)} bodies could not be separated "
            f"after {MAX_TREE_DEPTH} levels of octree subdivision (they are "
            "at or extremely near the same position) and were placed "
            "together in one 'bucket' leaf node; the tree walk still "
            "evaluates each of their forces individually and exactly "
            "(the same softened pairwise force direct summation would "
            "give), not as a combined monopole, so no force accuracy is "
            "lost -- only the usual octree traversal speedup for these "
            "particular bodies.",
            RuntimeWarning,
            stacklevel=2,
        )

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
            positions, masses, sub_idx, child_center, child_half, depth + 1
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
    center = tuple((0.5 * (lo + hi)).tolist())
    half_size = float(np.max(hi - lo)) * 0.5 * 1.001
    if not (half_size > 0.0):
        half_size = 1.0
    return _build_octree(positions, masses, np.arange(n), center, half_size, 0)


def _node_acceleration(root, i, pos_list, mass_list, theta, eps2):
    """
    Iterative (explicit-stack) tree walk for one body.

    Written with an explicit stack of nodes, rather than recursion, and
    with plain Python float arithmetic on 3-tuples rather than NumPy
    3-vectors, purely for wall-clock speed: see _OctreeNode's docstring.
    The algorithm is exactly the recursive Barnes-Hut walk described in
    compute_accelerations_tree's docstring; only the implementation is
    optimized.
    """
    xi, yi, zi = pos_list[i]
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
                dx, dy, dz = xj - xi, yj - yi, zj - zi
                r2 = dx * dx + dy * dy + dz * dz + eps2
                f = mass_list[j] * r2 ** -1.5
                ax += f * dx
                ay += f * dy
                az += f * dz
            continue

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
            f = node.mass * r2 ** -1.5
            ax += f * dx
            ay += f * dy
            az += f * dz
        else:
            stack.extend(node.children)
    return G * ax, G * ay, G * az


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
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    if positions.shape != (masses.size, 3):
        raise ValueError(
            f"positions must have shape ({masses.size}, 3) to match masses; "
            f"got {positions.shape}."
        )
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
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
        acc[i] = _node_acceleration(root, i, pos_list, mass_list, theta, eps2)

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
    return float(0.5 * np.sum(masses * np.einsum("ij,ij->i", velocities, velocities)))


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
    n = masses.size
    u = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        r = np.sqrt(np.einsum("ij,ij->i", d, d) + eps2)
        u -= G * masses[i] * np.sum(masses[i + 1:] / r)
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
    Wvir -> U exactly (the r_ij^2/(r_ij^2+eps^2) factor -> 1), which is the
    check test_virial_force_term_reduces_to_potential_energy_as_softening_
    vanishes verifies.

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
    n = masses.size
    w = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        r2 = np.einsum("ij,ij->i", d, d)
        # over='ignore' silences numpy's RuntimeWarning for an r2
        # extreme enough to overflow (r2+eps2)**1.5 to inf -- an
        # expected, already-handled case: the finiteness check
        # immediately below raises a clear ValueError for exactly this.
        with np.errstate(over="ignore"):
            denom = (r2 + eps2) ** 1.5
        # For sufficiently extreme (but each
        # individually finite) position magnitudes, denom can overflow to
        # inf for one pair while the rest of the sum stays normal --
        # silently contributing exactly 0.0 (not nan) for that pair's
        # r2/denom term rather than raising. Checking only the FINAL
        # summed w (below) would miss this: a run with enough well-
        # behaved pairs could still sum to something finite despite one
        # or more pairs being silently dropped. Checked per-pair here
        # instead, at the value that actually overflows.
        if not np.all(np.isfinite(denom)):
            raise ValueError(
                "virial force term overflowed (a pair separation was too "
                "large for the softened force-law denominator to remain "
                "finite); check that positions are physically reasonable."
            )
        w -= G * masses[i] * np.sum(masses[i + 1:] * r2 / denom)
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
    return 2.0 * kinetic / abs(work_term)


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
    total_mass = np.sum(masses)
    return np.sum(masses[:, None] * positions, axis=0) / total_mass


def center_of_mass_velocity(velocities, masses):
    """Mass-weighted mean velocity; see center_of_mass()'s docstring for
    the masses contract this shares."""
    masses = _require_masses(masses)
    velocities = _require_snapshot(velocities, "velocities", n_bodies=masses.size)
    total_mass = np.sum(masses)
    return np.sum(masses[:, None] * velocities, axis=0) / total_mass


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

    r = np.sqrt(np.einsum("ij,ij->i", positions - center, positions - center))
    order = np.argsort(r)
    r_sorted = r[order]
    cum_mass = np.cumsum(masses[order])
    cum_frac = cum_mass / cum_mass[-1]

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
    n = masses.size

    pos_cm, vel_cm = recenter(positions, velocities, masses)
    phi = np.zeros(n)
    for i in range(n):
        d = pos_cm - pos_cm[i]
        r = np.sqrt(np.einsum("ij,ij->i", d, d) + eps2)
        r[i] = np.inf                       # exclude self term
        phi[i] = -G * np.sum(masses / r)
    speed2 = np.einsum("ij,ij->i", vel_cm, vel_cm)
    return phi, speed2


def specific_energies(positions, velocities, masses, softening):
    """
    Per-body specific (per unit mass) mechanical energy in the center-of-
    mass frame, e_i = (1/2) v_i^2 + Phi_i, where Phi_i is the softened
    potential at body i due to every OTHER body. A positive e_i marks a
    body that is INSTANTANEOUSLY unbound from the rest of the system
    given its current position and velocity (see identify_unbound).

    O(N^2); intended for periodic diagnostics, not every integration step.
    """
    phi, speed2 = _phi_and_speed2(positions, velocities, masses, softening)
    return 0.5 * speed2 + phi


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
    phi, speed2 = _phi_and_speed2(positions, velocities, masses, softening)
    v_esc2 = 2.0 * np.abs(phi)
    bound = (0.5 * speed2 + phi) < 0.0
    fast = speed2 >= (threshold ** 2) * v_esc2
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
    # Cubing via the ** operator can raise a raw,
    # uncaught OverflowError for a sufficiently large but otherwise valid
    # radius (Python's float.__pow__ goes through C's pow(), which signals
    # ERANGE as OverflowError, unlike plain multiplication, which silently
    # saturates to inf) -- multiplying out by hand never raises, so the
    # extreme-input case is instead caught, with a clear message, by the
    # explicit finiteness check that follows.
    r_cubed = half_mass_radius_m * half_mass_radius_m * half_mass_radius_m
    t_cross_squared = r_cubed / (G * total_mass_kg)
    t_cross_squared = _require_positive("t_cross_squared", t_cross_squared)
    return math.sqrt(t_cross_squared)


def free_fall_time(mean_density_kg_m3):
    """Free-fall time of a uniform sphere, t_ff = sqrt(3 pi / (32 G rho))."""
    mean_density_kg_m3 = _require_positive("mean_density_kg_m3", mean_density_kg_m3)
    return math.sqrt(3.0 * math.pi / (32.0 * G * mean_density_kg_m3))


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
    relaxation accelerates as N shrinks (which is why a 200-body cluster
    relaxes, visibly, within a simulation a student can actually run,
    while a 200,000-body one for all practical purposes does not), but it
    is not a prediction of how fast THIS program's own softened,
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
                     method="tree", theta=0.5, snapshot_stride=1):
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
        kinetic        (n_snap,)      J (per the fixed body masses)
        potential      (n_snap,)      J
        virial_work    (n_snap,)      J  (see virial_force_term)
        energy         (n_snap,)      J  (kinetic + potential)
        momentum       (n_snap, 3)    kg m/s
        n_steps_taken  int

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
    pot_hist = np.empty(n_snapshots)
    vir_hist = np.empty(n_snapshots)
    mom_hist = np.empty((n_snapshots, 3))

    def _record(k, t, pos, vel):
        t_hist[k] = t
        pos_hist[k] = pos
        vel_hist[k] = vel
        kin_hist[k] = kinetic_energy(vel, masses)
        pot_hist[k] = potential_energy(pos, masses, softening)
        # Wvir (virial_force_term), not U (potential_energy), is the correct
        # denominator for the virial ratio once softening is nonzero -- see
        # the module note above potential_energy() and virial_force_term()'s
        # own docstring.
        vir_hist[k] = virial_force_term(pos, masses, softening)
        mom_hist[k] = np.sum(masses[:, None] * vel, axis=0)

    pos, vel = positions.copy(), velocities.copy()
    accel = compute_accelerations(pos, masses, softening, method, theta)
    snap_set = set(snap_steps)
    k = 0
    if 0 in snap_set:
        _record(k, 0.0, pos, vel)
        k += 1

    t = 0.0
    for step in range(1, n_steps + 1):
        pos, vel, accel = leapfrog_step(
            pos, vel, masses, dt, softening, method, theta, accel=accel
        )
        t += dt
        if step in snap_set:
            _record(k, t, pos, vel)
            k += 1

    return dict(
        t=t_hist, positions=pos_hist, velocities=vel_hist,
        kinetic=kin_hist, potential=pot_hist, virial_work=vir_hist,
        energy=kin_hist + pot_hist,
        momentum=mom_hist, n_steps_taken=n_steps, masses=masses,
        softening=softening, method=method, theta=theta, dt=dt,
    )


# ======================================================================
# Chaos / sensitivity-to-initial-conditions diagnostics
# ======================================================================
#: How far the ACTUAL realized RMS displacement (after floating-point
#: addition to the position array) is allowed to drift, as a fraction of
#: the requested target RMS, before perturb_positions() raises rather than
#: silently returning a degraded perturbation. Chosen empirically, sampled
#: across 100 perturbation-offset seeds for a representative N=40, 1-pc
#: Plummer realization: relative_perturbation=1e-16 realizes within this
#: 10% tolerance on every sampled seed (ratio range 0.953-1.090); 1e-17 is
#: right at the representability boundary and is only reliably (87% of
#: seeds) rejected, not reliably accepted, at this tolerance; 3e-18 and
#: smaller are rejected on every sampled seed. This boundary is inherently
#: seed-dependent (the specific random offset draw interacts with each
#: position coordinate's own floating-point resolution), so treat "roughly
#: 1e-16 to 1e-17" as an order-of-magnitude guide to where representability
#: starts to fail, not a guaranteed per-seed cutoff.
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
    centroid = positions.mean(axis=0)
    rms_radius = math.sqrt(float(np.mean(np.sum((positions - centroid) ** 2, axis=1))))
    if rms_radius == 0.0:
        rms_radius = 1.0
    rng = np.random.default_rng(seed)
    offset = rng.normal(size=positions.shape)
    if masses is not None:
        masses = _require_masses(masses, n_bodies=positions.shape[0])
        offset = offset - center_of_mass(offset, masses)
    achieved_rms = math.sqrt(float(np.mean(np.sum(offset ** 2, axis=1))))
    if achieved_rms == 0.0:
        raise RuntimeError(
            "the random offset draw was degenerate (all-zero, including "
            "after center-of-mass removal); this should not happen for "
            "any valid input and indicates an unlucky draw rather than a "
            "supported degenerate case -- retry with a different seed."
        )
    target_rms = relative_perturbation * rms_radius
    offset = offset * (target_rms / achieved_rms)

    new_positions = positions + offset
    realized_displacement = new_positions - positions
    realized_rms = math.sqrt(float(np.mean(np.sum(realized_displacement ** 2, axis=1))))
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
        mass_sum = np.sum(masses)
        com_a = np.sum(masses[:, None] * positions_a, axis=-2, keepdims=True) / mass_sum
        com_b = np.sum(masses[:, None] * positions_b, axis=-2, keepdims=True) / mass_sum
        positions_a = positions_a - com_a
        positions_b = positions_b - com_b
    diff2 = np.sum((positions_a - positions_b) ** 2, axis=-1)
    return np.sqrt(np.mean(diff2, axis=-1))


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
       n_used that a pure significance test does. Measured on this
       project's own data: representative real chaos-mode runs span
       roughly 16-17 e-folds within their selected window, the logistic
       fixture spans only about 2.1, and this project's own (deliberately
       modest) positive-control fixture spans about 3.2 -- all with
       comfortable margin from the default threshold.
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
       how many raw points were stored), while a synthetic fixture's
       genuine systematic curvature survives bin-averaging just as well
       as, or better than, it survives a raw-point fit, since binning
       averages down within-bin noise without averaging away a
       consistent trend.

       A numerically degenerate quadratic fit (essentially zero residual
       variance) is treated as showing no significant curvature rather
       than raising a division-by-zero, since a fit that close to perfect
       is, if anything, stronger evidence of a clean exponential, not
       weaker.

       Measured behavior of this binned statistic (curvature_n_bins=20):
       this project's own real chaos-mode output (n_bodies=40, default
       parameters, seeds 0-19) scores at most about 8.4 in magnitude
       (seed 5); a family of stretched/compressed-exponential negative
       controls, d(t) = exp(A*(t/T)^p) for p in {0.6, 0.8, 1.2, 1.4} with
       1-5% multiplicative noise, scores at least about 12 in magnitude
       across 200 seeds per (p, noise) cell -- comfortable, non-
       overlapping margin on both sides of the default threshold. A
       noisy quadratic, d(t) = (1+t^2)*(1 + 5% noise), t in [0, 10], 200
       samples, is rejected in every one of 300 sampled seeds by this
       binned design.

       KNOWN REMAINING LIMITATION: a logistic curve is, by construction,
       asymptotically exponential during its early growth, so no shape
       statistic computed purely within a narrow amplitude window --
       binned or not -- can in principle always distinguish a logistic's
       early-growth window from a true exponential; only requiring more
       dynamic range than a logistic's early phase can sustain (check 4)
       can do that in general, and check 4 only rejects a logistic whose
       selected window fails to reach min_window_efolds. A logistic whose
       selected window happens to clear that floor by a moderate margin
       can still show too little quadratic curvature, within that
       specific narrow window, for this check to catch: measured on an
       ordinary (not the named official fixture) taller logistic,
       d(t) = (1 + 99/(1+exp(-(t-5))))*(1 + 5% noise), t in [0, 10], 400
       samples, across 300 sampled seeds, 208/300 (about 69%) clear the
       e-folds floor at all, and of THOSE, 191/208 (about 92%) are then
       also accepted by the curvature check -- i.e. once this logistic's
       window clears the dynamic-range gate, the curvature check catches
       it only rarely. Overall, that is 191/300 (about 64%) of all
       sampled seeds accepted end to end. This is a real, measured gap,
       not "occasional" in the sense of a low percentage -- see
       NbodyGalaxySimulator.html's Known Model Artefacts section for the
       same caveat stated for students, and
       test_estimate_lyapunov_exponent_does_not_reliably_reject_a_taller_
       logistic in tests/test_physics_nbg.py for the regression that
       keeps this honest.

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
    # dependence (see docstring for the measured before/after numbers).
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
    virial = _virial_track(sim["kinetic"], sim["virial_work"])
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
                kinetic=sim["kinetic"], potential=sim["potential"],
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
    100 kpc) -- the collapse and relaxation PHYSICS is the same
    collisionless N-body process either way, but the absolute numbers
    should not be over-interpreted.

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
                           method=method, theta=theta, snapshot_stride=stride)

    lagrangian = _lagrangian_track(sim["positions"], masses, lagrangian_fractions)
    virial = _virial_track(sim["kinetic"], sim["virial_work"])
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
        max_fractional_energy_drift=energy_drift,
        lagrangian_fractions=tuple(lagrangian_fractions),
        seed=seed, warnings=warnings,
        model_version=MODEL_VERSION, build_id=BUILD_ID,
    )
    return dict(kind="galaxy", t=sim["t"], positions=sim["positions"],
                velocities=sim["velocities"], masses=masses,
                lagrangian_radii=lagrangian, virial_ratio=virial,
                energy=sim["energy"], kinetic=sim["kinetic"],
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
