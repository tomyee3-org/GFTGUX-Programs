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
import numpy as np

MODEL_VERSION = "1.0.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without
#: duplicating this list.
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
# Audit1 fix (Codex P2-9, 2026-09-03): MAX_BODIES and MAX_SNAPSHOTS were
# each individually bounded, but nothing bounded their PRODUCT -- the
# stored positions and velocities histories are each
# (n_snapshots, n_bodies, 3) float64 arrays, so MAX_BODIES=5000 combined
# with MAX_SNAPSHOTS=4000 could allocate about 5000*4000*3*8*2 bytes
# (position + velocity) = 960 MB for a single run, which is not a
# "memory-safety guard" in any practical sense for a teaching tool. This
# caps the product directly; see integrate_nbody() and
# TestLeapfrogAndIntegration.test_integrate_nbody_rejects_excessive_body_snapshot_product.
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
    Audit1 fix (Codex P2-9/P2-10, Copilot A14, 2026-09-03): run_cluster(),
    run_galaxy() and run_chaos() previously only checked theta for
    finiteness up front (compute_accelerations_tree's own [MIN_THETA,
    MAX_THETA] range check was the sole gate against an out-of-range
    theta, deep inside the first leapfrog step -- after initial
    conditions, timescales and the full step count had already been
    computed). An out-of-range theta is now rejected at the same point
    as every other run-mode argument, before any of that setup work.
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


def _validate_state(positions, velocities, masses):
    """Validate a full N-body state; return (positions, velocities, masses)."""
    masses = _as_finite_array(masses, "masses")
    if masses.ndim != 1:
        raise ValueError("masses must be one-dimensional.")
    n = masses.size
    if n < MIN_BODIES:
        raise ValueError(f"at least {MIN_BODIES} bodies are required; got {n}.")
    if n > MAX_BODIES:
        raise ValueError(f"at most {MAX_BODIES:,} bodies are supported; got {n:,}.")
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
    positions = _as_finite_array(positions, "positions", shape=(n, 3))
    velocities = _as_finite_array(velocities, "velocities", shape=(n, 3))
    return positions, velocities, masses


# ======================================================================
# Softened Newtonian gravity
# ======================================================================
def dehnen_softening(n_bodies, scale_radius):
    """
    Approximately optimal Plummer-softening length for an N-body
    realization of a smooth density profile with characteristic radius
    ``scale_radius``.

        eps_opt = 0.98 * a * N^(-0.26)

    This is the empirical scaling of Dehnen (2001, MNRAS 324, 273),
    "Towards optimal softening in three-dimensional N-body codes", derived
    by minimizing the mean-square force error against the true (smooth)
    field of a Plummer model. It is used here as a general order-of-
    magnitude default for any of this program's roughly-Plummer-like or
    roughly-uniform mass distributions, not only the exact Plummer
    profile Dehnen fit it to; treat it as a well-motivated starting point
    to experiment around, not a precise result for every initial
    condition this program can generate.
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
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    n = masses.size
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape ({n}, 3) to match masses; got "
            f"{positions.shape}."
        )
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
    softening = _require_positive("softening", softening)
    eps2 = softening * softening

    acc = np.zeros_like(positions)
    for i in range(n):
        d = positions - positions[i]              # (n, 3), points i -> all
        r2 = np.einsum("ij,ij->i", d, d) + eps2
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
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    n = masses.size
    if positions.shape != (n, 3):
        raise ValueError(
            f"positions must have shape ({n}, 3) to match masses; got "
            f"{positions.shape}."
        )
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
        # Audit1 fix (Codex P1-5, Copilot A10, 2026-09-03): an internal
        # node containing body i's own position must never be accepted as
        # a monopole, however small its opening angle appears -- accepting
        # it would fold body i's own mass and position into the node's
        # mass/center-of-mass and apply a spurious self-force. Node bounds
        # are exact octree-construction cubes (see _build_octree), so a
        # simple axis-aligned containment test against this node's own
        # GEOMETRIC center (cx, cy, cz -- NOT its mass-weighted comx/comy/
        # comz, which need not lie inside the cube at all) and half_size is
        # exact, not approximate; a contained body forces descent into the
        # children regardless of theta. Reproduced pre-fix: an 8-body
        # adversarial configuration with the target at one corner and 7
        # bodies clustered near the opposite corner gave a 49.27% relative
        # acceleration error at theta in [0.7, 1.0], entirely eliminated by
        # this check (see TestOctreeAndTreeAcceleration.
        # test_target_containing_node_is_never_accepted_as_monopole).
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
    eps2 = softening * softening

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
# Audit1 (Codex/Copilot, 2026-09-03) correctly found the original release
# conflating them: potential_energy() (U) is the right quantity for total-
# energy bookkeeping (E = T + U, conserved by the integrator up to numerical
# error), but it is NOT the quantity the scalar virial theorem uses once
# gravity is softened. virial_force_term() (Wvir) is that quantity; see its
# docstring. Use potential_energy() for energy conservation and
# virial_force_term() for virial-ratio/equilibrium diagnostics -- never the
# other way around.
# ======================================================================
def kinetic_energy(velocities, masses):
    velocities = _as_finite_array(velocities, "velocities")
    masses = _as_finite_array(masses, "masses")
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
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
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
    softening = _require_positive("softening", softening)
    eps2 = softening * softening
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

    Audit1 finding (Codex P1-1, Copilot A2, 2026-09-03): the original
    release reported virial_ratio(kinetic, potential_energy(...)) -- i.e.
    2T/|U| -- as "the" virial ratio and as an exact equilibrium diagnostic
    for the softened equations actually being integrated.  For eps/a of
    order the program's own Dehnen-softening default (roughly 0.2-0.25 for
    the default cluster N), the two ratios differ by several percent to
    tens of percent (measured: 2T/|U| = 1.298 vs 2T/|Wvir| = 1.380 for one
    N=200 Plummer realization at the default softening).  virial_ratio()
    itself is unchanged (it is simply 2T/|W| for whatever W is supplied);
    every CALLER that wants an equilibrium diagnostic for the softened
    force must now pass virial_force_term(...), not potential_energy(...),
    and every caller that wants total energy must keep using
    potential_energy(...).  See run_cluster()/run_galaxy() and the Help
    file's Governing Equations section.
    """
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    if np.any(masses <= 0.0):
        raise ValueError("all masses must be strictly positive.")
    softening = _require_positive("softening", softening)
    eps2 = softening * softening
    n = masses.size
    w = 0.0
    for i in range(n - 1):
        d = positions[i + 1:] - positions[i]
        r2 = np.einsum("ij,ij->i", d, d)
        denom = (r2 + eps2) ** 1.5
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
    W. 1.0 marks exact virial equilibrium. For Plummer-softened gravity,
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

    Audit1 fix (Codex P2-9, 2026-09-03): a non-positive total mass
    previously fell through to a silent 0/0 division, returning
    [nan, nan, nan] together with a RuntimeWarning rather than an
    actionable error -- masses are not required to be positive by this
    function's signature alone (unlike, e.g., kinetic_energy(), which
    does enforce it), so this checks the one thing that would otherwise
    make the division ill-defined.
    """
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
    total_mass = np.sum(masses)
    if total_mass <= 0.0:
        raise ValueError(
            f"total mass must be strictly positive to define a center of "
            f"mass; got {total_mass:g}."
        )
    return np.sum(masses[:, None] * positions, axis=0) / total_mass


def center_of_mass_velocity(velocities, masses):
    """Mass-weighted mean velocity; see center_of_mass()'s docstring for
    why a non-positive total mass is rejected rather than silently
    producing NaN (Audit1 P2-9, 2026-09-03)."""
    velocities = _as_finite_array(velocities, "velocities")
    masses = _as_finite_array(masses, "masses")
    total_mass = np.sum(masses)
    if total_mass <= 0.0:
        raise ValueError(
            f"total mass must be strictly positive to define a "
            f"center-of-mass velocity; got {total_mass:g}."
        )
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
    positions = _as_finite_array(positions, "positions")
    masses = _as_finite_array(masses, "masses")
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
    positions = _as_finite_array(positions, "positions")
    velocities = _as_finite_array(velocities, "velocities")
    masses = _as_finite_array(masses, "masses")
    softening = _require_positive("softening", softening)
    eps2 = softening * softening
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

    Two-body relaxation evaporates a cluster by slowly repopulating this
    high-velocity tail of the (locally Maxwellian) speed distribution
    until individual stars cross their local escape speed; with the
    force softening this program uses to keep the tree/direct force
    evaluation numerically well-behaved (see Domain of Validity in the
    Help file), that final crossing is suppressed far more than the
    gradual growth of this tail is, so this fraction is a far more
    sensitive, continuously varying leading indicator of relaxation than
    waiting for identify_unbound() to register a positive-energy body.

    ``threshold`` must be strictly positive but is not required to be
    below 1 (Audit1 note, Copilot A19, 2026-09-03): a value at or above 1
    is not unsafe -- it always returns 0.0, because the ``bound`` mask
    used here already excludes anything at or past the local escape
    speed, so nothing can simultaneously be bound and at or above
    threshold=1 times that speed. That is a documented degenerate case
    for an unusual argument, not an error, so it is not rejected; a
    threshold intended to mean "near escape but still bound" should be
    chosen in (0, 1).
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

    Audit1 fix (Codex P1-7, Copilot A13, 2026-09-03): renamed from
    identify_escapers(). A positive specific energy is the physically
    correct criterion for a body that WILL eventually escape to infinity
    in a static potential that falls to zero there, regardless of its
    instantaneous radial velocity sign -- a body can be energetically
    unbound while momentarily moving inward (e.g. on the incoming branch
    of a hyperbolic-like encounter), so no outward-motion requirement is
    added here (a prior release's Help file incorrectly claimed one was
    already applied; that Help claim is corrected, not this function).
    But in an N-body system the potential is NOT static -- it evolves as
    the other bodies move -- so a body flagged here can later return to
    negative energy as the potential changes; this count is therefore
    NOT necessarily monotonically increasing over a run, and a body
    counted here is "instantaneously unbound," not a confirmed,
    permanent escaper. Downstream summary fields (n_unbound,
    unbound_fraction_final) and documentation are named accordingly; see
    TestEscapersAndFastFraction.test_unbound_count_is_not_guaranteed_monotonic
    and .test_inward_moving_positive_energy_body_is_still_flagged_unbound.
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
    return math.sqrt(half_mass_radius_m ** 3 / (G * total_mass_kg))


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
    result by an order-unity factor, not a scaling. Treat this as an
    order-of-magnitude estimate: it correctly captures how relaxation
    accelerates as N shrinks (which is why a 200-body cluster relaxes,
    visibly, within a simulation a student can actually run, while a
    200,000-body one for all practical purposes does not), not a precise
    prediction for any specific realization.
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
    Sample an isotropic, isolated Plummer (1911) sphere in virial
    equilibrium, using the closed-form inversion for positions and the
    rejection method for velocities of Aarseth, Henon & Wielen (1974,
    A&A 37, 183), as also given in Hut & Makino's online "Moving Stars
    Around" text and in Binney & Tremaine's "Galactic Dynamics".

    Density profile:  rho(r) = (3M / 4 pi a^3) (1 + r^2/a^2)^(-5/2)
    Enclosed mass:     M(r)/M = r^3 / (r^2 + a^2)^(3/2) = x1  =>
                       r = a (x1^(-2/3) - 1)^(-1/2)

    Audit1 fix (Codex P1-3, Copilot A11, 2026-09-03): the velocity
    rejection sampler below draws speeds from the exact Plummer
    distribution function, which is in virial equilibrium for the
    SMOOTH, unsoftened Plummer potential Phi(r) = -GM/sqrt(r^2+a^2).
    This program integrates a discrete, Plummer-SOFTENED N-body
    realization instead (pairwise force ~ 1/(r^2+eps^2)), whose actual
    scalar-virial quantity Wvir (virial_force_term()) differs from the
    idealized continuum energy the DF was matched to -- both because
    N is finite and because eps > 0. Left uncorrected, this is a
    systematic energy-SCALE mismatch (not a shape mismatch: relative
    particle-to-particle speeds are still drawn correctly), and it was
    previously being measured and reported as if it were genuine
    two-body-relaxation-driven evolution during the first few crossing
    times of a "relaxation" run, when it was actually just the discrete
    system relaxing away from a mismatched initial condition. The fix:
    after sampling, rescale ALL velocities by one overall constant (so
    the DF's shape is unchanged) to put the actual discrete, softened
    realization into exact virial equilibrium, 2T/|Wvir| = 1, under the
    force law it will actually be integrated with -- see
    TestInitialConditions.test_plummer_sphere_starts_in_exact_softened_virial_equilibrium.

    Velocities are drawn from the exact isotropic Plummer distribution
    function f(E) ~ (-E)^(7/2) by rejection sampling q = v/v_esc(r)
    against g(q) = q^2 (1 - q^2)^(7/2). Audit1 correction (Codex P3-4,
    Copilot A16, 2026-09-03): the maximum of g is at q = 1/sqrt(4.5)
    approx 0.4714 (from dg/dq = 0 => 1 - 4.5 q^2 = 0), not at q = 2/3 as
    previously stated here; the true maximum value is g(0.4714) approx
    0.09221, not 0.0930033. The envelope constant 0.1 used below was
    already comfortably above the true maximum (0.1 > 0.09221), so the
    sampler itself was always correct -- only this derivation was wrong.
    The overall per-attempt acceptance probability is
    integral_0^1 g(q) dq / 0.1 approx 0.4295 / 0.1 approx 43%, not the
    ">~90%" previously claimed (see
    TestInitialConditions.test_plummer_velocity_envelope_is_a_valid_upper_bound).

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
    softening = softening if softening is not None else dehnen_softening(n_bodies, a)
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
    # Recenter BEFORE the equilibrium rescale below: scaling a zero-mean
    # velocity set by one overall constant preserves zero mean exactly, so
    # doing this first (rather than after, as a prior release did) makes
    # the rescale exact rather than only approximately so (Audit1 Copilot
    # P2-1; see uniform_sphere() for the N=3 worst case this order fixes).
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
    self-energy W0 = -3 G M^2 / (5 R) a prior release used instead.

    Audit1 fix (Codex P1-2/P1-3, Copilot A1/A11, 2026-09-03), two
    changes bundled together because they touch the same few lines:

    1. Convention: Q0 is now defined with the SAME 2T/|W| normalization
       as the virial_ratio() output diagnostic (equilibrium = 1), not
       the previous T/|W0| convention (equilibrium = 0.5) -- the two
       conventions previously in use for the same word "virial ratio"
       made the virial_ratio_init argument and the virial_ratio_initial
       summary field disagree by close to a factor of 2 for the same
       physical state, which is corrected here rather than merely
       documented.
    2. Energy scale: using the idealized continuum W0 instead of the
       actual discrete, softened Wvir0 as the equilibrium reference was
       itself a second, independent source of the same kind of scale
       mismatch documented in plummer_sphere()'s docstring (Codex P1-3):
       for any nonzero softening or finite N, W0 != Wvir0, so scaling to
       Q0 = 1 against W0 does not actually put the realization in
       equilibrium under the force law it is integrated with.

    Q0 = 0 (the default) is a perfectly cold start: every body begins at
    rest, and the sphere free-falls before violently relaxing into a
    quasi-equilibrium remnant -- the classic toy model of collisionless
    "violent relaxation" (Lynden-Bell 1967; the numerical experiment goes
    back to van Albada 1982, MNRAS 201, 939). Q0 = 1 is virial
    equilibrium for the actual discrete, softened realization; values in
    between give a "warm" collapse with a milder relaxation transient.

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
    softening = softening if softening is not None else dehnen_softening(n_bodies, r_sphere)
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
        # afterward (the prior order) lets the recentering step perturb T
        # away from target -- negligible at large N, but severe at small N
        # (Audit1 Copilot P2-1: measured final-T/target-T ratio at N=3
        # ranging 0.148-0.983 across 50 seeds under the old order, versus
        # 0.979-1.000 at N=300).
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
    method="direct". Audit1 qualification (Codex P2-8, 2026-09-03): the
    Barnes-Hut tree force (method="tree") is target-dependent and not
    pair-symmetric, and its approximation changes discontinuously as
    bodies cross cell boundaries between steps, so it has not been
    demonstrated to be the gradient of any single approximate
    Hamiltonian; the broad "symplectic, bounded nonsecular energy error"
    guarantee is therefore only established here for method="direct",
    not asserted for method="tree" merely because the same KDK update
    formula is used with either force.

    ``accel``, if given, is the already-computed acceleration at the
    input state (a(x0)), avoiding a redundant force evaluation when the
    caller is stepping in a loop and already has it from the previous
    step's final kick; if omitted it is computed here. Audit1 fix (Codex
    P2-9, 2026-09-03): a caller-supplied accel is now validated (shape
    and finiteness) rather than trusted -- a prior release passed it
    straight into the kick with no check, so a bad value from a
    misbehaving caller would silently corrupt the whole subsequent
    integration instead of failing at the point of the bad input.

    Returns (positions_new, velocities_new, accel_new).

    Audit1 fix (Copilot A11, 2026-09-03): dt must be strictly positive, not
    merely nonzero. Every documented mode of this program represents
    forward-in-time evolution, and a negative dt previously ran silently
    "backward" with no warning or documentation -- a forward-simulation
    API accepting an undocumented reversed-time mode by accident is a
    trap for a direct caller, not a supported feature. Reversed-time
    integration is not offered here; a caller who genuinely wants it can
    negate the returned velocities and re-run forward, which is
    mathematically equivalent for this reversible integrator.
    """
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
    does not scale with n_steps for a large stride. Audit1 correction
    (Copilot A12, 2026-09-03): this docstring previously also claimed
    angular momentum was evaluated here; no angular momentum array was
    ever computed or returned, and that claim is removed rather than
    implemented -- angular momentum for these initial conditions is
    formally zero and not a diagnostic this program currently tracks (see
    Experiment 16 in the Help file, which invites a student to add it).

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

    Audit1 fix (Copilot A11, 2026-09-03): dt must be strictly positive;
    see leapfrog_step's docstring for why a negative (reversed-time) dt
    is rejected rather than silently supported.
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
        # own docstring (Audit1 Codex P1-1, Copilot A2, 2026-09-03).
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
def perturb_positions(positions, relative_perturbation, masses=None, seed=None):
    """
    Return a copy of ``positions`` with every body's position displaced by
    an independent, isotropic random offset, such that the RMS per-body
    displacement magnitude -- sqrt(mean_i |offset_i|^2) -- equals
    ``relative_perturbation`` times the RMS radius of the configuration
    (about its own centroid).

    Audit1 fix (Codex P2-2, Copilot A9, 2026-09-03): each Cartesian
    component is drawn with sigma = relative_perturbation * rms_radius /
    sqrt(3), not sigma = relative_perturbation * rms_radius as a prior
    release used. Three independent Gaussian components each with
    variance sigma^2 sum to a squared vector magnitude with mean 3
    sigma^2, so the RMS vector displacement was previously
    sqrt(3) * relative_perturbation * rms_radius -- about 73% larger than
    documented (an N=1000 probe measured a ratio of 1.7356 against the
    predicted sqrt(3) = 1.7321). Dividing the per-component sigma by
    sqrt(3) here makes the RMS VECTOR displacement equal to
    relative_perturbation * rms_radius exactly, matching the documented
    and intended meaning of the parameter; see
    TestChaosDiagnostics.test_perturbation_rms_vector_magnitude_matches_documented_value.

    If ``masses`` is given, the random offsets are also recentered to
    remove their own mass-weighted mean before being added, so the
    perturbation does not impart a net center-of-mass shift to the
    perturbed copy relative to the original -- otherwise the subsequent
    divergence measurement would partly be tracking a spurious coherent
    translation rather than the system's internal chaotic divergence
    (see position_space_divergence()).

    This follows the classic gravitational N-body sensitivity experiment
    (Miller 1964, ApJ 140, 250): two realizations that start indistinguishably
    close together, evolved under exactly the same equations of motion and
    force method, separate exponentially -- the N-body problem is chaotic
    even though the underlying dynamics are entirely deterministic.
    """
    positions = _as_finite_array(positions, "positions")
    relative_perturbation = _require_positive(
        "relative_perturbation", relative_perturbation
    )
    centroid = positions.mean(axis=0)
    rms_radius = math.sqrt(float(np.mean(np.sum((positions - centroid) ** 2, axis=1))))
    if rms_radius == 0.0:
        rms_radius = 1.0
    rng = np.random.default_rng(seed)
    sigma_per_component = (relative_perturbation * rms_radius) / math.sqrt(3.0)
    offset = rng.normal(size=positions.shape) * sigma_per_component
    if masses is not None:
        masses = _as_finite_array(masses, "masses")
        offset = offset - center_of_mass(offset, masses)
    return positions + offset


def position_space_divergence(positions_a, positions_b, masses=None):
    """
    RMS per-body position separation (meters) between two realizations of
    the same system, sqrt( mean_i |x_a,i - x_b,i|^2 ).

    Audit1 fix (Codex P1-6, Copilot A12, 2026-09-03): renamed from
    phase_space_divergence() -- this measures separation in POSITION
    space only (velocities are not involved at all), so the previous
    name overstated what is actually computed; a genuine phase-space
    distance would need a velocity term too (with some, inherently
    arbitrary, choice of relative weighting between position and
    velocity units). If ``masses`` is given, each realization is
    recentered to its own mass-weighted center of mass before differencing,
    removing any coherent center-of-mass translation between the two
    realizations (e.g. from perturb_positions() displacing the system's
    own centroid, or from the tree method's imperfect momentum
    conservation letting the two realizations' centroids drift apart)
    from what should be a measurement of internal, relative chaotic
    divergence.
    """
    positions_a = _as_finite_array(positions_a, "positions_a")
    positions_b = _as_finite_array(positions_b, "positions_b")
    if positions_a.shape != positions_b.shape:
        raise ValueError("positions_a and positions_b must have the same shape.")
    if masses is not None:
        # Recenter per snapshot without assuming positions_a/positions_b
        # are a single (n_bodies, 3) snapshot rather than a whole
        # (n_snapshots, n_bodies, 3) history -- center_of_mass() only
        # handles the former, so the mass-weighted mean over the
        # body axis (second-to-last) is computed directly here, with
        # keepdims so it broadcasts against either shape.
        masses = _as_finite_array(masses, "masses")
        mass_sum = np.sum(masses)
        com_a = np.sum(masses[:, None] * positions_a, axis=-2, keepdims=True) / mass_sum
        com_b = np.sum(masses[:, None] * positions_b, axis=-2, keepdims=True) / mass_sum
        positions_a = positions_a - com_a
        positions_b = positions_b - com_b
    diff2 = np.sum((positions_a - positions_b) ** 2, axis=-1)
    return np.sqrt(np.mean(diff2, axis=-1))


def estimate_lyapunov_exponent(t, divergence, min_points=5, min_r_squared=0.98,
                                max_segment_slope_spread=0.15):
    """
    Estimate an exponential growth rate lambda from a divergence time
    series by an ordinary least-squares fit of ln(divergence) against t,
    restricted to a single CONTIGUOUS window where growth is plausibly
    exponential, and only accepted if that fit is actually good.

    Audit1 fix (Codex P1-6.4, Copilot A12, 2026-09-03): a prior release
    fit every point satisfying two amplitude thresholds (divergence in
    [3*d0, 0.5*d_max]) with no check that those points form a single
    contiguous interval, that the growth in it is close to exponential,
    or any reported goodness of fit -- so it could, and did, fit a
    physically meaningless "line" through scattered points from an
    oscillating series, or through smoothly curving (non-exponential)
    data whose net change happened to fall in the amplitude window.
    Measured false positives from that version: divergence = 1 + t gave
    lambda = 0.0476 (should be rejected, no exponential regime exists);
    divergence = 1 + t^2 gave lambda = 0.0750 (same). This version:

    1. Requires the amplitude-window points to form the single LONGEST
       contiguous run in time index (not a scattered subset) -- this
       alone rejects an oscillating series, which cannot sustain a long
       contiguous run through a monotonically-defined amplitude window.
    2. Requires at least ``min_points`` points in that run (default 5,
       raised from the prior version's 3, which is too few to assess fit
       quality at all).
    3. Requires the OLS fit of ln(divergence) against t over that window
       to reach R^2 >= ``min_r_squared`` (default 0.98) -- this alone
       rejects divergence = 1+t (R^2 approx 0.86 over the corresponding
       window) and divergence = 1+t^2 (R^2 approx 0.82), both far below
       threshold, because neither is close to exponential over a window
       spanning almost two orders of magnitude in amplitude.
    4. Splits the window into thirds and requires the three independently
       fit sub-slopes to agree with each other to within a fractional
       spread of ``max_segment_slope_spread`` (default 0.15) -- this
       catches smooth curvature subtle enough to still pass a whole-
       window R^2 test (verified against a logistic/saturating curve
       whose early-to-mid rise otherwise reaches R^2 approx 0.999 over
       the same amplitude window: its three sub-slopes disagree by a
       fractional spread of approximately 0.18, above threshold, and it
       is correctly rejected, whereas a genuine noisy exponential's
       sub-slopes agree to within a spread under 0.01 at up to 5%
       per-point multiplicative noise in the validation used to pick
       these thresholds).

    This is a heuristic gate, not a hypothesis test with a p-value; a
    sufficiently pathological adversarial input could still slip through
    (KNOWN LIMITATION -- documented rather than silently claimed solved,
    per this project's testing requirements). It substantially raises the
    bar over the prior version, which had no gate at all beyond the two
    amplitude thresholds, and it is validated in this codebase against
    linear, quadratic, oscillatory and saturating synthetic inputs (see
    TestChaosDiagnostics), all of which it rejects, while accepting
    genuine (possibly noisy) exponential growth.

    Returns a dict with 'lyapunov_exponent' (1/s), 'lyapunov_time' (s,
    = 1/lambda), 'n_points_used', 'r_squared' (whole-window fit quality),
    and 'segment_slope_spread' (the three-segment consistency check
    value, or nan if too few points to compute it). All numeric fields
    are nan (and lyapunov_time is left nan, not inf) when no window
    passes every gate.
    """
    t = _as_finite_array(t, "t")
    divergence = _as_finite_array(divergence, "divergence")
    if t.shape != divergence.shape:
        raise ValueError("t and divergence must have the same shape.")
    if np.any(divergence < 0.0):
        raise ValueError("divergence must be non-negative.")
    if t.size < 2 or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing with at least two points.")

    def _rejected(n_used, r_squared=float("nan"), spread=float("nan")):
        return dict(lyapunov_exponent=float("nan"), lyapunov_time=float("nan"),
                    n_points_used=n_used, r_squared=r_squared,
                    segment_slope_spread=spread)

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
    slope, intercept = np.polyfit(tw, logd, 1)
    fit = slope * tw + intercept
    ss_res = float(np.sum((logd - fit) ** 2))
    ss_tot = float(np.sum((logd - np.mean(logd)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if r_squared < min_r_squared:
        return _rejected(n_used, r_squared=r_squared)

    spread = float("nan")
    if n_used >= 6:
        edges = [0, n_used // 3, 2 * n_used // 3, n_used]
        seg_slopes = []
        for a, b in zip(edges[:-1], edges[1:]):
            if b - a < 2:
                return _rejected(n_used, r_squared=r_squared)
            s, _ = np.polyfit(tw[a:b], logd[a:b], 1)
            seg_slopes.append(s)
        if any(s <= 0.0 for s in seg_slopes):
            return _rejected(n_used, r_squared=r_squared)
        spread = (max(seg_slopes) - min(seg_slopes)) / max(abs(s) for s in seg_slopes)
        if spread > max_segment_slope_spread:
            return _rejected(n_used, r_squared=r_squared, spread=spread)

    if slope <= 0.0:
        return _rejected(n_used, r_squared=r_squared, spread=spread)

    lam = float(slope)
    return dict(lyapunov_exponent=lam, lyapunov_time=1.0 / lam,
                n_points_used=n_used, r_squared=r_squared,
                segment_slope_spread=spread)


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
    Audit1 fix (Copilot A16, 2026-09-03): floor division here previously
    let the actual snapshot count run to nearly DOUBLE target_snapshots
    whenever n_steps fell just under 2 * target_snapshots (e.g.
    n_steps=199, target_snapshots=100 gave stride = 199 // 100 = 1, i.e.
    close to 200 snapshots instead of ~100). Ceiling division keeps the
    realized count close to the requested target across that whole
    range, while still always storing at least the initial and final
    steps.
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
    potential-energy series -- see virial_ratio()'s docstring (Audit1
    Codex P1-1, Copilot A2, 2026-09-03).
    """
    return np.array([virial_ratio(k, w) for k, w in zip(kinetic, virial_work)])


def _energy_drift(energy):
    e0 = energy[0]
    if e0 == 0.0:
        return float("nan")
    return float(np.max(np.abs((energy - e0) / e0)))


def run_cluster(n_bodies=200, total_mass_msun=1.0e3, scale_radius_pc=1.0,
                 n_relax=5.0, steps_per_crossing=60, softening_pc=None,
                 theta=0.5, method="tree", target_snapshots=150, seed=None,
                 lagrangian_fractions=LAGRANGIAN_FRACTIONS_DEFAULT):
    """
    Star-cluster mode: a Plummer sphere evolved under mutual gravity long
    enough to show two-body-relaxation-driven evaporation.

    The run length is set as a multiple, ``n_relax``, of the cluster's own
    initial two-body relaxation time (relaxation_time()), and the
    timestep as a fraction, 1/steps_per_crossing, of its initial crossing
    time (crossing_time()) -- both computed from the requested N, mass and
    scale radius, so a run is numerically well resolved without the
    student having to already know what a sound dt looks like. Because
    t_relax ~ (N / 8 ln N) t_cross, a modest N (order 100-500, the range
    this teaching tool targets) relaxes within a run length a laptop can
    actually complete; a realistic star cluster (N ~ 10^5-10^6) would
    not, which is itself one of the lessons here.

    Two different diagnostics track the same underlying process at two
    different sensitivities. ``n_unbound`` (identify_unbound) counts
    bodies that are INSTANTANEOUSLY unbound (positive specific energy)
    right now -- not a confirmed, permanent escape count: a body counted
    here can return to negative energy at a later snapshot as the
    system's own potential evolves, so n_unbound is not guaranteed to be
    monotonically increasing over a run (Audit1 Codex/Copilot P1-7,
    2026-09-03; a prior release's naming, n_escaped/evaporated_fraction,
    and a prior test's final->=initial assertion both overstated this as
    confirmed evaporation). It can also stay at zero for the default run
    length AND the default (Dehnen-optimal) softening, because that
    softening is chosen to minimize the force error against the smooth
    mass distribution, which necessarily also damps the close, hard
    two-body encounters that physically drive relaxation and evaporation
    in the first place. ``high_velocity_fraction`` (the fraction of
    bodies already above 90% of their local escape speed) grows
    continuously and is visible with the default settings; it is the
    leading indicator of the same process. Seeing n_unbound > 0 within a
    practical run generally requires lowering ``softening_pc`` below its
    default AND raising ``steps_per_crossing`` to compensate (a smaller
    softening length demands a smaller timestep) -- an explicit exercise
    in the Help file, not a change made silently here.
    """
    n_bodies = _require_int("n_bodies", n_bodies, lo=MIN_BODIES, hi=MAX_BODIES)
    n_relax = _require_positive("n_relax", n_relax)
    steps_per_crossing = _require_int("steps_per_crossing", steps_per_crossing, lo=4)
    theta = _require_theta(theta)
    method = _require_method(method)
    target_snapshots = _require_int("target_snapshots", target_snapshots, lo=2)

    # Softening is computed here, before the initial conditions, because
    # plummer_sphere() now needs it (Audit1 P1-3: it rescales velocities
    # to the ACTUAL discrete, softened equilibrium, not an idealized
    # unsoftened one) -- dehnen_softening() depends only on n_bodies and
    # scale_radius_pc, not on the sampled realization, so this reordering
    # changes nothing about what softening value is chosen.
    softening = (softening_pc * PC if softening_pc is not None
                 else dehnen_softening(n_bodies, scale_radius_pc * PC))
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
    # comment there (Audit1 P1-3) -- dehnen_softening() does not depend on
    # the sampled realization, only on n_bodies and radius_pc.
    softening = (softening_pc * PC if softening_pc is not None
                 else dehnen_softening(n_bodies, r_sphere))
    softening = _require_positive("softening", softening)
    softening_explicit = softening_pc is not None

    ic = uniform_sphere(n_bodies, total_mass_msun, radius_pc,
                         virial_ratio_init=virial_ratio_init,
                         softening=softening, seed=seed)
    pos0, vel0, masses = ic["positions"], ic["velocities"], ic["masses"]
    total_mass_kg = float(masses.sum())

    mean_density = total_mass_kg / (4.0 / 3.0 * math.pi * r_sphere ** 3)
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

    Audit1 fix (Codex P1-6.3, Copilot A12, 2026-09-03): ``method``
    defaults to "direct", not "tree", for this mode specifically (the
    other two modes keep "tree" as their default). The Barnes-Hut tree
    is pair-asymmetric and rebuilt independently for the two
    realizations each step, so nearly-coincident bodies that fall on
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

    # Softening computed before the initial conditions -- see run_cluster()
    # (Audit1 P1-3): plummer_sphere() needs it to rescale to the actual
    # discrete, softened equilibrium.
    softening = (softening_pc * PC if softening_pc is not None
                 else dehnen_softening(n_bodies, scale_radius_pc * PC))
    softening = _require_positive("softening", softening)
    softening_explicit = softening_pc is not None

    ic = plummer_sphere(n_bodies, total_mass_msun, scale_radius_pc,
                         softening=softening, seed=seed)
    pos_a, vel0, masses = ic["positions"], ic["velocities"], ic["masses"]
    total_mass_kg = float(masses.sum())
    # masses= removes any net center-of-mass shift the random perturbation
    # itself would otherwise impart to realization B relative to A (Audit1
    # P1-6.2, Copilot A12): without it, a coherent COM translation between
    # the two realizations is folded into the measured divergence, which
    # is supposed to isolate their internal, relative chaotic divergence.
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
            "the separation between the two realizations never developed a "
            "single contiguous window that passed this program's "
            "exponential-growth-quality checks (r_squared >= 0.98 and "
            "consistent sub-segment slopes) -- it may still be too small, "
            "may have already saturated at the system size, or may be "
            "noisier than these checks tolerate; raise n_cross or "
            "relative_perturbation and re-run, or inspect the plotted "
            "divergence curve directly."
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
        n_points_used_in_fit=lyap["n_points_used"],
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
