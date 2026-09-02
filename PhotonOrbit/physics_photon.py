"""
physics_photon.py — PhotonOrbit

General-relativistic equations of motion and an RK4 integrator for null
geodesics in the equatorial plane of Schwarzschild spacetime.
"""

import math

MODEL_VERSION = "1.2.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_photon.py",
    "driver_photon.py",
    "main.py",
    "plot_photon.py",
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


_MAX_STEPS = 5_000_000


def _require_finite(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def radial_acceleration(r, L, GM_over_c2):
    """Return d²r/dλ² for an equatorial Schwarzschild null geodesic.

    Validates that r, L and GM_over_c2 are finite reals (Audit1 Copilot
    F-2: this public function previously checked only r>0, so an extreme
    but "finite" r such as 1e200 could raise a raw OverflowError out of
    r**3, and an extreme L such as 1e308 could silently return nan) and
    guards the arithmetic itself, so every rejection -- ordinary or
    extreme -- is the same documented ValueError rather than an
    uncaught OverflowError or a silent nan.
    """
    _require_finite("r", r)
    _require_finite("L", L)
    _require_finite("GM_over_c2", GM_over_c2)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    try:
        value = (L * L / r**3) * (1.0 - 3.0 * GM_over_c2 / r)
    except (OverflowError, ZeroDivisionError) as exc:
        # ZeroDivisionError added post-Audit1 (found while reproducing
        # Copilot F-2/Codex P2-2 with a fresh set of extreme-but-finite
        # inputs, not itself raised by either audit): an r small enough
        # that r**3 underflows to a *literal* 0.0 float (e.g. r=1e-200,
        # since 1e-200**3=1e-600 is far below the smallest positive
        # float, ~5e-324) makes "L*L / r**3" a division by zero rather
        # than an overflow -- a distinct Python exception type that the
        # original OverflowError-only except clause let straight through
        # as an uncaught traceback. Folding it into the same finite-
        # value ValueError keeps a single, documented failure mode for
        # every extreme-input case, regardless of whether the extremity
        # manifests as overflow (too large) or underflow (too small).
        raise ValueError(
            "radial_acceleration overflowed for the given r, L, GM_over_c2; "
            "these values are too extreme for this program's geometric-unit scale."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            "radial_acceleration produced a non-finite result for the given "
            "r, L, GM_over_c2; these values are too extreme for this "
            "program's geometric-unit scale."
        )
    return value


def dphi_dlambda(r, L):
    """Return dφ/dλ = L/r².

    Validates r and L (Audit1 Copilot F-2, same rationale as
    radial_acceleration above) and guards the arithmetic itself.
    """
    _require_finite("r", r)
    _require_finite("L", L)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    try:
        value = L / (r * r)
    except (OverflowError, ZeroDivisionError) as exc:
        # ZeroDivisionError added post-Audit1; see the matching comment in
        # radial_acceleration() above -- r*r can underflow to a literal
        # 0.0 for a sufficiently small (but finite and positive) r.
        raise ValueError(
            "dphi_dlambda overflowed for the given r, L; these values are "
            "too extreme for this program's geometric-unit scale."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            "dphi_dlambda produced a non-finite result for the given r, L; "
            "these values are too extreme for this program's geometric-unit "
            "scale."
        )
    return value


def _derivatives(r, v_r, phi, L, GM_over_c2):
    """Derivatives of (r, v_r, φ) with respect to affine parameter λ."""
    del phi  # φ does not appear explicitly in the Schwarzschild equations.
    return (
        v_r,
        radial_acceleration(r, L, GM_over_c2),
        dphi_dlambda(r, L),
    )


def rk4_step(r, v_r, phi, L, GM_over_c2, d_lambda):
    """Advance (r, dr/dλ, φ) by one classical fourth-order RK step."""
    k1 = _derivatives(r, v_r, phi, L, GM_over_c2)

    r2 = r + 0.5 * d_lambda * k1[0]
    v2 = v_r + 0.5 * d_lambda * k1[1]
    p2 = phi + 0.5 * d_lambda * k1[2]
    k2 = _derivatives(r2, v2, p2, L, GM_over_c2)

    r3 = r + 0.5 * d_lambda * k2[0]
    v3 = v_r + 0.5 * d_lambda * k2[1]
    p3 = phi + 0.5 * d_lambda * k2[2]
    k3 = _derivatives(r3, v3, p3, L, GM_over_c2)

    r4 = r + d_lambda * k3[0]
    v4 = v_r + d_lambda * k3[1]
    p4 = phi + d_lambda * k3[2]
    k4 = _derivatives(r4, v4, p4, L, GM_over_c2)

    factor = d_lambda / 6.0
    return (
        r + factor * (k1[0] + 2.0*k2[0] + 2.0*k3[0] + k4[0]),
        v_r + factor * (k1[1] + 2.0*k2[1] + 2.0*k3[1] + k4[1]),
        phi + factor * (k1[2] + 2.0*k2[2] + 2.0*k3[2] + k4[2]),
    )


def integrate_photon_orbit(GM_over_c2, r0, b, lambda_max, d_lambda):
    """
    Integrate an initially ingoing photon orbit.

    The affine parameter is normalized so E=1, hence L=b.  The initial
    radial velocity is chosen from

        (dr/dλ)^2 = 1 - (b^2/r^2)(1 - 2M/r),

    with the negative root for an incoming photon.  Integrating the
    second-order radial equation allows dr/dλ to pass smoothly through zero
    at a turning point, so scattering trajectories correctly turn outward.

    Returns
    -------
    x_values, y_values, info
        Cartesian trajectory samples and a dictionary of diagnostics.
    """
    for name, value in (
        ("GM_over_c2", GM_over_c2), ("r0", r0), ("b", b),
        ("lambda_max", lambda_max), ("d_lambda", d_lambda),
    ):
        _require_finite(name, value)

    if GM_over_c2 <= 0.0:
        raise ValueError("GM_over_c2 must be greater than zero.")
    r_s = 2.0 * GM_over_c2
    r_photon = 3.0 * GM_over_c2
    if r0 <= r_s:
        raise ValueError(f"r0 must be outside the event horizon (r0 > {r_s:g}).")
    if b < 0.0:
        raise ValueError("b must be nonnegative; use its magnitude as the impact parameter.")
    if lambda_max <= 0.0:
        raise ValueError("lambda_max must be greater than zero.")
    if d_lambda <= 0.0:
        raise ValueError("d_lambda must be greater than zero.")

    # Audit1 Codex P1-2/Copilot F-1: compare the ratio to the step cap
    # BEFORE calling math.ceil() on it. lambda_max/d_lambda can itself
    # legitimately evaluate to float('inf') for extreme-but-finite inputs
    # (e.g. lambda_max=1e308, d_lambda=1e-308); Python's float division
    # returns inf silently for that, but math.ceil(inf) raises an
    # uncaught OverflowError. Testing the ratio against _MAX_STEPS first
    # (a plain float comparison, well-defined even against inf) rejects
    # that case with the normal documented ValueError instead.
    step_ratio = lambda_max / d_lambda
    if not math.isfinite(step_ratio) or step_ratio > _MAX_STEPS:
        raise ValueError(
            f"lambda_max/d_lambda would require more than {_MAX_STEPS:,} "
            "steps (or is not finite); limit the run to "
            f"{_MAX_STEPS:,} steps or fewer."
        )
    n_steps = math.ceil(step_ratio)

    E = 1.0
    L = b
    initial_radicand = E*E - (L*L / (r0*r0)) * (1.0 - r_s / r0)
    tolerance = 1e-14 * max(1.0, E*E)
    if initial_radicand < -tolerance:
        b_max = r0 / math.sqrt(1.0 - r_s / r0)
        raise ValueError(
            "The requested b is incompatible with an initially ingoing null geodesic "
            f"at r0={r0:g}. For these parameters, b must be <= {b_max:.8g}."
        )
    initial_radicand = max(0.0, initial_radicand)

    r = r0
    v_r = -math.sqrt(initial_radicand)
    phi = 0.0
    lambda_value = 0.0
    escape_radius = 2.0 * r0

    x_values = [r * math.cos(phi)]
    y_values = [r * math.sin(phi)]
    min_r = r0
    status = "lambda_max"
    turned_outward = False

    for _ in range(n_steps):
        if lambda_value >= lambda_max:
            break

        remaining = lambda_max - lambda_value
        h = min(d_lambda, remaining)
        previous_r = r
        previous_v = v_r
        previous_phi = phi
        previous_lambda = lambda_value

        try:
            new_r, new_v, new_phi = rk4_step(
                r, v_r, phi, L, GM_over_c2, h
            )
        except ValueError as exc:
            raise RuntimeError(
                "Integration stepped to a nonphysical radius. Reduce d_lambda."
            ) from exc

        if not all(math.isfinite(value) for value in (new_r, new_v, new_phi)):
            raise RuntimeError("Integration produced a non-finite value; reduce d_lambda.")

        # Stop at the horizon rather than storing a point inside it. Linear
        # interpolation within the final RK4 step locates the termination
        # surface without leaving a misleading sub-horizon trajectory sample.
        if new_r <= r_s:
            if new_r < previous_r:
                fraction = (previous_r - r_s) / (previous_r - new_r)
                fraction = min(1.0, max(0.0, fraction))
            else:
                fraction = 1.0

            r = r_s
            v_r = previous_v + fraction * (new_v - previous_v)
            phi = previous_phi + fraction * (new_phi - previous_phi)
            lambda_value = previous_lambda + fraction * h

            x_values.append(r * math.cos(phi))
            y_values.append(r * math.sin(phi))
            min_r = min(min_r, r)
            status = "captured"
            break

        # Has the photon turned outward by the END of this step? Audit1
        # Codex P1-1: the previous test, "previous_v < 0.0 <= v_r", required
        # a STRICTLY negative sample beforehand. A legal tangential start
        # (b at the local kinematic bound, v_r=0 initially) never satisfies
        # that: Python's -math.sqrt(0.0) is negative zero, and -0.0 < 0.0 is
        # False, so a photon that starts at rest and is immediately pushed
        # outward by a positive radial acceleration (any r0 > 3*GM_over_c2)
        # never had the flag set and could travel arbitrarily far while
        # still being reported as status="lambda_max". Testing new_v > 0.0
        # directly (with no requirement on the sign of the sample before
        # it) catches that first outward step, while still leaving the
        # exact circular orbit (r0=3*GM_over_c2, b=b_crit, where v_r and
        # the acceleration are both exactly 0.0 forever) never satisfying
        # new_v > 0.0, so it correctly stays "lambda_max".
        now_outward = turned_outward or new_v > 0.0

        # Stop at the escape radius rather than overshooting past it.
        # Audit1 Codex P1-2: escaping trajectories previously accepted and
        # appended a whole RK4 step before checking r>=escape_radius, so
        # the recorded lambda_final/delta_phi/closest-approach-adjacent
        # diagnostics carried a step-phase error that did not shrink
        # smoothly with d_lambda -- contaminating exactly the convergence
        # comparison EXP-9 asks students to make. Linear interpolation
        # within the crossing step, symmetric with the horizon-crossing
        # interpolation above, locates the escape_radius crossing itself.
        if now_outward and new_r >= escape_radius:
            if new_r > previous_r:
                fraction = (escape_radius - previous_r) / (new_r - previous_r)
                fraction = min(1.0, max(0.0, fraction))
            else:
                fraction = 1.0

            r = escape_radius
            v_r = previous_v + fraction * (new_v - previous_v)
            phi = previous_phi + fraction * (new_phi - previous_phi)
            lambda_value = previous_lambda + fraction * h

            x_values.append(r * math.cos(phi))
            y_values.append(r * math.sin(phi))
            min_r = min(min_r, r)
            status = "escaped"
            break

        r, v_r, phi = new_r, new_v, new_phi
        lambda_value += h
        turned_outward = now_outward

        x_values.append(r * math.cos(phi))
        y_values.append(r * math.sin(phi))
        min_r = min(min_r, r)

    info = {
        "status": status,
        "closest_approach": min_r,
        "delta_phi": phi,
        "lambda_final": lambda_value,
        "steps": len(x_values) - 1,
        "r_s": r_s,
        "r_photon": r_photon,
        "critical_b_infinity": 3.0 * math.sqrt(3.0) * GM_over_c2,
        "escape_radius": escape_radius,
        "model_version": MODEL_VERSION,
        "build_id": BUILD_ID,
    }
    return x_values, y_values, info
