"""
physics_photon.py — PhotonOrbit

General-relativistic equations of motion and an RK4 integrator for null
geodesics in the equatorial plane of Schwarzschild spacetime.
"""

import math

MODEL_VERSION = "1.3.0"


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
    """Reject anything that is not a finite, non-bool int/float.

    A Python int large enough that converting it to float would itself
    overflow (for example ``10**1000``) raises OverflowError from
    math.isfinite() rather than returning False; that is treated the same
    as any other non-finite value.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real number.")


def radial_acceleration(r, L, GM_over_c2):
    """Return d²r/dλ² = (L/r)²/r · (1 - 3·GM_over_c2/r).

    Algebraically identical to (L²/r³)(1-3M/r), but computed as q=L/r
    first so that a single scale change in r, L, and GM_over_c2 together
    (as EXP-6 exercises) does not force the intermediate r³ or L² to
    underflow or overflow before the final division, even when the final
    result itself is representable.

    Validates that r, L and GM_over_c2 are finite reals, and that the
    result is finite, raising ValueError (never a raw OverflowError or
    ZeroDivisionError, and never a silent nan) for inputs too extreme for
    this program's geometric-unit scale.
    """
    _require_finite("r", r)
    _require_finite("L", L)
    _require_finite("GM_over_c2", GM_over_c2)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    try:
        q = L / r
        value = (q * q / r) * (1.0 - 3.0 * GM_over_c2 / r)
    except (OverflowError, ZeroDivisionError) as exc:
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
    """Return dφ/dλ = (L/r)/r.

    Algebraically identical to L/r², but divides by r twice rather than
    squaring r first, for the same scale-range reason as
    radial_acceleration() above.

    Validates r and L, and the result, the same way radial_acceleration()
    does.
    """
    _require_finite("r", r)
    _require_finite("L", L)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    try:
        value = (L / r) / r
    except (OverflowError, ZeroDivisionError) as exc:
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


def _require_finite_derived(name, value):
    """Validate a quantity DERIVED from already-validated inputs (r_s,
    r_photon, escape_radius, critical_b_infinity). These are simple
    products/roots of finite inputs and are finite for every ordinary
    input, but an extreme-enough GM_over_c2 (for example 1e308) can still
    push a derived multiple of it past float range; report that the same
    way as any other out-of-range input rather than letting a non-finite
    value silently reach the returned diagnostics or the integrator.
    """
    if not math.isfinite(value):
        raise ValueError(
            f"{name} is not finite for the given GM_over_c2; choose a "
            "smaller GM_over_c2 for this program's geometric-unit scale."
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
    critical_b_infinity = 3.0 * math.sqrt(3.0) * GM_over_c2
    _require_finite_derived("r_s (2*GM_over_c2)", r_s)
    _require_finite_derived("r_photon (3*GM_over_c2)", r_photon)
    _require_finite_derived("the critical impact parameter (3*sqrt(3)*GM_over_c2)",
                             critical_b_infinity)
    if r0 <= r_s:
        raise ValueError(f"r0 must be outside the event horizon (r0 > {r_s:g}).")
    if b < 0.0:
        raise ValueError("b must be nonnegative; use its magnitude as the impact parameter.")
    if lambda_max <= 0.0:
        raise ValueError("lambda_max must be greater than zero.")
    if d_lambda <= 0.0:
        raise ValueError("d_lambda must be greater than zero.")

    # Compare the ratio to the step cap BEFORE calling math.ceil() on it.
    # lambda_max/d_lambda can itself legitimately evaluate to float('inf')
    # for extreme-but-finite inputs; testing the ratio against _MAX_STEPS
    # first (a plain float comparison, well-defined even against inf)
    # rejects that case with the normal documented ValueError instead of
    # letting math.ceil(inf) raise OverflowError.
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
    escape_radius = 2.0 * r0
    _require_finite_derived("the escape radius (2*r0)", escape_radius)

    # Ratio-first, matching radial_acceleration()/dphi_dlambda(): forming
    # L*L and r0*r0 separately can underflow both to 0.0 well before the
    # final radicand -- an ordinary, dimensionless-scale-invariant case --
    # becomes representable again once divided out.
    q0 = L / r0
    initial_radicand = E * E - q0 * q0 * (1.0 - r_s / r0)
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

        # Has the photon turned outward by the END of this step? Testing
        # the CURRENT step's velocity (new_v > 0.0) directly, rather than
        # requiring a strictly negative sample beforehand, correctly
        # recognizes a purely tangential start (v_r=0.0 exactly, e.g. b at
        # the local kinematic bound outside the photon sphere) as outward
        # on its very first outward step. The exact circular orbit
        # (r0=3*GM_over_c2, b=b_crit, where v_r and the acceleration are
        # both identically 0.0 forever) never satisfies new_v > 0.0, so it
        # correctly stays status="lambda_max".
        now_outward = turned_outward or new_v > 0.0

        # Stop at the escape radius rather than overshooting past it:
        # interpolate the crossing within the step, symmetric with the
        # horizon-crossing interpolation above, so escaped trajectories'
        # lambda_final/delta_phi are located at the exact escape_radius
        # crossing rather than one whole RK4 step beyond it.
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
        "critical_b_infinity": critical_b_infinity,
        "escape_radius": escape_radius,
        "model_version": MODEL_VERSION,
        "build_id": BUILD_ID,
    }
    return x_values, y_values, info
