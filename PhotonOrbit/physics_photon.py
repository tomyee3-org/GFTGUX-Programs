"""
physics_photon.py — PhotonOrbit

General-relativistic equations of motion and an RK4 integrator for null
geodesics in the equatorial plane of Schwarzschild spacetime.
"""

import math

_MAX_STEPS = 5_000_000


def _require_finite(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")


def radial_acceleration(r, L, GM_over_c2):
    """Return d²r/dλ² for an equatorial Schwarzschild null geodesic."""
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    return (L * L / r**3) * (1.0 - 3.0 * GM_over_c2 / r)


def dphi_dlambda(r, L):
    """Return dφ/dλ = L/r²."""
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    return L / (r * r)


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

    n_steps = math.ceil(lambda_max / d_lambda)
    if n_steps > _MAX_STEPS:
        raise ValueError(
            f"lambda_max/d_lambda would require about {n_steps:,} steps; "
            f"limit the run to {_MAX_STEPS:,} steps or fewer."
        )

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

    x_values = []
    y_values = []
    min_r = r0
    status = "lambda_max"
    turned_outward = False

    for _ in range(n_steps + 1):
        x_values.append(r * math.cos(phi))
        y_values.append(r * math.sin(phi))
        min_r = min(min_r, r)

        if r <= r_s:
            status = "captured"
            break
        if turned_outward and r >= escape_radius:
            status = "escaped"
            break
        if lambda_value >= lambda_max:
            break

        remaining = lambda_max - lambda_value
        h = min(d_lambda, remaining)
        previous_v = v_r
        try:
            r, v_r, phi = rk4_step(r, v_r, phi, L, GM_over_c2, h)
        except ValueError as exc:
            raise RuntimeError(
                "Integration stepped to a nonphysical radius. Reduce d_lambda."
            ) from exc
        lambda_value += h

        if previous_v < 0.0 <= v_r:
            turned_outward = True
        if not all(math.isfinite(value) for value in (r, v_r, phi)):
            raise RuntimeError("Integration produced a non-finite value; reduce d_lambda.")

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
    }
    return x_values, y_values, info
