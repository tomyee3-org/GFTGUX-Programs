"""High-precision Schwarzschild ray helpers for BlackHoleShadow.

Near the critical impact parameter, keep intermediate values as mpmath
numbers; convert back to float64 only at the named output boundaries.
The caller supplies the captured/escaping classification so that the
high-precision integrator cannot disagree with the float64 image grid
about which side of the critical curve a ray occupies.

Ordinary rays use the faster float64 path in physics_bhs.py. These helpers
require mpmath>=1.2 and use MP_DPS decimal digits for near-critical work.
"""

from __future__ import annotations

import math

import mpmath as mp

# Working precision for near-critical calculations, also identified in Help.
MP_DPS = 30

# How close (in |eps|, i.e. |beta**2-27|) a ray must be to critical before
# the mpmath path is worth its cost.  Matches physics_bhs.py's existing
# _NEAR_CRIT_REL-derived gate; exposed here so both modules read the same
# number instead of keeping their own copies.
NEAR_CRIT_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def to_mpf(x) -> mp.mpf:
    """Convert a Python float/int to mpf, capturing exactly the float64
    value's bit pattern (no additional rounding beyond what float64
    already introduced)."""
    return mp.mpf(float(x))


def is_near_critical(eps_float: float, threshold: float = NEAR_CRIT_EPS) -> bool:
    """Cheap float64 pre-check: is |eps| small enough to justify mpmath?

    Callers that already have a float64 eps (from physics_bhs.py's own
    _beta_sq_minus_27) can use this instead of recomputing eps here, so
    the "should I bother with mpmath" decision stays free of mpmath
    overhead until the answer is yes.
    """
    return math.isfinite(eps_float) and abs(eps_float) < threshold


# ---------------------------------------------------------------------------
# Dimensionless radicand, in mpf throughout.
# ---------------------------------------------------------------------------

def _R_hat_mpf(x: mp.mpf, beta: mp.mpf) -> mp.mpf:
    """R_hat(x) = 1/beta**2 - x**2 + 2 x**3, at whatever workdps is active."""
    return 1 / (beta * beta) - x * x + 2 * x ** 3


def _dR_hat_dx_mpf(x: mp.mpf) -> mp.mpf:
    """d/dx of -x**2+2x**3 (the beta-independent part; 1/beta**2 is constant)."""
    return -2 * x + 6 * x * x


# ---------------------------------------------------------------------------
# Periapsis (escaping rays only): rho = r/M > 3, dimensionless.
# ---------------------------------------------------------------------------

def periapsis_over_M_mpf(beta: mp.mpf, dps: int = MP_DPS) -> mp.mpf:
    """Outer turning point rho=r/M, solving rho**3 - beta**2 rho + 2 beta**2 = 0.

    Same sigma-Newton parameterization physics_bhs.py's float64 code uses
    (rho=3+sigma, beta**2=27+eps => sigma**2(9+sigma) = eps(1+sigma),
    stable at the near-double root), just carried out entirely in mpf.

    Returns an mpf.  Deliberately dimensionless (rho, not r=rho*M): a
    caller who needs r in physical units multiplies by M themselves, in
    whatever precision that final step needs, rather than this function
    baking a specific M in (and rather than risking overflow/underflow at
    the M~1e-170/1e150 extremes this exchange has tested at, for no
    benefit -- the M-scaling is exact and doesn't need 30 digits of help).

    Raises ValueError if beta is not > sqrt(27) (i.e. not escaping) --
    callers are expected to have already classified the ray dimensionally
    in physics_bhs.py; this function does not re-derive that.
    """
    with mp.workdps(dps + 10):  # headroom so the returned digits are clean at dps
        beta = mp.mpf(beta)
        eps = beta * beta - 27
        if eps <= 0:
            raise ValueError(
                "periapsis_over_M_mpf requires beta > sqrt(27) (an escaping ray); "
                "got beta**2-27 = %s. Classify escaped/captured in physics_bhs.py "
                "before calling this." % eps
            )
        sig = mp.sqrt(eps / 9)
        tol = mp.mpf(10) ** (-(dps + 5))
        for _ in range(100):
            f = sig * sig * (9 + sig) - eps * (1 + sig)
            df = 2 * sig * (9 + sig) + sig * sig - eps
            if df == 0:
                break
            sig_new = sig - f / df
            if sig_new <= 0:
                sig_new = sig / 2
            if abs(sig_new - sig) <= tol * max(mp.mpf(1), abs(sig)):
                sig = sig_new
                break
            sig = sig_new
        rho = 3 + sig
        with mp.workdps(dps):
            rho = +rho
    return rho


def periapsis_over_M(b, M, dps: int = MP_DPS) -> float:
    """Public float64 entry point for the escaping periapsis divided by M.

    Call the combined endpoint/azimuth helper when both quantities are
    needed; do not round the turning point and feed it back into mpmath.
    """
    with mp.workdps(dps + 10):
        beta = to_mpf(b) / to_mpf(M)
        rho = periapsis_over_M_mpf(beta, dps=dps)
    return float(rho)


# ---------------------------------------------------------------------------
# Inbound phi integral, in mpf throughout, from x_lo out to x_hi.
# ---------------------------------------------------------------------------

def _turning_segment_mpf(beta: mp.mpf, a: mp.mpf, c: mp.mpf) -> mp.mpf:
    """integral_a^c dx/sqrt(R_hat(x)) where R_hat(c)=0 (c is a turning point).

    t**2 = c-x substitution; the t=0 endpoint (the sqrt singularity) is
    handled analytically via R_hat's derivative rather than sampled, so
    there's no grid to under-resolve no matter how close a and c sit to
    the near-double root at x=1/3.
    """
    def g(t):
        if t == 0:
            d = _dR_hat_dx_mpf(c)
            return 2 / mp.sqrt(abs(d)) if d != 0 else mp.mpf(0)
        x = c - t * t
        rv = _R_hat_mpf(x, beta)
        return 2 * t / mp.sqrt(rv) if rv > 0 else mp.mpf(0)

    t_max = mp.sqrt(c - a)
    return mp.quad(g, [mp.mpf(0), t_max])


def _plain_segment_mpf(beta: mp.mpf, a: mp.mpf, c: mp.mpf) -> mp.mpf:
    """integral_a^c dx/sqrt(R_hat(x)) with no singularity expected in (a, c)."""
    def f(x):
        rv = _R_hat_mpf(x, beta)
        return 1 / mp.sqrt(rv) if rv > 0 else mp.mpf(0)

    return mp.quad(f, [a, c])


def phi_segment_mpf(
    x_lo,
    x_hi,
    beta: mp.mpf,
    dps: int = MP_DPS,
    turning: bool | None = None,
) -> mp.mpf:
    """Integrate the inbound azimuth segment while retaining mpmath precision.

    Split at x=1/3 whenever the photon-sphere feature lies inside the
    integration interval. Turning points use a squared-distance change
    of variable; ordinary endpoints use direct quadrature. A supplied
    turning flag avoids inferring the endpoint type from a rounded root.
    """
    with mp.workdps(dps + 10):
        x_lo = mp.mpf(x_lo)
        x_hi = mp.mpf(x_hi)
        beta = mp.mpf(beta)
        if x_hi <= x_lo:
            return mp.mpf(0)

        if turning is None:
            turning = abs(_R_hat_mpf(x_hi, beta)) < mp.mpf(10) ** (-(dps - 5))

        third = mp.mpf(1) / 3
        cuts = [x_lo]
        if cuts[0] < third < x_hi:
            cuts.append(third)
        cuts.append(x_hi)

        total = mp.mpf(0)
        for a, c in zip(cuts[:-1], cuts[1:]):
            if c <= a:
                continue
            if turning and c == cuts[-1]:
                total += _turning_segment_mpf(beta, a, c)
            else:
                total += _plain_segment_mpf(beta, a, c)
        with mp.workdps(dps):
            total = +total
    return total


def phi_segment(x_lo, x_hi, b, M, dps: int = MP_DPS, turning: bool | None = None) -> float:
    """Public float64 entry point for phi_segment_mpf, given physical b, M
    and dimensionless x=M*u endpoints already computed by the caller.

    Suitable as the drop-in fallback inside physics_bhs.py's
    _phi_quad_segment: replace
        return _phi_factored_trap(x_lo, x_hi, beta, eps, n=4096)
    with
        return utilities_bhs.phi_segment(x_lo, x_hi, b, M)
    (physics_bhs.py already has beta on hand there; passing b, M instead
    of beta directly is deliberate, so this module's own to_mpf(b)/to_mpf(M)
    is what feeds the mpf beta, not a beta that itself already round-
    tripped through anything upstream).
    """
    with mp.workdps(dps + 10):
        beta = to_mpf(b) / to_mpf(M)
        result = phi_segment_mpf(x_lo, x_hi, beta, dps=dps, turning=turning)
    return float(result)


def phi_to_endpoint_over_M(
    b,
    M,
    captured: bool,
    horizon_over_M: float = 2.0,
    horizon_margin: float = 1.0000001,
    dps: int = MP_DPS,
) -> float:
    """Return the inbound azimuth to the turning or near-horizon endpoint.

    Keep the periapsis and integration endpoint in mpmath precision until
    the final float64 return. The caller provides the captured status
    based on its dimensional critical-impact-parameter comparison.
    """
    with mp.workdps(dps + 10):
        beta = to_mpf(b) / to_mpf(M)
        if captured:
            rho_hi = mp.mpf(horizon_over_M) * mp.mpf(horizon_margin)
            x_hi = 1 / rho_hi
            phi = phi_segment_mpf(0, x_hi, beta, dps=dps, turning=False)
        else:
            rho = periapsis_over_M_mpf(beta, dps=dps)  # mpf; never touches float()
            x_hi = 1 / rho
            phi = phi_segment_mpf(0, x_hi, beta, dps=dps, turning=True)
    return float(phi)
