"""utilities_bhs.py -- shared high-precision (mpmath) numerics for BlackHoleShadow.

Proposed new fifth module, per Tom's direction following Claude's Audit 18
(BlackHoleShadow-Grok-Response-to-Audit17, v0.18.0) and the follow-up
discussion about that round's two related findings:

  1. _phi_quad_segment's fallback threshold discarded accurate SciPy
     answers in favor of a hand-rolled trapezoid that was demonstrably
     worse (rejecting a good answer at a boundary that doesn't trust it).
  2. _mp_periapsis returned float(...) and that already-rounded value was
     reused to build x_hi one level up in _mp_phi_to_endpoint, discarding
     most of the 30-digit solve (discarding a good answer at a boundary
     that doesn't preserve it).

Both are the same class of bug: a correct high-precision result loses its
precision -- or gets thrown away outright -- while crossing a boundary
between components.  This module exists to make that class of bug
mechanically harder to write, via one governing rule:

    GOVERNING RULE: every function in this module that does high-precision
    work takes mpmath values (mpf) in and returns mpf out.  Nothing in
    here ever rounds an intermediate result to a Python float and then
    keeps computing with it.  The ONLY functions that return a plain
    float are the ones named *_over_M / *_float below, which exist
    specifically to be the single, designated conversion point back to
    physics_bhs.py's float64 world -- call one of THOSE from physics_bhs.py,
    never re-derive an mpf from a float64 value that this module already
    had at full precision.

A second design rule, worth stating explicitly because getting it wrong
is exactly how a previous round's _dimensionless_state saga happened:
this module does NOT independently decide whether a ray is escaping or
captured.  Classification (b vs. b_crit) stays the sole responsibility of
physics_bhs.py's own critical_impact_parameter(M) and the dimensional
b > b_crit / b < b_crit comparison already used everywhere else in that
file.  Every function below that needs to know which regime it's in takes
an explicit `captured` flag from the caller, rather than recomputing its
own high-precision b_crit and risking disagreement with physics_bhs.py's
float64 classification at the boundary.

Also folds in, as one place instead of five: the eps=beta**2-27 helper,
the near-critical detector, and the periapsis/phi-integral solvers that
have been rewritten piecemeal across many rounds (_signed_eps,
_beta_offset, _dimensionless_state, _beta_sq_minus_27, _two_square,
_mp_periapsis, _mp_phi_to_endpoint, _phi_factored_trap all overlap with
what lives here now). physics_bhs.py's Dekker/FMA compensation for
beta**2-27 is a float64-only trick and has no equivalent need here: at
MP_DPS=30 the cancellation in beta*beta-27 costs far fewer digits than
float64 ever had to spare in the first place, so plain mpmath arithmetic
is already exact enough.

Requires mpmath>=1.2 (pure Python, no compiled extension -- no Python
version constraint beyond what BlackHoleShadow already requires).
"""

from __future__ import annotations

import math

import mpmath as mp

# Keep this in sync with the Help file's stated precision (Audit 18 found
# them disagreeing: code said 30, Help said "~50" -- whichever number is
# true, it should only be written down in one place).
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
    return +rho  # round back to the requested dps on return (mpmath idiom)


def periapsis_over_M(b, M, dps: int = MP_DPS) -> float:
    """Public float64 entry point: periapsis/M for an escaping ray.

    This is the ONLY function in this module allowed to hand back a
    plain float for the periapsis -- call this from physics_bhs.py's
    periapsis(), never reconstruct rho from a float you already rounded.
    If you need the periapsis AND the phi-integral to the periapsis in
    the same call, use phi_to_endpoint_over_M below instead of calling
    this first and feeding its float result back in: that round-trip
    through float64 is exactly Audit 18's second finding.
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
    """integral_{x_lo}^{x_hi} dx/sqrt(R_hat(x)), all mpf, splitting at x=1/3
    whenever that point lies strictly inside the domain.

    This is the general-purpose building block: it is the direct,
    trustworthy replacement for BOTH _phi_factored_trap (physics_bhs.py's
    fixed-grid fallback) and, when called from the near-critical gate,
    _mp_phi_to_endpoint's integral.  Use it as _phi_quad_segment's
    fallback -- instead of falling back to a method that is *worse* than
    the SciPy answer being discarded, fall back to a method that is
    *unconditionally at least as good*, which is the actual fix for
    Audit 18's first finding (the fallback threshold itself can stay
    exactly as trigger-happy as it likes once the fallback is trustworthy).

    x=1/3 is where the photon-sphere feature narrows as eps -> 0; missing
    this split is the mechanism behind every captured-ray near-critical
    bug this exchange has found (Audits 15 and 17).  It is included here
    unconditionally, not just "when near-critical", because it costs
    nothing when eps is not small (the two mp.quad calls are just as fast
    as one over a smooth integrand) and removes an entire class of "did
    someone remember to split this time" mistakes.

    turning=True asserts x_hi is a genuine turning point (R_hat(x_hi)=0);
    turning=False asserts it is not (an ordinary endpoint, e.g. a
    bisection trial point or the near-horizon endpoint for a captured
    ray). Leave as None to auto-detect via |R_hat(x_hi)| against a
    dps-scaled tolerance -- safe, but pass the flag explicitly when the
    caller already knows, since auto-detection near a genuine but not
    yet fully mpmath-converged turning point is exactly the kind of
    judgment call this module exists to make once, correctly, rather
    than leaving to a heuristic in every caller.
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
    return +total


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
    """Public float64 entry point: inbound phi from infinity to the outer
    turning point (escaping) or the near-horizon endpoint (captured).

    This is the direct replacement for _mp_phi_to_endpoint, and it is
    where Audit 18's second finding is actually fixed: the escaping
    branch calls periapsis_over_M_mpf and keeps rho as an mpf all the way
    through building x_hi -- it never calls periapsis_over_M (the
    float-returning wrapper) internally, so nothing here rounds to
    float64 before the phi integral has consumed the full-precision
    turning point.

    `captured` must come from physics_bhs.py's own dimensional b vs.
    b_crit comparison (see the module docstring's second design rule).
    `horizon_over_M` / `horizon_margin` default to Schwarzschild's r_s=2M
    with the existing 1.0000001 near-horizon convention, matching
    physics_bhs.py's _max_inbound_u exactly -- pass physics_bhs.py's own
    event_horizon(M)/M if that convention ever changes, so this module
    never hard-codes a physical assumption physics_bhs.py doesn't also
    make.
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
