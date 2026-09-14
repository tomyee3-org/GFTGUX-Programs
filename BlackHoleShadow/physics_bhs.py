"""
physics_bhs.py
==============
Schwarzschild image-plane engine for the BlackHoleShadow teaching program.

PhotonOrbit integrates one equatorial null geodesic and reports its status
and accumulated azimuth.  GravitationalLensing maps a source plane through
a thin-lens kink.  This module does neither of those jobs a second time.
It takes a grid of image-plane impact parameters (b_x, b_y) and classifies
each direction on the sky as captured, escaped, or a high-winding escaper
near the photon sphere.

Geometric units: the single scale is M = GM/c^2.  Command-line length
flags are multiples of M (r_cam/M, b/M, fov/M, r_hot/M).  The integrator
converts those ratios to absolute lengths by multiplying by M.  Camera
grids are built so a fixed --fov (in units of M) frames the same
dimensionless picture at every M.

Nothing here is Kerr, plasma, radiative transfer, or an EHT pipeline.
The face-on image uses static emitters (no orbital Doppler).
Inclination is not a student parameter in this version.
"""

from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache

import numpy as np

try:
    from scipy.integrate import quad as _scipy_quad
    import mpmath as mp
    import utilities_bhs
except Exception as exc:
    raise ImportError(
        "BlackHoleShadow requires SciPy, NumPy, Matplotlib, and mpmath. "
        "Install the packages listed in requirements.txt."
    ) from exc

MODEL_VERSION = "1.0.0"
_MP_DPS = utilities_bhs.MP_DPS
# Highest crossing index computed for the toy image (m = 1..MAX_IMAGE_M).
# Photon-ring panels start at m=3.
MAX_IMAGE_M = 4


def photon_order_label(max_m=None):
    """User-facing photon-ring range m=3..MAX_IMAGE_M."""
    max_m = MAX_IMAGE_M if max_m is None else int(max_m)
    if max_m < 3:
        return ""
    if max_m == 3:
        return "m=3"
    if max_m == 4:
        return "m=3,4"
    return f"m=3..{max_m}"


def image_order_pair_label(max_m=None):
    """Back-compat alias for photon_order_label."""
    return photon_order_label(max_m)


def omitted_order_label(max_m=None):
    max_m = MAX_IMAGE_M if max_m is None else int(max_m)
    return f"m>={max_m + 1}"


def image_sum_clause(max_m=None):
    max_m = MAX_IMAGE_M if max_m is None else int(max_m)
    return f"sum_{{m=1..{max_m}}} g^4 I_em;  {omitted_order_label(max_m)} omitted"

# RK4 null-geodesic stepper is kept in lockstep with PhotonOrbit 1.4.0
# (GFTGUX-Programs/PhotonOrbit).  This package does not import that
# program; the copy exists so a BlackHoleShadow zip runs standalone.
PHOTONORBIT_SYNC_VERSION = "1.4.0"

# Tom's explicit intent: BUILD_ID hashes the live program modules.
# Help and tests are versioned separately and are not part of BUILD_ID.
BUILD_ID_COVERS = (
    "physics_bhs.py",
    "utilities_bhs.py",
    "driver_bhs.py",
    "main.py",
    "plot_bhs.py",
)


def compute_build_id_from_directory(directory):
    """Hash the covered program modules found in ``directory``.

    Tests copy those files into a temporary folder and call this; they
    must not write the live source tree.
    """
    digest = hashlib.sha256()
    for name in BUILD_ID_COVERS:
        path = os.path.join(directory, name)
        with open(path, "r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def _compute_build_id():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return compute_build_id_from_directory(here)
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# ----------------------------------------------------------------------
# Physical constants (SI, CODATA 2022 / IAU) — compare mode only
# ----------------------------------------------------------------------
G = 6.674_30e-11
C_LIGHT = 2.997_924_58e8
M_SUN = 1.988_47e30
PC = 3.085_677_581e16
GPC = 1.0e9 * PC
D_L_DEFAULT = 1.0 * GPC
D_S_DEFAULT = 2.0 * GPC

# Smallest number of grid intervals the shadow diameter 2 b_crit must span.
SHADOW_MIN_INTERVALS = 8.0

# Automatic n_pix cap.  Wider fields need an explicit --n_pix.
N_PIX_AUTO_MAX = 401

# Default camera radius as a multiple of M.  Large compared with 3 so the
# conserved b is essentially the impact parameter at infinity.
R_CAM_DEFAULT = 40.0

# Window around the photon sphere, as a multiple of M, used to tag a
# high-winding overlay pixel.  Not a fitted photon-ring profile.
PHOTON_RING_R_WINDOW = 0.35

# Coarse-camera cap for pixels mode.  Dense maps use --n_pix as given.
PIXELS_N_PIX_MAX = 41
PIXELS_N_PIX_DEFAULT = 17
DENSE_N_PIX_DEFAULT = 161

# Capture/ring maps must contain b_crit with this half-width margin.
FOV_CONTAIN_MARGIN = 1.25

# compare-mode log10(M/M_sun) teaching range.
LOGM_MIN = 6.0
LOGM_MAX = 15.0

# Azimuth threshold that counts as "wound once" on an escaping ray.
WINDING_DELTA_PHI = 2.0 * math.pi

# Emitted-annulus Gaussian half-width as a multiple of M.
HOT_RING_WIDTH_DEFAULT = 0.45
# Source-table / image contract.  emitted_intensity itself accepts any
# positive width; the table sampler does not resolve narrower Gaussians.
HOT_RING_WIDTH_MIN = 1.0e-2
# asymptotic_deflection is not claimed at closer offsets than this
# relative gap (b-b_crit)/b_crit.
DEFLECTION_MIN_REL = 1.0e-4

# Image-mode r_hot/M must stay below this at the n_pix cap so
# containment (1.4 r_hot) and the 8-interval shadow rule can both hold.
def max_image_r_hot_over_M(n_pix=None):
    """Largest r_hot/M that still satisfies image framing and sampling."""
    if n_pix is None:
        n_pix = N_PIX_AUTO_MAX
    n_pix = max(int(n_pix), 9)
    b_c = 3.0 * math.sqrt(3.0)
    fov_max = (n_pix - 1) * b_c / 4.0
    return fov_max / 2.8

# Default hot-ring coordinate radius (a few r_s, outside the photon sphere).
HOT_RING_R_DEFAULT = 6.0

_MAX_STEPS = 5_000_000


def _require_finite(name, value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _require_positive(name, value):
    value = _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value:g}")
    return value


def validate_user_value(name, value, *, positive=False, min_value=None,
                        max_value=None, exclusive_max=None):
    value = _require_finite(name, value)
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value:g}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value:g}, got {value:g}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value:g}, got {value:g}")
    if exclusive_max is not None and value >= exclusive_max:
        raise ValueError(f"{name} must be < {exclusive_max:g}, got {value:g}")
    return value


def event_horizon(M):
    M = _require_positive("M", M)
    r_s = 2.0 * M
    if not math.isfinite(r_s):
        raise ValueError("r_s is not finite for the given M.")
    return r_s


def photon_sphere(M):
    M = _require_positive("M", M)
    r_ph = 3.0 * M
    if not math.isfinite(r_ph):
        raise ValueError("r_photon is not finite for the given M.")
    return r_ph


def critical_impact_parameter(M):
    """b_crit = 3 sqrt(3) M, the capture threshold from infinity."""
    M = _require_positive("M", M)
    b_crit = 3.0 * math.sqrt(3.0) * M
    if not math.isfinite(b_crit):
        raise ValueError("b_crit is not finite for the given M.")
    return b_crit


def weak_field_deflection(b, M):
    """hat{alpha} = 4M/b, the large-b limit of a Schwarzschild photon."""
    M = _require_positive("M", M)
    b = _require_positive("b", b)
    return 4.0 * M / b


def radicand(r, b, M):
    """1 - (1 - 2M/r) (b/r)^2, the quantity under the square root of (dr/dλ)^2.

    Affine parameter normalized so E = 1 and L = b, matching PhotonOrbit.
    """
    r = _require_finite("r", r)
    b = _require_finite("b", b)
    M = _require_positive("M", M)
    if r <= 0.0:
        raise ValueError("r must remain positive.")
    q = b / r
    return 1.0 - q * q * (1.0 - 2.0 * M / r)


def is_captured(b, M):
    """True if an ingoing ray from outside the photon sphere is captured.

    For a static camera at any r_cam > 3M the condition is the same as
    the condition at infinity: b < b_crit.  Equality is the unstable
    circular photon orbit and is treated as captured for the silhouette
    (a set of measure zero on the grid).
    """
    b = _require_finite("b", b)
    if b < 0.0:
        raise ValueError("b must be nonnegative.")
    return b <= critical_impact_parameter(M)


def periapsis(b, M):
    """Outer turning point of an escaping null geodesic, or None if captured.

    Solves r^3 - b^2 r + 2 M b^2 = 0 for the root r > 3M.
    """
    b = _require_finite("b", b)
    M = _require_positive("M", M)
    if b < 0.0:
        raise ValueError("b must be nonnegative.")
    b_crit = critical_impact_parameter(M)
    if b <= b_crit:
        return None
    beta, eps = _beta_offset(b, M)
    if eps <= 0.0:
        # b > b_crit already; do not reclassify by a rounded beta.
        eps = math.ulp(27.0)
    rel = (b - b_crit) / b_crit
    if _is_near_critical(b, M):
        return _mp_periapsis(b, M)
    if rel < 1.0e-2:
        # ρ = 3+σ, β² = 27+ε  =>  σ²(9+σ) = ε(1+σ).  Stable at the double root.
        sig = math.sqrt(eps / 9.0)
        for _ in range(40):
            f = sig * sig * (9.0 + sig) - eps * (1.0 + sig)
            df = 2.0 * sig * (9.0 + sig) + sig * sig - eps
            if df == 0.0:
                break
            sig_new = sig - f / df
            if sig_new <= 0.0:
                sig_new = 0.5 * sig
            if abs(sig_new - sig) <= 1.0e-16 * max(1.0, abs(sig)):
                sig = sig_new
                break
            sig = sig_new
        rho = 3.0 + sig
    else:
        rho = max(beta, 3.25)
        for _ in range(80):
            f = rho ** 3 - beta * beta * rho + 2.0 * beta * beta
            df = 3.0 * rho * rho - beta * beta
            if df == 0.0:
                rho = 3.5 + 0.5 * beta
                continue
            rho_new = rho - f / df
            if rho_new <= 3.0:
                rho_new = 0.5 * (rho + 3.0) + 0.05
            if abs(rho_new - rho) <= 1.0e-14 * max(1.0, abs(rho)):
                rho = rho_new
                break
            rho = rho_new
    if rho <= 3.0 or not math.isfinite(rho):
        raise RuntimeError("periapsis solver failed for the given b, M.")
    return rho * M


def asymptotic_deflection(b, M, n_u=None):
    """Asymptotic scattering deflection hat{alpha} = Delta phi_inf - pi.

    Uses the standard substitution u = 1/r:

        Delta phi_inf = 2 * integral_0^{u_min} b / sqrt(1 - b^2 u^2 + 2 M b^2 u^3) du

    Captured rays (b <= b_crit) have no scattering deflection; this
    function raises ValueError for them.  The integrand is integrable at
    the turning point.  Near b_crit the result grows without bound.  That strong-deflection
    divergence permits higher-order images; it is not itself a photon ring.
    Offsets closer than DEFLECTION_MIN_REL are refused here.  Use
    escaping_azimuth_from_infinity when only a winding threshold is needed.
    """
    b = _require_positive("b", b)
    M = _require_positive("M", M)
    b_crit = critical_impact_parameter(M)
    if is_captured(b, M):
        raise ValueError(
            "asymptotic_deflection is defined only for escaping rays "
            f"(b > b_crit = {b_crit:g})."
        )
    if (b - b_crit) / b_crit < DEFLECTION_MIN_REL:
        raise ValueError(
            "asymptotic_deflection is not resolved closer than "
            f"{DEFLECTION_MIN_REL:g} relative to b_crit; "
            "the strong-field divergence is real, the number is not."
        )
    r_min = periapsis(b, M)
    beta = b / M
    x_min = M / r_min
    delta = abs(beta - 3.0 * math.sqrt(3.0))
    if n_u is None:
        n_u = 2048
        if delta < 1.0e-2:
            n_u = 8192
        if delta < 1.0e-4:
            n_u = 32768
    n_u = max(int(n_u), 512)
    t = np.linspace(0.0, math.sqrt(x_min), n_u)
    x = np.clip(x_min - t * t, 0.0, x_min)
    rad = 1.0 - (beta * beta) * x * x + 2.0 * (beta * beta) * x * x * x
    rad = np.maximum(rad, 0.0)
    piece = np.zeros_like(t)
    safe = rad > 0.0
    piece[safe] = (2.0 * t[safe]) / np.sqrt(rad[safe])
    R_min = float(1.0 - (beta * beta) * x_min * x_min
                  + 2.0 * (beta * beta) * x_min * x_min * x_min)
    if R_min <= 1.0e-14:
        drad_dx = -2.0 * beta * beta * x_min + 6.0 * beta * beta * x_min * x_min
        slope = abs(float(drad_dx))
        if slope > 0.0:
            piece[0] = 2.0 / math.sqrt(slope)
    delta_phi_inf = 2.0 * beta * _trapz(piece, t)
    return delta_phi_inf - math.pi


def escaping_azimuth_from_infinity(b, M):
    """Total asymptotic azimuth Delta phi_inf for an escaping ray.

    Uses asymptotic_deflection when the offset is resolved.  Closer to
    b_crit, uses the Schwarzschild logarithmic strong-deflection tail
    calibrated at DEFLECTION_MIN_REL so a winding threshold cannot be
    mistaken for an empty physical window.
    """
    b = _require_positive("b", b)
    M = _require_positive("M", M)
    b_crit = critical_impact_parameter(M)
    if b <= b_crit:
        raise ValueError("escaping_azimuth_from_infinity needs b > b_crit")
    rel = (b - b_crit) / b_crit
    if rel >= DEFLECTION_MIN_REL:
        return asymptotic_deflection(b, M) + math.pi
    rel_ref = 2.0 * DEFLECTION_MIN_REL
    b_ref = b_crit * (1.0 + rel_ref)
    alpha_ref = asymptotic_deflection(b_ref, M)
    alpha = -math.log(rel) + alpha_ref + math.log(rel_ref)
    return alpha + math.pi


def radial_acceleration(r, L, M):
    """d²r/dλ² = (L/r)²/r · (1 - 3M/r), PhotonOrbit's first integral."""
    r = _require_finite("r", r)
    L = _require_finite("L", L)
    M = _require_positive("M", M)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    q = L / r
    value = (q * q / r) * (1.0 - 3.0 * M / r)
    if not math.isfinite(value):
        raise ValueError("radial_acceleration produced a non-finite result.")
    return value


def dphi_dlambda(r, L):
    """dφ/dλ = L/r²."""
    r = _require_finite("r", r)
    L = _require_finite("L", L)
    if r <= 0.0:
        raise ValueError("r must remain positive during integration.")
    value = (L / r) / r
    if not math.isfinite(value):
        raise ValueError("dphi_dlambda produced a non-finite result.")
    return value


def _derivatives(r, v_r, phi, L, M):
    del phi
    return (v_r, radial_acceleration(r, L, M), dphi_dlambda(r, L))


def rk4_step(r, v_r, phi, L, M, d_lambda):
    k1 = _derivatives(r, v_r, phi, L, M)
    r2 = r + 0.5 * d_lambda * k1[0]
    v2 = v_r + 0.5 * d_lambda * k1[1]
    p2 = phi + 0.5 * d_lambda * k1[2]
    k2 = _derivatives(r2, v2, p2, L, M)
    r3 = r + 0.5 * d_lambda * k2[0]
    v3 = v_r + 0.5 * d_lambda * k2[1]
    p3 = phi + 0.5 * d_lambda * k2[2]
    k3 = _derivatives(r3, v3, p3, L, M)
    r4 = r + d_lambda * k3[0]
    v4 = v_r + d_lambda * k3[1]
    p4 = phi + d_lambda * k3[2]
    k4 = _derivatives(r4, v4, p4, L, M)
    factor = d_lambda / 6.0
    return (
        r + factor * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        v_r + factor * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        phi + factor * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    )


def integrate_photon_orbit(M, r0, b, lambda_max=200.0, d_lambda=0.02):
    """Initially ingoing equatorial null geodesic.  PhotonOrbit's ODE.

    Affine parameter normalized so E = 1, L = b.  ``lambda_max`` and
    ``d_lambda`` are absolute lengths in the same unit as M (so they
    scale with M when the caller treats CLI flags as multiples of M).
    The stepper itself runs in units of M, so the dimensionless
    trajectory is independent of the numerical value of M.
    Returns (x, y, info) with status in {'captured', 'escaped', 'lambda_max'}.
    """
    M = _require_positive("M", M)
    r0 = _require_finite("r0", r0)
    b = _require_finite("b", b)
    lambda_max = _require_positive("lambda_max", lambda_max)
    d_lambda = _require_positive("d_lambda", d_lambda)
    xs_hat, ys_hat, info = _integrate_photon_orbit_hat(
        r0 / M, b / M, lambda_max / M, d_lambda / M,
    )
    xs = [x * M for x in xs_hat]
    ys = [y * M for y in ys_hat]
    info["r_s"] = event_horizon(M)
    info["r_photon"] = photon_sphere(M)
    info["critical_b_infinity"] = critical_impact_parameter(M)
    info["escape_radius"] = info["escape_radius_over_M"] * M
    info["closest_approach"] = info["closest_approach_over_M"] * M
    info["lambda_final"] = info["lambda_final_over_M"] * M
    return xs, ys, info


def _integrate_photon_orbit_hat(r0, b, lambda_max, d_lambda):
    """Integrator in units of M (internal M = 1)."""
    M = 1.0
    r_s = 2.0
    r_ph = 3.0
    b_crit = 3.0 * math.sqrt(3.0)
    if r0 <= r_s:
        raise ValueError(
            f"--r_cam must be outside the event horizon "
            f"(r_cam > 2M = {r_s:g} in the same units as M)."
        )
    if b < 0.0:
        raise ValueError("b must be nonnegative.")
    step_ratio = lambda_max / d_lambda
    if not math.isfinite(step_ratio) or step_ratio > _MAX_STEPS:
        raise ValueError(
            f"lambda_max/d_lambda would require more than {_MAX_STEPS:,} steps."
        )
    n_steps = math.ceil(step_ratio)
    L = b
    escape_radius = 2.0 * r0
    b_max = r0 * math.sqrt(r0 / (r0 - r_s))
    if b > b_max + 4.0 * math.ulp(b_max):
        raise ValueError(
            "The requested b is incompatible with an initially ingoing "
            f"null geodesic at r0={r0:g}.  b must be <= {b_max:.8g}."
        )
    q0 = L / r0
    initial_radicand = max(0.0, 1.0 - q0 * q0 * (1.0 - r_s / r0))
    r = r0
    v_r = -math.sqrt(initial_radicand)
    phi = 0.0
    lambda_value = 0.0
    xs = [r * math.cos(phi)]
    ys = [r * math.sin(phi)]
    min_r = r0
    status = "lambda_max"
    turned_outward = False
    for _ in range(n_steps):
        if lambda_value >= lambda_max:
            break
        h = min(d_lambda, lambda_max - lambda_value)
        previous_r, previous_v, previous_phi, previous_lambda = r, v_r, phi, lambda_value
        try:
            new_r, new_v, new_phi = rk4_step(r, v_r, phi, L, M, h)
        except ValueError as exc:
            raise RuntimeError(
                "Integration stepped to a nonphysical radius. Reduce d_lambda."
            ) from exc
        if not all(math.isfinite(v) for v in (new_r, new_v, new_phi)):
            raise RuntimeError("Integration produced a non-finite value.")
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
            xs.append(r * math.cos(phi))
            ys.append(r * math.sin(phi))
            min_r = min(min_r, r)
            status = "captured"
            break
        now_outward = turned_outward or new_v > 0.0
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
            xs.append(r * math.cos(phi))
            ys.append(r * math.sin(phi))
            min_r = min(min_r, r)
            status = "escaped"
            break
        r, v_r, phi = new_r, new_v, new_phi
        lambda_value += h
        turned_outward = now_outward
        xs.append(r * math.cos(phi))
        ys.append(r * math.sin(phi))
        min_r = min(min_r, r)
    info = {
        "status": status,
        "closest_approach_over_M": min_r,
        "delta_phi": phi,
        "lambda_final_over_M": lambda_value,
        "steps": len(xs) - 1,
        "escape_radius_over_M": escape_radius,
        "model_version": MODEL_VERSION,
        "build_id": BUILD_ID,
    }
    return xs, ys, info


def to_absolute_length(value_over_M, M, name="length"):
    """Convert a user length given in units of M to an absolute length."""
    value_over_M = _require_finite(name, value_over_M)
    M = _require_positive("M", M)
    if value_over_M < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value_over_M:g}")
    return value_over_M * M


def require_camera_outside_photon_sphere(r_cam, M):
    """This lesson's capture predicate needs r_cam > 3M."""
    r_cam = _require_positive("r_cam", r_cam)
    r_ph = photon_sphere(M)
    if r_cam <= r_ph:
        raise ValueError(
            f"--r_cam must lie outside the photon sphere "
            f"(r_cam > 3M = {r_ph:g} in the same units as M).  "
            "Inside 3M the capture cone is a different lesson."
        )
    return r_cam


def make_impact_grid(n_pix, fov_over_M, M=1.0):
    """Square camera grid of conserved impact parameters.

    ``fov_over_M`` is the field of view on a side, in units of M.
    Returned ``bx, by`` are absolute lengths (same unit as M), running
    from -0.5*fov_over_M*M to +0.5*fov_over_M*M inclusive.  n_pix is
    rounded up to the next odd integer so a pixel sits on the axis.
    """
    n_pix = int(n_pix)
    if n_pix < 9:
        raise ValueError("n_pix must be an integer >= 9")
    if n_pix % 2 == 0:
        n_pix += 1
    fov_over_M = _require_positive("fov", fov_over_M)
    M = _require_positive("M", M)
    half = 0.5 * fov_over_M * M
    axis = np.linspace(-half, half, n_pix)
    bx, by = np.meshgrid(axis, axis)
    return bx, by, n_pix, fov_over_M


def pixel_edge_extent(bx, by):
    """imshow extent using pixel *edges*, not pixel centres."""
    x = np.asarray(bx[0, :], dtype=float)
    y = np.asarray(by[:, 0], dtype=float)
    dx = x[1] - x[0] if x.size > 1 else 1.0
    dy = y[1] - y[0] if y.size > 1 else 1.0
    return [float(x[0] - 0.5 * dx), float(x[-1] + 0.5 * dx),
            float(y[0] - 0.5 * dy), float(y[-1] + 0.5 * dy)]


def require_resolved_shadow(M, fov_over_M, n_pix):
    """Reject a camera grid on which the shadow is smaller than a few pixels.

    The check is M-invariant once ``fov_over_M`` is a field in units of M.
    """
    b_crit = critical_impact_parameter(M)
    fov_over_M = _require_positive("fov", fov_over_M)
    n_pix = int(n_pix)
    width = fov_over_M * M
    spacing = width / max(n_pix - 1, 1)
    intervals = (2.0 * b_crit) / spacing
    if intervals < SHADOW_MIN_INTERVALS:
        raise ValueError(
            f"the shadow diameter 2 b_crit = {2.0 * b_crit / M:.4g} M spans "
            f"only {intervals:.2f} grid intervals.  Capture maps need at "
            f"least {SHADOW_MIN_INTERVALS:g} intervals so the rim is "
            "visible.  Use a smaller --fov or a larger --n_pix "
            f"(n_pix cannot exceed {N_PIX_AUTO_MAX}).  "
            "If --r_hot forced the field to grow, lower --r_hot "
            f"(image-mode maximum is {max_image_r_hot_over_M(n_pix):.3g} M "
            "at this n_pix)."
        )
    return intervals


def contained_fov(fov_over_M, M, extra_radius_over_M=0.0):
    """Grow ``fov`` so the half-width contains b_crit (and any extra radius)."""
    fov_over_M = _require_positive("fov", fov_over_M)
    need_half = max(
        FOV_CONTAIN_MARGIN * critical_impact_parameter(M) / M,
        float(extra_radius_over_M),
    )
    need = 2.0 * need_half
    return max(fov_over_M, need), need


def require_shadow_in_frame(M, fov_over_M, n_pix, extra_radius_over_M=0.0):
    """Containment then sampling.  Returns (effective_fov, intervals)."""
    fov_eff, need = contained_fov(fov_over_M, M, extra_radius_over_M)
    intervals = require_resolved_shadow(M, fov_eff, n_pix)
    return fov_eff, need, intervals


def impact_parameter_of_periapsis(r_min, M):
    """b of an escaping ray whose outer turning point is r_min > 3M."""
    M = _require_positive("M", M)
    r_min = _require_positive("r_min", r_min)
    if r_min <= photon_sphere(M):
        raise ValueError("r_min must lie outside the photon sphere.")
    return math.sqrt(r_min ** 3 / (r_min - 2.0 * M))


def high_winding_b_window(M, r_window=None, delta_phi_min=None):
    """Inclusive-outer interval (b_lo, b_hi) for the high-winding overlay.

    b_lo is just above b_crit (no artificial 1e-4 gap).  b_hi is the
    largest b whose periapsis still sits within r_window of 3M *and*
    whose asymptotic azimuth exceeds delta_phi_min.  This is a
    pedagogical highlighter, not a photon-ring brightness profile.
    """
    M = _require_positive("M", M)
    if r_window is None:
        r_window = PHOTON_RING_R_WINDOW * M
    else:
        r_window = _require_positive("r_window", r_window)
    if delta_phi_min is None:
        delta_phi_min = WINDING_DELTA_PHI
    else:
        delta_phi_min = _require_positive("delta_phi_min", delta_phi_min)
    b_crit = critical_impact_parameter(M)
    r_ph = photon_sphere(M)
    r_outer = r_ph + r_window
    b_from_r = impact_parameter_of_periapsis(r_outer, M)
    rel_max = max((b_from_r - b_crit) / b_crit, 1.0e-15)

    def qualifies(rel):
        bv = b_crit * (1.0 + rel)
        rmin = periapsis(bv, M)
        if rmin is None or rmin > r_outer:
            return False
        return escaping_azimuth_from_infinity(bv, M) > delta_phi_min

    # Search in log(rel) from the first representable escaping b.
    rel_lo = (math.nextafter(b_crit, math.inf) - b_crit) / b_crit
    if rel_lo <= 0.0:
        rel_lo = math.ulp(1.0)
    if not qualifies(rel_lo):
        return b_crit, b_crit
    if qualifies(rel_max):
        return b_crit, b_from_r
    lo, hi = rel_lo, rel_max
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if qualifies(mid):
            lo = mid
        else:
            hi = mid
    return b_crit, b_crit * (1.0 + lo)


def capture_map(bx, by, M):
    """Boolean array: True where the ray is captured (the geometric shadow)."""
    b = np.hypot(bx, by)
    return b <= critical_impact_parameter(M)


def photon_ring_mask(bx, by, M, r_window=None, delta_phi_min=None):
    """High-winding overlay: escaped pixels with b_crit < b <= b_hi.

    This is a pedagogical highlighter of strongly wound escaping rays,
    not a measurement of photon-ring brightness.  There is no artificial
    gap above b_crit.
    """
    M = _require_positive("M", M)
    if r_window is None:
        r_window = PHOTON_RING_R_WINDOW * M
    if delta_phi_min is None:
        delta_phi_min = WINDING_DELTA_PHI
    b = np.hypot(np.asarray(bx, dtype=float), np.asarray(by, dtype=float))
    b_lo, b_hi = high_winding_b_window(M, r_window=r_window,
                                       delta_phi_min=delta_phi_min)
    mask = np.zeros(b.shape, dtype=bool)
    if b_hi <= b_lo:
        return mask
    mask[(b > b_lo) & (b <= b_hi)] = True
    return mask


def require_resolved_high_winding(M, fov_over_M, n_pix):
    """Reject a ring-mode grid that cannot sample the high-winding window."""
    require_resolved_shadow(M, fov_over_M, n_pix)
    b_lo, b_hi = high_winding_b_window(M)
    width = _require_positive("fov", fov_over_M) * _require_positive("M", M)
    spacing = width / max(int(n_pix) - 1, 1)
    if b_hi - b_lo < 0.5 * spacing:
        raise ValueError(
            "the high-winding overlay spans less than half a grid interval.  "
            "Use a smaller --fov or a larger --n_pix so the overlay is visible."
        )
    return b_lo, b_hi


def deflection_curve(M, b_min_factor=1.02, b_max_factor=8.0, n=48):
    """(b, hat_alpha_exact, hat_alpha_weak) for the weak-field beat."""
    M = _require_positive("M", M)
    b_crit = critical_impact_parameter(M)
    b_lo = b_min_factor * b_crit
    b_hi = b_max_factor * b_crit
    bs = np.geomspace(b_lo, b_hi, int(n))
    exact = np.array([asymptotic_deflection(float(b), M) for b in bs])
    weak = np.array([weak_field_deflection(float(b), M) for b in bs])
    return bs, exact, weak


def einstein_radius_point_mass(m_kg, d_l=D_L_DEFAULT, d_s=D_S_DEFAULT):
    """Einstein radius of a transparent point-mass lens, in radians.

    Identical formula to GravitationalLensing.  Used only in compare mode
    so the student can put theta_E and b_crit on the same page and refuse
    to call them the same ring.
    """
    m_kg = _require_positive("m_kg", m_kg)
    d_l = _require_positive("d_l", d_l)
    d_s = _require_positive("d_s", d_s)
    if d_s <= d_l:
        raise ValueError("d_s must exceed d_l so that D_ls is positive")
    d_ls = d_s - d_l
    return math.sqrt((4.0 * G * m_kg / C_LIGHT**2) * (d_ls / (d_l * d_s)))


def geometric_length_of_sun(m_over_msun=1.0):
    """GM/c^2 for a mass in solar units, in metres."""
    m_over_msun = _require_positive("m_over_msun", m_over_msun)
    return G * m_over_msun * M_SUN / C_LIGHT**2


def compare_rings(log10_m_galaxy=12.0, d_l=D_L_DEFAULT, d_s=D_S_DEFAULT):
    """Numbers that keep the Einstein critical curve and b_crit apart.

    Returns a dict.  Lengths are metres; angles are radians and arcsec.
    The galaxy Einstein radius in units of that galaxy's own GM/c^2 is
    enormous — that is the point of the beat.
    """
    log10_m_galaxy = _require_finite("log10_m_galaxy", log10_m_galaxy)
    if log10_m_galaxy < LOGM_MIN or log10_m_galaxy > LOGM_MAX:
        raise ValueError(
            f"log10_m_galaxy must lie in [{LOGM_MIN:g}, {LOGM_MAX:g}]."
        )
    m_kg = (10.0 ** log10_m_galaxy) * M_SUN
    theta_e = einstein_radius_point_mass(m_kg, d_l=d_l, d_s=d_s)
    M_gal_m = geometric_length_of_sun(10.0 ** log10_m_galaxy)
    r_e_physical = theta_e * d_l
    return {
        "log10_m_galaxy": log10_m_galaxy,
        "m_kg": m_kg,
        "theta_e_rad": theta_e,
        "theta_e_arcsec": math.degrees(theta_e) * 3600.0,
        "M_galaxy_m": M_gal_m,
        "r_s_galaxy_m": 2.0 * M_gal_m,
        "r_photon_galaxy_m": 3.0 * M_gal_m,
        "b_crit_galaxy_m": 3.0 * math.sqrt(3.0) * M_gal_m,
        "r_e_physical_m": r_e_physical,
        "r_e_over_M": r_e_physical / M_gal_m,
        "b_crit_over_M": 3.0 * math.sqrt(3.0),
        "r_s_over_M": 2.0,
        "r_photon_over_M": 3.0,
        "d_l_m": d_l,
        "d_s_m": d_s,
    }


def _R_of_u(u, b, M):
    """(du/dφ)^2 = 1/b^2 - u^2 + 2 M u^3."""
    return 1.0 / (b * b) - u * u + 2.0 * M * u * u * u


def _trapz(y, x):
    """NumPy 1.x trapz / 2.x trapezoid compatibility."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


_SPLITTER = 134217729.0  # 2**27 + 1, Dekker split


def _two_square(a):
    """Return (a*a, rounding_error) without requiring math.fma."""
    a = float(a)
    prod = a * a
    t = _SPLITTER * a
    hi = t - (t - a)
    lo = a - hi
    err = ((hi * hi - prod) + 2.0 * hi * lo) + lo * lo
    return prod, err


def _beta_sq_minus_27(beta):
    """Compensated beta^2 - 27.  math.fma on 3.13+; Dekker on 3.10–3.12."""
    beta = float(beta)
    if hasattr(math, "fma"):
        return math.fma(beta, beta, -27.0)
    prod, err = _two_square(beta)
    return (prod - 27.0) + err


def _dimensionless_state(b, M):
    """beta = b/M and eps = beta^2-27 from the supplied float ratio.

    Classification is dimensional (b vs b_crit).  The polynomial residual
    is the compensated square of that same ratio.  Rebuild only when the
    ratio has lost the capture sign.
    """
    b = float(b)
    M = float(M)
    b_crit = critical_impact_parameter(M)
    beta_c = 3.0 * math.sqrt(3.0)
    if b == b_crit:
        return beta_c, 0.0, False
    escaped = b > b_crit
    captured = b < b_crit
    beta = b / M
    eps = _beta_sq_minus_27(beta)
    sign_ok = (escaped and eps > 0.0) or (captured and eps < 0.0)
    if sign_ok and not utilities_bhs.is_near_critical(eps):
        return beta, eps, escaped
    with mp.workdps(_MP_DPS):
        beta_mp = mp.mpf(b) / mp.mpf(M)
        eps_mp = beta_mp * beta_mp - 27
        if escaped and eps_mp > 0:
            return float(beta_mp), float(eps_mp), True
        if captured and eps_mp < 0:
            return float(beta_mp), float(eps_mp), False
        extra = mp.mpf(beta_c) * mp.mpf(beta_c) - 27
        dbeta_mp = beta_mp - mp.mpf(beta_c)
        eps = float(extra + (2 * mp.mpf(beta_c) + dbeta_mp) * dbeta_mp)
        beta = float(mp.mpf(beta_c) + dbeta_mp)
    return beta, eps, escaped


def _signed_eps(b, M):
    """Scale-free eps = ((b-b_crit)/M) * ((b+b_crit)/M) when needed."""
    return _dimensionless_state(b, M)[1]


def _beta_offset(b, M):
    beta, eps, _escaped = _dimensionless_state(b, M)
    return beta, eps


def _is_near_critical(b, M):
    """Single production gate: |eps| from the supplied-float ratio."""
    b = float(b)
    M = float(M)
    if M == 0.0 or not math.isfinite(b) or not math.isfinite(M):
        return False
    eps = _beta_sq_minus_27(b / M)
    return utilities_bhs.is_near_critical(eps)


@lru_cache(maxsize=256)
def _mp_periapsis(b, M):
    """Public float periapsis via utilities_bhs; not used to build x_hi."""
    return utilities_bhs.periapsis_over_M(b, M, dps=_MP_DPS) * float(M)


@lru_cache(maxsize=256)
def _mp_phi_to_endpoint(b, M):
    """Inbound φ with periapsis kept as mpf inside utilities_bhs."""
    captured = is_captured(b, M)
    return utilities_bhs.phi_to_endpoint_over_M(
        b, M, captured=captured,
        horizon_over_M=event_horizon(M) / float(M),
        dps=_MP_DPS,
    )


@lru_cache(maxsize=512)
def _phi_segment_cached(x_lo, x_hi, b, M, turning):
    return utilities_bhs.phi_segment(
        x_lo, x_hi, b, M, dps=_MP_DPS, turning=turning,
    )


def _R_hat(x, beta, eps=None):
    """Dimensionless first integral: x = M u, beta = b/M."""
    if eps is None:
        eps = _beta_sq_minus_27(beta)
    delta = -eps / (27.0 * beta * beta)
    xm = x - (1.0 / 3.0)
    return delta + 2.0 * xm * xm * (x + (1.0 / 6.0))


def _phi_quad_segment(x_lo, x_hi, beta, eps=None, b=None, M=None):
    """One SciPy segment of ∫ dx/sqrt(R_hat), with a t² sub at a turning point."""
    t_max = math.sqrt(max(x_hi - x_lo, 0.0))
    if t_max == 0.0:
        return 0.0
    x_hi_f = float(x_hi)
    xm = x_hi_f - 1.0 / 3.0
    dR = 4.0 * xm * (x_hi_f + 1.0 / 6.0) + 2.0 * xm * xm

    def g(t):
        if t <= 0.0:
            rad0 = float(_R_hat(x_hi_f, beta, eps))
            if rad0 > 1.0e-18:
                return 0.0
            if abs(dR) > 0.0:
                return 2.0 / math.sqrt(abs(dR))
            return 0.0
        x = x_hi_f - t * t
        rad = float(_R_hat(x, beta, eps))
        if rad <= 0.0:
            return 0.0
        return (2.0 * t) / math.sqrt(rad)

    import warnings
    from scipy.integrate import IntegrationWarning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", IntegrationWarning)
        val, err = _scipy_quad(g, 0.0, t_max, epsabs=1.0e-10, limit=500)
    warned = any(issubclass(w.category, IntegrationWarning) for w in caught)
    rel_ok = (
        err is None
        or abs(val) < 1.0e-30
        or float(err) <= 1.0e-6 * max(1.0, abs(val))
    )
    if (
        not math.isfinite(val)
        or (warned and not rel_ok)
        or (err is not None and float(err) > 1.0e-5)
    ):
        if b is None or M is None:
            raise RuntimeError("near-critical quadrature fallback requires b and M")
        turning = abs(float(_R_hat(x_hi_f, beta, eps))) < 1.0e-14
        return _phi_segment_cached(
            float(x_lo), float(x_hi), float(b), float(M), turning,
        )
    return float(val)


def _phi_quad(x_lo, x_hi, beta, eps=None, b=None, M=None):
    """Adaptive integral, split at the photon-sphere coordinate x=1/3."""
    cuts = [x_lo]
    third = 1.0 / 3.0
    if x_lo < third < x_hi:
        cuts.append(third)
    cuts.append(x_hi)
    total = 0.0
    for a, bseg in zip(cuts[:-1], cuts[1:]):
        if bseg > a:
            total += _phi_quad_segment(a, bseg, beta, eps, b=b, M=M)
    return total


def _phi_integral(u_lo, u_hi, b, M, n=None):
    """∫ du / sqrt(R) computed in scale-free (x, beta) variables."""
    M = float(M)
    beta, eps, _escaped = _dimensionless_state(b, M)
    x_lo = M * float(u_lo)
    x_hi = M * float(u_hi)
    if x_hi < x_lo:
        x_lo, x_hi = x_hi, x_lo
    if x_hi <= x_lo:
        return 0.0
    R_hi = float(_R_hat(x_hi, beta, eps))
    delta = abs(beta - 3.0 * math.sqrt(3.0))
    if n is None and (delta < 1.0e-3 or abs(eps) < 1.0e-8):
        return _phi_quad(x_lo, x_hi, beta, eps, b=b, M=M)
    if n is None:
        n = 8192 if R_hi > 1.0e-8 else 4096
    n = max(int(n), 256)
    use_plain = R_hi > 1.0e-12
    if use_plain:
        x = np.linspace(x_lo, x_hi, n)
        rad = np.maximum(_R_hat(x, beta, eps), 0.0)
        inv = np.zeros_like(x)
        safe = rad > 0.0
        inv[safe] = 1.0 / np.sqrt(rad[safe])
        return _trapz(inv, x)
    span = x_hi - x_lo
    t = np.linspace(0.0, math.sqrt(max(span, 0.0)), n)
    x = x_hi - t * t
    rad = np.maximum(_R_hat(x, beta, eps), 0.0)
    piece = np.zeros_like(t)
    safe = rad > 0.0
    piece[safe] = (2.0 * t[safe]) / np.sqrt(rad[safe])
    if R_hi <= 1.0e-14:
        dR = -2.0 * x_hi + 6.0 * x_hi * x_hi
        if abs(dR) > 0.0:
            piece[0] = 2.0 / math.sqrt(abs(dR))
    return _trapz(piece, t)


def phi_from_infinity_inbound(u_target, b, M, n=None):
    """Orbital angle from r=∞ down to r=1/u_target, inbound, no turning."""
    u_target = float(u_target)
    if u_target <= 0.0:
        return 0.0
    if float(b) == critical_impact_parameter(M):
        return critical_phi_of_x(float(M) * u_target)
    return _phi_integral(0.0, u_target, b, M, n=n)


def face_on_crossing_angle(m):
    """Face-on equatorial crossing m=1,2,3,... occurs at this orbital angle.

    Observer at infinity on +z, thin disk in z=0.  The first crossing is
    at φ=π/2 from the incoming asymptote; later crossings add π.
    """
    m = int(m)
    if m < 1:
        raise ValueError("crossing index m starts at 1")
    return 0.5 * math.pi + (m - 1) * math.pi


def _max_inbound_u(b, M):
    if is_captured(b, M):
        return 1.0 / (event_horizon(M) * 1.0000001)
    return 1.0 / periapsis(b, M)


def _phi_to_turning_or_horizon(b, M):
    if float(b) == critical_impact_parameter(M):
        return float("inf")
    if _is_near_critical(b, M):
        return _mp_phi_to_endpoint(b, M)
    return phi_from_infinity_inbound(_max_inbound_u(b, M), b, M)


def face_on_crossing_radii(b, M, max_m=4):
    """r_m(b) for the first ``max_m`` face-on equatorial crossings.

    Returns a list of length ``max_m`` with None where that crossing
    does not occur.  Captured rays (b <= b_crit) only travel inbound;
    they still contribute any crossings they make before the horizon.
    Escaping rays travel inbound to periapsis and back out.
    """
    b = _require_finite("b", b)
    M = _require_positive("M", M)
    if b < 0.0:
        raise ValueError("b must be nonnegative.")
    if b == 0.0:
        # Radial ray: hits the disk at the origin of the image plane
        # only in the sense of a polar plunge; no finite-r crossing
        # of z=0 at φ=π/2 in this construction.
        return [None] * max_m
    b_crit = critical_impact_parameter(M)
    if b == b_crit:
        return _critical_crossing_radii(M, max_m)
    captured = b < b_crit
    u_end = _max_inbound_u(b, M)
    phi_in = _phi_to_turning_or_horizon(b, M)
    phi_total = phi_in if captured else 2.0 * phi_in
    radii = []
    for m in range(1, max_m + 1):
        target = face_on_crossing_angle(m)
        if target > phi_total + 1.0e-9:
            radii.append(None)
            continue
        if target <= phi_in + 1.0e-12:
            u = _invert_phi_inbound(target, b, M, u_end)
        else:
            remaining = target - phi_in
            u = _invert_phi_inbound(phi_in - remaining, b, M, u_end)
        if u is None or u <= 0.0:
            radii.append(None)
        else:
            radii.append(1.0 / u)
    return radii


def _invert_phi_inbound(target_phi, b, M, u_end):
    """Find u in (0, u_end] with inbound φ(u) = target_phi."""
    if target_phi <= 0.0:
        return 0.0
    lo, hi = 0.0, float(u_end)
    phi_hi = _phi_to_turning_or_horizon(b, M)
    if target_phi > phi_hi + 1.0e-12:
        return None
    for _ in range(56):
        mid = 0.5 * (lo + hi)
        phi_mid = phi_from_infinity_inbound(mid, b, M)
        if phi_mid < target_phi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_ATANH_ONE_OVER_SQRT3 = math.atanh(1.0 / math.sqrt(3.0))


def critical_phi_of_x(x):
    """Analytic inbound φ(x) on the exact critical geodesic.  x=M/r."""
    inside = 2.0 * float(x) + 1.0 / 3.0
    if inside <= 0.0:
        return 0.0
    if inside >= 1.0:
        return float("inf")
    return 2.0 * (math.atanh(math.sqrt(inside)) - _ATANH_ONE_OVER_SQRT3)


def critical_x_of_phi(phi):
    """Invert critical_phi_of_x.  Returns x=M/r in (0, 1/3)."""
    if phi <= 0.0:
        return 0.0
    a = math.tanh(0.5 * float(phi) + _ATANH_ONE_OVER_SQRT3)
    if a == 1.0:
        return None
    x = 0.5 * (a * a - 1.0 / 3.0)
    if x <= 0.0 or x >= 1.0 / 3.0:
        return None
    return x


def _critical_crossing_radii(M, max_m):
    """Finite-angle crossings of the exact critical geodesic.

    Closed form from R(x)=2(x-1/3)^2(x+1/6) on beta=3√3:
    φ(x)=2[atanh(√(2x+1/3))−atanh(1/√3)].
    """
    M = float(M)
    radii = []
    prev = None
    for m in range(1, max_m + 1):
        x = critical_x_of_phi(face_on_crossing_angle(m))
        if x is None or x <= 0.0 or x >= 1.0 / 3.0:
            radii.append(None)
            continue
        r = M / x
        if prev is not None and not (r < prev):
            radii.append(None)
            continue
        radii.append(r)
        prev = r
    return radii


def crossing_count(b, M, max_m=4):
    """How many face-on equatorial crossings exist at this b."""
    return sum(1 for r in face_on_crossing_radii(b, M, max_m=max_m) if r is not None)


def _require_table_width(width, M):
    """Table/image sampling cannot resolve Gaussians narrower than this."""
    width = _require_positive("width", width)
    floor = HOT_RING_WIDTH_MIN * M
    if width < floor:
        raise ValueError(
            f"width must be at least {HOT_RING_WIDTH_MIN:g} M "
            "for the source-aware intensity table."
        )
    return width


def emitted_intensity(r, M, r_hot=None, width=None):
    """Optically thin face-on annulus, I_em(r).  Zero inside the horizon."""
    M = _require_positive("M", M)
    r = _require_finite("r", r)
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
    width = _require_positive("width", width)
    if r <= event_horizon(M):
        return 0.0
    return math.exp(-0.5 * ((r - r_hot) / width) ** 2)


def gravitational_g(r, M):
    """sqrt(1-2M/r) for a static emitter at radius r."""
    r = _require_positive("r", r)
    M = _require_positive("M", M)
    arg = 1.0 - 2.0 * M / r
    if arg <= 0.0:
        return 0.0
    return math.sqrt(arg)


def observed_components(b, M, r_hot=None, width=None, max_m=4):
    """Per-crossing bolometric I_obs pieces: g^4 I_em(r_m).

    Returns (parts, radii) where parts[0] is the direct image, parts[1]
    the lensing-ring crossing, and parts[2:] photon-ring / subring
    crossings.  Captured rays are allowed to contribute before the
    horizon; they are not forced dark.
    """
    radii = face_on_crossing_radii(b, M, max_m=max_m)
    parts = []
    for r in radii:
        if r is None:
            parts.append(0.0)
        else:
            g = gravitational_g(r, M)
            parts.append((g ** 4) * emitted_intensity(r, M, r_hot, width))
    return parts, radii


def observed_intensity(b, M, r_hot=None, width=None, max_m=4, upto=None):
    parts, _ = observed_components(b, M, r_hot=r_hot, width=width, max_m=max_m)
    if upto is None:
        return float(sum(parts))
    return float(sum(parts[:int(upto)]))


def adaptive_impact_samples(M, b_max, n_outer=48, n_near=72):
    """b samples clustered logarithmically around b_crit on both sides."""
    M = _require_positive("M", M)
    b_max = _require_positive("b_max", b_max)
    b_crit = critical_impact_parameter(M)
    outer = np.linspace(0.02 * M, b_max, int(n_outer))
    # Offsets from 1e-6 M to 0.3 M.  The m=3 window is ~0.03 M wide.
    log_off = np.logspace(-6.0, math.log10(0.30), int(n_near)) * M
    near = np.concatenate([
        b_crit - log_off[::-1],
        b_crit + log_off,
    ])
    near = near[(near > 0.0) & (near <= b_max * 1.001)]
    bs = np.unique(np.concatenate([outer, near]))
    return np.sort(bs)


def transfer_curves(M, b_max_over_M=8.0, max_m=3):
    """b and r_m/M tables for the transfer-function beat."""
    M = _require_positive("M", M)
    b_crit = critical_impact_parameter(M)
    b_max = max(float(b_max_over_M) * M, 1.2 * b_crit)
    bs = adaptive_impact_samples(M, b_max)
    table = np.full((bs.size, max_m), np.nan)
    counts = np.zeros(bs.size, dtype=int)
    for i, bv in enumerate(bs):
        radii = face_on_crossing_radii(float(bv), M, max_m=max_m)
        counts[i] = sum(r is not None for r in radii)
        for m, r in enumerate(radii):
            if r is not None:
                table[i, m] = r / M
    return bs, table, counts, b_crit


def source_crossing_impacts(M, r_hot, max_m=4):
    """Bracketed roots of r_m(b) = r_hot, one per existing branch.

    Each transfer branch is scanned from below b_crit out to a large b.
    If r_m jumps from a finite value below r_hot to None, the solver
    hunts the branch endpoint (where r_m -> infinity) rather than
    declaring the root missing.
    """
    M = _require_positive("M", M)
    r_hot = _require_positive("r_hot", r_hot)
    b_crit = critical_impact_parameter(M)
    b_max = max(2.0 * r_hot, 3.0 * b_crit)
    inner = np.linspace(0.05 * M, 0.99 * b_crit, 40)
    outer_hi = math.log10(max(b_max / b_crit - 1.0, 1.0e-3))
    near_lo = b_crit * (1.0 - np.logspace(-12.0, -2.3, 28))
    near_hi = b_crit * (1.0 + np.logspace(-12.0, outer_hi, 48))
    bs = np.unique(np.concatenate([inner, near_lo, near_hi]))
    bs = bs[bs > 0.0]
    found = [None] * max_m
    prev_r = [None] * max_m
    prev_b = None
    for bv in bs:
        radii = face_on_crossing_radii(float(bv), M, max_m=max_m)
        for m, r in enumerate(radii):
            if found[m] is not None:
                continue
            if r is not None and prev_r[m] is not None and prev_b is not None:
                if (prev_r[m] - r_hot) * (r - r_hot) <= 0.0:
                    found[m] = _bisect_source_root(
                        prev_b, float(bv), M, m + 1, r_hot,
                    )
            elif r is None and prev_r[m] is not None and prev_b is not None:
                if prev_r[m] < r_hot:
                    found[m] = _hunt_root_to_endpoint(
                        prev_b, float(bv), M, m + 1, r_hot,
                    )
            prev_r[m] = r
        prev_b = float(bv)
    return found


def _hunt_root_to_endpoint(b_lo, b_hi, M, m, r_hot):
    """b_lo has finite r_m < r_hot; b_hi has no crossing.  r_m -> inf at the edge."""
    lo, hi = float(b_lo), float(b_hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = face_on_crossing_radii(mid, M, max_m=m)[m - 1]
        if r is None:
            hi = mid
        elif r >= r_hot:
            return _bisect_source_root(b_lo, mid, M, m, r_hot)
        else:
            lo = mid
    r_lo = face_on_crossing_radii(lo, M, max_m=m)[m - 1]
    if r_lo is not None and r_lo >= r_hot:
        return _bisect_source_root(b_lo, lo, M, m, r_hot)
    return None


def _bisect_source_root(b_lo, b_hi, M, m, r_hot):
    """Refine b in [b_lo, b_hi] until r_m(b) = r_hot."""
    def residual(bv):
        radii = face_on_crossing_radii(bv, M, max_m=m)
        r = radii[m - 1]
        if r is None:
            return None
        return r - r_hot

    flo = residual(b_lo)
    fhi = residual(b_hi)
    if flo is None or fhi is None or flo * fhi > 0.0:
        return None
    lo, hi, f_lo = b_lo, b_hi, flo
    mid = 0.5 * (lo + hi)
    fm = None
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = residual(mid)
        if fm is None:
            return None
        if abs(fm) <= 1.0e-8 * max(M, abs(r_hot)):
            return mid
        if f_lo * fm <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, fm
    if fm is not None and abs(fm) <= 1.0e-6 * max(M, abs(r_hot)):
        return mid
    return None


def source_aware_impact_samples(M, b_max, r_hot, width, max_m=4, peaks=None):
    """b nodes clustered at b_crit and at every source-image root.

    Returns the sample array only.  Pass ``peaks`` to reuse a root list
    already computed by source_crossing_impacts.
    """
    M = _require_positive("M", M)
    width = _require_table_width(width, M)
    chunks = [adaptive_impact_samples(M, b_max)]
    if peaks is None:
        peaks = source_crossing_impacts(M, r_hot, max_m=max_m)
    roots = [
        float(bp) for bp in peaks
        if bp is not None and bp > 0.0 and bp <= b_max * 1.05
    ]
    if roots:
        chunks.append(np.asarray(roots, dtype=float))
        span = max((width / M) if width else 1.0e-12, 1.0e-12)
        lo_off = min(1.0e-8, max(0.05 * span, 1.0e-12))
        log_off = np.logspace(math.log10(lo_off), math.log10(span), 28) * M
        near = np.concatenate(
            [np.concatenate([bp - log_off[::-1], bp + log_off]) for bp in roots]
        )
        near = near[(near > 0.0) & (near <= b_max * 1.05)]
        chunks.append(near)
    return np.sort(np.unique(np.concatenate(chunks)))


def intensity_table(M, b_max, r_hot=None, width=None, max_m=None, peaks=None):
    """Source-aware (b, I_m) table.  Returns (sample, parts)."""
    M = _require_positive("M", M)
    if max_m is None:
        max_m = MAX_IMAGE_M
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
    width = _require_table_width(width, M)
    if peaks is None:
        peaks = source_crossing_impacts(M, r_hot, max_m=max_m)
    sample = source_aware_impact_samples(
        M, b_max, r_hot, width, max_m=max_m, peaks=peaks,
    )
    parts = np.zeros((sample.size, max_m))
    for i, bv in enumerate(sample):
        comp, _ = observed_components(
            float(bv), M, r_hot=r_hot, width=width, max_m=max_m,
        )
        parts[i, :] = comp
    return sample, parts


def disk_image_components(bx, by, M, r_hot=None, width=None, max_m=None):
    """Return (I_direct, I_upto2, I_total, sample, parts, peaks).

    Each camera pixel is a point sample of I_obs at the pixel-centre
    impact parameter, interpolated from a source-aware radial table.
    Features narrower than the raster may miss every centre; the
    radial I(b) table is the record of those peaks.
    """
    M = _require_positive("M", M)
    if max_m is None:
        max_m = MAX_IMAGE_M
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
    width = _require_table_width(width, M)
    r_hot = _require_positive("r_hot", r_hot)
    if r_hot <= photon_sphere(M):
        raise ValueError(
            f"r_hot must lie outside the photon sphere ({photon_sphere(M):g})."
        )
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    b = np.hypot(bx, by)
    b_pix = float(np.max(b)) if b.size else 0.0
    b_max = max(b_pix, 1.25 * r_hot, 2.0 * critical_impact_parameter(M))
    peaks = source_crossing_impacts(M, r_hot, max_m=max_m)
    sample, parts = intensity_table(
        M, b_max, r_hot=r_hot, width=width, max_m=max_m, peaks=peaks,
    )
    cols = [
        np.interp(b, sample, parts[:, m], left=0.0, right=0.0)
        for m in range(max_m)
    ]
    img1 = cols[0]
    img2 = cols[0] + (cols[1] if max_m > 1 else 0.0)
    img_all = sum(cols)
    return img1, img2, img_all, sample, parts, peaks


def disk_image(bx, by, M, r_hot=None, width=None, max_m=None, upto=None):
    """Face-on optically thin image.  Wrapper around disk_image_components."""
    if max_m is None:
        max_m = MAX_IMAGE_M
    img1, img2, img_all, sample, parts, peaks = disk_image_components(
        bx, by, M, r_hot=r_hot, width=width, max_m=max_m,
    )
    marker = peaks[0] if peaks and peaks[0] is not None else float("nan")
    if upto == 1:
        return img1, marker
    if upto == 2:
        return img2, marker
    return img_all, marker


def odd_n_pix(n_pix):
    n_pix = int(n_pix)
    if n_pix < 9:
        raise ValueError("n_pix must be an integer >= 9")
    if n_pix % 2 == 0:
        n_pix += 1
    if n_pix > N_PIX_AUTO_MAX:
        raise ValueError(
            f"n_pix={n_pix} exceeds the automatic cap {N_PIX_AUTO_MAX}.  "
            "This is a teaching grid, not an EHT pipeline."
        )
    return n_pix


def patch_help_version(html_path):
    """Write MODEL_VERSION and BUILD_ID into the Help #version_build element."""
    import re
    path = os.fspath(html_path)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    pattern = r'(id="version_build"[^>]*>)(.*?)(</p>)'
    replacement = (
        rf'\1\n    Version {MODEL_VERSION}&nbsp;&nbsp;&nbsp;&nbsp;'
        rf'Build {BUILD_ID}\n  \3'
    )
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if n != 1:
        raise ValueError("could not find #version_build in Help file")
    new = re.sub(
        r"(\\sum_\{m=1\}\^\{)\d+(\})",
        rf"\g<1>{MAX_IMAGE_M}\2",
        new,
    )
    new = re.sub(
        r"Narrow \\\([^\\)]*\\\) peaks",
        "Narrow \\\\(" + photon_order_label() + "\\\\) peaks",
        new,
    )
    new = re.sub(
        r"(\\\(m\\ge )\d+(\\\) is omitted)",
        rf"\g<1>{MAX_IMAGE_M + 1}\2",
        new,
    )
    if new != text:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(new)
    return MODEL_VERSION, BUILD_ID
