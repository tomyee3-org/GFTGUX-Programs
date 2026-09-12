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

import numpy as np

MODEL_VERSION = "0.4.0"

# Highest crossing index computed for the toy image (m = 1..MAX_IMAGE_M).
# m>=5 is omitted; the figure must say so.
MAX_IMAGE_M = 4

# RK4 null-geodesic stepper is kept in lockstep with PhotonOrbit 1.4.0
# (GFTGUX-Programs/PhotonOrbit).  This package does not import that
# program; the copy exists so a BlackHoleShadow zip runs standalone.
PHOTONORBIT_SYNC_VERSION = "1.4.0"

BUILD_ID_COVERS = (
    "physics_bhs.py",
    "driver_bhs.py",
    "main.py",
    "plot_bhs.py",
)


def compute_build_id_from_directory(directory):
    """Hash the four covered files found in ``directory``.

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

# Capture/ring maps must contain b_crit with this half-width margin.
FOV_CONTAIN_MARGIN = 1.25

# compare-mode log10(M/M_sun) teaching range.
LOGM_MIN = 6.0
LOGM_MAX = 15.0

# Azimuth threshold that counts as "wound once" on an escaping ray.
WINDING_DELTA_PHI = 2.0 * math.pi

# Schematic backlight Gaussian half-width as a multiple of M.
HOT_RING_WIDTH_DEFAULT = 0.45

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
    r_ph = photon_sphere(M)
    r = max(float(b), r_ph + 0.25 * M)
    for _ in range(80):
        f = r * r * r - b * b * r + 2.0 * M * b * b
        df = 3.0 * r * r - b * b
        if df == 0.0:
            r = r_ph + 0.5 * M + 0.5 * float(b)
            continue
        r_new = r - f / df
        if r_new <= r_ph:
            r_new = 0.5 * (r + r_ph) + 0.05 * M
        if abs(r_new - r) <= 1.0e-14 * max(1.0, abs(r)):
            r = r_new
            break
        r = r_new
    if r <= r_ph or not math.isfinite(r):
        raise RuntimeError("periapsis solver failed for the given b, M.")
    return r


def asymptotic_deflection(b, M, n_u=800):
    """Asymptotic scattering deflection hat{alpha} = Delta phi_inf - pi.

    Uses the standard substitution u = 1/r:

        Delta phi_inf = 2 * integral_0^{u_min} b / sqrt(1 - b^2 u^2 + 2 M b^2 u^3) du

    Captured rays (b <= b_crit) have no scattering deflection; this
    function raises ValueError for them.  The integrand is integrable at
    the turning point.  Near b_crit the result grows without bound.  That strong-deflection
    divergence permits higher-order images; it is not itself a photon ring.
    """
    b = _require_positive("b", b)
    M = _require_positive("M", M)
    if is_captured(b, M):
        raise ValueError(
            "asymptotic_deflection is defined only for escaping rays "
            f"(b > b_crit = {critical_impact_parameter(M):g})."
        )
    r_min = periapsis(b, M)
    u_min = 1.0 / r_min
    # u = 1/r.  Write u = u_min - t^2 so the turning-point square root
    # cancels and the outer limit u = 0 is t = sqrt(u_min).
    n_u = max(int(n_u), 256)
    t = np.linspace(0.0, math.sqrt(u_min), n_u)
    u = np.clip(u_min - t * t, 0.0, u_min)
    rad = 1.0 - (b * b) * u * u + 2.0 * M * (b * b) * u * u * u
    rad = np.maximum(rad, 0.0)
    piece = np.zeros_like(t)
    safe = rad > 0.0
    piece[safe] = (2.0 * t[safe]) / np.sqrt(rad[safe])
    drad_du = -2.0 * b * b * u_min + 6.0 * M * b * b * u_min * u_min
    slope = abs(float(drad_du))
    if slope > 0.0:
        piece[0] = 2.0 / math.sqrt(slope)
    delta_phi_inf = 2.0 * b * float(np.trapezoid(piece, t))
    return delta_phi_inf - math.pi


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
            "visible.  Use a smaller --fov or a larger --n_pix."
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
    # Walk inward from b_from_r until Delta phi exceeds the threshold.
    b_hi = b_crit
    samples = np.geomspace(b_crit * (1.0 + 1.0e-8), b_from_r, 64)
    for bv in samples[::-1]:
        try:
            delta_phi = asymptotic_deflection(float(bv), M) + math.pi
        except ValueError:
            continue
        if delta_phi > delta_phi_min:
            b_hi = float(bv)
            break
    b_lo = b_crit
    if b_hi <= b_lo:
        return b_lo, b_lo
    return b_lo, b_hi


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


def _phi_integral(u_lo, u_hi, b, M, n=512):
    """∫_{u_lo}^{u_hi} du / sqrt(R(u)), R>=0 on the open interval."""
    u_lo = float(u_lo)
    u_hi = float(u_hi)
    if u_hi < u_lo:
        u_lo, u_hi = u_hi, u_lo
    if u_hi <= u_lo:
        return 0.0
    # t^2 substitution from the upper end in case R(u_hi)~0 (turning point).
    span = u_hi - u_lo
    t = np.linspace(0.0, math.sqrt(span), int(n))
    u = u_hi - t * t
    rad = _R_of_u(u, b, M)
    rad = np.maximum(rad, 0.0)
    piece = np.zeros_like(t)
    safe = rad > 0.0
    piece[safe] = (2.0 * t[safe]) / np.sqrt(rad[safe])
    # Endpoint slope of R at a turning point.
    if rad[0] <= 1.0e-18:
        dR = -2.0 * u_hi + 6.0 * M * u_hi * u_hi
        if abs(dR) > 0.0:
            piece[0] = 2.0 / math.sqrt(abs(dR))
    return float(np.trapezoid(piece, t))


def phi_from_infinity_inbound(u_target, b, M):
    """Orbital angle from r=∞ down to r=1/u_target, inbound, no turning."""
    u_target = float(u_target)
    if u_target <= 0.0:
        return 0.0
    return _phi_integral(0.0, u_target, b, M)


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
    r_ph = photon_sphere(M)
    if abs(b - b_crit) <= 1.0e-10 * M:
        # Unstable circular orbit: azimuth accumulates without bound
        # while r approaches 3M from above.  It does not plunge.
        return [r_ph * (1.0 + 1.0e-4 / float(m)) for m in range(1, max_m + 1)]
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
    phi_hi = phi_from_infinity_inbound(hi, b, M)
    if target_phi > phi_hi:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        phi_mid = phi_from_infinity_inbound(mid, b, M)
        if phi_mid < target_phi:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crossing_count(b, M, max_m=4):
    """How many face-on equatorial crossings exist at this b."""
    return sum(1 for r in face_on_crossing_radii(b, M, max_m=max_m) if r is not None)


def emitted_intensity(r, M, r_hot=None, width=None):
    """Optically thin face-on annulus, I_em(r).  Zero inside the horizon."""
    M = _require_positive("M", M)
    r = _require_finite("r", r)
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
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


def transfer_curves(M, b_max_over_M=8.0, n=81, max_m=3):
    """b and r_m/M tables for the transfer-function beat.

    ``n`` is accepted for compatibility; the actual grid is adaptive
    around b_crit so the m=3 branch is not a function of --fov luck.
    """
    del n
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
    """b values where r_m(b) = r_hot, one per existing branch."""
    M = _require_positive("M", M)
    r_hot = _require_positive("r_hot", r_hot)
    b_crit = critical_impact_parameter(M)
    b_max = max(1.6 * r_hot, 2.0 * b_crit)
    bs = adaptive_impact_samples(M, b_max, n_outer=60, n_near=96)
    found = [None] * max_m
    prev = [None] * max_m
    prev_b = None
    for bv in bs:
        radii = face_on_crossing_radii(float(bv), M, max_m=max_m)
        for m, r in enumerate(radii):
            if r is None or prev[m] is None or prev_b is None:
                prev[m] = r
                continue
            if (prev[m] - r_hot) * (r - r_hot) <= 0.0:
                # Linear interpolate in b.
                denom = (r - prev[m])
                if denom == 0.0:
                    found[m] = float(bv)
                else:
                    frac = (r_hot - prev[m]) / denom
                    found[m] = float(prev_b + frac * (bv - prev_b))
            prev[m] = r
        prev_b = float(bv)
    return found


def intensity_table(M, b_max, r_hot=None, width=None, max_m=None):
    """Adaptive (b, I_m) table.  Components computed once."""
    M = _require_positive("M", M)
    if max_m is None:
        max_m = MAX_IMAGE_M
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
    sample = adaptive_impact_samples(M, b_max)
    parts = np.zeros((sample.size, max_m))
    for i, bv in enumerate(sample):
        comp, _ = observed_components(
            float(bv), M, r_hot=r_hot, width=width, max_m=max_m,
        )
        parts[i, :] = comp
    return sample, parts


def disk_image_components(bx, by, M, r_hot=None, width=None, max_m=None):
    """Return (I_direct, I_upto2, I_total, sample, parts, b_hot).

    Pixel values are interpolated from an adaptive radial table, then
    each pixel takes the max of that interpolation and any source-peak
    sample that falls inside the pixel's radial half-width.  Narrow
    m>=3 features therefore remain visible instead of being averaged
    off the raster.
    """
    M = _require_positive("M", M)
    if max_m is None:
        max_m = MAX_IMAGE_M
    if r_hot is None:
        r_hot = HOT_RING_R_DEFAULT * M
    if width is None:
        width = HOT_RING_WIDTH_DEFAULT * M
    r_hot = _require_positive("r_hot", r_hot)
    if r_hot <= photon_sphere(M):
        raise ValueError(
            f"r_hot must lie outside the photon sphere ({photon_sphere(M):g})."
        )
    bx = np.asarray(bx, dtype=float)
    by = np.asarray(by, dtype=float)
    b = np.hypot(bx, by)
    b_max = float(np.max(b)) if b.size else 8.0 * M
    sample, parts = intensity_table(
        M, b_max, r_hot=r_hot, width=width, max_m=max_m,
    )
    cols = [
        np.interp(b, sample, parts[:, m], left=0.0, right=0.0)
        for m in range(max_m)
    ]
    # Pixel radial half-width from the grid, if it is a regular mesh.
    if bx.ndim == 2 and bx.shape[1] > 1:
        db = abs(float(bx[0, 1] - bx[0, 0]))
    else:
        db = 0.05 * M
    peaks = source_crossing_impacts(M, r_hot, max_m=max_m)
    for m, b_peak in enumerate(peaks):
        if b_peak is None:
            continue
        I_peak = observed_intensity(
            b_peak, M, r_hot=r_hot, width=width, max_m=max_m, upto=m + 1,
        ) - observed_intensity(
            b_peak, M, r_hot=r_hot, width=width, max_m=max_m, upto=m,
        )
        near = np.abs(b - b_peak) <= 0.55 * db
        cols[m] = np.where(near, np.maximum(cols[m], I_peak), cols[m])
    img1 = cols[0]
    img2 = cols[0] + (cols[1] if max_m > 1 else 0.0)
    img_all = sum(cols)
    try:
        b_hot = impact_parameter_of_periapsis(r_hot, M)
    except ValueError:
        b_hot = float("nan")
    return img1, img2, img_all, sample, parts, b_hot


def disk_image(bx, by, M, r_hot=None, width=None, max_m=None, upto=None):
    """Face-on optically thin image.  Wrapper around disk_image_components."""
    if max_m is None:
        max_m = MAX_IMAGE_M
    img1, img2, img_all, sample, parts, b_hot = disk_image_components(
        bx, by, M, r_hot=r_hot, width=width, max_m=max_m,
    )
    if upto == 1:
        return img1, b_hot
    if upto == 2:
        return img2, b_hot
    return img_all, b_hot


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
    text = open(path, encoding="utf-8").read()
    pattern = r'(id="version_build"[^>]*>)(.*?)(</p>)'
    replacement = (
        rf'\1\n    Version {MODEL_VERSION}&nbsp;&nbsp;&nbsp;&nbsp;'
        rf'Build {BUILD_ID}\n  \3'
    )
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if n != 1:
        raise ValueError("could not find #version_build in Help file")
    if new != text:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(new)
    return MODEL_VERSION, BUILD_ID
