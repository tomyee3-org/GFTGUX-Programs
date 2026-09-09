"""
physics_gl.py
=============
Thin-lens gravitational-lensing engine for the GravitationalLensing
teaching program.

Two mass models share this module:

  point   a point-mass lens (Einstein ring; caustic degenerates to a point)
  shear   a singular isothermal sphere plus external shear (diamond caustic)

Angular calculations are in radians internally.  Command-line source
positions and plot axes are in arcseconds.  Distances that enter the Einstein radius are
SI metres.  Nothing here integrates a geodesic: the thin-lens deflection
is applied in one plane, which is the right teaching model for galaxy-scale
lensing and the wrong model for a black-hole photon orbit (see PhotonOrbit).

Every result is labelled in the Help file as derived from the thin-lens
equation or as a schematic ray cartoon.  No N-body mass distribution is
fitted to data.
"""

import math
import hashlib
import os

import numpy as np

MODEL_VERSION = "1.0.0"

BUILD_ID_COVERS = (
    "physics_gl.py",
    "driver_gl.py",
    "main.py",
    "plot_gl.py",
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
    """Return a short identifier derived from the four core source files."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return compute_build_id_from_directory(here)
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# ----------------------------------------------------------------------
# Physical constants (SI, CODATA 2022 / IAU)
# ----------------------------------------------------------------------
G = 6.674_30e-11
C_LIGHT = 2.997_924_58e8
M_SUN = 1.988_47e30
PC = 3.085_677_581e16
GPC = 1.0e9 * PC

# Default geometry: lens halfway to a source at 2 Gpc.
D_L_DEFAULT = 1.0 * GPC
D_S_DEFAULT = 2.0 * GPC

# Soft floor so a pixel sitting on the lens centre does not divide by zero.
_R_FLOOR_FRAC = 1.0e-6

# Beginner shear range: keep the diamond inside the SIS pseudo-caustic.
# At |gamma| = 1/3 the long-axis cusps touch |beta| = theta_E; above that
# they become naked cusps and a three-image region appears.
GAMMA_MAX = 1.0 / 3.0

# Cluster Newton roots only when they sit closer than this fraction of
# theta_E.  3% of theta_E was large enough to merge distinct images just
# inside a fold or cusp; solver accuracy is ~1e-10 rad, so 1e-4 theta_E
# still treats genuine duplicates as one image.
IMAGE_DEDUP_FRAC = 1.0e-4

# Fractional distance to a fold or cusp inside which Newton image counts
# are not a promised teaching output.  Failures appear around 1e-8.
CAUSTIC_COUNT_BUFFER = 1.0e-4

# FOV half-width = FOV_MARGIN * largest required radius.
FOV_MARGIN = 1.2

# Keep at least this many pixels across one compact-source sigma.
PIXELS_PER_SIGMA = 3.0

# Automatic n_pix cap.  Wider fields need an explicit --n_pix.
N_PIX_AUTO_MAX = 401

# Fewest samples across a lensed image-plane scale for the image to be
# treated as visible on a uniform grid.  This is a detectability floor,
# not a morphology-resolution guarantee.
IMAGE_DETECT_SAMPLES = 0.8

# Smallest Einstein radius, in grid intervals, accepted by kappa mode.
KAPPA_MIN_EINSTEIN_INTERVALS = 4.0


def _require_finite(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _require_positive(name, value):
    value = _require_finite(name, value)
    if value <= 0.0:
        raise ValueError(f"{name} must be greater than zero, got {value:g}")
    return value


def arcsec_to_rad(x):
    return np.deg2rad(np.asarray(x, dtype=float) / 3600.0)


def rad_to_arcsec(x):
    return np.rad2deg(np.asarray(x, dtype=float)) * 3600.0


def einstein_radius_point_mass(m_kg, d_l=D_L_DEFAULT, d_s=D_S_DEFAULT):
    """Einstein radius of a point-mass lens, in radians."""
    m_kg = _require_positive("m_kg", m_kg)
    d_l = _require_positive("d_l", d_l)
    d_s = _require_positive("d_s", d_s)
    if d_s <= d_l:
        raise ValueError("d_s must exceed d_l so that D_ls is positive")
    d_ls = d_s - d_l
    return math.sqrt((4.0 * G * m_kg / C_LIGHT**2) * (d_ls / (d_l * d_s)))


def einstein_radius_sis(sigma_v_m_s, d_l=D_L_DEFAULT, d_s=D_S_DEFAULT):
    """Einstein radius of an SIS lens, in radians.  sigma_v in m/s."""
    sigma_v_m_s = _require_positive("sigma_v_m_s", sigma_v_m_s)
    d_l = _require_positive("d_l", d_l)
    d_s = _require_positive("d_s", d_s)
    if d_s <= d_l:
        raise ValueError("d_s must exceed d_l so that D_ls is positive")
    d_ls = d_s - d_l
    return 4.0 * math.pi * (sigma_v_m_s / C_LIGHT)**2 * (d_ls / d_s)


def make_grid(n_pix=201, fov_arcsec=6.0):
    """Square image-plane grid in radians, odd n_pix so a pixel sits on axis."""
    n_pix = int(n_pix)
    if n_pix < 9:
        raise ValueError("n_pix must be at least 9")
    if n_pix % 2 == 0:
        n_pix += 1
    fov_arcsec = _require_positive("fov_arcsec", fov_arcsec)
    half = arcsec_to_rad(fov_arcsec) / 2.0
    axis = np.linspace(-half, half, n_pix)
    theta_x, theta_y = np.meshgrid(axis, axis)
    return theta_x, theta_y


def _r_floor(theta_x, theta_y):
    span = max(float(np.nanmax(np.abs(theta_x))),
               float(np.nanmax(np.abs(theta_y))),
               1.0e-16)
    return span * _R_FLOOR_FRAC


def _radius(theta_x, theta_y):
    r = np.hypot(theta_x, theta_y)
    return np.maximum(r, _r_floor(theta_x, theta_y))


def deflection_point_mass(theta_x, theta_y, theta_e):
    """alpha = theta_E^2 * theta / |theta|^2."""
    theta_e = _require_positive("theta_e", theta_e)
    r2 = _radius(theta_x, theta_y) ** 2
    return (theta_e**2) * theta_x / r2, (theta_e**2) * theta_y / r2


def jacobian_det_point_mass(theta_x, theta_y, theta_e):
    """det A = 1 - (theta_E / |theta|)^4 for a point-mass lens."""
    theta_e = _require_positive("theta_e", theta_e)
    r2 = _radius(theta_x, theta_y) ** 2
    return 1.0 - (theta_e**4 / r2**2)


def point_mass_radial_stretch(theta, theta_e):
    """|d beta / d theta| along a radial line for a point-mass lens."""
    theta = _require_finite("theta", theta)
    theta_e = _require_positive("theta_e", theta_e)
    if theta == 0.0:
        raise ValueError("theta must be nonzero")
    return abs(1.0 + (theta_e / theta) ** 2)


def point_mass_image_plane_scale(theta, theta_e, source_scale):
    """Image-plane width of a source-plane scale at image radius theta."""
    source_scale = _require_positive("source_scale", source_scale)
    return source_scale / point_mass_radial_stretch(theta, theta_e)


def point_mass_mapped_radii(beta, theta_e, source_radius):
    """Image radii of the farthest source-plane point |beta| + source_radius."""
    beta = _require_finite("beta", beta)
    source_radius = _require_positive("source_radius", source_radius)
    return point_mass_image_radii(abs(beta) + source_radius, theta_e)


def point_mass_image_radii(beta, theta_e):
    """Analytic image radii along the source direction.

    Returns (theta_plus, theta_minus) with theta_plus > 0 outside the
    Einstein ring and theta_minus < 0 on the opposite side.  For beta = 0
    both sit at +/- theta_E (the ring).
    """
    beta = _require_finite("beta", beta)
    theta_e = _require_positive("theta_e", theta_e)
    disc = math.sqrt(beta * beta + 4.0 * theta_e * theta_e)
    return 0.5 * (beta + disc), 0.5 * (beta - disc)


def map_point_mass(theta_x, theta_y, theta_e):
    ax, ay = deflection_point_mass(theta_x, theta_y, theta_e)
    return theta_x - ax, theta_y - ay


def deflection_sis_shear(theta_x, theta_y, theta_e, gamma):
    """SIS deflection of fixed length theta_E, plus external shear along x.

    alpha_sis = theta_E * theta_hat
    alpha_shear = (gamma x, -gamma y)
    """
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    if abs(gamma) >= GAMMA_MAX:
        raise ValueError(
            f"shear |gamma| must be < {GAMMA_MAX:g} so the diamond "
            f"stays inside the pseudo-caustic, got {gamma:g}"
        )
    r = _radius(theta_x, theta_y)
    ax = theta_e * theta_x / r + gamma * theta_x
    ay = theta_e * theta_y / r - gamma * theta_y
    return ax, ay


def map_sis_shear(theta_x, theta_y, theta_e, gamma):
    ax, ay = deflection_sis_shear(theta_x, theta_y, theta_e, gamma)
    return theta_x - ax, theta_y - ay


def convergence_sis(theta_x, theta_y, theta_e):
    """SIS convergence kappa = theta_E / (2 |theta|)."""
    theta_e = _require_positive("theta_e", theta_e)
    return 0.5 * theta_e / _radius(theta_x, theta_y)


def jacobian_det_sis_shear(theta_x, theta_y, theta_e, gamma):
    """det A for SIS plus external shear Gamma_1 = gamma, Gamma_2 = 0."""
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    r = _radius(theta_x, theta_y)
    phi = np.arctan2(theta_y, theta_x)
    kap = 0.5 * theta_e / r
    g1 = -kap * np.cos(2.0 * phi) + gamma
    g2 = -kap * np.sin(2.0 * phi)
    return (1.0 - kap)**2 - (g1**2 + g2**2)


def jacobian_matrix_sis_shear(theta_x, theta_y, theta_e, gamma):
    """2x2 Jacobian A = d beta / d theta at one image-plane point."""
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    x = float(theta_x)
    y = float(theta_y)
    r = max(math.hypot(x, y), 1.0e-16)
    phi = math.atan2(y, x)
    kap = 0.5 * theta_e / r
    g1 = -kap * math.cos(2.0 * phi) + gamma
    g2 = -kap * math.sin(2.0 * phi)
    return np.array([[1.0 - kap - g1, -g2],
                     [-g2, 1.0 - kap + g1]], dtype=float)


def critical_radius_sis_shear(phi, theta_e, gamma):
    """Analytic tangential critical radius for this shear convention.

    r(phi) = theta_E [1 - gamma cos(2 phi)] / (1 - gamma^2).
    Maximum is theta_E / (1 - |gamma|).
    """
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    if abs(gamma) >= GAMMA_MAX:
        raise ValueError(
            f"shear |gamma| must be < {GAMMA_MAX:g} so the diamond "
            f"stays inside the pseudo-caustic, got {gamma:g}"
        )
    phi = _require_finite("phi", phi)
    return theta_e * (1.0 - gamma * math.cos(2.0 * phi)) / (1.0 - gamma * gamma)


def critical_curve_sis_shear(theta_e, gamma, n_theta=361):
    """Closed tangential critical curve from the analytic radius."""
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    if abs(gamma) >= GAMMA_MAX:
        raise ValueError(
            f"shear |gamma| must be < {GAMMA_MAX:g} so the diamond "
            f"stays inside the pseudo-caustic, got {gamma:g}"
        )
    n_theta = int(n_theta)
    if n_theta < 16:
        raise ValueError("n_theta must be at least 16")
    phis = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=True)
    rs = np.array([critical_radius_sis_shear(float(phi), theta_e, gamma)
                   for phi in phis], dtype=float)
    return rs * np.cos(phis), rs * np.sin(phis)


def pseudo_caustic_radius_sis_shear(theta_e):
    """Radius of the circular pseudo-caustic (image of the SIS singularity)."""
    return float(_require_positive("theta_e", theta_e))


def is_einstein_ring_case(beta_x, beta_y, gamma, theta_e=1.0):
    """Centered source, no shear: the image is the Einstein ring, not dots."""
    beta_x = _require_finite("beta_x", beta_x)
    beta_y = _require_finite("beta_y", beta_y)
    gamma = _require_finite("gamma", gamma)
    theta_e = _require_positive("theta_e", theta_e)
    scale = max(theta_e, 1.0e-16)
    return (abs(gamma) < 1.0e-12
            and math.hypot(beta_x, beta_y) < 1.0e-12 * scale)


def caustic_from_critical(theta_x, theta_y, theta_e, gamma):
    """Push a critical curve into the source plane."""
    return map_sis_shear(theta_x, theta_y, theta_e, gamma)


def images_sis_shear(beta_x, beta_y, theta_e, gamma,
                     n_start=24, r_max=None, tol=1.0e-10):
    """Newton solve of beta(theta) = beta_src.

    Returns discrete images.  The centered gamma=0 Einstein ring is a
    continuum and is reported as an empty list; call
    ``is_einstein_ring_case`` to distinguish that degeneracy from a
    genuine zero-image failure.

    The radial search adapts to |beta| so a source outside the
    pseudo-caustic still yields its one remaining image.
    """
    beta_x = _require_finite("beta_x", beta_x)
    beta_y = _require_finite("beta_y", beta_y)
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    if abs(gamma) >= GAMMA_MAX:
        raise ValueError(
            f"shear |gamma| must be < {GAMMA_MAX:g} so the diamond "
            f"stays inside the pseudo-caustic, got {gamma:g}"
        )
    if is_einstein_ring_case(beta_x, beta_y, gamma, theta_e):
        return []
    beta_r = math.hypot(beta_x, beta_y)
    far = (beta_r + theta_e) / max(1.0e-6, 1.0 - abs(gamma))
    if r_max is None:
        r_lim = max(8.0 * theta_e, 2.0 * far)
    else:
        r_lim = float(r_max) * theta_e
    starts = []
    r_facs = (0.25, 0.55, 0.85, 1.15, 1.6, 2.2, 3.0,
              far / theta_e, 1.2 * far / theta_e)
    for r_fac in r_facs:
        r0 = abs(r_fac) * theta_e
        if r0 < 0.05 * theta_e:
            continue
        for k in range(int(n_start)):
            phi = 2.0 * math.pi * k / n_start
            starts.append((r0 * math.cos(phi), r0 * math.sin(phi)))
    # Analytic on-axis guesses for this shear convention.
    den = 1.0 - gamma
    if abs(den) > 1.0e-12:
        r_plus = (beta_x + theta_e) / den
        r_minus = (-beta_x + theta_e) / den
        if r_plus > 0.0:
            starts.append((r_plus, 0.0))
        if r_minus > 0.0:
            starts.append((-r_minus, 0.0))
    images = []
    for sx, sy in starts:
        th = np.array([sx, sy], dtype=float)
        ok = False
        for _ in range(40):
            if math.hypot(th[0], th[1]) < 1.0e-8 * theta_e:
                break
            bx, by = map_sis_shear(th[0], th[1], theta_e, gamma)
            f = np.array([bx - beta_x, by - beta_y], dtype=float)
            J = jacobian_matrix_sis_shear(th[0], th[1], theta_e, gamma)
            try:
                step = np.linalg.solve(J, f)
            except np.linalg.LinAlgError:
                break
            th = th - step
            if float(np.hypot(step[0], step[1])) < tol:
                ok = True
                break
        if not ok:
            continue
        if math.hypot(th[0], th[1]) > r_lim:
            continue
        bx, by = map_sis_shear(th[0], th[1], theta_e, gamma)
        if math.hypot(bx - beta_x, by - beta_y) > 1.0e-7 * max(theta_e, beta_r, 1.0e-16):
            continue
        if all(math.hypot(th[0] - a, th[1] - b) > IMAGE_DEDUP_FRAC * theta_e
               for a, b in images):
            images.append((float(th[0]), float(th[1])))
    images.sort(key=lambda p: math.atan2(p[1], p[0]))
    return images


def gaussian_source(beta_x, beta_y, beta_x0, beta_y0, sigma):
    sigma = _require_positive("sigma", sigma)
    return np.exp(-((beta_x - beta_x0)**2 + (beta_y - beta_y0)**2)
                  / (2.0 * sigma**2))


def sigma_from_r_eff(r_eff):
    """Gaussian sigma implied by a 2-D half-light / effective radius.

    For I = exp[-r^2 / (2 sigma^2)], half the light lies inside
    R_e = sigma * sqrt(2 ln 2).
    """
    r_eff = _require_positive("r_eff", r_eff)
    return r_eff / math.sqrt(2.0 * math.log(2.0))


def elliptical_source(beta_x, beta_y, beta_x0, beta_y0, r_eff, q, phi):
    """Elliptical Gaussian; r_eff is the half-light radius along the major axis."""
    r_eff = _require_positive("r_eff", r_eff)
    q = _require_finite("q", q)
    if q <= 0.0 or q > 1.0:
        raise ValueError("axis ratio q must lie in (0, 1]")
    phi = _require_finite("phi", phi)
    sigma = sigma_from_r_eff(r_eff)
    c, s = math.cos(phi), math.sin(phi)
    dx = beta_x - beta_x0
    dy = beta_y - beta_y0
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    rr = np.sqrt(xr**2 + (yr / q)**2)
    return np.exp(-(rr**2) / (2.0 * sigma**2))


def render_point_mass_source(theta_x, theta_y, theta_e,
                             beta_x0, beta_y0, sigma):
    bx, by = map_point_mass(theta_x, theta_y, theta_e)
    return gaussian_source(bx, by, beta_x0, beta_y0, sigma)


def render_point_mass_extended(theta_x, theta_y, theta_e,
                               beta_x0, beta_y0, r_eff, q, phi):
    bx, by = map_point_mass(theta_x, theta_y, theta_e)
    return elliptical_source(bx, by, beta_x0, beta_y0, r_eff, q, phi)


def render_sis_shear_extended(theta_x, theta_y, theta_e, gamma,
                              beta_x0, beta_y0, r_eff, q, phi):
    bx, by = map_sis_shear(theta_x, theta_y, theta_e, gamma)
    return elliptical_source(bx, by, beta_x0, beta_y0, r_eff, q, phi)


def default_point_mass_theta_e(log10_m_over_msun=12.0):
    log10_m_over_msun = _require_finite("log10_m_over_msun", log10_m_over_msun)
    mass = (10.0 ** log10_m_over_msun) * M_SUN
    return einstein_radius_point_mass(mass)


def default_sis_theta_e(sigma_v_kms=300.0):
    sigma_v_kms = _require_positive("sigma_v_kms", sigma_v_kms)
    return einstein_radius_sis(sigma_v_kms * 1.0e3)


def forward_point_mass_ray(beta, theta, theta_e, d_l=1.0, d_s=2.0):
    """One forward thin-lens ray from a fixed source through impact theta.

    Distances are schematic (default D_l = D_ls = 1, D_s = 2) so a plotted
    height equals the corresponding angle in radians.  The slope change at
    the lens is chosen so that y_obs = 0 if and only if the lens equation
    beta = theta - theta_E^2/theta holds.  The kink is toward the mass.
    """
    beta = _require_finite("beta", beta)
    theta = _require_finite("theta", theta)
    theta_e = _require_positive("theta_e", theta_e)
    d_l = _require_positive("d_l", d_l)
    d_s = _require_positive("d_s", d_s)
    if d_s <= d_l:
        raise ValueError("d_s must exceed d_l")
    if theta == 0.0:
        raise ValueError("theta must be nonzero")
    d_ls = d_s - d_l
    y_src = beta * d_s
    y_lens = theta * d_l
    alpha_signed = (theta_e * theta_e) * (d_s / d_ls) / theta
    m_in = (y_lens - y_src) / d_ls
    m_out = m_in - alpha_signed
    y_obs = y_lens + m_out * d_l
    return {
        "beta": float(beta),
        "theta": float(theta),
        "alpha": float(alpha_signed),
        "y_src": float(y_src),
        "y_lens": float(y_lens),
        "y_obs": float(y_obs),
        "m_in": float(m_in),
        "m_out": float(m_out),
        "hits": abs(y_obs) <= 1.0e-12 * max(theta_e * d_l, 1.0e-16),
        "points": np.array([[0.0, y_src], [d_ls, y_lens], [d_s, y_obs]],
                           dtype=float),
        "d_l": d_l,
        "d_s": d_s,
        "d_ls": d_ls,
    }


def default_ray_thetas(theta_e):
    """Impact angles for mode rays: Einstein pair plus inner and outer misses."""
    theta_e = _require_positive("theta_e", theta_e)
    factors = (-3.2, -2.2, -1.5, -1.0, -0.70, -0.45,
               0.45, 0.70, 1.0, 1.5, 2.2, 3.2)
    return tuple(f * theta_e for f in factors)


def forward_point_mass_bundle(beta, theta_e, thetas=None, d_l=1.0, d_s=2.0):
    """Forward rays from one source through many impact parameters."""
    beta = _require_finite("beta", beta)
    theta_e = _require_positive("theta_e", theta_e)
    if thetas is None:
        plus, minus = point_mass_image_radii(beta, theta_e)
        thetas = tuple(sorted(set(default_ray_thetas(theta_e)) | {plus, minus}))
    rays = [forward_point_mass_ray(beta, th, theta_e, d_l=d_l, d_s=d_s)
            for th in thetas]
    return {
        "beta": float(beta),
        "theta_e": float(theta_e),
        "d_l": float(d_l),
        "d_s": float(d_s),
        "d_ls": float(d_s - d_l),
        "y_src": float(beta * d_s),
        "rays": rays,
        "hits": [r for r in rays if r["hits"]],
        "misses": [r for r in rays if not r["hits"]],
    }


COMPACT_SOURCE_SIGMA_ARCSEC = 0.08


def adapted_fov_arcsec(theta_e, fov_arcsec, extras=(), margin=None):
    """Grow the field so theta_E and extras still fit.

    ``fov_arcsec`` is a minimum side length.  The returned value is
    ``max(fov, 2 * margin * largest_radius)`` with margin defaulting to
    FOV_MARGIN (1.2, a 20% border).
    """
    if margin is None:
        margin = FOV_MARGIN
    fov_arcsec = _require_positive("fov_arcsec", fov_arcsec)
    margin = _require_positive("margin", margin)
    half = [float(rad_to_arcsec(_require_positive("theta_e", theta_e)))]
    for extra in extras:
        half.append(abs(float(rad_to_arcsec(extra))))
    return max(fov_arcsec, 2.0 * margin * max(half))


def sis_shear_outer_image_radius(beta_x, beta_y, source_radius, theta_e, gamma):
    """Conservative image-plane radius that contains the source contour.

    For this shear convention a bound valid throughout |gamma| < 1 is
    (|beta| + R + theta_E) / (1 - |gamma|).
    """
    beta_x = _require_finite("beta_x", beta_x)
    beta_y = _require_finite("beta_y", beta_y)
    source_radius = _require_positive("source_radius", source_radius)
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    if abs(gamma) >= GAMMA_MAX:
        raise ValueError(
            f"shear |gamma| must be < {GAMMA_MAX:g} so the diamond "
            f"stays inside the pseudo-caustic, got {gamma:g}"
        )
    return ((math.hypot(beta_x, beta_y) + source_radius + theta_e)
            / (1.0 - abs(gamma)))


def adapted_n_pix(n_pix, fov_arcsec, smallest_scale_arcsec,
                  max_n_pix=None, remedy=None, pixels_per_scale=None):
    """Raise n_pix so spacing FOV/(n_pix-1) is no larger than scale/PIXELS.

    ``smallest_scale_arcsec`` is the scale used to *build* the grid
    (usually a source-plane width).  It is not itself an image-plane
    proof that every lensed image is resolved.
    """
    n_pix = int(n_pix)
    if n_pix < 9:
        raise ValueError("n_pix must be at least 9")
    if n_pix % 2 == 0:
        n_pix += 1
    fov_arcsec = _require_positive("fov_arcsec", fov_arcsec)
    smallest_scale_arcsec = _require_positive("smallest_scale_arcsec",
                                              smallest_scale_arcsec)
    if max_n_pix is None:
        max_n_pix = N_PIX_AUTO_MAX
    if pixels_per_scale is None:
        pixels_per_scale = PIXELS_PER_SIGMA
    pixels_per_scale = _require_positive("pixels_per_scale", pixels_per_scale)
    pixel_need = smallest_scale_arcsec / pixels_per_scale
    n_intervals = int(math.ceil(fov_arcsec / pixel_need))
    n_need = n_intervals + 1
    if n_need % 2 == 0:
        n_need += 1
    n_need = max(n_need, 9)
    if n_pix >= n_need:
        return n_pix
    if n_need > max_n_pix:
        hint = remedy or (
            "Move the source closer to the lens or use a smaller source."
        )
        raise ValueError(
            f"field {fov_arcsec:.2f}\" with sampling scale "
            f"{smallest_scale_arcsec:.4f}\" needs {n_need} pixels "
            f"({PIXELS_PER_SIGMA:g} samples per scale; spacing is "
            f"FOV/(n_pix-1)). Automatic grids stop at {max_n_pix}. {hint}"
        )
    return n_need


def require_resolved_point_mass_images(theta_e, beta, source_scale,
                                       fov_arcsec, n_pix,
                                       min_samples=None):
    """Reject a grid that cannot show both members of the point-mass double.

    ``min_samples`` is the fewest pixels allowed across each image-plane
    scale.  A uniform grid that would need tens of thousands of pixels
    to resolve a highly demagnified inner image is outside the teaching
    domain; the student should move the source closer, not raise n_pix.
    """
    theta_e = _require_positive("theta_e", theta_e)
    beta = _require_finite("beta", beta)
    source_scale = _require_positive("source_scale", source_scale)
    fov_arcsec = _require_positive("fov_arcsec", fov_arcsec)
    if min_samples is None:
        min_samples = IMAGE_DETECT_SAMPLES
    n_pix = int(n_pix)
    spacing = fov_arcsec / max(n_pix - 1, 1)
    plus, minus = point_mass_image_radii(abs(beta), theta_e)
    for th, name in ((plus, "outer"), (minus, "inner")):
        scale_as = float(rad_to_arcsec(
            point_mass_image_plane_scale(th, theta_e, source_scale)
        ))
        if scale_as < min_samples * spacing:
            raise ValueError(
                f"the {name} image is compressed to {scale_as:.4f}\" "
                f"while the grid spacing is {spacing:.4f}\". "
                f"This program renders both images of a point-mass double; "
                f"that offset is outside the teaching domain. "
                f"Move the source closer to the lens."
            )


def require_resolved_kappa(theta_e, fov_arcsec, n_pix):
    """Reject a kappa map whose Einstein circle is smaller than a few pixels."""
    te = float(rad_to_arcsec(_require_positive("theta_e", theta_e)))
    fov_arcsec = _require_positive("fov_arcsec", fov_arcsec)
    n_pix = int(n_pix)
    spacing = fov_arcsec / max(n_pix - 1, 1)
    intervals = te / spacing
    if intervals < KAPPA_MIN_EINSTEIN_INTERVALS:
        raise ValueError(
            f"theta_E = {te:.4f}\" spans only {intervals:.2f} grid "
            f"intervals. kappa mode needs at least "
            f"{KAPPA_MIN_EINSTEIN_INTERVALS:g} intervals so the "
            f"kappa=1/2 circle is visible. Use a larger --sigma_v."
        )


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
