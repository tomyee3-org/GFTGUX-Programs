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

MODEL_VERSION = "0.1.4"

BUILD_ID_COVERS = (
    "physics_gl.py",
    "driver_gl.py",
    "main.py",
    "plot_gl.py",
)


def _compute_build_id():
    """Return a short identifier derived from the four core source files.

    MODEL_VERSION is the declared release.  BUILD_ID distinguishes source
    revisions that keep the same declared version.  The hash is independent
    of LF versus CRLF and frames each file with its name and length.

    Return ``"unknown"`` rather than preventing the program from running if
    the source files cannot be located.
    """
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
    if abs(gamma) >= 1.0:
        raise ValueError("shear |gamma| must be less than 1")
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


def critical_curve_sis_shear(theta_e, gamma, n_theta=361, r_max=3.0):
    """Sample det A = 0 on a polar grid and return (theta_x, theta_y)."""
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    phis = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=True)
    rs = np.linspace(0.05 * theta_e, r_max * theta_e, 400)
    xs = []
    ys = []
    for phi in phis:
        c, s = math.cos(phi), math.sin(phi)
        det_vals = jacobian_det_sis_shear(rs * c, rs * s, theta_e, gamma)
        sign = np.sign(det_vals)
        crossings = np.where(sign[1:] * sign[:-1] <= 0.0)[0]
        if crossings.size == 0:
            continue
        i = int(crossings[0])
        d0, d1 = float(det_vals[i]), float(det_vals[i + 1])
        if d1 == d0:
            r_c = rs[i]
        else:
            r_c = rs[i] - d0 * (rs[i + 1] - rs[i]) / (d1 - d0)
        xs.append(r_c * c)
        ys.append(r_c * s)
    if len(xs) < 8:
        raise RuntimeError("failed to trace a closed critical curve")
    return np.asarray(xs), np.asarray(ys)


def caustic_from_critical(theta_x, theta_y, theta_e, gamma):
    """Push a critical curve into the source plane."""
    return map_sis_shear(theta_x, theta_y, theta_e, gamma)


def images_sis_shear(beta_x, beta_y, theta_e, gamma,
                     n_start=24, r_max=2.5, tol=1.0e-10):
    """Newton solve of beta(theta) = beta_src from a polar ring of starts."""
    beta_x = _require_finite("beta_x", beta_x)
    beta_y = _require_finite("beta_y", beta_y)
    theta_e = _require_positive("theta_e", theta_e)
    gamma = _require_finite("gamma", gamma)
    starts = []
    for r_fac in (0.3, 0.7, 1.1, 1.6, 2.2):
        for k in range(int(n_start)):
            phi = 2.0 * math.pi * k / n_start
            starts.append((r_fac * theta_e * math.cos(phi),
                           r_fac * theta_e * math.sin(phi)))
    images = []
    for sx, sy in starts:
        th = np.array([sx, sy], dtype=float)
        ok = False
        for _ in range(25):
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
        bx, by = map_sis_shear(th[0], th[1], theta_e, gamma)
        if math.hypot(bx - beta_x, by - beta_y) > 1.0e-6:
            continue
        if math.hypot(th[0], th[1]) > r_max * theta_e * 1.2:
            continue
        if all(math.hypot(th[0] - a, th[1] - b) > 0.04 * theta_e
               for a, b in images):
            images.append((float(th[0]), float(th[1])))
    images.sort(key=lambda p: math.atan2(p[1], p[0]))
    return images


def gaussian_source(beta_x, beta_y, beta_x0, beta_y0, sigma):
    sigma = _require_positive("sigma", sigma)
    return np.exp(-((beta_x - beta_x0)**2 + (beta_y - beta_y0)**2)
                  / (2.0 * sigma**2))


def elliptical_source(beta_x, beta_y, beta_x0, beta_y0, r_eff, q, phi):
    """Elliptical Gaussian; q is the axis ratio in (0, 1]."""
    r_eff = _require_positive("r_eff", r_eff)
    q = _require_finite("q", q)
    if q <= 0.0 or q > 1.0:
        raise ValueError("axis ratio q must lie in (0, 1]")
    phi = _require_finite("phi", phi)
    c, s = math.cos(phi), math.sin(phi)
    dx = beta_x - beta_x0
    dy = beta_y - beta_y0
    xr = dx * c + dy * s
    yr = -dx * s + dy * c
    rr = np.sqrt(xr**2 + (yr / q)**2)
    return np.exp(-(rr**2) / (2.0 * r_eff**2))


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


def thin_lens_side_rays(thetas, beta, theta_e, model="point", gamma=0.0,
                        d_l=1.0, d_s=2.0):
    """Polyline rays in a side-view cartoon (optical axis along x).

    Heights are angular coordinates times the relevant distance, so the
    drawing is dimensionally a small-angle unwrapping of the thin-lens
    geometry rather than a full geodesic integration.

    Each ray is the triple
        source plane  ->  lens plane (height = image angle * D_l)
                      ->  observer (height 0).
    The kink at the lens *is* the deflection.  For ``model='point'`` the
    family of rays that reach the observer from an on-axis source is the
    Einstein cone.  Mapping the same family back into the source plane
    shows how a caustic is the envelope of those incoming rays.
    """
    theta_e = _require_positive("theta_e", theta_e)
    beta = _require_finite("beta", beta)
    d_l = _require_positive("d_l", d_l)
    d_s = _require_positive("d_s", d_s)
    if d_s <= d_l:
        raise ValueError("d_s must exceed d_l")
    d_ls = d_s - d_l
    thetas = np.asarray(thetas, dtype=float)
    rays = []
    for th in thetas:
        if model == "point":
            ax, ay = deflection_point_mass(0.0, th, theta_e)
        elif model == "shear":
            ax, ay = deflection_sis_shear(0.0, th, theta_e, gamma)
        else:
            raise ValueError(f"unknown model {model!r}")
        # Lens-plane height and the source-plane height implied by the
        # lens equation, y_s = (theta - alpha) * D_s.
        y_lens = float(th) * d_l
        y_src = float(th - ay) * d_s
        rays.append({
            "theta": float(th),
            "alpha": float(ay),
            "points": np.array([
                [0.0, y_src],
                [d_ls, y_lens],
                [d_s, 0.0],
            ], dtype=float),
        })
    return {"d_l": d_l, "d_s": d_s, "d_ls": d_ls, "beta": beta, "rays": rays}


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


def schematic_forward_rays(offset=False):
    """Deprecated alias kept so older tests still import a name."""
    theta_e = default_point_mass_theta_e(12.0)
    beta = 0.0 if not offset else -0.40 * theta_e
    return forward_point_mass_bundle(beta, theta_e)
