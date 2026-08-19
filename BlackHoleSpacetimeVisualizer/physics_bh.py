"""
physics_bh.py
=============
Core physics engine for the Black_hole_spacetime_visualizer program.

Four calculations share this module, all built on the static, spherically
symmetric Schwarzschild geometry (and, for the horizons calculation, its
simplest dynamical generalisation, the ingoing Vaidya metric):

  embed     Flamm's paraboloid: the embedding of the equatorial Schwarzschild
            spatial geometry in flat 3-D Euclidean space.

  tidal     The radial (stretching) and tangential (compressing) tidal
            acceleration between two nearby free-falling test particles, as
            a function of radius, together with a simple "does a human
            survive to the horizon" comparison across black-hole masses.

  infall    A test particle released from rest at some radius and dropped
            radially into the hole: its radial coordinate, its proper time,
            the Schwarzschild coordinate time recorded by a distant static
            observer, its local infall speed, and the redshift of light it
            sends outward, all integrated with fourth-order Runge-Kutta.

  horizons  The distinction between the (local, instantaneous) apparent
            horizon and the (global, teleological) event horizon of a
            black hole that gains mass by swallowing a shell of infalling
            null dust -- an ingoing Vaidya spacetime.  The event horizon is
            located by a numerical shooting method: it is the one outgoing
            null geodesic that neither escapes to large radius nor falls to
            the singularity.

Every calculation here is an exact consequence of the Schwarzschild or
Vaidya metric -- there is no equation of state to choose and no fitted
closure, unlike the stellar-structure programs in this suite.  What is
*idealised* is the physical scenario: point test particles with no mass or
extent of their own, radial free-fall with no angular momentum, and, for
the horizons calculation, a mass function chosen by the student rather than
a mass supplied by a real accretion process.  The help file
(Black_hole_spacetime_visualizer.html) states plainly where each result
comes from.

SI units are used internally; user-facing quantities are in solar masses,
Schwarzschild radii, metres, milliseconds and multiples of standard
gravity g.
"""

import math
import numpy as np

MODEL_VERSION = "1.0.0"

# ----------------------------------------------------------------------
# Physical constants (SI, CODATA / IAU nominal values)
# ----------------------------------------------------------------------
G     = 6.674_30e-11        # m^3 kg^-1 s^-2
c     = 2.997_924_58e8      # m s^-1
M_sun = 1.988_92e30         # kg
g0    = 9.806_65             # m s^-2  (standard gravity, for tidal accel. in "g")

MODES = ("embed", "tidal", "infall", "horizons")

# Masses this model is happy to run.  General relativity itself places no
# lower or upper bound on a Schwarzschild mass; the bounds below mark the
# range of *astrophysically known* black holes, not a limit of the physics.
TRUSTED_MASS_LO = 2.0        # Msun -- below the observed mass gap
TRUSTED_MASS_HI = 5.0e10     # Msun -- above the most massive black holes known

MAX_POINTS = 200_000
MAX_STEPS = 4_000_000


# ======================================================================
# Small validation helpers (same conventions as physics_sev.py)
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


def check_mass(name, m_msun):
    """
    Validate a black-hole mass in solar masses.  Returns (m_msun, warnings).

    Any positive mass is accepted -- the Schwarzschild solution has no
    intrinsic mass scale -- but masses outside the range of astrophysically
    known black holes are flagged, because the pedagogical point of running
    one (for example, a 1 solar-mass or a 1e12 solar-mass hole) is usually
    precisely to see the numbers become extreme.
    """
    m_msun = _require_positive(name, m_msun)
    warnings = []
    if m_msun < TRUSTED_MASS_LO:
        warnings.append(
            f"{name} = {m_msun:g} Msun is below about {TRUSTED_MASS_LO:g} Msun, "
            "the lower edge of the observed neutron-star/black-hole mass gap.  "
            "The Schwarzschild geometry is exact at any mass; nature is simply "
            "not known to make black holes this light through stellar collapse."
        )
    elif m_msun > TRUSTED_MASS_HI:
        warnings.append(
            f"{name} = {m_msun:g} Msun exceeds the mass of the most massive "
            f"black holes known (a few times 10^10 Msun).  The physics below "
            "is still exact; only the astrophysical realism is in question."
        )
    return m_msun, warnings


# ======================================================================
# Schwarzschild geometry: basic scales
# ======================================================================
def schwarzschild_radius(m_msun):
    """r_s = 2GM/c^2, in metres."""
    m_msun = _require_positive("mass", m_msun)
    return 2.0 * G * (m_msun * M_sun) / c**2


def schwarzschild_radius_km(m_msun):
    return schwarzschild_radius(m_msun) / 1.0e3


def light_crossing_time(m_msun):
    """r_s / c, in seconds -- the natural timescale of a black hole of this mass."""
    return schwarzschild_radius(m_msun) / c


# ======================================================================
# Mode: embed -- Flamm's paraboloid
# ======================================================================
def embedding_profile(m_msun=10.0, r_max_rs=8.0, n_r=400):
    """
    The embedding of the equatorial (theta = pi/2), constant-t Schwarzschild
    spatial slice in flat 3-D Euclidean space.

    The Schwarzschild spatial metric in the equatorial plane is

        ds^2 = dr^2 / (1 - r_s/r) + r^2 dphi^2 .

    Embedding it as a surface of revolution z(r) in cylindrical coordinates
    (r, phi, z) of flat 3-space, ds^2 = dz^2 + dr^2 + r^2 dphi^2, requires

        (dz/dr)^2 = r_s / (r - r_s),

    which integrates, choosing z(r_s) = 0, to

        z(r) = 2 sqrt[ r_s (r - r_s) ]           (Flamm's paraboloid, 1916).

    The embedding is defined only for r >= r_s: the curvature of the
    equatorial slice grows without bound as r -> r_s, and dz/dr diverges
    there, which is exactly the vertical "throat" seen in the plot.  Nothing
    in the static exterior geometry corresponds to r < r_s, so the surface
    is not extended inside the throat; the popular picture of a "wormhole
    tube" continuing on the far side belongs to the maximally extended
    (Kruskal) spacetime, a different and larger solution, not to this one.
    """
    m_msun, warn_m = check_mass("mass", m_msun)
    r_max_rs = _require_finite("r_max_rs", r_max_rs)
    if r_max_rs <= 1.0:
        raise ValueError(
            f"r_max_rs must exceed 1 (the embedding starts at the horizon); "
            f"got {r_max_rs:g}."
        )
    if r_max_rs > 1000.0:
        raise ValueError(
            f"r_max_rs = {r_max_rs:g} is unreasonably large for a picture "
            "whose whole point is the near-horizon throat; try 5-20."
        )
    n_r = _require_int("n_r", n_r, lo=10, hi=MAX_POINTS)

    rs = schwarzschild_radius(m_msun)
    r = np.linspace(rs, rs * r_max_rs, n_r)
    z = 2.0 * np.sqrt(rs * (r - rs))

    # Local proper circumference vs. the flat-space (Euclidean) expectation,
    # a second, purely geometric way to see the curvature: on the curved
    # slice the proper radial distance from r_s to r exceeds r - r_s, while
    # the circumference at r is still exactly 2*pi*r (a defining property of
    # the Schwarzschild r coordinate).
    proper_radial_distance = np.array([
        _proper_radial_distance(rs, rs, rr) for rr in r
    ])

    return dict(
        kind="embed",
        r=r, z=z, proper_radial_distance=proper_radial_distance,
        summary=dict(
            m_msun=m_msun, rs_m=rs, rs_km=rs / 1.0e3,
            r_max_rs=r_max_rs, n_points=n_r,
            throat_note="the surface has a vertical tangent at r = r_s; "
                        "this is the throat, not a numerical artefact",
            warnings=warn_m,
            model_version=MODEL_VERSION,
        ),
    )


def _proper_radial_distance(rs, r_from, r_to, n=4000):
    """
    Proper radial distance integral(dr / sqrt(1 - r_s/r)) from r_from to
    r_to (both >= r_s), by Simpson's rule.  Used only for the small
    illustrative comparison printed alongside the embedding diagram.
    """
    if r_to <= r_from:
        return 0.0
    # Integrable inverse-square-root singularity at r = r_s is handled by
    # substituting r = r_s + u^2, dr = 2u du: since 1 - r_s/r = u^2/(u^2+r_s),
    # the integrand 2u/sqrt(1-r_s/r) = 2u sqrt(u^2+r_s)/u = 2 sqrt(u^2+r_s),
    # which is finite (equal to 2 sqrt(r_s)) even at u = 0.
    u_from = math.sqrt(max(r_from - rs, 0.0))
    u_to = math.sqrt(r_to - rs)
    u = np.linspace(u_from, u_to, n)
    integrand = 2.0 * np.sqrt(u * u + rs)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(integrand, u))


# ======================================================================
# Mode: tidal -- geodesic-deviation (tidal) acceleration
# ======================================================================
def tidal_acceleration(m_msun, r_m, separation_m=1.8):
    """
    Radial (stretching) and tangential (compressing) tidal acceleration
    between the two ends of a rod of proper length `separation_m`, centred
    at Schwarzschild radial coordinate r_m, oriented radially or
    tangentially respectively.

    From the Riemann tensor of the Schwarzschild geometry projected into the
    local orthonormal frame of a free-falling observer, the tidal
    (geodesic-deviation) acceleration along a radial separation dr is

        a_radial     = + (2GM/r^3) dr        (stretching, along the line to the hole)
        a_tangential = - (GM/r^3)  dl        (compressing, in either transverse
                                               direction)

    These are exact results for Schwarzschild, and -- a fact worth pausing
    on -- they are the *same* whether the observer is static or in radial
    free fall: the relevant curvature components are invariant under boosts
    along the radial direction, so the tide felt by someone falling in is
    identical, at each instant, to the tide that would be felt by someone
    momentarily at rest at the same radius.
    """
    M_kg = m_msun * M_sun
    r_m = np.asarray(r_m, dtype=float)
    coeff = G * M_kg / r_m**3
    a_radial = 2.0 * coeff * separation_m
    a_tangential = coeff * separation_m
    return a_radial, a_tangential


def tidal_profile(m_msun=10.0, r_min_rs=1.01, r_max_rs=10.0, n_r=400,
                  separation_m=1.8):
    """Tidal acceleration vs. radius, from just outside the horizon outward."""
    m_msun, warn_m = check_mass("mass", m_msun)
    r_min_rs = _require_finite("r_min_rs", r_min_rs)
    r_max_rs = _require_finite("r_max_rs", r_max_rs)
    if r_min_rs <= 1.0:
        raise ValueError(f"r_min_rs must exceed 1; got {r_min_rs:g}.")
    if r_max_rs <= r_min_rs:
        raise ValueError("r_max_rs must exceed r_min_rs.")
    n_r = _require_int("n_r", n_r, lo=10, hi=MAX_POINTS)
    separation_m = _require_positive("separation_m", separation_m)

    rs = schwarzschild_radius(m_msun)
    r = np.geomspace(rs * r_min_rs, rs * r_max_rs, n_r)
    a_r, a_t = tidal_acceleration(m_msun, r, separation_m)

    a_r_horizon, a_t_horizon = tidal_acceleration(m_msun, rs, separation_m)

    return dict(
        kind="tidal",
        r=r, a_radial=a_r, a_tangential=a_t,
        summary=dict(
            m_msun=m_msun, rs_m=rs, rs_km=rs / 1.0e3,
            separation_m=separation_m,
            r_min_rs=r_min_rs, r_max_rs=r_max_rs, n_points=n_r,
            a_radial_horizon=a_r_horizon, a_tangential_horizon=a_t_horizon,
            a_radial_horizon_g=a_r_horizon / g0,
            a_tangential_horizon_g=a_t_horizon / g0,
            warnings=warn_m,
            model_version=MODEL_VERSION,
        ),
    )


def survival_radius(m_msun, separation_m=1.8, limit_g=100.0):
    """
    Radius at which the radial tidal acceleration across `separation_m`
    first reaches `limit_g` standard gravities, found by inverting
    a_radial = 2GM dr / r^3 for r:

        r_crit = [ 2 G M separation_m / (limit_g * g0) ]^(1/3) .

    limit_g is a deliberately round illustrative threshold for structural
    survival, not a precise physiological limit; the exercises invite the
    student to adopt a different one and see how little the qualitative
    conclusion changes.
    """
    m_msun = _require_positive("mass", m_msun)
    separation_m = _require_positive("separation_m", separation_m)
    limit_g = _require_positive("limit_g", limit_g)
    M_kg = m_msun * M_sun
    limit_a = limit_g * g0
    r_crit = (2.0 * G * M_kg * separation_m / limit_a) ** (1.0 / 3.0)
    return r_crit


def compare_tidal_across_masses(mass_list_msun, separation_m=1.8, limit_g=100.0):
    """
    For each mass in mass_list_msun: the Schwarzschild radius, the radial
    tidal acceleration at the horizon, the survival radius for `limit_g`,
    and whether that survival radius lies inside or outside the horizon
    (i.e. whether the tidal limit is reached before or after the horizon
    is crossed).
    """
    rows = []
    for m in mass_list_msun:
        m = _require_positive("mass", m)
        rs = schwarzschild_radius(m)
        a_r_h, a_t_h = tidal_acceleration(m, rs, separation_m)
        r_crit = survival_radius(m, separation_m, limit_g)
        rows.append(dict(
            m_msun=m, rs_m=rs, rs_km=rs / 1.0e3,
            a_radial_horizon=a_r_h, a_radial_horizon_g=a_r_h / g0,
            r_crit_m=r_crit, r_crit_over_rs=r_crit / rs,
            survives_horizon=bool(r_crit < rs),
        ))
    return rows


# ======================================================================
# Mode: infall -- radial free fall and what a distant observer sees
# ======================================================================
def _infall_state(rs, r0, E, r):
    """
    dr/dtau and dt/dtau for a test particle dropped from rest at r0,
    at current radius r (all in SI units, c retained explicitly).

    Conserved specific energy for a particle released from rest at r0:
        E = sqrt(1 - r_s/r0)                     (E = 1 as r0 -> infinity)
    Radial equation of motion (from g_ab u^a u^b = -c^2):
        (dr/dtau)^2 = c^2 [ r_s/r - r_s/r0 ]
        dt/dtau      = E / (1 - r_s/r)
    """
    val = rs / r - rs / r0
    drdtau = -c * math.sqrt(val) if val > 0.0 else 0.0
    dtdtau = E / (1.0 - rs / r)
    return drdtau, dtdtau


def local_infall_speed(rs, r0, E, r):
    """
    Speed of the infalling particle relative to a local static observer at
    r, in units of c:

        v_local(r) = sqrt(r_s/r - r_s/r0) / E .

    v_local -> 1 as r -> r_s for every r0: any infalling observer's speed,
    as clocked by the local static observers it passes, approaches the
    speed of light at the horizon (the static observers themselves are the
    ones who cease to be able to exist there).
    """
    val = rs / r - rs / r0
    return math.sqrt(val) / E if val > 0.0 else 0.0


def outgoing_redshift_factor(rs, r0, E, r):
    """
    Ratio nu_observed / nu_emitted for a photon sent radially outward, at
    the instant the infalling particle is at radius r, and received by a
    static observer at infinity:

        nu_obs/nu_emit = sqrt(1 - r_s/r) * sqrt[ (1 - v) / (1 + v) ],

    the product of the ordinary gravitational redshift factor for a static
    source at r and the special-relativistic Doppler factor for a source
    receding (falling away from the outgoing photon) at local speed v =
    v_local(r).  For a particle released from rest at infinity (E = 1,
    v = sqrt(r_s/r)) this collapses to the well-known closed form
    nu_obs/nu_emit = 1 - sqrt(r_s/r).  In every case the ratio falls to
    zero as r -> r_s: light emitted ever closer to the horizon is received,
    ever more faintly and ever more redshifted, only after ever longer
    delay -- it is never seen to cross.
    """
    v = local_infall_speed(rs, r0, E, r)
    grav = math.sqrt(max(1.0 - rs / r, 0.0))
    dopp = math.sqrt((1.0 - v) / (1.0 + v)) if v < 1.0 else 0.0
    return grav * dopp


def infall_radial(m_msun=10.0, r0_rs=6.0, n_points=4000, r_stop_rs=1.0005,
                  step_frac=0.02):
    """
    Integrate the radial free-fall of a test particle released from rest at
    r0 = r0_rs * r_s, from r0 down to r_stop = r_stop_rs * r_s, with RK4 in
    the particle's own proper time tau.

    The step size is refined geometrically as the particle approaches
    r_stop, in the same spirit as the structure integrators of the other
    GFtGU programs: the coordinate time t and the redshift factor both vary
    increasingly rapidly (in fact divergently, in the tau -> tau_horizon
    limit) as r -> r_s, while r(tau) itself remains perfectly smooth.

    r_stop_rs must exceed 1: the Schwarzschild t coordinate, and this
    integrator's use of it, both break down at the horizon itself, which is
    a coordinate problem, not a physical one -- the particle's own clock
    (tau) runs through the horizon without incident, and the *reason* t and
    the redshift diverge instead of the physics being singular there is
    itself one of the central lessons of this mode.
    """
    m_msun, warn_m = check_mass("mass", m_msun)
    r0_rs = _require_finite("r0_rs", r0_rs)
    if r0_rs <= 1.0:
        raise ValueError(f"r0_rs must exceed 1 (start outside the horizon); got {r0_rs:g}.")
    r_stop_rs = _require_finite("r_stop_rs", r_stop_rs)
    if not (1.0 < r_stop_rs < r0_rs):
        raise ValueError(
            f"r_stop_rs must satisfy 1 < r_stop_rs < r0_rs; got r_stop_rs = "
            f"{r_stop_rs:g}, r0_rs = {r0_rs:g}."
        )
    n_points = _require_int("n_points", n_points, lo=20, hi=MAX_POINTS)
    step_frac = _require_finite("step_frac", step_frac)
    if not (1.0e-5 < step_frac <= 0.2):
        raise ValueError(f"step_frac must lie in (1e-5, 0.2]; got {step_frac:g}.")

    rs = schwarzschild_radius(m_msun)
    r0 = r0_rs * rs
    r_stop = r_stop_rs * rs
    E = math.sqrt(1.0 - rs / r0)

    # Start an infinitesimal distance inside r0: r0 itself is an exact
    # (unstable) fixed point of dr/dtau, since the particle is released
    # from rest there, so a numerical integration begun exactly at r0
    # would never move.
    r = r0 * (1.0 - 1.0e-12)
    tau = 0.0
    t_coord = 0.0

    taus = [0.0]
    ts = [0.0]
    rs_list = [r]
    steps = 0
    while r > r_stop and steps < MAX_STEPS:
        drdtau, _ = _infall_state(rs, r0, E, r)
        speed = max(abs(drdtau), 1.0e-6 * c)
        dtau = min(step_frac * (r - r_stop) / speed, step_frac * rs / c * 50.0)
        dtau = max(dtau, 1.0e-12 * rs / c)

        def f(rr):
            return _infall_state(rs, r0, E, rr)

        k1r, k1t = f(r)
        k2r, k2t = f(r + 0.5 * dtau * k1r)
        k3r, k3t = f(r + 0.5 * dtau * k2r)
        k4r, k4t = f(r + dtau * k3r)
        r_next = r + (dtau / 6.0) * (k1r + 2 * k2r + 2 * k3r + k4r)
        t_next = t_coord + (dtau / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)

        if not (math.isfinite(r_next) and math.isfinite(t_next)):
            raise RuntimeError(
                "The infall integration produced a non-finite state; "
                "raise --r_stop or reduce --step_frac."
            )
        if r_next <= r_stop:
            # Linearly interpolate the final partial step so the recorded
            # track ends exactly at r_stop, in both tau and t.
            frac = (r - r_stop) / (r - r_next) if r != r_next else 1.0
            tau += frac * dtau
            t_coord += frac * (t_next - t_coord)
            r = r_stop
        else:
            tau += dtau
            t_coord = t_next
            r = r_next

        taus.append(tau)
        ts.append(t_coord)
        rs_list.append(r)
        steps += 1

    r_arr = np.array(rs_list)
    tau_arr = np.array(taus)
    t_arr = np.array(ts)

    # The geometric step refinement above concentrates points near r_stop;
    # resample onto exactly n_points, by array index, for clean plotting
    # and CSV output, keeping the first and last (r0 and r_stop) points.
    if r_arr.size > n_points:
        idx = np.unique(np.round(np.linspace(0, r_arr.size - 1, n_points)).astype(int))
        r_arr, tau_arr, t_arr = r_arr[idx], tau_arr[idx], t_arr[idx]

    v_local = np.array([local_infall_speed(rs, r0, E, rr) for rr in r_arr])
    redshift = np.array([outgoing_redshift_factor(rs, r0, E, rr) for rr in r_arr])
    dtau_dt = (1.0 - rs / r_arr) / E   # instantaneous d(proper time)/d(coordinate time)

    warnings = list(warn_m)
    warnings.append(
        f"integration was stopped at r = {r_stop_rs:g} r_s, just outside the "
        "horizon, because the Schwarzschild t coordinate used for the "
        "distant observer's clock diverges there; the particle's own "
        "proper time does not."
    )

    return dict(
        kind="infall",
        tau=tau_arr, t=t_arr, r=r_arr,
        v_local=v_local, redshift=redshift, dtau_dt=dtau_dt,
        summary=dict(
            m_msun=m_msun, rs_m=rs, rs_km=rs / 1.0e3,
            r0_rs=r0_rs, r_stop_rs=r_stop_rs,
            tau_total_s=tau_arr[-1], tau_total_ms=tau_arr[-1] * 1.0e3,
            t_total_s=t_arr[-1], t_total_ms=t_arr[-1] * 1.0e3,
            E=E, n_points=r_arr.size,
            v_local_final=v_local[-1], redshift_final=redshift[-1],
            warnings=warnings,
            model_version=MODEL_VERSION,
        ),
    )


# ======================================================================
# Mode: horizons -- Vaidya apparent vs. event horizon
# ======================================================================
def _mass_geom(m_msun):
    """GM/c^2 for a mass in solar masses, i.e. half the Schwarzschild radius, in metres."""
    return G * (m_msun * M_sun) / c**2


def vaidya_mass_of_v(v, m0, m1, v1, v2):
    """
    Mass function of the ingoing Vaidya metric, advanced time v (metres,
    geometrised): constant at m0 before v1, ramps linearly to m1 between v1
    and v2, constant at m1 after v2.  m0, m1, v, v1, v2 all in metres
    (geometrised units, i.e. mass expressed as GM/c^2).
    """
    v = np.asarray(v, dtype=float)
    frac = np.clip((v - v1) / (v2 - v1), 0.0, 1.0)
    return m0 + (m1 - m0) * frac


def _outgoing_null_deriv(v, r, m0, m1, v1, v2):
    """dr/dv for an outgoing radial null geodesic of the ingoing Vaidya metric:
    dr/dv = (1/2)( 1 - 2 M(v)/r )."""
    return 0.5 * (1.0 - 2.0 * vaidya_mass_of_v(v, m0, m1, v1, v2) / r)


def _integrate_null_geodesic(v_start, r_start, v_end, m0, m1, v1, v2,
                             step_frac=0.01, max_steps=300_000):
    """
    RK4-integrate one outgoing null geodesic from (v_start, r_start) to
    v_end; returns (v_array, r_array).

    The step size is adaptive, capped at a fraction `step_frac` of the
    *current* radius, exactly as the structure integrators of the other
    GFtGU programs cap their step by a fraction of the current integration
    variable.  This matters here more than usual: dr/dv = (1/2)(1-2M/r)
    diverges as r -> 0, and a naive fixed step overshoots straight through
    r = 0 into negative r, where the sign of 2M/r flips and the geodesic
    appears, spuriously, to turn around and escape.  A trajectory that
    reaches a small radius is instead declared to have plunged and the
    integration stops there.
    """
    v_scale = max(v2 - v1, m0 + m1, 1.0e-30)
    r_floor = 1.0e-4 * (m0 + m1)

    v_list = [v_start]
    r_list = [r_start]
    v, r = v_start, r_start
    steps = 0
    while v < v_end and steps < max_steps:
        if r <= r_floor:
            break

        def f(vv, rr):
            return _outgoing_null_deriv(vv, rr, m0, m1, v1, v2)

        deriv = f(v, r)
        dv_r = step_frac * r / max(abs(deriv), 1.0e-12)
        dv = min(dv_r, step_frac * v_scale, v_end - v)
        dv = max(dv, 1.0e-9 * v_scale)

        k1 = deriv
        r1 = r + 0.5 * dv * k1
        if r1 <= 0.0:
            r = r1
            v = v + dv
            v_list.append(v)
            r_list.append(r)
            break
        k2 = f(v + 0.5 * dv, r1)
        r2 = r + 0.5 * dv * k2
        if r2 <= 0.0:
            r = r2
            v = v + dv
            v_list.append(v)
            r_list.append(r)
            break
        k3 = f(v + 0.5 * dv, r2)
        r3 = r + dv * k3
        if r3 <= 0.0:
            r = r3
            v = v + dv
            v_list.append(v)
            r_list.append(r)
            break
        k4 = f(v + dv, r3)
        r_next = r + (dv / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        v = v + dv
        r = r_next
        v_list.append(v)
        r_list.append(r)
        steps += 1

        if r > 1.0e4 * (m0 + m1):     # unambiguously escaped
            break

    return np.asarray(v_list), np.asarray(r_list)


def vaidya_horizons(m0_msun=5.0, m1_msun=10.0, v1_rs0=5.0, duration_rs0=10.0,
                    v_start_margin_rs0=25.0, v_end_margin_rs0=15.0,
                    n_steps=6000, bisect_iters=60, n_family=9):
    """
    Locate the apparent and event horizons of a black hole whose mass grows
    from m0 to m1 (solar masses) between advanced times v1 and v1+duration,
    in units of the initial Schwarzschild radius r_s0 = 2 G m0/c^2.

    Apparent horizon (local, instantaneous):
        r_AH(v) = 2 M(v) -- the radius, at each v, where the expansion of
        the outgoing null congruence vanishes.  It can be read off the mass
        function alone.

    Event horizon (global, teleological):
        the boundary of the region from which outgoing light can never
        escape to infinity.  It is generated by the one outgoing null
        geodesic that neither falls to r = 0 nor escapes to r -> infinity;
        finding it requires following light rays all the way to the future,
        which is exactly why it cannot be computed locally.  It is located
        here by a bisection shooting method: outgoing geodesics are fired
        from deep in the static region before v1, where the true horizon
        generator sits exponentially close to r = 2 m0, and the initial
        radius is bisected until the geodesic neither clearly escapes nor
        clearly plunges by v_end.

    Because the shooting method can only be started at a *finite* v_start,
    not at v = -infinity, the located event horizon carries a small,
    controllable residual error before v1 that shrinks as v_start is moved
    earlier or the number of bisection iterations is increased -- itself a
    genuine feature of the calculation, not a numerical embarrassment, and
    one of the suggested exercises asks you to demonstrate it directly.
    """
    m0_msun, warn0 = check_mass("M0", m0_msun)
    m1_msun, warn1 = check_mass("M1", m1_msun)
    if m1_msun < m0_msun:
        raise ValueError(
            f"M1 = {m1_msun:g} Msun is less than M0 = {m0_msun:g} Msun; the "
            "Vaidya mass function used here is non-decreasing (accretion "
            "of positive-energy null dust). A shrinking hole would need the "
            "outgoing-flux Vaidya metric, which is a different solution."
        )
    v1_rs0 = _require_positive("v1_rs0", v1_rs0)
    duration_rs0 = _require_positive("duration_rs0", duration_rs0)
    v_start_margin_rs0 = _require_positive("v_start_margin_rs0", v_start_margin_rs0)
    v_end_margin_rs0 = _require_positive("v_end_margin_rs0", v_end_margin_rs0)
    n_steps = _require_int("n_steps", n_steps, lo=200, hi=200_000)
    bisect_iters = _require_int("bisect_iters", bisect_iters, lo=10, hi=200)
    n_family = _require_int("n_family", n_family, lo=1, hi=41)

    m0 = _mass_geom(m0_msun)
    m1 = _mass_geom(m1_msun)
    rs0 = 2.0 * m0
    rs1 = 2.0 * m1
    v1 = v1_rs0 * rs0
    v2 = v1 + duration_rs0 * rs0
    v_start = v1 - v_start_margin_rs0 * rs0
    v_end = v2 + v_end_margin_rs0 * rs0

    # --- bisection for the critical (event-horizon) initial radius -------
    lo = 0.5 * rs0
    hi = 1.5 * rs0

    def escapes(r_i):
        _, r_arr = _integrate_null_geodesic(v_start, r_i, v_end, m0, m1, v1, v2)
        return r_arr[-1] > rs1

    if escapes(lo) or not escapes(hi):
        raise RuntimeError(
            "The shooting-method bracket [0.5, 1.5] r_s0 did not bracket "
            "the event horizon; this should not happen for a physically "
            "reasonable mass ratio and accretion duration."
        )

    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if escapes(mid):
            hi = mid
        else:
            lo = mid
    r_crit = 0.5 * (lo + hi)
    residual_rs0 = (hi - lo) / rs0

    v_eh, r_eh = _integrate_null_geodesic(v_start, r_crit, v_end, m0, m1, v1, v2)

    v_grid = np.linspace(v_start, v_end, n_steps + 1)
    r_ah = vaidya_mass_of_v(v_grid, m0, m1, v1, v2) * 2.0
    # Interpolate r_eh (computed on its own, possibly-truncated grid) onto
    # the common v_grid for direct comparison and CSV output.
    r_eh_grid = np.interp(v_grid, v_eh, r_eh, left=r_eh[0], right=r_eh[-1])

    # --- a family of nearby geodesics, for the "how the horizon is found"
    # panel: some inside the critical radius (fall in), some outside
    # (escape). The offset from r_crit needed to reach a *visibly* distinct
    # fate (clearly escaped, clearly plunged) by v_end depends on the local
    # expansion rate of the geometry near the horizon, which scales with the
    # final horizon size r_s1, not with the tiny bisection residual -- for a
    # large mass ratio M1/M0 a residual-sized offset never has time to grow
    # into a visible separation before v_end. Search geometrically for the
    # smallest offset that clearly separates, so the family panel looks
    # right regardless of the mass ratio or the accretion duration chosen.
    family = []
    if n_family > 1:
        spread = max(20.0 * residual_rs0 * rs0, 1.0e-6 * rs0)
        for _ in range(60):
            _, r_hi = _integrate_null_geodesic(v_start, r_crit + spread, v_end,
                                               m0, m1, v1, v2)
            _, r_lo = _integrate_null_geodesic(v_start, max(r_crit - spread,
                                                             1.0e-6 * rs0),
                                               v_end, m0, m1, v1, v2)
            if r_hi[-1] > 2.0 * rs1 and r_lo[-1] < 0.5 * rs0:
                break
            spread *= 1.6
            if spread > 0.5 * rs0:
                break
        offsets = np.linspace(-spread, spread, n_family)
    else:
        offsets = [0.0]
    for off in offsets:
        r_i = r_crit + off
        v_f, r_f = _integrate_null_geodesic(v_start, max(r_i, 1.0e-3 * rs0),
                                            v_end, m0, m1, v1, v2)
        family.append(dict(r_i_over_rs0=r_i / rs0, v=v_f, r=r_f,
                           escapes=bool(r_f[-1] > rs1)))

    warnings = list(warn0) + list(warn1)
    warnings.append(
        f"the located event horizon carries a residual uncertainty of about "
        f"{residual_rs0:.2e} r_s0 from the finite v_start and the {bisect_iters}-"
        "iteration bisection; see the algorithm section of the help file."
    )

    return dict(
        kind="horizons",
        v=v_grid, r_AH=r_ah, r_EH=r_eh_grid,
        v_eh_raw=v_eh, r_eh_raw=r_eh,
        family=family,
        summary=dict(
            m0_msun=m0_msun, m1_msun=m1_msun,
            rs0_m=rs0, rs1_m=rs1,
            rs0_km=rs0 / 1.0e3, rs1_km=rs1 / 1.0e3,
            v1_rs0=v1_rs0, v2_rs0=v1_rs0 + duration_rs0, duration_rs0=duration_rs0,
            v_start_rs0=v_start / rs0, v_end_rs0=v_end / rs0,
            r_crit_over_rs0=r_crit / rs0,
            residual_rs0=residual_rs0,
            bisect_iters=bisect_iters, n_steps=n_steps,
            light_crossing_time_rs0_s=rs0 / c,
            n_points=v_grid.size,
            warnings=warnings,
            model_version=MODEL_VERSION,
        ),
    )
