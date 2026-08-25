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

MODEL_VERSION = "1.2.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_bh.py",
    "driver_bh.py",
    "main.py",
    "plot_bh.py",
)


def _compute_build_id():
    """A short, content-derived build identifier (Copilot Audit 7 P1-2).

    This project has no external version-control system available at
    build time, so MODEL_VERSION alone cannot distinguish two builds
    that share the same version string but differ in source content
    (e.g. a local patch applied without remembering to bump
    MODEL_VERSION). This hashes the actual on-disk source of the core
    computational modules listed in BUILD_ID_COVERS, giving CSV
    provenance and --version output a machine-checkable answer to "did
    these two runs use byte-identical code?" without requiring git or
    any other source-control tooling. It degrades to "unknown" (never
    raises) if the source files cannot be located or read -- e.g.
    inside a frozen/zipped distribution -- since a missing build id
    should never prevent the program from running.

    Two robustness fixes over an earlier version, both catching real
    ways two semantically identical checkouts could otherwise hash
    differently or collide:

    - Read in TEXT mode (universal-newline translation) and re-encoded
      to UTF-8 before hashing, not raw bytes: a checkout with CRLF line
      endings (e.g. from Windows) would otherwise hash differently from
      the exact same source checked out with LF endings.
    - Each file's name and byte length are hashed immediately before
      its own content, not just all four files' bytes concatenated back
      to back: without that framing, moving a shared byte sequence
      across a file boundary (e.g. content shifted from the end of one
      file to the start of the next) can reproduce the exact same
      concatenated byte stream and therefore the same hash, even though
      the two files individually differ.
    """
    import hashlib
    import os
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha256()
        for name in BUILD_ID_COVERS:
            with open(os.path.join(here, name), "r", encoding="utf-8",
                      newline=None) as f:
                content = f.read().encode("utf-8")
            h.update(name.encode("utf-8"))
            h.update(len(content).to_bytes(8, "big"))
            h.update(content)
        return h.hexdigest()[:12]
    except (OSError, UnicodeDecodeError):
        return "unknown"


BUILD_ID = _compute_build_id()


# ----------------------------------------------------------------------
# Physical constants (SI, CODATA / IAU nominal values)
# ----------------------------------------------------------------------
G     = 6.674_30e-11        # m^3 kg^-1 s^-2 (CODATA; G is known only to
                             # about 1 part in 10^4-10^5)
c     = 2.997_924_58e8      # m s^-1 (exact, by definition of the metre)
M_sun = 1.988_92e30         # kg -- a conventional kilogram value for one
                             # solar mass. NOT used below to compute any
                             # mass-dependent geometry (see GM_SUN_NOMINAL):
                             # kept only as a documented, labelled constant
                             # in case a caller wants a mass in kilograms.
g0    = 9.806_65             # m s^-2  (standard gravity, for tidal accel. in "g")

#: The IAU 2015 Resolution B3 nominal solar mass *parameter*, i.e. the
#: product GM_sun directly, in m^3 s^-2. This is what every mass-dependent
#: formula below actually uses (schwarzschild_radius, _mass_geom,
#: tidal_acceleration, survival_radius), instead of separately multiplying
#: G by a kilogram mass for the Sun. The reason: G itself is known only to
#: about 1 part in 10^4-10^5, but the *product* GM_sun is pinned by Solar
#: System dynamics (planetary/spacecraft tracking) to roughly 1 part in
#: 10^10 and is fixed by IAU convention independently of any specific value
#: of G or of the Sun's kilogram mass, which itself is uncertain at the
#: same ~1e-4-1e-5 level as G. Computing G*(m_msun*M_sun) -- the previous
#: approach -- therefore threw away five to six digits of precision and,
#: at 1 solar mass, overstated the Schwarzschild radius by about 0.76 m
#: (about 7.6 m at 10 Msun), a discrepancy large enough to show up in
#: several exercises' numerical comparisons (Reviewer Audit round 1,
#: Codex P2-2). Reference: IAU 2015 Resolution B3, arXiv:1510.07674.
GM_SUN_NOMINAL = 1.327_124_4e20   # m^3 s^-2, exact by IAU definition

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
    except OverflowError as exc:
        # float(x) raises OverflowError, not ValueError, when x is a Python
        # int too large to represent as a C double (e.g. a several-hundred-
        # digit integer typed on the command line). Previously this
        # propagated as an uncaught traceback instead of the intended
        # concise error message (Reviewer Audit round 1, Codex P2-3).
        raise ValueError(
            f"{name} is too large to represent as a floating-point number; "
            f"got {value!r}."
        ) from exc
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
    """r_s = 2GM/c^2, in metres, computed as 2*m_msun*GM_SUN_NOMINAL/c^2
    using the IAU nominal solar mass parameter directly rather than
    separately-sourced G and kilogram values for the Sun (Reviewer Audit
    round 1, Codex P2-2; see the GM_SUN_NOMINAL module-level comment)."""
    m_msun = _require_positive("mass", m_msun)
    rs = 2.0 * m_msun * GM_SUN_NOMINAL / c**2
    if not math.isfinite(rs):
        # A finite input mass does not guarantee a finite result: an
        # astronomically large (but technically finite) --M overflows the
        # multiplication before this check, e.g. schwarzschild_radius(1e300)
        # previously returned inf with no error at all (Reviewer Audit
        # round 1, Codex P2-3).
        raise ValueError(
            f"schwarzschild_radius is not finite for mass = {m_msun:g} "
            "Msun; the mass is too large to compute with."
        )
    return rs


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

    The embedding is defined only for r >= r_s: dz/dr diverges as r -> r_s,
    which is exactly the vertical "throat" seen in the plot.  It is worth
    being precise about what diverges and what does not: dz/dr is the slope
    of the *embedding* -- an extrinsic property of how the 2-D slice sits
    inside the auxiliary flat 3-D space used to draw it -- and it is this
    slope, not the slice's own intrinsic curvature, that blows up.  The
    intrinsic Gaussian curvature of the slice itself, computable from the
    2-D metric alone (K = -r_s / (2 r^3), by either the general orthogonal-
    coordinate formula applied to this metric or the equivalent surface-
    of-revolution formula f'f''/[r(1+f'^2)^2] applied to z(r) above -- both
    give the same result, as the Theorema Egregium requires), is finite
    everywhere on the domain r >= r_s, including at the throat, where it
    takes the finite value K(r_s) = -1/(2 r_s^2).  The sign is worth
    dwelling on: despite the throat's convex, bowl-like appearance in the
    embedding, the slice is intrinsically *negatively* curved everywhere,
    like a trumpet or a pseudosphere, not positively curved like a sphere
    or an ordinary paraboloid z ~ r^2 (whose curvature is +1/[a^2(1+r^2/
    a^2)^2], positive everywhere) -- the visual convexity is a fact about
    this particular embedding in flat 3-space, not about the intrinsic
    geometry it depicts.  The throat is a smooth, regular minimal
    surface, not a curvature singularity; nothing in this static exterior
    geometry is singular before r = 0.  Nothing in the static exterior
    geometry corresponds to r < r_s, so the surface is not extended inside
    the throat; the popular picture of a "wormhole tube" continuing on the
    far side belongs to the maximally extended (Kruskal) spacetime, a
    different and larger solution, not to this one.
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
    proper_radial_distance = _proper_radial_distance(rs, rs, r)

    # The intrinsic Gaussian curvature of the slice itself (see the module
    # docstring above for the derivation and its sign): exposed here as a
    # genuine program output, not merely discussed in prose, so it can be
    # plotted, exported to CSV, and used directly by an exercise asking
    # for "the throat's curvature scale" (Reviewer Audit round 1, Codex/
    # Copilot EXP-18: the previous help file asked for this quantity
    # without the program actually returning it).
    K = gaussian_curvature(rs, r)

    return dict(
        kind="embed",
        r=r, z=z, proper_radial_distance=proper_radial_distance, K=K,
        summary=dict(
            m_msun=m_msun, rs_m=rs, rs_km=rs / 1.0e3,
            r_max_rs=r_max_rs, n_points=n_r,
            K_at_horizon=float(K[0]),
            throat_note="the surface has a vertical tangent at r = r_s; "
                        "this is the throat, not a numerical artefact",
            warnings=warn_m,
            model_version=MODEL_VERSION,
            build_id=BUILD_ID,
        ),
    )


def gaussian_curvature(rs, r_m):
    """
    Intrinsic Gaussian curvature K(r) = -r_s / (2 r^3) of the equatorial,
    constant-t Schwarzschild spatial slice (see the derivation in this
    module's docstring, and embedding_profile's docstring, above). Accepts
    a scalar or array r_m, both in metres, and requires r_m >= rs.
    """
    r_m = np.asarray(r_m, dtype=float)
    if np.any(r_m < rs):
        raise ValueError("gaussian_curvature is defined only for r >= r_s.")
    return -rs / (2.0 * r_m**3)


def _proper_radial_distance(rs, r_from, r_to):
    """
    EXACT closed-form proper radial distance
    integral(dr / sqrt(1 - r_s/r)) from r_from to r_to (both >= r_s, and
    r_to may be a scalar or numpy array).

    Reviewer Audit round 1, Codex P2-6: the previous implementation
    evaluated this integral numerically (4000-point trapezoidal rule on a
    substitution-regularised grid) while the help file and this program's
    own "exact vs. numerical" classification called embed mode "closed-
    form ... no integration," which was not actually true of this one
    auxiliary output. The antiderivative is elementary (verified
    independently both by hand and with a computer-algebra system, and
    cross-checked here against the previous trapezoidal implementation to
    confirm they agree to numerical-integration precision before it was
    retired):

        F(r) = sqrt(r(r-r_s)) + r_s * asinh( sqrt((r-r_s)/r_s) )
        integral_{r_from}^{r_to} dr/sqrt(1-r_s/r) = F(r_to) - F(r_from)

    (equivalent to, but numerically better-conditioned than, the more
    commonly tabulated sqrt(r(r-r_s)) + r_s*ln(sqrt(r-r_s)+sqrt(r)) form,
    which carries an r_s-dependent additive constant that cancels in the
    difference F(r_to)-F(r_from) but is easy to drop incorrectly by hand;
    asinh avoids that pitfall and remains accurate as r_to/r_s -> 1).
    """
    r_from = np.asarray(r_from, dtype=float)
    r_to = np.asarray(r_to, dtype=float)

    def F(r):
        r = np.maximum(r, rs)   # F(r<=rs) = 0 exactly; avoid sqrt of a
                                 # tiny negative number from roundoff.
        return np.sqrt(r * (r - rs)) + rs * np.arcsinh(np.sqrt((r - rs) / rs))

    result = F(r_to) - F(r_from)
    result = np.where(r_to <= r_from, 0.0, result)
    if result.ndim == 0:
        return float(result)
    return result


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
    (geodesic-deviation) acceleration magnitudes across a separation dr (or
    dl) are

        a_radial     = (2GM/r^3) dr          (stretching: the two ends are
                                               pulled apart, along the line
                                               to the hole)
        a_tangential = (GM/r^3)  dl           (compressing: the two ends are
                                               pushed together, in either
                                               transverse direction)

    Both are returned here as positive magnitudes; the sense of each -- one
    stretching, the other compressing -- is carried by the words "radial"
    and "tangential" and stated explicitly wherever these numbers are
    printed, plotted or labelled in a CSV column, rather than by a sign on
    the number itself.

    Precision is warranted about exactly what is and is not exact here
    (Reviewer Audit round 1, Codex P1-5). The curvature *components*
    2GM/r^3 and GM/r^3 -- the coefficients multiplying `separation_m`
    above -- are exact results for Schwarzschild: a fact worth pausing on
    is that they are the *same* whether the observer is static or in
    radial free fall, since the relevant curvature components are
    invariant under boosts along the radial direction, so the tide felt by
    someone falling in is identical, at each instant, to the tide that
    would be felt by someone momentarily at rest at the same radius.
    Multiplying an exact local curvature by a *finite* separation_m,
    however, is the linearized (infinitesimal-separation) geodesic-
    deviation approximation, not an exact finite-body result at any order:
    it assumes the curvature does not change appreciably across the rod.
    That assumption is excellent for every astrophysical example this
    program is meant to illustrate (a person-scale separation is utterly
    negligible next to any stellar or supermassive horizon), but the
    program's own bounds on mass and separation do not by themselves
    guarantee it -- see `TIDAL_LINEARIZATION_RATIO_WARN` and the warning
    `tidal_profile` adds when the ratio is not small.
    """
    m_msun = _require_positive("mass", m_msun)
    separation_m = _require_positive("separation_m", separation_m)
    r_m = np.asarray(r_m, dtype=float)
    if not np.all(np.isfinite(r_m)):
        raise ValueError("r_m must be finite.")
    if np.any(r_m <= 0.0):
        raise ValueError("r_m must be strictly positive.")
    # GM directly from the IAU nominal solar mass parameter, not G times a
    # separately-sourced kilogram mass -- see GM_SUN_NOMINAL (Reviewer
    # Audit round 1, Codex P2-2). tidal_acceleration is a public helper
    # that tidal_profile does not gate behind its own validation, so this
    # function validates its own inputs directly (Codex P2-3).
    GM = m_msun * GM_SUN_NOMINAL
    coeff = GM / r_m**3
    a_radial = 2.0 * coeff * separation_m
    a_tangential = coeff * separation_m
    if not (np.all(np.isfinite(a_radial)) and np.all(np.isfinite(a_tangential))):
        raise ValueError(
            "tidal_acceleration produced a non-finite result; m_msun, r_m "
            "and separation_m are too extreme in combination."
        )
    return a_radial, a_tangential


#: Above this ratio of separation_m to the smallest radius sampled, the
#: linearized geodesic-deviation approximation (multiplying the local
#: curvature by a finite separation, as if it were infinitesimal) is no
#: longer defensible: the tidal *tensor* components 2GM/r^3 and -GM/r^3 are
#: exact, but treating their product with a finite-length rod as the rod's
#: actual acceleration assumes the curvature does not change appreciably
#: across the rod. 1e-3 is a deliberately generous threshold (a truly
#: rigorous bound would also involve the second derivative of the
#: curvature and the rod's material response) meant to catch only the
#: regime -- reachable at this program's permitted small-mass, default-
#: separation combinations -- where the approximation is qualitatively,
#: not just quantitatively, wrong (Reviewer Audit round 1, Codex P1-5: at
#: m_msun = 1e-6, r_s is about 3 mm and the default 1.8 m separation is
#: some 600 horizon radii, far outside any meaning of "linearized").
TIDAL_LINEARIZATION_RATIO_WARN = 1.0e-3


def tidal_profile(m_msun=10.0, r_min_rs=1.01, r_max_rs=10.0, n_r=400,
                  separation_m=1.8):
    """Tidal acceleration vs. radius, from just outside the horizon outward.

    a_radial and a_tangential are the *exact* local Schwarzschild tidal-
    tensor components (2GM/r^3 and GM/r^3) multiplied by `separation_m`,
    i.e. the linearized (infinitesimal-separation) geodesic-deviation
    approximation to the acceleration between the two ends of a rod of
    that proper length. This is an excellent approximation whenever
    separation_m is small compared with r -- true for every astrophysical
    example this program is meant to illustrate -- but it is not
    guaranteed by the code's own bounds on mass and separation, so a
    warning is added to the summary when separation_m exceeds
    TIDAL_LINEARIZATION_RATIO_WARN times the smallest sampled radius
    (Reviewer Audit round 1, Codex P1-5).
    """
    m_msun, warn_m = check_mass("mass", m_msun)
    r_min_rs = _require_finite("r_min_rs", r_min_rs)
    r_max_rs = _require_finite("r_max_rs", r_max_rs)
    if r_min_rs <= 1.0:
        raise ValueError(f"r_min_rs must exceed 1; got {r_min_rs:g}.")
    if r_max_rs <= r_min_rs:
        raise ValueError("r_max_rs must exceed r_min_rs.")
    # tidal_profile shares --r_max_rs with embedding_profile (Reviewer Audit
    # round 1, Codex P2-7), but the two modes have different reasons to
    # bound it: embedding_profile's picture is specifically about the
    # near-horizon throat (bound 1000), while tidal_profile is also used to
    # verify the far-field 1/r^3 power law (EXP-8) and so is allowed a much
    # larger range. The bound here exists only to keep r = rs*r_max_rs, and
    # everything computed from it, finite (Codex P2-3); it is not a
    # physical or pedagogical limit.
    if r_max_rs > 1.0e8:
        raise ValueError(
            f"r_max_rs = {r_max_rs:g} is too large; must not exceed 1e8."
        )
    n_r = _require_int("n_r", n_r, lo=10, hi=MAX_POINTS)
    separation_m = _require_positive("separation_m", separation_m)

    rs = schwarzschild_radius(m_msun)
    r = np.geomspace(rs * r_min_rs, rs * r_max_rs, n_r)
    if not np.all(np.isfinite(r)):
        raise ValueError(
            "tidal_profile's radius grid is not finite; reduce --M or "
            "--r_max_rs."
        )
    a_r, a_t = tidal_acceleration(m_msun, r, separation_m)

    a_r_horizon, a_t_horizon = tidal_acceleration(m_msun, rs, separation_m)

    linearization_ratio = separation_m / (rs * r_min_rs)
    warn_lin = list(warn_m)
    if linearization_ratio > TIDAL_LINEARIZATION_RATIO_WARN:
        warn_lin.append(
            f"separation_m = {separation_m:g} m is {linearization_ratio:.3g} "
            f"times the smallest sampled radius (r_min = {r_min_rs:g} r_s = "
            f"{rs * r_min_rs:.3g} m); the linearized geodesic-deviation "
            "approximation used here (multiplying the exact local tidal "
            "tensor by a finite separation, as if it were infinitesimal) "
            "is not reliable in this regime -- treat a_radial/a_tangential "
            "as indicative only, not as the exact acceleration of a rigid "
            "body of this length."
        )

    return dict(
        kind="tidal",
        r=r, a_radial=a_r, a_tangential=a_t,
        summary=dict(
            m_msun=m_msun, rs_m=rs, rs_km=rs / 1.0e3,
            separation_m=separation_m,
            r_min_rs=r_min_rs, r_max_rs=r_max_rs, n_points=n_r,
            linearization_ratio=linearization_ratio,
            a_radial_horizon=a_r_horizon, a_tangential_horizon=a_t_horizon,
            a_radial_horizon_g=a_r_horizon / g0,
            a_tangential_horizon_g=a_t_horizon / g0,
            warnings=warn_lin,
            model_version=MODEL_VERSION,
            build_id=BUILD_ID,
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
    # GM from the IAU nominal solar mass parameter directly (Reviewer Audit
    # round 1, Codex P2-2; see GM_SUN_NOMINAL).
    GM = m_msun * GM_SUN_NOMINAL
    limit_a = limit_g * g0
    r_crit = (2.0 * GM * separation_m / limit_a) ** (1.0 / 3.0)
    if not math.isfinite(r_crit):
        raise ValueError(
            "survival_radius produced a non-finite result; m_msun, "
            "separation_m and limit_g are too extreme in combination."
        )
    return r_crit


def compare_tidal_across_masses(mass_list_msun, separation_m=1.8, limit_g=100.0):
    """
    For each mass in mass_list_msun: the Schwarzschild radius, the radial
    tidal acceleration at the horizon, the survival radius for `limit_g`,
    and whether that survival radius lies inside or outside the horizon
    (i.e. whether the tidal limit is reached before or after the horizon
    is crossed).

    Each mass is validated the same way as the primary `--M` mass (via
    `check_mass`), so a comparison mass outside the astrophysically known
    range is flagged in the returned row's `warnings`, not silently accepted
    -- the earlier version of this function used a looser check that missed
    this.
    """
    limit_g = _require_positive("limit_g", limit_g)
    rows = []
    for m in mass_list_msun:
        m, warn_m = check_mass("compare_masses entry", m)
        rs = schwarzschild_radius(m)
        a_r_h, a_t_h = tidal_acceleration(m, rs, separation_m)
        r_crit = survival_radius(m, separation_m, limit_g)
        rows.append(dict(
            m_msun=m, rs_m=rs, rs_km=rs / 1.0e3,
            a_radial_horizon=a_r_h, a_radial_horizon_g=a_r_h / g0,
            r_crit_m=r_crit, r_crit_over_rs=r_crit / rs,
            survives_horizon=bool(r_crit < rs),
            warnings=warn_m,
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

    The step size is refined geometrically near *both* ends of the track,
    in the same spirit as the structure integrators of the other GFtGU
    programs. Near r_stop the refinement is needed because the coordinate
    time t and the redshift factor both vary increasingly rapidly (in fact
    divergently, in the tau -> tau_horizon limit) as r -> r_s. Near r0 it is
    needed for a different reason: dr/dtau, as a *function of r*, behaves as
    -const * sqrt(r0 - r) close to release (the particle starts from rest,
    so its speed rises like the square root of the distance fallen), which
    has an infinite slope in r at r = r0. RK4's fourth-order accuracy
    assumes the right-hand side is smooth on the scale of a step, and that
    assumption fails right at this branch point; without refining the step
    there too, the integrator silently drops to first-order accuracy for
    the whole run, however small --step_frac is made elsewhere. Capping the
    step by the (shrinking) distance to r0, exactly as is already done for
    the distance to r_stop, restores full fourth-order behaviour: with this
    version of the integrator, --step_frac 0.2 (the coarsest value allowed)
    already agrees with an exact closed-form (cycloid-parametrised) benchmark
    solution to better than one part in 10^6 for the default parameters.

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
    if r0_rs > 1.0e8:
        # Purely a finite-computation guard (Reviewer Audit round 1, Codex
        # P2-3: infall_radial(..., r0_rs=1e308) reached a non-finite
        # state), not a physical or pedagogical limit.
        raise ValueError(f"r0_rs = {r0_rs:g} is too large; must not exceed 1e8.")
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

    # --- Startup: release point r0 down to a numerical seed point --------
    # r0 itself is *not* a stationary point of the physical, second-order
    # equation of motion -- a particle released from rest there has a
    # perfectly ordinary nonzero inward proper acceleration. What is
    # singular at r0 is only the *first-order* form (dr/dtau)^2 = c^2(rs/r
    # - rs/r0) used by this integrator (via _infall_state): it is
    # non-Lipschitz at r=r0 (infinite dr/dr-slope there, as the docstring
    # above explains), which is a numerical-startup issue for this
    # particular ODE formulation, not a physical equilibrium (Reviewer
    # Audit round 1, Codex P1-3, correcting the previous round's "exact
    # (unstable) fixed point" language, which conflated the two).
    #
    # The previous implementation stepped an infinitesimal distance inside
    # r0 to dodge that non-Lipschitz point, but then assigned that seed
    # point tau=0 and t=0 -- silently discarding the (tiny but nonzero and
    # exactly quantifiable) proper and coordinate time actually elapsed
    # over that first sliver of fall. That omission, not RK4 convergence,
    # was the source of the ~1.3e-6 relative-error floor previously
    # observed and misattributed to two-sided step refinement (Codex
    # P1-3). It is fixed here by seeding the numerical integration with
    # the *exact* elapsed tau and t for that sliver, taken from the closed-
    # form cycloid solution of the same radial-infall problem (the same
    # solution already used elsewhere in this module as an independent
    # benchmark):
    #
    #   r(eta)   = (r0/2) (1 + cos eta)
    #   tau(eta) = sqrt(r0^3 / (8 m_geom)) (eta + sin eta) / c
    #
    # with m_geom = rs/2 = GM/c^2, and eta=0 at release (r=r0). Solving
    # r(eta0) = r0*(1-eps) for eta0 via the (numerically stable, no
    # cancellation for tiny eps) half-angle identity 1-cos(eta) =
    # 2 sin^2(eta/2) gives eta0 = 2 arcsin(sqrt(eps)); the seed radius is
    # unchanged from before (same eps = 1e-12, so the RK4 integrator below
    # starts from exactly the same r and inherits exactly the same
    # step-size behaviour), only its (tau, t) labels are corrected.
    #
    # t(eta) has no equally short closed form (it involves the tortoise
    # coordinate's logarithmic divergence at the horizon), but none is
    # needed here: for eta0 this small, r stays within a fractional eps of
    # r0 (r0-r ~ (r0/4)*eta0^2 = eps*r0/... utterly negligible), so
    # dt/dtau is, to relative accuracy O(eta0^2) ~ O(eps) ~ 1e-12 -- far
    # below any other error in this calculation -- just its release-point
    # value dt/dtau|_{r=r0} = E/(1-rs/r0) = 1/E, giving t_seed = tau_seed/E.
    eps = 1.0e-12
    eta0 = 2.0 * math.asin(math.sqrt(eps))
    m_geom = 0.5 * rs
    tau_seed = (math.sqrt(r0**3 / (8.0 * m_geom)) * (eta0 + math.sin(eta0))) / c
    t_seed = tau_seed / E

    r = r0 * (1.0 - eps)
    tau = tau_seed
    t_coord = t_seed

    # The exact release event (r0, tau=0, t=0) is prepended as its own
    # sample, separately from the numerical seed point above, so the
    # returned track's first row is genuinely r0 -- not r0*(1-eps) mislabeled
    # as r0 (Reviewer Audit round 1, Copilot P2-1) -- and so the omitted
    # startup interval is visible in the data as a distinct (tiny) first
    # step rather than folded invisibly into the second point.
    taus = [0.0, tau]
    ts = [0.0, t_coord]
    rs_list = [r0, r]
    steps = 0
    while r > r_stop and steps < MAX_STEPS:
        drdtau, _ = _infall_state(rs, r0, E, r)
        speed = max(abs(drdtau), 1.0e-6 * c)
        # Cap the step by the (shrinking) distance to r_stop, as before, AND
        # by the (shrinking) distance back to r0: dr/dtau has an infinite
        # slope in r right at r0 (release from rest), and without this
        # second cap RK4's accuracy silently degrades from fourth order to
        # first order for the whole run -- see the docstring above.
        dist_to_stop = r - r_stop
        dist_from_start = max(r0 - r, 1.0e-9 * rs)
        dtau = min(step_frac * dist_to_stop / speed,
                  step_frac * dist_from_start / speed,
                  step_frac * rs / c * 50.0)
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

    if r > r_stop:
        # The loop above can only exit with r > r_stop by exhausting
        # MAX_STEPS: silently reporting that truncated track as though it
        # were the finished run would misstate tau_total and t_total.  This
        # is not expected for any physically reasonable --r0_rs, --r_stop_rs
        # and --step_frac, but fail loudly rather than risk it.
        raise RuntimeError(
            f"The infall integration did not reach r_stop after the "
            f"maximum of {MAX_STEPS:,} steps (stopped at r = {r/rs:.6g} "
            f"r_s, {steps:,} steps taken); try a larger --step_frac or a "
            "--r_stop_rs further from the horizon."
        )

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
            build_id=BUILD_ID,
        ),
    )


# ======================================================================
# Mode: horizons -- Vaidya apparent vs. event horizon
# ======================================================================
def _mass_geom(m_msun):
    """GM/c^2 for a mass in solar masses, i.e. half the Schwarzschild radius,
    in metres, using the IAU nominal solar mass parameter directly (Reviewer
    Audit round 1, Codex P2-2; see GM_SUN_NOMINAL)."""
    return m_msun * GM_SUN_NOMINAL / c**2


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

    "Escapes" and "plunges" are both, necessarily, finite-time numerical
    verdicts, not the literal infinite-time statements those words describe:
    a trajectory is called plunged the moment r first drops to or below
    r_floor = 1e-4*(m0+m1), and unambiguously escaped the moment r first
    exceeds 1e4*(m0+m1); a trajectory that reaches v_end without doing
    either is classified by comparing its radius there with r_s1 (see the
    caller). All three thresholds are comfortably far from the interesting
    region near the horizon for any physically reasonable mass, but they
    are thresholds, not limits -- worth stating plainly rather than leaving
    implicit.
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
        # Align the step to land exactly on v1 or v2 if it would otherwise
        # straddle one of them (Reviewer Audit round 1, Gemini finding 1).
        # M(v) is only C0, not C1, at these two hinges (a piecewise-linear
        # ramp has a kinked, discontinuous derivative there), so a step
        # that mixes both sides in its k1..k4 evaluations is only
        # first-order accurate across that one step, however small
        # --step_frac is. Forcing a fresh step to start exactly at the
        # hinge restores the integrator's normal fourth-order behaviour
        # immediately on the other side.
        if v < v1 < v + dv:
            dv = v1 - v
        elif v < v2 < v + dv:
            dv = v2 - v

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

    if v < v_end and r > r_floor and r <= 1.0e4 * (m0 + m1) and steps >= max_steps:
        # Loop exited only because the internal step cap was reached, with
        # the trajectory neither plunged, escaped, nor at v_end: silently
        # returning this partial track would let the caller (in particular
        # the bisection shooting search) misclassify an unresolved
        # trajectory as one that has genuinely escaped or plunged. This
        # should not happen for any physically reasonable input -- surface
        # it loudly rather than let it corrupt the located event horizon.
        raise RuntimeError(
            f"An outgoing null-geodesic integration hit the internal step "
            f"cap ({max_steps:,} steps) before reaching v_end, r_floor, or "
            f"the escape radius (stopped at r/(m0+m1) = {r/(m0+m1):.3g}, "
            f"v short of v_end by {(v_end - v)/v_scale:.3g} in v_scale "
            "units). Try a larger --v_end_margin_rs0, a shorter "
            "--duration_rs0, or a smaller --M1/--M0 ratio."
        )

    return np.asarray(v_list), np.asarray(r_list)


def _integrate_event_horizon_backward(v_bc, r_bc, v_target, m0, m1, v1, v2,
                                      step_frac=0.01, max_steps=300_000):
    """
    Construct the event-horizon generator r_EH(v) directly, by RK4
    integrating dr/dv = (1/2)(1-2M(v)/r) BACKWARD in v from the exact
    boundary condition (v_bc, r_bc) down to v_target <= v_bc. Returns
    (v_array, r_array) with v increasing from v_target to v_bc.

    Why backward, and why this is the primary construction as of
    MODEL_VERSION 1.2.0 (Reviewer Audit round 1, Codex P1-1/P1-2/P2-1 and
    Copilot P1-1/P1-2, and this module's own M0==M1 special case from the
    previous round, which this construction now subsumes as an
    unremarkable limit rather than needing a separate code path):

    r = 2M(v) is a fixed point of this ODE whenever M is locally constant
    (before v1, and after v2). Linearising an offset epsilon = r - 2M
    gives d(epsilon)/dv = epsilon/(2M): epsilon grows exponentially
    FORWARD in v (the fixed point is unstable forward) and shrinks
    exponentially BACKWARD in v (the identical ODE run in reverse is
    stable/self-correcting). The forward "shooting" construction kept
    below (for the geodesic-family visualisation and as a secondary
    accuracy diagnostic) locates the horizon generator by bisecting a
    STARTING radius deep in the static era and integrating forward
    through however much v-range the student's --v_start_margin_rs0 and
    --v_end_margin_rs0 request; because that range spans one or two
    unstable epochs, any offset from the true generator -- even one at
    the level of a single double-precision ULP, which is all bisection
    can ever remove -- is amplified exponentially and can produce a
    grossly wrong curve (reproduced in Audit round 1 with M0=M1=8 Msun,
    --bisect_iters 20, and separately with M0=10, M1=10.5 at defaults,
    where the forward construction's own r_EH(v_end) undershot r_s1 by
    several parts in 1e6, i.e. numerically violated r_EH >= r_AH).

    The boundary condition r(v2) = 2*m1 is exact (once accretion has
    finished, the spacetime is exactly static Schwarzschild with mass
    m1, and the event and apparent horizons coincide there by
    definition), so integrating backward from it needs no bisection or
    search at all, and -- because offsets shrink rather than grow in
    this direction -- the result is numerically well-conditioned however
    far backward it is asked to go, unlike the forward construction.
    This is exactly the "numerically cleaner" alternative construction
    this module's help file already names in its Algorithm design note;
    Audit round 1 is what established that it is not merely cleaner but
    necessary for a correct default-parameter result.
    """
    v_scale = max(v2 - v1, m0 + m1, 1.0e-30)

    v_list = [v_bc]
    r_list = [r_bc]
    v, r = v_bc, r_bc
    steps = 0
    while v > v_target and steps < max_steps:
        def f(vv, rr):
            return _outgoing_null_deriv(vv, rr, m0, m1, v1, v2)

        deriv = f(v, r)
        dv_r = step_frac * r / max(abs(deriv), 1.0e-12)
        dv = min(dv_r, step_frac * v_scale, v - v_target)
        dv = max(dv, 1.0e-9 * v_scale)
        # Same hinge-alignment reasoning as the forward integrator: do not
        # let one RK4 step straddle v1 or v2, where M(v)'s derivative is
        # discontinuous.
        if v - dv < v2 < v:
            dv = v - v2
        elif v - dv < v1 < v:
            dv = v - v1

        k1 = deriv
        r1 = r - 0.5 * dv * k1
        k2 = f(v - 0.5 * dv, r1)
        r2 = r - 0.5 * dv * k2
        k3 = f(v - 0.5 * dv, r2)
        r3 = r - dv * k3
        k4 = f(v - dv, r3)
        r_next = r - (dv / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        v = v - dv
        r = r_next
        v_list.append(v)
        r_list.append(r)
        steps += 1

        if not (math.isfinite(v) and math.isfinite(r)) or r <= 0.0:
            raise RuntimeError(
                "The backward event-horizon integration produced a "
                "non-finite or non-positive radius; check that --M0, "
                "--M1 and the accretion window are physically reasonable."
            )

    if v > v_target and steps >= max_steps:
        raise RuntimeError(
            f"The backward event-horizon integration hit the internal "
            f"step cap ({max_steps:,} steps) before reaching v_start "
            f"(stopped at (v-v_target)/v_scale = {(v - v_target)/v_scale:.3g} "
            "short). Try a smaller --v_start_margin_rs0."
        )

    v_arr = np.asarray(v_list[::-1])
    r_arr = np.asarray(r_list[::-1])
    return v_arr, r_arr


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
        geodesic that neither falls to r = 0 nor escapes to r -> infinity.

    CHANGE OF METHOD as of MODEL_VERSION 1.2.0 (Reviewer Audit round 1:
    Codex findings P1-1/P1-2/P2-1, Copilot findings P1-1/P1-2). The
    reported r_EH(v) curve is now built by integrating the outgoing null
    geodesic BACKWARD in v from the exact boundary condition r = 2 m1 at
    v = v2 (once accretion has finished the spacetime is exactly static
    with mass m1, so the event and apparent horizons coincide there
    exactly, by definition, needing no search). This replaces the
    previous round's forward "shooting" construction -- bisecting a
    starting radius deep in the static era and integrating forward to
    v_end -- as the PRIMARY source of the reported horizon. The forward
    construction is retained below, but only as (a) the mechanism behind
    the geodesic-family visualisation panel, and (b) a secondary
    diagnostic (`shooting_vs_backward_rs0`) that quantifies its own
    error, no longer as the thing being reported.

    Why the change: r = 2M(v) is a fixed point of dr/dv = (1/2)(1-2M/r)
    whenever M is locally constant (before v1, and again after v2).
    Linearising an offset epsilon = r - 2M gives d(epsilon)/dv =
    epsilon/(2M): the fixed point is UNSTABLE forward in v and STABLE
    backward in v (same ODE, opposite direction). The forward shooting
    construction inherits that instability -- any offset from the true
    generator, even one at the single-ULP level that is all bisection can
    ever remove, is amplified exponentially over however much v-range
    --v_start_margin_rs0/--v_end_margin_rs0 request -- while the backward
    construction, starting from an *exact* boundary condition, is
    self-correcting in the same sense. Reviewer Audit round 1 demonstrated
    concretely that the previous forward-only construction could return a
    curve that numerically violates its own documented invariant
    r_EH >= r_AH (reproduced with M0=10, M1=10.5 at every documented
    default -- no unusual --bisect_iters or margin needed -- where the
    forward r_EH undershot r_s1, and hence r_AH there, by several parts in
    a million), and, at the low --bisect_iters values Exercise EXP-13
    directly told students to try, could be wrong by *several r_s0* (the
    M0==M1 case fixed in the previous round was, in retrospect, a special
    case of this same general fragility, not an isolated defect of its
    own -- the general backward construction below also gives the exact
    M0==M1 answer, without needing that round's special case at all,
    which is kept only as a fast path).

    `residual_rs0` (the forward bisection bracket's width) and
    `r_crit_shooting_over_rs0` (the forward-located critical radius
    itself) are still computed and reported, but strictly as what they
    are: `residual_rs0` measures only how many times a bracket was
    halved, not the accuracy of the geodesic inside it, and it is driven
    to floating-point noise (~2^-60 relative) by `bisect_iters` alone,
    independent of everything else -- do not read a small residual as
    evidence the forward-shooting horizon itself is accurate (Reviewer
    Audit round 1, Codex P2-1 and Copilot P1-5). The genuinely useful
    accuracy comparison is `shooting_vs_backward_rs0`, the actual
    difference between the forward and backward constructions at
    v_start: at generous settings the two agree to many significant
    figures (a real, meaningful convergence check); at the settings
    Exercise EXP-13 explores, they can differ by orders of r_s0, which is
    the point of that exercise once correctly framed (see EXP-13 in the
    help file).

    `r_crit_over_rs0 - 1` (from the now-primary backward construction) is
    the physical, not merely numerical, displacement of the true event
    horizon above the static value r = 2 m0 at the chosen v_start: because
    the exact event horizon only equals 2 m0 in the strict v_start ->
    -infinity limit, every finite v_start asks for the horizon's radius
    at a different, specific event on the one true horizon, not a more or
    less "truncated" estimate of a single fixed answer. Moving v_start
    earlier measures that physical displacement at an earlier event (where
    it is smaller, since the displacement itself decays exponentially into
    the static past); it does not converge a numerical error toward zero.
    (Reviewer Audit round 1, Codex P1-2 and Copilot P1-1: the previous
    docstring/help-file language calling this a "finite-v_start truncation
    error" was a genuine conceptual mistake, now corrected throughout.)
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
    # v1_rs0 (where accretion begins, on the arbitrary advanced-time axis)
    # may be any finite value, including zero or negative: only
    # --duration_rs0 and the two margins have a physically required sign
    # (Reviewer Audit round 1, Codex P3-1).
    v1_rs0 = _require_finite("v1_rs0", v1_rs0)
    duration_rs0 = _require_positive("duration_rs0", duration_rs0)
    v_start_margin_rs0 = _require_positive("v_start_margin_rs0", v_start_margin_rs0)
    v_end_margin_rs0 = _require_positive("v_end_margin_rs0", v_end_margin_rs0)
    n_steps = _require_int("n_steps", n_steps, lo=200, hi=200_000)
    bisect_iters = _require_int("bisect_iters", bisect_iters, lo=10, hi=200)
    n_family = _require_int("n_family", n_family, lo=1, hi=41)
    if n_family == 2:
        # n_family=1 (just the horizon generator itself, no bracketing
        # family -- given a dedicated, differently-labelled presentation
        # by plot_bh.plot_horizons) and n_family>=3 (a genuine bracketing
        # family, symmetric about r_crit) are both meaningful; n_family=2
        # is neither -- two trajectories cannot straddle r_crit
        # symmetrically the way np.linspace(-spread, spread, 2) actually
        # produces (both endpoints, no centre point), and it is not the
        # single-trajectory case either (Reviewer Audit round 1, Codex
        # "lower-level" observation on the n_family=1 panel).
        raise ValueError(
            "n_family must be 1 (no bracketing family, just the horizon "
            "generator) or at least 3 (a genuine bracketing family); got 2."
        )

    m0 = _mass_geom(m0_msun)
    m1 = _mass_geom(m1_msun)
    rs0 = 2.0 * m0
    rs1 = 2.0 * m1
    v1 = v1_rs0 * rs0
    v2 = v1 + duration_rs0 * rs0
    v_start = v1 - v_start_margin_rs0 * rs0
    v_end = v2 + v_end_margin_rs0 * rs0

    no_accretion = (m0_msun == m1_msun)

    # --- primary construction: backward integration from the exact
    # boundary condition r(v2) = r_s1 -----------------------------------
    if no_accretion:
        # Fast path only -- M(v) = m0 identically, so r_EH = r_AH = r_s0
        # is the exact algebraic answer and the general backward
        # integrator would (correctly, but needlessly slowly) rediscover
        # exactly this by finding the derivative is bit-for-bit zero at
        # r = r_s0 for the whole run.
        r_crit = rs0
        v_eh_back, r_eh_back = np.array([v_start, v2]), np.array([rs0, rs0])
    else:
        v_eh_back, r_eh_back = _integrate_event_horizon_backward(
            v2, rs1, v_start, m0, m1, v1, v2)
        r_crit = float(r_eh_back[0])

    v_grid = np.linspace(v_start, v_end, n_steps + 1)
    r_ah = vaidya_mass_of_v(v_grid, m0, m1, v1, v2) * 2.0
    # For v >= v2 the mass is exactly constant at m1 and r = r_s1 is an
    # exact fixed point (as for the no_accretion fast path above): no
    # integration is needed or performed there, only for v_start <= v < v2.
    r_eh_grid = np.where(
        v_grid < v2,
        np.interp(v_grid, v_eh_back, r_eh_back, left=r_eh_back[0], right=rs1),
        rs1,
    )
    v_eh, r_eh = v_grid, r_eh_grid.copy()  # kept for API/CSV compatibility

    # Postcondition checks on the now-primary result. These are expected
    # to hold essentially to floating-point/RK4-step precision given the
    # backward construction's self-correcting stability; if they do not,
    # something about the requested physical scenario or resolution is
    # genuinely outside what this integrator can resolve, and it is
    # better to say so than to export a curve that contradicts its own
    # documented invariants (Reviewer Audit round 1, Codex P1-1).
    ah_violation_rs0 = float(np.max((r_ah - r_eh_grid) / rs0))
    if ah_violation_rs0 > 1.0e-6:
        raise RuntimeError(
            f"The located event horizon fell below the apparent horizon "
            f"by up to {ah_violation_rs0:.3e} r_s0, which should not "
            "happen with the backward construction; try a larger "
            "--n_steps or check that --M0/--M1 and the accretion window "
            "are physically reasonable."
        )

    # --- secondary: forward bisection shooting, retained for the
    # geodesic-family visualisation panel and as an accuracy diagnostic
    # against the (now primary) backward result -- see the docstring.
    if no_accretion:
        r_crit_shooting = rs0
        residual_rs0 = 0.0
    else:
        lo = 0.5 * rs0
        hi = max(1.5 * rs0, 1.5 * rs1)

        def escapes(r_i):
            _, r_arr = _integrate_null_geodesic(v_start, r_i, v_end, m0, m1, v1, v2)
            return r_arr[-1] > rs1

        for _ in range(20):
            if not escapes(lo):
                break
            lo *= 0.5
        for _ in range(60):
            if escapes(hi):
                break
            hi *= 1.6

        if escapes(lo) or not escapes(hi):
            raise RuntimeError(
                f"Could not bracket the event horizon even after adaptively "
                f"widening the search to [{lo/rs0:.3g}, {hi/rs0:.3g}] r_s0; try "
                "a larger --v_end_margin_rs0 (so trial geodesics have enough "
                "advanced time to clearly escape or plunge) or check that "
                "--M0/--M1 and the accretion window are physically reasonable."
            )

        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            if escapes(mid):
                hi = mid
            else:
                lo = mid
        r_crit_shooting = 0.5 * (lo + hi)
        residual_rs0 = (hi - lo) / rs0

    shooting_vs_backward_rs0 = abs(r_crit_shooting - r_crit) / rs0

    # --- a family of nearby geodesics, for the "how the horizon is found"
    # panel: some inside the critical radius (fall in), some outside
    # (escape), forward-integrated and centred on the primary (backward-
    # derived) r_crit -- so the panel is correct even in the parameter
    # regimes where the forward-shooting search itself (above) is not.
    # The offset needed to reach a *visibly* distinct fate by v_end
    # depends on the local expansion rate of the geometry near the
    # horizon, which scales with the final horizon size r_s1, not with
    # the tiny bisection residual -- for a large mass ratio M1/M0 a
    # residual-sized offset never has time to grow into a visible
    # separation before v_end. Search geometrically for the smallest
    # offset that clearly separates, so the family panel looks right
    # regardless of the mass ratio or the accretion duration chosen.
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
    if no_accretion:
        warnings.append(
            "M0 = M1: no accretion, so r_AH(v) = r_EH(v) = r_s0 is the "
            "exact algebraic answer for the whole run (both the primary "
            "backward construction and the diagnostic forward-shooting "
            "search agree with it exactly here); residual_rs0 = 0 and "
            "bisect_iters is unused for this run."
        )
    else:
        warnings.append(
            f"primary result: backward integration from the exact "
            f"r(v2)=r_s1 boundary condition, giving "
            f"r_crit/r_s0 = {r_crit/rs0:.8f} at v_start -- the physical "
            "early-time displacement of the event horizon above r_s0, "
            "not a truncation error (see the docstring / help file). "
            f"diagnostic only: a forward bisection search (unused for "
            f"the reported curve) located r_crit_shooting/r_s0 = "
            f"{r_crit_shooting/rs0:.8f} with bracket width residual_rs0 "
            f"= {residual_rs0:.2e} r_s0 ({bisect_iters} iterations) -- "
            f"the two constructions differ by "
            f"{shooting_vs_backward_rs0:.2e} r_s0 here; residual_rs0 "
            "alone does not measure that difference (see EXP-13)."
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
            r_crit_shooting_over_rs0=r_crit_shooting / rs0,
            shooting_vs_backward_rs0=shooting_vs_backward_rs0,
            residual_rs0=residual_rs0,
            bisect_iters=bisect_iters, n_steps=n_steps,
            light_crossing_time_rs0_s=rs0 / c,
            n_points=v_grid.size,
            warnings=warnings,
            model_version=MODEL_VERSION,
            build_id=BUILD_ID,
        ),
    )
