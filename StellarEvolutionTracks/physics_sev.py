"""
physics_sev.py
==============
Core physics engine for the StellarEvolutionTracks program.

Four calculations share this module:

  tracks   Evolution of a single star.  The main sequence is obtained by
           integrating a nuclear-burning ODE coupled to homology relations
           for L and R.  The post-main-sequence portion integrates a
           shell-burning core-growth ODE closed by the core-mass-luminosity
           relation and an empirical Hayashi-line fit.

  hr       The same track calculation repeated over a grid of masses, with
           an optional set of isochrones constructed from the tracks.

  wdcool   White-dwarf structure from the zero-temperature Chandrasekhar
           degenerate-electron equation of state in Newtonian hydrostatic
           equilibrium (integrated as ODEs in radius), followed by a Mestel
           cooling ODE for the core temperature.

  nsmr     Neutron-star mass-radius relation from the Tolman-Oppenheimer-
           Volkoff equations with a choice of equation of state.

SI units are used internally.  User-facing quantities are in solar masses,
solar radii, solar luminosities, kelvin, kilometres and years.

Every model in this module is a deliberately simple teaching model.  The
help file (StellarEvolutionTracks.html) states explicitly which results are
integrated from stated differential equations and which are prescriptions
or empirical fits.  Nothing here is a substitute for a stellar-evolution
code: the tracks stop at helium ignition, and the compact-object modes are
separate calculations reached through a schematic remnant prescription
rather than a continuous integration.
"""

import math
import numpy as np

MODEL_VERSION = "1.3.0"


#: The exact source files this build identifier covers: a documentation-only
#: change, a sample-output file, or an edit to the test suite does not change
#: this value -- only the four core program modules listed here do.  Exposed
#: so callers can determine precisely what BUILD_ID covers without duplicating
#: this list.
BUILD_ID_COVERS = (
    "physics_sev.py",
    "driver_sev.py",
    "main.py",
    "plot_sev.py",
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


# ----------------------------------------------------------------------
# Physical constants (SI, CODATA 2022 / IAU nominal values)
# ----------------------------------------------------------------------
G       = 6.674_30e-11          # m^3 kg^-1 s^-2
c       = 2.997_924_58e8        # m s^-1
h_pl    = 6.626_070_15e-34      # J s
k_B     = 1.380_649e-23         # J K^-1
sigma_SB = 5.670_374_419e-8     # W m^-2 K^-4
a_rad   = 4.0 * sigma_SB / c    # J m^-3 K^-4
m_u     = 1.660_539_068_92e-27  # kg   (atomic mass constant, CODATA 2022)
m_e     = 9.109_383_713_9e-31   # kg   (electron mass, CODATA 2022)
m_n     = 1.674_927_500_56e-27  # kg   (neutron mass, CODATA 2022)

#: IAU 2015 Resolution B3 nominal solar mass parameter, exact by
#: definition.  GM_sun is known from solar-system dynamics to a relative
#: precision many orders better than G or M_sun individually, so the
#: IAU-recommended way to fix the solar mass is to divide this nominal GM
#: by the Newtonian constant actually used above -- not to hard-code an
#: independently rounded M_sun in kilograms, which would be inconsistent
#: with G everywhere the two appear together (surface gravity, the
#: Kelvin-Helmholtz time, the TOV and Newtonian structure integrations).
GM_SUN_NOMINAL = 1.327_124_4e20  # m^3 s^-2
M_sun   = GM_SUN_NOMINAL / G     # kg     (derived, consistent with G above)
R_sun   = 6.957e8               # m      (IAU nominal solar radius)
L_sun   = 3.828e26              # W      (IAU nominal solar luminosity)
TEFF_SUN = 5772.0               # K      (IAU nominal solar effective temp.)
R_EARTH = 6_371_008.7714        # m      (IUGG/GRS80 mean Earth radius;
                                 #         Moritz, J. Geodesy 74, 128 (2000))

YEAR    = 365.25 * 86400.0      # s      (Julian year, exactly 365.25 d;
                                 #         IAU-recommended definition of a
                                 #         year for astronomical ages)
GYR     = 1.0e9 * YEAR

EPS_NUC = 0.007 * c**2          # J kg^-1 released turning H into He

# Safety limits
MAX_TRACK_STEPS   = 2_000_000
MIN_TRACK_STEPS   = 50
MAX_STRUCT_STEPS  = 400_000
MAX_GRID_POINTS   = 20_000
MAX_MASSES        = 40
MAX_ISOCHRONES    = 10     # cap on the number of ages the hr mode will
                            # interpolate isochrones for in one run

# Range over which the closures of this one-zone model are pedagogically
# reliable.  Masses outside it are still accepted -- watching the
# approximations fail is one of the exercises -- but the run says so.
TRUSTED_MASS_LO = 0.35     # below this, stars are fully convective
TRUSTED_MASS_HI = 15.0     # above this, winds and radiation pressure matter


# ======================================================================
# Small validation helpers
# ======================================================================
def _require_finite(name, value):
    """Return value as float after giving a consistent user-facing error."""
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


def _require_bool(name, value):
    """
    Reject anything that is not literally True/False before it is used in
    a truthiness test.  Without this, a caller who passes a non-bool
    "truthy" value to a boolean-like direct-API parameter (a non-empty
    string such as "False", or 0/1) is silently reinterpreted by Python's
    ordinary truthiness rules instead of getting a clear error -- the
    CLI itself never produces anything but a real bool here, but the
    physics layer is a reusable API and a caller bypassing the CLI can
    pass anything.
    """
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False; got {value!r}.")
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


# ======================================================================
# Composition
# ======================================================================
def check_composition(X, Z):
    """Validate a (X, Z) pair and return (X, Y, Z) with Y = 1 - X - Z."""
    X = _require_finite("X", X)
    Z = _require_finite("Z", Z)
    if not (0.0 <= X <= 1.0):
        raise ValueError(f"Hydrogen mass fraction X must lie in [0, 1]; got {X:g}.")
    if not (0.0 <= Z <= 1.0):
        raise ValueError(f"Metal mass fraction Z must lie in [0, 1]; got {Z:g}.")
    Y = 1.0 - X - Z
    if Y < -1e-12:
        raise ValueError(
            f"X + Z = {X + Z:g} exceeds 1; the helium fraction Y would be negative."
        )
    return X, max(Y, 0.0), Z


def mean_molecular_weight(X, Z):
    """
    Mean molecular weight of a fully ionised ideal gas,

        1/mu = 2X + (3/4)Y + (1/2)Z,        Y = 1 - X - Z.

    Solar composition X = 0.70, Z = 0.02 gives mu = 0.6173.
    """
    X, Y, Z = check_composition(X, Z)
    inv = 2.0 * X + 0.75 * Y + 0.5 * Z
    if inv <= 0.0:
        raise ValueError("Composition gives a non-positive particle number density.")
    return 1.0 / inv


def mean_molecular_weight_per_electron(X):
    """mu_e = 2/(1 + X) for fully ionised matter with metals of A = 2Z."""
    X = _require_finite("X", X)
    if not (0.0 <= X <= 1.0):
        raise ValueError(f"X must lie in [0, 1]; got {X:g}.")
    return 2.0 / (1.0 + X)


# ======================================================================
# Homology relations for a radiative, ideal-gas main-sequence star
# ======================================================================
def homology_exponents(nu, kappa_a, kappa_b):
    """
    Return the homology exponents for a radiative star in which

        kappa = kappa_0 rho^a T^b      (opacity law)
        eps   = eps_0   rho   T^nu     (energy generation)

    Standard homology (hydrostatic equilibrium, ideal gas, radiative
    diffusion, nuclear energy balance) then gives

        R  ~  mu^p M^q,
        L  ~  mu^(4-b) M^(3-a-b) R^(b+3a),

    with

        D = nu + 3 + b + 3a,
        p = (nu - 4 + b)/D,
        q = (nu - 1 + a + b)/D.

    Substituting R gives the pure (mu, M) exponents also returned here.

    Returns a dict with keys p_R_mu, q_R_M, e_L_mu, e_L_M.
    """
    nu = _require_finite("nu", nu)
    kappa_a = _require_finite("kappa_a", kappa_a)
    kappa_b = _require_finite("kappa_b", kappa_b)

    D = nu + 3.0 + kappa_b + 3.0 * kappa_a
    if abs(D) < 1e-8:
        raise ValueError(
            "The chosen opacity and energy-generation exponents make the "
            "homology denominator nu + 3 + b + 3a vanish; no homologous "
            "solution exists for this combination."
        )
    p = (nu - 4.0 + kappa_b) / D
    q = (nu - 1.0 + kappa_a + kappa_b) / D

    r_exp = kappa_b + 3.0 * kappa_a
    e_L_mu = (4.0 - kappa_b) + r_exp * p
    e_L_M = (3.0 - kappa_a - kappa_b) + r_exp * q
    return dict(p_R_mu=p, q_R_M=q, e_L_mu=e_L_mu, e_L_M=e_L_M, D=D)


# Opacity laws selectable by the student.
OPACITY_LAWS = {
    # name          (a,    b)
    "thomson": (0.0, 0.0),      # electron scattering, kappa = const
    "kramers": (1.0, -3.5),     # bound-free / free-free
}

# Energy-generation temperature exponents.
BURNING_NU = {
    "pp": 4.0,
    "cno": 16.0,
}


def default_burning(m_msun):
    """pp chain below 1.2 solar masses, CNO cycle above."""
    m_msun = _require_positive("mass", m_msun)
    return "pp" if m_msun < 1.2 else "cno"


def default_core_fraction(m_msun):
    """
    Fraction q_c of the stellar mass that acts as a well-mixed hydrogen
    reservoir on the main sequence.

    Stars somewhat below the Sun have small radiative cores; stars above
    about 1.2 solar masses have convective cores that grow with mass.  The
    values below are calibrated so that a 1 solar-mass track has a
    main-sequence lifetime near 10 Gyr.

    The fixed q_c = 0.15 below 1.2 Msun does NOT describe stars below
    about 0.35 Msun, which are fully convective and can therefore burn a
    far larger fraction of their hydrogen.  Tracks below TRUSTED_MASS_LO
    carry a warning for that reason.
    """
    m_msun = _require_positive("mass", m_msun)
    if m_msun <= 1.2:
        return 0.15
    return min(0.15 * (m_msun / 1.2) ** 0.22, 0.32)


# ======================================================================
# Zero-age main sequence anchor
# ======================================================================
def zams_luminosity(m_msun):
    """
    Empirical zero-age main-sequence luminosity in solar units.

    A piecewise power law anchored so that the ZAMS Sun has
    L = 0.72 L_sun.  This is a fit, not a homology result; comparing it
    with the homology prediction is one of the exercises.
    """
    m = _require_positive("mass", m_msun)
    if m < 0.43:
        return 0.72 * 0.43 ** 4.0 * (m / 0.43) ** 2.3
    if m < 2.0:
        return 0.72 * m ** 4.0
    return 0.72 * 2.0 ** 4.0 * (m / 2.0) ** 3.5


def zams_radius(m_msun):
    """
    Empirical zero-age main-sequence radius in solar units, anchored so
    that the ZAMS Sun has R = 0.89 R_sun.
    """
    m = _require_positive("mass", m_msun)
    if m < 1.0:
        return 0.89 * m ** 0.80
    return 0.89 * m ** 0.57


def effective_temperature(L_lsun, R_rsun):
    """Effective temperature in K from L and R in solar units."""
    L_lsun = _require_positive("L", L_lsun)
    R_rsun = _require_positive("R", R_rsun)
    L = L_lsun * L_sun
    R = R_rsun * R_sun
    return (L / (4.0 * np.pi * R**2 * sigma_SB)) ** 0.25


def zams_curve(m_lo=0.15, m_hi=60.0, n=200):
    """Return (M, logTeff, logL) arrays tracing the ZAMS."""
    m_lo = _require_positive("m_lo", m_lo)
    m_hi = _require_positive("m_hi", m_hi)
    if m_hi <= m_lo:
        raise ValueError(f"m_hi ({m_hi:g}) must exceed m_lo ({m_lo:g}).")
    n = _require_int("n", n, lo=2, hi=MAX_GRID_POINTS)
    m = np.geomspace(m_lo, m_hi, n)
    L = np.array([zams_luminosity(v) for v in m])
    R = np.array([zams_radius(v) for v in m])
    T = np.array([effective_temperature(a, b) for a, b in zip(L, R)])
    return m, np.log10(T), np.log10(L)


# ======================================================================
# Post-main-sequence prescriptions
# ======================================================================
def core_mass_luminosity(mc_msun):
    """
    Schematic core-mass-luminosity closure for a hydrogen shell burning
    around a degenerate helium core,

        L/L_sun = 2.3e5 (M_c/M_sun)^6 .

    A steep power law of this kind is the standard qualitative statement
    of the red-giant-branch core-mass-luminosity relation for low-mass
    stars, and it is the reason the giant-branch tip is nearly a standard
    candle.  It is NOT the Paczynski relation, which describes shell
    burning on the thermally pulsing AGB and is close to linear in M_c.
    The particular coefficient and exponent used here are a pedagogical
    fit chosen to put the solar giant-branch tip near 2500 L_sun; treat
    them as a closure, not as a calibrated published relation.

    Reasonable roughly for 0.17 < M_c/M_sun < 0.5.  Used here,
    deliberately, over a slightly wider range as a smooth closure for the
    shell-burning phase; the help file discusses the consequences.
    """
    mc_msun = _require_positive("core mass", mc_msun)
    return 2.3e5 * mc_msun ** 6.0


def hayashi_teff(m_msun, L_lsun):
    """
    Empirical fit to the Hayashi (fully convective) line,

        T_eff = 6560 K (L/L_sun)^-0.092 (M/M_sun)^0.10 .

    This reproduces T_eff ~ 4300 K at L = 100 L_sun and ~3200 K at the
    tip of the red-giant branch for a 1 solar-mass star.  It is a fit to
    detailed models, not a solution of the structure equations.
    """
    m_msun = _require_positive("mass", m_msun)
    L_lsun = _require_positive("L", L_lsun)
    return 6560.0 * L_lsun ** (-0.092) * m_msun ** 0.10


# Above this mass the helium core is not degenerate and the star crosses
# the Hertzsprung gap on a thermal timescale instead of climbing a
# degenerate red-giant branch.
M_DEGENERATE_CORE = 2.0

# Core mass at the helium flash for a degenerate helium core.
MC_HELIUM_FLASH = 0.47


def helium_flash_core_mass():
    """
    Helium-core mass at the helium flash.

    A low-mass star builds an electron-degenerate helium core, which is
    nearly isothermal and therefore ignites at a mass that hardly depends
    on the mass of the star: about 0.47 solar masses.  This is why the tip
    of the red-giant branch is such a good standard candle.
    """
    return MC_HELIUM_FLASH


def kelvin_helmholtz_time(m_msun, r_rsun, l_lsun):
    """Thermal (Kelvin-Helmholtz) timescale t_KH = G M^2 / (R L), in seconds."""
    m_msun = _require_positive("mass", m_msun)
    r_rsun = _require_positive("radius", r_rsun)
    l_lsun = _require_positive("L", l_lsun)
    return (G * (m_msun * M_sun) ** 2
            / (r_rsun * R_sun * l_lsun * L_sun))


def predicted_remnant(m_msun):
    """
    Schematic end state.  Returns (kind, mass_msun, note).

    This is a classification, not a result of the integration: the track
    stops at helium ignition and everything after it -- core helium
    burning, the AGB, thermal pulses, mass loss and envelope ejection --
    is skipped.  The white-dwarf branch uses the linear initial-final
    mass relation of Kalirai et al. (2008).  That work extended direct
    constraints DOWN to an initial mass of 1.16 Msun and combined them
    with the existing higher-mass cluster sample, which reached about
    7 Msun.  Applying the fit below 1.16 Msun or above about 7 Msun is
    therefore an extrapolation.  Later work (for example Cummings et
    al. 2018) shows that the real relation is not globally linear.
    """
    m_msun = _require_positive("mass", m_msun)
    if m_msun < 0.5:
        return ("helium white dwarf",
                0.109 * m_msun + 0.394,
                "schematic; main-sequence lifetime exceeds the age of the "
                "Universe, so no such star has yet formed")
    if m_msun < 8.0:
        mf = 0.109 * m_msun + 0.394
        kind = "carbon-oxygen white dwarf" if m_msun < 6.5 else "oxygen-neon white dwarf"
        if m_msun < 1.16:
            note = ("schematic; Kalirai et al. (2008) linear initial-final "
                    "mass relation extrapolated below its directly "
                    "constrained progenitor range")
        elif m_msun <= 7.0:
            note = ("schematic; Kalirai et al. (2008) linear initial-final "
                    "mass relation (later work finds a non-linear relation)")
        else:
            note = ("schematic; Kalirai et al. (2008) linear initial-final "
                    "mass relation extrapolated above its directly "
                    "constrained progenitor range")
        return (kind, mf, note)
    if m_msun < 20.0:
        return ("neutron star", 1.4,
                "schematic classification; the remnant mass is set by the "
                "explosion and the nuclear EOS, neither of which is modelled")
    return ("black hole", 0.2 * m_msun,
            "schematic classification; very rough, and strongly dependent "
            "on mass loss and metallicity")


# ======================================================================
# Main-sequence + post-main-sequence track
# ======================================================================
def _mu_effective(mu_core, mu_env, w):
    """
    One-parameter closure for the luminosity-weighted mean molecular
    weight of a star with a helium-enriched core inside an unchanged
    envelope:

        mu_eff = mu_env^(1-w) mu_core^w .

    w = 0 ignores the core, w = 1 treats the whole star as core material.
    The default w = 0.36 is calibrated together with q_c so that a
    1 solar-mass track passes through L = 1 L_sun at an age of 4.57 Gyr
    and reaches L = 2.2 L_sun at the terminal-age main sequence, matching
    standard solar models.
    """
    return mu_env ** (1.0 - w) * mu_core ** w


def _radius_response(s_burn, expansion):
    """
    Empirical main-sequence radius law,

        R(t) = R_ZAMS * [ 1 + (f_exp - 1) s^(3/2) ],
        s = 1 - X_c/X_0  (fraction of the central hydrogen consumed).

    Homology for a chemically homogeneous star predicts R ~ mu^p with
    p = (nu - 4 + b)/D, which for the pp chain with electron scattering is
    exactly zero: homology predicts no expansion at all.  Detailed models
    show that a solar-type star grows by about 70 per cent in radius across
    the main sequence, because the growing molecular-weight gradient is not
    homologous.  The default f_exp = 1.7 reproduces that growth; the
    --homology switch discards this law and uses the homology exponent
    instead, so the two can be compared directly.
    """
    return 1.0 + (expansion - 1.0) * max(s_burn, 0.0) ** 1.5


def integrate_track(m_msun=1.0, X=0.70, Z=0.02,
                    qc=None, burning=None, opacity="thomson",
                    core_weight=0.36, expansion=1.70,
                    core_efficiency=0.75,
                    n_ms=3000, n_post=3000,
                    t_max_gyr=15.0, x_end=1.0e-3,
                    include_postms=True, homology_zams=False):
    """
    Integrate one stellar evolution track.

    Main sequence (integrated ODE)
    ------------------------------
        dX_c/dt = -L(t) / (EPS_NUC * q_c * M)

    with L obtained at each step from the homology scaling

        L(t) = L_ZAMS * (mu_eff(t)/mu_eff(0))^e_L_mu,
        R(t) = R_ZAMS * (mu_eff(t)/mu_eff(0))^p_R_mu.

    Post main sequence (integrated ODE + prescriptions)
    ---------------------------------------------------
        dM_c/dt = L(t) / (EPS_NUC * X_env)

    with L a smooth blend of the terminal-age main-sequence luminosity and
    the core-mass-luminosity relation, and T_eff blended from the TAMS
    value onto the empirical Hayashi line.

    Returns a dict of arrays plus a summary dict.
    """
    # ---------------- validation ----------------
    m_msun = _require_positive("mass", m_msun)
    if m_msun < 0.08:
        raise ValueError(
            f"mass = {m_msun:g} Msun is below the hydrogen-burning limit "
            "(about 0.08 Msun); this program models hydrogen-burning stars."
        )
    if m_msun > 120.0:
        raise ValueError(
            f"mass = {m_msun:g} Msun exceeds the 120 Msun limit of this "
            "simple model, where radiation pressure and mass loss dominate."
        )
    X, Y, Z = check_composition(X, Z)
    if X <= 0.0:
        raise ValueError("A hydrogen-burning track needs X > 0.")
    include_postms = _require_bool("include_postms", include_postms)
    homology_zams = _require_bool("homology_zams", homology_zams)

    core_weight = _require_finite("core_weight", core_weight)
    if not (0.0 <= core_weight <= 1.0):
        raise ValueError(f"core_weight must lie in [0, 1]; got {core_weight:g}.")
    core_efficiency = _require_finite("core_efficiency", core_efficiency)
    if not (0.0 < core_efficiency <= 1.0):
        raise ValueError(
            f"core_efficiency must lie in (0, 1]; got {core_efficiency:g}."
        )
    expansion = _require_positive("expansion", expansion)
    if expansion < 1.0:
        raise ValueError(
            f"expansion = {expansion:g} is below 1, but the parameter is "
            "defined as the growth factor R_TAMS/R_ZAMS and detailed models "
            "show main-sequence stars expand.  Use 1.0 for no expansion."
        )
    if expansion > 10.0:
        raise ValueError(
            f"expansion = {expansion:g} is unphysically large for the main "
            "sequence; values between 1 and 3 are sensible."
        )

    # Accepted range and trustworthy range are not the same thing.
    warnings = []
    if m_msun < TRUSTED_MASS_LO:
        warnings.append(
            f"M = {m_msun:g} Msun is below {TRUSTED_MASS_LO:g} Msun, where "
            "stars are fully convective.  The fixed hydrogen-reservoir "
            "fraction q_c of this model understates the available fuel, so "
            "the lifetime is a lower limit only."
        )
    elif m_msun > TRUSTED_MASS_HI:
        warnings.append(
            f"M = {m_msun:g} Msun is above {TRUSTED_MASS_HI:g} Msun, where "
            "mass loss, radiation pressure and convection dominate.  The "
            "track is qualitative at best; treat it as a demonstration of "
            "where the approximations fail."
        )

    n_ms = _require_int("n_ms", n_ms, lo=MIN_TRACK_STEPS, hi=MAX_TRACK_STEPS)
    n_post = _require_int("n_post", n_post, lo=MIN_TRACK_STEPS, hi=MAX_TRACK_STEPS)
    t_max_gyr = _require_positive("t_max_gyr", t_max_gyr)
    x_end = _require_finite("x_end", x_end)
    if not (0.0 <= x_end < X):
        raise ValueError(
            f"x_end must satisfy 0 <= x_end < X; got x_end = {x_end:g}, X = {X:g}."
        )

    if qc is None:
        qc = default_core_fraction(m_msun)
    qc = _require_finite("qc", qc)
    if not (0.0 < qc <= 1.0):
        raise ValueError(f"qc must lie in (0, 1]; got {qc:g}.")

    if burning is None:
        burning = default_burning(m_msun)
    if burning not in BURNING_NU:
        raise ValueError(
            f"burning must be one of {sorted(BURNING_NU)}; got {burning!r}."
        )
    if opacity not in OPACITY_LAWS:
        raise ValueError(
            f"opacity must be one of {sorted(OPACITY_LAWS)}; got {opacity!r}."
        )

    nu = BURNING_NU[burning]
    ka, kb = OPACITY_LAWS[opacity]
    exps = homology_exponents(nu, ka, kb)
    e_L_mu = exps["e_L_mu"]
    p_R_mu = exps["p_R_mu"]

    # ---------------- ZAMS anchor ----------------
    mu_env = mean_molecular_weight(X, Z)
    if homology_zams:
        # Pure homology, normalised on the ZAMS Sun of the same composition.
        L0 = 0.72 * m_msun ** exps["e_L_M"]
        R0 = 0.89 * m_msun ** exps["q_R_M"]
    else:
        L0 = zams_luminosity(m_msun)
        R0 = zams_radius(m_msun)

    M_kg = m_msun * M_sun
    reservoir_kg = qc * M_kg          # well-mixed hydrogen reservoir

    mu0 = _mu_effective(mu_env, mu_env, core_weight)   # = mu_env

    def L_of_Xc(Xc):
        """Return (L, R, mu_eff) in solar units at central abundance Xc."""
        mu_core = mean_molecular_weight(Xc, Z)
        mu_eff = _mu_effective(mu_core, mu_env, core_weight)
        L_l = L0 * (mu_eff / mu0) ** e_L_mu
        if homology_zams:
            R_r = R0 * (mu_eff / mu0) ** p_R_mu
        else:
            R_r = R0 * _radius_response(1.0 - Xc / X, expansion)
        return L_l, R_r, mu_eff

    # Estimated main-sequence lifetime; used only to pick the timestep.
    # A track that will be stopped at t_max is resolved over the interval
    # actually integrated, so a truncated run is not coarser than a
    # complete one.
    t_ms_est = EPS_NUC * (X - x_end) * reservoir_kg / (L0 * L_sun)
    dt = min(t_ms_est, t_max_gyr * GYR) / n_ms

    t_list, Xc_list, L_list, R_list, T_list, mu_list, mc_list = [], [], [], [], [], [], []
    phase_list = []

    t = 0.0
    Xc = X
    steps = 0
    truncated = False
    t_stop_s = t_max_gyr * GYR

    while True:
        L_l, R_r, mu_eff = L_of_Xc(Xc)
        t_list.append(t)
        Xc_list.append(Xc)
        L_list.append(L_l)
        R_list.append(R_r)
        T_list.append(effective_temperature(L_l, R_r))
        mu_list.append(mu_eff)
        mc_list.append(core_efficiency * qc * m_msun * (1.0 - Xc / X))
        phase_list.append(0)                      # 0 = main sequence

        if Xc <= x_end:
            break
        if t >= t_stop_s:
            truncated = True
            break
        steps += 1
        if steps > MAX_TRACK_STEPS:
            raise RuntimeError(
                "Main-sequence integration exceeded the internal step limit; "
                "reduce --n_ms or raise --x_end."
            )

        # RK4 on dXc/dt = -L(Xc)/(EPS_NUC * reservoir)
        def deriv(xc):
            xc = min(max(xc, 0.0), 1.0)
            return -L_of_Xc(xc)[0] * L_sun / (EPS_NUC * reservoir_kg)

        k1 = deriv(Xc)
        k2 = deriv(Xc + 0.5 * dt * k1)
        k3 = deriv(Xc + 0.5 * dt * k2)
        k4 = deriv(Xc + dt * k3)
        Xc_next = Xc + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if not math.isfinite(Xc_next):
            raise RuntimeError(
                "The main-sequence integration produced a non-finite state; "
                "try a larger --n_ms."
            )
        if Xc_next >= Xc:
            raise RuntimeError(
                "The main-sequence integration failed to consume hydrogen; "
                "check the composition and core fraction."
            )

        if Xc_next <= x_end and t + dt * (Xc - x_end) / (Xc - Xc_next) <= t_stop_s:
            # Interpolate the final partial step so the track ends exactly
            # at the requested central hydrogen abundance.
            frac = (Xc - x_end) / (Xc - Xc_next)
            t += frac * dt
            Xc = x_end
        elif t + dt >= t_stop_s:
            # Interpolate to t_max exactly, so a truncated track always
            # stops at the age the student asked for.
            frac = (t_stop_s - t) / dt
            Xc = Xc + frac * (Xc_next - Xc)
            t = t_stop_s
            truncated = True
        else:
            t += dt
            Xc = Xc_next

    reached_tams = not truncated
    t_stop_reached = t                       # age of the last main-sequence point
    t_ms = t if reached_tams else float("nan")
    Xc_end = Xc

    # These are terminal-age main-sequence quantities and are only defined
    # if the star actually reached the terminal-age main sequence.
    L_tams = L_list[-1] if reached_tams else float("nan")
    T_tams = T_list[-1] if reached_tams else float("nan")
    R_tams = R_list[-1] if reached_tams else float("nan")
    # Must match the array formula used inside the main-sequence loop
    # (mc_list.append(... * (1.0 - Xc/X)) above) evaluated at Xc = Xc_end,
    # or mc_tams disagrees with the star's own last plotted core mass by an
    # amount that depends on x_end -- a discontinuity, not a rounding error.
    mc_tams = (core_efficiency * qc * m_msun * (1.0 - Xc_end / X)
               if reached_tams else float("nan"))
    mc_end = mc_list[-1]

    phase_end = ("main sequence, integration stopped at t_max"
                 if truncated else "TAMS")
    if truncated:
        warnings.append(
            f"the star was still burning hydrogen at t_max = {t_max_gyr:g} "
            "Gyr, so the track was stopped there.  It has no terminal-age "
            "main sequence, no post-main-sequence phase and no meaningful "
            "main-sequence lifetime; raise --t_max to follow it further."
        )

    # ---------------- post-main-sequence ----------------
    #
    # Two regimes, separated at M_DEGENERATE_CORE = 2 Msun:
    #
    #   M <= 2 : the helium core is electron degenerate.  The core grows by
    #            hydrogen shell burning, dM_c/dt = L/(EPS_NUC X_env), and the
    #            luminosity follows the core-mass-luminosity relation.  The
    #            star climbs the red-giant branch until the helium flash at
    #            M_c = 0.47 Msun.
    #
    #   M >  2 : the helium core is not degenerate.  The star crosses the
    #            Hertzsprung gap at nearly constant luminosity on the
    #            Kelvin-Helmholtz timescale t_KH = G M^2/(R L) and ignites
    #            helium when it reaches the Hayashi line.
    #
    post_ok = bool(include_postms and reached_tams)
    t_he = float("nan")
    L_tip = float("nan")
    mc_ign = float("nan")
    t_cross = float("nan")
    regime = "none"
    # True only once helium has actually ignited (either at the tip of the
    # degenerate red-giant branch, or at the top of a Hertzsprung-gap
    # crossing).  post_ms alone is not enough to test this: it is also true
    # when the envelope is exhausted before the flash (a helium white
    # dwarf) and when a track is stopped at t_max, neither of which ends at
    # helium ignition.  driver_sev.py and plot_sev.py must use this flag,
    # not post_ms, to decide whether to report/annotate "helium ignition".
    helium_ignition = False

    if post_ok and m_msun <= M_DEGENERATE_CORE:
        regime = "degenerate red-giant branch"
        # The helium core cannot swallow the whole star: at most about 70
        # per cent of the mass can end up in the core before the hydrogen
        # envelope is gone.
        mc_cap = 0.70 * m_msun
        mc_ign = helium_flash_core_mass()
        flashes = mc_ign <= mc_cap
        if not flashes:
            mc_ign = mc_cap
        if mc_ign <= mc_tams:
            # A rare corner: the burned core already exceeds the flash mass.
            post_ok = False
            warnings.append(
                "no post-main-sequence phase is reported: the core mass "
                "already at the terminal-age main sequence "
                f"({mc_tams:.4f} Msun) is at or above the helium-flash core "
                f"mass ({mc_ign:.4f} Msun) used by this schematic model, so "
                "there is no growing-core phase left to integrate."
            )
        else:
            X_env = X
            mc_hayashi = min(mc_tams + 0.40 * (mc_ign - mc_tams), mc_ign)

            def L_post(mc):
                """Smooth blend of the TAMS luminosity and the CMLR."""
                l_shell = core_mass_luminosity(mc)
                k = 4.0
                return (L_tams ** k + l_shell ** k) ** (1.0 / k)

            def teff_post(mc, L_l):
                t_h = hayashi_teff(m_msun, L_l)
                if mc_hayashi <= mc_tams:
                    return t_h
                u = (mc - mc_tams) / (mc_hayashi - mc_tams)
                u = min(max(u, 0.0), 1.0)
                u = u * u * (3.0 - 2.0 * u)        # smoothstep in core mass
                return 10.0 ** ((1.0 - u) * math.log10(T_tams)
                                + u * math.log10(t_h))

            dmc = (mc_ign - mc_tams) / n_post
            mc = mc_tams
            pstep = 0
            while True:
                L_l = L_post(mc)
                T_e = teff_post(mc, L_l)
                R_r = math.sqrt(L_l * L_sun
                                / (4.0 * np.pi * sigma_SB * T_e**4)) / R_sun
                if pstep > 0:                      # TAMS point already stored
                    t_list.append(t)
                    Xc_list.append(0.0)
                    L_list.append(L_l)
                    R_list.append(R_r)
                    T_list.append(T_e)
                    mu_list.append(mu_list[-1])
                    mc_list.append(mc)
                    phase_list.append(1 if mc < mc_hayashi else 2)

                if mc >= mc_ign:
                    break
                pstep += 1
                if pstep > MAX_TRACK_STEPS:
                    raise RuntimeError(
                        "Post-main-sequence integration exceeded the step limit."
                    )

                # RK4 on dt/dM_c = EPS_NUC X_env M_sun / L
                def dtdmc(mc_val):
                    return EPS_NUC * X_env * M_sun / (L_post(mc_val) * L_sun)

                j1 = dtdmc(mc)
                j2 = dtdmc(mc + 0.5 * dmc)
                j3 = dtdmc(mc + 0.5 * dmc)
                j4 = dtdmc(mc + dmc)
                t += (dmc / 6.0) * (j1 + 2 * j2 + 2 * j3 + j4)
                mc += dmc

            t_he = t
            L_tip = L_list[-1]
            phase_end = ("helium flash at the tip of the red-giant branch"
                         if flashes else
                         "core-mass cap of this schematic model reached "
                         "before the helium core reached the flash mass "
                         "(helium white dwarf)")
            helium_ignition = bool(flashes)

    elif post_ok:
        regime = "Hertzsprung-gap crossing"
        mc_ign = mc_tams
        t_cross = kelvin_helmholtz_time(m_msun, R_tams, L_tams)
        T_hay = hayashi_teff(m_msun, L_tams)
        if T_hay >= T_tams:
            # Already at or beyond the Hayashi line; nothing to cross.
            post_ok = False
            warnings.append(
                "no post-main-sequence phase is reported: the "
                f"terminal-age main-sequence effective temperature "
                f"({T_tams:.0f} K) is already at or below the Hayashi-line "
                f"temperature this model predicts for the star ({T_hay:.0f} "
                "K), so there is no Hertzsprung-gap crossing left to "
                "integrate."
            )
        else:
            log_T0, log_T1 = math.log10(T_tams), math.log10(T_hay)
            for i in range(1, n_post + 1):
                u = i / n_post
                t_i = t_ms + u * t_cross
                L_l = L_tams
                T_e = 10.0 ** ((1.0 - u) * log_T0 + u * log_T1)
                R_r = math.sqrt(L_l * L_sun
                                / (4.0 * np.pi * sigma_SB * T_e**4)) / R_sun
                t_list.append(t_i)
                Xc_list.append(0.0)
                L_list.append(L_l)
                R_list.append(R_r)
                T_list.append(T_e)
                mu_list.append(mu_list[-1])
                mc_list.append(mc_tams)
                phase_list.append(1)
            t = t_ms + t_cross
            t_he = t
            L_tip = L_list[-1]
            phase_end = "helium ignition at the base of the giant branch"
            helium_ignition = True

    t_arr = np.asarray(t_list, dtype=float)
    L_arr = np.asarray(L_list, dtype=float)
    R_arr = np.asarray(R_list, dtype=float)
    T_arr = np.asarray(T_list, dtype=float)
    Xc_arr = np.asarray(Xc_list, dtype=float)
    mu_arr = np.asarray(mu_list, dtype=float)
    mc_arr = np.asarray(mc_list, dtype=float)
    ph_arr = np.asarray(phase_list, dtype=int)

    kind, mrem, note = predicted_remnant(m_msun)

    # predicted_remnant() is a standalone, mass-only a-priori classifier
    # (its own switch to "carbon-oxygen white dwarf" happens at 0.5 Msun),
    # independent of what THIS track's own post-main-sequence integration
    # actually found.  For initial masses roughly 0.5-0.67 Msun those two
    # disagree: the degenerate-RGB branch above finds the envelope
    # exhausted before the core reaches the helium-flash mass (a helium
    # white dwarf) at the same time predicted_remnant() reports a
    # carbon-oxygen white dwarf -- a genuine, simultaneous contradiction in
    # one summary, not merely a labelling gap.  When a full track was
    # actually integrated far enough to settle the question, that computed
    # outcome must take precedence over the generic a-priori classification
    # for this run's own remnant fields.
    # remnant_basis tells every consumer (driver, plot, CSV) which of the
    # two possible origins this run's remnant fields actually came from,
    # so none of them has to repeat (or contradict) the other's wording:
    # a fixed, always-true sentence describing predicted_remnant() cannot
    # be correct for both origins, because only one of them is "a
    # classification from the initial mass, not an integrated result".
    remnant_basis = "mass-only schematic classification"
    if post_ok and regime == "degenerate red-giant branch" and not flashes:
        kind = "helium white dwarf"
        mrem = mc_ign
        remnant_basis = "this track's own post-main-sequence integration"
        # The 0.70*m_msun cap is bookkeeping in this schematic model's
        # core-mass variable, not an integrated envelope-ejection
        # calculation -- no envelope-mass-loss physics is modeled here,
        # so the wording below deliberately says "reached the cap built
        # into this model" rather than "found the envelope exhausted",
        # to avoid overstating what was actually computed.
        note = ("this track's own post-main-sequence integration reached "
                "the hydrogen-envelope core-mass cap built into this "
                "schematic model (at most 0.70 of the initial mass can "
                "join the core) before the core reached the helium-flash "
                "mass; this supersedes the generic mass-only "
                "classification for this run, but is itself a bookkeeping "
                "endpoint, not an integrated envelope-ejection result")

    summary = dict(
        m_msun=m_msun, X=X, Y=Y, Z=Z, qc=qc, expansion=expansion,
        homology_zams=bool(homology_zams),
        burning=burning, nu=nu, opacity=opacity,
        kappa_a=ka, kappa_b=kb,
        core_weight=core_weight,
        e_L_mu=e_L_mu, p_R_mu=p_R_mu,
        e_L_M=exps["e_L_M"], q_R_M=exps["q_R_M"],
        mu_env=mu_env,
        L_zams=L_arr[0], R_zams=R_arr[0], T_zams=T_arr[0],
        L_tams=L_tams, T_tams=T_tams,
        t_ms_gyr=t_ms / GYR,
        t_ms_est_gyr=t_ms_est / GYR,
        truncated=truncated,
        reached_tams=reached_tams,
        t_stop_gyr=t_stop_reached / GYR,
        Xc_end=Xc_end,
        mc_end=mc_end,
        t_max_gyr=t_max_gyr,
        post_ms=post_ok,
        core_efficiency=core_efficiency,
        post_regime=(regime if post_ok else "none"),
        t_cross_gyr=(t_cross / GYR if post_ok and math.isfinite(t_cross)
                     else float("nan")),
        mc_tams=mc_tams, mc_ign=(mc_ign if post_ok else float("nan")),
        t_he_gyr=(t_he / GYR if post_ok else float("nan")),
        t_post_gyr=((t_he - t_ms) / GYR if post_ok else float("nan")),
        L_tip=L_tip,
        t_total_gyr=t_arr[-1] / GYR,
        phase_end=phase_end,
        helium_ignition=helium_ignition,
        remnant_kind=kind, remnant_msun=mrem, remnant_note=note,
        remnant_basis=remnant_basis,
        n_points=t_arr.size,
        warnings=warnings,
        model_version=MODEL_VERSION,
        build_id=BUILD_ID,
    )

    return dict(
        kind="track",
        t=t_arr, L=L_arr, R=R_arr, Teff=T_arr,
        Xc=Xc_arr, mu=mu_arr, Mcore=mc_arr, phase=ph_arr,
        summary=summary,
    )


# ======================================================================
# HR-diagram grid and isochrones
# ======================================================================
def turnoff_mass(masses, t_ms_gyr, age_gyr):
    """
    Main-sequence turn-off mass at a given age, from the computed t_MS(M).

    Interpolates log t_MS against log M over the tracks that actually
    reached the terminal-age main sequence.  Returns None if the age lies
    outside the range spanned by the mass grid, which is the honest answer
    for a sparse grid.
    """
    age_gyr = _require_positive("age_gyr", age_gyr)
    pairs = [(m, t) for m, t in zip(masses, t_ms_gyr)
             if t is not None and math.isfinite(t) and t > 0.0 and m > 0.0]
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda q: q[1])                 # increasing lifetime
    lt = [math.log10(t) for _, t in pairs]
    lm = [math.log10(m) for m, _ in pairs]
    la = math.log10(age_gyr)
    if la < lt[0] or la > lt[-1]:
        return None
    return float(10.0 ** np.interp(la, lt, lm))


def build_hr_grid(masses, isochrone_gyr=None, **track_kwargs):
    """
    Run integrate_track for each mass and, optionally, build isochrones by
    interpolating every track to a set of ages.

    Isochrone points carry the evolutionary phase of the interpolated
    point, so that a turn-off can be identified unambiguously rather than
    guessed from a sparse mass list.  The isochrones are pedagogical
    interpolants through a handful of simplified tracks, not a substitute
    for a modern isochrone library.
    """
    masses = [ _require_positive("mass", m) for m in masses ]
    if not masses:
        raise ValueError("At least one mass is required for the HR grid.")
    if len(masses) > MAX_MASSES:
        raise ValueError(
            f"At most {MAX_MASSES} masses may be requested; got {len(masses)}."
        )
    if len(set(masses)) != len(masses):
        raise ValueError("The mass list contains duplicates.")

    # Normalize isochrone_gyr to a plain Python list of validated floats
    # up front, before any expensive track integration starts, rather
    # than accepting whatever iterable was passed and discovering a
    # problem only partway through the isochrone loop below.  This also
    # fixes three direct-API correctness gaps together: len(isochrone_gyr)
    # raised TypeError for a generator; the later "if isochrone_gyr:"
    # truthiness check raised the ambiguous-truth-value ValueError for a
    # NumPy array; and duplicate ages were silently accepted, producing
    # redundant identical isochrones (unlike duplicate masses, which are
    # explicitly refused above).
    if isochrone_gyr is not None:
        isochrone_gyr = [_require_positive("isochrone age", a)
                         for a in isochrone_gyr]
        if len(isochrone_gyr) > MAX_ISOCHRONES:
            raise ValueError(
                f"At most {MAX_ISOCHRONES} isochrone ages may be requested; "
                f"got {len(isochrone_gyr)}."
            )
        if len(set(isochrone_gyr)) != len(isochrone_gyr):
            raise ValueError("The isochrone age list contains duplicates.")

    tracks = []
    for m in sorted(masses):
        tracks.append(integrate_track(m_msun=m, **track_kwargs))

    isochrones = []
    grid_warnings = []
    if isochrone_gyr:
        lifetimes = [tr["summary"]["t_ms_gyr"] for tr in tracks]
        grid_masses = [tr["summary"]["m_msun"] for tr in tracks]
        for age in isochrone_gyr:
            pts = []
            for tr in tracks:
                t_gyr = tr["t"] / GYR
                if t_gyr[0] <= age <= t_gyr[-1]:
                    logT = float(np.interp(age, t_gyr, np.log10(tr["Teff"])))
                    logL = float(np.interp(age, t_gyr, np.log10(tr["L"])))
                    j = int(np.searchsorted(t_gyr, age, side="right")) - 1
                    j = min(max(j, 0), tr["phase"].size - 1)
                    phase = int(tr["phase"][j])
                    tms = tr["summary"]["t_ms_gyr"]
                    on_ms = bool(phase == 0)
                    pts.append((tr["summary"]["m_msun"], logT, logL,
                                phase, on_ms, tms))
            if len(pts) >= 2:
                isochrones.append(dict(
                    age_gyr=age, points=pts,
                    turnoff_mass=turnoff_mass(grid_masses, lifetimes, age),
                ))
            else:
                # A requested isochrone age that no track (or only one
                # track) spans is silently unusable -- report why instead
                # of just shrinking n_isochrones with no explanation.
                grid_warnings.append(
                    f"isochrone at age = {age:g} Gyr was omitted: only "
                    f"{len(pts)} of {len(tracks)} tracks in the mass grid "
                    "span that age, and at least 2 are needed to draw an "
                    "isochrone.  Widen the mass range or choose an age "
                    "within the lifetimes actually spanned by this grid."
                )

    m_z, logT_z, logL_z = zams_curve()

    for tr in tracks:
        for w in tr["summary"]["warnings"]:
            grid_warnings.append(f"M = {tr['summary']['m_msun']:g} Msun: {w}")

    summary = dict(
        n_tracks=len(tracks),
        masses=[tr["summary"]["m_msun"] for tr in tracks],
        lifetimes_gyr=[tr["summary"]["t_ms_gyr"] for tr in tracks],
        reached_tams=[tr["summary"]["reached_tams"] for tr in tracks],
        stop_gyr=[tr["summary"]["t_stop_gyr"] for tr in tracks],
        totals_gyr=[tr["summary"]["t_total_gyr"] for tr in tracks],
        n_isochrones=len(isochrones),
        isochrone_ages=[iso["age_gyr"] for iso in isochrones],
        isochrone_turnoffs=[iso["turnoff_mass"] for iso in isochrones],
        t_max_gyr=tracks[0]["summary"]["t_max_gyr"],
        X=tracks[0]["summary"]["X"], Z=tracks[0]["summary"]["Z"],
        core_weight=tracks[0]["summary"]["core_weight"],
        warnings=grid_warnings,
        model_version=MODEL_VERSION,
        build_id=BUILD_ID,
    )
    return dict(kind="hr", tracks=tracks, isochrones=isochrones,
                zams=(m_z, logT_z, logL_z), summary=summary)


# ======================================================================
# Degenerate equations of state
# ======================================================================
# Below this relativity parameter x = p_F/(m c), FermiGasEOS.pressure()
# and .energy_density() switch from their closed forms (which cancel two
# nearly-equal terms and lose essentially all significant digits well
# before x reaches 1e-4) to the exact small-x Taylor series, which agrees
# with the closed form to better than 1e-10 relative accuracy at this
# threshold and only improves below it (see the two methods' docstrings).
_FERMI_SMALL_X = 0.05


class FermiGasEOS:
    """
    Ideal completely degenerate Fermi gas, exact special-relativistic form.

    With x = p_F/(m c) the relativity parameter and A = pi m^4 c^5/(3 h^3):

        P   = A [ x(2x^2 - 3) sqrt(1+x^2) + 3 asinh(x) ]
        eps = A [ 3x(2x^2 + 1) sqrt(1+x^2) - 3 asinh(x) ]   (energy density,
                                                             rest mass included)
        n   = (8 pi / 3) (m c / h)^3 x^3
        dP/dx = A * 8 x^4 / sqrt(1 + x^2)

    For a white dwarf the pressure comes from electrons (m = m_e) while the
    mass density comes from the nucleons, rho = mu_e m_u n_e.  For a neutron
    star both come from the neutrons (m = m_n, mu_e -> 1, m_u -> m_n).
    """

    def __init__(self, particle_mass, mass_per_particle):
        self.m = particle_mass
        self.mass_per_particle = mass_per_particle
        self.A = np.pi * particle_mass**4 * c**5 / (3.0 * h_pl**3)
        self.n0 = (8.0 * np.pi / 3.0) * (particle_mass * c / h_pl) ** 3

    def pressure(self, x):
        """
        P/A = x(2x^2-3)*sqrt(1+x^2) + 3*asinh(x), analytically exact but
        formed as a difference of two terms that are nearly equal for
        x well below 1 (both approach 3x as x -> 0).  IEEE double
        arithmetic cancels essentially all significant digits there --
        by x=1e-4 the direct evaluation underflows to exactly 0, and by
        x=1e-5 it goes slightly negative, which is unphysical for a
        pressure.  Below _FERMI_SMALL_X the exact Taylor series in x
        (whose leading terms are (8/5)x^5 - (4/7)x^7 + (1/3)x^9 - ...,
        confirmed against a symbolic expansion of the closed form) is
        used instead; it agrees with the closed form to better than
        1e-10 relative accuracy everywhere it is used, and is what the
        closed form itself would give at infinite precision.
        """
        x = np.asarray(x, dtype=float)
        small = x < _FERMI_SMALL_X
        if np.any(small):
            xs = x[small] if x.shape else x
            x2 = xs * xs
            series = xs**5 * (1.6 + x2 * (-4.0 / 7.0 + x2 * (1.0 / 3.0
                              + x2 * (-5.0 / 22.0 + x2 * 35.0 / 208.0))))
            if x.shape:
                out = np.empty_like(x)
                s = np.sqrt(1.0 + x * x)
                out[~small] = (x[~small] * (2.0 * x[~small]**2 - 3.0) * s[~small]
                              + 3.0 * np.arcsinh(x[~small]))
                out[small] = series
                return self.A * out
            return self.A * series
        s = np.sqrt(1.0 + x * x)
        return self.A * (x * (2.0 * x * x - 3.0) * s + 3.0 * np.arcsinh(x))

    def dP_dx(self, x):
        x = np.asarray(x, dtype=float)
        return self.A * 8.0 * x**4 / np.sqrt(1.0 + x * x)

    def number_density(self, x):
        return self.n0 * np.asarray(x, dtype=float) ** 3

    def rest_mass_density(self, x):
        return self.mass_per_particle * self.number_density(x)

    def energy_density(self, x):
        """
        Total energy density including rest mass, in J/m^3.  Same
        small-x cancellation problem as pressure() (both terms approach
        3x as x -> 0), with the same series-based remedy below
        _FERMI_SMALL_X: eps/A = 8x^3 + (12/5)x^5 - (3/7)x^7 + ...
        """
        x = np.asarray(x, dtype=float)
        small = x < _FERMI_SMALL_X
        if np.any(small):
            xs = x[small] if x.shape else x
            x2 = xs * xs
            series = xs**3 * (8.0 + x2 * (2.4 + x2 * (-3.0 / 7.0
                              + x2 * (1.0 / 6.0 + x2 * (-15.0 / 176.0
                              + x2 * 21.0 / 416.0)))))
            if x.shape:
                out = np.empty_like(x)
                s = np.sqrt(1.0 + x * x)
                out[~small] = (3.0 * x[~small] * (2.0 * x[~small]**2 + 1.0) * s[~small]
                              - 3.0 * np.arcsinh(x[~small]))
                out[small] = series
                return self.A * out
            return self.A * series
        s = np.sqrt(1.0 + x * x)
        return self.A * (3.0 * x * (2.0 * x * x + 1.0) * s - 3.0 * np.arcsinh(x))

    def mass_energy_density(self, x):
        """eps/c^2, the density that sources gravity in the TOV equations."""
        return self.energy_density(x) / c**2

    def sound_speed_ratio(self, x):
        """
        Adiabatic sound speed as a fraction of c.

        For the ideal Fermi gas dP/dx = 8A x^4/sqrt(1+x^2) and
        d(eps)/dx = 24 A x^2 sqrt(1+x^2), so

            (c_s/c)^2 = x^2 / (3 (1 + x^2)),

        which rises monotonically to 1/sqrt(3): an ideal Fermi gas is
        always causal.
        """
        x = np.asarray(x, dtype=float)
        return np.sqrt(x * x / (3.0 * (1.0 + x * x)))

    def x_from_density(self, rho):
        """Invert rho -> x for the rest-mass density."""
        rho = _require_positive("central density", rho)
        return (rho / (self.mass_per_particle * self.n0)) ** (1.0 / 3.0)


# Nuclear saturation density, about 0.16 baryons per cubic femtometre.
RHO_NUCLEAR = 2.7e17          # kg m^-3


class PolytropeEOS:
    """
    Stiff polytrope  P = K rho^Gamma  with rest-mass density rho as the
    integration variable and total energy density

        eps/c^2 = rho + P/((Gamma - 1) c^2).

    Rather than asking for K in SI units, the constant is set by a
    dimensionless stiffness

        p_nuc = P(rho_nuc) / (rho_nuc c^2),

    the pressure at nuclear saturation density expressed as a fraction of
    the rest-mass energy density there.  Larger p_nuc means a stiffer
    equation of state, a larger radius and a larger maximum mass.  The
    default p_nuc = 0.04 with Gamma = 2.5 gives a maximum mass of about
    2.2 solar masses at a radius near 11.6 km, comparable with the
    heaviest precisely measured neutron stars, and stays causal.
    """

    def __init__(self, p_nuc=0.04, gamma=2.5, K=None):
        self.gamma = _require_finite("gamma", gamma)
        if not (1.0 < self.gamma <= 5.0):
            raise ValueError(
                f"gamma must lie in (1, 5]; got {self.gamma:g}."
            )
        if K is not None:
            self.K = _require_positive("K", K)
            self.p_nuc = self.K * RHO_NUCLEAR ** self.gamma / (RHO_NUCLEAR * c**2)
        else:
            self.p_nuc = _require_positive("p_nuc", p_nuc)
            if self.p_nuc > 1.0:
                raise ValueError(
                    f"p_nuc = {self.p_nuc:g} exceeds 1: the pressure at "
                    "nuclear density would exceed the rest-mass energy "
                    "density there.  Causality is strictly a condition on "
                    "the sound speed, dP/d(eps) <= c^2, but P > rho c^2 at "
                    "saturation density is far outside anything nuclear "
                    "physics supports, so it is refused here.  Use the "
                    "reported c_s/c to test causality properly."
                )
            self.K = self.p_nuc * RHO_NUCLEAR * c**2 / RHO_NUCLEAR ** self.gamma
        self.mass_per_particle = m_n

    def pressure(self, rho):
        return self.K * np.asarray(rho, dtype=float) ** self.gamma

    def dP_dx(self, rho):
        return self.K * self.gamma * np.asarray(rho, dtype=float) ** (self.gamma - 1.0)

    def rest_mass_density(self, rho):
        return np.asarray(rho, dtype=float)

    def mass_energy_density(self, rho):
        rho = np.asarray(rho, dtype=float)
        return rho + self.pressure(rho) / ((self.gamma - 1.0) * c**2)

    def sound_speed_ratio(self, rho):
        """
        c_s/c from c_s^2 = dP/d(eps).  For this polytrope

            dP/drho   = Gamma K rho^(Gamma-1),
            deps/drho = c^2 + Gamma K rho^(Gamma-1)/(Gamma - 1),

        so a sufficiently stiff polytrope becomes acausal (c_s > c) above
        some density.  Checking where that happens is one of the exercises.
        """
        rho = np.asarray(rho, dtype=float)
        dPdrho = self.gamma * self.K * rho ** (self.gamma - 1.0)
        depsdrho = c**2 + dPdrho / (self.gamma - 1.0)
        return np.sqrt(dPdrho / depsdrho)

    def x_from_density(self, rho):
        return _require_positive("central density", rho)


def make_eos(name, p_nuc=None, gamma=None, K=None):
    """Factory for the equations of state offered to the student."""
    if name == "electron":
        raise ValueError("Use wd_structure() for the electron-gas white-dwarf EOS.")
    if name == "neutron":
        return FermiGasEOS(m_n, m_n)
    if name == "polytrope":
        return PolytropeEOS(p_nuc=0.04 if p_nuc is None else p_nuc,
                            gamma=2.5 if gamma is None else gamma,
                            K=K)
    raise ValueError(f"Unknown equation of state {name!r}; use 'neutron' or 'polytrope'.")


# ======================================================================
# Stellar structure integration (Newtonian and TOV)
# ======================================================================
def integrate_structure(eos, y_c, relativistic=False,
                        r_scale=1.0e7, y_floor=1.0e-8,
                        step_frac=0.01, max_steps=MAX_STRUCT_STEPS,
                        keep_profile=False):
    """
    Integrate a spherical star outward from the centre.

    Newtonian (relativistic=False):
        dm/dr = 4 pi r^2 rho
        dP/dr = -G m rho / r^2

    TOV (relativistic=True):
        dm/dr = 4 pi r^2 (eps/c^2)
        dP/dr = -G (eps/c^2 + P/c^2)(m + 4 pi r^3 P/c^2)
                 / ( r^2 (1 - 2 G m /(r c^2)) )

    The integration variable is the EOS parameter y (the Fermi relativity
    parameter x, or the rest-mass density for a polytrope), advanced with

        dy/dr = (dP/dr) / (dP/dy).

    Returns (M_kg, R_m, profile_or_None).
    """
    # _require_positive (not a bare float()/<=0 check) so that a NaN
    # central variable is rejected explicitly: NaN <= 0.0 is False in
    # Python, so the old check let a NaN y_c silently through to a tiny,
    # physically meaningless central seed instead of raising.
    y = _require_positive("y_c", y_c)
    relativistic = _require_bool("relativistic", relativistic)
    keep_profile = _require_bool("keep_profile", keep_profile)
    r_scale = _require_positive("r_scale", r_scale)
    y_floor = _require_finite("y_floor", y_floor)
    if not (0.0 < y_floor < 1.0):
        raise ValueError(f"y_floor must lie in (0, 1); got {y_floor:g}.")
    step_frac = _require_finite("step_frac", step_frac)
    if not (0.0 < step_frac <= 0.5):
        raise ValueError(f"step_frac must lie in (0, 0.5]; got {step_frac:g}.")
    max_steps = _require_int("max_steps", max_steps, lo=1)

    r = r_scale * step_frac * 1.0e-6
    rho_c = float(eos.mass_energy_density(y) if relativistic
                  else eos.rest_mass_density(y))
    m = (4.0 / 3.0) * np.pi * r**3 * rho_c

    rs, ms, ys = ([r], [m], [y]) if keep_profile else (None, None, None)

    def derivs(r_, m_, y_):
        rho_rest = float(eos.rest_mass_density(y_))
        if relativistic:
            rho_g = float(eos.mass_energy_density(y_))
            P = float(eos.pressure(y_))
            metric = 1.0 - 2.0 * G * m_ / (r_ * c**2)
            if metric <= 0.0:
                raise RuntimeError(
                    "The TOV integration reached a horizon (1 - 2Gm/rc^2 <= 0); "
                    "the central density is too high for this equation of state."
                )
            dP = -(G * (rho_g + P / c**2) * (m_ + 4.0 * np.pi * r_**3 * P / c**2)
                   / (r_**2 * metric))
            dm = 4.0 * np.pi * r_**2 * rho_g
        else:
            dP = -G * m_ * rho_rest / r_**2
            dm = 4.0 * np.pi * r_**2 * rho_rest
        dPdy = float(eos.dP_dx(y_))
        if dPdy <= 0.0:
            raise RuntimeError("dP/dy vanished during the structure integration.")
        return dm, dP / dPdy

    steps = 0
    while y > y_floor * y_c:
        # The initial derivs() call (k1) must be inside the same try/except
        # as the k2-k4 stages: it is evaluated at the current (r, m, y)
        # state using the same EOS calls that can fail (a bad y value can
        # make dP/dy vanish, or drive the TOV metric non-positive), so
        # leaving it outside let a first-stage failure propagate as an
        # uncaught RuntimeError/ValueError instead of the intended,
        # actionable message below.
        try:
            dm_dr, dy_dr = derivs(r, m, y)
            # Step control: never change y by more than step_frac, and
            # never advance by more than step_frac of the current radius.
            dr_cap = step_frac * max(r_scale, r)
            if dy_dr != 0.0:
                dr = min(dr_cap, step_frac * abs(y / dy_dr))
            else:
                dr = dr_cap
            dr = max(dr, 1.0e-6)

            k1m, k1y = dm_dr, dy_dr
            k2m, k2y = derivs(r + 0.5 * dr, m + 0.5 * dr * k1m, max(y + 0.5 * dr * k1y, 1e-30))
            k3m, k3y = derivs(r + 0.5 * dr, m + 0.5 * dr * k2m, max(y + 0.5 * dr * k2y, 1e-30))
            k4m, k4y = derivs(r + dr, m + dr * k3m, max(y + dr * k3y, 1e-30))
        except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
            # A failed Runge-Kutta stage is a numerical or equation-of-state
            # failure, not a stellar surface.  Reporting it as a surface
            # would return a plausible-looking but incomplete star.  This
            # also catches the horizon RuntimeError that derivs() itself
            # raises for the TOV metric, so that failure gets the same
            # actionable message as every other stage failure instead of
            # propagating with a different, inconsistent one.
            raise RuntimeError(
                "The structure integration failed at r = "
                f"{r:.4g} m (central EOS variable {y_c:.6g}): {exc}.  "
                "Reduce --step_frac, or move the central density into a "
                "range this equation of state can represent."
            ) from exc

        m_next = m + (dr / 6.0) * (k1m + 2 * k2m + 2 * k3m + k4m)
        y_next = y + (dr / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

        if not (math.isfinite(m_next) and math.isfinite(y_next)):
            raise RuntimeError(
                "The structure integration produced a non-finite state; "
                "reduce --step_frac."
            )
        if y_next <= 0.0:
            # Linearly interpolate the surface within this step.
            frac = y / (y - y_next)
            r += frac * dr
            m += frac * (m_next - m)
            y = 0.0
            if keep_profile:
                rs.append(r); ms.append(m); ys.append(0.0)
            break

        r += dr
        m = m_next
        y = y_next
        if keep_profile:
            rs.append(r); ms.append(m); ys.append(y)

        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"The structure integration exceeded {max_steps:,} steps; "
                "increase --step_frac or check the central density."
            )

    profile = None
    if keep_profile:
        profile = dict(r=np.asarray(rs), m=np.asarray(ms), y=np.asarray(ys))
    return m, r, profile


# ======================================================================
# White dwarfs: structure, mass-radius relation, Mestel cooling
# ======================================================================
def check_mu_e(mu_e):
    """
    Validate the electron mean molecular weight of degenerate matter.

    mu_e is the mass per electron in atomic mass units -- roughly the
    number of nucleons per electron -- so mu_e = 2 for helium, carbon and
    oxygen and mu_e = 1 for pure hydrogen.  Ordinary fully ionised matter
    cannot have fewer than one nucleon per electron, so mu_e >= 1.
    """
    mu_e = _require_positive("mu_e", mu_e)
    if mu_e < 1.0:
        raise ValueError(
            f"mu_e = {mu_e:g} means fewer than one nucleon per electron.  "
            "mu_e is the mass per electron in atomic mass units: 1 for pure "
            "hydrogen, 2 for helium, carbon or oxygen, about 2.15 for iron."
        )
    if mu_e > 3.0:
        raise ValueError(
            f"mu_e = {mu_e:g} is heavier per electron than any plausible "
            "white-dwarf composition (iron is about 2.15)."
        )
    return mu_e


def chandrasekhar_mass(mu_e):
    """The Chandrasekhar limit, M_Ch = 5.836 mu_e^-2 solar masses."""
    return 5.836 / mu_e**2


def wd_structure(m_target_msun, mu_e=2.0, step_frac=0.01,
                 rho_lo=1.0e7, rho_hi=1.0e13, tol=1.0e-5, max_iter=80,
                 max_bracket_expansions=40):
    """
    Find the central density that gives a white dwarf of the requested
    mass, by bisection on the exact Chandrasekhar structure integration.

    rho_lo and rho_hi are starting points, not hard limits: if they do not
    already bracket the requested mass the search widens them by decades
    (up to max_bracket_expansions times per side) before giving up, since a
    mass close to the Chandrasekhar limit needs a central density well
    above the default rho_hi.

    Returns (rho_c, M_kg, R_m).
    """
    m_target_msun = _require_positive("white-dwarf mass", m_target_msun)
    mu_e = check_mu_e(mu_e)
    tol = _require_positive("tol", tol)
    if tol >= 1.0:
        raise ValueError(f"tol must lie in (0, 1); got {tol:g}.")
    max_iter = _require_int("max_iter", max_iter, lo=1)
    max_bracket_expansions = _require_int("max_bracket_expansions",
                                          max_bracket_expansions, lo=0, hi=60)
    m_ch = chandrasekhar_mass(mu_e)
    if m_target_msun >= 0.999 * m_ch:
        raise ValueError(
            f"A white dwarf of {m_target_msun:g} Msun is not supported: the "
            f"Chandrasekhar limit for mu_e = {mu_e:g} is {m_ch:.3f} Msun."
        )

    eos = FermiGasEOS(m_e, mu_e * m_u)
    target = m_target_msun * M_sun

    def mass_of(rho_c):
        x_c = eos.x_from_density(rho_c)
        r_scale = 1.0e7 * (rho_c / 1.0e9) ** (-1.0 / 6.0)
        M, R, _ = integrate_structure(eos, x_c, relativistic=False,
                                      r_scale=max(r_scale, 1.0e5),
                                      step_frac=step_frac)
        return M, R

    lo, hi = _require_positive("rho_lo", rho_lo), _require_positive("rho_hi", rho_hi)
    if hi <= lo:
        raise ValueError(
            f"rho_hi ({hi:.3e}) must exceed rho_lo ({lo:.3e}); a bisection "
            "search cannot start from a reversed or degenerate bracket."
        )
    m_lo, _ = mass_of(lo)
    m_hi, _ = mass_of(hi)
    n = 0
    while m_lo > target and n < max_bracket_expansions:
        lo /= 10.0
        m_lo, _ = mass_of(lo)
        n += 1
    n = 0
    while m_hi < target and n < max_bracket_expansions:
        hi *= 10.0
        m_hi, _ = mass_of(hi)
        n += 1
    if m_lo > target or m_hi < target:
        raise RuntimeError(
            "Bisection could not bracket the requested white-dwarf mass "
            f"even after widening the search to rho_c in "
            f"[{lo:.3e}, {hi:.3e}] kg/m^3; try a mass further from the "
            "Chandrasekhar limit."
        )

    # Keep mid, M and R computed together at every step so the value
    # ultimately returned always reflects an actually-evaluated model,
    # rather than recomputing sqrt(lo*hi) from post-update bounds that no
    # longer correspond to the M, R last computed inside the loop.
    mid = math.sqrt(lo * hi)
    M, R = mass_of(mid)
    for _ in range(max_iter):
        if abs(M - target) / target < tol:
            return mid, M, R
        if M < target:
            lo = mid
        else:
            hi = mid
        mid = math.sqrt(lo * hi)
        M, R = mass_of(mid)
    # The loop ran max_iter times without the residual falling below tol.
    # Returning (mid, M, R) here anyway would silently hand back a model
    # that can be far from the requested mass (a single-iteration probe at
    # the default bracket returns a mass 58% off target) while looking
    # exactly like an ordinary successful result -- the caller has no way
    # to tell the two apart without independently recomputing the
    # residual itself.  Failing to converge is therefore reported the
    # same way every other public solver in this module reports failure:
    # a RuntimeError that states what was requested, what was achieved,
    # and where the search ended.
    residual = abs(M - target) / target
    raise RuntimeError(
        f"wd_structure() did not converge within max_iter={max_iter} "
        f"iterations: requested tol={tol:.3e}, achieved relative residual "
        f"{residual:.3e} at rho_c={mid:.3e} kg/m^3 (final bracket "
        f"[{lo:.3e}, {hi:.3e}] kg/m^3, M={M / M_sun:.6f} Msun against a "
        f"target of {m_target_msun:.6f} Msun).  Increase max_iter or "
        "relax tol."
    )


def wd_mass_radius_curve(mu_e=2.0, n=40, rho_lo=1.0e8, rho_hi=1.0e14,
                         step_frac=0.01):
    """
    Mass-radius relation for cold white dwarfs, using the zero-temperature
    electron equation of state in Newtonian hydrostatic equilibrium.
    """
    n = _require_int("n_mr", n, lo=3, hi=MAX_GRID_POINTS)
    mu_e = check_mu_e(mu_e)
    rho_lo = _require_positive("rho_lo", rho_lo)
    rho_hi = _require_positive("rho_hi", rho_hi)
    if rho_hi <= rho_lo:
        raise ValueError("rho_hi must exceed rho_lo.")
    eos = FermiGasEOS(m_e, mu_e * m_u)
    rho = np.geomspace(rho_lo, rho_hi, n)
    M = np.empty(n)
    R = np.empty(n)
    for i, rc in enumerate(rho):
        x_c = eos.x_from_density(rc)
        r_scale = max(1.0e7 * (rc / 1.0e9) ** (-1.0 / 6.0), 1.0e5)
        m_kg, r_m, _ = integrate_structure(eos, x_c, relativistic=False,
                                           r_scale=r_scale, step_frac=step_frac)
        M[i] = m_kg / M_sun
        R[i] = r_m / R_sun
    return rho, M, R


def mestel_constant(mu_env, mu_e_env, kappa0):
    """
    Coefficient C in the Mestel luminosity law L = C M T_c^{7/2}.

    Derivation (see the help file).  A radiative envelope with Kramers
    opacity kappa = kappa0 rho T^-3.5 and a zero surface boundary condition
    obeys

        P = sqrt(2 K' / 8.5) T^4.25,    K' = 16 pi a c G M k /(3 kappa0 mu m_u L).

    Matching that envelope to the isothermal degenerate core, where the
    non-relativistic electron pressure K1 (rho/mu_e)^{5/3} equals the ideal
    gas pressure, eliminates the transition density and leaves

        L = C M T_c^{7/2},
        C = (32 pi a c G k)/(25.5 kappa0 mu m_u) * [K1 (mu m_u/(k mu_e))^{5/3}]^3 .

    Because L is proportional to M and the ion heat content is also
    proportional to M, the core-temperature history T_c(t) predicted by
    this model does not depend on the mass of the white dwarf at all,
    although the luminosity does.
    """
    K1 = 1.0036e7          # SI: P = K1 (rho/mu_e)^{5/3}, P in Pa, rho in kg/m^3
    bracket = K1 * (mu_env * m_u / (k_B * mu_e_env)) ** (5.0 / 3.0)
    return (32.0 * np.pi * a_rad * c * G * k_B
            / (25.5 * kappa0 * mu_env * m_u)) * bracket**3


def kramers_kappa0(X, Z):
    """
    Combined bound-free plus free-free Kramers coefficient in SI units
    (kappa = kappa0 rho T^-3.5, kappa in m^2/kg, rho in kg/m^3).

    cgs coefficients: kappa_bf = 4.34e25 Z(1+X), kappa_ff = 3.68e22 (1-Z)(1+X),
    both in cm^2/g with rho in g/cm^3.  Converting kappa to m^2/kg and rho to
    kg/m^3 multiplies both by 1e-4.

    Note the strong dependence on Z.  Gravitational settling removes heavy
    elements from a white dwarf's envelope on a short timescale, so a
    settled envelope is close to metal free, free-free absorption
    dominates, and the envelope is far more transparent than a
    solar-composition envelope would be.  (Real white dwarfs can still show
    detectable metals from ongoing accretion; Z_env=0 models the settled,
    unpolluted case.)
    """
    X, Y, Z = check_composition(X, Z)
    kappa_cgs = 4.34e25 * Z * (1.0 + X) + 3.68e22 * (1.0 - Z) * (1.0 + X)
    return 1.0e-4 * kappa_cgs


def integrate_wd_cooling(m_msun=0.6, mu_e=2.0, A_ion=14.0,
                         X_env=0.70, Z_env=0.0,
                         Tc0=3.0e7, Tc_end=3.0e6, n_steps=4000,
                         step_frac=0.01):
    """
    Mestel cooling of a white dwarf.

    Structure: the zero-temperature Chandrasekhar electron EOS fixes R(M).
    Thermal content: in the Mestel approximation only the ions contribute,

        U = (3/2) (k/(A m_u)) M T_c ,

    because the heat capacity of the strongly degenerate electrons is
    smaller than the ionic one by roughly kT/E_F.  It is small, not zero.
    Cooling: dU/dt = -L with L = C M T_c^{7/2}, so

        dT_c/dt = -(2/3) (A m_u/k) C T_c^{7/2} .

    The exact solution with a finite starting temperature is

        t = [U'/(2.5 C M)] (T_c^{-5/2} - T_c0^{-5/2}),

    which is what this routine reports as the analytic age.  Only for
    T_c << T_c0 does that reduce to t proportional to T_c^{-5/2} and hence
    to the classic late-time Mestel result L proportional to t^{-7/5}.
    The program also integrates the ODE with RK4 so the two can be
    compared.

    Note which composition controls what: the core mu_e sets the structure
    (radius and central density), A_ion sets the ionic heat capacity, and
    the ENVELOPE composition (X_env, Z_env) sets the opacity and hence the
    Mestel coefficient C.  Students are often surprised by the last one.
    """
    m_msun = _require_positive("white-dwarf mass", m_msun)
    mu_e = check_mu_e(mu_e)
    A_ion = _require_positive("A_ion", A_ion)
    if not (1.0 <= A_ion <= 60.0):
        raise ValueError(
            f"A_ion = {A_ion:g} is outside [1, 60]: A_ion is the mean ionic "
            "mass number setting the ion heat capacity (about 4 for a "
            "helium core, 12-16 for carbon-oxygen, up to the mid-50s for "
            "an oxygen-neon or iron-group core); values outside this range "
            "do not correspond to any white-dwarf composition."
        )
    Tc0 = _require_positive("Tc0", Tc0)
    Tc_end = _require_positive("Tc_end", Tc_end)
    if Tc_end < 100.0:
        # The analytic-age formula below evaluates Tc_end**-2.5; for a
        # sufficiently small (but still positive) Tc_end that overflows a
        # Python float (OverflowError) well before the cooling model has
        # any physical content left to describe, since real white-dwarf
        # cooling curves are not extended to near-absolute-zero core
        # temperatures by this simple Mestel law.
        raise ValueError(
            f"Tc_end = {Tc_end:g} K is below 100 K, where this cooling "
            "model has no physical content and the analytic-age formula "
            "would overflow.  Use a larger --Tc_end."
        )
    if Tc_end >= Tc0:
        raise ValueError(
            f"Tc_end ({Tc_end:g} K) must be below the starting core "
            f"temperature Tc0 ({Tc0:g} K)."
        )
    if Tc0 > 1.0e9:
        raise ValueError("Tc0 above 1e9 K is outside the range of this cooling model.")
    n_steps = _require_int("n_cool", n_steps, lo=MIN_TRACK_STEPS, hi=MAX_TRACK_STEPS)

    X_env, Y_env, Z_env = check_composition(X_env, Z_env)
    mu_env = mean_molecular_weight(X_env, Z_env)
    mu_e_env = mean_molecular_weight_per_electron(X_env)
    kappa0 = kramers_kappa0(X_env, Z_env)

    rho_c, M_kg, R_m = wd_structure(m_msun, mu_e=mu_e, step_frac=step_frac)
    C = mestel_constant(mu_env, mu_e_env, kappa0)

    heat_capacity = 1.5 * k_B * M_kg / (A_ion * m_u)   # dU/dT_c

    def dTdt(T):
        return -C * M_kg * T ** 3.5 / heat_capacity

    # Analytic solution, kept for comparison with the RK4 result.
    coef = heat_capacity / (2.5 * C * M_kg)
    t_end_analytic = coef * (Tc_end ** -2.5 - Tc0 ** -2.5)

    # Because dT/dt goes as T^3.5 the cooling is extremely stiff at early
    # times.  Steps are therefore chosen so that each one changes T_c by a
    # fixed fractional amount, giving about n_steps steps in total.
    alpha = math.log(Tc0 / Tc_end) / n_steps

    t_list, T_list, L_list = [], [], []
    t = 0.0
    T = Tc0
    steps = 0
    while True:
        t_list.append(t)
        T_list.append(T)
        L_list.append(C * M_kg * T ** 3.5)
        if T <= Tc_end:
            break
        steps += 1
        if steps > MAX_TRACK_STEPS:
            raise RuntimeError("The cooling integration exceeded the step limit.")

        dt = alpha * T / abs(dTdt(T))

        k1 = dTdt(T)
        k2 = dTdt(max(T + 0.5 * dt * k1, 1.0))
        k3 = dTdt(max(T + 0.5 * dt * k2, 1.0))
        k4 = dTdt(max(T + dt * k3, 1.0))
        T_next = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not math.isfinite(T_next) or T_next >= T or T_next <= 0.0:
            raise RuntimeError(
                "The cooling integration failed to decrease the core "
                "temperature; increase --n_cool."
            )
        if T_next <= Tc_end:
            frac = (T - Tc_end) / (T - T_next)
            t += frac * dt
            T = Tc_end
        else:
            t += dt
            T = T_next

    t_arr = np.asarray(t_list)
    T_arr = np.asarray(T_list)
    L_arr = np.asarray(L_list)              # W
    L_lsun = L_arr / L_sun
    Teff = (L_arr / (4.0 * np.pi * R_m**2 * sigma_SB)) ** 0.25

    rho_mr, M_mr, R_mr = wd_mass_radius_curve(mu_e=mu_e, n=36, step_frac=step_frac)

    summary = dict(
        m_msun=M_kg / M_sun, m_requested=m_msun,
        mu_e=mu_e, A_ion=A_ion,
        X_env=X_env, Z_env=Z_env, mu_env=mu_env, mu_e_env=mu_e_env,
        kappa0=kappa0,
        rho_c=rho_c, R_km=R_m / 1.0e3, R_rsun=R_m / R_sun,
        R_rearth=R_m / R_EARTH,
        M_ch=chandrasekhar_mass(mu_e),
        Tc0=Tc0, Tc_end=Tc_end,
        L_start=L_lsun[0], L_end=L_lsun[-1],
        Teff_start=Teff[0], Teff_end=Teff[-1],
        t_end_gyr=t_arr[-1] / GYR,
        t_end_analytic_gyr=t_end_analytic / GYR,
        mestel_C=C,
        n_points=t_arr.size,
        mean_density=M_kg / ((4.0 / 3.0) * np.pi * R_m**3),
        surface_gravity=G * M_kg / R_m**2,
        warnings=[],
        model_version=MODEL_VERSION,
        build_id=BUILD_ID,
    )

    return dict(
        kind="wdcool",
        t=t_arr, Tc=T_arr, L=L_lsun, Teff=Teff,
        mr_rho=rho_mr, mr_M=M_mr, mr_R=R_mr,
        summary=summary,
    )


# ======================================================================
# Neutron stars: TOV mass-radius relation
# ======================================================================
def ns_mass_radius_curve(eos_name="neutron", n=40,
                         rho_lo=1.0e17, rho_hi=5.0e19,
                         relativistic=True, p_nuc=None, gamma=None, K=None,
                         step_frac=0.01):
    """
    Sequence of neutron-star models obtained by integrating the TOV (or,
    with relativistic=False, the Newtonian) equations for a range of
    central densities.

    Returns central density, mass, radius, compactness and (for a
    relativistic sequence only) surface gravitational redshift arrays.

    The turning point of M(rho_c) marks the onset of radial instability
    for cold, non-rotating, one-parameter equilibrium sequences of this
    kind.  It is reported only when the sequence actually turns over
    within the sampled range, and only for the TOV equations.
    """
    n = _require_int("n_mr", n, lo=3, hi=MAX_GRID_POINTS)
    relativistic = _require_bool("relativistic", relativistic)
    rho_lo = _require_positive("rho_lo", rho_lo)
    rho_hi = _require_positive("rho_hi", rho_hi)
    if rho_hi <= rho_lo:
        raise ValueError("rho_hi must exceed rho_lo.")

    eos = make_eos(eos_name, p_nuc=p_nuc, gamma=gamma, K=K)
    rho = np.geomspace(rho_lo, rho_hi, n)

    M = np.full(n, np.nan)
    R = np.full(n, np.nan)
    # Recording WHY each dropped model failed (rather than only that it
    # failed) lets the summary warning describe the actual failure mix
    # instead of assuming a horizon encounter, and keeps an unexpected
    # programming defect from being silently indistinguishable from an
    # ordinary non-convergence.
    fail_reasons = {}
    for i, rc in enumerate(rho):
        y_c = eos.x_from_density(rc)
        r_scale = 1.5e4
        try:
            m_kg, r_m, _ = integrate_structure(eos, y_c, relativistic=relativistic,
                                               r_scale=r_scale, step_frac=step_frac)
        except RuntimeError as exc:
            fail_reasons[i] = str(exc)
            continue
        M[i] = m_kg / M_sun
        R[i] = r_m / 1.0e3            # km

    # A model is only accepted if it is a star.  A very stiff equation of
    # state at extreme central density can drive the integrator into a
    # degenerate state that formally returns, and reporting that as a
    # zero-radius neutron star would be worse than dropping it.
    good = (np.isfinite(M) & np.isfinite(R)
            & (M > 1.0e-4) & (R > 1.0e-2))
    for i in np.where(~good)[0]:
        fail_reasons.setdefault(int(i), "model rejected: mass or radius too small to be a star")
    M = np.where(good, M, np.nan)
    R = np.where(good, R, np.nan)
    if good.sum() < 3:
        raise RuntimeError(
            "Fewer than three neutron-star models converged; adjust the "
            "central-density range."
        )

    warnings = []
    if int(good.sum()) < n:
        n_horizon = sum(1 for msg in fail_reasons.values() if "horizon" in msg)
        n_dropped = n - int(good.sum())
        if n_horizon == n_dropped:
            cause = "the integration reached a horizon in every dropped case"
        elif n_horizon == 0:
            cause = "none of the dropped cases reached a horizon (see warnings_detail)"
        else:
            cause = (f"{n_horizon} of {n_dropped} dropped cases reached a "
                     "horizon; the rest failed for other reasons "
                     "(see warnings_detail)")
        warnings.append(
            f"{n_dropped} of the {n} requested central densities did not "
            f"yield a valid stellar model and were dropped ({cause}).  "
            "With a very stiff equation of state near the horizon this "
            "usually means lowering --rho_hi will help; a failure that is "
            "NOT a horizon encounter may instead indicate a numerical or "
            "programming problem worth investigating directly."
        )

    # GM/Rc^2 is always computable.  For a TOV sequence it is the physical
    # compactness; for a Newtonian sequence it is only a diagnostic showing
    # where the Newtonian description stops being self-consistent.
    compact = np.full(n, np.nan)
    compact[good] = G * M[good] * M_sun / (R[good] * 1.0e3 * c**2)

    # The Schwarzschild surface redshift is a general-relativistic result
    # and is left undefined for a Newtonian sequence.
    z_surf = np.full(n, np.nan)
    if relativistic:
        idx = np.where(good)[0]
        idx = idx[1.0 - 2.0 * compact[idx] > 0.0]
        z_surf[idx] = 1.0 / np.sqrt(1.0 - 2.0 * compact[idx]) - 1.0

    gi = np.where(good)[0]
    i_max = int(gi[int(np.argmax(M[gi]))])

    # A genuine turning point needs converged models on BOTH sides of the
    # maximum.  i_max == gi[0] (mass already falling at the lowest sampled
    # density) is just as inconclusive as i_max == gi[-1] (mass still
    # rising at the highest sampled density): in both cases the true
    # extremum lies outside the sampled range.  It is not enough that
    # SOME converged point exists somewhere below and somewhere above
    # i_max, though: if either of i_max's own immediate neighbors failed
    # to converge, the sampled maximum sits at the edge of an unsolved
    # gap, and an even larger true mass could be hiding inside that gap.
    # Both conditions are required before a sampled peak may be reported
    # as a resolved turning point.
    interior = bool(gi[0] < i_max < gi[-1])
    neighbors_converged = (interior and good[i_max - 1] and good[i_max + 1])
    turning_point = bool(interior and neighbors_converged)
    if not turning_point:
        if i_max >= gi[-1]:
            warnings.append(
                "the mass is still rising at the highest central density "
                "sampled, so no turning point was found.  The value "
                "reported is the largest sampled mass, not a maximum mass, "
                "and no model in this sequence has been shown to be "
                "unstable.  Raise --rho_hi to look for the true turning "
                "point."
            )
        elif i_max <= gi[0]:
            warnings.append(
                "the mass is already falling at the lowest central density "
                "sampled, so the maximum lies below the sampled range and "
                "no turning point was found within it.  The value reported "
                "is the largest sampled mass, not a maximum mass, and no "
                "model in this sequence has been shown to be unstable.  "
                "Lower --rho_lo to look for the true turning point."
            )
        else:
            warnings.append(
                "the largest converged sampled mass sits next to a central "
                "density that did NOT converge, so the true maximum is "
                "unresolved across that convergence gap: a larger mass "
                "could exist inside the gap and would not have been seen.  "
                "The value reported is the largest converged sampled mass, "
                "not a maximum mass, and no model in this sequence has been "
                "shown to be unstable.  Increase --n_mr, or narrow the "
                "density range around the gap, to resolve it."
            )

    # Sound speed over the whole stable (or, without a turning point, the
    # whole sampled) branch, not only at the extremum.  The branch is the
    # single contiguous run of converged models ending at i_max, not every
    # converged index below it: an isolated non-converged density inside
    # that range must not be silently bridged over as if the branch were
    # unbroken on both sides of the gap.
    stable_idx = [i_max]
    _i = i_max - 1
    while _i >= 0 and good[_i]:
        stable_idx.append(_i)
        _i -= 1
    stable_idx = np.array(sorted(stable_idx))
    cs_branch = np.array([float(eos.sound_speed_ratio(eos.x_from_density(rho[i])))
                          for i in stable_idx])
    cs_max_branch = float(np.max(cs_branch))
    i_cs = int(stable_idx[int(np.argmax(cs_branch))])
    cs_at_max = float(eos.sound_speed_ratio(eos.x_from_density(rho[i_max])))
    causal = bool(cs_max_branch <= 1.0)
    if not causal:
        warnings.append(
            f"the sound speed reaches c_s/c = {cs_max_branch:.3f} on the "
            f"branch (at rho_c = {rho[i_cs]:.3e} kg/m^3).  This equation of "
            "state is acausal there and the models above that density are "
            "not physically admissible."
        )

    if not relativistic:
        warnings.append(
            "this is a Newtonian sequence.  GM/Rc^2 is reported only as a "
            "self-consistency diagnostic -- it shows where Newtonian "
            "gravity ceases to be an acceptable approximation -- and the "
            "gravitational redshift, the Buchdahl bound and the "
            "turning-point stability criterion, all of which are results of "
            "general relativity, are not reported."
        )

    summary = dict(
        eos=eos_name,
        relativistic=bool(relativistic),
        gamma=(getattr(eos, "gamma", float("nan"))),
        K=(getattr(eos, "K", float("nan"))),
        p_nuc=(getattr(eos, "p_nuc", float("nan"))),
        n_models=int(good.sum()),
        rho_lo=rho_lo, rho_hi=rho_hi,
        M_max=float(M[i_max]),
        R_at_Mmax=float(R[i_max]),
        rho_at_Mmax=float(rho[i_max]),
        compact_at_Mmax=float(compact[i_max]),
        cs_over_c_at_Mmax=cs_at_max,
        cs_over_c_max_branch=cs_max_branch,
        rho_at_cs_max=float(rho[i_cs]),
        causal=causal,
        z_at_Mmax=float(z_surf[i_max]),
        turning_point=turning_point,
        # Equal to turning_point by construction: a turning point is only
        # ever reported once a genuine stable/unstable split has been
        # identified, so there is currently no case where the two differ.
        # Kept as a separate field for API stability; do not rely on any
        # future difference between the two without checking this module's
        # current version.
        stable_branch=turning_point,
        M_min=float(np.nanmin(M[good])),
        R_min=float(np.nanmin(R[good])),
        R_max=float(np.nanmax(R[good])),
        warnings=warnings,
        # One entry per central density that did NOT yield an accepted
        # model, giving the actual reason instead of forcing a caller to
        # guess from the aggregate warning text above.
        warnings_detail=[f"rho_c={rho[i]:.4e} kg/m^3: {reason}"
                         for i, reason in sorted(fail_reasons.items())],
        model_version=MODEL_VERSION,
        build_id=BUILD_ID,
    )
    return dict(
        kind="nsmr",
        rho=rho, M=M, R=R, compact=compact, z=z_surf,
        i_max=i_max, summary=summary,
    )
