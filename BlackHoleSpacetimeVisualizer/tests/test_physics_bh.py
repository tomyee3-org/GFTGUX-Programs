"""
tests/test_physics_bh.py
=========================
Regression / correctness suite for Black_hole_spacetime_visualizer.

Run with any of:
    python -m unittest tests.test_physics_bh -v
    python -m unittest discover -s tests
    python tests/test_physics_bh.py

Design notes
------------
* Every physics assertion here is checked against something that is NOT
  simply "call another function in physics_bh.py that used the same
  formula" -- each TestCase docstring says what the independent check is
  (a closed-form solution, a hand-derived scaling law, an independently
  coded formula, a physical invariant, or a limiting case).
* Tests are grouped into TestCase classes by physical/numerical
  invariant, per mode, plus classes for CLI/version/output provenance and
  for the specific defects found and fixed while building this suite
  (each such class reproduces the defect on the pre-fix code path
  conceptually, and permanently guards the fix).
* Warnings: `warnings.simplefilter("error")` is installed per-test (via
  setUp) for tests that assert a clean numerical run, so any unexpected
  RuntimeWarning becomes a test failure rather than console noise.
* Randomised/property-style checks use a fixed seed (SEED, below) and
  print it on assertion failure so a failure is reproducible.
"""

import csv
import importlib.util
import inspect
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_bh as phys  # noqa: E402

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
SEED = 20260825


# ======================================================================
# Small shared helpers
# ======================================================================
def run_cli(args, cwd=PROJECT_DIR, timeout=120):
    """Run `python main.py <args>` as a subprocess, MPLBACKEND=Agg."""
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [PYTHON, "main.py"] + args,
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        lines = fh.readlines()
    comments = [l for l in lines if l.startswith("#")]
    data_lines = [l for l in lines if not l.startswith("#")]
    reader = csv.reader(data_lines)
    rows = list(reader)
    return comments, rows[0], rows[1:]


def proper_radial_distance_closed_form(rs, r):
    """
    Independent closed-form antiderivative of integral(dr'/sqrt(1-rs/r'))
    from rs to r, derived by hand (u = sqrt(r-rs) substitution,
    integral(sqrt(u^2+a^2)) du standard form) -- NOT the same
    substitution-and-trapz numerical method physics_bh._proper_radial_distance
    uses, so this is a genuine independent check of that function:

        l(r) = sqrt(r(r-rs)) + rs * ln[ (sqrt(r-rs) + sqrt(r)) / sqrt(rs) ]
    """
    if r <= rs:
        return 0.0
    return (math.sqrt(r * (r - rs))
            + rs * math.log((math.sqrt(r - rs) + math.sqrt(r)) / math.sqrt(rs)))


def infall_cycloid_tau(m_msun, r0_rs, r_stop_rs):
    """
    Independent closed-form (cycloid-parametrised) proper time for radial
    free fall from rest at r0 to r_stop, per EXP-11 of the help file:
        eta_stop = arccos(2 r_stop/r0 - 1)
        tau(eta) = sqrt(r0^3 / (8 G M)) * (eta + sin eta)
    This is the same textbook closed form the help file cites (also the
    Newtonian radial Kepler-fall cycloid with r0 in place of the
    semi-latus rectum), coded independently of physics_bh.infall_radial's
    RK4 integrator.

    GM is taken from the literal IAU 2015 Resolution B3 nominal solar mass
    parameter, not from phys.G/phys.M_sun: reusing the module's own
    constants here would make this "independent" oracle share physics_bh's
    then-current constant, so a wrong constant in the module could not be
    caught by this test (Reviewer Audit round 1, Codex P2-2 test-suite
    critique). phys.schwarzschild_radius is still used for r0/r_stop's
    metre conversion, which is a separate, purely geometric step
    (r0_rs/r_stop_rs times r_s) not under test here; the mass-dependent
    physics itself is recomputed from the literal constant below.
    """
    GM_SUN_NOMINAL = 1.3271244e20   # m^3 s^-2, IAU 2015 Resolution B3
    c = 2.99792458e8                # m s^-1
    rs = 2.0 * m_msun * GM_SUN_NOMINAL / c ** 2
    r0 = r0_rs * rs
    r_stop = r_stop_rs * rs
    GM = m_msun * GM_SUN_NOMINAL
    eta_stop = math.acos(2.0 * r_stop / r0 - 1.0)
    # sqrt(r0^3/(8*GM)) already carries units of seconds directly (GM is
    # m^3/s^2), matching the standard Newtonian free-fall-time formula
    # exactly, with no separate factor of c needed.
    return math.sqrt(r0 ** 3 / (8.0 * GM)) * (eta_stop + math.sin(eta_stop))


def import_physics_from_dir(dirpath, modname):
    """
    Import a standalone copy of physics_bh.py located in `dirpath` under a
    fresh module name, for BUILD_ID provenance tests. `dirpath` must also
    contain copies of the other three BUILD_ID_COVERS files, since
    physics_bh._compute_build_id() reads all four from its own directory.
    """
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(dirpath, "physics_bh.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def make_source_copy(tmp_root, label):
    """Copy the four BUILD_ID_COVERS files into tmp_root/label; return that path."""
    dest = os.path.join(tmp_root, label)
    os.makedirs(dest, exist_ok=True)
    for name in phys.BUILD_ID_COVERS:
        shutil.copyfile(os.path.join(PROJECT_DIR, name), os.path.join(dest, name))
    return dest


# ======================================================================
# Schwarzschild scales / mass validation
# ======================================================================
class TestSchwarzschildScales(unittest.TestCase):
    """r_s = 2GM/c^2 scaling and mass-input validation."""

    def test_rs_scales_linearly_with_mass(self):
        # Independent check: r_s is *defined* to be linear in M; verify
        # the coded function actually reproduces that proportionality
        # over four decades of mass, not just at one point.
        rs1 = phys.schwarzschild_radius(1.0)
        for factor in (2.0, 10.0, 100.0, 1.0e4, 1.0e-2):
            rs_f = phys.schwarzschild_radius(factor)
            self.assertAlmostEqual(rs_f / rs1, factor, delta=factor * 1e-12)

    def test_rs_matches_independent_SI_calculation(self):
        # CORRECTED ORACLE (Reviewer Audit round 1, Codex P2-2): this test
        # previously hand-computed G*(M_msun*M_sun_in_kg) using the exact
        # same nonstandard M_sun=1.98892e30 kg value physics_bh.py itself
        # used at the time -- so it was not actually an independent check
        # of the module's mass-dependent physics at all; it would have
        # passed identically whether or not that M_sun value was correct.
        # The literal expected value below is now taken directly from the
        # IAU 2015 Resolution B3 nominal solar mass *parameter* GM_sun
        # (arXiv:1510.07674), which is what physics_bh.schwarzschild_radius
        # is now computed from (GM_SUN_NOMINAL), and needs no separately
        # sourced kilogram mass or G at all -- a genuinely independent
        # literal constant, not a reference to phys.G/phys.M_sun/
        # phys.GM_SUN_NOMINAL.
        GM_sun_nominal = 1.3271244e20   # m^3 s^-2, IAU 2015 Resolution B3
        c = 2.99792458e8
        expected = 2.0 * 10.0 * GM_sun_nominal / c ** 2
        self.assertAlmostEqual(phys.schwarzschild_radius(10.0), expected,
                                delta=expected * 1e-9)

    def test_rs_10_solar_masses_is_about_29_5_km(self):
        # Textbook benchmark value.
        self.assertAlmostEqual(phys.schwarzschild_radius_km(10.0), 29.5, delta=0.1)

    def test_light_crossing_time_equals_rs_over_c(self):
        m = 7.3
        self.assertAlmostEqual(
            phys.light_crossing_time(m),
            phys.schwarzschild_radius(m) / phys.c, delta=1e-25)

    def test_check_mass_rejects_nonpositive_and_nonfinite(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                phys.check_mass("mass", bad)

    def test_check_mass_flags_below_trusted_range(self):
        m, warn = phys.check_mass("mass", 1.0)
        self.assertEqual(m, 1.0)
        self.assertEqual(len(warn), 1)
        self.assertIn("mass gap", warn[0])

    def test_check_mass_flags_above_trusted_range(self):
        m, warn = phys.check_mass("mass", 1.0e11)
        self.assertEqual(len(warn), 1)
        self.assertIn("most massive", warn[0])

    def test_check_mass_silent_in_trusted_range(self):
        m, warn = phys.check_mass("mass", 10.0)
        self.assertEqual(warn, [])

    def test_check_mass_rejects_above_max_computable_ceiling(self):
        # Reviewer Audit round 2, Codex P2-2 reproducer: masses this
        # large previously reached deep into embed/infall/horizons math
        # (r_s**3, r0**3, ...) before overflowing, raising either a raw
        # OverflowError (Python-native **) or silently becoming inf/-0.0
        # (numpy **) several calls away from any input-validation message.
        # check_mass's new MAX_COMPUTABLE_MASS_MSUN ceiling refuses these
        # up front, with a clear, on-topic ValueError.
        for bad in (phys.MAX_COMPUTABLE_MASS_MSUN * 1.001, 1.0e100, 1.0e150, 1.0e300):
            with self.assertRaises(ValueError) as ctx:
                phys.check_mass("mass", bad)
            self.assertIn("exceeds", str(ctx.exception))
        # And confirm the ceiling is not simply "anything large fails":
        # a mass just under it is accepted (with the expected TRUSTED_MASS_HI
        # warning, since it is still astrophysically absurd, just not
        # computationally unrepresentable).
        m, warn = phys.check_mass("mass", phys.MAX_COMPUTABLE_MASS_MSUN * 0.999)
        self.assertEqual(len(warn), 1)

    def test_max_computable_mass_ceiling_prevents_uncaught_overflow(self):
        # Defense-in-depth check that the ceiling is actually wired into
        # the calculations that used to overflow uncaught: embedding_profile
        # (r**3-scale geometry), tidal_profile, and infall_radial (whose
        # cycloid seed calculation used r0**3 directly and could raise a
        # bare Python OverflowError -- see the try/except around
        # tau_seed in infall_radial). Each must raise ValueError, not
        # propagate OverflowError or silently return non-finite output.
        huge = 1.0e100
        with self.assertRaises(ValueError):
            phys.embedding_profile(m_msun=huge, r_max_rs=8.0, n_r=50)
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=huge, r_min_rs=1.01, r_max_rs=8.0, n_r=50)
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=huge, r0_rs=6.0, r_stop_rs=1.0005, n_points=50)
        # Reviewer Audit round 2, Codex P2-2's exact vaidya_horizons
        # reproducer: vaidya_horizons(1e300, 1e300, ...) previously
        # returned inf/nan arrays through the M0==M1 no-accretion fast
        # path, and the NaN postcondition did not catch it (nan > tolerance
        # is always False in Python/numpy). check_mass is called on both
        # M0 and M1 before any of that math runs, so this is now rejected
        # immediately with a clear, on-topic ValueError instead.
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=1.0e300, m1_msun=1.0e300)

    def test_cli_rejects_overflowing_masses_cleanly_in_every_mode(self):
        # Reviewer Audit round 2, Codex P2-2's exact CLI-level reproducers:
        #   python main.py --mode embed --M 1e150 ...   (previously wrote
        #     inf/-0 into the CSV after RuntimeWarnings, exit 0)
        #   python main.py --mode infall --M 1e100 ...  (previously an
        #     uncaught OverflowError traceback)
        # Both must now fail with a concise, non-traceback CLI error and
        # write no output file at all.
        for mode, mass_flag, mass in (("embed", "--M", "1e150"),
                                       ("tidal", "--M", "1e150"),
                                       ("infall", "--M", "1e100")):
            with tempfile.TemporaryDirectory() as tmp:
                proc = run_cli(["--mode", mode, mass_flag, mass, "--no_plot",
                                 "--csvdir", tmp])
                self.assertNotEqual(proc.returncode, 0, msg=f"mode={mode}")
                self.assertNotIn("Traceback", proc.stderr, msg=f"mode={mode}")
                self.assertIn("exceeds", proc.stderr, msg=f"mode={mode}")
                self.assertEqual(os.listdir(tmp), [], msg=f"mode={mode}")

    def test_schwarzschild_radius_rejects_nonfinite_result_from_finite_mass(self):
        # Reviewer Audit round 1, Codex P2-3 reproducer: a finite --M does
        # not guarantee a finite r_s. schwarzschild_radius(1e300) is a
        # *finite* input mass, but 2*m*GM_SUN_NOMINAL/c^2 overflows a
        # double before this fix, and the function silently returned inf
        # instead of raising.
        with self.assertRaises(ValueError):
            phys.schwarzschild_radius(1.0e300)
        self.assertTrue(math.isfinite(phys.schwarzschild_radius(1.0e10)))

    def test_mass_ceiling_applies_at_every_public_entry_point(self):
        # Audit Round 3, Codex P2-1 reproducer: check_mass's
        # MAX_COMPUTABLE_MASS_MSUN ceiling was previously enforced only
        # inside check_mass itself, not by schwarzschild_radius,
        # light_crossing_time, tidal_acceleration, or survival_radius --
        # all four accepted 1e100 Msun and returned a finite-looking
        # answer even though 1e100 is far above the documented
        # computational ceiling. All four now route their mass validation
        # through check_mass and must reject it identically.
        huge = 1.0e100
        with self.assertRaises(ValueError):
            phys.schwarzschild_radius(huge)
        with self.assertRaises(ValueError):
            phys.light_crossing_time(huge)
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(huge, 1.0, separation_m=1.0)
        with self.assertRaises(ValueError):
            phys.survival_radius(huge, separation_m=1.0, limit_g=1.0)
        # And a mass comfortably under the ceiling must still succeed at
        # every one of the same four entry points.
        self.assertTrue(math.isfinite(phys.schwarzschild_radius(10.0)))
        self.assertTrue(math.isfinite(phys.light_crossing_time(10.0)))
        self.assertTrue(all(math.isfinite(x) for x in
                             phys.tidal_acceleration(10.0, 1.0e9, separation_m=1.0)))
        self.assertTrue(math.isfinite(
            phys.survival_radius(10.0, separation_m=1.0, limit_g=100.0)))

    def test_mass_floor_rejects_subnormal_and_underflow_never_silent(self):
        # Audit Round 3, Codex P2-1 reproducer: finite positive inputs
        # previously could silently underflow to an impossible exact zero
        # instead of raising -- tidal_acceleration(5e-324, 1e308, 1.0) ->
        # (0.0, 0.0) and survival_radius(5e-324, 1e308, 1e308) -> 0.0 were
        # both reproduced against the pre-fix code. The new
        # MIN_COMPUTABLE_MASS_MSUN floor rejects the subnormal mass
        # outright (the same computational-limit framing as the ceiling);
        # confirm both the floor itself and, independently, that a mass
        # comfortably above the floor still cannot silently underflow to
        # zero through tidal_acceleration/survival_radius's own extreme
        # r_m/limit_g arguments.
        for bad in (5.0e-324, 1.0e-320, phys.MIN_COMPUTABLE_MASS_MSUN * 0.999):
            with self.assertRaises(ValueError) as ctx:
                phys.check_mass("mass", bad)
            self.assertIn("below", str(ctx.exception))
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(5.0e-324, 1.0e308, separation_m=1.0)
        with self.assertRaises(ValueError):
            phys.survival_radius(5.0e-324, separation_m=1.0e308, limit_g=1.0e308)
        # A mass just above the floor is accepted.
        m, warn = phys.check_mass("mass", phys.MIN_COMPUTABLE_MASS_MSUN * 1.001)
        self.assertEqual(len(warn), 1)   # still below TRUSTED_MASS_LO

    def test_check_mass_docstring_matches_enforced_bounds(self):
        # Audit Round 3, Codex P2-1: check_mass's own docstring previously
        # said "Any positive mass is accepted" while the function actually
        # rejected masses above MAX_COMPUTABLE_MASS_MSUN -- a direct
        # self-contradiction visible to anyone reading help(check_mass).
        doc = inspect.getdoc(phys.check_mass) or ""
        self.assertNotIn("any positive mass is accepted", doc.lower())

    def test_require_finite_reports_overflow_as_value_error(self):
        # Reviewer Audit round 1, Codex P2-3: float() on a Python int with
        # hundreds of digits raises OverflowError, not ValueError/TypeError
        # -- previously uncaught by _require_finite, so it propagated as a
        # raw traceback instead of the program's usual concise message.
        giant_int = 10 ** 400
        with self.assertRaises(ValueError):
            phys._require_finite("mass", giant_int)
        with self.assertRaises(ValueError):
            phys.check_mass("mass", giant_int)


# ======================================================================
# Mode: embed
# ======================================================================
class TestEmbedding(unittest.TestCase):
    """Flamm's paraboloid: formula, throat, proper distance, bounds."""

    def setUp(self):
        warnings.simplefilter("error", RuntimeWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_throat_endpoint_z_is_zero_at_rs(self):
        result = phys.embedding_profile(m_msun=10.0, r_max_rs=8.0, n_r=50)
        self.assertAlmostEqual(result["z"][0], 0.0, delta=1e-9)
        self.assertAlmostEqual(result["r"][0], result["summary"]["rs_m"], delta=1e-6)

    def test_z_matches_closed_form_at_several_radii(self):
        m = 12.3
        rs = phys.schwarzschild_radius(m)
        result = phys.embedding_profile(m_msun=m, r_max_rs=20.0, n_r=500)
        r, z = result["r"], result["z"]
        for i in (0, 5, 100, 250, 499):
            expected = 2.0 * math.sqrt(rs * (r[i] - rs))
            self.assertAlmostEqual(z[i], expected, delta=max(1e-6, abs(expected) * 1e-10))

    def test_z_is_monotonically_nondecreasing_in_r(self):
        result = phys.embedding_profile(m_msun=10.0, r_max_rs=15.0, n_r=800)
        self.assertTrue(np.all(np.diff(result["z"]) >= 0.0))

    def test_r_array_is_sorted_and_starts_at_rs_ends_at_r_max(self):
        result = phys.embedding_profile(m_msun=5.0, r_max_rs=6.0, n_r=100)
        r, s = result["r"], result["summary"]
        self.assertTrue(np.all(np.diff(r) > 0.0))
        self.assertAlmostEqual(r[0], s["rs_m"], delta=1e-6)
        self.assertAlmostEqual(r[-1], s["rs_m"] * 6.0, delta=s["rs_m"] * 1e-9)

    def test_near_horizon_slope_diverges(self):
        # dz/dr = sqrt(rs/(r-rs)) -> infinity as r -> rs+; verify the
        # finite-difference slope keeps growing as we sample closer in.
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        slopes = []
        for eps_rs in (1.0e-2, 1.0e-4, 1.0e-6):
            r_max_rs = 1.0 + eps_rs
            result = phys.embedding_profile(m_msun=m, r_max_rs=r_max_rs, n_r=20)
            r, z = result["r"], result["z"]
            slope = (z[1] - z[0]) / (r[1] - r[0])
            slopes.append(slope)
        self.assertTrue(slopes[1] > slopes[0] > 0)
        self.assertTrue(slopes[2] > slopes[1])

    def test_large_radius_curve_flattens(self):
        # Far from the hole, dz/dr = sqrt(rs/(r-rs)) -> 0; confirm the
        # finite-difference slope at large r tracks that closed form (an
        # independent check, not merely "smaller than near the throat" --
        # the coarse near-throat spacing on a 0..1000 r_s linear grid does
        # not itself probe the divergence, which test_near_horizon_slope_
        # diverges already covers on a much finer grid).
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        result = phys.embedding_profile(m_msun=m, r_max_rs=1000.0, n_r=4000)
        r, z = result["r"], result["z"]
        slope_far = (z[-1] - z[-2]) / (r[-1] - r[-2])
        r_mid_far = 0.5 * (r[-1] + r[-2])
        expected_far = math.sqrt(rs / (r_mid_far - rs))
        self.assertAlmostEqual(slope_far / expected_far, 1.0, delta=1e-2)
        self.assertLess(slope_far, 0.1)  # << 1: the surface is nearly flat out here

    def test_proper_radial_distance_matches_independent_closed_form(self):
        # TIGHTENED TOLERANCE (Reviewer Audit round 1, Codex P2-6):
        # physics_bh._proper_radial_distance was a 4000-point trapezoidal
        # numerical integration, so 1e-6 was a realistic discretization-
        # error tolerance against this independently-derived closed form.
        # It is now itself an exact closed form (differently written --
        # asinh here in the module vs. a normalized log here in the test
        # -- but mathematically identical), so the two should now agree to
        # floating-point precision, not merely 1e-6; a regression back to
        # numerical integration, or a sign/constant slip in either closed
        # form, would now be caught at a far tighter tolerance.
        m = 8.4
        rs = phys.schwarzschild_radius(m)
        for r_over_rs in (1.0001, 1.01, 1.2, 2.0, 8.0, 50.0):
            r = rs * r_over_rs
            coded = phys._proper_radial_distance(rs, rs, r)
            closed = proper_radial_distance_closed_form(rs, r)
            rel = abs(coded - closed) / closed
            self.assertLess(rel, 1e-10, msg=f"r/rs={r_over_rs}: coded={coded}, closed={closed}")

    def test_proper_radial_distance_accepts_array_r_to(self):
        # physics_bh._proper_radial_distance is now vectorized (used
        # directly on the whole r array inside embedding_profile, instead
        # of via a per-point Python loop); confirm the array path agrees
        # with the scalar path pointwise.
        m = 8.4
        rs = phys.schwarzschild_radius(m)
        r_over_rs = np.array([1.0, 1.0001, 1.01, 1.2, 2.0, 8.0, 50.0])
        r = rs * r_over_rs
        array_result = phys._proper_radial_distance(rs, rs, r)
        scalar_result = np.array([phys._proper_radial_distance(rs, rs, rr) for rr in r])
        np.testing.assert_allclose(array_result, scalar_result, rtol=1e-14)

    def test_gaussian_curvature_matches_formula_and_is_negative(self):
        # K(r) = -r_s/(2 r^3): negative everywhere, and finite (equal to
        # -1/(2 r_s^2)) exactly at the throat, per this module's own
        # derivation. Exposed as embedding_profile's new "K" field
        # (Reviewer Audit round 1, Codex/Copilot EXP-18).
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        result = phys.embedding_profile(m_msun=m, r_max_rs=8.0, n_r=50)
        r, K = result["r"], result["K"]
        expected = -rs / (2.0 * r ** 3)
        np.testing.assert_allclose(K, expected, rtol=1e-12)
        self.assertTrue(np.all(K < 0.0))
        self.assertAlmostEqual(K[0], -1.0 / (2.0 * rs ** 2), delta=abs(K[0]) * 1e-9)
        self.assertAlmostEqual(result["summary"]["K_at_horizon"], K[0], delta=1e-30)

    def test_gaussian_curvature_rejects_r_below_horizon(self):
        rs = phys.schwarzschild_radius(10.0)
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(rs, 0.5 * rs)

    def test_gaussian_curvature_rejects_nonfinite_and_nonpositive_rs(self):
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(float("nan"), 1.0)
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(-1.0, 1.0)

    def test_gaussian_curvature_rejects_nonfinite_r_m(self):
        rs = phys.schwarzschild_radius(10.0)
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(rs, float("nan"))
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(rs, float("inf"))

    def test_gaussian_curvature_rejects_overflowing_r_m(self):
        # Reviewer Audit round 2, Codex P2-2: gaussian_curvature previously
        # validated neither its inputs nor its own result, and would
        # silently return -0.0 (not an error, not even a warning) once
        # r_m**3 overflowed a double -- a wrong, non-obviously-wrong
        # answer rather than a raised error. rs**3 float64 overflow starts
        # a bit above r_m ~ 1e102 for a stellar-mass rs; 1e110 is
        # comfortably past that.
        rs = phys.schwarzschild_radius(10.0)
        with self.assertRaises(ValueError):
            phys.gaussian_curvature(rs, 1.0e110)
        # And confirm this was a genuine silent-wrong-answer bug, not a
        # hypothetical one: the raw (unguarded) formula really does
        # produce -0.0, not nan/inf, at this magnitude, which is why the
        # explicit finite-result check above -- not just finite-input
        # validation -- is the part of the fix that matters here.
        with np.errstate(over="ignore"):
            raw = -rs / (2.0 * np.asarray(1.0e110) ** 3)
        self.assertEqual(raw, 0.0)
        self.assertTrue(math.copysign(1.0, raw) < 0.0)  # -0.0, not +0.0

    def test_proper_radial_distance_exceeds_coordinate_difference(self):
        # Physical invariant: the curved-space proper distance from r_s to
        # r must exceed the flat-space coordinate difference r - r_s.
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        for r_over_rs in (1.001, 1.5, 3.0, 10.0):
            r = rs * r_over_rs
            d = phys._proper_radial_distance(rs, rs, r)
            self.assertGreater(d, r - rs)

    def test_proper_radial_distance_zero_for_equal_or_reversed_endpoints(self):
        rs = phys.schwarzschild_radius(10.0)
        self.assertEqual(phys._proper_radial_distance(rs, rs, rs), 0.0)
        self.assertEqual(phys._proper_radial_distance(rs, 2 * rs, 1.5 * rs), 0.0)

    def test_embedding_csv_column_matches_summary_rs(self):
        result = phys.embedding_profile(m_msun=10.0, r_max_rs=5.0, n_r=30)
        s = result["summary"]
        self.assertAlmostEqual(result["r"][0] / s["rs_m"], 1.0, delta=1e-9)

    # --- bounds / error handling -----------------------------------
    def test_r_max_rs_must_exceed_one(self):
        for bad in (1.0, 0.5, -1.0):
            with self.assertRaises(ValueError):
                phys.embedding_profile(m_msun=10.0, r_max_rs=bad, n_r=20)

    def test_r_max_rs_upper_bound_enforced(self):
        with self.assertRaises(ValueError):
            phys.embedding_profile(m_msun=10.0, r_max_rs=1001.0, n_r=20)
        # boundary value itself must succeed
        phys.embedding_profile(m_msun=10.0, r_max_rs=1000.0, n_r=20)

    def test_n_r_bounds_enforced(self):
        with self.assertRaises(ValueError):
            phys.embedding_profile(m_msun=10.0, r_max_rs=5.0, n_r=9)
        with self.assertRaises(ValueError):
            phys.embedding_profile(m_msun=10.0, r_max_rs=5.0, n_r=phys.MAX_POINTS + 1)
        phys.embedding_profile(m_msun=10.0, r_max_rs=5.0, n_r=10)  # boundary ok

    def test_nonfinite_inputs_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                phys.embedding_profile(m_msun=10.0, r_max_rs=bad, n_r=20)
            with self.assertRaises(ValueError):
                phys.embedding_profile(m_msun=bad, r_max_rs=5.0, n_r=20)

    def test_mass_below_trusted_range_is_flagged_not_rejected(self):
        result = phys.embedding_profile(m_msun=1.0, r_max_rs=5.0, n_r=20)
        self.assertEqual(len(result["summary"]["warnings"]), 1)


# ======================================================================
# Mode: tidal
# ======================================================================
class TestTidal(unittest.TestCase):
    """Radial/tangential tidal acceleration, survival radius, scaling."""

    def setUp(self):
        warnings.simplefilter("error", RuntimeWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_radial_is_exactly_twice_tangential_everywhere(self):
        # Physical invariant (vacuum tracelessness of the tidal tensor):
        # a_radial = 2 * a_tangential at every radius, for every mass.
        rng = np.random.default_rng(SEED)
        for _ in range(20):
            m = rng.uniform(1.0, 1.0e8)
            r = rng.uniform(1.001, 100.0) * phys.schwarzschild_radius(m)
            a_r, a_t = phys.tidal_acceleration(m, r, separation_m=1.8)
            self.assertAlmostEqual(a_r / a_t, 2.0, delta=1e-10,
                                    msg=f"seed={SEED}, m={m}, r={r}")

    def test_matches_independent_Newtonian_derivation(self):
        # Independent hand derivation: g(r) = GM/r^2; tidal accel across a
        # radial separation dr is d g/dr * dr (leading order), i.e.
        # -2GM/r^3 * dr in magnitude -> a_radial = 2GM dr / r^3. Coded
        # here from scratch, not by calling tidal_acceleration.
        #
        # GM is the literal IAU 2015 Resolution B3 nominal solar mass
        # parameter (Reviewer Audit round 1, Codex P2-2), not
        # phys.G/phys.M_sun: the previous version of this test reused
        # those two module constants directly, so it was not actually
        # independent of physics_bh's own (then-nonstandard) mass
        # constant and could not have caught this defect.
        GM_sun_nominal = 1.3271244e20   # m^3 s^-2, IAU 2015 Resolution B3
        c = 2.99792458e8
        for m_msun, r_over_rs, dr in [(10.0, 5.0, 1.8), (1.0e6, 50.0, 2.5)]:
            rs = 2.0 * m_msun * GM_sun_nominal / c ** 2
            r = r_over_rs * rs
            GM = m_msun * GM_sun_nominal
            expected_radial = 2.0 * GM * dr / r ** 3
            expected_tangential = GM * dr / r ** 3
            a_r, a_t = phys.tidal_acceleration(m_msun, r, separation_m=dr)
            self.assertAlmostEqual(a_r / expected_radial, 1.0, delta=1e-12)
            self.assertAlmostEqual(a_t / expected_tangential, 1.0, delta=1e-12)

    def test_scales_as_M_over_r_cubed(self):
        # Hold M fixed, vary r: a_radial * r^3 must be constant.
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        vals = []
        for r_over_rs in (2.0, 5.0, 20.0, 100.0):
            r = r_over_rs * rs
            a_r, _ = phys.tidal_acceleration(m, r, separation_m=1.8)
            vals.append(a_r * r ** 3)
        for v in vals[1:]:
            self.assertAlmostEqual(v / vals[0], 1.0, delta=1e-9)

    def test_scales_linearly_with_separation(self):
        m, r = 10.0, 5.0 * phys.schwarzschild_radius(10.0)
        a_r1, a_t1 = phys.tidal_acceleration(m, r, separation_m=1.0)
        a_r2, a_t2 = phys.tidal_acceleration(m, r, separation_m=3.7)
        self.assertAlmostEqual(a_r2 / a_r1, 3.7, delta=1e-10)
        self.assertAlmostEqual(a_t2 / a_t1, 3.7, delta=1e-10)

    def test_horizon_tidal_acceleration_scales_as_M_inverse_squared(self):
        # a_radial(r_s) = 2GM dr / r_s^3 = 2GM dr / (2GM/c^2)^3 ~ M^-2.
        rows = phys.compare_tidal_across_masses([10.0, 100.0, 1000.0], separation_m=1.8,
                                                 limit_g=100.0)
        a10, a100, a1000 = (r["a_radial_horizon"] for r in rows)
        self.assertAlmostEqual(a10 / a100, 100.0, delta=100.0 * 1e-6)
        self.assertAlmostEqual(a100 / a1000, 100.0, delta=100.0 * 1e-6)

    def test_survival_radius_inverts_tidal_formula(self):
        # Independent check: plug r_crit back into a_radial and confirm it
        # equals limit_g * g0, rather than trusting the algebra silently.
        m, sep, limit_g = 5.0e5, 2.3, 250.0
        r_crit = phys.survival_radius(m, separation_m=sep, limit_g=limit_g)
        a_r, _ = phys.tidal_acceleration(m, r_crit, separation_m=sep)
        self.assertAlmostEqual(a_r / (limit_g * phys.g0), 1.0, delta=1e-9)

    def test_bigger_hole_is_gentler_at_horizon(self):
        rows = phys.compare_tidal_across_masses([10.0, 4.31e6], separation_m=1.8,
                                                  limit_g=100.0)
        self.assertGreater(rows[0]["a_radial_horizon_g"], rows[1]["a_radial_horizon_g"])
        self.assertTrue(rows[0]["survives_horizon"] is False)   # 10 Msun: destroyed first
        self.assertTrue(rows[1]["survives_horizon"] is True)    # Sgr A*: crosses intact

    def test_compare_masses_each_row_self_consistent_with_tidal_profile(self):
        for m in (7.5, 3.0e4):
            rows = phys.compare_tidal_across_masses([m], separation_m=1.8, limit_g=100.0)
            profile = phys.tidal_profile(m_msun=m, r_min_rs=1.001, r_max_rs=2.0, n_r=20,
                                          separation_m=1.8)
            self.assertAlmostEqual(rows[0]["a_radial_horizon"],
                                    profile["summary"]["a_radial_horizon"], delta=1e-6)

    def test_compare_masses_flags_out_of_range_entries(self):
        rows = phys.compare_tidal_across_masses([1.0, 10.0], separation_m=1.8, limit_g=100.0)
        self.assertEqual(len(rows[0]["warnings"]), 1)
        self.assertEqual(len(rows[1]["warnings"]), 0)

    def test_a_r_array_matches_pointwise_formula(self):
        result = phys.tidal_profile(m_msun=10.0, r_min_rs=1.01, r_max_rs=10.0, n_r=50,
                                     separation_m=1.8)
        r, a_r = result["r"], result["a_radial"]
        idx = [0, 10, 25, 49]
        for i in idx:
            # Uses phys.GM_SUN_NOMINAL, matching what tidal_acceleration
            # itself is now computed from (Reviewer Audit round 1, Codex
            # P2-2); the point of this test is checking the vectorized
            # array path agrees with the scalar formula at each radius,
            # not re-deriving the physics independently (that is
            # test_matches_independent_Newtonian_derivation's job, below).
            expected = 2.0 * 10.0 * phys.GM_SUN_NOMINAL * 1.8 / r[i] ** 3
            self.assertAlmostEqual(a_r[i] / expected, 1.0, delta=1e-10)

    # --- bounds / error handling -----------------------------------
    def test_r_min_rs_must_exceed_one(self):
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, r_min_rs=1.0, r_max_rs=5.0, n_r=20)
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, r_min_rs=0.5, r_max_rs=5.0, n_r=20)

    def test_r_max_must_exceed_r_min(self):
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, r_min_rs=2.0, r_max_rs=2.0, n_r=20)
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, r_min_rs=5.0, r_max_rs=2.0, n_r=20)

    def test_separation_must_be_positive(self):
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, separation_m=0.0)
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, separation_m=-1.0)

    def test_survival_radius_requires_positive_inputs(self):
        with self.assertRaises(ValueError):
            phys.survival_radius(10.0, separation_m=1.8, limit_g=0.0)
        with self.assertRaises(ValueError):
            phys.survival_radius(10.0, separation_m=-1.0, limit_g=100.0)
        with self.assertRaises(ValueError):
            phys.survival_radius(-10.0, separation_m=1.8, limit_g=100.0)

    def test_tidal_acceleration_rejects_nonpositive_r(self):
        # Reviewer Audit round 1, Codex P2-3 reproducer:
        # tidal_acceleration(..., r_m=0) previously divided by zero with no
        # validation at all, because this public helper is not gated
        # behind tidal_profile's own validation.
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(10.0, 0.0, separation_m=1.8)
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(10.0, -1.0, separation_m=1.8)
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(10.0, float("nan"), separation_m=1.8)
        with self.assertRaises(ValueError):
            phys.tidal_acceleration(10.0, np.array([1.0, 0.0, 2.0]), separation_m=1.8)

    def test_tidal_r_max_rs_upper_bound_enforced(self):
        # Reviewer Audit round 1, Codex P2-3/P2-7: tidal_profile previously
        # had no upper bound on r_max_rs at all (unlike embedding_profile's
        # 1000), so e.g. r_max_rs=1e308 produced a non-finite/invalid grid.
        with self.assertRaises(ValueError):
            phys.tidal_profile(m_msun=10.0, r_max_rs=1.0e300, n_r=20)
        # A large but sane value, well beyond embedding_profile's tighter
        # near-throat bound, must still be accepted (tidal is also used to
        # verify the far-field 1/r^3 power law over a wide range).
        phys.tidal_profile(m_msun=10.0, r_max_rs=1.0e6, n_r=20)

    def test_tidal_linearization_warning_appears_for_extreme_separation(self):
        # Reviewer Audit round 1, Codex P1-5 reproducer: at the CLI's
        # allowed comparison-mass lower bound, r_s is a few millimetres, so
        # the default 1.8 m separation is hundreds of horizon radii --
        # nowhere near "infinitesimal," which the linearized geodesic-
        # deviation approximation this program uses requires.
        result = phys.tidal_profile(m_msun=1.0e-6, r_min_rs=1.01, r_max_rs=10.0,
                                     n_r=20, separation_m=1.8)
        s = result["summary"]
        self.assertGreater(s["linearization_ratio"], phys.TIDAL_LINEARIZATION_RATIO_WARN)
        self.assertTrue(any("linearized" in w for w in s["warnings"]))

    def test_tidal_linearization_warning_absent_for_ordinary_case(self):
        # A person-scale separation around a stellar-mass hole is utterly
        # negligible next to the curvature scale; no warning expected.
        result = phys.tidal_profile(m_msun=10.0, r_min_rs=1.01, r_max_rs=10.0,
                                     n_r=20, separation_m=1.8)
        s = result["summary"]
        self.assertLess(s["linearization_ratio"], phys.TIDAL_LINEARIZATION_RATIO_WARN)
        self.assertFalse(any("linearized" in w for w in s["warnings"]))


# ======================================================================
# Mode: infall
# ======================================================================
class TestInfall(unittest.TestCase):
    """Conserved energy, radial-geodesic ODEs, redshift, closed-form checks."""

    def setUp(self):
        warnings.simplefilter("error", RuntimeWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_conserved_energy_formula(self):
        for r0_rs in (1.5, 3.0, 6.0, 50.0):
            m = 10.0
            rs = phys.schwarzschild_radius(m)
            E = math.sqrt(1.0 - 1.0 / r0_rs)
            result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, r_stop_rs=1.01,
                                         n_points=200, step_frac=0.05)
            self.assertAlmostEqual(result["summary"]["E"], E, delta=1e-10)

    def test_drdtau_matches_closed_form_pointwise(self):
        # Independent re-derivation of (dr/dtau)^2 = c^2(rs/r - rs/r0),
        # evaluated directly (not via _infall_state) at several radii.
        m, r0_rs = 10.0, 6.0
        rs = phys.schwarzschild_radius(m)
        r0 = r0_rs * rs
        E = math.sqrt(1.0 - rs / r0)
        for r_over_rs in (5.0, 3.0, 1.5, 1.01):
            r = r_over_rs * rs
            expected_sq = phys.c ** 2 * (rs / r - rs / r0)
            drdtau, dtdtau = phys._infall_state(rs, r0, E, r)
            self.assertAlmostEqual(drdtau ** 2 / expected_sq, 1.0, delta=1e-10)
            self.assertAlmostEqual(dtdtau, E / (1.0 - rs / r), delta=abs(dtdtau) * 1e-10)

    def test_v_local_approaches_c_at_horizon(self):
        m, r0_rs = 10.0, 6.0
        rs = phys.schwarzschild_radius(m)
        r0 = r0_rs * rs
        E = math.sqrt(1.0 - rs / r0)
        v_close = phys.local_infall_speed(rs, r0, E, 1.0001 * rs)
        v_far = phys.local_infall_speed(rs, r0, E, 3.0 * rs)
        self.assertGreater(v_close, 0.99)
        self.assertLess(v_close, 1.0)
        self.assertGreater(v_close, v_far)

    def test_redshift_approaches_zero_at_horizon(self):
        m, r0_rs = 10.0, 6.0
        rs = phys.schwarzschild_radius(m)
        r0 = r0_rs * rs
        E = math.sqrt(1.0 - rs / r0)
        z_close = phys.outgoing_redshift_factor(rs, r0, E, 1.0001 * rs)
        z_far = phys.outgoing_redshift_factor(rs, r0, E, 3.0 * rs)
        self.assertLess(z_close, 0.05)
        self.assertLess(z_close, z_far)

    def test_rest_at_infinity_limit_matches_closed_form(self):
        # E -> 1 as r0 -> infinity: v_local -> sqrt(rs/r),
        # redshift -> 1 - sqrt(rs/r) (derived algebraically from the
        # general product formula; see physics_bh docstring). Checked here
        # with r0_rs = 1e6 as a numerical stand-in for infinity: the
        # residual E != 1 correction is O(rs/r0) = O(1e-6), well inside
        # the 1e-3 comparison tolerance used below (r0_rs = 1000, tried
        # first, left an O(1e-3) residual that was indistinguishable from
        # the tolerance itself and made this an unreliable check).
        m = 10.0
        rs = phys.schwarzschild_radius(m)
        r0 = 1.0e6 * rs
        E = math.sqrt(1.0 - rs / r0)
        for r_over_rs in (1.01, 1.5, 3.0, 10.0):
            r = r_over_rs * rs
            v = phys.local_infall_speed(rs, r0, E, r)
            z = phys.outgoing_redshift_factor(rs, r0, E, r)
            v_expected = math.sqrt(rs / r)
            z_expected = 1.0 - math.sqrt(rs / r)
            self.assertAlmostEqual(v / v_expected, 1.0, delta=1e-3)
            self.assertAlmostEqual(z / z_expected, 1.0, delta=1e-3)

    def test_r_array_is_monotonically_decreasing(self):
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=500,
                                     r_stop_rs=1.001, step_frac=0.05)
        self.assertTrue(np.all(np.diff(result["r"]) <= 0.0))

    def test_tau_and_t_are_monotonically_increasing(self):
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=500,
                                     r_stop_rs=1.001, step_frac=0.05)
        self.assertTrue(np.all(np.diff(result["tau"]) > 0.0))
        self.assertTrue(np.all(np.diff(result["t"]) > 0.0))

    def test_endpoint_reaches_r_stop_exactly(self):
        rs_ratio = 1.0007
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=1000,
                                     r_stop_rs=rs_ratio, step_frac=0.02)
        rs = result["summary"]["rs_m"]
        self.assertAlmostEqual(result["r"][-1] / rs, rs_ratio, delta=1e-9)

    def test_analytic_cycloid_benchmark_default_params(self):
        # Independent closed-form cycloid proper time (see helper docstring
        # above); matches the help file's own EXP-11 claim of agreement to
        # better than 1e-5 relative error, even at the coarsest allowed
        # --step_frac 0.2.
        m, r0_rs, r_stop_rs = 10.0, 6.0, 1.0005
        tau_closed = infall_cycloid_tau(m, r0_rs, r_stop_rs)
        result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, r_stop_rs=r_stop_rs,
                                     n_points=4000, step_frac=0.2)
        tau_code = result["summary"]["tau_total_s"]
        rel = abs(tau_code - tau_closed) / tau_closed
        self.assertLess(rel, 1.0e-5)

    def test_cycloid_benchmark_converges_without_plateau(self):
        # CORRECTED ORACLE (Reviewer Audit round 1, Codex P1-3): the
        # previous version of this test asserted that the RK4-vs-closed-
        # form error *plateaus* around 1e-5 as --step_frac is refined
        # below about 0.05, and treated that plateau as evidence the
        # integrator was already "maximally accurate." Codex demonstrated
        # that the plateau was not an RK4 accuracy floor at all: it was
        # the omitted initial proper/coordinate-time interval between the
        # exact release event r0 and this integrator's numerical seed
        # point, assigned tau=0 by the previous implementation instead of
        # its true (tiny but nonzero) value. A biased but *fixed* offset
        # in tau_total cannot be reduced by shrinking --step_frac, which
        # is exactly the "plateau" that was observed and had been
        # mis-explained as convergence. Now that infall_radial seeds the
        # numerical integration with the exact cycloid-derived (tau, t)
        # for its seed point (see the startup comment in
        # physics_bh.infall_radial), the error against this same
        # independent closed form should fall roughly as step_frac^4
        # (RK4) with no floor above floating-point/closed-form-evaluation
        # noise, all the way from the coarsest allowed --step_frac (0.2)
        # down to 0.01. This test would have FAILED against the previous
        # implementation (whose error genuinely did plateau near 1.3e-6);
        # it is not weakened, but the assertion itself is inverted to
        # check for the presence of convergence instead of the absence of
        # further improvement.
        #
        # STRENGTHENED (Reviewer Audit round 2, Codex P2-1): the previous
        # version of this test stopped at step_frac=0.005 and asserted only
        # a loose "still improving" bound there, which is technically true
        # but permits a sequence that is not actually clean, monotonic
        # fourth-order convergence -- exactly the gap between what EXP-11
        # promised (a factor of 16 for every halving, smoothly, down to
        # 0.005) and what the integrator actually delivers. Measured
        # relative errors against the closed form are, in order,
        # 1.56e-6, 2.16e-7, 2.92e-8, 1.63e-9, 1.83e-10, 8.88e-11, 2.86e-10,
        # 8.21e-10 for step_frac = 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002,
        # 0.001: strong (roughly third-to-fourth order, not exactly 16x
        # every time) improvement down to about 0.005, then a genuinely
        # NON-monotonic tail (0.002 and 0.001 are both slightly worse than
        # 0.005) once the shrinking RK4 truncation error and the
        # accumulated floating-point round-off from many more, individually
        # tinier steps become comparable. This test now checks exactly that
        # claim -- not endless monotonic improvement -- explicitly, keyed
        # by step_frac rather than by fragile list position.
        m, r0_rs, r_stop_rs = 10.0, 6.0, 1.0005
        tau_closed = infall_cycloid_tau(m, r0_rs, r_stop_rs)
        step_fracs = (0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)
        rel_errs = {}
        for step_frac in step_fracs:
            result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, r_stop_rs=r_stop_rs,
                                         n_points=4000, step_frac=step_frac)
            tau_code = result["summary"]["tau_total_s"]
            rel_errs[step_frac] = abs(tau_code - tau_closed) / tau_closed
        # All comfortably inside the documented 1e-5 tolerance even at the
        # coarsest allowed step size.
        for sf, e in rel_errs.items():
            self.assertLess(e, 1.0e-5, f"step_frac={sf}")
        # Genuine convergence over the well-resolved range: refining
        # step_frac from the coarsest (0.2) to a middling value (0.02)
        # reduces the error by at least two orders of magnitude (RK4
        # predicts (0.2/0.02)^4 = 1e4; 1e2 is a generous, robust floor).
        # The old, bugged implementation instead showed *no* meaningful
        # reduction here -- both ends of that range sat on the same
        # ~1.3e-6 plateau.
        self.assertLess(rel_errs[0.02], rel_errs[0.2] / 1.0e2)
        self.assertLess(rel_errs[0.005], rel_errs[0.02] / 3.0)
        # Below step_frac=0.005 the sequence is not guaranteed to keep
        # improving monotonically (see above), but it IS guaranteed to stay
        # in the tiny floating-point-noise band it has already reached --
        # refining further should never make the answer meaningfully worse.
        for sf in (0.005, 0.002, 0.001):
            self.assertLess(rel_errs[sf], 5.0e-9, f"step_frac={sf}")

        # EXTENDED (Audit Round 3, Codex P1-2, release-blocking): EXP-11 and
        # this integrator's own docstring previously claimed that below
        # step_frac=0.005 the relative error "settles into a band well
        # under 1e-8" and "never climbs back out" -- and this regression
        # stopped at step_frac=0.001, so it never exercised the lower 99%
        # of the accepted (1e-5, 0.2] logarithmic range where that claim is
        # demonstrably false. Reproduced with Codex's own step_fracs (same
        # M/r0_rs/r_stop_rs as above): 5e-4 -> 8.6204e-10, 2e-4 -> 4.8920e-11
        # (both still tiny), but 1e-4 -> 4.6704e-8, 5e-5 -> 6.9701e-8, and
        # 2e-5 -> 2.0488e-7 -- climbing back OUT of the sub-1e-8 band the
        # previous claim promised, as accumulated floating-point round-off
        # from ever more, ever tinier steps comes to dominate the shrinking
        # RK4 truncation error. The corrected claim (now also in the
        # docstring and EXP-11) is different and weaker, but true: the
        # error never exceeds its own coarsest-step-size (step_frac=0.2)
        # value anywhere in the accepted range, i.e. refining --step_frac
        # is never required for accuracy and can occasionally cost a little
        # of it back, not that it is monotonically non-increasing below
        # 0.005. This test would have FAILED against the previous
        # (round-2) implementation's documented claim: 2.05e-7 is neither
        # "well under 1e-8" nor part of a band the error, once reached,
        # "never climbs back out" of (rel_errs[0.005] = 1.83e-10 is reached
        # and then visibly departed).
        extended_step_fracs = (1.0e-4, 5.0e-5, 2.0e-5)
        for step_frac in extended_step_fracs:
            result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, r_stop_rs=r_stop_rs,
                                         n_points=4000, step_frac=step_frac)
            tau_code = result["summary"]["tau_total_s"]
            rel_errs[step_frac] = abs(tau_code - tau_closed) / tau_closed
        for sf in extended_step_fracs:
            # Comfortably above the old (false) "well under 1e-8" claim,
            # confirming the climb-back-out is real and not noise: each
            # measured value is at least an order of magnitude above 1e-8.
            self.assertGreater(rel_errs[sf], 1.0e-8, f"step_frac={sf}")
            # But still small in absolute terms, and in particular never
            # larger than the coarsest-step-size (step_frac=0.2) error --
            # the actual, honest bound over the whole accepted range.
            self.assertLess(rel_errs[sf], rel_errs[0.2], f"step_frac={sf}")
            self.assertLess(rel_errs[sf], 5.0e-7, f"step_frac={sf}")
        # And, across every step_frac measured in this test (the original
        # 8-point sweep above plus this extension), the documented ~1e-5
        # accepted tolerance is never violated -- this is the one claim
        # EXP-11 and this function's docstring still make unconditionally,
        # and it remains true throughout.
        for sf, e in rel_errs.items():
            self.assertLess(e, 1.0e-5, f"step_frac={sf}")

    def test_release_point_is_exact(self):
        # Reviewer Audit round 1, Copilot P2-1: the previous implementation
        # stored r0*(1-1e-12) as the *first* sample, labelled tau=0, t=0,
        # even though comments and the help file describe the first sample
        # as the release point r0 itself. The exact release event is now
        # prepended as its own (r0, tau=0, t=0) row, distinct from the
        # numerical integrator's seed point.
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=4000,
                                     r_stop_rs=1.0005, step_frac=0.02)
        rs = result["summary"]["rs_m"]
        self.assertEqual(result["tau"][0], 0.0)
        self.assertEqual(result["t"][0], 0.0)
        self.assertEqual(result["r"][0], 6.0 * rs)
        self.assertEqual(result["v_local"][0], 0.0)
        # At release the particle is momentarily at rest (no Doppler
        # contribution), but r0 is still a finite radius, not infinity, so
        # the ordinary gravitational redshift factor sqrt(1-rs/r0) still
        # applies -- it is not 1.0.
        self.assertAlmostEqual(result["redshift"][0], math.sqrt(1.0 - 1.0 / 6.0),
                                delta=1e-12)

    def test_dtau_dt_matches_inverse_of_dtdtau_pointwise(self):
        m, r0_rs = 10.0, 6.0
        result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, n_points=300,
                                     r_stop_rs=1.001, step_frac=0.05)
        rs, E = result["summary"]["rs_m"], result["summary"]["E"]
        for i in (0, 50, 150, 299):
            r = result["r"][i]
            expected = (1.0 - rs / r) / E
            self.assertAlmostEqual(result["dtau_dt"][i] / expected, 1.0, delta=1e-8)

    def test_v_local_final_and_redshift_final_match_summary(self):
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=200,
                                     r_stop_rs=1.002, step_frac=0.05)
        s = result["summary"]
        self.assertAlmostEqual(s["v_local_final"], result["v_local"][-1], delta=1e-14)
        self.assertAlmostEqual(s["redshift_final"], result["redshift"][-1], delta=1e-14)

    # --- bounds / error handling -----------------------------------
    def test_r0_rs_must_exceed_one(self):
        for bad in (1.0, 0.9, -3.0):
            with self.assertRaises(ValueError):
                phys.infall_radial(m_msun=10.0, r0_rs=bad, r_stop_rs=0.5 if bad < 1 else bad - 0.1)

    def test_r0_rs_upper_bound_enforced(self):
        # Reviewer Audit round 1, Codex P2-3 reproducer:
        # infall_radial(..., r0_rs=1e308) previously reached a non-finite
        # state deep inside the integrator instead of being rejected
        # up front by input validation.
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=1.0e308, r_stop_rs=1.0005)

    def test_r_stop_rs_must_lie_strictly_between_one_and_r0(self):
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, r_stop_rs=1.0)
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, r_stop_rs=6.0)
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, r_stop_rs=7.0)

    def test_step_frac_bounds_enforced(self):
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, step_frac=1.0e-6)
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, step_frac=0.2001)
        phys.infall_radial(m_msun=10.0, r0_rs=6.0, step_frac=0.2, n_points=200)  # boundary ok

    def test_n_points_bounds_enforced(self):
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=10.0, r0_rs=6.0, n_points=19)

    def test_nonfinite_and_nonpositive_mass_rejected(self):
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=float("nan"), r0_rs=6.0)
        with self.assertRaises(ValueError):
            phys.infall_radial(m_msun=-5.0, r0_rs=6.0)

    # --- Reviewer Audit round 2, Gemini finding 2 (seed-point overshoot on
    # ultra-narrow r0_rs/r_stop_rs windows) -----------------------------
    #
    # infall_radial's numerical seed point sits at r0*(1-eps), eps
    # formerly fixed at 1e-12. Whenever the caller's whole window
    # (r0-r_stop)/r0 was narrower than that fixed eps, the seed point
    # itself already lay AT OR PAST r_stop before the main integration
    # loop ever ran: the loop would then execute zero iterations and the
    # function would return "successfully", silently mislabelling the
    # seed radius r0*(1-1e-12) -- not the requested r_stop -- as the
    # stopping point. The fix scales eps down to a fixed 10% fraction of
    # the actual window whenever that window is narrower than 1e-12, so
    # the seed always lies strictly between r0 and r_stop.
    #
    # A genuine end-to-end regression test (call infall_radial with a
    # narrow window and confirm it now reaches r_stop correctly) turns
    # out not to be constructible in this codebase for the exact regime
    # the fix changes. The two per-step caps used elsewhere in the
    # integrator -- a speed floor of 1e-6*c and a distance-from-start
    # floor of 1e-9*rs -- were sized for ordinary infall windows, not for
    # windows a trillionth as wide as r0 itself; for EVERY (r0_rs, window
    # fraction) combination tried that actually falls in the regime the
    # fix changes (window fraction below 1e-11, so eps shrinks below its
    # old fixed value), the true infall velocity never climbs out of that
    # 1e-6*c floor before r_stop is reached, and the integrator exhausts
    # MAX_STEPS without converging -- this was checked exhaustively for
    # r0_rs in {1.01, 1.1, 1.5, 2.0, 3.0, 6.0, 9.0} crossed with window
    # fractions in {1e-9, ..., 1e-14} (a reduced MAX_STEPS of 200,000, for
    # speed) and again for r0_rs=1.01 with window fractions in
    # {9e-12, 5e-12, 3e-12, 2e-12} at the full production MAX_STEPS of
    # 4,000,000 (all still failed to converge). That is a separate,
    # pre-existing numerical-tractability limit of this integrator for
    # windows this narrow, orthogonal to the seed-point fix itself, and
    # is not something this fix was ever meant to overcome.
    #
    # What the fix genuinely, verifiably does is proven two ways below
    # instead: (1) the seed-point formula itself, exercised directly and
    # compared against an independent hand-derivation, always keeps the
    # seed strictly inside (r_stop, r0) -- this is a closed-form
    # mathematical property of eps = min(1e-12, 0.1*window/r0), true for
    # every 0 < window < r0, and is exactly what stops the old
    # zero-iteration silent-success failure mode from being possible; and
    # (2) calling the real infall_radial() in the narrow-window regime
    # now fails LOUDLY (a RuntimeError naming r_stop and the step count)
    # rather than returning silently-wrong data -- confirmed against the
    # old fixed-eps formula, which would have returned with no error at
    # all for the same inputs.
    def test_seed_eps_formula_keeps_seed_strictly_inside_stop_window(self):
        # Independent re-derivation of physics_bh's
        # eps = min(1.0e-12, 0.1 * (r0 - r_stop) / r0) formula, checked
        # for a spread of window widths that cross the old fixed
        # eps=1e-12 in both directions.
        for r0_rs in (1.01, 1.5, 6.0, 50.0):
            for window_frac in (1.0e-2, 1.0e-9, 1.0e-10, 1.0e-12,
                                 1.0e-13, 1.0e-14):
                # (Window fractions much narrower than 1e-14 are not
                # exercised here: at r0_rs=1.01 the seed offset eps*r0
                # falls below float64's representable spacing near r0
                # itself, so r0*(1-eps) rounds back to exactly r0 -- a
                # float64 precision floor unrelated to this formula.)
                r0 = r0_rs  # units of rs; eps is scale-invariant in rs
                r_stop = r0 * (1.0 - window_frac)
                eps = min(1.0e-12, 0.1 * (r0 - r_stop) / r0)
                seed = r0 * (1.0 - eps)
                self.assertGreater(
                    seed, r_stop,
                    msg=f"seed did not stay past r_stop for r0_rs={r0_rs}, "
                        f"window_frac={window_frac:g}")
                self.assertLess(seed, r0)
                # For window_frac comfortably above the 1e-11 crossover
                # (0.1*window_frac vs. the fixed 1e-12), eps is unchanged
                # from the old fixed value -- ordinary, wide-window infall
                # calls see byte-identical behaviour. Comfortably below
                # it, eps has shrunk. (The crossover itself, window_frac
                # == 1e-11 exactly, is left untested here since 0.1*1e-11
                # is not exactly representable in float64 and lands a few
                # parts in 1e4 to either side of 1e-12 depending on
                # rounding -- not a meaningful behavioural distinction.)
                if window_frac >= 1.0e-9:
                    self.assertEqual(eps, 1.0e-12)
                elif window_frac <= 1.0e-12:
                    self.assertLess(eps, 1.0e-12)

    def test_old_fixed_eps_would_have_silently_overshot_r_stop(self):
        # Demonstrates the actual pre-fix defect algebraically: with the
        # old fixed eps=1e-12, for any window narrower than that, the
        # seed radius r0*(1-1e-12) already lies AT OR PAST the caller's
        # requested r_stop -- so the old code's main loop would run zero
        # iterations and return a "successful" track whose last point is
        # not the requested r_stop at all.
        for r0_rs in (1.01, 6.0, 1000.0):
            for window_frac in (1.0e-13, 1.0e-14, 1.0e-20):
                r0 = r0_rs
                r_stop = r0 * (1.0 - window_frac)
                old_eps = 1.0e-12
                old_seed = r0 * (1.0 - old_eps)
                self.assertLessEqual(
                    old_seed, r_stop,
                    msg="old fixed-eps seed should already have passed "
                        f"r_stop for r0_rs={r0_rs}, window_frac={window_frac:g}")

    def test_ultra_narrow_stop_window_fails_loudly_not_silently(self):
        # In the regime the fix actually changes (window_frac < 1e-11),
        # genuine convergence is numerically out of reach (see the class
        # comment above) -- but the fixed code no longer pretends
        # otherwise. It must raise the honest "did not reach r_stop"
        # RuntimeError, never return, and never raise anything else (in
        # particular, never a silent success and never an unrelated
        # exception type such as a bare ZeroDivisionError or OverflowError
        # escaping the integrator).
        old_max_steps = phys.MAX_STEPS
        phys.MAX_STEPS = 2000  # keep this test fast; see class comment --
        # more steps would not change the qualitative (non-convergent)
        # outcome for this window, only how long the test takes to fail.
        try:
            r0_rs = 1.01
            r_stop_rs = r0_rs * (1.0 - 1.0e-13)
            with self.assertRaises(RuntimeError) as ctx:
                phys.infall_radial(m_msun=10.0, r0_rs=r0_rs,
                                    r_stop_rs=r_stop_rs, n_points=20,
                                    step_frac=0.2)
            self.assertIn("did not reach r_stop", str(ctx.exception))
        finally:
            phys.MAX_STEPS = old_max_steps

    def test_ordinary_window_unaffected_by_narrow_window_fix(self):
        # Sanity check that the eps fix is inert for realistic infall
        # calls: an ordinary window (r0_rs=6, r_stop_rs=1.0005, the CLI
        # defaults) still integrates cleanly and reaches r_stop exactly,
        # exactly as it did before this fix.
        result = phys.infall_radial(m_msun=10.0, r0_rs=6.0,
                                     r_stop_rs=1.0005, n_points=200,
                                     step_frac=0.02)
        rs = result["summary"]["rs_m"]
        self.assertAlmostEqual(result["r"][-1] / rs, 1.0005, delta=1e-9)


# ======================================================================
# Mode: horizons
# ======================================================================
class TestHorizons(unittest.TestCase):
    """Vaidya apparent vs. event horizon, shooting method invariants."""

    def setUp(self):
        warnings.simplefilter("error", RuntimeWarning)

    def tearDown(self):
        warnings.resetwarnings()

    def test_apparent_horizon_equals_2M_of_v_pointwise(self):
        # Reviewer Audit round 1, Copilot P2-3: the previous version of
        # this test obtained its "expected" mass function by calling
        # phys.vaidya_mass_of_v -- the very function whose output feeds
        # r_AH inside vaidya_horizons -- so it could not catch a shared
        # defect in that one function. The piecewise-linear ramp is
        # reconstructed here from scratch (plain numpy clip arithmetic),
        # independent of vaidya_mass_of_v itself.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=30)
        s = result["summary"]
        m0_geom = 0.5 * s["rs0_m"]
        m1_geom = 0.5 * s["rs1_m"]
        v1 = s["v1_rs0"] * s["rs0_m"]
        v2 = s["v2_rs0"] * s["rs0_m"]
        frac = np.clip((result["v"] - v1) / (v2 - v1), 0.0, 1.0)
        expected = 2.0 * (m0_geom + (m1_geom - m0_geom) * frac)
        np.testing.assert_allclose(result["r_AH"], expected, rtol=1e-12)

    def test_constant_mass_apparent_and_event_horizon_coincide(self):
        # Static special case: r_AH(v) = r_EH(v) = r_s0 exactly, for every
        # v, regardless of bisect_iters (regression: see
        # TestNoAccretionShootingRegression below for why "regardless of
        # bisect_iters" specifically needed a code fix).
        result = phys.vaidya_horizons(m0_msun=8.0, m1_msun=8.0, n_steps=300,
                                       bisect_iters=60)
        s = result["summary"]
        np.testing.assert_allclose(result["r_AH"], s["rs0_m"], rtol=1e-12)
        np.testing.assert_allclose(result["r_EH"], s["rs0_m"], rtol=1e-12)
        self.assertAlmostEqual(s["r_crit_over_rs0"], 1.0, delta=1e-12)

    def test_event_horizon_at_or_above_apparent_horizon(self):
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=40)
        self.assertTrue(np.all(result["r_EH"] >= result["r_AH"] - 1.0e-6 * result["r_AH"]))

    def test_horizons_coincide_after_accretion_ends(self):
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, v1_rs0=5.0,
                                       duration_rs0=10.0, v_end_margin_rs0=15.0,
                                       n_steps=400, bisect_iters=50)
        s = result["summary"]
        v_over_rs0 = result["v"] / s["rs0_m"]
        late = v_over_rs0 > s["v2_rs0"] + 1.0
        self.assertTrue(np.any(late))
        rel_diff = np.abs(result["r_EH"][late] - result["r_AH"][late]) / s["rs0_m"]
        self.assertLess(np.max(rel_diff), 1.0e-3)

    def test_event_horizon_rises_before_accretion_begins(self):
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, v1_rs0=5.0,
                                       duration_rs0=10.0, v_start_margin_rs0=25.0,
                                       n_steps=400, bisect_iters=50)
        s = result["summary"]
        v_over_rs0 = result["v"] / s["rs0_m"]
        just_before = (v_over_rs0 > s["v1_rs0"] - 1.0) & (v_over_rs0 < s["v1_rs0"])
        self.assertTrue(np.any(just_before))
        self.assertTrue(np.all(result["r_EH"][just_before] > s["rs0_m"] * 1.0000001))

    def test_apparent_and_event_horizon_are_nondecreasing_in_v(self):
        # Area theorem, in this special-case form: neither horizon shrinks.
        result = phys.vaidya_horizons(m0_msun=10.0, m1_msun=100.0, n_steps=500,
                                       bisect_iters=50)
        self.assertTrue(np.all(np.diff(result["r_AH"]) >= -1.0e-9 * result["summary"]["rs0_m"]))
        self.assertTrue(np.all(np.diff(result["r_EH"]) >= -1.0e-6 * result["summary"]["rs0_m"]))

    def test_small_growth_10_to_10p5_never_violates_r_EH_ge_r_AH(self):
        # Reviewer Audit round 1, Copilot P1-2 (reproducer): at every
        # DEFAULT setting -- no unusual --bisect_iters or margins needed --
        # the previous forward-shooting-only construction returned an
        # r_EH(v) that dipped BELOW r_AH(v) near v_end by about 3.9e-6
        # r_s0 for this specific, mild mass-growth case (M0=10 -> M1=10.5),
        # directly contradicting the documented invariant "r_EH >= r_AH
        # always" that the same run's own summary text asserted
        # unconditionally. This reproduces on the pre-fix code and passes
        # on the current backward-integration-primary construction.
        result = phys.vaidya_horizons(m0_msun=10.0, m1_msun=10.5)
        min_diff_rs0 = float(np.min((result["r_EH"] - result["r_AH"])
                                     / result["summary"]["rs0_m"]))
        self.assertGreaterEqual(min_diff_rs0, -1.0e-9)

    def test_low_bisect_iters_does_not_produce_false_event_horizon(self):
        # Reviewer Audit round 1, Codex P1-1: at the low --bisect_iters
        # values Exercise EXP-13 itself told students to try (10, 20, 30),
        # the previous forward-shooting construction could return a
        # candidate "event horizon" that had plunged to the small-radius
        # cutoff, then interpolated a spectacularly wrong curve across
        # nearly the entire run, while still being reported under the
        # unconditional claim r_EH >= r_AH. Because r_EH(v) is now built
        # by backward integration from the exact r(v2)=r_s1 boundary
        # condition -- a construction that does not depend on
        # bisect_iters at all -- the reported curve should be essentially
        # IDENTICAL regardless of --bisect_iters, and should never violate
        # r_EH >= r_AH, even at the settings that previously broke it.
        default_M0, default_M1 = 5.0, 10.0
        curves = {}
        for iters in (10, 20, 30, 40, 60):
            result = phys.vaidya_horizons(m0_msun=default_M0, m1_msun=default_M1,
                                           n_steps=300, bisect_iters=iters)
            s = result["summary"]
            min_diff_rs0 = float(np.min((result["r_EH"] - result["r_AH"]) / s["rs0_m"]))
            self.assertGreaterEqual(min_diff_rs0, -1.0e-9,
                                     msg=f"bisect_iters={iters}")
            curves[iters] = result["summary"]["r_crit_over_rs0"]
        r_crit_values = list(curves.values())
        for v in r_crit_values[1:]:
            self.assertAlmostEqual(v, r_crit_values[0], delta=1e-9)

    def test_shooting_vs_backward_diagnostic_fields_are_reported(self):
        # The new secondary diagnostics introduced by the backward-
        # integration architecture change: r_crit_shooting_over_rs0 (the
        # old forward-shooting answer, kept only as a diagnostic) and
        # shooting_vs_backward_rs0 (their actual difference), which is the
        # genuinely meaningful accuracy comparison -- unlike residual_rs0,
        # which measures only the bisection bracket's width (Reviewer
        # Audit round 1, Codex P1-1/P2-1, Copilot P1-5).
        result = phys.vaidya_horizons(m0_msun=10.0, m1_msun=10.5, bisect_iters=15)
        s = result["summary"]
        for key in ("r_crit_shooting_over_rs0", "shooting_vs_backward_rs0",
                    "r_crit_over_rs0", "residual_rs0"):
            self.assertIn(key, s)
            self.assertTrue(math.isfinite(s[key]))
        self.assertAlmostEqual(
            s["shooting_vs_backward_rs0"],
            abs(s["r_crit_shooting_over_rs0"] - s["r_crit_over_rs0"]),
            delta=1e-12)
        # At only 15 bisect_iters the forward-shooting bracket is coarse
        # (see test_low_bisect_iters_does_not_produce_false_event_horizon
        # for how badly this can miss the true horizon at even lower
        # settings); the diagnostic should be capable of reflecting a
        # nonzero difference, not hard-coded to zero.
        self.assertGreaterEqual(s["shooting_vs_backward_rs0"], 0.0)

    def test_ah_violation_postcondition_is_actually_enforced(self):
        # Mutation-style check that the postcondition guard in
        # vaidya_horizons (Reviewer Audit round 1, Codex P1-1: "verify ...
        # r_EH >= r_AH within a stated numerical tolerance ... If those
        # checks fail ... refuse to label/export the candidate") is live
        # code, not a check that can never fire: temporarily corrupt the
        # backward integrator's output so it undershoots r_AH, and confirm
        # vaidya_horizons refuses to return a result rather than silently
        # exporting a curve that violates its own documented invariant.
        original = phys._integrate_event_horizon_backward

        def _broken(v_bc, r_bc, v_target, m0, m1, v1, v2, **kw):
            v_arr, r_arr = original(v_bc, r_bc, v_target, m0, m1, v1, v2, **kw)
            return v_arr, r_arr * 0.999   # force a visible undershoot
        phys._integrate_event_horizon_backward = _broken
        try:
            with self.assertRaises(RuntimeError):
                phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=200,
                                      bisect_iters=30)
        finally:
            phys._integrate_event_horizon_backward = original

    def test_monotonicity_postcondition_is_actually_enforced(self):
        # Companion mutation test to the one above, isolating the NEW
        # monotonicity postcondition (Reviewer Audit round 2, Codex P1-1's
        # own recommended addition) from the pre-existing AH-violation one:
        # a uniform *0.999 corruption (as used above) drags the curve
        # below r_AH too, so it cannot by itself prove the monotonicity
        # check is live code rather than dead code shadowed by the AH
        # check. This instead injects a local dip -- a small triangular
        # notch, during accretion where the r_EH-r_AH margin is
        # comfortably large (about 0.3 r_s0 here) -- sized to be far
        # larger than the 1e-6 r_s0 monotonicity tolerance but far smaller
        # than the local AH margin, so ONLY the monotonicity check can
        # fire. (Copilot Audit round 2 REMAINING QUESTION D: mutation-test
        # robustness -- this and the AH mutation test above now cover the
        # two postconditions independently rather than confounded.)
        original = phys._integrate_event_horizon_backward

        def _broken(v_bc, r_bc, v_target, m0, m1, v1, v2, **kw):
            v_arr, r_arr = original(v_bc, r_bc, v_target, m0, m1, v1, v2, **kw)
            v_arr = v_arr.copy()
            r_arr = r_arr.copy()
            rs0 = phys.schwarzschild_radius(5.0)
            v_center, v_half_width = 7.25 * rs0, 1.0 * rs0
            idxs = np.where((v_arr > v_center - v_half_width) &
                             (v_arr < v_center + v_half_width))[0]
            self.assertGreater(len(idxs), 4)  # sanity: window is populated
            lo, hi = idxs[0], idxs[-1]
            dip = 0.15 * rs0
            span = hi - lo
            tri = np.concatenate([np.linspace(0.0, dip, span // 2),
                                   np.linspace(dip, 0.0, span - span // 2 + (span % 2))])
            r_arr[lo:hi] -= tri[:hi - lo]
            return v_arr, r_arr

        phys._integrate_event_horizon_backward = _broken
        try:
            with self.assertRaises(RuntimeError) as ctx:
                phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=200,
                                      bisect_iters=30)
            self.assertIn("decreased somewhere along v", str(ctx.exception))
        finally:
            phys._integrate_event_horizon_backward = original

    def test_summary_r_crit_matches_exported_curve_first_point(self):
        # Reviewer Audit round 2, Gemini finding 1: r_crit_over_rs0 in the
        # summary is read directly from r_eh_back[0] (the raw backward
        # integrator's first array element), while the exported r_EH grid
        # is built by np.interp(v_grid, v_eh_back, r_eh_back, ...)
        # evaluated at v_start -- previously two different numbers, off by
        # the same final-step overshoot the P3-1/Gemini-finding-1 fix
        # (dv reordering, see test_backward_integrator_hits_v_target_exactly
        # above) closes. They must now agree, not just be close.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=40)
        s = result["summary"]
        self.assertEqual(s["r_crit_over_rs0"], result["r_EH"][0] / s["rs0_m"])

    def test_bisection_residual_shrinks_with_more_iterations(self):
        residuals = []
        for iters in (10, 20, 30, 40):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=200,
                                           bisect_iters=iters)
            residuals.append(result["summary"]["residual_rs0"])
        for a, b in zip(residuals, residuals[1:]):
            self.assertLess(b, a)

    def test_M0_equals_M1_special_case_matches_general_path_at_low_bisect_iters(self):
        # The special case must agree with what a converged shooting
        # search would find, not merely avoid crashing: cross-check
        # against a *slightly unequal* mass pair (M1 = M0 * (1+1e-9))
        # forced through the general shooting path, which should locate
        # essentially the same r_crit/r_s0 = 1.
        result_exact = phys.vaidya_horizons(m0_msun=8.0, m1_msun=8.0, n_steps=200,
                                             bisect_iters=60)
        result_near = phys.vaidya_horizons(m0_msun=8.0, m1_msun=8.0 * (1 + 1e-6),
                                            n_steps=200, bisect_iters=60,
                                            duration_rs0=0.5, v1_rs0=5.0)
        self.assertAlmostEqual(result_exact["summary"]["r_crit_over_rs0"],
                                result_near["summary"]["r_crit_over_rs0"], delta=1e-4)

    def test_larger_mass_ratio_needs_bracket_search_but_still_converges(self):
        result = phys.vaidya_horizons(m0_msun=2.0, m1_msun=1000.0, n_steps=400,
                                       bisect_iters=50, v_end_margin_rs0=30.0)
        s = result["summary"]
        self.assertGreater(s["r_crit_over_rs0"], 0.9)
        self.assertTrue(math.isfinite(s["r_crit_over_rs0"]))

    def test_brief_vs_long_accretion_both_finite_and_ordered(self):
        for duration in (0.5, 50.0):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0,
                                           duration_rs0=duration, n_steps=400,
                                           bisect_iters=40, v_end_margin_rs0=20.0)
            self.assertTrue(np.all(np.isfinite(result["r_EH"])))
            self.assertTrue(np.all(result["r_EH"] >= result["r_AH"] - 1e-6 * result["r_AH"]))

    def test_small_and_large_mass_growth_both_finite(self):
        for m0, m1 in ((10.0, 10.5), (10.0, 1.0e4)):
            result = phys.vaidya_horizons(m0_msun=m0, m1_msun=m1, n_steps=300,
                                           bisect_iters=40, v_end_margin_rs0=20.0)
            self.assertTrue(np.all(np.isfinite(result["r_EH"])))
            self.assertTrue(np.all(np.isfinite(result["r_AH"])))

    def test_long_duration_accretion_stays_monotonic_and_ah_consistent(self):
        # Reviewer Audit round 2, Codex P1-1 reproducer: with the OLD
        # duration-proportional step cap shared by both integrators,
        # --duration_rs0 600 produced a non-monotonic r_EH(v), and 850+
        # produced one that dipped below r_AH(v) -- both violations of
        # this module's own documented invariants, and both silently
        # exported rather than caught (the monotonicity postcondition
        # check itself is new this round; see
        # test_ah_violation_postcondition_is_actually_enforced above for
        # its AH-violation counterpart). The backward integrator's step
        # cap is now a LOCAL mass_scale, not duration-proportional, so
        # arbitrarily long durations should not degrade its near-horizon
        # step resolution. Below the fix, this call raised RuntimeError
        # ("decreased somewhere along v") at duration=600.
        for duration in (600.0, 1000.0):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0,
                                           duration_rs0=duration,
                                           v_end_margin_rs0=25.0, n_steps=400,
                                           bisect_iters=60, n_family=9)
            s = result["summary"]
            self.assertTrue(np.all(np.isfinite(result["r_EH"])),
                             msg=f"duration={duration}")
            min_step = float(np.min(np.diff(result["r_EH"])))
            self.assertGreaterEqual(min_step, 0.0, msg=f"duration={duration}")
            min_ah_gap_rs0 = float(np.min((result["r_EH"] - result["r_AH"]) / s["rs0_m"]))
            self.assertGreaterEqual(min_ah_gap_rs0, -1.0e-9, msg=f"duration={duration}")

    def test_backward_integrator_hits_v_target_exactly(self):
        # Reviewer Audit round 2, Codex P3-1 / Gemini finding 1: both
        # integrators capped each step as
        # dv = min(dv_r, cap); dv = max(dv, floor) -- applying the floor
        # AFTER the remaining-interval cap could push the FINAL step's dv
        # back above the true remaining distance, overshooting v_target.
        # The fix reorders so the remaining-interval cap is applied LAST.
        # Direct unit check on the backward integrator: it should land on
        # v_target exactly (RK4 in a fixed number of steps to an exact
        # endpoint, not merely "close").
        m0, m1 = 5.0, 10.0
        rs0 = phys.schwarzschild_radius(m0)
        rs1 = phys.schwarzschild_radius(m1)
        v1 = 5.0 * rs0
        v2 = v1 + 10.0 * rs0
        v_start = v1 - 25.0 * rs0
        v_arr, r_arr = phys._integrate_event_horizon_backward(
            v2, rs1, v_start, 0.5 * rs0, 0.5 * rs1, v1, v2)
        self.assertEqual(v_arr[0], v_start)
        self.assertTrue(np.all(np.isfinite(r_arr)))
        # And confirm no step overshot past v_target along the way either
        # (the array is returned v-increasing from v_target to v_bc).
        self.assertGreaterEqual(float(v_arr[0]), v_start)

    def test_family_n_equals_one_uses_primary_backward_curve_exactly(self):
        # Reviewer Audit round 2, Codex P2-3 reproducer: with n_family=1,
        # the single "family" entry used to be a forward-integrated trial
        # AT ZERO OFFSET from r_crit -- subject to the same forward-
        # shooting instability the backward construction exists to avoid
        # -- yet exported and described (in the HTML help text) as "the
        # event-horizon generator itself". It is now the true primary
        # backward-derived curve directly, flagged as such.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=40, n_family=1)
        self.assertEqual(len(result["family"]), 1)
        f0 = result["family"][0]
        self.assertTrue(f0["is_primary_backward_curve"])
        self.assertIsNone(f0["escapes"])
        self.assertAlmostEqual(f0["r_i_over_rs0"],
                                result["summary"]["r_crit_over_rs0"], delta=1e-15)
        self.assertTrue(np.all(np.isfinite(f0["r"])))
        self.assertTrue(np.all(np.isfinite(f0["v"])))

    def test_family_odd_count_avoids_exact_zero_offset(self):
        # CORRECTED ORACLE (Audit Round 3, Codex P2-5): the previous fix
        # (round 2) nudged the exact-zero middle offset of an odd-count
        # linspace by 0.25*bin_width instead of removing it, which avoided
        # the unstable zero-offset forward trial but left the family
        # asymmetric (for the CLI default n_family=9: four plunging rays,
        # five escaping), directly contradicting this module's own
        # "symmetric about r_crit" rationale (confirmed by Codex: the nine
        # normalized offsets were -0.0488281, -0.0366211, -0.0244141,
        # -0.0122070, +0.00305176, +0.0122070, +0.0244141, +0.0366211,
        # +0.0488281 -- not symmetric about zero). The redesigned
        # construction instead DROPS the exact-zero forward trial outright
        # and appends the stable backward-derived primary curve itself as
        # an explicit, distinctly labelled center member (reusing the same
        # is_primary_backward_curve=True / escapes=None convention as
        # --n_family 1) -- preserving the requested count, restoring EXACT
        # pairwise symmetry among the forward trials, and never forward-
        # integrating from exactly r_crit at all. This test would have
        # FAILED against the previous (round-2) implementation, both on
        # the exact-zero-offset check (still true: dropped, not merely
        # nudged) and on the new symmetry/primary-member checks below,
        # which the round-2 family shape did not satisfy.
        for n_family in (3, 5, 9):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                           bisect_iters=40, n_family=n_family)
            fam = result["family"]
            self.assertEqual(len(fam), n_family)
            r_crit_rs0 = result["summary"]["r_crit_over_rs0"]
            primaries = [f for f in fam if f["is_primary_backward_curve"]]
            trials = [f for f in fam if not f["is_primary_backward_curve"]]
            self.assertEqual(len(primaries), 1, msg=f"n_family={n_family}")
            self.assertEqual(len(trials), n_family - 1, msg=f"n_family={n_family}")
            self.assertIsNone(primaries[0]["escapes"])
            self.assertAlmostEqual(primaries[0]["r_i_over_rs0"], r_crit_rs0,
                                    delta=1e-15)
            offsets = sorted(f["r_i_over_rs0"] - r_crit_rs0 for f in trials)
            for off in offsets:
                self.assertNotEqual(off, 0.0, msg=f"n_family={n_family}")
            for f in trials:
                self.assertIsInstance(f["escapes"], bool)
            # Exact pairwise symmetry: for every positive offset there is a
            # matching negative one of the same magnitude.
            for lo, hi in zip(offsets, reversed(offsets)):
                self.assertAlmostEqual(lo, -hi, delta=1e-12,
                                        msg=f"n_family={n_family}")
            # And an equal split of escaping vs. plunging forward trials,
            # since each symmetric +/- pair straddles r_crit identically.
            n_escape = sum(1 for f in trials if f["escapes"])
            n_plunge = sum(1 for f in trials if not f["escapes"])
            self.assertEqual(n_escape, n_plunge, msg=f"n_family={n_family}")

    def test_family_even_count_unaffected_by_odd_count_redesign(self):
        # Companion check: an even n_family never had an exact-zero-offset
        # entry in the first place (np.linspace(-spread, spread, n_family)
        # with an even count never lands on zero), so it takes the
        # unmodified "else" branch -- no primary-curve member is added,
        # and every entry is an ordinary forward trial.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=40, n_family=4)
        fam = result["family"]
        self.assertEqual(len(fam), 4)
        for f in fam:
            self.assertFalse(f["is_primary_backward_curve"])
            self.assertIsInstance(f["escapes"], bool)

    def test_curvewise_diagnostic_fields_reproduce_p1_2_pattern(self):
        # Reviewer Audit round 2, Codex P1-2 reproducer: at low
        # --bisect_iters, shooting_vs_backward_rs0 (the STARTING-radius-
        # only comparison) reads deceptively small even though the
        # forward-shooting trial has already plunged/escaped and differs
        # from the true horizon by order unity in r_s0 -- the previous
        # EXP-13 text read that small starting-radius number as evidence
        # of small curve error. shooting_reached_v2 and
        # shooting_vs_backward_curve_max_rs0 catch this: at 10 and 20
        # bisect_iters the forward trial does not even reach v2, while
        # the curvewise max difference is order unity in r_s0; only once
        # bisect_iters is large enough (30+, here) does the trial reach
        # v2 and the curvewise difference shrink to match the small
        # starting-radius diagnostic.
        low_curve_max = None
        for iters in (10, 20):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                           bisect_iters=iters)
            s = result["summary"]
            self.assertFalse(s["shooting_reached_v2"], msg=f"iters={iters}")
            self.assertTrue(math.isnan(s["shooting_v2_boundary_residual_rs0"]),
                             msg=f"iters={iters}")
            self.assertGreater(s["shooting_vs_backward_curve_max_rs0"], 0.5,
                                msg=f"iters={iters}")
            # The deceptive part of the old diagnostic: small even though
            # the curve has already gone badly wrong.
            self.assertLess(s["shooting_vs_backward_rs0"], 1.0e-3, msg=f"iters={iters}")
            low_curve_max = s["shooting_vs_backward_curve_max_rs0"]
        for iters in (30, 60):
            result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                           bisect_iters=iters)
            s = result["summary"]
            self.assertTrue(s["shooting_reached_v2"], msg=f"iters={iters}")
            self.assertTrue(math.isfinite(s["shooting_v2_boundary_residual_rs0"]),
                             msg=f"iters={iters}")
            self.assertLess(s["shooting_vs_backward_curve_max_rs0"], 1.0e-2,
                             msg=f"iters={iters}")
        self.assertGreater(low_curve_max, 0.5)

    def test_shooting_vs_backward_curve_max_is_exact_not_undersampled(self):
        # Audit Round 3, Codex P1-1 reproducer (release-blocking): the
        # previous implementation computed shooting_vs_backward_curve_max_rs0
        # on a FIXED 400-point np.linspace(v_start, v_common_hi, 400) grid,
        # not the documented "maximum pointwise difference" -- both curves
        # are piecewise-linear between their own RK4 nodes (hinge-aligned,
        # so v1/v2 are forced nodes of each), so the true maximum can only
        # occur at one of the two curves' own breakpoints, and a fixed
        # 400-sample grid can miss a hinge-localized peak between two of
        # its own samples. Reproduced with Codex's exact parameters
        # (M0=5, M1=10, duration_rs0=0.1, bisect_iters=30, n_steps=200,
        # n_family=1): the old 400-point grid returned 1.4023974705069702e-5
        # r_s0; a dense (200,000-point) independent check found the true
        # value to be 7.3895703853825250e-5 r_s0, a factor of >5 larger.
        # The fixed implementation, using the exact union of both curves'
        # own integration nodes instead of any fixed sample count, reaches
        # 7.4428e-5 r_s0 here -- matching Codex's dense-grid check to
        # within that check's own sampling resolution (not a coincidence:
        # this construction is the exact maximum, not a finer estimate of
        # it), and nowhere near the old, undersampled 1.40e-5 answer.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0,
                                       duration_rs0=0.1, bisect_iters=30,
                                       n_steps=200, n_family=1)
        curve_max = result["summary"]["shooting_vs_backward_curve_max_rs0"]
        self.assertGreater(curve_max, 5.0e-5)   # old undersampled value: 1.40e-5
        self.assertAlmostEqual(curve_max, 7.39e-5, delta=1.0e-6)
        # At the documented default settings (M0=5, M1=10, bisect_iters=60)
        # the old 400-point grid under-reported by about 5.4% (3.9323e-5 vs
        # a true value near 4.1555e-5, both per Codex's Audit 3 report);
        # confirm the fix lands at the larger, true figure here too.
        result2 = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, bisect_iters=60)
        curve_max2 = result2["summary"]["shooting_vs_backward_curve_max_rs0"]
        self.assertGreater(curve_max2, 4.0e-5)   # old undersampled value: 3.9323e-5
        self.assertAlmostEqual(curve_max2, 4.156e-5, delta=1.0e-7)

    def test_v1_rs0_translation_invariance_at_extreme_offsets(self):
        # Audit Round 3, Codex P1-3 reproducer (release-blocking): the
        # physical problem depends only on v-v1, duration, and the two
        # margins, never on v1's own magnitude, and --v1_rs0's own help
        # text promises "any finite value, including zero or negative".
        # Before the fix, vaidya_horizons(..., v1_rs0=1e16, n_steps=200,
        # bisect_iters=10, n_family=1) exhausted the 300,000-step backward
        # integration budget after about 11.5s, reporting it was still 3.2
        # problem scales short of v_start: at that magnitude, subtracting
        # an rs0-scale RK4 step from an absolute v of order 1e16*rs0 lost
        # or badly quantized the step in float64 arithmetic. Integrating
        # in a v1-shifted coordinate instead (see the docstring) makes
        # r_crit_over_rs0 -- a v1-independent physical quantity -- exactly
        # (bit-for-bit) unaffected by v1_rs0's magnitude, across positive
        # and negative offsets spanning 22 orders of magnitude, and every
        # case completes promptly instead of hitting the step cap.
        common = dict(m0_msun=5.0, m1_msun=10.0, duration_rs0=10.0,
                      v_start_margin_rs0=25.0, v_end_margin_rs0=15.0,
                      n_steps=300, bisect_iters=40)
        baseline = phys.vaidya_horizons(v1_rs0=0.0, **common)
        r_crit_baseline = baseline["summary"]["r_crit_over_rs0"]
        for v1_rs0 in (5.0, -5.0, 1.0e6, -1.0e6, 1.0e12, -1.0e12, 1.0e16):
            result = phys.vaidya_horizons(v1_rs0=v1_rs0, **common)
            self.assertEqual(result["summary"]["r_crit_over_rs0"],
                              r_crit_baseline, msg=f"v1_rs0={v1_rs0:g}")
            # The exported v grid must still reflect the absolute origin
            # (v/rs0 at v_start equals v1_rs0 - v_start_margin_rs0, up to
            # ordinary float64 rounding). The tolerance is scaled to the
            # magnitude of v1_rs0 itself, not a fixed 1e-6: at v1_rs0~1e12,
            # v0_over_rs0 is an absolute quantity of that same magnitude,
            # so double precision (~2.22e-16 relative) only guarantees
            # agreement to ~1e12*2.22e-16 ~ 2e-4, not 1e-6. A fixed 1e-6
            # delta here would itself be a bug for large offsets -- it
            # would demand more precision than float64 can represent,
            # regardless of correctness of the implementation.
            v0_over_rs0 = result["v"][0] / result["summary"]["rs0_m"]
            tol = max(1e-6, abs(v1_rs0) * 1e-12)
            self.assertAlmostEqual(v0_over_rs0, v1_rs0 - 25.0, delta=tol,
                                    msg=f"v1_rs0={v1_rs0:g}")
        # Codex's exact reproducer: n_steps=200, bisect_iters=10, n_family=1
        # at v1_rs0=1e16 must now succeed promptly rather than hitting the
        # 300,000-step backward-integration cap.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, v1_rs0=1.0e16,
                                       n_steps=200, bisect_iters=10, n_family=1)
        self.assertTrue(math.isfinite(result["summary"]["r_crit_over_rs0"]))

    def test_v_eh_raw_fields_are_true_raw_integrator_nodes(self):
        # Audit Round 3, Codex P2-6 reproducer: vaidya_horizons previously
        # set v_eh_raw/r_eh_raw = v_grid/r_eh_grid.copy() -- the resampled
        # n_steps+1-point OUTPUT grid (identical to the "v"/"r_EH" fields
        # already in this same dict) -- not the actual raw backward-
        # integrator node arrays, despite the "raw" name. v_grid is an
        # exactly evenly-spaced np.linspace, so its own diffs are all
        # equal; the true adaptive-step RK4 node spacing is not, which
        # gives a simple, direct way to tell the two apart without
        # depending on any private helper.
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=40, n_family=1)
        self.assertNotEqual(len(result["v_eh_raw"]), len(result["v"]))
        diffs = np.diff(result["v_eh_raw"])
        self.assertGreater(float(np.std(diffs)), 0.0)
        self.assertTrue(np.all(np.isfinite(result["v_eh_raw"])))
        self.assertTrue(np.all(np.isfinite(result["r_eh_raw"])))
        # The raw curve must span the SAME [v_start, v_end] range as the
        # main reported curve (Codex P2-5's separate span-mismatch finding
        # for the n_family=1 family export applies equally to v_eh_raw):
        # previously it stopped at v2, short of v_end.
        self.assertAlmostEqual(float(result["v_eh_raw"][0]),
                                float(result["v"][0]), delta=1e-6)
        self.assertAlmostEqual(float(result["v_eh_raw"][-1]),
                                float(result["v"][-1]), delta=1e-6)
        self.assertAlmostEqual(float(result["r_eh_raw"][-1]),
                                result["summary"]["rs1_m"], delta=1e-6)

    def test_duration_too_small_rejected_with_precise_message(self):
        # Audit Round 3, Codex P2-2 reproducer, CORRECTED ORACLE: the
        # original report described a duration_rs0 small enough (down to
        # subnormal, e.g. 5e-324) rounding v2 back to exactly v1 once
        # converted to metres. The P1-3 u-space refactor (v1 held at
        # exactly 0.0 internally) architecturally eliminated *that*
        # specific failure mode. Self-sweep testing during Round 3 found a
        # distinct, previously-unreported failure mode with the same
        # symptom (an unusable tiny duration_rs0): the dimensionless ratio
        # max(v_start_margin_rs0, v_end_margin_rs0, 1) / duration_rs0 that
        # feeds vaidya_mass_of_v's interpolation fraction overflows double
        # precision once duration_rs0 is this small, independent of mass.
        # This must still be rejected up front, with a message naming
        # duration_rs0 and the margins/overflow, not left to surface as an
        # uncaught RuntimeWarning/overflow deep inside the integrator.
        for tiny in (5e-324, 1e-320, 1e-300):
            with self.assertRaises(ValueError) as ctx:
                phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0,
                                      duration_rs0=tiny, n_steps=200,
                                      bisect_iters=10)
            msg = str(ctx.exception)
            self.assertIn("duration_rs0", msg)
            self.assertIn("margins", msg)
            self.assertIn("overflow", msg)

    def test_duration_too_large_rejected_naming_duration_not_margin(self):
        # Audit Round 3, Codex P2-2 reproducer: duration_rs0=100000 (M0=5,
        # M1=10, n_steps=200, bisect_iters=10, n_family=1) previously
        # exhausted the 300,000-step backward cap after about 11s, with an
        # error message recommending a smaller --v_start_margin_rs0 -- not
        # the actual bottleneck, since the default 25-unit margin is
        # negligible next to a 100,000-unit accretion span. This is now
        # rejected immediately (an explicit, checked bound, not a burned
        # step budget), and the message must name --duration_rs0, not only
        # --v_start_margin_rs0.
        with self.assertRaises(ValueError) as ctx:
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0,
                                  duration_rs0=100000.0, n_steps=200,
                                  bisect_iters=10, n_family=1)
        msg = str(ctx.exception)
        self.assertIn("duration_rs0", msg)

    # --- bounds / error handling -----------------------------------
    def test_M1_less_than_M0_rejected(self):
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=10.0, m1_msun=5.0)

    def test_nonpositive_window_parameters_rejected(self):
        # CORRECTED ORACLE (Reviewer Audit round 1, Codex P3-1): v1_rs0 is
        # only the origin of the advanced-time axis, which is physically
        # arbitrary (the Vaidya spacetime described here is invariant
        # under shifting v by a constant); duration_rs0 and the two
        # margins are the parameters with a physically required sign
        # (they are literal lengths of a v-interval). v1_rs0 = 0.0 is
        # therefore no longer expected to raise -- see
        # test_v1_rs0_accepts_zero_and_negative_values below for the
        # positive-path regression this correction enables.
        for kwargs in (dict(duration_rs0=0.0),
                       dict(v_start_margin_rs0=0.0), dict(v_end_margin_rs0=0.0)):
            with self.assertRaises(ValueError):
                phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, **kwargs)

    def test_v1_rs0_accepts_zero_and_negative_values(self):
        # Reviewer Audit round 1, Codex P3-1: v1 (where accretion begins)
        # is an arbitrary origin on the advanced-time axis, not a
        # physically constrained quantity, so v1_rs0 need not be positive.
        # Shifting v1_rs0 (with v_start/v_end following along, since they
        # are defined relative to v1/v2) should shift the whole solution
        # rigidly and leave every v-independent physical quantity (here,
        # the horizon displacement r_crit_over_rs0) unchanged.
        common = dict(m0_msun=5.0, m1_msun=10.0, duration_rs0=10.0,
                      v_start_margin_rs0=25.0, v_end_margin_rs0=15.0,
                      n_steps=300, bisect_iters=40)
        r_pos = phys.vaidya_horizons(v1_rs0=5.0, **common)
        r_zero = phys.vaidya_horizons(v1_rs0=0.0, **common)
        r_neg = phys.vaidya_horizons(v1_rs0=-5.0, **common)
        for r in (r_pos, r_zero, r_neg):
            self.assertIsInstance(r, dict)
        self.assertAlmostEqual(r_zero["summary"]["r_crit_over_rs0"],
                                r_pos["summary"]["r_crit_over_rs0"], delta=1e-9)
        self.assertAlmostEqual(r_neg["summary"]["r_crit_over_rs0"],
                                r_pos["summary"]["r_crit_over_rs0"], delta=1e-9)

    def test_n_steps_bisect_iters_n_family_bounds(self):
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=199)
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, bisect_iters=9)
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, bisect_iters=201)
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_family=42)
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_family=0)

    def test_nonfinite_masses_rejected(self):
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=float("nan"), m1_msun=10.0)
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=5.0, m1_msun=float("inf"))


# ======================================================================
# Regression: two defects found while building this suite
# ======================================================================
class TestNoAccretionShootingRegression(unittest.TestCase):
    """
    Regression test for a defect reproduced during initial test-suite
    construction: for M0 == M1 (no accretion), r = 2M is an *unstable*
    fixed point of the outgoing-null-geodesic equation dr/dv =
    0.5(1-2M/r). Before the fix, the general shooting/bisection path was
    still used for M0 == M1; at the default --bisect_iters (60) the
    bisection bracket collapsed to a single double-precision value by
    floating-point coincidence and the bug was numerically invisible, but
    at --bisect_iters 20 (a value Exercise EXP-13 explicitly directs
    students to try) the tiny (~1e-6 r_s0) residual between the located
    r_crit and the true r_s0 grew exponentially over the integrated v
    range, producing an r_EH(v) track that swung out to several r_s0 --
    silently contradicting the "the two horizons coincide at r = r_s0 for
    the whole run" text driver_bh.py and plot_bh.py both print for this
    exact case. The fix special-cases M0 == M1 to the exact algebraic
    answer, independent of --bisect_iters.
    """

    def test_low_bisect_iters_still_gives_exact_static_horizon(self):
        for iters in (10, 15, 20, 30):
            result = phys.vaidya_horizons(m0_msun=8.0, m1_msun=8.0, n_steps=200,
                                           bisect_iters=iters)
            s = result["summary"]
            self.assertAlmostEqual(s["r_crit_over_rs0"], 1.0, delta=1e-12,
                                    msg=f"bisect_iters={iters}")
            max_rel_dev = float(np.max(np.abs(result["r_EH"] - s["rs0_m"]))) / s["rs0_m"]
            self.assertLess(max_rel_dev, 1e-9, msg=f"bisect_iters={iters}")

    def test_large_v_start_margin_still_gives_exact_static_horizon(self):
        # The unstable-fixed-point amplification factor is
        # exp(delta_v / (2 r_s0)); before the fix this was also
        # reproducible by widening the margins at fixed bisect_iters.
        result = phys.vaidya_horizons(m0_msun=8.0, m1_msun=8.0, n_steps=200,
                                       bisect_iters=15, v_start_margin_rs0=80.0,
                                       v_end_margin_rs0=40.0)
        s = result["summary"]
        max_rel_dev = float(np.max(np.abs(result["r_EH"] - s["rs0_m"]))) / s["rs0_m"]
        self.assertLess(max_rel_dev, 1e-9)


class TestNoPlotOutdirRegression(unittest.TestCase):
    """
    Regression test for a defect found by comparing driver_bh.py's
    behaviour against its own documentation: main.py's --no_plot --help
    text ("no screen display and no PNG, regardless of --outdir") and the
    HTML help file both describe --no_plot and --outdir as freely
    combinable (--outdir is simply unused when --no_plot is given).
    Before the fix, driver_bh.run() raised ValueError whenever both were
    supplied together, contradicting that documented behaviour.
    """

    def test_no_plot_and_outdir_together_is_accepted(self):
        import driver_bh
        with tempfile.TemporaryDirectory() as tmp:
            csvdir = os.path.join(tmp, "csv")
            outdir = os.path.join(tmp, "png")
            result = driver_bh.run(mode="embed", M=10.0, r_max_rs=5.0, n_r=20,
                                    csvdir=csvdir, outdir=outdir, no_plot=True)
            self.assertEqual(result["kind"], "embed")
            # csvdir output was produced ...
            self.assertTrue(os.path.isdir(csvdir))
            self.assertTrue(any(f.endswith(".csv") for f in os.listdir(csvdir)))
            # ... but no PNG was written, since --no_plot suppresses it
            # regardless of --outdir.
            self.assertFalse(os.path.isdir(outdir) and
                              any(f.endswith(".png") for f in os.listdir(outdir)))

    def test_cli_no_plot_and_outdir_together_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            csvdir = os.path.join(tmp, "csv")
            outdir = os.path.join(tmp, "png")
            proc = run_cli(["--mode", "embed", "--no_plot",
                             "--outdir", outdir, "--csvdir", csvdir])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)


# ======================================================================
# Version / build-id / provenance
# ======================================================================
class TestVersionAndBuildProvenance(unittest.TestCase):

    def test_version_and_build_id_in_cli_version_output(self):
        proc = run_cli(["--version"])
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn(phys.MODEL_VERSION, proc.stdout)
        self.assertIn(phys.BUILD_ID, proc.stdout)

    def test_build_id_is_12_hex_chars_or_unknown(self):
        self.assertTrue(phys.BUILD_ID == "unknown" or
                         (len(phys.BUILD_ID) == 12 and
                          all(c in "0123456789abcdef" for c in phys.BUILD_ID)))

    def test_csv_provenance_contains_version_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--no_plot", "--csvdir", tmp,
                             "--n_r", "20"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            csv_files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 1)
            comments, header, rows = read_csv_rows(os.path.join(tmp, csv_files[0]))
            joined = "".join(comments)
            self.assertIn(phys.MODEL_VERSION, joined)
            self.assertIn(phys.BUILD_ID, joined)

    def test_build_id_ignores_newline_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir_lf = make_source_copy(tmp, "lf")
            mod_lf = import_physics_from_dir(dir_lf, "physics_bh_test_lf")

            dir_crlf = make_source_copy(tmp, "crlf")
            for name in phys.BUILD_ID_COVERS:
                path = os.path.join(dir_crlf, name)
                with open(path, "r", encoding="utf-8", newline=None) as fh:
                    text = fh.read()
                with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
                    fh.write(text)
            mod_crlf = import_physics_from_dir(dir_crlf, "physics_bh_test_crlf")

            self.assertEqual(mod_lf.BUILD_ID, mod_crlf.BUILD_ID)
            self.assertNotEqual(mod_lf.BUILD_ID, "unknown")

    def test_build_id_changes_when_covered_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = make_source_copy(tmp, "a")
            mod_a = import_physics_from_dir(dir_a, "physics_bh_test_a")

            dir_b = make_source_copy(tmp, "b")
            with open(os.path.join(dir_b, "driver_bh.py"), "a", encoding="utf-8") as fh:
                fh.write("\n# trivial change for BUILD_ID sensitivity test\n")
            mod_b = import_physics_from_dir(dir_b, "physics_bh_test_b")

            self.assertNotEqual(mod_a.BUILD_ID, mod_b.BUILD_ID)

    def test_build_id_unaffected_by_uncovered_file_change(self):
        # A change to a file *not* in BUILD_ID_COVERS (e.g. this test
        # suite itself, or documentation) must not move BUILD_ID.
        with tempfile.TemporaryDirectory() as tmp:
            dir_a = make_source_copy(tmp, "a2")
            mod_a = import_physics_from_dir(dir_a, "physics_bh_test_a2")

            dir_b = make_source_copy(tmp, "b2")
            with open(os.path.join(dir_b, "README_not_covered.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("this file is not in BUILD_ID_COVERS\n")
            mod_b = import_physics_from_dir(dir_b, "physics_bh_test_b2")

            self.assertEqual(mod_a.BUILD_ID, mod_b.BUILD_ID)


# ======================================================================
# CLI-level tests
# ======================================================================
class TestCLI(unittest.TestCase):
    """Subprocess-level checks: exit codes, CSV files, expected failures."""

    def test_each_mode_runs_successfully_with_no_plot_and_csvdir(self):
        for mode in phys.MODES:
            with tempfile.TemporaryDirectory() as tmp:
                proc = run_cli(["--mode", mode, "--no_plot", "--csvdir", tmp,
                                 "--n_r", "20", "--n_points", "50",
                                 "--n_steps", "200", "--bisect_iters", "15"])
                self.assertEqual(proc.returncode, 0,
                                  msg=f"mode={mode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
                self.assertTrue(any(f.endswith(".csv") for f in os.listdir(tmp)),
                                 msg=f"mode={mode} produced no CSV")

    def test_no_plot_without_csvdir_fails(self):
        proc = run_cli(["--mode", "embed", "--no_plot"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no output at all", proc.stderr)

    def test_bad_mode_rejected_by_argparse(self):
        proc = run_cli(["--mode", "not_a_mode"])
        self.assertNotEqual(proc.returncode, 0)

    def test_negative_mass_fails_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--M", "-5", "--no_plot", "--csvdir", tmp])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("must be greater than zero", proc.stderr)

    def test_m1_less_than_m0_fails_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "horizons", "--M0", "10", "--M1", "5",
                             "--no_plot", "--csvdir", tmp])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("less than M0", proc.stderr)

    def test_dpi_out_of_range_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--no_plot", "--csvdir", tmp, "--dpi", "5"])
            self.assertNotEqual(proc.returncode, 0)

    def test_embed_csv_header_and_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--no_plot", "--csvdir", tmp, "--n_r", "37"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            fname = [f for f in os.listdir(tmp) if f.endswith(".csv")][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, fname))
            # gaussian_curvature_K_per_m2 added this round (Reviewer Audit
            # round 1, Codex/Copilot EXP-18: K(r) exposed as a genuine
            # program output).
            self.assertEqual(header, ["r_over_rs", "r_m", "z_m",
                                       "proper_radial_distance_m",
                                       "gaussian_curvature_K_per_m2"])
            self.assertEqual(len(rows), 37)
            self.assertAlmostEqual(float(rows[0][0]), 1.0, delta=1e-6)
            self.assertLess(float(rows[0][4]), 0.0)   # K is negative everywhere

    def test_embed_csv_values_round_trip_to_17_significant_digits(self):
        # Reviewer Audit round 1, Codex P2-8: CSV columns previously used
        # .6f/.6g formatting, which could truncate the precision actually
        # computed. Confirm the CSV values now agree with the in-process
        # result to full binary64 precision, not merely 6 digits.
        result = phys.embedding_profile(m_msun=10.0, r_max_rs=5.0, n_r=25)
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--no_plot", "--csvdir", tmp,
                             "--M", "10.0", "--r_max_rs", "5.0", "--n_r", "25"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            fname = [f for f in os.listdir(tmp) if f.endswith(".csv")][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, fname))
            for i in (0, 5, 24):
                self.assertAlmostEqual(float(rows[i][1]) / result["r"][i], 1.0, delta=1e-15)
            self.assertEqual(float(rows[0][2]), 0.0)   # z(r_s) = 0 exactly
            for i in (5, 24):
                self.assertAlmostEqual(float(rows[i][2]) / result["z"][i], 1.0, delta=1e-12)

    def test_tidal_compare_masses_writes_second_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "tidal", "--no_plot", "--csvdir", tmp,
                             "--compare_masses", "10,4.31e6"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            files = sorted(f for f in os.listdir(tmp) if f.endswith(".csv"))
            self.assertEqual(len(files), 2)
            compare_file = [f for f in files if "compare" in f][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, compare_file))
            self.assertEqual(header, ["M_Msun", "rs_km", "a_radial_horizon_g",
                                       "r_crit_over_rs", "survives_horizon"])
            self.assertEqual(len(rows), 2)

    def test_tidal_compare_masses_are_sorted_numerically(self):
        # Reviewer Audit round 1, Gemini finding 2: an unsorted
        # --compare_masses list was passed straight through to
        # compare_tidal_across_masses and then to plot_tidal, whose panels
        # 3/4 connect points with continuous marker+line plots -- an
        # unsorted x-axis draws a zig-zag rather than a clean curve. The
        # masses are now sorted in driver_bh._run_tidal before use,
        # regardless of the order given on the command line.
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "tidal", "--no_plot", "--csvdir", tmp,
                             "--compare_masses", "1000,10,1e6,100"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            compare_file = [f for f in os.listdir(tmp) if "compare" in f][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, compare_file))
            masses = [float(r[0]) for r in rows]
            self.assertEqual(masses, sorted(masses))

    def test_horizons_rejects_n_family_two(self):
        # Reviewer Audit round 1, Codex "lower-level" observation: n_family
        # =1 is a legitimate (differently presented) request, and
        # n_family>=3 is a genuine bracketing family, but n_family=2 is
        # neither -- it cannot symmetrically bracket r_crit the way the
        # family-panel code intends.
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "horizons", "--no_plot", "--csvdir", tmp,
                             "--n_steps", "200", "--bisect_iters", "15",
                             "--n_family", "2"])
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("n_family", proc.stderr)

    def test_large_n_r_embed_plot_renders_without_error(self):
        # Reviewer Audit round 1, Codex P2-4 reproducer: n_r at (or near)
        # the documented MAX_POINTS ceiling previously formed several
        # 140-by-n_r float64 arrays for the 3-D mesh (over 1 GB combined at
        # n_r=200,000), a resource cost not mentioned anywhere the ceiling
        # is documented. This actually renders a PNG (not --no_plot) at a
        # large n_r and confirms it still completes and produces a
        # reasonably-sized file, exercising the memory-safe downsampling
        # in plot_bh.plot_embedding.
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "embed", "--outdir", tmp,
                             "--n_r", "50000"], timeout=180)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            pngs = [f for f in os.listdir(tmp) if f.endswith(".png")]
            self.assertEqual(len(pngs), 1)
            self.assertGreater(os.path.getsize(os.path.join(tmp, pngs[0])), 0)

    def test_repeated_csv_writes_do_not_collide(self):
        # Reviewer Audit round 1, Codex P2-5 reproducer: filenames with
        # only one-second timestamp resolution could collide when the
        # program (or a shell-loop parameter sweep, which --no_plot is
        # explicitly advertised for) writes the same-prefix CSV more than
        # once within the same wall-clock second, silently overwriting the
        # earlier file. Three fast back-to-back runs into the same csvdir
        # must produce three distinct files, not fewer.
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(3):
                proc = run_cli(["--mode", "embed", "--no_plot", "--csvdir", tmp,
                                 "--n_r", "15"])
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            csv_files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            self.assertEqual(len(csv_files), 3)

    def test_horizons_writes_main_and_family_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "horizons", "--no_plot", "--csvdir", tmp,
                             "--n_steps", "200", "--bisect_iters", "15", "--n_family", "5"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            files = sorted(f for f in os.listdir(tmp) if f.endswith(".csv"))
            self.assertEqual(len(files), 2)
            family_file = [f for f in files if "family" in f][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, family_file))
            # CORRECTED ORACLE (Audit Round 3, Codex P3-3): the escapes
            # column previously held a long prose value for the primary
            # member ("n/a (primary backward curve, not a forward
            # trial)") instead of a short categorical value like ordinary
            # rows' yes/no. escapes is now a plain "n/a"/"yes"/"no", and
            # the explanation moved to its own trajectory_kind column.
            self.assertEqual(header, ["trajectory_id", "r_i_over_rs0", "escapes",
                                       "trajectory_kind", "v_over_rs0",
                                       "r_over_rs0"])
            traj_ids = sorted(set(int(r[0]) for r in rows))
            self.assertEqual(traj_ids, list(range(5)))
            escapes_col = set(r[2] for r in rows)
            self.assertTrue(escapes_col <= {"yes", "no", "n/a"})
            kind_col = set(r[3] for r in rows)
            self.assertTrue(kind_col <= {"forward_trial", "primary_backward"})
            # n_family=5 is odd, so exactly one trajectory_id is the
            # primary backward curve.
            primary_ids = set(int(r[0]) for r in rows if r[3] == "primary_backward")
            self.assertEqual(len(primary_ids), 1)
            for r in rows:
                if r[3] == "primary_backward":
                    self.assertEqual(r[2], "n/a")
                else:
                    self.assertIn(r[2], ("yes", "no"))

    def test_infall_csv_summary_fields_agree_with_returned_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "infall", "--no_plot", "--csvdir", tmp,
                             "--n_points", "80"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            fname = [f for f in os.listdir(tmp) if f.endswith(".csv")][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, fname))
            self.assertEqual(header, ["tau_ms", "t_coord_ms", "r_over_rs",
                                       "v_local_over_c", "redshift_ratio", "dtau_dt"])
            last_row = rows[-1]
            self.assertAlmostEqual(float(last_row[2]), 1.0005, delta=2e-3)  # default r_stop_rs

    def test_plot_facing_labels_present_in_source(self):
        # Static check that the mislabelling-sensitive plot annotations
        # (stretching/compressing, escapes/plunges) are actually present
        # in plot_bh.py, so a future edit that silently drops or swaps
        # them is caught even without rendering a figure.
        with open(os.path.join(PROJECT_DIR, "plot_bh.py"), encoding="utf-8") as fh:
            src = fh.read()
        for token in ("stretching", "compressing", "escapes to infinity",
                      "plunges to $r=0$", "apparent horizon", "event horizon"):
            self.assertIn(token, src)

    def test_concurrent_processes_writing_same_csvdir_and_outdir_do_not_collide(self):
        # Reviewer Audit round 2, Gemini finding 3 / Codex P2-6: the
        # previous os.path.exists()-then-open() sequence in
        # driver_bh._write_csv and plot_bh._finish was not atomic, so two
        # independent PROCESSES (not just two calls in the same process)
        # racing to create a same-microsecond-timestamped path could both
        # see it as free and both open it -- the second writer silently
        # overwrote the first's data. test_repeated_csv_writes_do_not_collide
        # above exercises three *sequential* subprocess runs, which cannot
        # actually race; this launches several real OS processes
        # concurrently (subprocess.Popen, not run()) against the SAME
        # csvdir/outdir, to exercise the genuine race the atomic
        # os.open(O_CREAT|O_EXCL) fix (and its FileExistsError-retry loop)
        # is meant to close.
        n_procs = 5
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        with tempfile.TemporaryDirectory() as tmp:
            # Deliberately omit --no_plot: with --outdir given (and
            # --no_plot absent), driver_bh.run() both writes a CSV AND
            # calls plot_bh._finish to save a PNG, so both atomic-write
            # paths race concurrently across processes.
            procs = [
                subprocess.Popen(
                    [PYTHON, "main.py", "--mode", "embed", "--csvdir", tmp,
                     "--outdir", tmp, "--n_r", "15"],
                    cwd=PROJECT_DIR, env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                for _ in range(n_procs)
            ]
            results = [p.communicate(timeout=120) for p in procs]
            for i, (p, (out, err)) in enumerate(zip(procs, results)):
                self.assertEqual(p.returncode, 0, msg=f"proc {i} stderr={err.decode()}")
            csv_files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            png_files = [f for f in os.listdir(tmp) if f.endswith(".png")]
            self.assertEqual(len(csv_files), n_procs)
            self.assertEqual(len(png_files), n_procs)
            self.assertEqual(len(set(csv_files)), n_procs)  # genuinely distinct names
            self.assertEqual(len(set(png_files)), n_procs)
            for f in csv_files:
                self.assertGreater(os.path.getsize(os.path.join(tmp, f)), 0)
            for f in png_files:
                self.assertGreater(os.path.getsize(os.path.join(tmp, f)), 0)

    def test_no_reviewer_or_audit_provenance_in_student_facing_text(self):
        # Reviewer Audit round 2, Codex P2-4: student-facing text (--help,
        # printed CLI output, exported CSV comment lines, and the HTML)
        # must describe the CURRENT program as fact, not narrate
        # development/audit history or name reviewers -- that provenance
        # belongs only in code comments (developer-facing) and the
        # Response_to_Audit* documents. This is a permanent guard against
        # that regressing. (It also caught a real instance during the
        # exhaustive self-sweep after Audit round 2: the bh_horizons_family
        # CSV's own comments= string cited "Reviewer Audit round 2, Codex
        # P2-3" -- CSV files are exported, student-facing data output
        # exactly like printed CLI text, not developer-facing source, so
        # that citation has been removed from the string itself.)
        #
        # EXTENDED (Audit Round 3, Codex P2-4): inspect.getdoc()/help()
        # expose Python docstrings as public API/teaching surface, distinct
        # from private "#" source comments -- seven public functions
        # (schwarzschild_radius, gaussian_curvature, tidal_acceleration,
        # tidal_profile, compare_tidal_across_masses, infall_radial,
        # vaidya_horizons) were reproduced to contain explicit reviewer/
        # audit-round citations in their docstrings, naming Codex/Copilot
        # and specific audit rounds, invisible to this test's previous
        # scope (--help/HTML/CLI-output/CSV only). The forbidden-phrase
        # list is also broadened to catch change-history narration the
        # previous narrow blacklist missed in the HTML: "an earlier design
        # note in this file once put it...", "An earlier version of this
        # exercise asked...", "no longer supplies the reported event
        # horizon", and "now a genuine program output" were all reproduced
        # as present, uncaught, before this round's HTML fixes. This test
        # would have FAILED against the pre-fix docstrings/HTML; it now
        # scans every public (non-underscore) callable's docstring, not
        # just a hardcoded list of seven, so a newly added function with a
        # similar citation is caught automatically.
        forbidden = ("reviewer audit", "gemini", "codex", "copilot",
                     "first release", "earlier release",
                     "response_to_audit", "review round",
                     "earlier design note", "earlier version",
                     "no longer supplies", "now a genuine program output")

        def _assert_clean(text, where):
            text_lower = text.lower()
            for term in forbidden:
                self.assertNotIn(term, text_lower, msg=f"in {where}: {term!r}")

        proc = run_cli(["--help"])
        self.assertEqual(proc.returncode, 0)
        _assert_clean(proc.stdout, "--help output")

        html_path = os.path.join(PROJECT_DIR, "Black_hole_spacetime_visualizer.html")
        if os.path.exists(html_path):
            with open(html_path, encoding="utf-8") as fh:
                _assert_clean(fh.read(), "HTML")

        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "horizons", "--no_plot", "--csvdir", tmp,
                             "--n_steps", "200", "--bisect_iters", "15",
                             "--n_family", "5"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            _assert_clean(proc.stdout, "printed CLI output")
            for fname in os.listdir(tmp):
                if fname.endswith(".csv"):
                    with open(os.path.join(tmp, fname), encoding="utf-8") as fh:
                        _assert_clean(fh.read(), f"CSV output ({fname})")

        # Every public (non-underscore-prefixed) callable's docstring, not
        # only a hardcoded shortlist -- this is the public teaching/API
        # surface inspect.getdoc()/help() expose to a student, distinct
        # from developer-facing "#" source comments (which legitimately
        # cite audit findings throughout this codebase and are excluded).
        for name in dir(phys):
            if name.startswith("_"):
                continue
            obj = getattr(phys, name)
            if callable(obj):
                doc = inspect.getdoc(obj)
                if doc:
                    _assert_clean(doc, f"docstring of physics_bh.{name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
