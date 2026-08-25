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
    """
    rs = phys.schwarzschild_radius(m_msun)
    r0 = r0_rs * rs
    r_stop = r_stop_rs * rs
    GM = phys.G * (m_msun * phys.M_sun)
    eta_stop = math.acos(2.0 * r_stop / r0 - 1.0)
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
        # Hand-computed from CODATA-style constants, independent of the
        # module's own G/c/M_sun constants being wired up correctly.
        G = 6.67430e-11
        c = 2.99792458e8
        Msun = 1.98892e30
        expected = 2.0 * G * (10.0 * Msun) / c ** 2
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
        m = 8.4
        rs = phys.schwarzschild_radius(m)
        for r_over_rs in (1.0001, 1.01, 1.2, 2.0, 8.0, 50.0):
            r = rs * r_over_rs
            coded = phys._proper_radial_distance(rs, rs, r)
            closed = proper_radial_distance_closed_form(rs, r)
            rel = abs(coded - closed) / closed
            self.assertLess(rel, 1e-6, msg=f"r/rs={r_over_rs}: coded={coded}, closed={closed}")

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
        G, M_sun = phys.G, phys.M_sun
        for m_msun, r_over_rs, dr in [(10.0, 5.0, 1.8), (1.0e6, 50.0, 2.5)]:
            rs = phys.schwarzschild_radius(m_msun)
            r = r_over_rs * rs
            M_kg = m_msun * M_sun
            expected_radial = 2.0 * G * M_kg * dr / r ** 3
            expected_tangential = G * M_kg * dr / r ** 3
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
            expected = 2.0 * phys.G * (10.0 * phys.M_sun) * 1.8 / r[i] ** 3
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

    def test_cycloid_benchmark_converges_and_plateaus(self):
        m, r0_rs, r_stop_rs = 10.0, 6.0, 1.0005
        tau_closed = infall_cycloid_tau(m, r0_rs, r_stop_rs)
        rel_errs = []
        for step_frac in (0.2, 0.1, 0.05, 0.02, 0.01):
            result = phys.infall_radial(m_msun=m, r0_rs=r0_rs, r_stop_rs=r_stop_rs,
                                         n_points=4000, step_frac=step_frac)
            tau_code = result["summary"]["tau_total_s"]
            rel_errs.append(abs(tau_code - tau_closed) / tau_closed)
        # All within the documented 1e-5 tolerance, at every step_frac
        # from the coarsest allowed (0.2) down to 0.01 ...
        for e in rel_errs:
            self.assertLess(e, 1.0e-5)
        # ... and, among the finer step_fracs (0.05, 0.02, 0.01), the
        # error has settled into a plateau rather than continuing to fall
        # by orders of magnitude -- consistent with the two-sided step
        # refinement already making even 0.2 highly accurate, so further
        # refinement mostly meets the same RK4-vs-closed-form floor. (The
        # coarsest step_frac=0.2 is not part of this particular
        # comparison: its error can land anywhere below the 1e-5 ceiling
        # already checked above, including -- as observed -- below the
        # plateau itself, since a single coarse step size can happen to
        # benefit from fortuitous truncation-error cancellation.)
        fine = rel_errs[2:]  # step_frac = 0.05, 0.02, 0.01
        self.assertLessEqual(max(fine), min(fine) * 5.0)

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
        result = phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, n_steps=300,
                                       bisect_iters=30)
        s = result["summary"]
        m0_geom = phys._mass_geom(s["m0_msun"])
        m1_geom = phys._mass_geom(s["m1_msun"])
        v1 = s["v1_rs0"] * s["rs0_m"]
        v2 = s["v2_rs0"] * s["rs0_m"]
        expected = phys.vaidya_mass_of_v(result["v"], m0_geom, m1_geom, v1, v2) * 2.0
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

    # --- bounds / error handling -----------------------------------
    def test_M1_less_than_M0_rejected(self):
        with self.assertRaises(ValueError):
            phys.vaidya_horizons(m0_msun=10.0, m1_msun=5.0)

    def test_nonpositive_window_parameters_rejected(self):
        for kwargs in (dict(v1_rs0=0.0), dict(duration_rs0=0.0),
                       dict(v_start_margin_rs0=0.0), dict(v_end_margin_rs0=0.0)):
            with self.assertRaises(ValueError):
                phys.vaidya_horizons(m0_msun=5.0, m1_msun=10.0, **kwargs)

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
            self.assertEqual(header, ["r_over_rs", "r_m", "z_m", "proper_radial_distance_m"])
            self.assertEqual(len(rows), 37)
            self.assertAlmostEqual(float(rows[0][0]), 1.0, delta=1e-6)

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

    def test_horizons_writes_main_and_family_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_cli(["--mode", "horizons", "--no_plot", "--csvdir", tmp,
                             "--n_steps", "200", "--bisect_iters", "15", "--n_family", "5"])
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            files = sorted(f for f in os.listdir(tmp) if f.endswith(".csv"))
            self.assertEqual(len(files), 2)
            family_file = [f for f in files if "family" in f][0]
            comments, header, rows = read_csv_rows(os.path.join(tmp, family_file))
            self.assertEqual(header, ["trajectory_id", "r_i_over_rs0", "escapes",
                                       "v_over_rs0", "r_over_rs0"])
            traj_ids = sorted(set(int(r[0]) for r in rows))
            self.assertEqual(traj_ids, list(range(5)))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
