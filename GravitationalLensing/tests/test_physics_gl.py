"""Regression tests for the GravitationalLensing program module.

Discovery supports both the repository layout
(``tests/test_physics_gl.py``) and a flattened upload layout.  The full
suite is run once from ``tests/``.

Development history (audit trail -- developers only; never surfaced to
students in the Help file or in main.py/driver_gl.py/physics_gl.py/
plot_gl.py docstrings or output):

  2026-09-07  Grok (principal developer).  Kickoff submission.  First
    program and first unittest suite.  Version 0.1.0.  BUILD_ID is
    computed from the four core modules after this file is written; the
    Help file's #version_build element is filled to match.  Coverage in
    this round is the thin-lens invariants a later audit will lean on:
    Einstein-radius units, the analytic two-image formula, det A = 0 on
    the Einstein ring, SIS kappa = 1/2 on that ring, image-count change
    across an SIS+shear diamond, CLI --version agreement, and Help-file
    version/build consistency.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CORE_MODULE_FILES = (
    "physics_gl.py",
    "driver_gl.py",
    "main.py",
    "plot_gl.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all four core program modules."""
    candidate = Path(start).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if all((directory / name).is_file() for name in CORE_MODULE_FILES):
            return directory
    required = ", ".join(CORE_MODULE_FILES)
    raise FileNotFoundError(
        f"could not find a directory containing all core modules: {required}"
    )


MODULE_DIR = find_module_dir(Path(__file__))
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import driver_gl as driver  # noqa: E402
import physics_gl as phys  # noqa: E402
import plot_gl as plotting  # noqa: E402


def recompute_build_id(directory):
    digest = hashlib.sha256()
    for name in phys.BUILD_ID_COVERS:
        with (directory / name).open("r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


def find_help_file():
    here = MODULE_DIR
    candidates = [
        here / "GravitationalLensing.html",
        here.parent / "GravitationalLensing.html",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def run_cli(args, cwd=MODULE_DIR, timeout=60):
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class TestModuleDiscovery(unittest.TestCase):
    def test_canonical_layout_finds_four_core_modules(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)
        for name in CORE_MODULE_FILES:
            self.assertTrue((MODULE_DIR / name).is_file())


class TestBuildIdentity(unittest.TestCase):
    def test_build_id_matches_independent_hash(self):
        self.assertEqual(recompute_build_id(MODULE_DIR), phys.BUILD_ID)
        self.assertNotEqual(phys.BUILD_ID, "unknown")
        self.assertEqual(len(phys.BUILD_ID), 12)

    def test_build_id_covers_exactly_the_four_core_files(self):
        self.assertEqual(tuple(phys.BUILD_ID_COVERS), CORE_MODULE_FILES)

    def test_cli_version_string(self):
        result = run_cli(["--version"])
        self.assertEqual(result.returncode, 0)
        text = (result.stdout + result.stderr).strip()
        self.assertIn(phys.MODEL_VERSION, text)
        self.assertIn(phys.BUILD_ID, text)


class TestEinsteinRadius(unittest.TestCase):
    def test_onee12_msun_at_default_geometry_is_about_two_arcsec(self):
        theta_e = phys.einstein_radius_point_mass(1.0e12 * phys.M_SUN)
        arcsec = phys.rad_to_arcsec(theta_e)
        self.assertGreater(arcsec, 1.5)
        self.assertLess(arcsec, 2.5)

    def test_point_mass_theta_e_scales_as_sqrt_mass(self):
        t1 = phys.einstein_radius_point_mass(1.0e12 * phys.M_SUN)
        t4 = phys.einstein_radius_point_mass(4.0e12 * phys.M_SUN)
        self.assertAlmostEqual(t4 / t1, 2.0, places=6)

    def test_rejects_source_in_front_of_lens(self):
        with self.assertRaises(ValueError):
            phys.einstein_radius_point_mass(phys.M_SUN, d_l=2.0, d_s=1.0)

    def test_sis_theta_e_scales_as_sigma_squared(self):
        t1 = phys.einstein_radius_sis(200.0e3)
        t2 = phys.einstein_radius_sis(400.0e3)
        self.assertAlmostEqual(t2 / t1, 4.0, places=6)


class TestPointMassLensEquation(unittest.TestCase):
    def test_analytic_images_recover_the_source(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        beta = 0.4 * theta_e
        th_p, th_m = phys.point_mass_image_radii(beta, theta_e)
        for th in (th_p, th_m):
            bx, by = phys.map_point_mass(th, 0.0, theta_e)
            self.assertAlmostEqual(float(bx), beta, places=10)
            self.assertAlmostEqual(float(by), 0.0, places=10)

    def test_on_axis_images_sit_on_the_einstein_ring(self):
        theta_e = 1.0e-5
        plus, minus = phys.point_mass_image_radii(0.0, theta_e)
        self.assertAlmostEqual(plus, theta_e, places=12)
        self.assertAlmostEqual(minus, -theta_e, places=12)

    def test_det_a_vanishes_on_the_einstein_ring(self):
        theta_e = 1.0e-5
        det = phys.jacobian_det_point_mass(theta_e, 0.0, theta_e)
        self.assertAlmostEqual(float(det), 0.0, places=10)

    def test_det_a_is_positive_outside_and_negative_inside(self):
        theta_e = 1.0e-5
        self.assertGreater(float(phys.jacobian_det_point_mass(2 * theta_e, 0.0, theta_e)), 0.0)
        self.assertLess(float(phys.jacobian_det_point_mass(0.5 * theta_e, 0.0, theta_e)), 0.0)

    def test_two_images_of_an_off_axis_source_have_opposite_parity(self):
        theta_e = 1.0e-5
        beta = 0.3 * theta_e
        plus, minus = phys.point_mass_image_radii(beta, theta_e)
        self.assertGreater(float(phys.jacobian_det_point_mass(plus, 0.0, theta_e)), 0.0)
        self.assertLess(float(phys.jacobian_det_point_mass(minus, 0.0, theta_e)), 0.0)


class TestSISShear(unittest.TestCase):
    def test_sis_kappa_is_one_half_on_the_einstein_ring(self):
        theta_e = 1.0e-5
        kap = phys.convergence_sis(theta_e, 0.0, theta_e)
        self.assertAlmostEqual(float(kap), 0.5, places=10)

    def test_shear_rejected_at_or_above_unity(self):
        with self.assertRaises(ValueError):
            phys.deflection_sis_shear(1.0, 0.0, 1.0, 1.0)

    def test_diamond_caustic_is_closed_and_has_four_tips(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        cx, cy = phys.critical_curve_sis_shear(theta_e, gamma)
        sx, sy = phys.caustic_from_critical(cx, cy, theta_e, gamma)
        self.assertGreater(cx.size, 20)
        self.assertAlmostEqual(cx[0], cx[-1], places=5)
        self.assertAlmostEqual(cy[0], cy[-1], places=5)
        # Four axial extrema of the diamond.
        self.assertGreater(float(sx.max()), 0.0)
        self.assertLess(float(sx.min()), 0.0)
        self.assertGreater(float(sy.max()), 0.0)
        self.assertLess(float(sy.min()), 0.0)

    def test_source_outside_diamond_has_two_images(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        cx, cy = phys.critical_curve_sis_shear(theta_e, gamma)
        sx, sy = phys.caustic_from_critical(cx, cy, theta_e, gamma)
        outside = 1.4 * float(sx.max())
        images = phys.images_sis_shear(outside, 0.0, theta_e, gamma)
        self.assertEqual(len(images), 2)

    def test_source_inside_diamond_has_four_images(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        images = phys.images_sis_shear(0.05 * theta_e, 0.03 * theta_e,
                                       theta_e, gamma)
        self.assertEqual(len(images), 4)


class TestRendering(unittest.TestCase):
    def test_centered_point_source_peaks_on_the_ring(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        grid_x, grid_y = phys.make_grid(n_pix=101, fov_arcsec=6.0)
        image = phys.render_point_mass_source(
            grid_x, grid_y, theta_e, 0.0, 0.0, theta_e / 8.0,
        )
        r = np_hypot = phys._radius(grid_x, grid_y)
        ring = (np.abs(r - theta_e) < 0.15 * theta_e)
        self.assertGreater(float(image[ring].max()), float(image[r > 1.6 * theta_e].max()))

    def test_q_must_be_positive_and_at_most_one(self):
        with self.assertRaises(ValueError):
            phys.elliptical_source(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            phys.elliptical_source(0.0, 0.0, 0.0, 0.0, 1.0, 1.1, 0.0)


class TestSideRays(unittest.TestCase):
    def test_on_axis_einstein_ray_hits_the_observer(self):
        theta_e = 1.0e-5
        bundle = phys.thin_lens_side_rays(
            np.array([-theta_e, theta_e]), 0.0, theta_e, model="point",
        )
        for ray in bundle["rays"]:
            self.assertAlmostEqual(ray["points"][-1, 1], 0.0, places=12)
            # Source-plane height of an on-axis Einstein ray is zero:
            # beta = theta - alpha = theta - theta_E^2/theta = 0.
            self.assertAlmostEqual(ray["points"][0, 1], 0.0, places=12)

    def test_forward_rays_share_one_source(self):
        theta_e = 1.0e-5
        for beta in (0.0, -0.4 * theta_e):
            bundle = phys.forward_point_mass_bundle(beta, theta_e)
            y0 = bundle["y_src"]
            for ray in bundle["rays"]:
                self.assertAlmostEqual(ray["points"][0, 1], y0)
                self.assertAlmostEqual(ray["points"][0, 0], 0.0)
                self.assertAlmostEqual(ray["points"][1, 0], bundle["d_ls"])

    def test_analytic_images_are_the_only_hits(self):
        theta_e = 1.0e-5
        beta = -0.4 * theta_e
        plus, minus = phys.point_mass_image_radii(beta, theta_e)
        for th in (plus, minus):
            ray = phys.forward_point_mass_ray(beta, th, theta_e)
            self.assertTrue(ray["hits"])
            self.assertAlmostEqual(ray["y_obs"], 0.0, places=12)
        miss = phys.forward_point_mass_ray(beta, 2.2 * theta_e, theta_e)
        self.assertFalse(miss["hits"])

    def test_critical_side_view_tracks_beta_y_not_beta_x(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        beta = -0.4 * theta_e
        bundle = phys.forward_point_mass_bundle(beta, theta_e)
        self.assertLess(bundle["y_src"], 0.0)
        self.assertEqual(len(bundle["hits"]), 2)

    def test_kink_is_always_toward_the_mass(self):
        theta_e = 1.0e-5
        beta = -0.4 * theta_e
        for th in (-2.2 * theta_e, -0.45 * theta_e, 0.45 * theta_e, 2.2 * theta_e):
            ray = phys.forward_point_mass_ray(beta, th, theta_e)
            # alpha_signed has the sign of theta: subtracting it from
            # m_in bends the ray toward y = 0.
            self.assertGreater(th * ray["alpha"], 0.0)
            if th > 0.0:
                self.assertLess(ray["m_out"], ray["m_in"])
            else:
                self.assertGreater(ray["m_out"], ray["m_in"])


class TestDriverAndPlots(unittest.TestCase):
    def test_every_mode_writes_a_png_and_provenance_under_agg(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in driver.MODES:
                _fig, saved = driver.run(mode, outdir=tmp, show=False, dpi=80)
                self.assertIsNotNone(saved)
                self.assertTrue(Path(saved).is_file())
                sidecar = Path(saved).with_suffix("").as_posix() + ".provenance.txt"
                # _finish writes stem + .provenance.txt next to .png
                stem = saved[:-4]
                sidecar = stem + ".provenance.txt"
                self.assertTrue(Path(sidecar).is_file(), sidecar)
                text = Path(sidecar).read_text(encoding="utf-8")
                self.assertIn(phys.MODEL_VERSION, text)
                self.assertIn(phys.BUILD_ID, text)
                self.assertIn(f"mode = {mode}", text)

    def test_cli_point_mode_saves_under_agg(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(["--mode", "point", "--outdir", tmp, "--n_pix", "51"])
            self.assertEqual(result.returncode, 0, result.stderr)
            pngs = list(Path(tmp).glob("*.png"))
            self.assertTrue(pngs)


class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.help_path = find_help_file()

    def test_help_file_is_present(self):
        self.assertIsNotNone(self.help_path, "GravitationalLensing.html missing")

    def test_version_build_element_matches_physics_module(self):
        raw = self.help_path.read_text(encoding="utf-8")
        self.assertIn('id="version_build"', raw)
        self.assertIn(f"Version {phys.MODEL_VERSION}", raw)
        self.assertIn(f"Build {phys.BUILD_ID}", raw)

    def test_help_names_every_mode(self):
        text = self.help_path.read_text(encoding="utf-8")
        for mode in driver.MODES:
            self.assertIn(mode, text)


if __name__ == "__main__":
    unittest.main()
