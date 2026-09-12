"""Regression tests for the GravitationalLensing program module.

Discovery supports both the repository layout
(``tests/test_physics_gl.py``) and a flattened upload layout.  The full
suite is run once from ``tests/``.

Development history (audit trail -- developers only; never surfaced to
students in the Help file or in main.py/driver_gl.py/physics_gl.py/
plot_gl.py docstrings or output):

  2026-09-07  Grok.  Kickoff.  Version 0.1.0.  Thin-lens invariants.
    Artifact: GravitationalLensing-Grok-Kickoff-20260908.txt

  2026-09-08  Codex+Copilot Audit1 / Grok Response.  Version 0.2.0.
    SIS+shear finder, analytic critical curve, |gamma|<1/3 scope later.
    Artifacts: *-Codex-Audit1-20260908.txt, *-Copilot-Audit1-20260908.txt,
    *-Grok-Response-to-Audit1-20260908.txt

  2026-09-08  Audit2 / Response.  Version 0.3.0.
    Teaching domain |gamma|<1/3; FOV extras; BUILD_ID temp-dir test.
    Artifacts: *-Audit2-20260908.txt, *-Response-to-Audit2-20260908.txt

  2026-09-09  Audit3 / Response.  Version 0.4.0.
    FOV-vs-resolution contract; n_pix grows; large offsets rejected.
    Artifacts: *-Codex-Audit3-20260909.txt, *-Response-to-Audit3-20260908.txt

  2026-09-09  Audit4 / Response.  Version 0.5.0.
    Image-plane visibility of the point-mass double; mapped half-light.
    Artifacts: *-Codex-Audit4-20260909.txt, *-Response-to-Audit4-20260908.txt

  2026-09-09  Audit5 / Response.  Version 0.6.0.
    SIS+shear off-axis FOV bound; honest 0.8-interval visibility floor.
    Artifacts: *-Codex-Audit5-20260909.txt, *-Response-to-Audit5-20260908.txt

  2026-09-08  Audit6 / Response.  Version 1.0.0 release candidate.
    Reversed r_eff remedy; kappa interval contract; developer trail.
    Artifacts: *-Codex-Audit6-20260909.txt, *-Copilot-Audit6-20260908.txt,
    *-Grok-Response-to-Audit6-20260908.txt
    Complete texts live with the other GFTGUX audit records on GitHub;
    these lines are locators only.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
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


class TestMinimumPythonVersionCompatibility(unittest.TestCase):
    """1.0.1 regression guard.

    1.0.0 shipped three provenance f-strings in plot_gl.py whose ``{...}``
    expression spanned a physical line break inside a single-quoted
    f-string.  That is a SyntaxError on every Python before 3.12 (PEP 701
    relaxed the grammar); it is legal only from 3.12 onward.  The Help
    file promises "Python 3.10 or newer," so every core module must at
    least *parse* on 3.10.  ``ast.parse`` under the running interpreter
    cannot catch this retroactively once run under 3.12+, so this test
    shells out to the documented minimum interpreter directly.  It skips
    (rather than fails) when that interpreter isn't installed on the
    machine running the suite -- CI for this project should make sure it
    is.
    """

    MIN_PYTHON = "python3.10"

    def test_core_modules_parse_on_the_documented_minimum_python(self):
        if shutil.which(self.MIN_PYTHON) is None:
            self.skipTest(f"{self.MIN_PYTHON} not installed on this machine")
        for name in CORE_MODULE_FILES:
            path = MODULE_DIR / name
            result = subprocess.run(
                [self.MIN_PYTHON, "-c",
                 "import ast, sys; ast.parse(open(sys.argv[1]).read())",
                 str(path)],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(
                result.returncode, 0,
                msg=f"{name} does not parse on {self.MIN_PYTHON}:\n{result.stderr}",
            )


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

    def test_source_between_diamond_and_pseudo_caustic_has_two_images(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        cx, cy = phys.critical_curve_sis_shear(theta_e, gamma)
        sx, sy = phys.caustic_from_critical(cx, cy, theta_e, gamma)
        diamond = float(np.max(np.hypot(sx, sy)))
        mid = 0.5 * (diamond + theta_e)
        self.assertGreater(mid, diamond)
        self.assertLess(mid, theta_e)
        images = phys.images_sis_shear(mid, 0.0, theta_e, gamma)
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
        for th in (-theta_e, theta_e):
            ray = phys.forward_point_mass_ray(0.0, th, theta_e)
            self.assertTrue(ray["hits"])
            self.assertAlmostEqual(ray["y_src"], 0.0, places=12)
            self.assertAlmostEqual(ray["y_obs"], 0.0, places=12)

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
        match = re.search(
            r'id="version_build"[^>]*>(.*?)</p>', raw, flags=re.S
        )
        self.assertIsNotNone(match)
        text = re.sub(r"\s+", " ", match.group(1))
        self.assertIn(f"Version {phys.MODEL_VERSION}", text)
        self.assertIn(f"Build {phys.BUILD_ID}", text)
        stale = text.replace(phys.MODEL_VERSION, "0.0.0")
        self.assertNotIn(f"Version {phys.MODEL_VERSION}", stale)

    def test_help_names_every_mode(self):
        text = self.help_path.read_text(encoding="utf-8")
        for mode in driver.MODES:
            self.assertIn(mode, text)


class TestAudit1Fixes(unittest.TestCase):
    def test_analytic_critical_radius_and_closure(self):
        theta_e = phys.default_sis_theta_e(300.0)
        for gamma in (-0.30, -0.25, 0.0, 0.25, 0.30):
            n = 361
            cx, cy = phys.critical_curve_sis_shear(theta_e, gamma, n_theta=n)
            self.assertEqual(cx.size, n)
            self.assertAlmostEqual(float(cx[0]), float(cx[-1]), places=10)
            self.assertAlmostEqual(float(cy[0]), float(cy[-1]), places=10)
            phis = np.linspace(0.0, 2.0 * math.pi, n)
            for i, phi in enumerate(phis[::30]):
                r_exp = phys.critical_radius_sis_shear(float(phi), theta_e, gamma)
                r_got = math.hypot(float(cx[i * 30]), float(cy[i * 30]))
                self.assertAlmostEqual(r_got, r_exp, places=8)
                det = phys.jacobian_det_sis_shear(cx[i * 30], cy[i * 30],
                                                  theta_e, gamma)
                self.assertAlmostEqual(float(det), 0.0, places=6)

    def test_high_shear_is_rejected_at_the_naked_cusp_bound(self):
        theta_e = 1.0e-5
        with self.assertRaises(ValueError):
            phys.critical_curve_sis_shear(theta_e, phys.GAMMA_MAX)
        with self.assertRaises(ValueError):
            phys.images_sis_shear(0.0, 0.0, theta_e, -phys.GAMMA_MAX)
        with self.assertRaises(ValueError):
            phys.deflection_sis_shear(1.0, 0.0, 1.0, 0.8)

    def test_remote_source_has_one_image_that_solves_the_lens_equation(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        beta_x = phys.arcsec_to_rad(2.0)
        images = phys.images_sis_shear(beta_x, 0.0, theta_e, gamma)
        self.assertEqual(len(images), 1)
        bx, by = phys.map_sis_shear(images[0][0], images[0][1], theta_e, gamma)
        self.assertAlmostEqual(float(bx), float(beta_x), places=7)
        self.assertAlmostEqual(float(by), 0.0, places=7)

    def test_returned_images_map_back_to_the_source(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        for bx, by in ((0.05 * theta_e, 0.03 * theta_e),
                       (0.7 * theta_e, 0.0),
                       (1.5 * theta_e, 0.0)):
            images = phys.images_sis_shear(bx, by, theta_e, gamma)
            self.assertGreaterEqual(len(images), 1)
            for ix, iy in images:
                mx, my = phys.map_sis_shear(ix, iy, theta_e, gamma)
                self.assertAlmostEqual(float(mx), float(bx), places=6)
                self.assertAlmostEqual(float(my), float(by), places=6)

    def test_fold_crossing_changes_image_count_from_two_to_four(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        phi = 0.25 * math.pi
        r = phys.critical_radius_sis_shear(phi, theta_e, gamma)
        tx, ty = r * math.cos(phi), r * math.sin(phi)
        bx, by = phys.map_sis_shear(tx, ty, theta_e, gamma)
        # The fold point is off-axis.  Step along the source-plane radius.
        rad = math.hypot(float(bx), float(by))
        ux, uy = float(bx) / rad, float(by) / rad
        outside = phys.images_sis_shear(1.12 * rad * ux, 1.12 * rad * uy,
                                        theta_e, gamma)
        inside = phys.images_sis_shear(0.70 * rad * ux, 0.70 * rad * uy,
                                       theta_e, gamma)
        self.assertEqual(len(outside), 2)
        self.assertEqual(len(inside), 4)
        seps = []
        for i, (ix, iy) in enumerate(inside):
            for jx, jy in inside[i + 1:]:
                seps.append(math.hypot(ix - jx, iy - jy))
        self.assertGreater(min(seps), phys.IMAGE_DEDUP_FRAC * theta_e)

    def test_images_just_inside_a_cusp_stay_four(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        cusp = 2.0 * theta_e * abs(gamma) / (1.0 + abs(gamma))
        frac = 1.0 - 10.0 * phys.CAUSTIC_COUNT_BUFFER
        images = phys.images_sis_shear(frac * cusp, 0.0, theta_e, gamma)
        self.assertEqual(len(images), 4)

    def test_centered_zero_shear_is_an_einstein_ring_not_dots(self):
        theta_e = phys.default_sis_theta_e(300.0)
        self.assertTrue(phys.is_einstein_ring_case(0.0, 0.0, 0.0, theta_e))
        self.assertEqual(phys.images_sis_shear(0.0, 0.0, theta_e, 0.0), [])

    def test_signed_rays_mirror(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        beta = phys.arcsec_to_rad(0.35)
        up = phys.forward_point_mass_bundle(beta, theta_e)
        down = phys.forward_point_mass_bundle(-beta, theta_e)
        self.assertAlmostEqual(up["y_src"], -down["y_src"])
        self.assertEqual(len(up["hits"]), 2)
        self.assertEqual(len(down["hits"]), 2)
        up_th = sorted(r["theta"] for r in up["hits"])
        down_th = sorted(r["theta"] for r in down["hits"])
        self.assertAlmostEqual(up_th[0], -down_th[1], places=10)
        self.assertAlmostEqual(up_th[1], -down_th[0], places=10)

    def test_r_eff_is_half_light_radius(self):
        r_eff = 1.0
        sigma = phys.sigma_from_r_eff(r_eff)
        # Enclosed fraction of a circular 2-D Gaussian inside R_e is 1/2.
        enclosed = 1.0 - math.exp(-(r_eff ** 2) / (2.0 * sigma ** 2))
        self.assertAlmostEqual(enclosed, 0.5, places=12)

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            phys.validate_user_value("beta_x", float("nan"))
        with self.assertRaises(ValueError):
            phys.validate_user_value("beta_y", float("inf"))
        with self.assertRaises(ValueError):
            phys.validate_user_value("gamma", 1.0, exclusive_max=1.0)

    def test_build_id_changes_when_a_core_file_changes(self):
        original = phys.compute_build_id_from_directory(MODULE_DIR)
        self.assertEqual(original, phys.BUILD_ID)
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            for name in phys.BUILD_ID_COVERS:
                (dest / name).write_text(
                    (MODULE_DIR / name).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            before = phys.compute_build_id_from_directory(dest)
            self.assertEqual(before, original)
            target = dest / "physics_gl.py"
            target.write_text(target.read_text(encoding="utf-8")
                              + "\n# audit-sensitivity\n",
                              encoding="utf-8")
            mutated = phys.compute_build_id_from_directory(dest)
            self.assertNotEqual(mutated, original)

    def test_cli_rejects_nonfinite_beta(self):
        result = run_cli(["--mode", "point", "--beta_x", "inf"])
        self.assertNotEqual(result.returncode, 0)

    def test_adapted_fov_uses_twenty_percent_margin(self):
        te = phys.arcsec_to_rad(2.0)
        fov = phys.adapted_fov_arcsec(te, 1.0, extras=())
        self.assertAlmostEqual(fov, 2.0 * phys.FOV_MARGIN * 2.0, places=6)

    def test_default_point_image_recovers_a_bright_peak(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        beta_x = phys.arcsec_to_rad(0.50)
        sigma = phys.arcsec_to_rad(phys.COMPACT_SOURCE_SIGMA_ARCSEC)
        plus, minus = phys.point_mass_image_radii(abs(beta_x), theta_e)
        fov = phys.adapted_fov_arcsec(
            theta_e, 6.0, extras=(beta_x, plus, minus, sigma),
        )
        n_pix = phys.adapted_n_pix(181, fov, phys.COMPACT_SOURCE_SIGMA_ARCSEC)
        gx, gy = phys.make_grid(n_pix=n_pix, fov_arcsec=fov)
        image = phys.render_point_mass_source(gx, gy, theta_e, beta_x, 0.0, sigma)
        self.assertGreater(float(image.max()), 0.5)
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_point(outdir=tmp, show=False, dpi=60)
            text = Path(saved[:-4] + ".provenance.txt").read_text(encoding="utf-8")
            self.assertIn("n_pix =", text)
            self.assertIn("fov_arcsec =", text)

    def test_large_offset_is_rejected_as_outside_teaching_domain(self):
        with self.assertRaises(ValueError) as ctx:
            driver.run_point(beta_x_arcsec=20.0, show=False)
        msg = str(ctx.exception)
        self.assertNotIn("pass --n_pix", msg)
        self.assertNotIn("--r_eff", msg)
        self.assertTrue(
            "teaching domain" in msg or "Move the source closer" in msg
        )

    def test_default_point_resolves_both_images(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        beta = phys.arcsec_to_rad(0.50)
        sigma = phys.arcsec_to_rad(phys.COMPACT_SOURCE_SIGMA_ARCSEC)
        extras = driver._point_mass_grid_args(
            theta_e, beta, 0.0, 3.0 * sigma,
        )
        fov = phys.adapted_fov_arcsec(theta_e, 6.0, extras=extras)
        n_pix = phys.adapted_n_pix(181, fov, phys.COMPACT_SOURCE_SIGMA_ARCSEC)
        phys.require_resolved_point_mass_images(
            theta_e, beta, sigma, fov, n_pix,
        )
        gx, gy = phys.make_grid(n_pix=n_pix, fov_arcsec=fov)
        image = phys.render_point_mass_source(gx, gy, theta_e, beta, 0.0, sigma)
        plus, minus = phys.point_mass_image_radii(beta, theta_e)
        spacing = fov / (n_pix - 1)
        inner = float(phys.rad_to_arcsec(
            phys.point_mass_image_plane_scale(minus, theta_e, sigma)
        ))
        self.assertGreaterEqual(inner / spacing, phys.IMAGE_DETECT_SAMPLES - 1e-9)

        def peak_near(th):
            iy = int(np.argmin(np.abs(gy[:, 0] - 0.0)))
            ix = int(np.argmin(np.abs(gx[0, :] - th)))
            sl = image[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]
            return float(sl.max())
        self.assertGreater(peak_near(plus), 0.4)
        self.assertGreater(peak_near(minus), 0.4)

    def test_mapped_half_light_sets_blob_field(self):
        theta_e = phys.default_point_mass_theta_e(12.0)
        r_eff = phys.arcsec_to_rad(2.0)
        far_p, _far_m = phys.point_mass_mapped_radii(0.0, theta_e, r_eff)
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_blob(
                beta_x_arcsec=0.0, beta_y_arcsec=0.0,
                r_eff_arcsec=2.0, q=1.0,
                outdir=tmp, show=False, dpi=40,
            )
            text = Path(saved[:-4] + ".provenance.txt").read_text(encoding="utf-8")
            fov_run = float(next(ln.split("=")[1]
                                 for ln in text.splitlines()
                                 if "fov_arcsec" in ln))
        self.assertGreater(0.5 * fov_run, float(phys.rad_to_arcsec(far_p)))

    def test_arcs_field_contains_off_axis_image(self):
        theta_e = phys.default_sis_theta_e(300.0)
        gamma = 0.25
        beta_x = phys.arcsec_to_rad(2.0)
        r_eff = phys.arcsec_to_rad(0.25)
        r_img = phys.sis_shear_outer_image_radius(
            beta_x, 0.0, r_eff, theta_e, gamma,
        )
        fov = phys.adapted_fov_arcsec(theta_e, 6.0, extras=(r_img,))
        self.assertGreater(0.5 * fov, float(phys.rad_to_arcsec(r_img)))
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_arcs(
                beta_x_arcsec=2.0, beta_y_arcsec=0.0,
                r_eff_arcsec=0.25, q=0.60, phi_deg=0.0,
                outdir=tmp, show=False, dpi=60,
            )
            text = Path(saved[:-4] + ".provenance.txt").read_text(encoding="utf-8")
            self.assertIn("theta_e_arcsec =", text)
            self.assertIn("sigma_v_kms =", text)
            fov_run = float(next(ln.split("=")[1]
                                for ln in text.splitlines()
                                if "fov_arcsec" in ln))
            self.assertGreater(0.5 * fov_run, float(phys.rad_to_arcsec(r_img)))
        n_pix = phys.adapted_n_pix(181, fov, 0.25 * 0.60)
        gx, gy = phys.make_grid(n_pix=n_pix, fov_arcsec=fov)
        image = phys.render_sis_shear_extended(
            gx, gy, theta_e, gamma, beta_x, 0.0, r_eff, 0.60, 0.0,
        )
        self.assertGreater(float(image.max()), 0.5)

    def test_small_sigma_v_is_rejected_by_kappa(self):
        with self.assertRaises(ValueError):
            driver.run_kappa(sigma_v_kms=10.0, show=False)

    def test_galaxy_scale_sigma_v_is_accepted_by_kappa(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_kappa(
                sigma_v_kms=180.0, outdir=tmp, show=False, dpi=40,
            )
            self.assertTrue(saved.endswith(".png"))

    def test_larger_r_eff_can_rescue_a_rejected_blob(self):
        with self.assertRaises(ValueError) as ctx:
            driver.run_blob(beta_x_arcsec=2.0, r_eff_arcsec=0.05,
                            q=0.60, show=False)
        self.assertIn("larger --r_eff", str(ctx.exception))
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_blob(
                beta_x_arcsec=2.0, r_eff_arcsec=0.25, q=0.60,
                outdir=tmp, show=False, dpi=40,
            )
            self.assertTrue(saved.endswith(".png"))

    def test_critical_provenance_records_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_critical(outdir=tmp, show=False, dpi=60)
            text = Path(saved[:-4] + ".provenance.txt").read_text(encoding="utf-8")
            self.assertIn("n_pix =", text)
            self.assertIn("fov_arcsec =", text)
            self.assertIn("sigma_src_arcsec =", text)

    def test_help_examples_fit_in_the_adapted_field(self):
        theta_e = phys.default_point_mass_theta_e(12.5)
        beta = phys.arcsec_to_rad(0.3 * math.sqrt(2.0))
        plus, minus = phys.point_mass_image_radii(beta, theta_e)
        fov = phys.adapted_fov_arcsec(
            theta_e, 6.0, extras=(plus, minus, phys.arcsec_to_rad(0.3)),
        )
        half = 0.5 * fov
        self.assertGreater(half, float(phys.rad_to_arcsec(theta_e)))
        self.assertGreater(half, abs(float(phys.rad_to_arcsec(plus))))
        self.assertGreater(half, abs(float(phys.rad_to_arcsec(minus))))
        n_pix = phys.adapted_n_pix(181, fov, phys.COMPACT_SOURCE_SIGMA_ARCSEC)
        spacing = fov / (n_pix - 1)
        self.assertLessEqual(
            spacing,
            phys.COMPACT_SOURCE_SIGMA_ARCSEC / phys.PIXELS_PER_SIGMA + 1e-9,
        )

    def test_cli_rejects_naked_cusp_shear(self):
        result = run_cli(["--mode", "shear", "--gamma", "0.8"])
        self.assertNotEqual(result.returncode, 0)

    def test_shear_provenance_records_gamma(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fig, saved = driver.run_shear(outdir=tmp, show=False, dpi=60,
                                           gamma=0.25)
            text = Path(saved[:-4] + ".provenance.txt").read_text(encoding="utf-8")
            self.assertIn("gamma =", text)
            self.assertIn("beta_x_arcsec =", text)


if __name__ == "__main__":
    unittest.main()
