"""Regression tests for the BlackHoleShadow program module.

Discovery supports both the repository layout
(``tests/test_physics_bhs.py``) and a flattened upload layout.  The full
suite is run once from ``tests/``.

Development history (audit trail -- developers only; never surfaced to
students in the Help file or in main.py/driver_bhs.py/physics_bhs.py/
plot_bhs.py docstrings or output):

  2026-09-11  Grok.  Kickoff.  Version 0.1.0.
    Artifact: BlackHoleShadow-Grok-Kickoff-2026091118.txt

  2026-09-11  Grok.  Response to Audit1.  Version 0.2.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit1-2026091122.txt

  2026-09-11  Grok.  Response to Audit2.  Version 0.3.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit2-2026091123.txt
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CORE_MODULE_FILES = (
    "physics_bhs.py",
    "driver_bhs.py",
    "main.py",
    "plot_bhs.py",
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

import driver_bhs as driver  # noqa: E402
import physics_bhs as phys  # noqa: E402
import plot_bhs as plotting  # noqa: E402


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
        here / "BlackHoleShadow.html",
        here.parent / "BlackHoleShadow.html",
        here.parent / "BlackHoleShadow-Documentation" / "BlackHoleShadow.html",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def run_cli(args, cwd=MODULE_DIR, timeout=90):
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

    def test_build_id_from_copied_tree_does_not_touch_live_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in CORE_MODULE_FILES:
                (tmp_path / name).write_text(
                    (MODULE_DIR / name).read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            copied = phys.compute_build_id_from_directory(tmp_path)
            self.assertEqual(copied, phys.BUILD_ID)


class TestScales(unittest.TestCase):
    def test_critical_b_at_unit_mass(self):
        self.assertAlmostEqual(
            phys.critical_impact_parameter(1.0), 3.0 * math.sqrt(3.0), places=12
        )
        self.assertAlmostEqual(phys.event_horizon(1.0), 2.0, places=12)
        self.assertAlmostEqual(phys.photon_sphere(1.0), 3.0, places=12)

    def test_scales_linear_in_M(self):
        self.assertAlmostEqual(
            phys.critical_impact_parameter(2.5),
            2.5 * phys.critical_impact_parameter(1.0),
            places=12,
        )

    def test_rejects_nonpositive_M(self):
        with self.assertRaises(ValueError):
            phys.critical_impact_parameter(0.0)
        with self.assertRaises(ValueError):
            phys.critical_impact_parameter(-1.0)


class TestCaptureClassification(unittest.TestCase):
    def test_photonorbit_demo_rays_at_unit_mass(self):
        self.assertTrue(phys.is_captured(5.0, 1.0))
        self.assertFalse(phys.is_captured(6.0, 1.0))

    def test_threshold_is_b_crit_inclusive(self):
        b_crit = phys.critical_impact_parameter(1.0)
        self.assertTrue(phys.is_captured(b_crit, 1.0))
        self.assertFalse(phys.is_captured(b_crit + 1.0e-8, 1.0))

    def test_negative_b_rejected(self):
        with self.assertRaises(ValueError):
            phys.is_captured(-0.1, 1.0)


class TestPeriapsis(unittest.TestCase):
    def test_captured_has_no_periapsis(self):
        self.assertIsNone(phys.periapsis(5.0, 1.0))

    def test_large_b_periapsis_approaches_b(self):
        rmin = phys.periapsis(50.0, 1.0)
        self.assertGreater(rmin, phys.photon_sphere(1.0))
        # Weak-field periapsis is slightly inside b.
        self.assertLess(rmin, 50.0)
        self.assertGreater(rmin, 45.0)

    def test_near_critical_periapsis_approaches_photon_sphere(self):
        b_crit = phys.critical_impact_parameter(1.0)
        rmin = phys.periapsis(b_crit * 1.001, 1.0)
        self.assertAlmostEqual(rmin, 3.0, delta=0.10)


class TestDeflection(unittest.TestCase):
    def test_weak_field_formula(self):
        self.assertAlmostEqual(phys.weak_field_deflection(100.0, 1.0), 0.04, places=12)

    def test_large_b_matches_weak_field(self):
        for b in (50.0, 200.0, 500.0, 1000.0):
            exact = phys.asymptotic_deflection(b, 1.0)
            weak = phys.weak_field_deflection(b, 1.0)
            series = weak + (15.0 * math.pi / 4.0) / (b * b)
            self.assertLess(abs(exact - series) / series, 0.02, msg=f"b={b}")

    def test_deflection_diverges_toward_b_crit(self):
        b_crit = phys.critical_impact_parameter(1.0)
        near = phys.asymptotic_deflection(b_crit * 1.001, 1.0)
        mid = phys.asymptotic_deflection(b_crit * 1.02, 1.0)
        self.assertGreater(near, mid)
        self.assertGreater(near, math.pi)

    def test_deflection_grows_toward_the_photon_sphere(self):
        b_crit = phys.critical_impact_parameter(1.0)
        far = phys.asymptotic_deflection(8.0, 1.0)
        near = phys.asymptotic_deflection(b_crit * 1.02, 1.0)
        self.assertGreater(near, far)
        self.assertGreater(near, math.pi / 2.0)

    def test_captured_ray_has_no_scattering_deflection(self):
        with self.assertRaises(ValueError):
            phys.asymptotic_deflection(5.0, 1.0)


class TestIntegratorAgreesWithCaptureRule(unittest.TestCase):
    def test_b5_is_captured_and_b6_escapes(self):
        _, _, info5 = phys.integrate_photon_orbit(1.0, 40.0, 5.0, 200.0, 0.05)
        _, _, info6 = phys.integrate_photon_orbit(1.0, 40.0, 6.0, 200.0, 0.05)
        self.assertEqual(info5["status"], "captured")
        self.assertEqual(info6["status"], "escaped")
        self.assertGreater(info6["closest_approach"], phys.photon_sphere(1.0))
        self.assertLessEqual(info5["closest_approach"], phys.event_horizon(1.0) + 1.0e-9)

    def test_delta_phi_is_accumulated_azimuth_not_hat_alpha(self):
        _, _, info = phys.integrate_photon_orbit(1.0, 40.0, 6.0, 200.0, 0.05)
        # Finite-camera Delta phi is not the asymptotic deflection.
        hat = phys.asymptotic_deflection(6.0, 1.0)
        self.assertNotAlmostEqual(info["delta_phi"], hat, places=2)


class TestCameraGrid(unittest.TestCase):
    def test_odd_n_pix_and_axis_pixel(self):
        bx, by, n_pix, fov = phys.make_impact_grid(20, 16.0)
        self.assertEqual(n_pix, 21)
        mid = n_pix // 2
        self.assertAlmostEqual(float(bx[mid, mid]), 0.0, places=12)
        self.assertAlmostEqual(float(by[mid, mid]), 0.0, places=12)

    def test_capture_map_disk_radius_is_b_crit(self):
        bx, by, _, _ = phys.make_impact_grid(81, 16.0)
        captured = phys.capture_map(bx, by, 1.0)
        b = np.hypot(bx, by)
        b_crit = phys.critical_impact_parameter(1.0)
        self.assertTrue(np.all(captured[b <= b_crit]))
        self.assertFalse(np.any(captured[b > b_crit]))

    def test_shadow_resolution_contract(self):
        phys.require_resolved_shadow(1.0, 16.0, 161)
        with self.assertRaises(ValueError):
            phys.require_resolved_shadow(1.0, 200.0, 11)

    def test_auto_n_pix_cap(self):
        with self.assertRaises(ValueError):
            phys.odd_n_pix(1000)


class TestPhotonRingMask(unittest.TestCase):
    def test_ring_sits_outside_the_shadow(self):
        bx, by, _, _ = phys.make_impact_grid(81, 16.0, 1.0)
        captured = phys.capture_map(bx, by, 1.0)
        ring = phys.photon_ring_mask(bx, by, 1.0)
        self.assertTrue(np.any(ring))
        self.assertFalse(np.any(ring & captured))
        b = np.hypot(bx, by)
        b_crit = phys.critical_impact_parameter(1.0)
        self.assertTrue(np.all(b[ring] > b_crit))

    def test_near_critical_escapers_are_in_the_overlay(self):
        M = 1.0
        b_crit = phys.critical_impact_parameter(M)
        bx = np.array([[b_crit * (1.0 + 1.0e-6)]])
        by = np.array([[0.0]])
        mask = phys.photon_ring_mask(bx, by, M)
        self.assertTrue(bool(mask[0, 0]))
        captured = np.array([[b_crit]])
        self.assertFalse(bool(phys.photon_ring_mask(captured, by, M)[0, 0]))


class TestDiskImage(unittest.TestCase):
    def test_zero_emissivity_gives_zero_image(self):
        real = phys.emitted_intensity
        phys.emitted_intensity = lambda *a, **k: 0.0
        try:
            self.assertEqual(phys.observed_intensity(6.93, 1.0), 0.0)
            bx, by, _, _ = phys.make_impact_grid(21, 16.0, 1.0)
            image, _ = phys.disk_image(bx, by, 1.0, r_hot=6.0)
            self.assertEqual(float(np.max(image)), 0.0)
        finally:
            phys.emitted_intensity = real

    def test_hot_ring_must_lie_outside_photon_sphere(self):
        bx, by, _, _ = phys.make_impact_grid(21, 16.0, 1.0)
        with self.assertRaises(ValueError):
            phys.disk_image(bx, by, 1.0, r_hot=2.5)

    def test_components_sum_to_total(self):
        parts, _ = phys.observed_components(7.0, 1.0)
        self.assertAlmostEqual(sum(parts), phys.observed_intensity(7.0, 1.0), places=12)

    def test_crossings_scale_with_M(self):
        r1 = phys.face_on_crossing_radii(6.0, 1.0)
        r2 = phys.face_on_crossing_radii(12.0, 2.0)
        for a, b in zip(r1, r2):
            if a is None:
                self.assertIsNone(b)
            else:
                self.assertAlmostEqual(a, b/2.0, places=8)


    def test_critical_ray_does_not_plunge(self):
        b = phys.critical_impact_parameter(1.0)
        rs = phys.face_on_crossing_radii(b, 1.0, max_m=8)
        self.assertTrue(all(r >= 3.0 for r in rs))

    def test_source_roots_for_default_annulus(self):
        peaks = phys.source_crossing_impacts(1.0, 6.0, max_m=4)
        self.assertAlmostEqual(peaks[0]/1.0, 6.93215, places=2)
        self.assertAlmostEqual(peaks[1]/1.0, 5.4789, places=2)
        self.assertAlmostEqual(peaks[2]/1.0, 5.2080, places=2)

    def test_transfer_resolves_r3_independent_of_fov(self):
        for fov in (8.0, 16.0, 24.0):
            bs, table, _, _ = phys.transfer_curves(1.0, b_max_over_M=0.5*fov, max_m=3)
            n3 = int(np.isfinite(table[:, 2]).sum())
            self.assertGreater(n3, 10, msg=f"fov={fov} n3={n3}")

    def test_photon_increment_stable_across_fov(self):
        incs = []
        for fov in (16.0, 20.0):
            bx, by, _, _ = phys.make_impact_grid(41, fov, 1.0)
            _, img2, img3, _, _, _ = phys.disk_image_components(bx, by, 1.0)
            incs.append(float((img3-img2).max()))
        self.assertGreater(incs[0], 0.05)
        self.assertAlmostEqual(incs[0], incs[1], delta=0.05)

    def test_captured_ray_may_still_cross(self):
        rs = phys.face_on_crossing_radii(5.0, 1.0)
        self.assertIsNotNone(rs[0])
        self.assertGreater(rs[0], 2.0)


class TestCompareRings(unittest.TestCase):
    def test_default_galaxy_einstein_radius_is_a_few_arcsec(self):
        numbers = phys.compare_rings(12.0)
        self.assertGreater(numbers["theta_e_arcsec"], 1.0)
        self.assertLess(numbers["theta_e_arcsec"], 4.0)

    def test_einstein_radius_in_units_of_M_is_not_the_photon_sphere(self):
        numbers = phys.compare_rings(12.0)
        self.assertGreater(numbers["r_e_over_M"], 1.0e5)
        self.assertAlmostEqual(numbers["b_crit_over_M"], 3.0 * math.sqrt(3.0), places=12)
        self.assertNotAlmostEqual(numbers["r_e_over_M"], numbers["b_crit_over_M"])


class TestModesAndCLI(unittest.TestCase):
    def test_known_modes(self):
        self.assertEqual(
            driver.MODES,
            ("rays", "pixels", "capture", "ring", "transfer", "image",
             "radii", "weak", "compare"),
        )

    def test_cli_rejects_unknown_mode(self):
        result = run_cli(["--mode", "kerr"])
        self.assertNotEqual(result.returncode, 0)

    def test_cli_rejects_nonpositive_M(self):
        result = run_cli(["--mode", "capture", "--M", "0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BlackHoleShadow:", result.stderr)

    def test_cli_capture_headless(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                ["--mode", "capture", "--n_pix", "41", "--fov", "16",
                 "--outdir", tmp],
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            pngs = list(Path(tmp).glob("capture_*.png"))
            sides = list(Path(tmp).glob("capture_*.provenance.txt"))
            self.assertEqual(len(pngs), 1)
            self.assertEqual(len(sides), 1)
            text = sides[0].read_text(encoding="utf-8")
            self.assertIn(phys.MODEL_VERSION, text)
            self.assertIn(phys.BUILD_ID, text)

    def test_cli_weak_and_compare_and_radii(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("weak", "compare", "radii"):
                extra = ["--n_pix", "41", "--fov", "16"] if mode == "radii" else []
                result = run_cli(["--mode", mode, "--outdir", tmp, *extra], timeout=60)
                self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)

    def test_driver_rays_classifies_demo_b(self):
        fig, saved = driver.run_rays(show=False, outdir=None)
        self.assertIsNotNone(fig)
        self.assertIsNone(saved)


class TestHelpFile(unittest.TestCase):
    def test_help_version_line_matches_when_present(self):
        path = find_help_file()
        if path is None:
            self.skipTest("Help file not shipped next to this build")
        text = path.read_text(encoding="utf-8")
        self.assertIn('id="version_build"', text)
        self.assertIn(phys.MODEL_VERSION, text)
        self.assertIn(phys.BUILD_ID, text)

    def test_cli_rejects_camera_inside_photon_sphere(self):
        result = run_cli(["--mode", "rays", "--r_cam", "2.5"])
        self.assertNotEqual(result.returncode, 0)

    def test_cli_rejects_inclination_outside_range(self):
        result = run_cli(["--mode", "image", "--inclination", "30"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("face-on", result.stderr.lower() + result.stdout.lower())

    def test_pixels_narration_follows_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                ["--mode", "pixels", "--b_in", "6", "--b_out", "5",
                 "--outdir", tmp],
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("escaped", result.stdout)
            self.assertIn("captured", result.stdout)


class TestScaleInvariance(unittest.TestCase):
    def test_capture_fraction_independent_of_M_at_fixed_fov_over_M(self):
        fractions = []
        for M in (0.5, 1.0, 2.0, 5.0):
            bx, by, _, _ = phys.make_impact_grid(41, 16.0, M)
            captured = phys.capture_map(bx, by, M)
            fractions.append(captured.mean())
        for frac in fractions[1:]:
            self.assertAlmostEqual(frac, fractions[0], places=10)

    def test_resolved_shadow_is_M_invariant(self):
        a = phys.require_resolved_shadow(1.0, 16.0, 81)
        b = phys.require_resolved_shadow(7.0, 16.0, 81)
        self.assertAlmostEqual(a, b, places=10)


    def test_undersized_fov_is_expanded_not_blank(self):
        fov_eff, need = phys.contained_fov(6.0, 1.0)
        self.assertGreater(fov_eff, 6.0)
        self.assertGreaterEqual(0.5 * fov_eff, phys.critical_impact_parameter(1.0))

    def test_cli_capture_at_M_equals_two_is_not_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_cli(
                ["--mode", "capture", "--M", "2", "--n_pix", "41",
                 "--fov", "16", "--outdir", tmp],
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("captured pixels", result.stdout)
            # Must not be 100% dark on a 16M field.
            self.assertNotIn("1681 / 1681", result.stdout)

    def test_r_cam_scales_with_M(self):
        statuses=[]
        phis=[]
        rmins=[]
        for M in (0.5, 1.0, 2.0, 5.0):
            _, _, info = phys.integrate_photon_orbit(
                M,
                phys.to_absolute_length(40.0, M, "r_cam"),
                phys.to_absolute_length(6.0, M, "b"),
                phys.to_absolute_length(200.0, M, "lambda_max"),
                phys.to_absolute_length(0.02, M, "d_lambda"),
            )
            statuses.append(info["status"])
            phis.append(info["delta_phi"])
            rmins.append(info["closest_approach"]/M)
        self.assertEqual(set(statuses), {"escaped"})
        for phi in phis[1:]:
            self.assertAlmostEqual(phi, phis[0], places=5)
        for r in rmins[1:]:
            self.assertAlmostEqual(r, rmins[0], places=5)


if __name__ == "__main__":
    unittest.main()

