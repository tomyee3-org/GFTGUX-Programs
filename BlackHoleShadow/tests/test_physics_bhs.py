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

  2026-09-12  Grok.  Response to Audit3.  Version 0.4.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit3-2026091203.txt

  2026-09-12  Grok.  Response to Audit4.  Version 0.5.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit4-2026091204.txt

  2026-09-12  Grok.  Response to Audit5.  Version 0.6.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit5-2026091207.txt

  2026-09-12  Grok.  Response to Audit6.  Version 0.7.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit6-2026091213.txt

  2026-09-12  Grok.  Response to Audit7.  Version 0.8.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit7-2026091218.txt

  2026-09-12  Grok.  Response to Audit8.  Version 0.9.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit8-2026091221.txt

  2026-09-13  Grok.  Response to Audit9.  Version 0.10.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit9-2026091303.txt

  2026-09-13  Grok.  Response to Audit10.  Version 0.11.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit10-2026091304.txt

  2026-09-13  Grok.  Response to Audit11.  Version 0.12.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit11-2026091305.txt

  2026-09-13  Grok.  Response to Audit12.  Version 0.13.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit12-2026091306.txt

  2026-09-13  Grok.  Response to Audit13.  Version 0.14.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit13-2026091306.txt

  2026-09-13  Grok.  Response to Audit14.  Version 0.15.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit14-2026091316.txt

  2026-09-13  Grok.  Response to Audit15.  Version 0.16.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit15-2026091317.txt

  2026-09-13  Grok.  Response to Audit16.  Version 0.17.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit16-2026091318.txt

  2026-09-13  Grok.  Response to Audit17.  Version 0.18.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit17-2026091320.txt

  2026-09-13  Grok.  Response to Audit18.  Version 0.19.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit18-2026091323.txt

  2026-09-14  Grok.  Response to Audit19.  Version 0.20.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit19-2026091401.txt

  2026-09-14  Grok.  Response to Audit20.  Version 0.21.0.
    Artifact: BlackHoleShadow-Grok-Response-to-Audit20-2026091402.txt

  2026-09-14  Grok.  Version 1.0.0 first numbered release.
    ReleaseNotes and SampleOutputs_Guide added after Audit 20.
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
    "utilities_bhs.py",
    "driver_bhs.py",
    "main.py",
    "plot_bhs.py",
)


def find_module_dir(start):
    """Find the nearest ancestor containing all core program modules."""
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
    def test_canonical_layout_finds_core_modules(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)
        for name in CORE_MODULE_FILES:
            self.assertTrue((MODULE_DIR / name).is_file())


class TestBuildIdentity(unittest.TestCase):
    def test_build_id_matches_independent_hash(self):
        self.assertEqual(recompute_build_id(MODULE_DIR), phys.BUILD_ID)
        self.assertNotEqual(phys.BUILD_ID, "unknown")
        self.assertEqual(len(phys.BUILD_ID), 12)

    def test_build_id_covers_exactly_the_core_files(self):
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
            (tmp_path / CORE_MODULE_FILES[0]).unlink()
            with self.assertRaises((OSError, FileNotFoundError)):
                phys.compute_build_id_from_directory(tmp_path)

    def test_mutating_help_does_not_change_build_id(self):
        help_path = find_help_file()
        if help_path is None:
            self.skipTest("Help file not shipped next to this build")
        before = phys.BUILD_ID
        original = help_path.read_text(encoding="utf-8")
        try:
            help_path.write_text(original + "\n<!-- probe -->\n", encoding="utf-8")
            self.assertEqual(recompute_build_id(MODULE_DIR), before)
        finally:
            help_path.write_text(original, encoding="utf-8")


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

    def test_deflection_rejects_unresolved_near_critical_offset(self):
        b_crit = phys.critical_impact_parameter(1.0)
        with self.assertRaises(ValueError):
            phys.asymptotic_deflection(math.nextafter(b_crit, math.inf), 1.0)


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
        rs = phys.face_on_crossing_radii(b, 1.0, max_m=4)
        self.assertTrue(all(r > 3.0 for r in rs))
        self.assertAlmostEqual(rs[0], 4.28492065, places=6)
        self.assertAlmostEqual(rs[1], 3.04374815, places=6)
        self.assertAlmostEqual(rs[2], 3.00187312, places=6)
        self.assertAlmostEqual(rs[3], 3.00008091, places=6)
        self.assertGreater(rs[0], rs[1])
        self.assertGreater(rs[1], rs[2])
        self.assertGreater(rs[2], rs[3])
        # Nearby captured/escaping rays keep their own geometry.
        below = phys.face_on_crossing_radii(b - 1.0e-6, 1.0, max_m=4)
        above = phys.face_on_crossing_radii(b + 1.0e-6, 1.0, max_m=4)
        self.assertAlmostEqual(below[0], rs[0], places=4)
        self.assertAlmostEqual(above[0], rs[0], places=4)
        self.assertTrue(all(r > 3.0 for r in above))
        self.assertTrue(all(r > 2.0 for r in below if r is not None))
        self.assertAlmostEqual(above[2], 3.002027687, places=4)
        self.assertAlmostEqual(above[3], 3.003651952, places=4)

    def test_source_roots_for_default_annulus(self):
        peaks = phys.source_crossing_impacts(1.0, 6.0, max_m=4)
        for m, b in enumerate(peaks, start=1):
            self.assertIsNotNone(b, msg=f"missing root m={m}")
            r = phys.face_on_crossing_radii(b, 1.0, max_m=m)[m-1]
            self.assertAlmostEqual(r, 6.0, places=5)

    def test_captured_direct_root_below_critical(self):
        peaks = phys.source_crossing_impacts(1.0, 4.0, max_m=4)
        self.assertIsNotNone(peaks[0])
        r = phys.face_on_crossing_radii(peaks[0], 1.0, max_m=1)[0]
        self.assertAlmostEqual(r, 4.0, places=5)
        self.assertLess(peaks[0], phys.critical_impact_parameter(1.0))

    def test_near_photon_sphere_source_keeps_high_order_roots(self):
        peaks = phys.source_crossing_impacts(1.0, 3.0001, max_m=4)
        for m, b in enumerate(peaks, start=1):
            self.assertIsNotNone(b, msg=f"missing m={m} at r_hot=3.0001")
            r = phys.face_on_crossing_radii(b, 1.0, max_m=m)[m - 1]
            self.assertAlmostEqual(r, 3.0001, places=4)

    def test_source_root_is_a_verified_crossing(self):
        for r_hot in (3.000081, 3.00012, 3.00013, 6.0):
            peaks = phys.source_crossing_impacts(1.0, r_hot, max_m=4)
            for m, b in enumerate(peaks, start=1):
                self.assertIsNotNone(b, msg=f"r_hot={r_hot} missing m={m}")
                r = phys.face_on_crossing_radii(b, 1.0, max_m=m)[m - 1]
                self.assertIsNotNone(r)
                self.assertAlmostEqual(r, r_hot, places=5)
            if r_hot in (3.00012, 3.00013):
                self.assertGreater(peaks[3], phys.critical_impact_parameter(1.0))

    def test_critical_high_m_stops_when_indistinguishable(self):
        rs = phys.face_on_crossing_radii(
            phys.critical_impact_parameter(1.0), 1.0, max_m=20,
        )
        self.assertIsNotNone(rs[10])
        self.assertIsNotNone(rs[11])
        self.assertGreater(rs[10], rs[11])
        self.assertTrue(all(r is None for r in rs[12:]))

    def test_analytic_critical_radii_only_at_exact_b_crit(self):
        b = phys.critical_impact_parameter(1.0)
        exact = phys.face_on_crossing_radii(b, 1.0, max_m=4)
        above = phys.face_on_crossing_radii(math.nextafter(b, math.inf), 1.0, max_m=4)
        below = phys.face_on_crossing_radii(math.nextafter(b, -math.inf), 1.0, max_m=4)
        self.assertAlmostEqual(above[3], exact[3], delta=3.0e-9)
        self.assertAlmostEqual(below[3], exact[3], delta=1.0e-9)

    def test_transfer_resolves_r3_independent_of_fov(self):
        for fov in (8.0, 16.0, 24.0):
            bs, table, _, _ = phys.transfer_curves(1.0, b_max_over_M=0.5*fov, max_m=3)
            n3 = int(np.isfinite(table[:, 2]).sum())
            self.assertGreater(n3, 10, msg=f"fov={fov} n3={n3}")

    def test_table_photon_peak_is_physical(self):
        g4 = (1.0 - 2.0 / 6.0) ** 2
        _, parts = phys.intensity_table(1.0, 12.0, r_hot=6.0)
        for m in range(4):
            self.assertAlmostEqual(float(parts[:, m].max()), g4, delta=0.02)

    def test_table_resolves_documented_r_hot_8(self):
        g4 = (1.0 - 2.0 / 8.0) ** 2
        _, parts = phys.intensity_table(1.0, 16.0, r_hot=8.0)
        for m in range(4):
            self.assertAlmostEqual(
                float(parts[:, m].max()), g4, delta=0.03, msg=f"m={m+1}",
            )

    def test_raster_cannot_exceed_table_peak(self):
        bx, by, _, _ = phys.make_impact_grid(41, 16.8, 1.0)
        _, img2, img3, _, parts, _ = phys.disk_image_components(bx, by, 1.0)
        table_peak = float((parts[:, 2] + parts[:, 3]).max())
        self.assertLessEqual(float((img3 - img2).max()), table_peak + 1.0e-6)

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

    def test_compare_rings_rejects_out_of_range_logM(self):
        with self.assertRaises(ValueError):
            phys.compare_rings(40.0)


class TestModesAndCLI(unittest.TestCase):
    def test_known_modes(self):
        self.assertEqual(
            driver.MODES,
            ("rays", "pixels", "capture", "ring", "transfer", "image",
             "radii", "weak", "compare"),
        )

    def test_pixel_legend_uses_snapped_cell_status(self):
        bx, by, _, _ = phys.make_impact_grid(9, 16.0, 1.0)
        captured = phys.capture_map(bx, by, 1.0)
        fig, _ = plotting.plot_pixels(
            bx, by, captured, M=1.0, b_in=5.1, b_out=6.0,
            n_pix_requested=9, fov_over_M=16.0, show=False,
        )
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        joined = " ".join(labels)
        self.assertIn("escaped", joined)
        self.assertNotIn("captured", joined)

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
        self.assertIn(rf"\sum_{{m=1}}^{{{phys.MAX_IMAGE_M}}}", text)
        self.assertIn(rf"m\ge {phys.MAX_IMAGE_M + 1}", text)
        self.assertIn(phys.image_order_pair_label(), text)
        self.assertNotIn("PhotonOrbit’s EXP-8", text)
        self.assertNotIn("PhotonOrbit EXP-4", text)
        self.assertIn("near-critical whirling", text)
        self.assertIn("mpmath", text)
        self.assertIn("SciPy", text)
        self.assertIn("utilities_bhs", text)
        self.assertNotIn("~50 digits", text)
        self.assertIn("utilities_bhs.MP_DPS", text)
        self.assertNotIn("GFTGU-Documentation", text)
        self.assertIn("GFTGUX-Documentation", text)
        self.assertNotIn("SciPy is optional", text)
        self.assertIn("pixel centre", text)
        self.assertEqual(text.count(r"\("), text.count(r"\)"))
        self.assertIn(r"Narrow \(" + phys.photon_order_label() + r"\) peaks", text)
        self.assertIn(r"photon-ring (\(m\ge 3\))", text)
        self.assertIn(rf"\(m\ge {phys.MAX_IMAGE_M + 1}\) is omitted", text)

    def test_patch_help_version_is_idempotent(self):
        src = find_help_file()
        if src is None:
            self.skipTest("Help file not shipped next to this build")
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "BlackHoleShadow.html"
            dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            phys.patch_help_version(dest)
            phys.patch_help_version(dest)
            text = dest.read_text(encoding="utf-8")
            self.assertEqual(text.count('id="version_build"'), 1)
            self.assertEqual(text.count(phys.BUILD_ID), 1)
            self.assertEqual(text.count(r"\("), text.count(r"\)"))

    def test_help_sync_follows_max_image_m(self):
        src = find_help_file()
        if src is None:
            self.skipTest("Help file not shipped next to this build")
        old = phys.MAX_IMAGE_M
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "BlackHoleShadow.html"
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                phys.MAX_IMAGE_M = 5
                phys.patch_help_version(dest)
                text = dest.read_text(encoding="utf-8")
                self.assertIn(r"\sum_{m=1}^{5}", text)
                self.assertIn(r"photon-ring (\(m\ge 3\))", text)
                self.assertIn(r"\(m\ge 6\) is omitted", text)
                self.assertIn(r"\(m=3..5\)", text)
                phys.MAX_IMAGE_M = 6
                phys.patch_help_version(dest)
                text = dest.read_text(encoding="utf-8")
                self.assertIn(r"\sum_{m=1}^{6}", text)
                self.assertIn(r"photon-ring (\(m\ge 3\))", text)
                self.assertIn(r"\(m\ge 7\) is omitted", text)
                self.assertIn(r"\(m=3..6\)", text)
                self.assertEqual(text.count(r"\("), text.count(r"\)"))
        finally:
            phys.MAX_IMAGE_M = old

    def test_cli_rejects_camera_inside_photon_sphere(self):
        result = run_cli(["--mode", "rays", "--r_cam", "2.5"])
        self.assertNotEqual(result.returncode, 0)

    def test_cli_rejects_unrenderable_image_r_hot(self):
        result = run_cli(["--mode", "image", "--r_hot", "200"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("renderable", (result.stderr + result.stdout).lower())

    def test_even_n_pix_uses_odd_r_hot_cap(self):
        raw = phys.max_image_r_hot_over_M(400)
        odd = phys.max_image_r_hot_over_M(phys.odd_n_pix(400))
        self.assertLess(raw, 185.5)
        self.assertGreater(odd, 185.5)

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

    def test_crossings_are_scale_free(self):
        ref = phys.face_on_crossing_radii(5.3, 1.0, max_m=2)
        ref_a = phys.asymptotic_deflection(1.02 * phys.critical_impact_parameter(1.0), 1.0)
        for M in (1.0e-14, 1.0e-6, 1.0, 1.0e6):
            got = phys.face_on_crossing_radii(5.3 * M, M, max_m=2)
            self.assertAlmostEqual(got[0] / M, ref[0], places=5)
            self.assertAlmostEqual(got[1] / M, ref[1], places=5)
            peri = phys.periapsis(5.3 * M, M)
            self.assertAlmostEqual(peri / M, 3.403321312, places=6)
            alpha = phys.asymptotic_deflection(
                1.02 * phys.critical_impact_parameter(M), M,
            )
            self.assertAlmostEqual(alpha, ref_a, places=5)

    def test_large_r_hot_finds_all_four_branches(self):
        for r_hot in (80.0,):
            peaks = phys.source_crossing_impacts(1.0, r_hot, max_m=4)
            for m, b in enumerate(peaks, start=1):
                self.assertIsNotNone(b, msg=f"r_hot={r_hot} missing m={m}")
                r = phys.face_on_crossing_radii(b, 1.0, max_m=m)[m - 1]
                self.assertAlmostEqual(r / r_hot, 1.0, places=6)

    def test_width_must_be_positive(self):
        with self.assertRaises(ValueError):
            phys.emitted_intensity(6.0, 1.0, r_hot=6.0, width=0.0)
        self.assertGreater(
            phys.emitted_intensity(6.0, 1.0, r_hot=6.0, width=1.0e-12), 0.0,
        )
        with self.assertRaises(ValueError):
            phys.intensity_table(1.0, 12.0, r_hot=6.0, width=1.0e-12)

    def test_bisect_returns_none_without_a_bracket(self):
        self.assertIsNone(phys._bisect_source_root(8.0, 9.0, 1.0, 1, 6.0))

    def test_high_winding_window_survives_large_azimuth_threshold(self):
        lo, hi = phys.high_winding_b_window(1.0, delta_phi_min=12.0)
        self.assertGreater(hi, lo)
        interior = phys.critical_impact_parameter(1.0) * (1.0 + 1.0e-6)
        self.assertTrue(lo < interior <= hi)
        self.assertGreater(phys.escaping_azimuth_from_infinity(hi, 1.0), 12.0)

    def test_higher_order_table_max_is_a_sum(self):
        sample, parts = phys.intensity_table(1.0, 16.0, r_hot=3.000081)
        combined = parts[:, 2:].sum(axis=1)
        self.assertGreater(float(combined.max()), float(parts[:, 2:].max()))

    def test_trapz_wrapper_exists(self):
        self.assertTrue(hasattr(np, "trapezoid") or hasattr(np, "trapz"))
        self.assertAlmostEqual(phys._trapz([0.0, 1.0], [0.0, 1.0]), 0.5)

    def test_plot_image_titles_follow_parts_columns(self):
        bx, by, _, _ = phys.make_impact_grid(9, 16.0, 1.0)
        z = np.zeros_like(bx)
        sample = np.linspace(5.2, 8.0, 8)
        parts = np.zeros((8, 5))
        fig, saved = plotting.plot_image(
            bx, by, z, z, z, M=1.0, r_hot=6.0, peaks=None,
            fov_over_M=16.0, sample=sample, parts=parts, show=False,
        )
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        joined = " ".join(titles) + " " + fig._suptitle.get_text()
        self.assertIn("m=3..5", joined)
        self.assertNotIn("m=3,4", joined)
        self.assertIn("{5}", fig._suptitle.get_text() or "")
        self.assertIn("m>=6", fig._suptitle.get_text() or "")

    def test_photon_order_label_starts_at_m3(self):
        self.assertEqual(phys.photon_order_label(3), "m=3")
        self.assertEqual(phys.photon_order_label(4), "m=3,4")
        self.assertEqual(phys.photon_order_label(5), "m=3..5")

    def test_escaping_ulps_keep_high_order_crossings(self):
        b = phys.critical_impact_parameter(1.0) * (1.0 + 2.05e-15)
        self.assertAlmostEqual(phys.periapsis(b, 1.0) - 3.0, 1.11678677e-7, delta=1.0e-12)
        rho = phys.periapsis(b, 1.0)
        self.assertAlmostEqual(rho**3 - b*b*rho + 2.0*b*b, 0.0, delta=1.0e-12)
        rs = phys.face_on_crossing_radii(b, 1.0, max_m=12)
        none_at = [i for i, r in enumerate(rs) if r is None]
        if none_at:
            self.assertTrue(all(r is None for r in rs[none_at[0]:]))
        self.assertIsNotNone(rs[6])
        self.assertIsNotNone(rs[11])
        self.assertAlmostEqual(rs[11], 12.6952956793829, delta=0.005)
        self.assertLess(rs[11], 13.0)
        self.assertAlmostEqual(2.0 * phys._phi_to_turning_or_horizon(b, 1.0),
                               36.5484022654517, delta=2.0e-7)
        self.assertAlmostEqual(phys._beta_sq_minus_27(6.0), 9.0)

    def test_captured_near_crit_crossing_count(self):
        b = phys.critical_impact_parameter(1.0) * (1.0 - 1.0e-13)
        rs = phys.face_on_crossing_radii(b, 1.0, max_m=12)
        self.assertEqual([r is not None for r in rs],
                         [True] * 10 + [False, False])
        self.assertAlmostEqual(phys._phi_to_turning_or_horizon(b, 1.0),
                               31.358345399346, delta=1.0e-8)

    def test_near_crit_observables_are_scale_free(self):
        beta0 = phys.critical_impact_parameter(1.0) * (1.0 + 2.05e-15)
        ref = phys.periapsis(beta0, 1.0)
        for M in (0.139, 1.0, 7.0):
            self.assertAlmostEqual(phys.periapsis(beta0 * M, M) / M, ref, delta=5.0e-9)

    def test_crossing_lists_have_no_isolated_holes(self):
        b_crit = phys.critical_impact_parameter(1.0)
        for rel in (1.0e-3, 1.0e-8):
            rs = phys.face_on_crossing_radii(b_crit * (1.0 + rel), 1.0, max_m=12)
            seen_none = False
            for r in rs:
                if r is None:
                    seen_none = True
                elif seen_none:
                    self.fail("isolated missing crossing at rel=%g" % rel)

    def test_endpoint_phi_independent_of_ambient_mp_dps(self):
        import mpmath as mp
        b = phys.critical_impact_parameter(1.0) * (1.0 + 2.05e-15)
        phys._mp_phi_to_endpoint.cache_clear()
        phys._mp_periapsis.cache_clear()
        old = mp.mp.dps
        try:
            mp.mp.dps = 15
            a = phys._phi_to_turning_or_horizon(b, 1.0)
            phys._mp_phi_to_endpoint.cache_clear()
            phys._mp_periapsis.cache_clear()
            mp.mp.dps = 80
            c = phys._phi_to_turning_or_horizon(b, 1.0)
        finally:
            mp.mp.dps = old
        self.assertAlmostEqual(2.0 * a, 36.5484022654517, delta=2.0e-7)
        self.assertAlmostEqual(2.0 * a, 2.0 * c, delta=1.0e-12)

    def test_utilities_module_exports(self):
        import utilities_bhs
        self.assertEqual(utilities_bhs.MP_DPS, phys._MP_DPS)
        self.assertTrue(hasattr(utilities_bhs, "phi_to_endpoint_over_M"))
        self.assertTrue(hasattr(utilities_bhs, "periapsis_over_M_mpf"))

    def test_exact_b_crit_is_the_analytic_geodesic(self):
        for M in (0.023, 0.139, 1.0):
            b = phys.critical_impact_parameter(M)
            beta, eps, escaped = phys._dimensionless_state(b, M)
            self.assertFalse(escaped)
            self.assertEqual(eps, 0.0)
            self.assertAlmostEqual(beta, 3.0 * math.sqrt(3.0))
            self.assertTrue(math.isinf(phys._phi_to_turning_or_horizon(b, M)))
            x = (1.0 / 3.0) - 1.0e-10
            u = x / M
            self.assertAlmostEqual(
                phys.phi_from_infinity_inbound(u, b, M),
                phys.critical_phi_of_x(x),
                delta=1.0e-12,
            )

    def test_ordinary_state_avoids_mpmath(self):
        beta, eps, escaped = phys._dimensionless_state(6.0, 1.0)
        self.assertTrue(escaped)
        self.assertAlmostEqual(eps, phys._beta_sq_minus_27(6.0), delta=0.0)

    def test_dekker_compensates_near_beta_c(self):
        beta_c = 3.0 * math.sqrt(3.0)
        prod, err = phys._two_square(beta_c)
        self.assertTrue(math.isfinite(err))
        self.assertTrue(math.isfinite(phys._beta_sq_minus_27(beta_c)))
        self.assertAlmostEqual((prod - 27.0) + err, phys._beta_sq_minus_27(beta_c), delta=1e-18)

    def test_dimensionless_state_keeps_capture_sign(self):
        for M in (0.023, 0.139, 1.0):
            b_crit = phys.critical_impact_parameter(M)
            b_esc = math.nextafter(b_crit, math.inf)
            b_cap = math.nextafter(b_crit, -math.inf)
            _b, eps_e, esc = phys._dimensionless_state(b_esc, M)
            self.assertTrue(esc)
            self.assertGreater(eps_e, 0.0)
            _b, eps_c, esc_c = phys._dimensionless_state(b_cap, M)
            self.assertFalse(esc_c)
            self.assertLess(eps_c, 0.0)
            import mpmath as mp
            with mp.workdps(phys._MP_DPS):
                exact_c = float((mp.mpf(b_cap) / mp.mpf(M)) ** 2 - 27)
            self.assertAlmostEqual(eps_c, exact_c, delta=abs(exact_c) * 1.0e-9 + 1.0e-18)

    def test_captured_nextafter_high_order_radii(self):
        refs = {
            0.023: 2.93004406104,
            0.139: 2.72158614174,
            1.0: 2.79155590796,
        }
        for M, r12_over_M in refs.items():
            b = math.nextafter(phys.critical_impact_parameter(M), -math.inf)
            rs = phys.face_on_crossing_radii(b, M, max_m=12)
            self.assertIsNotNone(rs[11])
            self.assertAlmostEqual(rs[11] / M, r12_over_M, delta=5.0e-9)

    def test_mpf_apis_keep_requested_dps(self):
        import mpmath as mp
        import utilities_bhs
        beta = utilities_bhs.to_mpf(6.0)
        old = mp.mp.dps
        try:
            mp.mp.dps = 15
            a = utilities_bhs.phi_segment_mpf(0, 0.2, beta, dps=30, turning=False)
            mp.mp.dps = 80
            c = utilities_bhs.phi_segment_mpf(0, 0.2, beta, dps=30, turning=False)
        finally:
            mp.mp.dps = old
        self.assertGreaterEqual(a._mpf_[3], 96)
        self.assertGreaterEqual(c._mpf_[3], 96)
        self.assertAlmostEqual(float(a), float(c), places=12)

    def test_quad_fallback_calls_utilities_phi_segment(self):
        import utilities_bhs
        called = {"n": 0}
        orig = utilities_bhs.phi_segment

        def wrapped(*args, **kwargs):
            called["n"] += 1
            return orig(*args, **kwargs)

        utilities_bhs.phi_segment = wrapped
        phys._phi_segment_cached.cache_clear()
        try:
            def fake_quad(*args, **kwargs):
                return (1.0, 1.0)
            old = phys._scipy_quad
            phys._scipy_quad = fake_quad
            try:
                val = phys._phi_quad_segment(0.0, 0.2, 6.0, b=6.0, M=1.0)
            finally:
                phys._scipy_quad = old
        finally:
            utilities_bhs.phi_segment = orig
            phys._phi_segment_cached.cache_clear()
        self.assertGreaterEqual(called["n"], 1)
        self.assertTrue(math.isfinite(val))

    def test_first_escaping_ray_survives_awkward_masses(self):
        for M in (0.139, 0.278, 1.0):
            b_crit = phys.critical_impact_parameter(M)
            b = math.nextafter(b_crit, math.inf)
            self.assertFalse(phys.is_captured(b, M))
            rmin = phys.periapsis(b, M)
            self.assertIsNotNone(rmin)
            self.assertGreater(rmin, 3.0 * M)
            rs = phys.face_on_crossing_radii(b, M, max_m=4)
            self.assertTrue(any(r is not None for r in rs))
            lo, hi = phys.high_winding_b_window(M)
            self.assertGreater(hi, lo)

    def test_high_winding_includes_first_representable_escaping_ray(self):
        lo, hi = phys.high_winding_b_window(1.0, delta_phi_min=38.0)
        b = math.nextafter(phys.critical_impact_parameter(1.0), math.inf)
        self.assertTrue(lo < b <= hi)

    def test_resolved_shadow_is_M_invariant(self):
        a = phys.require_resolved_shadow(1.0, 16.0, 81)
        b = phys.require_resolved_shadow(7.0, 16.0, 81)
        self.assertAlmostEqual(a, b, places=10)


    def test_cli_transfer_and_image_render(self):
        for mode in ("transfer", "image"):
            with tempfile.TemporaryDirectory() as tmp:
                result = run_cli(["--mode", mode, "--n_pix", "21", "--outdir", tmp])
                self.assertEqual(result.returncode, 0, msg=result.stderr)
                pngs = list(Path(tmp).glob("*.png"))
                self.assertTrue(pngs)
                sides = list(Path(tmp).glob("*.provenance.txt"))
                self.assertTrue(sides)

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

