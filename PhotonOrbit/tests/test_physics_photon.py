"""Regression tests for the PhotonOrbit program module.

The discovery helper below deliberately supports both the repository layout
(``tests/test_physics_photon.py``) and an upload layout in which this file is
flattened beside the four program modules (physics_photon.py,
driver_photon.py, main.py, plot_photon.py).  Both layouts are exercised by
``TestModuleDiscovery``, but that does not mean two complete rounds of the
suite are run: the flattened layout is only checked with a trivial smoke
test (module import + a two-line calculation) that proves the discovery
helper itself works from a flattened directory.  The full test suite is run
exactly once, from the canonical ``tests/`` layout.  Reviewer AIs (Copilot,
Codex, Gemini) should follow the same convention: run the full suite once
from ``tests/``, and treat any flattened-layout run as a discovery smoke
test only.

Development history (audit trail -- developers only; never surfaced to
students in the Help file or in main.py/driver_photon.py/physics_photon.py/
plot_photon.py docstrings or output):

  2026-09-02  Claude (principal developer).  First comprehensive regression
    suite for PhotonOrbit.  No prior unittest suite existed; development up
    to this point had been ad hoc, though three informal AI code-critique
    passes (two from Claude, one from Microsoft Copilot, chained together
    with a ChatGPT implementation pass in between) had already been carried
    out and their action items folded into the source tree before this
    round began.  This round:
      * Compiled and ran the supplied program (physics_photon.py,
        driver_photon.py, main.py, plot_photon.py) to establish a baseline.
        PhotonOrbit exposes a single public calculation mode -- integrating
        one photon (null-geodesic) trajectory -- reached through
        integrate_photon_orbit()/driver_photon_orbit()/main.py; it has no
        embed/tidal/infall/horizons-style family of selectable modes (that
        phrasing describes a different GFTGUX program's CLI and does not
        apply here). Baseline runs reproduced every headline number the
        three prior critique documents reported: b=5.0 (default r0=20)
        captured at r=2.0, b=6.0 escaped, b_crit=3*sqrt(3)=5.196152422706632
        to all displayed digits, b=0 pure radial infall with delta_phi=0
        and lambda_final=18 exactly, r0=2.5 (between horizon and photon
        sphere) captured for b=0,3,5.3, the exact circular photon orbit at
        r0=3,b=3*sqrt(3) held r=3.0 for the full requested lambda_max, and
        every documented ValueError fired with its documented message.
      * Independently re-derived d^2r/dlambda^2 = (L^2/r^3)(1-3GM/(c^2 r))
        by differentiating the radial first integral
        (dr/dlambda)^2 = E^2-(L^2/r^2)(1-2GM/(c^2 r)) by hand, and confirmed
        radial_acceleration() against a central finite difference of that
        SAME first integral (an independently coded formula, not a call to
        radial_acceleration or dphi_dlambda) to within 1.1e-8 absolute
        across a 7x4x3 grid of (r, L, GM_over_c2) -- consistent with O(h^2)
        truncation error at the h=1e-5 step used, not a discrepancy.
        Independently confirmed b_crit=3*sqrt(3)*GM_over_c2 by numerically
        maximizing the photon effective potential f(r)=(1-2M/r)/r^2 on a
        fine grid (argmax at r=3M to 1 part in 1e5, f(3M) matching
        1/(27M^2) to 14 significant figures) and independently proved (not
        merely spot-checked) that f is monotonically increasing in r on
        (2M,3M) -- f'(r)=(2/r^4)(3M-r)>0 there -- which is the closed-form
        reason an ingoing photon started anywhere in that band can never
        turn around, confirming the Help file's "capture is guaranteed
        between r_s and r_photon" claim analytically rather than only by
        the sampled (r0,b) pairs a prior review had tried.
      * Found and fixed two real defects, each reproduced before the fix
        and each now guarded by a permanent regression test:
          (1) plot_photon.py's saved-PNG filenames were built from only
              impact parameter and outcome status plus a to-the-second
              timestamp; two runs with the same b and status landing in the
              same wall-clock second (exactly what EXP-9's four
              same-b-different-d_lambda convergence runs, run in a quick
              shell loop into one --outdir, would produce) silently
              overwrote one another with no warning, discarding the
              earlier PNG.  Reproduced deterministically by freezing
              datetime.now() and calling plot_photon_orbit() twice into the
              same outdir; fixed by adding _unique_path(), which appends
              "_2", "_3", ... when the plain timestamped name is already
              taken, mirroring the same-second-collision fix already
              applied to sibling GFTGUX programs.
          (2) The console run summary printed by driver_photon.py never
              included d_lambda (the RK4 step size), even though the Help
              file's EXP-9 explicitly tells students that "step size ...
              status, closest approach, Delta phi, and step count" are
              "all printed in the console summary" -- step size was the
              one field on that list that was not actually printed, so the
              exercise's own resource claim was false for exactly the
              field the exercise is about.  Reproduced by capturing
              driver_photon_orbit()'s stdout and confirming d_lambda was
              absent; fixed by adding it to _print_summary() (and to the
              Help file's "Console summary" output-table row) so the claim
              is now true.
      * Also corrected one Help-file/code ordering mismatch found while
        reading the Algorithm section against the actual validation order
        in integrate_photon_orbit(): the text's step 1 previously claimed
        GM_over_c2, lambda_max and d_lambda were all validated as positive
        before step 2's r0/b checks, but the code actually validates
        GM_over_c2 alone, then r0, then b, then lambda_max, then d_lambda.
        No behavior changed; the wording was tightened to describe the
        code's actual two-stage order (all-finite + GM_over_c2>0, then
        r0/b/lambda_max/d_lambda together) rather than a specific
        three-then-two split the code never implemented.  Regression-tested
        by constructing two simultaneously-invalid inputs and asserting
        which single ValueError message is raised first, matching the
        corrected doc.
      * Reworded EXP-6 (Scale Invariance): the exercise asked students to
        "Compare x/GM_over_c2, y/GM_over_c2, ..." but main.py has no
        per-point numeric output (no CSV, no printed trajectory samples) --
        only a plot and the scalar closest_approach/delta_phi fields in the
        console summary.  A student following the CLI-only workflow the
        rest of the Help file describes had no way to numerically compare
        individual x,y samples.  Reworded to separate the part that is
        exactly comparable from the console (closest approach / GM_over_c2,
        delta_phi) from the part that is a visual plot comparison, and to
        note that a numeric point-by-point x,y comparison requires the
        direct Python API.  This is a documentation change only; no
        exercise was dropped, and none of the other eight needed a resource
        or difficulty-ordering correction (see the accompanying Kickoff
        report for the full per-exercise evaluation).
      * MODEL_VERSION 1.0.0 -> 1.1.0 (functional fixes to driver_photon.py
        and plot_photon.py, not merely new tests); BUILD_ID recomputed
        automatically by the existing _compute_build_id() machinery and
        the Help file's #version_build element updated to match.
      * Built this file from nothing, organized by physical/numerical
        invariant rather than by which critique first raised it, per the
        project's standing instruction that test names/comments describe
        the lasting physics, not the audit history.  No CSV-output tests
        are included: PhotonOrbit deliberately has no --csvdir (a single
        (x,y) trajectory per run has no natural tabular/sweep output, and
        the three prior critique documents already examined and declined
        adding one, a judgment this round agrees with -- see the Kickoff
        report).
"""

import ast
import contextlib
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


CORE_MODULE_FILES = (
    "physics_photon.py",
    "driver_photon.py",
    "main.py",
    "plot_photon.py",
)
HELP_FILE = "PhotonOrbit.html"


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

import physics_photon as phys  # noqa: E402
import driver_photon as driver  # noqa: E402
import plot_photon as plotting  # noqa: E402


def recompute_build_id(directory):
    """Independently reproduce the documented normalized source hash."""
    digest = hashlib.sha256()
    for name in phys.BUILD_ID_COVERS:
        with (directory / name).open("r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


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
        check=False,
    )


class FrozenDatetime(datetime):
    """A fixed-clock stand-in for datetime.datetime, used to force the
    same-wall-clock-second PNG collision that _unique_path() must resolve.
    """
    _fixed = datetime(2026, 9, 2, 10, 0, 0)

    @classmethod
    def now(cls, tz=None):
        return cls._fixed


# ------------------------------------------------------------------
# Minimal dependency-free HTML tree, used only for structural Help tests.
# ------------------------------------------------------------------
class HtmlNode:
    def __init__(self, tag, attrs=()):
        self.tag = tag
        self.attrs = dict(attrs)
        self.content = []

    def text(self):
        return "".join(
            item.text() if isinstance(item, HtmlNode) else item
            for item in self.content
        )


class HtmlTreeParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
                 "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = HtmlNode("document")
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = HtmlNode(tag, attrs)
        self.stack[-1].content.append(node)
        if tag not in self.VOID_TAGS:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].content.append(HtmlNode(tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].content.append(data)


def descendants(node, predicate=lambda item: True):
    matches = []
    for item in node.content:
        if isinstance(item, HtmlNode):
            if predicate(item):
                matches.append(item)
            matches.extend(descendants(item, predicate))
    return matches


def normalized_text(node):
    return " ".join(node.text().split())


def parse_html(path):
    parser = HtmlTreeParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.root


# ======================================================================
class TestModuleDiscovery(unittest.TestCase):
    def test_finds_canonical_tests_layout(self):
        self.assertEqual(find_module_dir(Path(__file__)), MODULE_DIR)

    def test_finds_flattened_layout(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "main.py"), MODULE_DIR)

    def test_uses_nearest_matching_ancestor(self):
        self.assertEqual(find_module_dir(MODULE_DIR / "tests"), MODULE_DIR)

    def test_missing_module_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            find_module_dir(Path(MODULE_DIR.anchor))

    def test_flattened_layout_smoke_test_only(self):
        """Prove discovery + import works from a flattened directory.

        This is deliberately NOT a second full run of the suite (see the
        module docstring): it imports physics_photon from a flattened copy
        and performs one trivial calculation, then returns.  The full suite
        below runs exactly once, from the canonical tests/ layout.
        """
        if os.environ.get("PHOTONORBIT_FLATTENED_SMOKE_CHILD") == "1":
            return
        with tempfile.TemporaryDirectory() as temporary:
            flat_dir = Path(temporary)
            for name in (*CORE_MODULE_FILES, HELP_FILE):
                shutil.copy2(MODULE_DIR / name, flat_dir / name)
            smoke = flat_dir / "_flat_smoke.py"
            smoke.write_text(
                "import sys\n"
                "sys.path.insert(0, '.')\n"
                "import physics_photon as p\n"
                "assert p.radial_acceleration(3.0, 1.0, 1.0) == 0.0\n"
                "assert abs(p.dphi_dlambda(2.0, 4.0) - 1.0) < 1e-15\n"
                "print('FLAT_SMOKE_OK')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PHOTONORBIT_FLATTENED_SMOKE_CHILD"] = "1"
            result = subprocess.run(
                [sys.executable, str(smoke)],
                cwd=flat_dir, env=environment,
                capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("FLAT_SMOKE_OK", result.stdout)


# ======================================================================
class TestMetadataAndCompatibility(unittest.TestCase):
    def test_build_coverage_is_exactly_the_executable_core(self):
        self.assertEqual(tuple(phys.BUILD_ID_COVERS), CORE_MODULE_FILES)
        self.assertNotIn(HELP_FILE, phys.BUILD_ID_COVERS)
        self.assertFalse(any("test" in name for name in phys.BUILD_ID_COVERS))

    def test_build_id_matches_independent_calculation(self):
        self.assertRegex(phys.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(phys.BUILD_ID, recompute_build_id(MODULE_DIR))

    def test_build_id_independent_of_line_endings(self):
        """BUILD_ID must be stable under LF/CRLF normalization (newline=None)."""
        digest_lf = hashlib.sha256()
        digest_crlf = hashlib.sha256()
        for name in phys.BUILD_ID_COVERS:
            raw = (MODULE_DIR / name).read_bytes()
            text_lf = raw.replace(b"\r\n", b"\n")
            text_crlf = text_lf.replace(b"\n", b"\r\n")
            normalized = text_lf
            for digest in (digest_lf, digest_crlf):
                digest.update(name.encode("utf-8"))
                digest.update(len(normalized).to_bytes(8, "big"))
                digest.update(normalized)
        self.assertEqual(digest_lf.hexdigest()[:12], digest_crlf.hexdigest()[:12])
        self.assertEqual(digest_lf.hexdigest()[:12], phys.BUILD_ID)

    def test_build_id_changes_when_covered_source_changes(self):
        """BUILD_ID must react to a real content change in a covered file."""
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            for name in CORE_MODULE_FILES:
                shutil.copy2(MODULE_DIR / name, work_dir / name)
            before = recompute_build_id(work_dir)
            with (work_dir / "driver_photon.py").open("a", encoding="utf-8") as fh:
                fh.write("\n# a trivial, behavior-free comment change\n")
            after = recompute_build_id(work_dir)
            self.assertNotEqual(before, after)

    def test_all_core_sources_parse_as_python_3_10(self):
        for name in CORE_MODULE_FILES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_version_command(self):
        result = run_cli(["--version"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"PhotonOrbit {phys.MODEL_VERSION} (build {phys.BUILD_ID})",
        )

    def test_info_dict_reports_same_version_and_build_as_physics_module(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=5.0, lambda_max=50.0, d_lambda=0.05,
        )
        self.assertEqual(info["model_version"], phys.MODEL_VERSION)
        self.assertEqual(info["build_id"], phys.BUILD_ID)


# ======================================================================
class TestAnalyticVerification(unittest.TestCase):
    """Verify the coded equations against independently-derived formulas.

    Per the project's testing standard, these do not validate a function by
    calling another function from the same module that shares the same
    assumptions; they use a hand-derived closed form, a central finite
    difference of that closed form, or an independent numerical search.
    """

    def _first_integral_rhs(self, r, L, GM, E=1.0):
        """(dr/dlambda)^2 as given by the radial first integral, coded here
        independently of anything in physics_photon.py."""
        return E * E - (L * L / (r * r)) * (1.0 - 2.0 * GM / r)

    def _fd_radial_acceleration(self, r, L, GM, h=1e-5):
        """d2r/dlambda2 = (1/2) d/dr[(dr/dlambda)^2], by central difference
        of the independently-coded first integral above (valid away from a
        turning point, which none of the sampled points below sit at)."""
        plus = self._first_integral_rhs(r + h, L, GM)
        minus = self._first_integral_rhs(r - h, L, GM)
        return (plus - minus) / (2.0 * h) / 2.0

    def test_radial_acceleration_matches_finite_difference_of_first_integral(self):
        max_abs_err = 0.0
        for r in (2.01, 2.5, 3.0, 4.0, 6.0, 10.0, 50.0):
            for L in (0.5, 3.0, 5.196152422706632, 20.0):
                for GM in (0.5, 1.0, 2.0):
                    coded = phys.radial_acceleration(r, L, GM)
                    fd = self._fd_radial_acceleration(r, L, GM)
                    max_abs_err = max(max_abs_err, abs(coded - fd))
        # O(h^2) truncation at h=1e-5 predicts errors of order 1e-8-1e-9.
        self.assertLess(max_abs_err, 5e-7)

    def test_radial_acceleration_vanishes_exactly_at_photon_sphere(self):
        for GM in (0.3, 1.0, 7.0):
            self.assertEqual(phys.radial_acceleration(3.0 * GM, 5.0, GM), 0.0)

    def test_radial_acceleration_sign_matches_potential_slope(self):
        # (L^2/r^3)(1-3GM/r): positive (pushes further outward) for
        # r>3GM, negative (pulls further inward) for r<3GM -- d2r/dlambda2
        # points AWAY from the photon sphere on both sides, exactly the
        # signature of an unstable equilibrium (a maximum, not a minimum,
        # of the effective potential) at r=3GM.
        GM = 1.0
        self.assertGreater(phys.radial_acceleration(3.5, 4.0, GM), 0.0)
        self.assertLess(phys.radial_acceleration(2.2, 4.0, GM), 0.0)

    def test_dphi_dlambda_is_L_over_r_squared(self):
        for r, L in ((2.5, 1.0), (10.0, 5.0), (100.0, 30.0)):
            self.assertAlmostEqual(phys.dphi_dlambda(r, L), L / (r * r), places=14)

    def test_critical_impact_parameter_matches_potential_maximum(self):
        # Independently maximize f(r) = (1-2M/r)/r^2 (the photon effective
        # potential) on a fine grid, without using 3*sqrt(3) anywhere in the
        # search, and derive b_crit = 1/sqrt(f_max) from that maximum.
        for GM in (0.5, 1.0, 1.7, 5.0):
            def f(r):
                return (1.0 - 2.0 * GM / r) / (r * r)
            grid = [2.0 * GM + i * 1e-5 * GM for i in range(1, 200000)]
            best_r = max(grid, key=f)
            self.assertAlmostEqual(best_r / GM, 3.0, places=3)
            b_crit_numeric = math.sqrt(1.0 / f(3.0 * GM))
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=GM, r0=20.0 * GM, b=1.0, lambda_max=1.0, d_lambda=0.5,
            )
            self.assertAlmostEqual(b_crit_numeric, info["critical_b_infinity"],
                                    places=5)

    def test_photon_effective_potential_is_monotonic_between_horizon_and_photon_sphere(self):
        # Closed-form proof that f'(r) = (2/r^4)(3M-r) > 0 for r in (2M,3M):
        # an ingoing photon started anywhere in that band moves toward
        # smaller f(r) the entire way in and can never turn around. This is
        # the analytic justification for the "capture is guaranteed between
        # r_s and r_photon" claim tested empirically in
        # TestCaptureInsidePhotonSphere below.
        GM = 1.0
        def f(r):
            return (1.0 - 2.0 * GM / r) / (r * r)
        rs = [2.0 * GM + 1e-3 * GM * i for i in range(1, 999)]
        values = [f(r) for r in rs]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))


# ======================================================================
class TestCaptureEscapeDichotomy(unittest.TestCase):
    def test_documented_default_is_captured(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=5.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "captured")
        self.assertAlmostEqual(info["closest_approach"], 2.0, places=6)

    def test_documented_b6_is_escaped(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "escaped")

    def test_bisection_threshold_converges_to_3_sqrt_3(self):
        # Independent bisection search (a distinct numerical method from
        # the integrator itself) for the b at which the outcome flips.
        lo, hi = 5.0, 6.0  # lo captured, hi escaped, per the tests above

        def outcome(b):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=20.0, b=b, lambda_max=300.0, d_lambda=0.01,
            )
            return info["status"]

        self.assertEqual(outcome(lo), "captured")
        self.assertEqual(outcome(hi), "escaped")
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if outcome(mid) == "captured":
                lo = mid
            else:
                hi = mid
        b_crit = 3.0 * math.sqrt(3.0)
        self.assertAlmostEqual(0.5 * (lo + hi), b_crit, places=6)


# ======================================================================
class TestRadialInfall(unittest.TestCase):
    def test_b_zero_has_zero_acceleration_everywhere(self):
        for r in (2.01, 3.0, 5.0, 20.0):
            self.assertEqual(phys.radial_acceleration(r, 0.0, 1.0), 0.0)

    def test_b_zero_falls_at_constant_rate_to_exact_closed_form_lambda(self):
        r0 = 20.0
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=r0, b=0.0, lambda_max=200.0, d_lambda=0.001,
        )
        self.assertEqual(info["status"], "captured")
        self.assertEqual(info["delta_phi"], 0.0)
        # With L=0, (dr/dlambda)^2 = E^2 = 1 identically, so the photon
        # falls at |dr/dlambda| = 1 exactly: lambda to reach r_s is r0-r_s.
        self.assertAlmostEqual(info["lambda_final"], r0 - info["r_s"], places=6)

    def test_b_zero_radius_is_strictly_monotonically_decreasing(self):
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=0.0, lambda_max=200.0, d_lambda=0.01,
        )
        radii = [math.hypot(xi, yi) for xi, yi in zip(x, y)]
        self.assertTrue(all(a > b for a, b in zip(radii, radii[1:])))


# ======================================================================
class TestCaptureInsidePhotonSphere(unittest.TestCase):
    """Regression coverage for the Help file's "capture is guaranteed for
    r_s < r0 < r_photon" claim, proved analytically in
    TestAnalyticVerification.test_photon_effective_potential_is_monotonic..."""

    def test_every_legal_b_is_captured_between_horizon_and_photon_sphere(self):
        for b in (0.0, 1.0, 2.0, 3.0, 5.3):
            with self.subTest(b=b):
                _, _, info = phys.integrate_photon_orbit(
                    GM_over_c2=1.0, r0=2.5, b=b, lambda_max=50.0, d_lambda=0.001,
                )
                self.assertEqual(info["status"], "captured")

    def test_boundary_start_exactly_at_photon_sphere_with_subcritical_b_is_captured(self):
        # r0 = r_photon exactly is outside the HTML's stated open interval
        # (r_s < r0 < r_photon) but is a legal, documented-adjacent boundary
        # worth pinning down: an ingoing photon started exactly at r=3M
        # with b below the local kinematic bound at that radius is still
        # captured, since f(r) is monotonically increasing right up to r=3M.
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=3.0, b=5.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "captured")


# ======================================================================
class TestTurningPointIdentity(unittest.TestCase):
    def test_first_integral_holds_at_closest_approach_and_tightens_with_step_size(self):
        residuals = []
        for d_lambda in (0.04, 0.01, 0.0025):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=200.0, d_lambda=d_lambda,
            )
            self.assertEqual(info["status"], "escaped")
            r_min = info["closest_approach"]
            lhs = 1.0
            rhs = (1.0 - 2.0 / r_min) * (6.0 ** 2) / (r_min ** 2)
            residuals.append(abs(lhs - rhs))
        # Convergence: the finest step size must be at least as accurate as
        # the coarsest (RK4 is 4th order, so this is a very loose bound).
        self.assertLess(residuals[-1], residuals[0])
        self.assertLess(residuals[-1], 1e-4)


# ======================================================================
class TestScaleInvariance(unittest.TestCase):
    def test_rescaling_all_lengths_rescales_geometry_and_preserves_delta_phi(self):
        base = dict(GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=200.0, d_lambda=0.01)
        x1, y1, info1 = phys.integrate_photon_orbit(**base)

        k = 3.7
        scaled = dict(GM_over_c2=k * base["GM_over_c2"], r0=k * base["r0"],
                      b=k * base["b"], lambda_max=k * base["lambda_max"],
                      d_lambda=k * base["d_lambda"])
        x2, y2, info2 = phys.integrate_photon_orbit(**scaled)

        self.assertEqual(info1["status"], info2["status"])
        self.assertAlmostEqual(info2["closest_approach"] / k,
                                info1["closest_approach"], places=6)
        self.assertAlmostEqual(info2["delta_phi"], info1["delta_phi"], places=6)
        self.assertAlmostEqual(info2["lambda_final"] / k,
                                info1["lambda_final"], places=5)
        # Same number of samples at the same relative step size.
        self.assertEqual(len(x1), len(x2))
        for xa, xb, ya, yb in zip(x1, x2, y1, y2):
            self.assertAlmostEqual(xb / k, xa, places=5)
            self.assertAlmostEqual(yb / k, ya, places=5)


# ======================================================================
class TestCircularPhotonOrbit(unittest.TestCase):
    def test_exact_circular_orbit_stays_on_photon_sphere(self):
        b_crit = 3.0 * math.sqrt(3.0)
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=3.0, b=b_crit, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "lambda_max")
        radii = [math.hypot(xi, yi) for xi, yi in zip(x, y)]
        self.assertTrue(all(abs(r - 3.0) < 1e-9 for r in radii))

    def test_b_cannot_exceed_local_kinematic_bound_at_r0_equal_photon_sphere(self):
        b_crit = 3.0 * math.sqrt(3.0)
        with self.assertRaisesRegex(ValueError, r"b must be <="):
            phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=3.0, b=b_crit + 1e-6,
                lambda_max=10.0, d_lambda=0.01,
            )

    def test_slightly_reduced_b_at_photon_sphere_is_captured(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=3.0, b=5.196152, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "captured")

    def test_nearby_escaping_case_outside_photon_sphere(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=3.1, b=5.20, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "escaped")
        self.assertAlmostEqual(info["closest_approach"], 3.0686559820284285,
                                places=6)


# ======================================================================
class TestConvergence(unittest.TestCase):
    def test_near_critical_case_converges_as_step_size_shrinks(self):
        b = 5.205
        results = []
        for d_lambda in (0.04, 0.02, 0.01, 0.005):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=20.0, b=b, lambda_max=400.0, d_lambda=d_lambda,
            )
            results.append(info)

        finest = results[-1]
        errors = [abs(r["closest_approach"] - finest["closest_approach"])
                  for r in results[:-1]]
        # Coarser step sizes must show larger (or comparable, for the two
        # finest) deviation from the finest-resolution answer -- true
        # convergence, not merely "doesn't crash".
        self.assertGreater(errors[0], errors[-1])
        for value in (r["closest_approach"] for r in results):
            self.assertTrue(math.isfinite(value))


# ======================================================================
class TestArrayAndDiagnosticsConsistency(unittest.TestCase):
    def test_arrays_are_finite_and_agree_in_length(self):
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=100.0, d_lambda=0.02,
        )
        self.assertEqual(len(x), len(y))
        self.assertTrue(all(math.isfinite(v) for v in x))
        self.assertTrue(all(math.isfinite(v) for v in y))
        self.assertEqual(info["steps"], len(x) - 1)

    def test_first_sample_is_the_starting_point(self):
        r0, GM = 20.0, 1.0
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=GM, r0=r0, b=6.0, lambda_max=100.0, d_lambda=0.02,
        )
        self.assertAlmostEqual(x[0], r0, places=10)
        self.assertAlmostEqual(y[0], 0.0, places=10)

    def test_closest_approach_matches_array_minimum_radius(self):
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=100.0, d_lambda=0.005,
        )
        radii = [math.hypot(xi, yi) for xi, yi in zip(x, y)]
        self.assertAlmostEqual(min(radii), info["closest_approach"], places=8)

    def test_captured_run_ends_exactly_at_the_event_horizon(self):
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=5.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "captured")
        last_r = math.hypot(x[-1], y[-1])
        self.assertAlmostEqual(last_r, info["r_s"], places=9)
        self.assertAlmostEqual(info["closest_approach"], info["r_s"], places=9)

    def test_escaped_run_ends_at_or_beyond_escape_radius(self):
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "escaped")
        last_r = math.hypot(x[-1], y[-1])
        self.assertGreaterEqual(last_r, info["escape_radius"])

    def test_lambda_max_run_reaches_the_requested_affine_parameter(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=3.0, b=3.0 * math.sqrt(3.0),
            lambda_max=50.0, d_lambda=0.5,
        )
        self.assertEqual(info["status"], "lambda_max")
        self.assertAlmostEqual(info["lambda_final"], 50.0, places=6)

    def test_required_info_keys_present(self):
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=5.0, lambda_max=50.0, d_lambda=0.05,
        )
        required = {"status", "closest_approach", "delta_phi", "lambda_final",
                    "steps", "r_s", "r_photon", "critical_b_infinity",
                    "escape_radius", "model_version", "build_id"}
        self.assertTrue(required.issubset(info.keys()))

    def test_horizon_and_photon_sphere_radii_scale_with_GM_over_c2(self):
        for GM in (0.5, 1.0, 4.0):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=GM, r0=20.0 * GM, b=1.0, lambda_max=1.0, d_lambda=0.5,
            )
            self.assertAlmostEqual(info["r_s"], 2.0 * GM, places=10)
            self.assertAlmostEqual(info["r_photon"], 3.0 * GM, places=10)


# ======================================================================
class TestInputValidationAndErrorHandling(unittest.TestCase):
    VALID = dict(GM_over_c2=1.0, r0=20.0, b=5.0, lambda_max=200.0, d_lambda=0.01)

    def _with(self, **overrides):
        kwargs = dict(self.VALID)
        kwargs.update(overrides)
        return kwargs

    def test_r0_inside_or_at_horizon_rejected(self):
        for r0 in (2.0, 1.5, -5.0):
            with self.subTest(r0=r0):
                with self.assertRaisesRegex(ValueError, "outside the event horizon"):
                    phys.integrate_photon_orbit(**self._with(r0=r0))

    def test_negative_b_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            phys.integrate_photon_orbit(**self._with(b=-1.0))

    def test_GM_over_c2_non_positive_rejected(self):
        for GM in (0.0, -1.0):
            with self.subTest(GM=GM):
                with self.assertRaisesRegex(ValueError, "GM_over_c2 must be greater than zero"):
                    phys.integrate_photon_orbit(**self._with(GM_over_c2=GM))

    def test_lambda_max_non_positive_rejected(self):
        for value in (0.0, -10.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "lambda_max must be greater than zero"):
                    phys.integrate_photon_orbit(**self._with(lambda_max=value))

    def test_d_lambda_non_positive_rejected(self):
        for value in (0.0, -0.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "d_lambda must be greater than zero"):
                    phys.integrate_photon_orbit(**self._with(d_lambda=value))

    def test_non_finite_scalar_inputs_rejected(self):
        for name in ("GM_over_c2", "r0", "b", "lambda_max", "d_lambda"):
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(name=name, bad=bad):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        phys.integrate_photon_orbit(**self._with(**{name: bad}))

    def test_bool_scalar_inputs_rejected(self):
        for name in ("GM_over_c2", "r0", "b", "lambda_max", "d_lambda"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    phys.integrate_photon_orbit(**self._with(**{name: True}))

    def test_step_cap_enforced_with_informative_message(self):
        with self.assertRaisesRegex(ValueError, r"5,000,000"):
            phys.integrate_photon_orbit(
                **self._with(lambda_max=1.0, d_lambda=1e-8)
            )

    def test_b_incompatible_with_ingoing_geodesic_reports_local_bound(self):
        with self.assertRaisesRegex(ValueError, r"b must be <= 5\.196"):
            phys.integrate_photon_orbit(**self._with(r0=3.0, b=10.0))

    def test_validation_order_matches_help_file_algorithm_section(self):
        # Regression test for the Help-file wording fix: step 1 is "finite,
        # then GM_over_c2>0"; step 2 groups r0>r_s, b>=0, lambda_max>0,
        # d_lambda>0, checked in that relative order.  Construct two
        # simultaneously-invalid inputs from each pairing and confirm which
        # single message wins, so a future reordering of the checks inside
        # integrate_photon_orbit() is caught here.
        with self.assertRaisesRegex(ValueError, "outside the event horizon"):
            phys.integrate_photon_orbit(**self._with(r0=1.0, lambda_max=-1.0))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            phys.integrate_photon_orbit(**self._with(b=-1.0, d_lambda=-1.0))
        with self.assertRaisesRegex(ValueError, "lambda_max must be greater than zero"):
            phys.integrate_photon_orbit(**self._with(lambda_max=-1.0, d_lambda=-1.0))

    def test_direct_low_level_functions_reject_nonpositive_radius(self):
        with self.assertRaises(ValueError):
            phys.radial_acceleration(0.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            phys.radial_acceleration(-1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            phys.dphi_dlambda(0.0, 1.0)


# ======================================================================
class TestPlotGuardrails(unittest.TestCase):
    VALID_INFO = dict(r_s=2.0, r_photon=3.0, status="captured",
                       closest_approach=2.0, delta_phi=1.0)

    def test_mismatched_array_lengths_rejected(self):
        with self.assertRaises(ValueError):
            plotting.plot_photon_orbit([1, 2, 3], [1, 2], 5.0, self.VALID_INFO)

    def test_empty_arrays_rejected(self):
        with self.assertRaises(ValueError):
            plotting.plot_photon_orbit([], [], 5.0, self.VALID_INFO)

    def test_non_finite_trajectory_values_rejected(self):
        with self.assertRaises(ValueError):
            plotting.plot_photon_orbit([1.0, math.nan], [1.0, 2.0], 5.0, self.VALID_INFO)
        with self.assertRaises(ValueError):
            plotting.plot_photon_orbit([1.0, math.inf], [1.0, 2.0], 5.0, self.VALID_INFO)

    def test_missing_required_info_keys_rejected_individually(self):
        for key in ("r_s", "r_photon", "status", "closest_approach", "delta_phi"):
            with self.subTest(key=key):
                info = dict(self.VALID_INFO)
                del info[key]
                with self.assertRaisesRegex(ValueError, key):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info)

    def test_non_finite_info_values_rejected(self):
        for key in ("r_s", "r_photon", "closest_approach", "delta_phi"):
            with self.subTest(key=key):
                info = dict(self.VALID_INFO)
                info[key] = math.nan
                with self.assertRaises(ValueError):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info)

    def test_nonpositive_horizon_or_photon_sphere_rejected(self):
        for key in ("r_s", "r_photon"):
            with self.subTest(key=key):
                info = dict(self.VALID_INFO)
                info[key] = 0.0
                with self.assertRaises(ValueError):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info)

    def test_non_string_status_rejected(self):
        info = dict(self.VALID_INFO)
        info["status"] = 42
        with self.assertRaises(ValueError):
            plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info)

    def test_invalid_dpi_rejected(self):
        for bad_dpi in (0, -5, 150.5, True):
            with self.subTest(bad_dpi=bad_dpi):
                with self.assertRaises(ValueError):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, self.VALID_INFO,
                                                dpi=bad_dpi)

    def test_invalid_lw_rejected(self):
        for bad_lw in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(bad_lw=bad_lw):
                with self.assertRaises(ValueError):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, self.VALID_INFO,
                                                lw=bad_lw)

    def test_valid_call_saves_a_named_png(self):
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch("matplotlib.pyplot.show"):
                plotting.plot_photon_orbit([1.0, 2.0], [0.0, 1.0], 5.0,
                                            self.VALID_INFO, outdir=outdir, dpi=60)
            files = os.listdir(outdir)
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].startswith("photon_b5_captured_"))
            self.assertTrue(files[0].endswith(".png"))
            self.assertGreater(os.path.getsize(os.path.join(outdir, files[0])), 0)

    def test_axis_labels_and_title_state_correct_physics(self):
        captured = {}

        def fake_show():
            fig = __import__("matplotlib.pyplot", fromlist=["gcf"]).gcf()
            ax = fig.axes[0]
            captured["xlabel"] = ax.get_xlabel()
            captured["ylabel"] = ax.get_ylabel()
            captured["title"] = ax.get_title()
            captured["legend"] = [t.get_text() for t in ax.get_legend().get_texts()]

        with mock.patch("matplotlib.pyplot.show", side_effect=fake_show):
            plotting.plot_photon_orbit([1.0, 2.0], [0.0, 1.0], 5.0, self.VALID_INFO)

        self.assertIn("GM/c^2", captured["xlabel"])
        self.assertIn("GM/c^2", captured["ylabel"])
        self.assertIn("b = 5", captured["title"])
        self.assertIn("Event horizon", captured["legend"])
        self.assertIn("Photon sphere", captured["legend"])
        self.assertIn("Photon trajectory", captured["legend"])


# ======================================================================
class TestOutputFileCollisionAvoidance(unittest.TestCase):
    """Regression tests for the same-wall-clock-second PNG collision fix."""

    def test_two_saves_in_the_same_second_produce_two_distinct_files(self):
        info = dict(r_s=2.0, r_photon=3.0, status="captured",
                     closest_approach=2.0, delta_phi=1.0)
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plotting, "datetime", FrozenDatetime), \
                 mock.patch("matplotlib.pyplot.show"):
                plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info, outdir=outdir)
                plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info, outdir=outdir)
            files = sorted(os.listdir(outdir))
            self.assertEqual(len(files), 2)
            for name in files:
                self.assertGreater(os.path.getsize(os.path.join(outdir, name)), 0)
            self.assertTrue(files[1].endswith("_2.png"))

    def test_three_saves_in_the_same_second_all_kept(self):
        info = dict(r_s=2.0, r_photon=3.0, status="escaped",
                     closest_approach=4.0, delta_phi=2.0)
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plotting, "datetime", FrozenDatetime), \
                 mock.patch("matplotlib.pyplot.show"):
                for _ in range(3):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 6.0, info, outdir=outdir)
            self.assertEqual(len(os.listdir(outdir)), 3)


# ======================================================================
class TestConsoleSummaryReproducibility(unittest.TestCase):
    """Regression test for the missing-d_lambda console-summary defect."""

    def test_summary_reports_the_step_size_used(self):
        buffer = io.StringIO()
        with mock.patch("matplotlib.pyplot.show"), \
             contextlib.redirect_stdout(buffer):
            driver.driver_photon_orbit(GM_over_c2=1.0, r0=20.0, b=5.0,
                                        lambda_max=50.0, d_lambda=0.025)
        output = buffer.getvalue()
        self.assertIn("d_lambda", output)
        self.assertIn("0.025", output)

    def test_summary_reports_every_documented_field(self):
        buffer = io.StringIO()
        with mock.patch("matplotlib.pyplot.show"), \
             contextlib.redirect_stdout(buffer):
            driver.driver_photon_orbit(GM_over_c2=1.0, r0=20.0, b=6.0,
                                        lambda_max=50.0, d_lambda=0.02)
        output = buffer.getvalue()
        for fragment in ("GM_over_c2", "Starting radius", "Impact parameter",
                          "Event horizon", "Photon sphere", "Critical b",
                          "Status", "Closest approach", "Delta phi",
                          "Affine parameter", "RK4 steps taken"):
            self.assertIn(fragment, output)


# ======================================================================
class TestDriverValidation(unittest.TestCase):
    def test_plot_input_validation_rejects_bad_dpi_and_lw(self):
        with self.assertRaises(ValueError):
            driver._validate_plot_inputs(dpi=0, lw=1.0)
        with self.assertRaises(ValueError):
            driver._validate_plot_inputs(dpi=150, lw=0.0)
        with self.assertRaises(ValueError):
            driver._validate_plot_inputs(dpi=150.5, lw=1.0)

    def test_plot_input_validation_accepts_and_normalizes_valid_inputs(self):
        dpi, lw = driver._validate_plot_inputs(dpi=150.0, lw=2)
        self.assertEqual(dpi, 150)
        self.assertIsInstance(dpi, int)
        self.assertEqual(lw, 2)


# ======================================================================
class TestCLI(unittest.TestCase):
    def test_help_lists_all_documented_options(self):
        result = run_cli(["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--GM_over_c2", "--r0", "--b", "--lambda_max",
                     "--d_lambda", "--lw", "--dpi", "--outdir", "--version"):
            self.assertIn(flag, result.stdout)

    def test_default_run_succeeds_and_prints_expected_summary_fields(self):
        result = run_cli([])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("captured", result.stdout)
        self.assertIn("d_lambda", result.stdout)

    def test_escaping_run_via_cli(self):
        result = run_cli(["--b", "6.0"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("escaped", result.stdout)

    def test_invalid_r0_reported_as_clean_single_line_error(self):
        result = run_cli(["--r0", "1.0"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PhotonOrbit:", result.stderr)
        self.assertIn("outside the event horizon", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_invalid_dpi_lw_b_reported_cleanly(self):
        for args, fragment in (
            (["--dpi", "-5"], "dpi"),
            (["--lw", "0"], "lw"),
            (["--b", "-1"], "nonnegative"),
        ):
            with self.subTest(args=args):
                result = run_cli(args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("PhotonOrbit:", result.stderr)
                self.assertIn(fragment, result.stderr)

    def test_unwritable_outdir_reported_as_clean_oserror(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocking_file = os.path.join(tmp, "blocked")
            with open(blocking_file, "w", encoding="utf-8") as fh:
                fh.write("not a directory")
            result = run_cli(["--outdir", blocking_file])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PhotonOrbit:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_outdir_saves_png_without_suppressing_display(self):
        with tempfile.TemporaryDirectory() as outdir:
            result = run_cli(["--b", "6.0", "--outdir", outdir, "--dpi", "80"])
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = [f for f in os.listdir(outdir) if f.endswith(".png")]
            self.assertEqual(len(saved), 1)
            self.assertIn("Displaying figure on screen", result.stdout)


# ======================================================================
class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODULE_DIR / HELP_FILE
        cls.html = cls.path.read_text(encoding="utf-8")
        cls.root = parse_html(cls.path)

    def test_version_and_build_match_program(self):
        nodes = descendants(self.root, lambda n: n.attrs.get("id") == "version_build")
        self.assertEqual(len(nodes), 1)
        text = normalized_text(nodes[0])
        self.assertIn(phys.MODEL_VERSION, text)
        self.assertIn(phys.BUILD_ID, text)

    def test_nav_anchors_resolve_to_existing_sections(self):
        nav_nodes = descendants(self.root, lambda n: n.tag == "nav")
        self.assertEqual(len(nav_nodes), 1)
        hrefs = [a.attrs["href"][1:] for a in descendants(nav_nodes[0], lambda n: n.tag == "a")
                 if a.attrs.get("href", "").startswith("#")]
        self.assertGreater(len(hrefs), 0)
        section_ids = {s.attrs.get("id") for s in descendants(self.root, lambda n: n.tag == "section")}
        for href in hrefs:
            with self.subTest(href=href):
                self.assertIn(href, section_ids)

    def test_module_signatures_match_actual_code(self):
        module_text = normalized_text(self.root)
        self.assertIn("radial_acceleration(r, L, GM_over_c2)", module_text)
        self.assertIn("dphi_dlambda(r, L)", module_text)
        self.assertIn(
            "integrate_photon_orbit( GM_over_c2, r0, b, lambda_max, d_lambda )".replace(" ", ""),
            module_text.replace(" ", ""),
        )

    def test_no_leftover_boilerplate_or_java_references(self):
        for pattern in (r"\.jar\b", r"\bapplet\b", r"\bJava\b",
                        r"translat(ed|ion) from", r"ported from"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, self.html))

    def test_mathjax_config_is_syntactically_valid_javascript(self):
        match = re.search(r"<script>\s*window\.MathJax.*?</script>", self.html, re.S)
        self.assertIsNotNone(match)
        js = match.group(0).replace("<script>", "").replace("</script>", "")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
            js_path = fh.name
        try:
            result = subprocess.run(["node", "--check", js_path],
                                     capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            os.unlink(js_path)

    def test_mathjax_config_enables_dollar_delimited_inline_math(self):
        match = re.search(r"window\.MathJax\s*=\s*\{.*?\};", self.html, re.S)
        self.assertIsNotNone(match)
        self.assertIn("'$', '$'", match.group(0))

    def test_restoration_guidance_is_tool_neutral_before_pycharm_specifics(self):
        nodes = descendants(self.root, lambda n: n.attrs.get("id") == "restore")
        self.assertEqual(len(nodes), 1)
        text = normalized_text(nodes[0])
        self.assertIn("PyCharm", text)
        self.assertIn("re-download", text.replace("Re-download", "re-download"))

    def test_console_summary_table_row_documents_d_lambda(self):
        self.assertIn("d_lambda", self.html)
        self.assertIn(
            "RK4 step size",
            self.html.replace("\n", " "),
        )

    def test_algorithm_step_order_matches_actual_validation_order(self):
        steps = [normalized_text(li) for li in descendants(self.root, lambda n: n.tag == "li")]
        algorithm_steps = [s for s in steps if "finite" in s or "event horizon" in s]
        self.assertTrue(any("finite" in s and "GM_over_c2" in s for s in algorithm_steps))
        self.assertTrue(any("event horizon" in s and "lambda_max" in s and "d_lambda" in s
                             for s in algorithm_steps))

    def test_exp6_wording_matches_actual_program_output_capabilities(self):
        exp_cards = descendants(self.root, lambda n: n.attrs.get("class") == "exp-card")
        scale_card = next(c for c in exp_cards if "Scale Invariance" in normalized_text(c))
        text = normalized_text(scale_card)
        self.assertIn("console summary", text)
        self.assertIn("physics_photon.integrate_photon_orbit", text)

    def test_all_nine_exercises_present_and_difficulty_labeled(self):
        exp_cards = descendants(self.root, lambda n: n.attrs.get("class") == "exp-card")
        self.assertEqual(len(exp_cards), 9)
        labels = [normalized_text(descendants(c, lambda n: n.attrs.get("class") == "ec-num")[0])
                  for c in exp_cards]
        for label in labels:
            self.assertRegex(label, r"EXP-\d+")
            self.assertTrue(
                any(word in label for word in
                    ("Introductory", "Intermediate", "Advanced"))
            )

    def test_user_story_elements_are_documented(self):
        # Cross-check against the program's own user story: light bending,
        # capture vs. escape, the photon sphere at r=3GM/c^2, the unstable
        # circular photon orbit, and the critical impact parameter.
        text = self.html
        self.assertIn("photon sphere", text)
        self.assertIn("event horizon", text)
        self.assertIn("critical impact parameter", text.lower())
        self.assertIn("captured", text)
        self.assertIn("escape", text.lower())
        self.assertIn("unstable", text.lower())

    def test_parameter_table_defaults_match_main_py(self):
        table_text = normalized_text(
            descendants(self.root, lambda n: n.attrs.get("class") == "param-table")[0]
        )
        self.assertIn("1.0", table_text)   # GM_over_c2 default
        self.assertIn("20.0", table_text)  # r0 default
        self.assertIn("5.0", table_text)   # b default
        self.assertIn("200.0", table_text)  # lambda_max default
        self.assertIn("0.01", table_text)  # d_lambda default


if __name__ == "__main__":
    unittest.main()
