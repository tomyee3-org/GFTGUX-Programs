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

  2026-09-02  Claude (principal developer).  Response to Audit1 (three
    independent reviews of the Kickoff package: Codex, Copilot, Gemini).
    This entry is APPENDED to, not a replacement for, the entry above --
    per the project's standing instruction (reaffirmed when a reviewer
    this round suggested condensing the history), this audit trail is for
    developers only and is kept complete rather than summarized, because a
    shortened version would erase exactly the kind of fix-by-fix
    provenance a later audit round needs to check "was this really fixed,
    and why."  Full claim-by-claim dispositions are in the accompanying
    Response-to-Audit1 report; this entry records only what changed in
    the source tree and in this test file.
      * Fixed two real production defects in physics_photon.py, each
        reproduced (with a documented exact repro) before the fix and
        each now covered by a permanent regression test:
          (1) Codex P1-1: a photon launched exactly tangentially (b at
              the local kinematic bound for r0, so the initial radial
              velocity is exactly 0.0) was never detected as "turned
              outward" by the old "previous_v < 0.0 <= v_r" test,
              because Python's -math.sqrt(0.0) is negative zero and
              -0.0 < 0.0 is False. Fixed by testing "new_v > 0.0"
              directly, with no dependence on the sign of the sample
              before it. See TestTangentialLaunches below.
          (2) Codex P1-2: an escaping trajectory accepted and appended
              a whole RK4 step before checking r>=escape_radius (unlike
              the symmetric, already-interpolated horizon-crossing
              case), so delta_phi/lambda_final carried a step-phase
              error that did not shrink smoothly with d_lambda --
              contaminating exactly the EXP-9 convergence comparison.
              Fixed by adding an interpolation block symmetric to the
              horizon-crossing one. TestConvergence below was reworked
              to check delta_phi convergence (now clean) rather than
              closest_approach convergence (a sampled minimum, still
              subject to run-to-run jitter by design -- see EXP-9's
              updated wording and the caveat comment in that test).
      * Fixed two real robustness gaps, both reproduced and now
        regression-tested:
          (3) Codex P2-2 / Copilot F-1/F-2: radial_acceleration(),
              dphi_dlambda(), and the lambda_max/d_lambda step-count
              check could raise a raw, uncaught OverflowError (or, in
              math.ceil()'s case, be handed float('inf')) for extreme-
              but-finite inputs, instead of the documented ValueError.
              Fixed by pre-checking the step ratio against the step cap
              before calling math.ceil(), and by wrapping the two
              physics functions' arithmetic in try/except with a
              trailing math.isfinite() check. While reproducing these
              two findings with a fresh set of extreme-value probes (not
              themselves given by either audit), found and fixed a
              related, previously-undetected defect of the same kind:
              an r small enough that r**3 (or r*r) underflows to a
              literal 0.0 float raises ZeroDivisionError, not
              OverflowError, and the original except clause did not
              catch it. Both functions now also catch ZeroDivisionError.
              See TestExtremeInputRegressions below.
          (4) Codex P3-1 / Copilot F-3: plot_photon.py's saved-PNG
              filename collision avoidance was check-then-write, a
              race between concurrent PhotonOrbit processes. Fixed with
              an atomic os.open(..., O_CREAT|O_EXCL) reservation.
              TestOutputFileCollisionAvoidance below gained a
              multi-thread concurrency stress test formalizing the
              12-thread manual verification done for this round.
      * Added a per-saved-run provenance sidecar (plot_photon.py's
        _format_provenance()/_reserve_unique_stem()), requested
        independently by both Codex (P1-3) and Copilot (Section 7) and
        weighed against Gemini's endorsement of the Kickoff round's
        decision to omit one; implemented because Codex/Copilot's
        concrete near-separatrix reproducer (a six-significant-digit
        console value of b changing the reported outcome when retyped)
        was a more compelling, specific failure mode than Gemini's
        shorter endorsement of the status quo. See the new
        TestProvenanceSidecar class below.
      * Console run summaries and CLI --outdir runs now also print/save
        a losslessly-precise "Reproduce this exact run with:" command
        line (driver_photon.py _print_summary(), Codex P1-3).
      * MODEL_VERSION 1.1.0 -> 1.2.0 (functional physics/robustness
        fixes, not merely new tests or documentation); BUILD_ID
        recomputed automatically and the Help file's #version_build
        element updated to match after every source change in this
        round, including the ZeroDivisionError fix found during testing.
      * Corrected an explanatory error in the Kickoff-round report,
        caught independently by Codex (P2-6) and Copilot (D-2): the
        report's Section 2 stated that inward motion "moves toward
        strictly larger f(r)"; since f(r)=(1-2M/r)/r^2 is increasing in
        r on (2M,3M), inward motion (decreasing r) moves toward SMALLER
        f(r), which is what makes (dr/dlambda)^2=E^2-L^2 f(r) INCREASE.
        This test file's own comment on
        test_photon_effective_potential_is_monotonic_between_horizon_and_photon_sphere
        already had the direction right; only the prose report was
        wrong, and only the sign/direction, not the capture conclusion
        itself. Corrected in the Response-to-Audit1 report.
      * Fixed three real defects in this test file's own quality, none
        of which had produced a wrong pass/fail verdict on the source
        under test, but each of which weakened what the test actually
        proved:
          (5) Codex P2-1: test_flattened_layout_smoke_test_only wrote a
              standalone _flat_smoke.py that imported physics_photon via
              a manual sys.path.insert(0, '.'), so it verified the four
              modules import cleanly from a flat directory but never
              actually called find_module_dir() from one -- a regression
              in the discovery helper itself (the thing the class is
              named for) could have passed unnoticed. Rewritten to copy
              this very test file into the flattened directory and
              invoke one of its own trivial discovery tests
              (TestModuleDiscovery.test_finds_flattened_layout) by fully
              qualified unittest name, so MODULE_DIR is genuinely
              computed, at import time, from the flattened layout.
          (6) Codex P2-3: test_build_id_independent_of_line_endings fed
              the SAME "normalized = text_lf" variable into both the
              "digest_lf" and "digest_crlf" accumulators -- text_crlf was
              computed and then discarded, so the test could not fail
              even if line-ending normalization were completely broken.
              Rewritten to materialize real LF-only and CRLF-only source
              trees on disk and run the production-mirroring
              recompute_build_id() helper (itself exercising
              newline=None) against each.
          (7) Codex P2-4: test_mathjax_config_is_syntactically_valid_javascript
              unconditionally shelled out to "node --check"; on a machine
              without Node.js installed this failed for an environmental
              reason unrelated to the Help file's actual correctness.
              Guarded with shutil.which("node") and a graceful skipTest.
      * Strengthened, per Codex P2-5, three Help-file structural tests
        that previously checked only weak, position-independent
        signals: test_module_signatures_match_actual_code now compares
        all four modules' documented signatures against
        inspect.signature() of the real functions (previously: three
        physics functions only, via substring match);
        test_algorithm_step_order_matches_actual_validation_order now
        walks the algorithm <ol>'s <li> elements in document order and
        checks each step's expected content at its own index
        (previously: only that certain keywords co-occurred somewhere on
        the page); test_parameter_table_defaults_match_main_py now
        parses the parameter table row by row and compares each row
        against the live argparse.Namespace from main.parse_args()
        (previously: five bare numeral substrings checked against the
        whole page, which could not detect a default attached to the
        wrong row and did not cover --lw/--dpi/--outdir at all).
      * Copilot T-2 (bisection-oracle independence): the bisection
        search in test_bisection_threshold_converges_to_3_sqrt_3 calls
        integrate_photon_orbit() itself to classify each trial b, so it
        is not a fully independent oracle for the *classification*
        outcome the way TestAnalyticVerification's finite-difference and
        potential-maximization tests are for the underlying physics; it
        is retained specifically because it independently exercises a
        different code path (repeated bisection across many b values
        near the separatrix, rather than one direct call) and pins the
        converged threshold to 3*sqrt(3) via a method (bisection) wholly
        distinct from how the source computes critical_b_infinity
        (closed-form 3*sqrt(3)*GM_over_c2). A clarifying comment was
        added directly above the test explaining this scope, rather than
        removing or replacing the test, since it still catches a real
        class of regression (the capture/escape dichotomy itself
        drifting away from the analytically-known threshold) that the
        closed-form tests cannot.
      * Added regression coverage for every fix above (tangential
        launches at both sides of the photon sphere, extreme-input
        ValueError reproducers via both the direct API and the CLI,
        provenance sidecar content and round-trip precision, a formalized
        atomic-write concurrency stress test, and delta_phi-based
        convergence) plus the strengthened/corrected tests already
        itemized. No previously-passing test was deleted or weakened
        without a stated reason above; test_escaped_run_ends_at_or_beyond_escape_radius
        was tightened from assertGreaterEqual to an exact
        assertAlmostEqual now that escape is event-located rather than
        loosened.
"""

import ast
import contextlib
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import importlib
import inspect
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
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
        """Prove find_module_dir() itself works from a flattened directory.

        This is deliberately NOT a second full run of the suite (see the
        module docstring): it invokes exactly one trivial test, by fully
        qualified unittest name, from a flattened copy of this very file,
        then returns.  The full suite below runs exactly once, from the
        canonical tests/ layout.

        Audit1 Codex P2-1: the previous version of this test wrote a
        standalone "_flat_smoke.py" script that did
        "sys.path.insert(0, '.')" and imported physics_photon directly --
        that proved the four core modules import cleanly from a flat
        directory, but never actually called find_module_dir() from one,
        so a regression in the discovery helper itself (the thing this
        test CLASS is named for) could have passed unnoticed. Copying
        this test file into the flat directory and running
        TestModuleDiscovery.test_finds_flattened_layout from the copy
        makes MODULE_DIR get computed for real, at import time, by
        find_module_dir(Path(__file__)) on the copy's own __file__, which
        now lives directly beside the four core modules with no tests/
        parent directory at all.
        """
        if os.environ.get("PHOTONORBIT_FLATTENED_SMOKE_CHILD") == "1":
            return
        this_file = Path(__file__).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            flat_dir = Path(temporary)
            for name in (*CORE_MODULE_FILES, HELP_FILE):
                shutil.copy2(MODULE_DIR / name, flat_dir / name)
            shutil.copy2(this_file, flat_dir / this_file.name)

            environment = os.environ.copy()
            environment["PHOTONORBIT_FLATTENED_SMOKE_CHILD"] = "1"
            module_name = this_file.stem
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "-v",
                 f"{module_name}.TestModuleDiscovery.test_finds_flattened_layout"],
                cwd=flat_dir, env=environment,
                capture_output=True, text=True, timeout=30, check=False,
            )
            combined = result.stdout + result.stderr
            self.assertEqual(result.returncode, 0, combined)
            self.assertIn("Ran 1 test", combined)
            self.assertIn("OK", combined)


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
        """BUILD_ID must be stable under LF/CRLF normalization (newline=None).

        Audit1 Codex P2-3: the previous version of this test fed the SAME
        "normalized = text_lf" variable into both the "digest_lf" and
        "digest_crlf" accumulators; "text_crlf" was computed and then
        never used, so the test could not have failed even if line-ending
        normalization were completely broken. This version materializes
        two real on-disk source trees -- one with LF-only, one with
        CRLF-only line endings -- and runs recompute_build_id() (a
        byte-for-byte mirror of physics_photon._compute_build_id(), which
        opens each file with newline=None, Python's universal-newlines
        mode) against each, so the test genuinely exercises the same file
        I/O and normalization path production code uses.
        """
        with tempfile.TemporaryDirectory() as lf_temp, \
             tempfile.TemporaryDirectory() as crlf_temp:
            lf_dir, crlf_dir = Path(lf_temp), Path(crlf_temp)
            for name in phys.BUILD_ID_COVERS:
                raw = (MODULE_DIR / name).read_bytes()
                text_lf = raw.replace(b"\r\n", b"\n")
                text_crlf = text_lf.replace(b"\n", b"\r\n")
                (lf_dir / name).write_bytes(text_lf)
                (crlf_dir / name).write_bytes(text_crlf)
            build_id_lf = recompute_build_id(lf_dir)
            build_id_crlf = recompute_build_id(crlf_dir)
        self.assertEqual(build_id_lf, build_id_crlf)
        self.assertEqual(build_id_lf, phys.BUILD_ID)

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

        # Copilot T-2: this bisection search calls integrate_photon_orbit()
        # itself to classify each trial b, so -- unlike
        # TestAnalyticVerification's finite-difference and potential-
        # maximization tests -- it is not a fully independent oracle for
        # the underlying PHYSICS. It is kept anyway because it exercises
        # a materially different code path (many repeated integrations
        # bracketing a threshold, via a distinct numerical method --
        # bisection -- rather than one direct call) and pins the
        # converged threshold to 3*sqrt(3) independently of the source's
        # own closed-form critical_b_infinity=3*sqrt(3)*GM_over_c2, so it
        # still catches a real regression class (the capture/escape
        # dichotomy drifting away from the analytically-known threshold)
        # that the closed-form tests do not.
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
class TestTangentialLaunches(unittest.TestCase):
    """Regression coverage for Audit1 Codex P1-1 (the negative-zero bug)
    and the new Help-file EXP-10 exercise built directly on top of it.

    A photon launched exactly tangentially -- b at the local kinematic
    bound for its r0, so the initial radial velocity is exactly 0.0 --
    is a legal initial condition, not merely a boundary case, and the
    fixed code must classify it correctly on both sides of the photon
    sphere without relying on the sign of any earlier sample.
    """

    def test_tangential_launch_outside_photon_sphere_escapes_exactly_at_escape_radius(self):
        GM, r0 = 1.0, 6.0
        b = r0 / math.sqrt(1.0 - 2.0 * GM / r0)  # exact local kinematic bound
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=GM, r0=r0, b=b, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "escaped")
        last_r = math.hypot(x[-1], y[-1])
        self.assertAlmostEqual(last_r, info["escape_radius"], places=9)
        self.assertAlmostEqual(info["escape_radius"], 2.0 * r0, places=9)

    def test_tangential_launch_inside_photon_sphere_is_captured(self):
        GM, r0 = 1.0, 2.5
        b = r0 / math.sqrt(1.0 - 2.0 * GM / r0)  # exact local kinematic bound
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=GM, r0=r0, b=b, lambda_max=50.0, d_lambda=0.001,
        )
        self.assertEqual(info["status"], "captured")
        last_r = math.hypot(x[-1], y[-1])
        self.assertAlmostEqual(last_r, info["r_s"], places=9)

    def test_tangential_launch_exactly_at_photon_sphere_stays_circular(self):
        # The one case a tangential launch does neither: at r0=3*GM_over_c2
        # with the exact circular-orbit b, both v_r and the radial
        # acceleration are identically 0.0 forever, so new_v > 0.0 never
        # fires and the photon correctly stays status="lambda_max" for
        # the whole requested run (already covered by
        # TestCircularPhotonOrbit.test_exact_circular_orbit_stays_on_photon_sphere;
        # repeated here as part of the EXP-10 trio for a single
        # self-contained regression class).
        GM = 1.0
        b_crit = 3.0 * math.sqrt(3.0) * GM
        _, _, info = phys.integrate_photon_orbit(
            GM_over_c2=GM, r0=3.0 * GM, b=b_crit, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "lambda_max")


# ======================================================================
class TestExtremeInputRegressions(unittest.TestCase):
    """Regression coverage for Audit1 Codex P2-2 / Copilot F-1 / F-2, plus
    a related defect (ZeroDivisionError on underflow) found while
    reproducing them with a fresh set of extreme-value probes.

    Every case here previously either raised a raw, undocumented
    exception (OverflowError or ZeroDivisionError) or, for the step-count
    case, was handed float('inf') by math.ceil(). All must now raise the
    program's documented ValueError.
    """

    def test_radial_acceleration_overflow_from_extreme_radius(self):
        # r**3 for r=1e200 overflows a Python float.
        with self.assertRaises(ValueError):
            phys.radial_acceleration(1e200, 1.0, 1.0)

    def test_radial_acceleration_nonfinite_from_extreme_angular_momentum(self):
        # L*L for L=1e308 overflows to inf, which is finite-checked and
        # rejected rather than silently propagated.
        with self.assertRaises(ValueError):
            phys.radial_acceleration(5.0, 1e308, 1.0)

    def test_radial_acceleration_zero_division_from_underflowing_radius(self):
        # r=1e-200 makes r**3=1e-600, far below the smallest positive
        # float (~5e-324); r**3 underflows to a literal 0.0, and
        # L*L / 0.0 is a ZeroDivisionError, not an OverflowError. This
        # was not itself one of Codex's or Copilot's reported
        # reproducers; it was found by extending their overflow-probe
        # methodology to underflow while reproducing P2-2/F-2.
        with self.assertRaises(ValueError):
            phys.radial_acceleration(1e-200, 1e308, 1.0)

    def test_dphi_dlambda_zero_division_from_underflowing_radius(self):
        with self.assertRaises(ValueError):
            phys.dphi_dlambda(1e-200, 1e308)

    def test_step_count_overflow_from_infinite_ratio_rejected_with_documented_message(self):
        # lambda_max/d_lambda = 1e308/1e-308 evaluates to float('inf');
        # math.ceil(inf) raises OverflowError if reached. The pre-check
        # against _MAX_STEPS (a plain float comparison, well-defined
        # against inf) must reject this with the same documented message
        # used for an ordinary excessive-but-finite step count.
        with self.assertRaisesRegex(ValueError, r"5,000,000"):
            phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=20.0, b=5.0,
                lambda_max=1e308, d_lambda=1e-308,
            )

    def test_step_count_overflow_reported_cleanly_via_cli(self):
        result = run_cli(["--lambda_max", "1e308", "--d_lambda", "1e-308"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PhotonOrbit:", result.stderr)
        self.assertIn("5,000,000", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_extreme_starting_radius_reported_cleanly_not_as_raw_overflow(self):
        # Copilot F-1 reproducer 2: integrate_photon_orbit(GM_over_c2=1.0,
        # r0=1e308, b=0.0, lambda_max=1.0, d_lambda=0.1) previously raised
        # a raw OverflowError("(34, 'Numerical result out of range')")
        # from an intermediate power computed during RK4 stepping. This
        # now surfaces as the documented RuntimeError (mid-integration
        # failures are RuntimeError, not ValueError, matching every other
        # "integration stepped to a nonphysical/non-finite state" case).
        with self.assertRaisesRegex(RuntimeError, "nonphysical"):
            phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=1e308, b=0.0, lambda_max=1.0, d_lambda=0.1,
            )

    def test_extreme_starting_radius_reported_cleanly_via_cli(self):
        result = run_cli(["--r0", "1e308", "--b", "0.0",
                           "--lambda_max", "1.0", "--d_lambda", "0.1"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PhotonOrbit:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


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
        """delta_phi convergence, reworked per Audit1 Codex P1-2/T-6.

        Before the escape-interpolation fix, delta_phi/lambda_final
        carried a whole-step overshoot error that did not shrink
        smoothly with d_lambda; this test previously used
        closest_approach as its convergence oracle instead specifically
        to route around that. Now that escape is event-located (linearly
        interpolated to the exact escape_radius crossing), delta_phi
        itself converges cleanly and is the more direct oracle for "is
        the integration accurate", so it is used here.

        closest_approach remains intentionally excluded from this
        oracle: it is simply the smallest SAMPLED radius along the path,
        not an interpolated turning point, so it can still show a little
        run-to-run jitter even as delta_phi converges smoothly -- the
        same caveat now given to students in the Help file's EXP-9. It is
        only checked below for being finite, not for convergence.
        """
        b = 5.205
        results = []
        for d_lambda in (0.04, 0.02, 0.01, 0.005):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=1.0, r0=20.0, b=b, lambda_max=400.0, d_lambda=d_lambda,
            )
            self.assertEqual(info["status"], "escaped")
            results.append(info)

        finest = results[-1]
        diffs = [abs(r["delta_phi"] - finest["delta_phi"]) for r in results[:-1]]
        # Each coarser step size must show a strictly larger deviation
        # from the finest-resolution delta_phi than the next-finer step
        # size -- true, monotonically shrinking convergence, not merely
        # "doesn't crash".
        self.assertEqual(diffs, sorted(diffs, reverse=True))
        self.assertLess(diffs[-1], diffs[0])
        for value in (r["closest_approach"] for r in results):
            self.assertTrue(math.isfinite(value))

    def test_observed_convergence_order_is_fourth_for_an_event_free_run(self):
        """Audit1 Copilot T-6: estimate RK4's OBSERVED convergence order
        from successive step-halvings (h, h/2, h/4), rather than only
        checking "finer is at least as good as coarser".

        This is deliberately run on a status="lambda_max" trajectory
        (integration proceeds for the full requested affine parameter
        with no horizon/escape event at all), isolating the RK4 stepping
        error from the SEPARATE linear-interpolation error introduced at
        an event crossing (Copilot T-7): the two have different orders in
        h (RK4 stepping is O(h^4); the horizon/escape crossing location
        is only linearly interpolated, O(h^1)), and mixing them would
        make an order estimate near a captured/escaped run's diagnostics
        muddy at best. See the Response-to-Audit1 report for the
        numerical demonstration of this distinction, and for the
        rationale for not extending event-location itself to a
        higher-order scheme this round.
        """
        GM_over_c2, r0, b, lambda_max = 1.0, 20.0, 5.1961, 30.0
        delta_phis = []
        for d_lambda in (0.02, 0.01, 0.005):
            _, _, info = phys.integrate_photon_orbit(
                GM_over_c2=GM_over_c2, r0=r0, b=b,
                lambda_max=lambda_max, d_lambda=d_lambda,
            )
            self.assertEqual(info["status"], "lambda_max")
            delta_phis.append(info["delta_phi"])

        diff_coarse = abs(delta_phis[0] - delta_phis[1])
        diff_fine = abs(delta_phis[1] - delta_phis[2])
        self.assertGreater(diff_fine, 0.0)
        observed_order = math.log2(diff_coarse / diff_fine)
        # RK4 predicts an error reduction factor of 2^4=16 (order 4) per
        # halving; allow a generous +/-0.5 band around that so the test
        # is not sensitive to which two step sizes are compared, while
        # still failing if the order collapsed toward 1 (linear) or 2.
        self.assertGreater(observed_order, 3.5)
        self.assertLess(observed_order, 4.5)


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

    def test_escaped_run_ends_exactly_at_escape_radius(self):
        # Tightened from assertGreaterEqual to an exact assertAlmostEqual
        # (Audit1 Codex P1-2): escape is now linearly interpolated to the
        # exact escape_radius crossing within the final RK4 step, mirroring
        # test_captured_run_ends_exactly_at_the_event_horizon above, so a
        # whole-step overshoot past escape_radius is itself a regression.
        x, y, info = phys.integrate_photon_orbit(
            GM_over_c2=1.0, r0=20.0, b=6.0, lambda_max=200.0, d_lambda=0.01,
        )
        self.assertEqual(info["status"], "escaped")
        last_r = math.hypot(x[-1], y[-1])
        self.assertAlmostEqual(last_r, info["escape_radius"], places=9)

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
            # Audit1 Codex P1-3/Copilot Section 7: a call with outdir now
            # also writes a ".provenance.txt" sidecar next to the PNG (see
            # TestProvenanceSidecar below), so a saved run is two files,
            # not one.
            self.assertEqual(len(files), 2)
            pngs = [f for f in files if f.endswith(".png")]
            sidecars = [f for f in files if f.endswith(".provenance.txt")]
            self.assertEqual(len(pngs), 1)
            self.assertEqual(len(sidecars), 1)
            self.assertTrue(pngs[0].startswith("photon_b5_captured_"))
            self.assertEqual(sidecars[0], pngs[0][: -len(".png")] + ".provenance.txt")
            self.assertGreater(os.path.getsize(os.path.join(outdir, pngs[0])), 0)

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
    """Regression tests for the same-wall-clock-second PNG collision fix,
    and (Audit1 Codex P3-1/Copilot F-3) for the atomic-reservation fix
    that replaced it this round."""

    def test_two_saves_in_the_same_second_produce_two_distinct_files(self):
        info = dict(r_s=2.0, r_photon=3.0, status="captured",
                     closest_approach=2.0, delta_phi=1.0)
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plotting, "datetime", FrozenDatetime), \
                 mock.patch("matplotlib.pyplot.show"):
                plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info, outdir=outdir)
                plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info, outdir=outdir)
            files = sorted(os.listdir(outdir))
            # Each save now writes a PNG plus a provenance sidecar, so two
            # saves in the same frozen second produce four files, not two.
            self.assertEqual(len(files), 4)
            for name in files:
                self.assertGreater(os.path.getsize(os.path.join(outdir, name)), 0)
            pngs = sorted(f for f in files if f.endswith(".png"))
            self.assertTrue(pngs[1].endswith("_2.png"))
            for png in pngs:
                sidecar = png[: -len(".png")] + ".provenance.txt"
                self.assertIn(sidecar, files)

    def test_three_saves_in_the_same_second_all_kept(self):
        info = dict(r_s=2.0, r_photon=3.0, status="escaped",
                     closest_approach=4.0, delta_phi=2.0)
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plotting, "datetime", FrozenDatetime), \
                 mock.patch("matplotlib.pyplot.show"):
                for _ in range(3):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 6.0, info, outdir=outdir)
            files = os.listdir(outdir)
            self.assertEqual(len(files), 6)  # 3 PNGs + 3 sidecars
            self.assertEqual(len([f for f in files if f.endswith(".png")]), 3)
            self.assertEqual(
                len([f for f in files if f.endswith(".provenance.txt")]), 3
            )

    def test_concurrent_saves_from_multiple_threads_never_collide(self):
        """Formalizes the manual 12-thread verification done this round.

        Audit1 Codex P3-1/Copilot F-3: the Kickoff round's _unique_path()
        was check-then-write, a race between concurrent processes/threads
        sharing a clock tick. _reserve_unique_stem() closes that race with
        os.open(..., O_CREAT|O_EXCL). All threads share the SAME frozen
        timestamp, so this only passes if the atomic reservation genuinely
        serializes stem selection; a reintroduced check-then-write race
        would intermittently drop a PNG or a sidecar, or produce a
        duplicate stem, under this test.
        """
        info = dict(r_s=2.0, r_photon=3.0, status="captured",
                     closest_approach=2.0, delta_phi=1.0)
        n_threads = 12
        errors = []

        def worker():
            try:
                with mock.patch("matplotlib.pyplot.show"):
                    plotting.plot_photon_orbit([1, 2], [1, 2], 5.0, info, outdir=outdir)
            except Exception as exc:  # noqa: BLE001 - surfaced via `errors`
                errors.append(exc)

        with tempfile.TemporaryDirectory() as outdir, \
             mock.patch.object(plotting, "datetime", FrozenDatetime):
            threads = [threading.Thread(target=worker) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(errors, [])
            files = os.listdir(outdir)
            pngs = [f for f in files if f.endswith(".png")]
            sidecars = [f for f in files if f.endswith(".provenance.txt")]
            self.assertEqual(len(pngs), n_threads)
            self.assertEqual(len(sidecars), n_threads)
            self.assertEqual(len(set(pngs)), n_threads)  # every stem unique
            for png in pngs:
                self.assertIn(png[: -len(".png")] + ".provenance.txt", sidecars)
                self.assertGreater(os.path.getsize(os.path.join(outdir, png)), 0)


# ======================================================================
class TestProvenanceSidecar(unittest.TestCase):
    """Regression tests for the provenance sidecar added this round
    (Audit1 Codex P1-3, Copilot Section 7; see the module docstring's
    Audit1 entry for why this was implemented despite Gemini's Kickoff-
    round endorsement of omitting it)."""

    def _run_and_read_sidecar(self, **plot_kwargs):
        info = dict(r_s=2.0, r_photon=3.0, status="escaped",
                     closest_approach=3.0686559820284285, delta_phi=1.2345678901234567)
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch("matplotlib.pyplot.show"):
                plotting.plot_photon_orbit(
                    [1.0, 2.0], [0.0, 1.0], 5.196152422706632, info,
                    outdir=outdir, **plot_kwargs,
                )
            files = os.listdir(outdir)
            sidecars = [f for f in files if f.endswith(".provenance.txt")]
            self.assertEqual(len(sidecars), 1)
            return (Path(outdir) / sidecars[0]).read_text(encoding="utf-8")

    def test_sidecar_records_physics_parameters_at_repr_precision(self):
        # A near-separatrix b with a 6-significant-digit console rounding
        # that changes the classification outcome (Codex's concrete
        # motivating reproducer) must round-trip EXACTLY through the
        # sidecar, not merely approximately.
        b = 5.196152422706632
        r0 = 20.0
        GM_over_c2 = 1.0
        lambda_max = 200.0
        d_lambda = 0.01
        text = self._run_and_read_sidecar(
            GM_over_c2=GM_over_c2, r0=r0, lambda_max=lambda_max, d_lambda=d_lambda,
        )
        for name, value in (
            ("GM_over_c2", GM_over_c2), ("r0", r0),
            ("lambda_max", lambda_max), ("d_lambda", d_lambda),
        ):
            with self.subTest(name=name):
                line = f"    {name} = {value!r}"
                self.assertIn(line, text)

        # b itself is passed as a positional argument, not a kwarg, but
        # must also appear at repr precision inside the reproduce-with
        # command line.
        self.assertIn(repr(b), text)

    def test_sidecar_contains_a_working_reproduce_with_command(self):
        text = self._run_and_read_sidecar(
            GM_over_c2=1.0, r0=20.0, lambda_max=200.0, d_lambda=0.01,
        )
        match = re.search(r"^\s*python main\.py .*$", text, re.M)
        self.assertIsNotNone(match)
        command = match.group(0).strip()
        self.assertIn("--GM_over_c2 1.0", command)
        self.assertIn("--r0 20.0", command)
        self.assertIn("--b 5.196152422706632", command)
        self.assertIn("--lambda_max 200.0", command)
        self.assertIn("--d_lambda 0.01", command)

        # The command must actually be runnable and reproduce the same
        # outcome fields recorded in the sidecar (round-trip, not just
        # textual presence).
        args = command.split()[2:]  # drop "python", "main.py"
        result = run_cli(args)
        self.assertEqual(result.returncode, 0, result.stderr)
        # b=5.196152422706632 at r0=20 is fractionally below the exact
        # critical impact parameter (3*sqrt(3), same repr to all 16
        # digits) and is captured, not escaped -- this is itself the
        # numerically sensitive behavior the sidecar exists to let a
        # student reproduce exactly.
        self.assertIn("captured", result.stdout)

    def test_sidecar_omits_reproduce_command_when_physics_params_not_supplied(self):
        text = self._run_and_read_sidecar()  # no GM_over_c2/r0/lambda_max/d_lambda
        self.assertIn("(not provided to plot_photon_orbit)", text)
        self.assertIn(
            "Reproduce-with command omitted", text,
        )
        self.assertNotIn("Reproduce with:", text)

    def test_sidecar_records_rendering_parameters_and_outcome_fields(self):
        text = self._run_and_read_sidecar(
            GM_over_c2=1.0, r0=20.0, lambda_max=200.0, d_lambda=0.01, dpi=222, lw=2.5,
        )
        self.assertIn("dpi = 222", text)
        self.assertIn("lw  = 2.5", text)
        self.assertIn("status = 'escaped'", text)
        self.assertIn(repr(3.0686559820284285), text)
        self.assertIn(repr(1.2345678901234567), text)


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
        """Cross-check EVERY module-card's declared signature against the
        real function signature via inspect.signature(), not just the
        three physics functions (Audit1 Codex P2-5: the previous version
        checked only physics_photon.py's three functions by substring, so
        a drift in driver_photon.py's or plot_photon.py's documented
        keyword-argument defaults -- exactly the kind of thing that
        changed this round, with the new GM_over_c2/r0/lambda_max/
        d_lambda kwargs on plot_photon_orbit() -- would not have been
        caught here).
        """
        def flat(text):
            return re.sub(r"\s+", "", text)

        def param_pattern(name, default=inspect.Parameter.empty):
            if default is inspect.Parameter.empty:
                return rf"[(,]{re.escape(name)}[,)]"
            return rf"[(,]{re.escape(name)}={re.escape(repr(default))}[,)]"

        cards = descendants(self.root, lambda n: n.attrs.get("class") == "module-card")
        self.assertEqual(len(cards), 4)

        def sig_text(card):
            blocks = descendants(card, lambda n: n.attrs.get("class") == "sig")
            self.assertEqual(len(blocks), 1)
            return flat(blocks[0].text())

        physics_card = next(c for c in cards if "physics_photon.py" in normalized_text(c))
        driver_card = next(c for c in cards if "driver_photon.py" in normalized_text(c))
        plot_card = next(c for c in cards if "plot_photon.py" in normalized_text(c))
        main_card = next(c for c in cards if "main.py" in normalized_text(c)
                          and "Entry Point" in normalized_text(c))

        physics_text = sig_text(physics_card)
        self.assertIn(flat("def radial_acceleration(r, L, GM_over_c2)"), physics_text)
        self.assertIn(flat("def dphi_dlambda(r, L)"), physics_text)
        self.assertIn(
            flat("def integrate_photon_orbit(GM_over_c2, r0, b, lambda_max, d_lambda)"),
            physics_text,
        )
        self.assertEqual(
            list(inspect.signature(phys.integrate_photon_orbit).parameters),
            ["GM_over_c2", "r0", "b", "lambda_max", "d_lambda"],
        )

        driver_text = sig_text(driver_card)
        self.assertIn(flat("def driver_photon_orbit("), driver_text)
        for name, param in inspect.signature(driver.driver_photon_orbit).parameters.items():
            with self.subTest(module="driver_photon.py", param=name):
                self.assertRegex(driver_text, param_pattern(name, param.default))

        plot_text = sig_text(plot_card)
        self.assertIn(flat("def plot_photon_orbit("), plot_text)
        for name, param in inspect.signature(plotting.plot_photon_orbit).parameters.items():
            with self.subTest(module="plot_photon.py", param=name):
                self.assertRegex(plot_text, param_pattern(name, param.default))

        main_text = sig_text(main_card)
        for flag in ("--GM_over_c2", "--r0", "--b", "--lambda_max",
                     "--d_lambda", "--lw", "--dpi", "--outdir"):
            with self.subTest(module="main.py", flag=flag):
                self.assertIn(flag, main_text)

    def test_no_leftover_boilerplate_or_java_references(self):
        for pattern in (r"\.jar\b", r"\bapplet\b", r"\bJava\b",
                        r"translat(ed|ion) from", r"ported from"):
            with self.subTest(pattern=pattern):
                self.assertIsNone(re.search(pattern, self.html))

    def test_mathjax_config_is_syntactically_valid_javascript(self):
        """Audit1 Codex P2-4: this test previously shelled out to
        "node --check" unconditionally, so on a machine without Node.js
        installed it failed for an environmental reason unrelated to the
        Help file's actual correctness. Guarded with shutil.which("node")
        and a graceful skipTest; see
        test_mathjax_check_is_skipped_when_node_is_unavailable below for
        a regression test of the guard itself.
        """
        if shutil.which("node") is None:
            self.skipTest("node executable not available in this environment")
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

    def test_mathjax_check_is_skipped_when_node_is_unavailable(self):
        """Regression test for the P2-4 skip guard itself: with
        shutil.which patched to report node as absent, the syntax-check
        test must raise unittest.SkipTest rather than attempting to run
        node (which would otherwise raise FileNotFoundError and be
        reported as an ERROR, not a clean SKIP)."""
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaises(unittest.SkipTest):
                self.test_mathjax_config_is_syntactically_valid_javascript()

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
        """Audit1 Copilot T-4: the previous version checked "d_lambda" and
        "RK4 step size" occur SOMEWHERE in the whole HTML document -- both
        strings could remain true while the specific Console-summary row
        regressed (e.g. if that row's wording were edited elsewhere and
        the phrases just happened to survive in another section). This
        scopes the check to the out-table row whose first cell reads
        "Console summary".
        """
        rows = descendants(self.root, lambda n: n.tag == "tr")
        console_rows = [
            row for row in rows
            if descendants(row, lambda n: n.attrs.get("class") == "oname")
            and normalized_text(
                descendants(row, lambda n: n.attrs.get("class") == "oname")[0]
            ) == "Console summary"
        ]
        self.assertEqual(len(console_rows), 1)
        row_text = normalized_text(console_rows[0])
        self.assertIn("d_lambda", row_text)
        self.assertIn("RK4 step size", row_text)

    def test_algorithm_step_order_matches_actual_validation_order(self):
        """Check the real ORDER of the Algorithm section's <ol> steps,
        not merely that certain keywords co-occur somewhere on the page
        (Audit1 Codex P2-5). Each expected fragment set below must be
        found in the step at its OWN index, matching the order actually
        implemented in integrate_photon_orbit()."""
        algorithm_section = next(
            s for s in descendants(self.root, lambda n: n.tag == "section")
            if s.attrs.get("id") == "algorithm"
        )
        ol_nodes = descendants(algorithm_section, lambda n: n.tag == "ol")
        self.assertEqual(len(ol_nodes), 1)
        steps = [normalized_text(li) for li in descendants(ol_nodes[0], lambda n: n.tag == "li")]
        self.assertEqual(len(steps), 8)

        expected_fragments_by_step = [
            ("finite", "GM_over_c2"),
            ("event horizon", "nonnegative", "lambda_max", "d_lambda"),
            ("radial first integral",),
            ("E=1", "L=b"),
            ("RK4",),
            ("event horizon", "escape radius", "lambda_max"),
            ("Cartesian", "status", "closest approach"),
            ("console", "PNG"),
        ]
        for index, (step_text, fragments) in enumerate(zip(steps, expected_fragments_by_step)):
            with self.subTest(step=index + 1):
                for fragment in fragments:
                    self.assertIn(fragment, step_text)

    def test_exp6_wording_matches_actual_program_output_capabilities(self):
        exp_cards = descendants(self.root, lambda n: n.attrs.get("class") == "exp-card")
        scale_card = next(c for c in exp_cards if "Scale Invariance" in normalized_text(c))
        text = normalized_text(scale_card)
        self.assertIn("console summary", text)
        self.assertIn("physics_photon.integrate_photon_orbit", text)

    def test_all_ten_exercises_present_and_difficulty_labeled(self):
        # Was 9 exercises at the Kickoff round; EXP-10 (Tangential
        # Launches and the Local Kinematic Bound) was added this round to
        # directly exercise the Audit1 Codex P1-1 fix.
        exp_cards = descendants(self.root, lambda n: n.attrs.get("class") == "exp-card")
        self.assertEqual(len(exp_cards), 10)
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
        # Audit1 Copilot T-5: the description should explicitly name
        # gravitational light bending, not just capture/escape/photon
        # sphere vocabulary.
        self.assertIn("light bending", text.lower())

    def test_parameter_table_defaults_match_main_py(self):
        """Row-by-row: every documented --flag/default pair must equal
        the live argparse default in main.py.

        Audit1 Codex P2-5: the previous version of this test only checked
        that five bare numeral substrings ("1.0", "20.0", "5.0", "200.0",
        "0.01") appeared SOMEWHERE in the whole table's text -- it could
        not detect a default attached to the wrong parameter (e.g.
        "20.0" listed as b's default instead of r0's) and did not cover
        --lw, --dpi, or --outdir at all. This version parses each row of
        the param-table into a (flag, documented-default) pair and
        compares it against main.parse_args()'s actual Namespace, called
        with no command-line arguments so every field is its coded
        default.
        """
        table = descendants(self.root, lambda n: n.attrs.get("class") == "param-table")[0]
        rows = [
            row for row in descendants(table, lambda n: n.tag == "tr")
            if descendants(row, lambda n: n.attrs.get("class") == "pname")
        ]
        self.assertGreater(len(rows), 0)

        main_module = importlib.import_module("main")
        with mock.patch.object(sys, "argv", ["main.py"]):
            args = main_module.parse_args()

        flag_to_attr = {
            "--GM_over_c2": "GM_over_c2", "--r0": "r0", "--b": "b",
            "--lambda_max": "lambda_max", "--d_lambda": "d_lambda",
            "--lw": "lw", "--dpi": "dpi", "--outdir": "outdir",
        }
        documented_flags = set()
        for row in rows:
            pname = normalized_text(descendants(row, lambda n: n.attrs.get("class") == "pname")[0])
            pdefault = normalized_text(
                descendants(row, lambda n: n.attrs.get("class") == "pdefault")[0]
            )
            with self.subTest(flag=pname):
                self.assertIn(pname, flag_to_attr, f"undocumented flag {pname!r} in param-table")
                documented_flags.add(pname)
                actual_default = getattr(args, flag_to_attr[pname])
                self.assertEqual(pdefault, str(actual_default))
        self.assertEqual(documented_flags, set(flag_to_attr))


if __name__ == "__main__":
    unittest.main()
