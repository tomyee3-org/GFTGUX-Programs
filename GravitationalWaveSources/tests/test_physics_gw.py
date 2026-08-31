"""Regression tests for the GravitationalWaveSources program module.

The discovery helper deliberately supports both the repository layout
(``tests/test_physics_gw.py``) and an upload layout in which this file is
flattened beside the four program modules -- mirroring the convention
established by EXAMPLE_test_physics_cannon.py.

Independent-oracle policy
--------------------------
Wherever practical, the physics tests below do NOT validate a formula by
calling another function in this module that shares the same derivation.
Instead each quantity is cross-checked against at least one of:

  * a hand-derived closed-form expression written from scratch in this file
    (using its own copies of the physical constants, not physics_gw's);
  * an algebraically distinct rearrangement of the same physics (e.g. the
    symmetric-mass-ratio form of chirp mass, or the Kepler/ISCO-radius
    derivation of f_ISCO instead of the closed-form c^3/(6^1.5 pi G M));
  * a genuinely different numerical method (finite-difference
    differentiation of the analytic inspiral-time formula to obtain df/dt,
    rather than reading dfdt() directly); or
  * a physical invariant that must hold regardless of implementation
    (QNM f*M and tau/M scale invariance, distance-independence of the
    chirp timeline, etc).
"""

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import inspect
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

import numpy as np


CORE_MODULE_FILES = (
    "physics_gw.py",
    "driver_gw.py",
    "main.py",
    "plot_gw.py",
)
HELP_FILE = "GravitationalWaveSources.html"


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

import driver_gw as driver  # noqa: E402
import physics_gw as physics  # noqa: E402
import plot_gw as plotting  # noqa: E402


def read_gw_csv(path):
    """Split a GravitationalWaveSources CSV export into its commented
    metadata block (as a dict) and its ordinary CSV rows (header + data).

    Audit2 (Codex P1-1 / Copilot A2-5) added a commented metadata block
    above the header row so an exported file can be identified/attributed
    on its own. Some metadata lines themselves contain commas (e.g. the
    "# columns: t_s,f_hz,A,h,phase_rad -- ..." line), so tests must not
    assume csv.reader's rows[0] is the header row -- "#"-prefixed lines
    are stripped out here before csv parsing, mirroring how a human (or
    any '#'-comment-aware CSV consumer) would read the file.

    Audit3 (Codex P3-4) asked that a duplicate metadata key be explicitly
    handled rather than silently overwritten by a plain dict. This helper
    now raises AssertionError the moment a "# key: ..." line repeats a key
    already seen in this file -- a real writer bug (two meta_lines entries
    for the same field) would otherwise be invisible to every test that
    only inspects the resulting dict, since the second value would just
    silently win.
    """
    import csv as csv_module
    meta = {}
    data_lines = []
    with open(path, newline="", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                stripped = line[1:].strip()
                if ":" in stripped:
                    key, _, value = stripped.partition(":")
                    key = key.strip()
                    assert key not in meta, (
                        f"duplicate CSV metadata key {key!r} in {path!r} -- "
                        f"first value {meta.get(key)!r}, repeated value "
                        f"{value.strip()!r}"
                    )
                    meta[key] = value.strip()
            else:
                data_lines.append(line)
    rows = list(csv_module.reader(data_lines))
    return meta, rows


def recompute_build_id(directory):
    """Independently reproduce the documented normalized source hash."""
    digest = hashlib.sha256()
    for name in physics.BUILD_ID_COVERS:
        with (directory / name).open("r", encoding="utf-8", newline=None) as source:
            content = source.read().encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Independent physical-constant set and from-scratch reference formulas.
#
# These constants are copied literally (not imported) so that a typo or a
# unit error introduced into physics_gw.py's G/c/M_sun/MPC_M would not be
# silently "confirmed" by a test that unknowingly shares the same numbers.
# ---------------------------------------------------------------------------
_G = 6.674_30e-11
_C = 2.997_924_58e8
_MSUN = 1.988_92e30
_MPC = 3.085_677_581_49e22


def ref_chirp_mass_kg(m1_kg, m2_kg):
    """Chirp mass via the standard product/sum closed form."""
    return (m1_kg * m2_kg) ** 0.6 / (m1_kg + m2_kg) ** 0.2


def ref_chirp_mass_via_symmetric_ratio_kg(m1_kg, m2_kg):
    """Chirp mass via Mc = M_total * eta^(3/5), eta = m1 m2 / (m1+m2)^2.

    Algebraically identical to the product/sum form but derived and coded
    independently, so it exercises a different arithmetic path.
    """
    M_total = m1_kg + m2_kg
    eta = (m1_kg * m2_kg) / M_total**2
    return M_total * eta ** 0.6


def ref_f_isco_hz_closed_form(M_total_kg):
    return _C**3 / (6.0 ** 1.5 * math.pi * _G * M_total_kg)


def ref_f_isco_hz_via_kepler(M_total_kg):
    """f_ISCO from Kepler's third law at the Schwarzschild ISCO radius.

    r_isco = 6 G M / c^2 (Schwarzschild), orbital angular frequency from
    Kepler's law Omega = sqrt(GM/r^3), and the dominant (l=m=2) GW frequency
    is twice the orbital frequency. This is a different derivation path
    from the closed-form c^3/(6^1.5 pi G M) used in physics_gw.f_isco.
    """
    r_isco = 6.0 * _G * M_total_kg / _C**2
    omega_orb = math.sqrt(_G * M_total_kg / r_isco**3)
    f_orb = omega_orb / (2.0 * math.pi)
    return 2.0 * f_orb


def ref_strain_amplitude(f_hz, Mc_kg, d_m):
    """Textbook face-on amplitude scale A = 4 (G Mc)^(5/3) (pi f)^(2/3)/(c^4 d).

    Coded directly from this expanded form rather than physics_gw's
    factored implementation (4 G Mc/(c^2 d)) * (pi G Mc f/c^3)^(2/3).
    """
    return 4.0 * (_G * Mc_kg) ** (5.0 / 3.0) * (math.pi * f_hz) ** (2.0 / 3.0) / (_C**4 * d_m)


def ref_inspiral_time_s(f1_hz, f2_hz, Mc_kg):
    Mc_geom = _G * Mc_kg / _C**3
    coeff = 5.0 / (256.0 * math.pi ** (8.0 / 3.0) * Mc_geom ** (5.0 / 3.0))
    return coeff * (f1_hz ** (-8.0 / 3.0) - f2_hz ** (-8.0 / 3.0))


def ref_qnm(M_kg):
    scale = _G * M_kg / _C**3
    return 0.3737 / (2.0 * math.pi * scale), scale / 0.0890


def ref_dfdt_hz_per_s(f_hz, Mc_kg):
    """Independent from-scratch copy of the leading-order df/dt relation,
    coded from the local _G/_C constants rather than importing dfdt()."""
    Mc_geom = _G * Mc_kg / _C**3
    return (96.0 / 5.0) * math.pi ** (8.0 / 3.0) * Mc_geom ** (5.0 / 3.0) * f_hz ** (11.0 / 3.0)


def ref_phase_from_frequency(f0_hz, f_hz, Mc_kg):
    """Closed-form leading-order accumulated GW phase between f0 and f,
    derived (and coded) from scratch here -- NOT by calling physics_gw's
    RK4 phase integration or dfdt(). Starting from the same two governing
    ODEs the model integrates (dPhi/dt = 2 pi f, df/dt = K f^(11/3)):

        dPhi/df = (dPhi/dt)/(df/dt) = 2 pi f / (K f^(11/3)) = (2 pi/K) f^(-8/3)
        Phi(f) - Phi(f0) = (2 pi/K) * integral_{f0}^{f} u^(-8/3) du
                          = (6 pi)/(5K) * [f0^(-5/3) - f^(-5/3)]

    Audit2 (Codex P3-4 / Copilot A2-8) points out that
    test_h_equals_amplitude_times_cosine_of_phase_exactly only re-reads
    physics_gw's own stored h/phase arrays against each other, so a
    coherently wrong phase trajectory (phase and h wrong together) would
    still pass it. This closed-form oracle is algebraically independent of
    the RK4 stepper physics_gw actually runs, so a wrong phase trajectory
    generally will *not* satisfy it.
    """
    Mc_geom = _G * Mc_kg / _C**3
    K = (96.0 / 5.0) * math.pi ** (8.0 / 3.0) * Mc_geom ** (5.0 / 3.0)
    return (6.0 * math.pi / (5.0 * K)) * (f0_hz ** (-5.0 / 3.0) - f_hz ** (-5.0 / 3.0))


class HtmlNode:
    """Small dependency-free HTML tree node used for structural Help tests."""

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
    """Build just enough of a DOM to test sections, tables, and cards."""

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


def has_class(node, class_name):
    return class_name in node.attrs.get("class", "").split()


def nodes_by_id(root, element_id):
    return descendants(root, lambda node: node.attrs.get("id") == element_id)


def main_argparse_defaults(directory):
    """Extract --flag default values straight from main.py's argparse calls."""
    tree = ast.parse((directory / "main.py").read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            flag = node.args[0].value
            for keyword in node.keywords:
                if keyword.arg == "default":
                    defaults[flag] = ast.literal_eval(keyword.value)
                elif keyword.arg == "action" and keyword.value.value == "store_true":
                    defaults[flag] = False
    return defaults


# ===========================================================================
# Module discovery
# ===========================================================================
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

    def test_can_run_from_a_flattened_layout(self):
        """Confirms this test file can locate and run against its target
        modules when copied beside them in a flat directory (not the
        canonical tests/ layout) -- e.g. an upload interface that presents
        every file as a single flat list.

        Audit3: the user's own words are the reason this changed --
        "I ONLY specified the need to run in a flattened folder because
        multiple AIs were complaining about test_physics_xxx.py not being
        able to find its target files from a flat folder... Human end-users
        would only be interested in running test_physics_xxx.py from within
        the tests/ folder." Prior to this round, this test re-ran the
        *entire* 141-test suite a second time as a subprocess from the flat
        copy, which silently doubled the cost of every validation pass
        without the user having asked for that -- it only ever needed to
        prove that module discovery itself works from a flat layout, which
        test_finds_flattened_layout above already checks in-process. This
        version instead runs just that one trivial, dependency-free test
        as a subprocess from a real flat copy on disk, as an end-to-end
        smoke check that nothing about a *literal file copy* (as opposed to
        the in-process Path arithmetic test_finds_flattened_layout checks)
        breaks discovery -- without re-running the rest of the suite.
        """
        if os.environ.get("GW_FLATTENED_TEST_CHILD") == "1":
            return

        with tempfile.TemporaryDirectory() as temporary:
            flat_dir = Path(temporary)
            for name in (*CORE_MODULE_FILES, HELP_FILE):
                shutil.copy2(MODULE_DIR / name, flat_dir / name)
            flat_test = flat_dir / "test_physics_gw.py"
            shutil.copy2(Path(__file__), flat_test)

            environment = os.environ.copy()
            environment["GW_FLATTENED_TEST_CHILD"] = "1"
            environment["MPLBACKEND"] = "Agg"
            result = subprocess.run(
                [sys.executable, str(flat_test),
                 "TestModuleDiscovery.test_finds_flattened_layout"],
                cwd=flat_dir,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OK", result.stdout + result.stderr)
            self.assertIn("Ran 1 test", result.stdout + result.stderr)


# ===========================================================================
# Metadata, versioning, and Python-version compatibility
# ===========================================================================
class TestMetadataAndCompatibility(unittest.TestCase):
    def test_model_version(self):
        # Audit3 round: bumped 1.2.0 -> 1.3.0 for this round's behavior
        # changes (CSV export_timestamp_utc field rename, atomic
        # publish-or-cleanup write semantics for both CSV and PNG output,
        # and the corrected chirp_mass_from_fdot() subnormal-boundary
        # error message).
        self.assertEqual(physics.MODEL_VERSION, "1.3.0")

    def test_build_coverage_is_exactly_the_executable_core(self):
        self.assertEqual(tuple(physics.BUILD_ID_COVERS), CORE_MODULE_FILES)
        self.assertNotIn(HELP_FILE, physics.BUILD_ID_COVERS)
        self.assertFalse(any("test" in name for name in physics.BUILD_ID_COVERS))

    def test_build_id_matches_independent_calculation(self):
        self.assertRegex(physics.BUILD_ID, r"^[0-9a-f]{12}$")
        self.assertEqual(physics.BUILD_ID, recompute_build_id(MODULE_DIR))

    def test_build_id_is_insensitive_to_line_ending_convention(self):
        crlf_digest = hashlib.sha256()
        for name in physics.BUILD_ID_COVERS:
            text = (MODULE_DIR / name).read_text(encoding="utf-8")
            content = text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")
            # BUILD_ID is defined on LF-normalized content, so re-derive the
            # LF-normalized digest and confirm CRLF text normalizes to it.
            normalized = content.replace(b"\r\n", b"\n")
            crlf_digest.update(name.encode("utf-8"))
            crlf_digest.update(len(normalized).to_bytes(8, "big"))
            crlf_digest.update(normalized)
        self.assertEqual(crlf_digest.hexdigest()[:12], physics.BUILD_ID)

    def test_driver_reports_same_metadata(self):
        self.assertEqual(
            driver.version_info(),
            {"model_version": physics.MODEL_VERSION, "build_id": physics.BUILD_ID},
        )

    def test_all_core_sources_parse_as_python_3_10(self):
        for name in CORE_MODULE_FILES:
            with self.subTest(name=name):
                source = (MODULE_DIR / name).read_text(encoding="utf-8")
                ast.parse(source, filename=name, feature_version=(3, 10))

    def test_build_id_returns_unknown_when_source_unreadable(self):
        """A2 regression: in a frozen/zipped/relocated distribution the four
        core files may not be readable beside physics_gw.py on disk (e.g.
        packed into a single-file executable). _compute_build_id() must
        degrade to the documented literal "unknown" rather than raising and
        preventing the program from running at all."""
        with mock.patch("builtins.open", side_effect=OSError("simulated frozen distribution")):
            self.assertEqual(physics._compute_build_id(), "unknown")

    def test_version_command(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"],
            cwd=MODULE_DIR,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            f"GravitationalWaveSources {physics.MODEL_VERSION} (build {physics.BUILD_ID})",
        )


# ===========================================================================
# Physics: independent-oracle correctness
# ===========================================================================
class TestPhysicsIndependentReferences(unittest.TestCase):
    """Cross-check every formula against a from-scratch reference, never by
    calling another physics_gw function that shares its derivation."""

    def test_constants_match_conventional_values(self):
        self.assertAlmostEqual(physics.G, _G, delta=1e-16)
        self.assertAlmostEqual(physics.c, _C, delta=1.0)
        self.assertEqual(physics.c, 2.99792458e8)  # defined exactly by SI
        self.assertAlmostEqual(physics.M_sun, _MSUN, delta=1e20)
        self.assertAlmostEqual(physics.MPC_M, _MPC, delta=1e10)

    def test_chirp_mass_matches_symmetric_mass_ratio_form(self):
        rng = np.random.default_rng(20260831)  # deterministic seed
        for _ in range(25):
            m1 = rng.uniform(1.0, 80.0) * physics.M_sun
            m2 = rng.uniform(1.0, 80.0) * physics.M_sun
            code_value = physics.chirp_mass(m1, m2)
            ref_value = ref_chirp_mass_via_symmetric_ratio_kg(m1, m2)
            self.assertAlmostEqual(code_value / ref_value, 1.0, places=12)

    def test_chirp_mass_is_symmetric_under_component_swap(self):
        m1, m2 = 36.0 * physics.M_sun, 29.0 * physics.M_sun
        self.assertEqual(physics.chirp_mass(m1, m2), physics.chirp_mass(m2, m1))

    def test_chirp_mass_of_equal_masses_is_2_to_the_neg_1_5_times_component(self):
        # For m1=m2=m: Mc = (m^2)^0.6/(2m)^0.2 = m^1.2/(2^0.2 m^0.2)
        #             = m * 2^(-0.2)
        m = 1.4 * physics.M_sun
        expected = m * 2.0 ** (-0.2)
        self.assertAlmostEqual(physics.chirp_mass(m, m) / expected, 1.0, places=12)

    def test_f_isco_matches_kepler_isco_radius_derivation(self):
        for M_msun in (2.8, 20.0, 65.0, 400.0):
            M = M_msun * physics.M_sun
            with self.subTest(M_msun=M_msun):
                code_value = physics.f_isco(M)
                ref_value = ref_f_isco_hz_via_kepler(M)
                self.assertAlmostEqual(code_value / ref_value, 1.0, places=10)

    def test_f_isco_scales_as_inverse_total_mass(self):
        f1 = physics.f_isco(20.0 * physics.M_sun)
        f2 = physics.f_isco(40.0 * physics.M_sun)
        self.assertAlmostEqual(f1 / f2, 2.0, places=10)

    def test_dfdt_matches_finite_difference_of_independent_inspiral_time(self):
        """dt(f)/df = 1/(df/dt); cross-check dfdt() via numerical
        differentiation of a from-scratch inspiral-time reference -- a
        genuinely different numerical method, not a re-reading of dfdt().
        """
        Mc = ref_chirp_mass_kg(1.4 * physics.M_sun, 1.4 * physics.M_sun)
        f0 = 100.0
        h = 1e-3

        def T(f):
            return ref_inspiral_time_s(f0, f, Mc)

        dT_df = (T(f0 + h) - T(f0 - h)) / (2 * h)
        analytic_dfdt = 1.0 / dT_df
        code_dfdt = physics.dfdt(f0, Mc)
        self.assertAlmostEqual(code_dfdt / analytic_dfdt, 1.0, places=7)

    def test_dfdt_scales_as_f_to_the_11_over_3(self):
        Mc = 1.2 * physics.M_sun
        f_lo, f_hi = 30.0, 60.0
        ratio = physics.dfdt(f_hi, Mc) / physics.dfdt(f_lo, Mc)
        expected = (f_hi / f_lo) ** (11.0 / 3.0)
        self.assertAlmostEqual(ratio / expected, 1.0, places=10)

    def test_strain_amplitude_matches_textbook_expanded_form(self):
        Mc = ref_chirp_mass_kg(1.4 * physics.M_sun, 1.4 * physics.M_sun)
        d = 400.0 * physics.MPC_M
        for f in (20.0, 100.0, 1000.0, 1570.0):
            with self.subTest(f=f):
                code_value = physics.strain_amplitude(f, Mc, d)
                ref_value = ref_strain_amplitude(f, Mc, d)
                self.assertAlmostEqual(code_value / ref_value, 1.0, places=12)

    def test_strain_amplitude_scales_inversely_with_distance(self):
        Mc = 1.2 * physics.M_sun
        f = 100.0
        A1 = physics.strain_amplitude(f, Mc, 100.0 * physics.MPC_M)
        A2 = physics.strain_amplitude(f, Mc, 400.0 * physics.MPC_M)
        self.assertAlmostEqual(A1 / A2, 4.0, places=10)

    def test_inspiral_time_matches_independent_reference(self):
        Mc = ref_chirp_mass_kg(1.4 * physics.M_sun, 1.4 * physics.M_sun)
        code_value = physics.inspiral_time(20.0, 1570.0167673975536, Mc)
        ref_value = ref_inspiral_time_s(20.0, 1570.0167673975536, Mc)
        self.assertAlmostEqual(code_value / ref_value, 1.0, places=12)

    def test_inspiral_time_power_law_in_f_start(self):
        """T(f1->fend) / T(f2->fend) ~ f1^(-8/3)/f2^(-8/3) when the upper
        cutoff term is comparatively small (matches Help EXP-6)."""
        Mc = 1.2188 * physics.M_sun
        f_isco_hz = 1570.0
        T15 = physics.inspiral_time(15.0, f_isco_hz, Mc)
        T20 = physics.inspiral_time(20.0, f_isco_hz, Mc)
        measured_ratio = T15 / T20
        predicted_ratio = (15.0 / 20.0) ** (-8.0 / 3.0)
        self.assertAlmostEqual(measured_ratio, predicted_ratio, delta=2e-3)

    def test_qnm_params_match_independent_reference(self):
        M_final = 0.95 * 65.0 * physics.M_sun
        code_f, code_tau = physics.qnm_params(M_final)
        ref_f, ref_tau = ref_qnm(M_final)
        self.assertAlmostEqual(code_f / ref_f, 1.0, places=12)
        self.assertAlmostEqual(code_tau / ref_tau, 1.0, places=12)

    def test_qnm_frequency_scales_as_inverse_remnant_mass(self):
        f1, _ = physics.qnm_params(20.0 * physics.M_sun)
        f2, _ = physics.qnm_params(40.0 * physics.M_sun)
        self.assertAlmostEqual(f1 / f2, 2.0, places=10)

    def test_qnm_decay_time_scales_as_remnant_mass(self):
        _, tau1 = physics.qnm_params(20.0 * physics.M_sun)
        _, tau2 = physics.qnm_params(40.0 * physics.M_sun)
        self.assertAlmostEqual(tau2 / tau1, 2.0, places=10)

    def test_qnm_f_times_m_and_tau_over_m_are_exactly_scale_invariant(self):
        """A fixed dimensionless M*omega forces f*M and tau/M to be exact
        physical invariants across arbitrary remnant masses."""
        products, ratios = [], []
        for M_msun in (10.0, 40.0, 160.0, 2.8):
            M = M_msun * physics.M_sun
            f_qnm, tau_qnm = physics.qnm_params(M)
            products.append(f_qnm * M)
            ratios.append(tau_qnm / M)
        for value in products[1:]:
            self.assertAlmostEqual(value / products[0], 1.0, places=12)
        for value in ratios[1:]:
            self.assertAlmostEqual(value / ratios[0], 1.0, places=12)


# ===========================================================================
# chirp_mass_from_fdot: the observational inverse (Codex P2-1 / Gemini)
# ===========================================================================
class TestChirpMassFromFdot(unittest.TestCase):
    """chirp_mass_from_fdot() is the leading-order *observational* inference
    -- Mc from a measured (f, df/dt) -- as distinct from chirp_mass(), which
    requires already-known component masses. Cross-checked against a
    from-scratch df/dt reference (ref_dfdt_hz_per_s, coded independently of
    physics_gw.dfdt) by injecting a known Mc, generating fdot from it, and
    confirming the inverse recovers the same Mc."""

    def test_recovers_injected_chirp_mass_at_several_frequencies(self):
        Mc_kg = ref_chirp_mass_kg(1.4 * _MSUN, 1.4 * _MSUN)
        for f in (20.0, 100.0, 500.0, 1000.0):
            with self.subTest(f=f):
                fdot = ref_dfdt_hz_per_s(f, Mc_kg)
                recovered = physics.chirp_mass_from_fdot(f, fdot)
                self.assertAlmostEqual(recovered / Mc_kg, 1.0, places=9)

    def test_recovers_injected_chirp_mass_for_bbh_case(self):
        Mc_kg = ref_chirp_mass_kg(36.0 * _MSUN, 29.0 * _MSUN)
        fdot = ref_dfdt_hz_per_s(50.0, Mc_kg)
        recovered = physics.chirp_mass_from_fdot(50.0, fdot)
        self.assertAlmostEqual(recovered / Mc_kg, 1.0, places=9)

    def test_round_trips_exactly_against_dfdt(self):
        Mc_kg = 1.2188 * physics.M_sun
        f = 20.0
        fdot = physics.dfdt(f, Mc_kg)
        recovered = physics.chirp_mass_from_fdot(f, fdot)
        self.assertAlmostEqual(recovered / Mc_kg, 1.0, places=12)

    def test_matches_default_run_via_finite_difference_of_two_samples(self):
        """Emulates the EXP-7 student workflow: two (t, f) samples from the
        default binary run at EXP-7's documented --f_start 100 (as a
        student would read from --csvdir output), a two-point finite-
        difference fdot estimate, then inversion -- must recover the run's
        own printed chirp mass. Audit3 (Codex P2-2) moved EXP-7's
        documented data-generation command from the default f_start=20 Hz
        (a 789,004-row/~78 MiB export) to --f_start 100 (a ~10,788-row/
        ~1 MiB export); this test follows that same command so it keeps
        emulating what a student following the Help text would actually
        run, not a larger export nobody is instructed to generate."""
        result = physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=100.0)
        t, f = result["t"], result["f"]
        f1, f2 = f[0], f[10]
        t1, t2 = t[0], t[10]
        # These are also the exact two rows EXP-7's copyable snippet uses --
        # the run and the two-point spacing (dt=2e-4, indices 0 and 10, i.e.
        # 0.002 s apart) are identical, so this pins down the Help text's
        # own numbers, not merely a similar independent scenario.
        self.assertEqual((t1, f1), (0.0, 100.0))
        self.assertEqual((t2, f2), (0.0020000000000000005, 100.03476529382071))
        fdot_est = (f2 - f1) / (t2 - t1)
        f_mid = 0.5 * (f1 + f2)
        Mc_kg = physics.chirp_mass_from_fdot(f_mid, fdot_est)
        self.assertEqual(f"{Mc_kg / physics.M_sun:.4f}", "1.2188")
        self.assertAlmostEqual(Mc_kg / physics.M_sun, result["summary"]["Mc_msun"], places=3)

    def test_f_must_be_positive(self):
        for value in (0.0, -1.0, -100.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "f_hz must be greater than zero"):
                    physics.chirp_mass_from_fdot(value, 0.05)

    def test_fdot_must_be_positive(self):
        for value in (0.0, -1e-3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "fdot_hz_per_s must be greater than zero"):
                    physics.chirp_mass_from_fdot(20.0, value)

    def test_rejects_non_finite_and_non_numeric(self):
        for value in (math.nan, math.inf, -math.inf, None, [], {}, "abc", 1 + 2j):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    physics.chirp_mass_from_fdot(value, 0.05)
                with self.assertRaises(ValueError):
                    physics.chirp_mass_from_fdot(20.0, value)

    def test_bool_is_rejected_not_silently_coerced(self):
        with self.assertRaisesRegex(ValueError, "bool is not"):
            physics.chirp_mass_from_fdot(True, 0.05)
        with self.assertRaisesRegex(ValueError, "bool is not"):
            physics.chirp_mass_from_fdot(20.0, False)

    def test_extreme_inputs_never_silently_underflow_or_overflow(self):
        """Audit2 (Codex P2-1) reproducer table: before the log-domain
        rewrite, some of these cases raised an uncaught OverflowError
        (from Python float exponentiation/multiplication) and others
        silently returned 0.0 kg (from premature underflow of an
        intermediate factor) instead of either a valid finite chirp mass
        or a clean ValueError. Every case below must now do one of the
        two acceptable things: return a positive finite float, or raise
        ValueError -- never raise OverflowError and never return exactly
        0.0 (a physically meaningless "successful" answer)."""
        # (f_hz, fdot_hz_per_s, expect) -- expect is "raises" or "finite"
        cases = [
            (5e-324, 1.0, "raises"),   # smallest subnormal double as f
            (1e-300, 1.0, "raises"),
            (1e308, 1.0, "raises"),
            (1e308, 1e308, "raises"),
            (20.0, 5e-324, "finite"),  # smallest subnormal double as fdot
            (20.0, 1e308, "finite"),
        ]
        for f_hz, fdot, expect in cases:
            with self.subTest(f_hz=f_hz, fdot=fdot):
                if expect == "raises":
                    with self.assertRaises(ValueError) as ctx:
                        physics.chirp_mass_from_fdot(f_hz, fdot)
                    self.assertNotIsInstance(ctx.exception, OverflowError)
                else:
                    Mc_kg = physics.chirp_mass_from_fdot(f_hz, fdot)
                    self.assertTrue(math.isfinite(Mc_kg))
                    self.assertGreater(Mc_kg, 0.0)

    def test_log_domain_bounds_are_finite_and_ordered(self):
        """physics_gw._LOG_FLOAT_MIN/_MAX must bracket a genuine finite
        range (used to reject an out-of-range implied chirp mass while
        still working entirely in the log domain); a regression that
        broke this constant setup would silently defeat every boundary
        check above."""
        self.assertTrue(math.isfinite(physics._LOG_FLOAT_MAX))
        self.assertTrue(math.isfinite(physics._LOG_FLOAT_MIN))
        self.assertLess(physics._LOG_FLOAT_MIN, physics._LOG_FLOAT_MAX)

    def test_lower_bound_is_exercised_immediately_below_and_above_float_min(self):
        """Audit3 (Codex P3-2): pin the exact deterministic boundary between
        a rejected (sub-float_info.min) and accepted (normal-range) result,
        rather than only testing far-away extreme values. f_hz=1e200 and the
        two fdot values below straddle the point where the implied chirp
        mass in kg crosses sys.float_info.min -- the smallest positive
        *normal* double -- to within one ULP in the log domain. This also
        pins down the reworded error message (previously the factually
        wrong "too small to represent"; a positive value this small *is*
        representable, just only as a reduced-precision subnormal float)."""
        f_hz = 1e200
        fdot_just_below = 6.990590547523025e+163   # implied Mc just under float_info.min: rejected
        fdot_just_above = 6.9906045287181e+163     # implied Mc just over float_info.min: accepted

        with self.assertRaisesRegex(
            ValueError,
            "below this program's supported normal-precision range",
        ) as ctx:
            physics.chirp_mass_from_fdot(f_hz, fdot_just_below)
        self.assertNotRegex(str(ctx.exception), "too small to represent")

        Mc_kg = physics.chirp_mass_from_fdot(f_hz, fdot_just_above)
        self.assertTrue(math.isfinite(Mc_kg))
        self.assertGreater(Mc_kg, 0.0)
        self.assertGreaterEqual(Mc_kg, sys.float_info.min)
        # Barely above the normal-float floor, not some unrelated magnitude.
        self.assertLess(Mc_kg, sys.float_info.min * 1.01)


class TestLowLevelHelperDomainContract(unittest.TestCase):
    """chirp_mass(), dfdt(), strain_amplitude(), f_isco(), qnm_params(), and
    inspiral_time() are documented (Audit3 Codex P3-3) as low-level helpers
    that deliberately perform no input validation of their own -- validation
    lives in integrate_inspiral(), which is on the only path a CLI user or
    the public integration API can reach these functions through. Adding a
    validation check to each of these would add per-call overhead on a hot
    path: dfdt() alone is called 4 times per RK4 step, so a multi-hundred-
    thousand-step integration would pay that cost roughly a million times
    for a safety net integrate_inspiral() already provides. These tests
    pin down -- not endorse as "correct" -- exactly what happens today when
    these helpers are called directly with out-of-domain inputs, so a
    future accidental change in that undefined behavior is caught."""

    def test_chirp_mass_of_negative_masses_returns_complex_not_an_exception(self):
        result = physics.chirp_mass(-1.0, -1.0)
        self.assertIsInstance(result, complex)
        self.assertAlmostEqual(result.real, 0.7042902001692477, places=9)
        self.assertAlmostEqual(result.imag, -0.5116967824803669, places=9)

    def test_dfdt_of_negative_chirp_mass_returns_complex_not_an_exception(self):
        result = physics.dfdt(-1.0, 1.0)
        self.assertIsInstance(result, complex)

    def test_f_isco_of_zero_total_mass_raises_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            physics.f_isco(0.0)

    def test_qnm_params_of_zero_remnant_mass_raises_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            physics.qnm_params(0.0)

    def test_strain_amplitude_of_zero_distance_raises_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            physics.strain_amplitude(20.0, 1.0, 0.0)

    def test_inspiral_time_of_zero_chirp_mass_raises_zero_division_error(self):
        with self.assertRaises(ZeroDivisionError):
            physics.inspiral_time(20.0, 20.0, 0.0)


class TestDefaultAndBBHReferenceCase(unittest.TestCase):
    """Bit-for-bit-independent hand recomputation of the two headline
    example cases quoted throughout the Help file, computed here from raw
    physical constants (not by importing physics_gw's helper functions)."""

    def test_default_bns_case_matches_hand_computation(self):
        m1 = m2 = 1.4 * _MSUN
        Mc = ref_chirp_mass_kg(m1, m2)
        M = m1 + m2
        f_isco_hz = ref_f_isco_hz_closed_form(M)
        T = ref_inspiral_time_s(20.0, f_isco_hz, Mc)
        A = ref_strain_amplitude(f_isco_hz, Mc, 400.0 * _MPC)

        self.assertEqual(f"{Mc/_MSUN:.4f}", "1.2188")
        self.assertEqual(f"{f_isco_hz:.1f}", "1570.0")
        self.assertEqual(f"{T:.3f}", "157.800")
        self.assertEqual(f"{A:.3e}", "5.584e-23")

        result = physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=20.0)
        s = result["summary"]
        self.assertAlmostEqual(s["Mc_msun"] / (Mc / _MSUN), 1.0, places=9)
        self.assertAlmostEqual(s["f_isco_hz"] / f_isco_hz, 1.0, places=9)
        self.assertAlmostEqual(s["T_band_s"] / T, 1.0, delta=1e-5)
        self.assertAlmostEqual(s["A_isco"] / A, 1.0, places=9)
        self.assertEqual(s["inspiral_steps"], 789_004)

    def test_bbh_ringdown_case_matches_hand_computation(self):
        m1, m2 = 36.0 * _MSUN, 29.0 * _MSUN
        Mc = ref_chirp_mass_kg(m1, m2)
        M = m1 + m2
        f_isco_hz = ref_f_isco_hz_closed_form(M)
        M_final = 0.95 * M
        f_qnm, tau_qnm = ref_qnm(M_final)

        self.assertEqual(f"{f_isco_hz:.1f}", "67.6")
        self.assertEqual(f"{f_qnm:.1f}", "195.5")
        self.assertEqual(f"{tau_qnm*1e3:.3f}", "3.418")

        result = physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=2e-5, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=6, ringdown_pts=4000,
        )
        s = result["summary"]
        self.assertAlmostEqual(s["f_isco_hz"] / f_isco_hz, 1.0, places=9)
        self.assertAlmostEqual(s["f_qnm_hz"] / f_qnm, 1.0, places=9)
        self.assertAlmostEqual(s["tau_qnm_ms"] / (tau_qnm * 1e3), 1.0, places=9)


# ===========================================================================
# RK4 stepper: hand-checked single step and convergence order
# ===========================================================================
class TestRK4Stepper(unittest.TestCase):
    def test_single_step_matches_hand_computed_rk4_stages(self):
        Mc = 1.2 * physics.M_sun
        f, phase, dt = 100.0, 0.5, 1e-4

        Mc_geom = _G * Mc / _C**3

        def dfdt_ref(fv):
            return (96.0 / 5.0) * math.pi ** (8.0 / 3.0) * Mc_geom ** (5.0 / 3.0) * fv ** (11.0 / 3.0)

        k1_f = dfdt_ref(f)
        k1_p = 2 * math.pi * f
        f2 = f + 0.5 * dt * k1_f
        k2_f = dfdt_ref(f2)
        k2_p = 2 * math.pi * f2
        f3 = f + 0.5 * dt * k2_f
        k3_f = dfdt_ref(f3)
        k3_p = 2 * math.pi * f3
        f4 = f + dt * k3_f
        k4_f = dfdt_ref(f4)
        k4_p = 2 * math.pi * f4
        expected_f = f + (dt / 6.0) * (k1_f + 2 * k2_f + 2 * k3_f + k4_f)
        expected_phase = phase + (dt / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)

        got_f, got_phase = physics._rk4_frequency_phase_step(f, phase, dt, Mc)
        self.assertAlmostEqual(got_f, expected_f, delta=1e-12)
        self.assertAlmostEqual(got_phase, expected_phase, delta=1e-12)

    def test_frequency_strictly_increases_each_step(self):
        Mc = 1.2 * physics.M_sun
        f, phase = 20.0, 0.0
        for _ in range(200):
            f_next, phase_next = physics._rk4_frequency_phase_step(f, phase, 1e-4, Mc)
            self.assertGreater(f_next, f)
            f, phase = f_next, phase_next

    def test_bulk_stepper_is_fourth_order_accurate_away_from_cutoff(self):
        """Away from the ISCO crossing (where no interpolation happens),
        halving dt should shrink the error against a high-resolution
        reference by close to 2^4 = 16, confirming true RK4 order."""
        Mc = physics.chirp_mass(36.0 * physics.M_sun, 29.0 * physics.M_sun)

        def integrate_n_steps(f0, dt, n):
            f, phase = f0, 0.0
            for _ in range(n):
                f, phase = physics._rk4_frequency_phase_step(f, phase, dt, Mc)
            return f, phase

        f0 = 20.0
        T_fixed = 0.3  # well below this system's f_isco (~67.6 Hz)
        f_ref, _ = integrate_n_steps(f0, 1e-5, round(T_fixed / 1e-5))

        errors = []
        for dt in (4e-3, 2e-3, 1e-3):
            f_val, _ = integrate_n_steps(f0, dt, round(T_fixed / dt))
            errors.append(abs(f_val - f_ref))

        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]
        # Generous bracket around 16x: the second halving approaches the
        # double-precision noise floor (errors ~1e-13), so only require the
        # first, cleanly-resolved halving to show clear quartic behavior.
        self.assertGreater(ratio1, 10.0)
        self.assertLess(ratio1, 24.0)
        self.assertGreater(ratio2, 1.0)  # still improving, not noise-dominated blowup

    def test_numerically_integrated_time_to_isco_converges_to_analytic_value(self):
        """The *reported* time to ISCO is limited by the single linearly
        interpolated final step (see Help Algorithm step 6 / EXP-8), so this
        checks convergence to the analytic answer without assuming a clean
        fourth-order rate."""
        Mc = physics.chirp_mass(36.0 * physics.M_sun, 29.0 * physics.M_sun)
        T_analytic = physics.inspiral_time(20.0, physics.f_isco(65.0 * physics.M_sun), Mc)
        errors = []
        for dt in (0.002, 0.001, 0.0005):
            result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=dt, f_start=20.0)
            errors.append(abs(result["t_isco"] - T_analytic))
        self.assertLess(errors[-1], errors[0])
        self.assertLess(errors[0], 2e-4)  # sub-millisecond-scale even at the coarsest dt tried


# ===========================================================================
# integrate_inspiral: nominal behavior and array/summary consistency
# ===========================================================================
class TestIntegrateInspiralNominal(unittest.TestCase):
    def test_structure_and_finiteness_pure_inspiral(self):
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        for key in ("t", "h", "A", "f", "phase"):
            self.assertTrue(np.all(np.isfinite(result[key])), key)
        self.assertEqual(result["t"][0], 0.0)
        self.assertTrue(np.all(np.diff(result["t"]) >= 0.0))
        self.assertTrue(np.all(np.diff(result["f"]) >= 0.0))
        self.assertEqual(result["f"][-1], result["f_isco_hz"])
        self.assertEqual(len(result["t"]), result["summary"]["inspiral_steps"])

    def test_summary_fields_agree_with_returned_arrays(self):
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        s = result["summary"]
        self.assertEqual(s["t_isco_s"], result["t_isco"])
        self.assertEqual(s["T_band_s"], result["t_isco"])
        self.assertEqual(s["f_isco_hz"], result["f_isco_hz"])
        self.assertEqual(s["Mc_msun"], result["Mc_msun"])
        self.assertEqual(s["A_isco"], result["A"][-1])
        self.assertEqual(s["model_version"], physics.MODEL_VERSION)
        self.assertEqual(s["build_id"], physics.BUILD_ID)

    def test_envelope_bounds_the_waveform_amplitude(self):
        """Weaker envelope-only check: |h| <= A everywhere on the inspiral.
        This alone would also pass for h=0, h=0.5*A, or an incorrect phase
        evolution -- see test_h_equals_amplitude_times_cosine_of_phase_
        exactly below for the actual equation check (P3-2 regression: this
        test previously carried the "...cosine_of_accumulated_phase" name
        while only ever testing this inequality)."""
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        self.assertTrue(np.all(np.abs(result["h"]) <= result["A"] * (1.0 + 1e-9)))

    def test_h_equals_amplitude_times_cosine_of_phase_exactly(self):
        """P3-2 regression: with the accumulated GW phase now exposed as
        result["phase"], directly assert the architectural invariant
        h(t) = A(t) cos(Phi(t)) on the inspiral segment, rather than only
        the much weaker |h| <= A bound. This would fail for h=0, h=0.5*A*
        cos(phase), or a wrong phase array, none of which the envelope-only
        check above can catch.

        Audit2 (Codex P3-4 / Copilot A2-8): this is an *internal storage
        consistency* check, not an independent scientific validation of the
        phase trajectory itself -- it recomputes A*cos(phase) from arrays
        physics_gw already produced together, so a phase array that is
        coherently wrong (with h wrong to match) would still satisfy this
        exact identity. See
        test_accumulated_phase_matches_independent_closed_form_oracle below
        for the algebraically-independent check that phase itself is
        physically correct as a function of frequency."""
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        insp_mask = np.isfinite(result["f"])
        expected_h = result["A"][insp_mask] * np.cos(result["phase"][insp_mask])
        np.testing.assert_array_equal(result["h"][insp_mask], expected_h)

    def test_accumulated_phase_matches_independent_closed_form_oracle(self):
        """Audit2 (Codex P3-4 / Copilot A2-8): validate result["phase"]
        itself against ref_phase_from_frequency -- a closed-form expression
        derived and coded from scratch in this file, never calling
        physics_gw.dfdt() or reusing its RK4 phase-accumulation code -- at
        several pre-cutoff frequencies plus the final, linearly interpolated
        ISCO point. A coherently wrong phase trajectory (e.g. an off-by-a-
        constant-factor error in the RK4 phase ODE) would fail this even
        though it would still pass the exact-identity test above."""
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        Mc_kg = result["summary"]["Mc_msun"] * physics.M_sun
        f = result["f"]
        phase = result["phase"]
        f0 = f[0]
        n = len(f)
        # Several ordinary RK4-stepped points: tight relative tolerance.
        for idx in (1, 100, n // 4, n // 2, n - 2):
            with self.subTest(idx=idx):
                expected = ref_phase_from_frequency(f0, f[idx], Mc_kg)
                self.assertAlmostEqual(phase[idx] / expected, 1.0, places=8)
        # Final point is linearly interpolated to the exact ISCO frequency
        # (see the code comment on the final RK4 step), so it carries a
        # somewhat larger, but still small, interpolation error.
        expected_last = ref_phase_from_frequency(f0, f[-1], Mc_kg)
        self.assertAlmostEqual(phase[-1] / expected_last, 1.0, places=4)

    def test_ringdown_h_equals_peak_amplitude_times_damped_cosine_of_phase(self):
        """Same architectural invariant on the ringdown segment: h_rd(t) =
        A_peak * cos(phase_rd(t)) * exp(-t_rd/tau_qnm), independently
        recomputed here from the returned phase, t_isco, f_qnm, and
        tau_qnm_ms rather than re-deriving the ringdown internally."""
        result = physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=6, ringdown_pts=4000,
        )
        insp_mask = np.isfinite(result["f"])
        rd_mask = ~insp_mask
        s = result["summary"]
        A_peak = result["A"][insp_mask][-1]
        tau_qnm_s = s["tau_qnm_ms"] * 1e-3
        t_rd = result["t"][rd_mask] - result["t_isco"]
        expected_h_rd = A_peak * np.cos(result["phase"][rd_mask]) * np.exp(-t_rd / tau_qnm_s)
        np.testing.assert_allclose(result["h"][rd_mask], expected_h_rd, rtol=1e-12)

    def test_ringdown_segment_marks_f_and_a_as_nan_but_h_and_t_finite(self):
        result = physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=6, ringdown_pts=100,
        )
        insp_mask = np.isfinite(result["f"])
        rd_mask = ~insp_mask
        self.assertTrue(np.any(rd_mask))
        self.assertTrue(np.all(np.isnan(result["A"][rd_mask])))
        self.assertTrue(np.all(np.isfinite(result["h"][rd_mask])))
        self.assertTrue(np.all(np.isfinite(result["t"][rd_mask])))
        self.assertTrue(np.all(np.diff(result["t"]) >= 0.0))

    def test_ringdown_disabled_leaves_qnm_fields_as_nan(self):
        result = physics.integrate_inspiral(36.0, 29.0, 440.0, dt=1e-4, f_start=20.0)
        s = result["summary"]
        self.assertFalse(s["include_ringdown"])
        self.assertTrue(math.isnan(s["f_qnm_hz"]))
        self.assertTrue(math.isnan(s["tau_qnm_ms"]))
        self.assertTrue(math.isnan(s["M_final_msun"]))
        self.assertTrue(np.all(np.isfinite(result["f"])))

    def test_ringdown_seam_is_continuous_to_within_one_ringdown_sample_step(self):
        """The stored ringdown segment intentionally begins one ringdown
        sample spacing after the exact ISCO instant (see the code comment
        in integrate_inspiral about skipping t_rd[0] to avoid a duplicate
        timestamp), so the first ringdown h should be *close to* but not
        necessarily bit-identical to the last inspiral h."""
        result = physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-5, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=6, ringdown_pts=4000,
        )
        insp_mask = np.isfinite(result["f"])
        last_insp_h = result["h"][insp_mask][-1]
        first_rd_h = result["h"][~insp_mask][0]
        first_rd_dt = result["t"][~insp_mask][0] - result["t_isco"]
        self.assertGreater(first_rd_dt, 0.0)
        self.assertLess(first_rd_dt, 1e-4)  # a small fraction of tau_qnm (~3.4ms here)
        self.assertLess(abs(first_rd_h - last_insp_h), 0.01 * abs(last_insp_h))

    def test_mass_ratio_at_fixed_total_mass_leaves_f_isco_unchanged(self):
        """Help EXP-5 invariant: f_ISCO depends only on total mass."""
        f_iscos = []
        for m1, m2 in ((20.0, 20.0), (25.0, 15.0), (30.0, 10.0)):
            result = physics.integrate_inspiral(m1, m2, 400.0, dt=1e-3, f_start=20.0)
            f_iscos.append(result["f_isco_hz"])
        for value in f_iscos[1:]:
            self.assertAlmostEqual(value, f_iscos[0], places=6)

    def test_distance_does_not_affect_timeline_only_amplitude(self):
        """Help EXP-2 invariant: A_isco ~ 1/d while t_isco, f_isco fixed."""
        results = {}
        for d in (100.0, 200.0, 400.0, 800.0):
            results[d] = physics.integrate_inspiral(1.4, 1.4, d, dt=2e-4, f_start=100.0)
        t_isco_values = {r["t_isco"] for r in results.values()}
        f_isco_values = {r["f_isco_hz"] for r in results.values()}
        self.assertEqual(len(t_isco_values), 1)
        self.assertEqual(len(f_isco_values), 1)
        self.assertAlmostEqual(
            results[100.0]["summary"]["A_isco"] / results[200.0]["summary"]["A_isco"],
            2.0, places=8,
        )
        self.assertAlmostEqual(
            results[100.0]["summary"]["A_isco"] / results[800.0]["summary"]["A_isco"],
            8.0, places=8,
        )


# ===========================================================================
# integrate_inspiral: validation, bounds, and error handling
# ===========================================================================
class TestIntegrateInspiralValidation(unittest.TestCase):
    def assert_all_rejected(self, keyword, values, exception=ValueError):
        for value in values:
            with self.subTest(keyword=keyword, value=value):
                kwargs = dict(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0, dt=2e-4, f_start=20.0)
                kwargs[keyword] = value
                with self.assertRaises(exception):
                    physics.integrate_inspiral(**kwargs)

    def test_mass_value_validation(self):
        self.assert_all_rejected("m1_msun", [0.0, -1.0, math.nan, math.inf, -math.inf])
        self.assert_all_rejected("m2_msun", [0.0, -1.0, math.nan, math.inf, -math.inf])

    def test_distance_value_validation(self):
        self.assert_all_rejected("d_mpc", [0.0, -1.0, math.nan, math.inf, -math.inf])

    def test_timestep_value_validation(self):
        self.assert_all_rejected("dt", [0.0, -1e-4, math.nan, math.inf, -math.inf])

    def test_f_start_value_validation(self):
        self.assert_all_rejected("f_start", [0.0, -1.0, math.nan, math.inf, -math.inf])

    def test_type_validation_rejects_non_numeric_values(self):
        for keyword in ("m1_msun", "m2_msun", "d_mpc", "dt", "f_start"):
            self.assert_all_rejected(keyword, [None, [], {}, "abc", 1 + 2j])

    def test_numeric_strings_are_permissively_accepted_like_float(self):
        """_require_finite deliberately uses float(value) rather than a
        strict isinstance check, so a numeric string such as "1.4" is
        accepted and normalized -- documented (not a defect) behavior that
        differs from EXAMPLE_physics_cannon's stricter TypeError-on-string
        convention."""
        result = physics.integrate_inspiral("1.4", "1.4", "400.0", dt="2e-4", f_start="20.0")
        self.assertEqual(result["summary"]["m1_msun"], 1.4)
        self.assertIsInstance(result["summary"]["m1_msun"], float)

    def test_bool_is_explicitly_rejected_not_silently_coerced(self):
        """Regression test for a defect found during this audit round:
        float(True) == 1.0, so before this fix a caller passing
        m1_msun=True (or any of the other numeric keywords) had it silently
        accepted as 1.0 solar mass instead of being rejected. Reproduced
        against the pre-fix code (float() conversion with no bool guard)
        before the guard in physics_gw._require_finite was added.
        """
        for keyword in ("m1_msun", "m2_msun", "d_mpc", "dt", "f_start",
                        "n_ringdown_tau", "ringdown_pts"):
            with self.subTest(keyword=keyword):
                # n_ringdown_tau/ringdown_pts are only validated (Audit2
                # Copilot A2-4) when include_ringdown=True, so this test
                # must request ringdown to still exercise their bool guard.
                kwargs = dict(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0, dt=2e-4,
                               f_start=20.0, include_ringdown=True,
                               n_ringdown_tau=6, ringdown_pts=4000)
                kwargs[keyword] = True
                with self.assertRaisesRegex(ValueError, "bool is not"):
                    physics.integrate_inspiral(**kwargs)
                # Confirm it is rejected, not merely coerced to 1.0/1.
                kwargs[keyword] = False
                with self.assertRaisesRegex(ValueError, "bool is not"):
                    physics.integrate_inspiral(**kwargs)

    def test_numpy_bool_is_also_rejected(self):
        with self.assertRaisesRegex(ValueError, "bool is not"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=np.bool_(True), f_start=20.0)

    def test_include_ringdown_must_be_boolean(self):
        for value in (1, 0, "true", None, [], 1.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "include_ringdown must be True or False"):
                    physics.integrate_inspiral(
                        1.4, 1.4, 400.0, dt=2e-4, f_start=20.0, include_ringdown=value
                    )
        # True/False (and numpy bool) remain accepted.
        physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                                    include_ringdown=np.bool_(False))

    def test_include_ringdown_docstring_matches_np_bool_behavior(self):
        """Audit2 (Copilot 'Prior A5' follow-up): the docstring explicitly
        promises numpy.bool_ is accepted (it is not treated as one of the
        rejected duck-typed boolean-like objects) -- confirm both the
        wording and the actual behavior agree, so a future edit to either
        cannot silently drift out of sync."""
        doc = physics.integrate_inspiral.__doc__
        self.assertIn("numpy.bool_", doc)
        result = physics.integrate_inspiral(
            1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
            include_ringdown=np.bool_(True),
            n_ringdown_tau=6, ringdown_pts=4000,
        )
        self.assertTrue(result["summary"]["include_ringdown"])

    def test_f_start_at_or_above_isco_is_rejected(self):
        f_isco_hz = physics.f_isco(2.8 * physics.M_sun)
        with self.assertRaisesRegex(ValueError, "must be below the Schwarzschild ISCO"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=f_isco_hz)
        with self.assertRaisesRegex(ValueError, "must be below the Schwarzschild ISCO"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=f_isco_hz + 1.0)

    def test_nyquist_violating_timestep_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too large to sample the waveform"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=0.01, f_start=20.0)

    def test_near_isco_start_frequency_hits_min_step_floor_not_the_isco_check(self):
        """A1 regression: dt is not checked against the physical timescale
        of the *requested* run, only the ISCO Nyquist limit -- so a tiny
        residual band (f_start just below f_ISCO) must still be caught by
        MIN_INSPIRAL_STEPS rather than silently returning a 1-2 sample
        'inspiral'."""
        f_isco_hz = physics.f_isco(2.8 * physics.M_sun)
        with self.assertRaisesRegex(ValueError, "too large to resolve this inspiral"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=f_isco_hz - 0.01)

    def test_nyquist_bound_is_always_tighter_than_the_min_step_bound_for_dt_alone(self):
        """For any dt coarse enough to trip MIN_INSPIRAL_STEPS on a normal
        (non-near-ISCO) request, the Nyquist check has already rejected it
        first -- so MIN_INSPIRAL_STEPS is specifically a near-ISCO-start
        safeguard (see test_near_isco_start_frequency_hits_min_step_floor_
        not_the_isco_check), not a general coarse-dt safeguard. This test
        documents that ordering rather than assuming it.
        """
        with self.assertRaisesRegex(ValueError, "too large to sample the waveform"):
            physics.integrate_inspiral(36.0, 29.0, 440.0, dt=0.05, f_start=20.0)

    def test_estimated_step_ceiling_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "exceeding the safety limit"):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=1e-6)

    def test_n_ringdown_tau_bounds(self):
        for value in (0, -1, 0.5, 2.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "n_ringdown_tau must be a positive integer"):
                    physics.integrate_inspiral(
                        1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                        include_ringdown=True, n_ringdown_tau=value,
                    )

    def test_ringdown_pts_lower_bound(self):
        for value in (0, 1, -5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "ringdown_pts must be an integer of at least 2"):
                    physics.integrate_inspiral(
                        1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                        include_ringdown=True, ringdown_pts=value,
                    )

    def test_ringdown_pts_upper_bound(self):
        with self.assertRaisesRegex(ValueError, r"must not exceed 500,000"):
            physics.integrate_inspiral(
                1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                include_ringdown=True, ringdown_pts=20_000_000,
            )
        # The documented ceiling itself must still be accepted (slow but valid).
        result = physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, ringdown_pts=physics.MAX_RINGDOWN_POINTS,
            n_ringdown_tau=1,
        )
        self.assertEqual(
            np.count_nonzero(~np.isfinite(result["f"])),
            physics.MAX_RINGDOWN_POINTS - 1,
        )

    def test_ringdown_samples_per_cycle_below_threshold_is_rejected(self):
        """P1-1 regression: Codex's exact reproducer grid (36+29 Msun,
        dt=1e-4) of settings that were previously accepted but produced an
        aliased or effectively invisible (single-point) ringdown."""
        for n_tau, rd_pts in ((6, 2), (6, 9), (6, 10), (10_000, 4_000)):
            with self.subTest(n_ringdown_tau=n_tau, ringdown_pts=rd_pts):
                with self.assertRaisesRegex(ValueError, "too small to resolve the"):
                    physics.integrate_inspiral(
                        36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                        include_ringdown=True, n_ringdown_tau=n_tau, ringdown_pts=rd_pts,
                    )

    def test_ringdown_samples_per_cycle_at_and_above_threshold_is_accepted(self):
        """Boundary check for the 36+29 Msun / n_ringdown_tau=6 case: the
        computed minimum ringdown_pts (34, independently recomputed here
        from qnm_params rather than hardcoded) must be the exact accept/
        reject boundary."""
        f_qnm, tau_qnm = physics.qnm_params(0.95 * 65.0 * physics.M_sun)
        min_pts_needed = math.ceil(physics.MIN_RINGDOWN_SAMPLES_PER_CYCLE
                                    * 6 * tau_qnm * f_qnm) + 1
        self.assertEqual(min_pts_needed, 34)
        with self.assertRaisesRegex(ValueError, "too small to resolve the"):
            physics.integrate_inspiral(
                36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=6, ringdown_pts=min_pts_needed - 1,
            )
        for rd_pts in (min_pts_needed, min_pts_needed + 1):
            physics.integrate_inspiral(
                36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=6, ringdown_pts=rd_pts,
            )  # must not raise

    def test_n_ringdown_tau_boundary_at_max_ringdown_points_ceiling(self):
        """Codex Audit2's 'Recommended Regression Additions' #6: an
        n_ringdown_tau boundary test just below/at/above the largest value
        still resolvable within MAX_RINGDOWN_POINTS. f_QNM*tau_QNM is a
        fixed dimensionless invariant for a given remnant mass (see
        test_qnm_f_times_m_and_tau_over_m_are_exactly_scale_invariant), so
        the minimum ringdown_pts needed grows linearly with n_ringdown_tau;
        this independently recomputes that boundary from qnm_params rather
        than hardcoding it."""
        f_qnm, tau_qnm = physics.qnm_params(0.95 * 65.0 * physics.M_sun)

        def min_pts_needed(n_tau):
            return math.ceil(
                physics.MIN_RINGDOWN_SAMPLES_PER_CYCLE * n_tau * tau_qnm * f_qnm
            ) + 1

        # Binary search the largest integer n_tau still resolvable at the
        # documented ceiling, MAX_RINGDOWN_POINTS.
        lo, hi = 1, 10_000_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if min_pts_needed(mid) <= physics.MAX_RINGDOWN_POINTS:
                lo = mid
            else:
                hi = mid - 1
        n_tau_max = lo
        self.assertLessEqual(min_pts_needed(n_tau_max), physics.MAX_RINGDOWN_POINTS)
        self.assertGreater(min_pts_needed(n_tau_max + 1), physics.MAX_RINGDOWN_POINTS)

        # At the boundary: MAX_RINGDOWN_POINTS is (just barely) sufficient.
        physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=n_tau_max,
            ringdown_pts=physics.MAX_RINGDOWN_POINTS,
        )  # must not raise
        # One below the boundary is also fine (less demanding).
        physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=n_tau_max - 1,
            ringdown_pts=physics.MAX_RINGDOWN_POINTS,
        )  # must not raise
        # One above the boundary: no ringdown_pts up to MAX_RINGDOWN_POINTS
        # can resolve it, so even the ceiling itself must be rejected.
        with self.assertRaisesRegex(ValueError, "too small to resolve the"):
            physics.integrate_inspiral(
                36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=n_tau_max + 1,
                ringdown_pts=physics.MAX_RINGDOWN_POINTS,
            )

    def test_ringdown_default_settings_remain_valid(self):
        """The documented default (--n_tau 6 --rd_pts 4000) must remain
        comfortably accepted after the P1-1 fix -- for the 36+29 case this
        needs only 34 points (see above), far below the 4000 default."""
        physics.integrate_inspiral(
            36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
            include_ringdown=True, n_ringdown_tau=6, ringdown_pts=4000,
        )  # must not raise

    def test_out_of_range_ringdown_params_are_ignored_when_ringdown_disabled(self):
        """Audit2 (Copilot A2-4): n_ringdown_tau/ringdown_pts are dormant
        configuration when include_ringdown=False and must not be able to
        break a pure-inspiral call, however invalid their values are on
        their own terms."""
        for n_tau, rd_pts in ((-5, 4000), (0, 4000), (0.5, 4000),
                               (6, 0), (6, 1), (6, -5), (6, 20_000_000)):
            with self.subTest(n_ringdown_tau=n_tau, ringdown_pts=rd_pts):
                result = physics.integrate_inspiral(
                    1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                    include_ringdown=False,
                    n_ringdown_tau=n_tau, ringdown_pts=rd_pts,
                )  # must not raise
                self.assertFalse(result["summary"]["include_ringdown"])

    def test_same_out_of_range_ringdown_params_still_rejected_when_enabled(self):
        """Companion to the A2-4 test above: the same values must still be
        rejected once include_ringdown=True actually requests a ringdown
        segment -- gating must not have accidentally disabled the checks
        outright."""
        for n_tau, rd_pts in ((-5, 4000), (0, 4000)):
            with self.subTest(n_ringdown_tau=n_tau, ringdown_pts=rd_pts):
                with self.assertRaisesRegex(ValueError, "n_ringdown_tau must be a positive integer"):
                    physics.integrate_inspiral(
                        1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                        include_ringdown=True, n_ringdown_tau=n_tau, ringdown_pts=rd_pts,
                    )

    def test_huge_n_ringdown_tau_raises_cleanly_not_overflowerror(self):
        """Audit2 (Codex P2-2) reproducer: a --n_tau value large enough that
        the implied minimum-sample-count arithmetic overflows to inf must
        raise a clean ValueError (math.isfinite guard / OverflowError
        catch around math.ceil), never an uncaught OverflowError."""
        with self.assertRaises(ValueError) as ctx:
            physics.integrate_inspiral(
                36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=1e308, ringdown_pts=4000,
            )
        self.assertNotIsInstance(ctx.exception, OverflowError)
        self.assertIn("too large", str(ctx.exception))

    def test_invalid_ringdown_sampling_is_rejected_before_any_rk4_step(self):
        """Audit2 (Codex P2-3 / Copilot A2-2): invalid coupled ringdown
        sampling must be detected before the (potentially long) inspiral
        integration runs, not after. A wall-clock timing comparison is not
        a reliable deterministic assertion, so this instead spies on the
        RK4 stepper itself and asserts it is never called for a request
        that is invalid on ringdown-sampling grounds alone."""
        with mock.patch.object(
            physics, "_rk4_frequency_phase_step", wraps=physics._rk4_frequency_phase_step
        ) as spy:
            with self.assertRaisesRegex(ValueError, "too small to resolve the"):
                physics.integrate_inspiral(
                    36.0, 29.0, 440.0, dt=1e-4, f_start=20.0,
                    include_ringdown=True, n_ringdown_tau=6, ringdown_pts=2,
                )
            spy.assert_not_called()

    def test_huge_integer_n_ringdown_tau_rd_pts_produce_clean_value_error(self):
        """P2-2 regression: a CLI-style huge Python int (401 digits, as in
        Codex's reproducer) previously escaped as a bare OverflowError from
        float() inside _require_finite; must now be one clean ValueError."""
        huge = 10 ** 400
        with self.assertRaises(ValueError):
            physics.integrate_inspiral(
                1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=huge, ringdown_pts=4000,
            )
        with self.assertRaises(ValueError):
            physics.integrate_inspiral(
                1.4, 1.4, 400.0, dt=2e-4, f_start=20.0,
                include_ringdown=True, n_ringdown_tau=6, ringdown_pts=huge,
            )

    def test_subnormal_dt_produces_clean_value_error(self):
        """P2-2 regression: --dt 5e-324 previously escaped as a bare
        OverflowError ("cannot convert float infinity to integer") from
        math.ceil(T_est / dt); must now be one clean ValueError."""
        with self.assertRaises(ValueError):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=5e-324, f_start=20.0)

    def test_subnormal_f_start_produces_clean_value_error(self):
        """P2-2 regression: --f_start 5e-324 previously escaped as a bare
        OverflowError ("Numerical result out of range") from
        f_start**(-8/3) inside inspiral_time(); must now be one clean
        ValueError."""
        with self.assertRaises(ValueError):
            physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=5e-324)

    def test_non_finite_rk4_state_raises_runtime_error_with_dt_hint(self):
        # Deterministically force the internal isfinite guard by making the
        # frequency derivative blow up, rather than hunting for organically
        # extreme physical inputs that also survive the earlier Nyquist and
        # step-count guards.
        with mock.patch.object(physics, "dfdt", return_value=np.inf):
            with self.assertRaisesRegex(RuntimeError, "non-finite.*smaller --dt"):
                physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=20.0)

    def test_non_increasing_frequency_raises_runtime_error_with_dt_hint(self):
        with mock.patch.object(physics, "dfdt", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "failed to increase GW frequency.*smaller --dt"):
                physics.integrate_inspiral(1.4, 1.4, 400.0, dt=2e-4, f_start=20.0)


# ===========================================================================
# driver_gw: plot-input validation (including the matching bool-rejection fix)
# ===========================================================================
class TestValidatePlotInputs(unittest.TestCase):
    def test_lw_must_be_positive(self):
        for value in (0.0, -0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    driver._validate_plot_inputs(None, None, value, 150)

    def test_dpi_must_be_a_positive_integer(self):
        for value in (0, -150, 150.5, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    driver._validate_plot_inputs(None, None, 0.4, value)

    def test_dpi_bool_is_rejected_not_silently_coerced(self):
        """Regression test: dpi=True previously passed float()/int() checks
        silently (dpi became 1) because bool is an int subclass."""
        with self.assertRaisesRegex(ValueError, "bool is not"):
            driver._validate_plot_inputs(None, None, 0.4, True)
        with self.assertRaisesRegex(ValueError, "bool is not"):
            driver._validate_plot_inputs(None, None, True, 150)

    def test_dpi_upper_bound_is_enforced(self):
        """P2-4 regression: --dpi previously had no upper bound at all, so
        an oversized (but not overflow-triggering) request could allocate a
        huge in-memory image before any error could be produced."""
        with self.assertRaisesRegex(ValueError, "dpi must not exceed 600"):
            driver._validate_plot_inputs(None, None, 0.4, driver.MAX_DPI + 1)
        # The documented ceiling itself must still be accepted.
        t_before, t_after, lw, dpi = driver._validate_plot_inputs(None, None, 0.4, driver.MAX_DPI)
        self.assertEqual(dpi, driver.MAX_DPI)

    def test_huge_integer_dpi_produces_clean_value_error(self):
        """P2-2 regression: a 401-digit --dpi previously escaped as a bare
        OverflowError from float() inside _finite_number; must now be one
        clean ValueError (and, independently, still rejected by MAX_DPI)."""
        with self.assertRaises(ValueError):
            driver._validate_plot_inputs(None, None, 0.4, 10 ** 400)

    def test_zoom_widths_must_be_non_negative_or_none(self):
        self.assertEqual(driver._validate_plot_inputs(None, None, 0.4, 150)[:2], (None, None))
        for value in (-0.1, math.nan, math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    driver._validate_plot_inputs(value, None, 0.4, 150)
                with self.assertRaises(ValueError):
                    driver._validate_plot_inputs(None, value, 0.4, 150)

    def test_valid_inputs_pass_through_normalized(self):
        t_before, t_after, lw, dpi = driver._validate_plot_inputs(0.2, 0.05, 0.4, 150.0)
        self.assertEqual((t_before, t_after, lw), (0.2, 0.05, 0.4))
        self.assertEqual(dpi, 150)
        self.assertIsInstance(dpi, int)


# ===========================================================================
# driver_gw.run: end-to-end smoke test
# ===========================================================================
class TestDriverRun(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_run_executes_and_returns_the_physics_result(self):
        import matplotlib.pyplot as plt
        with mock.patch.object(plt, "show"):
            result = driver.run(
                m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                dt=1e-4, f_start=20.0,
            )
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["model_version"], physics.MODEL_VERSION)

    def test_run_saves_png_in_addition_to_displaying_when_outdir_given(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plt, "show") as show:
                driver.run(
                    m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                    dt=1e-4, f_start=20.0, outdir=outdir,
                )
            show.assert_called_once_with()
            saved = list(Path(outdir).glob("*.png"))
            self.assertEqual(len(saved), 1)

    def test_run_saves_csv_in_addition_to_displaying_when_csvdir_given(self):
        """P2-1/Gemini regression: --csvdir gives students a no-programming
        route to the numerical arrays (e.g. for chirp-mass extraction).
        Audit2 (Codex P1-1 / Copilot A2-5, A2-6) additionally requires a
        commented metadata block above the header, and a fifth phase_rad
        column."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                result = driver.run(
                    m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
                    dt=2e-4, f_start=20.0, csvdir=csvdir,
                )
            saved = list(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 1)
            meta, rows = read_gw_csv(saved[0])
            self.assertEqual(rows[0], ["t_s", "f_hz", "A", "h", "phase_rad"])
            self.assertEqual(len(rows) - 1, result["summary"]["inspiral_steps"])
            self.assertEqual(float(rows[1][0]), 0.0)
            self.assertEqual(float(rows[1][1]), 20.0)
            # Audit3 (Codex P3-4): .17g is a round-trip-exact format for a
            # binary64 value (verified empirically: 100,000 random samples
            # across a wide magnitude range round-tripped with zero
            # mismatches), so parsed-back A/h values -- including strain
            # near 1e-23, where an absolute places= tolerance is meaningless
            # -- must compare exactly equal, not merely "almost".
            self.assertEqual(float(rows[-1][1]), result["f"][-1])
            self.assertEqual(float(rows[-1][2]), result["A"][-1])
            self.assertEqual(float(rows[-1][3]), result["h"][-1])
            self.assertEqual(float(rows[-1][4]), result["phase"][-1])
            # Metadata block: exact key set (Audit3 Codex P3-4 asked for the
            # *complete* set, not just a handful of fields) plus a
            # cross-check of every value against the run's own summary.
            self.assertEqual(
                set(meta.keys()),
                {
                    "model_version", "build_id", "export_timestamp_utc",
                    "m1_msun", "m2_msun", "d_mpc", "dt_s", "f_start_hz",
                    "include_ringdown", "n_ringdown_tau", "ringdown_pts",
                    "f_qnm_hz", "tau_qnm_ms", "ringdown_model", "columns",
                },
            )
            self.assertEqual(meta["model_version"], result["summary"]["model_version"])
            self.assertEqual(meta["build_id"], result["summary"]["build_id"])
            self.assertAlmostEqual(float(meta["m1_msun"]), 1.4, places=9)
            self.assertAlmostEqual(float(meta["m2_msun"]), 1.4, places=9)
            self.assertAlmostEqual(float(meta["d_mpc"]), 400.0, places=9)
            self.assertAlmostEqual(float(meta["dt_s"]), 2e-4, places=12)
            self.assertAlmostEqual(float(meta["f_start_hz"]), 20.0, places=9)
            self.assertEqual(meta["include_ringdown"], "False")
            self.assertEqual(meta["n_ringdown_tau"], "n/a")
            self.assertEqual(meta["ringdown_pts"], "n/a")
            self.assertEqual(meta["f_qnm_hz"], "n/a")
            self.assertEqual(meta["tau_qnm_ms"], "n/a")
            self.assertEqual(meta["ringdown_model"], "n/a")
            # export_timestamp_utc must be a genuinely parseable UTC instant
            # (Audit3 Codex P3-4), not merely a non-empty string.
            parsed = datetime.strptime(
                meta["export_timestamp_utc"], "%Y-%m-%dT%H:%M:%S.%fZ"
            )
            now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
            self.assertLessEqual(abs((now_utc_naive - parsed).total_seconds()), 120)

    def test_csv_metadata_distinguishes_runs_that_differ_only_by_dt(self):
        """Audit2 (Codex P1-1) specifically calls out EXP-8: four CSVs that
        differ only by --dt must remain individually identifiable once
        separated from the terminal that produced them."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                driver.run(m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                           dt=8e-4, f_start=20.0, csvdir=csvdir)
                driver.run(m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                           dt=4e-4, f_start=20.0, csvdir=csvdir)
            saved = sorted(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 2)
            dts = set()
            for path in saved:
                meta, _ = read_gw_csv(path)
                dts.add(meta["dt_s"])
            self.assertEqual(len(dts), 2)  # both files self-report a different dt_s

    def test_csv_blanks_ringdown_rows_with_no_inspiral_f_or_a(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                result = driver.run(
                    m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                    dt=1e-4, f_start=20.0, csvdir=csvdir,
                    include_ringdown=True, n_ringdown_tau=6, ringdown_pts=4000,
                )
            saved = list(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 1)
            meta, rows = read_gw_csv(saved[0])
            self.assertEqual(rows[0], ["t_s", "f_hz", "A", "h", "phase_rad"])
            data_rows = rows[1:]
            blank_f_rows = [row for row in data_rows if row[1] == ""]
            self.assertTrue(blank_f_rows)
            for row in blank_f_rows:
                self.assertEqual(row[2], "")  # A also blank on ringdown rows
                self.assertNotEqual(row[0], "")  # t always present
                self.assertNotEqual(row[3], "")  # h always present
                self.assertNotEqual(row[4], "")  # phase always present
            # Audit3 (Codex P3-4): exact metadata-key set, plus the QNM
            # fields the earlier version of this test never checked at all.
            self.assertEqual(
                set(meta.keys()),
                {
                    "model_version", "build_id", "export_timestamp_utc",
                    "m1_msun", "m2_msun", "d_mpc", "dt_s", "f_start_hz",
                    "include_ringdown", "n_ringdown_tau", "ringdown_pts",
                    "f_qnm_hz", "tau_qnm_ms", "ringdown_model", "columns",
                },
            )
            s = result["summary"]
            self.assertEqual(meta["include_ringdown"], "True")
            self.assertEqual(meta["n_ringdown_tau"], "6")
            self.assertEqual(meta["ringdown_pts"], "4000")
            self.assertAlmostEqual(float(meta["f_qnm_hz"]), s["f_qnm_hz"], places=6)
            self.assertAlmostEqual(float(meta["tau_qnm_ms"]), s["tau_qnm_ms"], places=6)
            self.assertIn("illustrative Schwarzschild QNM", meta["ringdown_model"])
            self.assertIn("not a physical merger", meta["ringdown_model"])

    def test_csv_timestamp_filename_has_microsecond_resolution(self):
        """P3-4 regression, driver-layer counterpart to the plot_gw check:
        the CSV filename must also carry microsecond resolution."""
        name = driver._timestamp_fname()
        self.assertRegex(name, r"^gw_inspiral_\d{8}_\d{6}_\d{6}\.csv$")

    def test_read_gw_csv_helper_rejects_a_duplicated_metadata_key(self):
        """Audit3 (Codex P3-4): read_gw_csv() must not silently let a
        repeated "# key: ..." line overwrite an earlier value -- that would
        hide a real _write_csv() regression (e.g. an accidentally
        duplicated meta_lines entry) from every other CSV test, since they
        only ever inspect the resulting dict. This directly tests the test
        helper's own contract with a synthetic file, independent of
        driver.run()."""
        with tempfile.TemporaryDirectory() as csvdir:
            bad_path = Path(csvdir) / "bad.csv"
            bad_path.write_text(
                "# model_version: 1.2.0\n"
                "# model_version: 1.3.0\n"
                "t_s,f_hz,A,h,phase_rad\n",
                encoding="utf-8",
            )
            with self.assertRaises(AssertionError):
                read_gw_csv(bad_path)

    def test_reserve_unique_path_does_not_overwrite_on_timestamp_collision(self):
        """Audit2 (Copilot A2-3): a repeated (e.g. mocked, or genuinely
        coincident) timestamp must never silently overwrite an earlier
        file -- _reserve_unique_path must claim a distinct "_1", "_2", ...
        suffixed name instead. Uses a frozen datetime.now() so the
        collision is deterministic rather than relying on real-clock
        timing."""
        fixed = driver.datetime(2026, 1, 1, 12, 0, 0, 123456)

        class FrozenDatetime(driver.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is None else fixed.astimezone(tz)

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(driver, "datetime", FrozenDatetime):
                p1 = driver._reserve_unique_path(d, "gw_inspiral", "csv")
                p2 = driver._reserve_unique_path(d, "gw_inspiral", "csv")
                p3 = driver._reserve_unique_path(d, "gw_inspiral", "csv")
            self.assertEqual(len({p1, p2, p3}), 3)  # three distinct paths
            for p in (p1, p2, p3):
                self.assertTrue(os.path.exists(p))
            self.assertEqual(len(list(Path(d).glob("*.csv"))), 3)  # nothing overwritten

    def test_frozen_timestamp_two_complete_writes_both_survive_intact(self):
        """Audit3 (Codex P2-1, required regression #5): the collision test
        above only proves _reserve_unique_path() claims three distinct
        empty placeholders. This proves the stronger claim that matters --
        two full, real _write_csv() runs that land on the exact same frozen
        timestamp each publish their own complete, distinguishable content
        rather than one clobbering the other or either being left
        truncated by the atomic-publish rename."""
        fixed = driver.datetime(2026, 1, 1, 12, 0, 0, 123456)

        class FrozenDatetime(driver.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is None else fixed.astimezone(tz)

        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(driver, "datetime", FrozenDatetime):
                with mock.patch.object(plt, "show"):
                    driver.run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
                               dt=2e-4, f_start=20.0, csvdir=csvdir)
                    driver.run(m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                               dt=2e-4, f_start=20.0, csvdir=csvdir)
            saved = sorted(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 2)  # "_1" suffix, not an overwrite
            m1_values = set()
            for path in saved:
                meta, rows = read_gw_csv(path)
                self.assertGreater(len(rows), 1)  # a real header + data rows
                m1_values.add(meta["m1_msun"])
            self.assertEqual(len(m1_values), 2)  # both runs' content survived distinctly

    def test_csv_write_failure_leaves_no_csv_behind(self):
        """Audit3 (Codex P2-1 / Copilot A3-2, required regression): inject a
        mid-write failure inside the csv.writer and confirm the atomic
        publish-or-cleanup path removes the temporary file and the
        pre-reserved placeholder, leaving zero .csv files -- not the
        empty/partial file Audit3 originally reported."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                with mock.patch.object(
                    driver.csv, "writer", side_effect=OSError("simulated disk error")
                ):
                    with self.assertRaises(OSError):
                        driver.run(m1_msun=1.4, m2_msun=1.4, d_mpc=400.0,
                                   dt=2e-4, f_start=20.0, csvdir=csvdir)
            self.assertEqual(list(Path(csvdir).glob("*.csv")), [])
            self.assertEqual(list(Path(csvdir).glob("*.csv.tmp")), [])
            self.assertEqual(list(Path(csvdir).iterdir()), [])

    def test_outdir_as_existing_regular_file_leaves_no_csv_behind(self):
        """Audit3 (Codex P2-1) exact reproducer: a valid --csvdir plus an
        --outdir path that is already an existing regular file (so
        os.makedirs(outdir) fails with FileExistsError). The early
        directory pre-validation in driver.run() now checks/creates
        --csvdir *and* --outdir before writing either artifact, so this
        fails before the CSV write is ever attempted -- no CSV is left
        behind. (The csvdir directory itself may still be created as a
        side effect of the pre-validation step, same as a bare
        os.makedirs(csvdir) would do; what must never happen is a
        completed CSV export sitting next to a hard failure caused by an
        unrelated, later plotting mistake -- the surprising behavior this
        reproducer originally exposed.)"""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmp:
            csvdir = os.path.join(tmp, "csvout")
            outdir_as_file = os.path.join(tmp, "not_a_directory")
            with open(outdir_as_file, "w", encoding="utf-8") as handle:
                handle.write("occupied")
            with mock.patch.object(plt, "show"):
                with self.assertRaises(OSError):
                    driver.run(m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                               dt=1e-4, f_start=20.0,
                               csvdir=csvdir, outdir=outdir_as_file)
            self.assertEqual(
                list(Path(csvdir).glob("*.csv")) if os.path.isdir(csvdir) else [], []
            )

    def test_no_csvdir_does_not_write_a_csv_file(self):
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                driver.run(m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                           dt=1e-4, f_start=20.0)
            self.assertEqual(list(Path(csvdir).glob("*.csv")), [])

    def test_empty_zoom_window_leaves_no_csv_behind(self):
        """Audit2 (Codex P3-5): a request whose zoom window is empty
        (--t_before 0 --t_after 0, which collapses hi<=lo once t_isco is
        not exactly 0 or the data endpoint) must fail all-or-nothing -- a
        non-zero exit and *zero* files written, not a CSV left behind by a
        write that happened before the (later) plot-window check failed."""
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as csvdir:
            with mock.patch.object(plt, "show"):
                with self.assertRaises(ValueError):
                    driver.run(
                        m1_msun=36.0, m2_msun=29.0, d_mpc=440.0,
                        dt=1e-4, f_start=20.0, csvdir=csvdir,
                        t_before=0.0, t_after=0.0,
                    )
            self.assertEqual(list(Path(csvdir).glob("*.csv")), [])


# ===========================================================================
# plot_gw: plot-facing data and labels
# ===========================================================================
def _synthetic_result(include_ringdown=False):
    t_insp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    f_insp = np.array([20.0, 30.0, 45.0, 60.0, 67.6])
    A_insp = np.array([1e-22, 1.1e-22, 1.3e-22, 1.6e-22, 2.0e-22])
    phase = np.linspace(0.0, 8 * np.pi, 5)
    h_insp = A_insp * np.cos(phase)
    summary = dict(
        m1_msun=36.0, m2_msun=29.0, Mc_msun=28.096, d_mpc=440.0,
        f_start_hz=20.0, f_isco_hz=67.6, include_ringdown=include_ringdown,
        f_qnm_hz=195.5, tau_qnm_ms=3.418,
    )
    if include_ringdown:
        t_rd = np.array([1.0005, 1.001, 1.002])
        h_rd = np.array([1.9e-22, 1.2e-22, 0.5e-22])
        t = np.concatenate([t_insp, t_rd])
        h = np.concatenate([h_insp, h_rd])
        A = np.concatenate([A_insp, [np.nan, np.nan, np.nan]])
        f = np.concatenate([f_insp, [np.nan, np.nan, np.nan]])
    else:
        t, h, A, f = t_insp, h_insp, A_insp, f_insp
    return dict(t=t, h=h, A=A, f=f, t_isco=1.0, f_isco_hz=67.6, summary=summary)


class TestPlotting(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def _render_and_capture(self, result, **kwargs):
        """Render result and return the Figure object, captured at the
        moment plt.show() is invoked. plot_inspiral() unconditionally calls
        plt.close(fig) right after plt.show(), so plt.gcf() called *after*
        plot_inspiral() returns would hand back a fresh, empty figure
        instead -- the Figure/Axes Python objects remain fully inspectable
        after close(), only pyplot's own figure registry entry is dropped.
        """
        import matplotlib.pyplot as plt
        captured = {}

        def _capture():
            captured["fig"] = plt.gcf()

        with mock.patch.object(plt, "show", side_effect=_capture) as show:
            plotting.plot_inspiral(result, **kwargs)
        show.assert_called_once_with()
        self.assertIn("fig", captured, "plt.show() was never called")
        return captured["fig"]

    def test_panel_titles_labels_and_legend_pure_inspiral(self):
        result = _synthetic_result(include_ringdown=False)
        fig = self._render_and_capture(result)
        self.assertEqual(len(fig.axes), 3)
        ax1, ax2, ax3 = fig.axes
        self.assertEqual(ax1.get_ylabel(), "Strain scale $h(t)$ (dimensionless)")
        self.assertEqual(ax2.get_ylabel(), "Amplitude $A(t)$ (dimensionless)")
        self.assertEqual(ax3.get_xlabel(), "Time $t$ [s]")
        self.assertEqual(ax3.get_ylabel(), "GW frequency [Hz]")
        legend1 = [text.get_text() for text in ax1.get_legend().get_texts()]
        self.assertIn("Inspiral strain scale $h(t)$", legend1)
        self.assertNotIn("Illustrative Schwarzschild QNM", legend1)

    def test_ringdown_segment_is_drawn_in_distinct_color_and_legend_entry(self):
        result = _synthetic_result(include_ringdown=True)
        fig = self._render_and_capture(result)
        ax1 = fig.axes[0]
        legend1 = [text.get_text() for text in ax1.get_legend().get_texts()]
        self.assertIn("Illustrative Schwarzschild QNM", legend1)
        rd_lines = [ln for ln in ax1.lines if ln.get_label() == "Illustrative Schwarzschild QNM"]
        self.assertEqual(len(rd_lines), 1)
        self.assertEqual(rd_lines[0].get_color(), plotting.C_RD)

    def test_envelope_excludes_nan_ringdown_samples(self):
        result = _synthetic_result(include_ringdown=True)
        fig = self._render_and_capture(result)
        ax1 = fig.axes[0]
        envelope_lines = [ln for ln in ax1.lines if "Envelope" in ln.get_label()]
        self.assertEqual(len(envelope_lines), 1)
        # Envelope x-data must never include a ringdown (NaN-amplitude) sample.
        self.assertTrue(np.all(np.isfinite(envelope_lines[0].get_xdata())))
        self.assertEqual(len(envelope_lines[0].get_xdata()), 5)  # inspiral-only count

    def test_isco_cutoff_line_and_annotation_present(self):
        result = _synthetic_result(include_ringdown=True)
        fig = self._render_and_capture(result)
        ax1 = fig.axes[0]
        cutoff_lines = [ln for ln in ax1.lines if "ISCO cutoff" in (ln.get_label() or "")]
        self.assertEqual(len(cutoff_lines), 1)
        self.assertEqual(cutoff_lines[0].get_xdata()[0], 1.0)
        texts = [t.get_text() for t in ax1.texts]
        self.assertTrue(any("QNM" in t for t in texts))

    def test_ringdown_annotation_states_not_a_physical_merger(self):
        """A3/A4 regression: the plot itself (not only the Help file) must
        carry a prominent reminder that the ringdown is illustrative, so a
        viewer of the saved PNG alone (without the Help text) is not misled
        into reading it as a physical merger-ringdown continuation."""
        result = _synthetic_result(include_ringdown=True)
        fig = self._render_and_capture(result)
        ax1 = fig.axes[0]
        texts = " ".join(t.get_text() for t in ax1.texts)
        self.assertIn("illustrative ringdown", texts)
        self.assertIn("not a physical merger", texts)

    def test_no_ringdown_annotation_when_ringdown_not_included(self):
        result = _synthetic_result(include_ringdown=False)
        fig = self._render_and_capture(result)
        ax1 = fig.axes[0]
        texts = " ".join(t.get_text() for t in ax1.texts)
        self.assertNotIn("illustrative ringdown", texts)

    def test_png_timestamp_filename_has_microsecond_resolution(self):
        """P3-4 regression: the previous second-resolution timestamp let two
        rapid programmatic saves to the same --outdir collide and silently
        overwrite each other; the filename must now carry a 6-digit
        microsecond field."""
        name = plotting._timestamp_fname()
        self.assertRegex(name, r"^gw_inspiral_\d{8}_\d{6}_\d{6}\.png$")

    def test_reserve_unique_path_does_not_overwrite_on_timestamp_collision(self):
        """Audit2 (Copilot A2-3), plot_gw-layer counterpart to the
        driver_gw test of the same name: a repeated timestamp must claim a
        distinct "_1", "_2", ... suffixed PNG path rather than silently
        overwriting an earlier one."""
        fixed = plotting.datetime(2026, 1, 1, 12, 0, 0, 123456)

        class FrozenDatetime(plotting.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed if tz is None else fixed.astimezone(tz)

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(plotting, "datetime", FrozenDatetime):
                p1 = plotting._reserve_unique_path(d)
                p2 = plotting._reserve_unique_path(d)
                p3 = plotting._reserve_unique_path(d)
            self.assertEqual(len({p1, p2, p3}), 3)
            for p in (p1, p2, p3):
                self.assertTrue(os.path.exists(p))
            self.assertEqual(len(list(Path(d).glob("*.png"))), 3)

    def test_savefig_failure_leaves_no_png_behind(self):
        """Audit3 (Codex P2-1 / Copilot A3-3, required regression): patch
        Figure.savefig to raise mid-write and confirm the atomic
        publish-or-cleanup path removes the temporary file and the
        pre-reserved placeholder -- zero .png files remain, not the
        zero-byte placeholder Audit3 originally reported. Also confirms
        the figure is still closed (finally block) despite the failure."""
        import matplotlib.figure
        import matplotlib.pyplot as plt
        result = _synthetic_result(include_ringdown=False)
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(
                matplotlib.figure.Figure, "savefig",
                side_effect=OSError("simulated disk error"),
            ):
                with self.assertRaises(OSError):
                    plotting.plot_inspiral(result, outdir=outdir)
            self.assertEqual(list(Path(outdir).glob("*.png")), [])
            self.assertEqual(list(Path(outdir).glob("*.png.tmp")), [])
            self.assertEqual(list(Path(outdir).iterdir()), [])
        # No figure leaked into pyplot's registry despite the exception.
        self.assertEqual(set(plt.get_fignums()), before)

    def test_show_failure_still_closes_the_figure_after_a_completed_png(self):
        """Audit3 (Codex P2-1, required regression): if plt.show() itself
        raises -- e.g. a headless/broken display -- after a --outdir PNG
        has already been completely and atomically saved, that saved PNG
        must survive (the failure is unrelated to the save), and the
        finally block must still close the figure so no live figure leaks
        into pyplot's registry."""
        import matplotlib.pyplot as plt
        result = _synthetic_result(include_ringdown=False)
        before = set(plt.get_fignums())
        with tempfile.TemporaryDirectory() as outdir:
            with mock.patch.object(plt, "show", side_effect=RuntimeError("no display")):
                with self.assertRaises(RuntimeError):
                    plotting.plot_inspiral(result, outdir=outdir)
            saved = list(Path(outdir).glob("*.png"))
            self.assertEqual(len(saved), 1)  # the completed save is not rolled back
            self.assertGreater(saved[0].stat().st_size, 0)
        self.assertEqual(set(plt.get_fignums()), before)

    def test_zoom_interval_rejects_an_empty_window(self):
        # A window fully outside the data range collapses to empty (hi<=lo):
        with self.assertRaisesRegex(ValueError, "no plotted data"):
            plotting._xlim(t_isco=100.0, t_before=1.0, t_after=1.0, t_min=0.0, t_max=2.0)
        with self.assertRaisesRegex(ValueError, "no plotted data"):
            plotting._xlim(t_isco=1.0, t_before=None, t_after=-2.5, t_min=0.0, t_max=2.0)

    def test_zoom_interval_clamps_to_available_data(self):
        lo, hi = plotting._xlim(t_isco=1.0, t_before=10.0, t_after=10.0, t_min=0.0, t_max=2.0)
        self.assertEqual((lo, hi), (0.0, 2.0))


# ===========================================================================
# Command-line interface
# ===========================================================================
class TestCLI(unittest.TestCase):
    def _run(self, args, timeout=30):
        environment = os.environ.copy()
        environment["MPLBACKEND"] = "Agg"
        return subprocess.run(
            [sys.executable, "main.py", *args],
            cwd=MODULE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def test_default_run_smoke_test(self):
        result = self._run([], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"GravitationalWaveSources {physics.MODEL_VERSION}", result.stdout)
        self.assertIn(f"(build {physics.BUILD_ID})", result.stdout)
        self.assertIn("Chirp mass          : 1.2188", result.stdout)
        self.assertIn("ISCO cutoff         : 1570.0", result.stdout)
        self.assertIn("Time to ISCO        : 157.800", result.stdout)
        self.assertIn("Strain scale at ISCO: 5.584e-23", result.stdout)
        self.assertIn("Inspiral samples    : 789,004", result.stdout)
        self.assertIn("Ringdown            : not included", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_ringdown_run_smoke_test(self):
        result = self._run(
            ["--m1", "36", "--m2", "29", "--d", "440", "--ringdown",
             "--t_before", "0.2", "--t_after", "0.05"],
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("QNM frequency       : 195.5", result.stdout)
        self.assertIn("QNM decay time      : 3.418", result.stdout)

    def test_invalid_inputs_produce_single_clean_error_line_no_traceback(self):
        cases = {
            ("--m1", "-1.0"): "Component masses must both be greater than zero.",
            ("--d", "0"): "Luminosity distance d_mpc must be greater than zero.",
            ("--dt", "0"): "Integration timestep dt must be greater than zero.",
            ("--f_start", "2000"): "must be below the Schwarzschild ISCO",
            ("--dt", "50"): "too large to sample the waveform",
            ("--rd_pts", "20000000", "--ringdown"): "must not exceed 500,000",
            ("--dpi", "0"): "dpi must be a positive integer.",
            ("--lw", "-1"): "lw must be greater than zero.",
            ("--dpi", "601"): "dpi must not exceed 600",
            ("--ringdown", "--n_tau", "6", "--rd_pts", "2",
             "--m1", "36", "--m2", "29", "--d", "440",
             "--dt", "1e-4"): "too small to resolve the",
            ("--n_tau", "1" + "0" * 400, "--ringdown"): None,
            ("--rd_pts", "1" + "0" * 400, "--ringdown"): None,
            ("--dpi", "1" + "0" * 400): None,
            ("--dt", "5e-324",): None,
            ("--f_start", "5e-324",): None,
        }
        for args, expected_fragment in cases.items():
            with self.subTest(args=args):
                result = self._run(list(args), timeout=15)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stderr)
                stderr_lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
                self.assertEqual(len(stderr_lines), 1)
                self.assertTrue(stderr_lines[0].startswith("GravitationalWaveSources: "))
                if expected_fragment is not None:
                    self.assertIn(expected_fragment, stderr_lines[0])

    def test_dpi_non_integer_is_rejected_by_argparse_itself(self):
        result = self._run(["--dpi", "150.5"], timeout=15)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid int value", result.stderr)

    def test_outdir_saves_and_still_displays(self):
        with tempfile.TemporaryDirectory() as outdir:
            result = self._run(
                ["--m1", "36", "--m2", "29", "--d", "440", "--dt", "1e-4",
                 "--outdir", outdir],
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PNG saved ->", result.stdout)
            self.assertIn("Displaying figure on screen", result.stdout)
            saved = list(Path(outdir).glob("*.png"))
            self.assertEqual(len(saved), 1)

    def test_no_outdir_does_not_write_a_file(self):
        with tempfile.TemporaryDirectory() as outdir:
            result = self._run(["--m1", "36", "--m2", "29", "--d", "440",
                                 "--dt", "1e-4"], timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("PNG saved", result.stdout)
            self.assertEqual(list(Path(outdir).glob("*.png")), [])

    def test_csvdir_saves_a_csv_end_to_end(self):
        with tempfile.TemporaryDirectory() as csvdir:
            result = self._run(
                ["--m1", "36", "--m2", "29", "--d", "440", "--dt", "1e-4",
                 "--csvdir", csvdir],
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CSV saved ->", result.stdout)
            saved = list(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 1)
            lines = saved[0].read_text(encoding="utf-8").splitlines()
            # Audit2 (Codex P1-1 / Copilot A2-5): a commented metadata block
            # now precedes the header row.
            self.assertTrue(lines[0].startswith("#"))
            header_lines = [ln for ln in lines if not ln.startswith("#")]
            self.assertEqual(header_lines[0], "t_s,f_hz,A,h,phase_rad")
            meta_text = "\n".join(ln for ln in lines if ln.startswith("#"))
            self.assertIn(f"model_version: {physics.MODEL_VERSION}", meta_text)
            self.assertIn(f"build_id: {physics.BUILD_ID}", meta_text)
            self.assertIn("dt_s:", meta_text)

    def test_exp7_revised_command_produces_a_small_csv_end_to_end(self):
        """Audit3 (Codex P2-2, recommended regression): the revised EXP-7
        data-generation command (--f_start 100, in place of the old
        default --f_start 20 which produces a 789,004-row/~78 MiB export)
        must actually produce a small file end-to-end through the real
        CLI, and the exact two-point method described in the Help snippet
        must recover 1.2188 from rows this real run actually wrote --
        not merely from a direct physics_gw.integrate_inspiral() call."""
        with tempfile.TemporaryDirectory() as csvdir:
            result = self._run(["--f_start", "100", "--csvdir", csvdir], timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = list(Path(csvdir).glob("*.csv"))
            self.assertEqual(len(saved), 1)
            meta, rows = read_gw_csv(saved[0])
            data_rows = rows[1:]
            # A generous ceiling: comfortably above the documented ~10,788
            # rows, but two orders of magnitude below the old 789,004-row
            # default-f_start export this exercise used to require.
            self.assertLess(len(data_rows), 50_000)
            self.assertLess(saved[0].stat().st_size, 5 * 1024 * 1024)  # < 5 MiB
            t1, f1 = float(data_rows[0][0]), float(data_rows[0][1])
            t2, f2 = float(data_rows[10][0]), float(data_rows[10][1])
            fdot_est = (f2 - f1) / (t2 - t1)
            f_mid = 0.5 * (f1 + f2)
            Mc_kg = physics.chirp_mass_from_fdot(f_mid, fdot_est)
            self.assertEqual(f"{Mc_kg / physics.M_sun:.4f}", "1.2188")

    def test_n_tau_huge_value_fails_cleanly_end_to_end(self):
        """Audit2 (Codex P2-2) reproducer, CLI-level: --n_tau is parsed as
        argparse type=int, so the reachable huge value is a huge digit
        string (as a real user might paste), not scientific notation
        (which argparse's int() rejects earlier, with its own clean
        usage/error message -- a different, already-covered case). This
        huge-but-valid Python int previously reached math.ceil(inf) inside
        physics_gw and raised an uncaught OverflowError; it must now
        produce a single clean error line and a non-zero exit, never a
        traceback."""
        result = self._run(
            ["--m1", "36", "--m2", "29", "--d", "440", "--dt", "1e-4",
             "--ringdown", "--n_tau", "1" + "0" * 308],
            timeout=15,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        stderr_lines = [ln for ln in result.stderr.splitlines() if ln.strip()]
        self.assertEqual(len(stderr_lines), 1)
        self.assertIn("too large", stderr_lines[0])

    def test_n_tau_out_of_range_is_ignored_without_ringdown_end_to_end(self):
        """Audit2 (Copilot A2-4): an out-of-range --n_tau has no effect
        when --ringdown is not requested."""
        result = self._run(
            ["--m1", "1.4", "--m2", "1.4", "--d", "400", "--dt", "2e-4",
             "--n_tau", "0"],
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_dpi_at_max_dpi_is_accepted_end_to_end(self):
        with tempfile.TemporaryDirectory() as outdir:
            result = self._run(
                ["--m1", "36", "--m2", "29", "--d", "440", "--dt", "1e-4",
                 "--outdir", outdir, "--dpi", "600"],
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(list(Path(outdir).glob("*.png"))), 1)


# ===========================================================================
# Help file
# ===========================================================================
class TestHelpFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = MODULE_DIR / HELP_FILE
        cls.html = cls.path.read_text(encoding="utf-8")
        parser = HtmlTreeParser()
        parser.feed(cls.html)
        parser.close()
        cls.root = parser.root

    def test_help_file_exists(self):
        self.assertTrue(self.path.is_file())

    def test_version_and_build_match_program(self):
        version_nodes = nodes_by_id(self.root, "version_build")
        self.assertEqual(len(version_nodes), 1)
        self.assertEqual(version_nodes[0].tag, "p")
        self.assertEqual(
            normalized_text(version_nodes[0]),
            f"Version {physics.MODEL_VERSION} Build {physics.BUILD_ID}",
        )

    def test_build_id_provenance_wording_does_not_overstate_exact_bytes(self):
        """Audit2 (Codex P3-2 / Copilot A2-1): BUILD_ID is computed from
        UTF-8 source *text* after universal-newline (CRLF/CR -> LF)
        normalization, not from the literal on-disk bytes -- a CRLF and an
        LF copy of the same source intentionally hash identically. The
        Help text previously (incorrectly) said "exact bytes"; it must not
        say that, and must describe the normalization instead."""
        self.assertNotIn("exact bytes", self.html)
        self.assertIn("normaliz", self.html.lower())
        provenance_idx = self.html.find("Build identifier provenance")
        self.assertNotEqual(provenance_idx, -1)
        # The normalization wording should appear near the provenance note
        # itself, not merely somewhere else on the page.
        nearby = self.html[provenance_idx:provenance_idx + 1200]
        self.assertIn("normaliz", nearby.lower())
        self.assertIn("CRLF", nearby)

    def test_default_case_numbers_are_current(self):
        for required in ("158 seconds",):
            with self.subTest(required=required):
                self.assertIn(required, self.html)

    def test_parameter_table_matches_argparse_defaults(self):
        parameter_sections = nodes_by_id(self.root, "parameters")
        self.assertEqual(len(parameter_sections), 1)
        tables = descendants(
            parameter_sections[0], lambda node: has_class(node, "param-table")
        )
        self.assertEqual(len(tables), 1)
        rows = {}
        for row in descendants(tables[0], lambda node: node.tag == "tr"):
            cells = [normalized_text(cell) for cell in descendants(row, lambda n: n.tag == "td")]
            if cells:
                rows[cells[0]] = cells[1:]

        argparse_defaults = main_argparse_defaults(MODULE_DIR)
        expected = {
            "--m1": 1.4, "--m2": 1.4, "--d": 400.0, "--dt": 2e-4,
            "--f_start": 20.0, "--ringdown": False, "--n_tau": 6,
            "--rd_pts": 4000, "--t_before": None, "--t_after": None,
            "--lw": 0.4, "--dpi": 150, "--outdir": None, "--csvdir": None,
        }
        self.assertEqual(argparse_defaults, expected)

        self.assertEqual(float(rows["--m1"][0]), argparse_defaults["--m1"])
        self.assertEqual(float(rows["--m2"][0]), argparse_defaults["--m2"])
        self.assertEqual(float(rows["--d"][0]), argparse_defaults["--d"])
        self.assertEqual(float(rows["--dt"][0]), argparse_defaults["--dt"])
        self.assertEqual(float(rows["--f_start"][0]), argparse_defaults["--f_start"])
        self.assertEqual(int(rows["--n_tau"][0]), argparse_defaults["--n_tau"])
        self.assertEqual(int(rows["--rd_pts"][0]), argparse_defaults["--rd_pts"])
        self.assertEqual(float(rows["--lw"][0]), argparse_defaults["--lw"])
        self.assertEqual(int(rows["--dpi"][0]), argparse_defaults["--dpi"])

    def test_help_states_python_version_requirement(self):
        """P3-3 regression: the Help previously never told students which
        Python version is required, even though the test suite already
        confirmed 3.10 syntax compatibility."""
        self.assertIn("Python 3.10", self.html)

    def test_help_documents_ringdown_samples_per_cycle_requirement(self):
        """P1-1 regression: the parameter table and safeguards note
        previously described --rd_pts as valid over "2 through 500,000"
        alone, with no mention that --n_tau and --rd_pts are jointly
        constrained by the QNM sampling requirement."""
        self.assertIn("samples per QNM", self.html)

    def test_help_documents_dpi_upper_bound(self):
        """P2-4 regression: the Help previously described --dpi as simply
        "a positive integer" with no documented upper bound."""
        self.assertIn("600", self.html)
        params_text = normalized_text(nodes_by_id(self.root, "parameters")[0])
        self.assertIn("1 through 600", params_text)

    def test_help_documents_csvdir(self):
        """P2-1/P2-3/Gemini regression: --csvdir must be documented as a
        parameter and referenced from the chirp-mass-extraction and
        convergence exercises."""
        self.assertIn("--csvdir", self.html)

    def test_csv_schema_wording_is_synchronized_across_cli_help_and_html(self):
        """Audit3 (Codex P2-3 / Copilot A3-1): the module docstring, the
        live "python main.py --help" text, and this Help file all
        previously described the CSV export as a bare four-column
        "t, f, A, h" table, while the real writer emits a commented
        metadata preamble plus a five-column t_s,f_hz,A,h,phase_rad table.
        This must never regress on any of the three student-facing
        surfaces at once."""
        proc = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=MODULE_DIR, capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        help_text = proc.stdout
        # Every "t_s,f_hz,A,h" column-list mention must be the full,
        # current five-column form -- never the stale bare four-column one.
        bare_four_column = re.compile(r"t_s,f_hz,A,h(?!,phase_rad)")
        for label, text in (
            ("live --help", help_text),
            ("main.py source", (MODULE_DIR / "main.py").read_text(encoding="utf-8")),
            ("Help HTML", self.html),
        ):
            with self.subTest(surface=label):
                self.assertIn("phase_rad", text)
                self.assertNotIn("t, f, A, h", text)
                self.assertNotIn("CSV of t, f, A, h", text)
                self.assertIsNone(
                    bare_four_column.search(text),
                    f"{label} still describes the stale 4-column CSV schema",
                )

    def test_exp7_pre_snippet_executes_and_prints_documented_result(self):
        """Extract the exact copyable Python snippet from the EXP-7 <pre>
        block and actually run it (against the real installed physics_gw),
        confirming it prints the documented 1.2188 -- rather than merely
        trusting that the HTML text and the real behavior still agree."""
        match = re.search(r"<pre>(import physics_gw as gw.*?)</pre>", self.html, re.S)
        self.assertIsNotNone(match, "EXP-7 <pre> snippet not found in Help file")
        snippet = html.unescape(match.group(1))
        self.assertIn("chirp_mass_from_fdot", snippet)
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=MODULE_DIR, capture_output=True, text=True, timeout=15, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "1.2188")

    def test_help_documents_mathjax_connectivity_plainly(self):
        self.assertIn("cdn.jsdelivr.net/npm/mathjax@3", self.html)
        self.assertIn("an internet connection is needed", self.html)
        self.assertNotIn("navigator.onLine", self.html)

    def test_envelope_ringdown_scope_is_clarified(self):
        output_text = normalized_text(nodes_by_id(self.root, "output")[0])
        self.assertIn("stops exactly at the ISCO cutoff", output_text)

    def test_algorithm_section_explains_time_to_isco_convergence_limit(self):
        algorithm_text = normalized_text(nodes_by_id(self.root, "algorithm")[0])
        self.assertIn("fourth order", algorithm_text)
        self.assertIn("time to isco", algorithm_text.lower())

    def test_exercise_cards_have_expected_rank_and_title(self):
        experiment_section = nodes_by_id(self.root, "experiments")
        self.assertEqual(len(experiment_section), 1)
        cards = descendants(
            experiment_section[0], lambda node: has_class(node, "exp-card")
        )
        actual = []
        for card in cards:
            num = descendants(card, lambda n: has_class(n, "ec-num"))
            title = descendants(card, lambda n: n.tag == "h4")
            self.assertEqual(len(num), 1)
            self.assertEqual(len(title), 1)
            actual.append((normalized_text(num[0]), normalized_text(title[0])))

        expected = [
            ("EXP-1 · introductory", "Identify the Chirp"),
            ("EXP-2 · introductory", "Verify Inverse-Distance Scaling"),
            ("EXP-3 · introductory/intermediate", "Starting Frequency and Time in Band"),
            ("EXP-4 · intermediate", "Chirp Mass Controls the Sweep"),
            ("EXP-5 · intermediate", "Mass Ratio at Fixed Total Mass"),
            ("EXP-6 · intermediate", "Test the Inspiral-Time Power Law"),
            ("EXP-7 · intermediate", "Extract Chirp Mass from the Frequency Sweep"),
            ("EXP-8 · intermediate/advanced", "Numerical Convergence"),
            ("EXP-9 · advanced", "Illustrative Black-Hole Ringdown"),
            ("EXP-10 · advanced", "QNM Scaling with Remnant Mass"),
            ("EXP-11 · synthesis", "Map the Model's Domain of Validity"),
        ]
        self.assertEqual(actual, expected)

    def test_all_internal_navigation_targets_exist_and_ids_are_unique(self):
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        counts = Counter(ids)
        self.assertFalse({name: count for name, count in counts.items() if count > 1})
        targets = [
            target
            for target in re.findall(r'href="#([^"]+)"', self.html)
            if not target.startswith("$")
        ]
        self.assertTrue(targets)
        for target in targets:
            with self.subTest(target=target):
                self.assertIn(target, counts)

    def test_no_tab_characters(self):
        self.assertNotIn("\t", self.html)

    def test_no_review_or_audit_history_leaked_into_student_help(self):
        for phrase in ("Claude", "Copilot", "Gemini", "ChatGPT", "Codex",
                       "Critique", "Audit1", "Kickoff"):
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.html)

    def test_module_overview_cards_describe_actual_responsibilities(self):
        module_section = nodes_by_id(self.root, "modules")[0]
        cards = descendants(module_section, lambda n: has_class(n, "module-card"))
        names = [normalized_text(descendants(c, lambda n: has_class(n, "mc-name"))[0])
                  for c in cards]
        self.assertEqual(set(names), set(CORE_MODULE_FILES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
