"""
test_physics_cosmo.py
======================
Automated regression suite for Cosmology_expansion_simulator.

Coverage, by category:

  * Analytic closed-form ages (Einstein-de Sitter, radiation-only, Milne,
    flat LambdaCDM) as a baseline sanity check on ordinary integration.
  * Event-precedence at a turning point: a_max and t_max must behave as
    hard stopping bounds, including the boundary cases where a_max or
    t_max sits exactly at, or extremely close to, a genuine turnaround
    -- the numerically hardest regime, since da/dt ~ sqrt(a_turn - a)
    has an infinite derivative there and a naive large integration step
    cannot land on it safely.
  * A non-monotonic E(a)^2 with two positive roots: the physical
    expanding-branch solution must stop at the FIRST root, not be waved
    through because a later point happens to test non-negative again.
  * A parameterized sweep of near-double-root separations, from
    comfortably wide down to a razor-thin fraction of the resolution a
    fixed-grid scan could ever see, and the complementary "confirmed no
    root" case just past the critical separation (a loitering model)
    integrating cleanly to a_max with no discarded trajectory gap.
  * Boundary coincidence: a_max is a HARD numeric bound (max(a) <=
    a_max) at every offset from an exact analytic turning point,
    without exception. At a genuine (non-trivial) offset the label is
    deterministically 'a_max'; at literally one ULP below the turning
    point, which label results depends on which side of the true root
    the independent bisection for a_turn happens to converge -- an
    implementation detail this program does not promise to control --
    so only the hard numeric bound is asserted there, not a specific
    label. (An earlier revision instead reclassified a coincidental
    a_max/a_turn tie as 'turnaround' using a floating-point tolerance,
    which could return an a_next slightly PAST the literal a_max the
    caller supplied -- a strictly worse defect than the label
    inconsistency it was meant to smooth over; the tolerance was
    removed for exactly this reason.)
  * Extreme/pathological CPL (w0, wa) parameters, which must raise a
    clear domain error rather than silently overflow to NaN/inf and
    have that non-finite value mistaken for a physical turning point.
  * The H(t) sign flip on the mirrored contracting branch.
  * Input validation (t_max <= t_i, excessive step counts, non-positive
    age_ref_gyr) and CSV output (headers, rows, and provenance/result
    metadata, including the resolved-parameter fields and model version
    added for Audit 5).
  * That the module's internal _Overshoot control-flow exception never
    escapes the public API.
  * Tri-state root-search semantics added for Audit 5: the certified
    search must raise _IndeterminateRootSearch (never silently return
    "no root") both when its evaluation budget is exhausted and when a
    sub-interval is bisected down to float64's own subdivision limit,
    and that raise must propagate as a genuine failure -- not a
    quietly-assumed "safe" or "no recollapse" result -- through evolve,
    compare, and age-scan mode alike.
  * Randomized/property-based verification that the analytic Lipschitz
    bound on |dE(a)^2/da| actually dominates dense sampling of the
    exact derivative, including nonzero wa with an interior CPL
    extremum, and a direct check that every bracket the certified
    search returns satisfies E(a_lo)^2>=0, E(a_hi)^2<0.
  * A Big-Rip-vs-recollapse precedence regression using the exact
    reproducer from Codex Audit 5 (a recollapse beyond the requested
    a_max must suppress a reported Big Rip), alongside a genuine flat-
    phantom case confirming the stricter check still reports a real
    Big Rip when one is actually present.
  * Interior-trajectory accuracy (not just gap size) for the exact
    matter-only closed-universe cycloid at several times approaching
    the turnaround, and a dedicated large-time-gap check for a TRUE
    near-double-root turnaround.
  * A general sweep asserting the a_max/t_max hard-bound invariant
    across many configurations, and a dedicated continue_collapse case
    where --t_max falls strictly on the mirrored contracting branch.
  * A CLI subprocess smoke test (--version, a successful run, and a
    deliberately invalid run), robust to both the normal repo/tests/
    layout and a flattened upload directory (skipped, never failed,
    if main.py cannot be located in either), and CSV assertions for
    the exact model version and the resolved-parameter provenance
    fields.
  * Audit 6 additions: Big-Bang branch connectivity certification
    (a hidden forbidden interval below a_i, and a CPL dark-energy term
    that dominates E(a)^2 NEGATIVELY as a->0, both rejected; a standard
    connected model and a genuinely closed recollapsing model are not
    false-positively rejected); Big-Rip dominance-scale overflow safety
    for w0 immediately below -1 across flat/open/closed curvature;
    structured fate_status/future_turnaround_a/fate_search_limit_a
    fields covering every fate outcome (recollapse, big_rip,
    future_recollapse, unresolved, and no-fate-question-applies); the
    MODEL_VERSION bump off "1.2.0"; a_eq_rm's reachability gate (an
    algebraically-derived milestone must not be reported when the model
    recollapses before ever reaching it); total_lifetime_gyr reported
    without requiring --continue_collapse; and the NaN policy in
    omega_fractions masking only non-positive E(a)^2, never a small-but-
    finite value near a loitering dip.

Independent cross-checks in this file (the turnaround-time and
two-root benchmarks below) are computed by methods coded independently
of physics_cosmo.py's own quadrature and root-finding routines -- a
separate fixed-resolution midpoint-rule integrator and, for the
two-root case, a direct cubic-polynomial root solve -- rather than by
calling back into the code under test.

Run with:   python -m pytest tests/test_physics_cosmo.py -v
       or:  python -m unittest tests.test_physics_cosmo -v
       or:  python tests/test_physics_cosmo.py
(No pytest dependency is required -- everything here is plain unittest.)
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_cosmo as phys
import driver_cosmo as drv

def _find_main_py():
    """
    Locate the directory containing main.py relative to THIS test file,
    trying a short list of candidate directories rather than assuming a
    single fixed layout. The normal repository layout is
    repo/tests/test_physics_cosmo.py with repo/main.py, but an upload or
    packaging step can flatten everything (main.py and
    test_physics_cosmo.py side by side in one directory), and
    REPO_ROOT = dirname(dirname(__file__)) is only correct for the
    former. Returns the directory containing main.py, or None if it
    cannot be found in any candidate location -- callers should skip
    (never fail) the CLI subprocess tests in that case, since it may be
    an artifact of how the files were uploaded/flattened for review
    rather than a defect in the actual release layout (Copilot Audit 6
    P1-1).
    """
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.dirname(here),  # repo/tests/ -> repo/ (normal layout)
        here,                   # flattened: main.py beside this file
        os.path.dirname(os.path.dirname(here)),
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "main.py")):
            return cand
    return None


REPO_ROOT = _find_main_py()


def H0_pgyr(H0_kms_mpc):
    return phys.H0_per_Gyr(H0_kms_mpc)


# ----------------------------------------------------------------------
# Independent numerical cross-checks, deliberately NOT sharing code with
# physics_cosmo.py's own _quad_time_between / _quad_time_to_turnaround /
# _first_root_ahead / _bisect_root_a. A fixed-resolution midpoint rule is
# a different quadrature family from the module's Gauss-Legendre nodes,
# and is simple enough to trust by inspection.
# ----------------------------------------------------------------------
def _indep_E2(a, omega_m, omega_r, omega_k, omega_de=0.0, w0=-1.0, wa=0.0):
    a = np.asarray(a, dtype=float)
    g = 0.0
    if omega_de != 0.0:
        g = a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    return omega_m * a ** -3 + omega_r * a ** -4 + omega_k * a ** -2 + omega_de * g


def _indep_time_between(a_lo, a_hi, H0_kms_mpc, omega_m, omega_r, omega_k,
                         omega_de=0.0, w0=-1.0, wa=0.0, n=400_000):
    """t = Integral_{a_lo}^{a_hi} da / [a H0 E(a)], ordinary (non-singular)
    interval, by composite midpoint rule."""
    H0p = H0_pgyr(H0_kms_mpc)
    du = (a_hi - a_lo) / n
    a_vals = a_lo + (np.arange(n) + 0.5) * du
    e2 = _indep_E2(a_vals, omega_m, omega_r, omega_k, omega_de, w0, wa)
    integrand = 1.0 / (a_vals * H0p * np.sqrt(e2))
    return float(np.sum(integrand) * du)


def _indep_time_to_turnaround(a_lo, a_turn, H0_kms_mpc, omega_m, omega_r,
                               omega_k, omega_de=0.0, w0=-1.0, wa=0.0,
                               n=400_000):
    """Same integral, but from a_lo up to a genuine turning point a_turn,
    using the substitution a = a_turn - u^2 (du = -da/(2u)) to remove the
    1/sqrt(a_turn-a) singularity, with an OPEN (midpoint) grid in u so
    u=0 -- the singular endpoint itself -- is never evaluated directly."""
    H0p = H0_pgyr(H0_kms_mpc)
    u_max = math.sqrt(a_turn - a_lo)
    du = u_max / n
    u_vals = (np.arange(n) + 0.5) * du
    a_vals = a_turn - u_vals ** 2
    e2 = _indep_E2(a_vals, omega_m, omega_r, omega_k, omega_de, w0, wa)
    # For u_max small enough that u^2 underflows the float64 precision of
    # a_turn itself (a_turn - u^2 rounds back to exactly a_turn), e2 can
    # come out as exactly 0 for a handful of grid points nearest u=0.
    # Each such point has vanishing measure (spacing ~u_max/n) and a
    # finite true integrand there (the u->0 limit of 2u/(a H E) is
    # finite, not singular), so treating its contribution as 0 rather
    # than the true small finite value introduces an error far below
    # this helper's intended few-times-1e-5-relative precision.
    good = e2 > 0.0
    integrand = np.zeros_like(u_vals)
    integrand[good] = 2.0 * u_vals[good] / (a_vals[good] * H0p
                                             * np.sqrt(e2[good]))
    return float(np.sum(integrand) * du)


class AnalyticAges(unittest.TestCase):
    """Single-component universes have closed-form ages; the numerical
    integrator should reproduce them to within its documented tolerance
    (a few times step_frac in relative terms, at default step_frac)."""

    def test_einstein_de_sitter(self):
        # Flat, matter-only: t0 = (2/3) / H0
        r = phys.integrate_evolution(H0=70.0, omega_m=1.0, omega_r=0.0,
                                      omega_de=0.0, a_max=1.0,
                                      step_frac=0.002)
        age = r["summary"]["age_today_gyr"]
        expected = (2.0 / 3.0) / H0_pgyr(70.0)
        self.assertAlmostEqual(age, expected, delta=expected * 1e-3)

    def test_radiation_only(self):
        # Flat, radiation-only: t0 = (1/2) / H0
        r = phys.integrate_evolution(H0=70.0, omega_m=0.0, omega_r=1.0,
                                      omega_de=0.0, a_i=1.0e-10,
                                      a_max=1.0, step_frac=0.002)
        age = r["summary"]["age_today_gyr"]
        expected = 0.5 / H0_pgyr(70.0)
        self.assertAlmostEqual(age, expected, delta=expected * 1e-3)

    def test_milne(self):
        # Empty, pure curvature: a = H0 t exactly, so H0*t0 = 1.
        r = phys.integrate_evolution(H0=70.0, omega_m=0.0, omega_r=0.0,
                                      omega_de=0.0, a_max=1.0,
                                      step_frac=0.002)
        age = r["summary"]["age_today_gyr"]
        h0t0 = age * H0_pgyr(70.0)
        self.assertAlmostEqual(h0t0, 1.0, delta=1e-3)

    def test_flat_lambda_cdm_sanity(self):
        # Not a closed form; cross-checked against an independently
        # implemented quadrature (composite midpoint rule) of the same
        # Friedmann integral, rather than a wide hand-picked range.
        r = phys.integrate_evolution(H0=67.4, omega_m=0.315,
                                      omega_r=9.24e-5, a_max=1.0,
                                      step_frac=0.005)
        age = r["summary"]["age_today_gyr"]
        # Flat by construction (omega_de omitted => forced flat): Ok=0,
        # Ode = 1 - Om - Or.
        omega_de = 1.0 - 0.315 - 9.24e-5
        expected = _indep_time_between(1.0e-8, 1.0, 67.4, 0.315, 9.24e-5,
                                        0.0, omega_de=omega_de)
        self.assertAlmostEqual(age, expected, delta=expected * 1e-3)


class TurnaroundBoundaryPrecedence(unittest.TestCase):
    """a_max/t_max must be respected as hard bounds even when a genuine
    recollapse would occur beyond them, INCLUDING when the requested
    bound sits exactly at, or arbitrarily close to, the true turning
    point -- the numerically hardest case, since a naive single large
    integration step cannot safely land on a sqrt-singularity."""

    # Closed, matter-only: Omega_m0=2 => Omega_k0=-1, and the turning
    # point 2*a^-3 = 1*a^-2 is exactly a_turn=2 (t_turn = pi/H0 exactly,
    # the standard closed-matter-dominated cycloid solution).
    ARGS = dict(H0=70.0, omega_m=2.0, omega_r=0.0, omega_de=0.0,
                a_i=1.0e-5, step_frac=0.005)
    A_TURN_EXACT = 2.0
    T_TURN_EXACT = math.pi / H0_pgyr(70.0)

    def test_a_max_below_turnaround_is_respected(self):
        r = phys.integrate_evolution(a_max=1.8, t_max_gyr=None, **self.ARGS)
        s = r["summary"]
        self.assertAlmostEqual(r["a"][-1], 1.8, places=6)
        self.assertIsNone(s["turnaround"])
        self.assertEqual(s["stop_reason"], "a_max")

    def test_a_max_above_turnaround_reports_turnaround(self):
        r = phys.integrate_evolution(a_max=5.0, t_max_gyr=None, **self.ARGS)
        s = r["summary"]
        self.assertIsNotNone(s["turnaround"])
        self.assertEqual(s["stop_reason"], "turnaround")
        self.assertAlmostEqual(s["turnaround"]["a_turn"], self.A_TURN_EXACT,
                                places=6)
        self.assertAlmostEqual(s["turnaround"]["t_turn_gyr"],
                                self.T_TURN_EXACT, delta=1e-6)

    def test_a_max_exactly_at_turnaround(self):
        # a_max coincides exactly with the true turnaround: this must
        # stop cleanly and accurately, not raise.
        r = phys.integrate_evolution(a_max=self.A_TURN_EXACT,
                                      t_max_gyr=None, **self.ARGS)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "turnaround")
        self.assertAlmostEqual(s["turnaround"]["a_turn"], self.A_TURN_EXACT,
                                places=6)
        self.assertAlmostEqual(s["turnaround"]["t_turn_gyr"],
                                self.T_TURN_EXACT, delta=1e-6)

    def test_a_max_approaching_turnaround_matrix(self):
        # a_max/a_turn ratios approaching 1 from below: each must stop
        # cleanly (no RuntimeError, no leaked exception) at the requested
        # a_max, with the elapsed time matching the independent
        # singularity-removing quadrature to high precision -- NOT the
        # badly-degraded plain-quadrature result a naive implementation
        # would give this close to a genuine turning point.
        for frac in (0.90, 0.95, 0.975, 0.99, 0.999, 0.999999):
            a_max = frac * self.A_TURN_EXACT
            with self.subTest(frac=frac):
                r = phys.integrate_evolution(a_max=a_max, t_max_gyr=None,
                                              **self.ARGS)
                s = r["summary"]
                self.assertEqual(s["stop_reason"], "a_max")
                self.assertIsNone(s["turnaround"])
                self.assertAlmostEqual(r["a"][-1], a_max, places=6)
                expected_t = _indep_time_to_turnaround(
                    self.ARGS["a_i"], self.A_TURN_EXACT, self.ARGS["H0"],
                    self.ARGS["omega_m"], self.ARGS["omega_r"], -1.0) \
                    - _indep_time_to_turnaround(
                        a_max, self.A_TURN_EXACT, self.ARGS["H0"],
                        self.ARGS["omega_m"], self.ARGS["omega_r"], -1.0)
                self.assertAlmostEqual(r["t_gyr"][-1], expected_t,
                                        delta=max(1e-5, expected_t * 1e-4))

    def test_t_max_before_turnaround_is_respected(self):
        r = phys.integrate_evolution(a_max=5.0,
                                      t_max_gyr=self.T_TURN_EXACT * 0.5,
                                      **self.ARGS)
        s = r["summary"]
        self.assertIsNone(s["turnaround"])
        self.assertEqual(s["stop_reason"], "t_max")
        self.assertAlmostEqual(r["t_gyr"][-1], self.T_TURN_EXACT * 0.5,
                                places=6)

    def test_t_max_after_turnaround_does_not_suppress_it(self):
        r = phys.integrate_evolution(a_max=5.0,
                                      t_max_gyr=self.T_TURN_EXACT * 2.0,
                                      **self.ARGS)
        s = r["summary"]
        self.assertIsNotNone(s["turnaround"])
        self.assertEqual(s["stop_reason"], "turnaround")

    def test_t_max_approaching_turnaround_matrix(self):
        # t_max/t_turn ratios close to (and just past) 1: no RuntimeError
        # and, crucially, no leaked internal _Overshoot exception (the
        # t_max analog of the a_max-near-turnaround case above). At
        # frac exactly 1.0, t_max (computed here
        # from the independent analytic pi/H0) and the module's own
        # internally resolved t_turn need not be bit-identical, so either
        # stop_reason is an acceptable, physically equivalent outcome
        # there -- what must hold is that the reported state is correct
        # either way, not that one specific tie-breaking label wins.
        for frac in (0.7, 0.8, 0.9, 0.99, 0.999, 1.0, 1.000001, 1.1):
            t_max = self.T_TURN_EXACT * frac
            with self.subTest(frac=frac):
                r = phys.integrate_evolution(a_max=5.0, t_max_gyr=t_max,
                                              **self.ARGS)
                s = r["summary"]
                self.assertTrue(math.isfinite(r["a"][-1]))
                self.assertTrue(0.0 < r["a"][-1] <= self.A_TURN_EXACT + 1e-9)
                if frac < 0.999999:
                    self.assertEqual(s["stop_reason"], "t_max")
                    self.assertAlmostEqual(r["t_gyr"][-1], t_max, places=6)
                elif frac > 1.000001:
                    self.assertEqual(s["stop_reason"], "turnaround")
                else:
                    self.assertIn(s["stop_reason"], ("t_max", "turnaround"))
                    self.assertAlmostEqual(r["a"][-1], self.A_TURN_EXACT,
                                            delta=1e-6)

    def test_turnaround_time_matches_independent_benchmark(self):
        # A second closed model (Omega_m0=1.5, default radiation), cross-
        # checked against the independent midpoint-rule quadrature above
        # rather than a hardcoded literal.
        args = dict(H0=70.0, omega_m=1.5, omega_r=9.24e-5, omega_de=0.0,
                    a_i=1.0e-8, a_max=5.0, step_frac=0.005)
        r = phys.integrate_evolution(**args)
        s = r["summary"]
        a_turn = s["turnaround"]["a_turn"]
        expected_t = _indep_time_to_turnaround(
            args["a_i"], a_turn, args["H0"], args["omega_m"],
            args["omega_r"], 1.0 - args["omega_m"] - args["omega_r"])
        self.assertAlmostEqual(s["turnaround"]["t_turn_gyr"], expected_t,
                                delta=max(1e-4, expected_t * 1e-4))

    def test_age_just_above_turnaround_reports_correct_age(self):
        # A model whose turnaround is JUST above a=1 exercises the a=1
        # checkpoint specifically: age_today_gyr must be resolved by an
        # accurate, verified method (quadrature anchored to the actual
        # turning point), not an unverified bisection bracket that could
        # silently return a plausible-looking but wrong time.
        for omega_m in (10.0, 20.0, 101.0):
            with self.subTest(omega_m=omega_m):
                args = dict(H0=70.0, omega_m=omega_m, omega_r=9.24e-5,
                            omega_de=0.0, a_i=1.0e-8, a_max=1.0,
                            step_frac=0.005)
                r = phys.integrate_evolution(**args)
                age = r["summary"]["age_today_gyr"]
                omega_k = 1.0 - omega_m - args["omega_r"]
                expected = _indep_time_between(
                    args["a_i"], 1.0, args["H0"], omega_m, args["omega_r"],
                    omega_k)
                self.assertAlmostEqual(age, expected,
                                        delta=max(1e-5, expected * 1e-4))


class NonMonotonicE2(unittest.TestCase):
    """E(a)^2 need not be monotonic: a closed universe with matter and a
    small positive cosmological constant can have TWO positive roots (a
    forbidden interval), with E(a_max)^2 testing positive again beyond
    it. The physical expanding-branch solution must stop at the FIRST
    root; checking only the sign of E(a_max)^2 cannot detect this."""

    def test_stops_at_first_of_two_roots(self):
        H0, omega_m, omega_de = 70.0, 1.5, 0.005
        omega_k = 1.0 - omega_m - omega_de  # -0.505, default w0=-1,wa=0
        # Independent check: E(a)^2=0 multiplied through by a^3 is the
        # cubic omega_de*a^3 + omega_k*a + omega_m = 0 (w0=-1,wa=0 =>
        # g(a)=1 identically). Solved directly, independent of the
        # module's own root-finding.
        roots = np.roots([omega_de, 0.0, omega_k, omega_m])
        positive_real_roots = sorted(
            r.real for r in roots
            if abs(r.imag) < 1e-9 and r.real > 0.0)
        self.assertEqual(len(positive_real_roots), 2,
                          "test setup error: expected two positive roots")
        a_root1, a_root2 = positive_real_roots

        r = phys.integrate_evolution(H0=H0, omega_m=omega_m, omega_r=0.0,
                                      omega_de=omega_de, a_i=1.0e-8,
                                      a_max=20.0, step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "turnaround")
        self.assertIsNotNone(s["turnaround"])
        self.assertAlmostEqual(s["turnaround"]["a_turn"], a_root1, places=4)
        # Must NOT have been waved through to (or anywhere near) the
        # second root.
        self.assertLess(r["a"][-1], 0.5 * (a_root1 + a_root2))


class NearDoubleRootRegression(unittest.TestCase):
    """Adversarial regression coverage for arbitrarily-close pairs of
    positive roots of E(a)^2=0. A fixed-resolution grid scan can miss a
    forbidden interval narrower than its own spacing; these cases probe
    a range of separations spanning above and below that former
    (2000-point, ~a_max/2000 wide) grid spacing, with roots found
    independently via numpy.roots on the matter+curvature+Lambda cubic
    a^3 E(a)^2 = omega_de*a^3 + omega_k*a + omega_m (w0=-1, wa=0 makes
    the CPL shape g(a)=1 identically, so this is exactly cubic)."""

    H0 = 70.0
    OMEGA_M = 1.5

    @classmethod
    def _positive_real_roots(cls, omega_de):
        omega_k = 1.0 - cls.OMEGA_M - omega_de
        roots = np.roots([omega_de, 0.0, omega_k, cls.OMEGA_M])
        return sorted(r.real for r in roots
                      if abs(r.imag) < 1.0e-9 and r.real > 0.0)

    @classmethod
    def _find_critical_omega_de(cls):
        # Bisect on omega_de for the transition between "two positive
        # real roots" (a forbidden interval exists) and "zero positive
        # real roots" (the pair has gone complex, i.e. E(a)^2 dips close
        # to zero but never crosses it) -- independent of
        # physics_cosmo.py, using only numpy.roots on the cubic above.
        lo, hi = 1.0e-6, 0.5
        assert len(cls._positive_real_roots(lo)) == 2
        assert len(cls._positive_real_roots(hi)) == 0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if len(cls._positive_real_roots(mid)) == 2:
                lo = mid
            else:
                hi = mid
        return lo

    def test_first_root_found_across_a_range_of_separations(self):
        ode_crit = self._find_critical_omega_de()
        # Separations from comfortably above the old grid spacing
        # (~1e-2) down to a razor-thin fraction of it, by moving
        # omega_de closer to the critical value from below.
        for eps in (1.0e-3, 1.0e-5, 1.0e-7, 1.0e-9, 1.0e-10):
            omega_de = ode_crit - eps
            roots = self._positive_real_roots(omega_de)
            with self.subTest(eps=eps):
                self.assertEqual(len(roots), 2,
                                  "test setup error: expected two "
                                  "positive real roots")
                a_root1, a_root2 = roots
                separation = a_root2 - a_root1
                r = phys.integrate_evolution(
                    H0=self.H0, omega_m=self.OMEGA_M, omega_r=0.0,
                    omega_de=omega_de, w0=-1.0, wa=0.0,
                    a_i=1.0e-8, a_max=20.0, step_frac=0.005)
                s = r["summary"]
                self.assertEqual(
                    s["stop_reason"], "turnaround",
                    f"eps={eps:.3g} (root separation={separation:.3g}) "
                    "was not detected as a turning point")
                self.assertAlmostEqual(
                    s["turnaround"]["a_turn"], a_root1, places=4,
                    msg=f"eps={eps:.3g}: a_turn did not match the "
                    "independently-computed first root")
                # No returned sample of the expanding branch may exceed
                # the first root -- the forbidden interval must never be
                # crossed, however narrow it is.
                self.assertLessEqual(max(r["a"]), a_root1 * (1.0 + 1e-9))

    def test_confirmed_no_root_reaches_a_max_without_large_gaps(self):
        # Just ABOVE critical: the same near-cancellation structure, but
        # now the two roots have gone complex -- E(a)^2 dips close to
        # zero and rebounds without crossing it (a "loitering" model).
        # This must integrate cleanly all the way to a_max, and -- since
        # an earlier version of this integrator fast-forwarded across
        # such a confirmed false alarm by quadrature alone, discarding
        # the RK4 trajectory in between -- must not leave any
        # anomalously large gap in the returned (a, t) samples.
        ode_crit = self._find_critical_omega_de()
        r = phys.integrate_evolution(
            H0=self.H0, omega_m=self.OMEGA_M, omega_r=0.0,
            omega_de=ode_crit + 1.0e-8, w0=-1.0, wa=0.0,
            a_i=1.0e-8, a_max=20.0, step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "a_max")
        self.assertEqual(r["a"][-1], 20.0)
        a_arr = np.asarray(r["a"])
        t_arr = np.asarray(r["t_gyr"])
        # An ordinary step near a=20 with step_frac=0.005 changes a by a
        # small fraction of a itself; a gap even a few times that large
        # would indicate a discarded stretch of trajectory, not just an
        # ordinary variable step size.
        self.assertLess(np.diff(a_arr).max(), 1.0)
        self.assertLess(np.diff(t_arr).max(), 10.0)


class AMaxHardBoundInvariant(unittest.TestCase):
    """a_max is documented as a HARD bound: the returned trajectory must
    never contain a point past it, however close a_max sits to a
    genuine turning point. An earlier revision reclassified a
    coincidental a_max/a_turn tie as 'turnaround' using a floating-
    point tolerance -- which could return an a_next slightly PAST the
    literal a_max the caller supplied, a strictly worse defect than
    the label inconsistency it was meant to smooth over. The corrected
    behavior compares a_max and a_turn with a plain, tolerance-free
    "<": a user who sets a_max a hair below a genuine turning point
    gets exactly that a_max back, labeled 'a_max', not a substituted
    a_turn that would violate the bound."""

    def test_a_max_hard_bound_across_offsets_from_turnaround(self):
        # Pure matter+curvature (no radiation, no dark energy): the
        # turning point is the exact closed-form a_turn = Om/(Om-1),
        # from E(a)^2 = Om*a^-3 + (1-Om)*a^-2 = 0.
        #
        # At a genuine offset (1e-9 relative and wider), the margin is
        # many orders of magnitude larger than _bisect_root_a's own
        # ~1e-15-relative convergence noise, so the label is expected
        # to be deterministically 'a_max'. At literally one ULP, which
        # of a_max/computed-a_turn ends up numerically larger depends
        # on which side of the true root the bisection happens to
        # converge -- an implementation detail this program does not
        # promise to control -- so only the HARD NUMERIC bound
        # (max(a) <= a_max) is asserted there, not a specific label;
        # that bound must hold at every offset without exception.
        for omega_m in (1.5, 2.5, 3.0, 7.0):
            a_turn_analytic = omega_m / (omega_m - 1.0)
            offsets = {
                "one ULP below": np.nextafter(a_turn_analytic, -np.inf),
                "1e-9 relative below": a_turn_analytic * (1.0 - 1.0e-9),
                "0.1% below": a_turn_analytic * 0.999,
                "1% below": a_turn_analytic * 0.99,
            }
            for label, a_max in offsets.items():
                with self.subTest(omega_m=omega_m, offset=label):
                    r = phys.integrate_evolution(
                        H0=70.0, omega_m=omega_m, omega_r=0.0,
                        omega_de=0.0, w0=-1.0, wa=0.0, a_i=1.0e-8,
                        a_max=a_max, step_frac=0.005)
                    s = r["summary"]
                    # The hard bound: no returned sample may exceed the
                    # literal a_max supplied, at any offset, however
                    # close to the true turning point.
                    self.assertLessEqual(max(r["a"]), a_max)
                    if label != "one ULP below":
                        self.assertEqual(s["stop_reason"], "a_max")
                        self.assertAlmostEqual(r["a"][-1], a_max, places=9)
                    else:
                        self.assertIn(s["stop_reason"], ("a_max", "turnaround"))

    def test_a_max_at_or_beyond_turnaround_reports_turnaround(self):
        omega_m = 3.0
        a_turn_analytic = omega_m / (omega_m - 1.0)
        for a_max in (a_turn_analytic, a_turn_analytic * 1.001):
            with self.subTest(a_max=a_max):
                r = phys.integrate_evolution(
                    H0=70.0, omega_m=omega_m, omega_r=0.0, omega_de=0.0,
                    w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=a_max,
                    step_frac=0.005)
                s = r["summary"]
                self.assertEqual(s["stop_reason"], "turnaround")
                self.assertLessEqual(max(r["a"]), a_max + 1.0e-9)


class ExtremeCPLRejection(unittest.TestCase):
    """Extreme CPL parameters must raise a clear error rather than
    silently overflow to NaN/inf and have that non-finite value mistaken
    for a physical turning point."""

    def test_absurd_w0_wa_raises_cleanly(self):
        with self.assertRaises(ValueError):
            phys.integrate_evolution(H0=70.0, omega_m=0.3, omega_de=0.7,
                                      w0=1000.0, wa=1000.0, a_max=5.0,
                                      step_frac=0.005)

    def test_large_wa_at_moderate_a_max_raises_or_is_finite(self):
        # Either it raises the domain error cleanly, or (for milder
        # combinations) it succeeds -- but if it succeeds, the ENTIRE
        # returned trajectory must be well-formed: finite throughout, any
        # reported turning point strictly within the requested a-range,
        # a consistent stop_reason, and E(a)^2 non-negative everywhere
        # along the returned expanding branch (as it must be for any
        # point the integrator actually visited).
        try:
            r = phys.integrate_evolution(H0=70.0, omega_m=0.3, omega_de=0.7,
                                          w0=-1.0, wa=10.0, a_max=5.0,
                                          step_frac=0.005)
        except ValueError:
            return
        s = r["summary"]
        self.assertIn(s["stop_reason"], ("a_max", "t_max", "turnaround"))
        self.assertTrue(np.all(np.isfinite(r["a"])))
        self.assertTrue(np.all(np.isfinite(r["t_gyr"])))
        self.assertTrue(np.all(np.isfinite(r["H_kms_mpc"])))
        e2_along_path = _indep_E2(r["a"], 0.3, 9.24e-5,
                                   1.0 - 0.3 - 9.24e-5 - 0.7, 0.7, -1.0, 10.0)
        self.assertTrue(np.all(e2_along_path >= -1e-8))
        if s["turnaround"] is not None:
            a_turn = s["turnaround"]["a_turn"]
            self.assertTrue(math.isfinite(a_turn))
            self.assertTrue(math.isfinite(s["turnaround"]["t_turn_gyr"]))
            self.assertGreater(a_turn, 0.0)
            self.assertLessEqual(a_turn, 5.0 + 1e-9)

    def test_de_density_shape_raises_instead_of_overflowing(self):
        with self.assertRaises(ValueError):
            phys.de_density_shape(1.0e-8, 1000.0, 1000.0)


class OvershootNeverLeaksToPublicAPI(unittest.TestCase):
    """_Overshoot is an internal control-flow signal (E(a)^2 went
    negative during an RK4 sub-step). It must never cross the public
    API boundary -- callers should see either a normal result or a
    documented ValueError/RuntimeError, never a bare internal exception
    with an empty message."""

    def test_overshoot_class_is_not_a_value_or_runtime_error(self):
        # Guards against a future refactor accidentally making the
        # public exception types swallow _Overshoot silently instead of
        # the reverse.
        self.assertFalse(issubclass(phys._Overshoot, ValueError))
        self.assertFalse(issubclass(phys._Overshoot, RuntimeError))

    def test_dense_t_max_sweep_never_leaks_overshoot(self):
        # Sweep densely across and past a known turnaround with t_max,
        # the regime where an unbounded internal RK4 sub-step is most
        # likely to overshoot into E(a)^2 < 0.
        H0, omega_m = 70.0, 2.0
        t_turn = math.pi / H0_pgyr(H0)
        for frac in np.linspace(0.5, 1.5, 25):
            t_max = t_turn * float(frac)
            try:
                r = phys.integrate_evolution(
                    H0=H0, omega_m=omega_m, omega_r=0.0, omega_de=0.0,
                    a_i=1.0e-5, a_max=5.0, t_max_gyr=t_max,
                    step_frac=0.005)
            except phys._Overshoot:
                self.fail(f"_Overshoot leaked to caller at t_max/t_turn="
                          f"{frac:.6f}")
            except (ValueError, RuntimeError):
                continue  # a documented, well-typed failure is acceptable
            self.assertTrue(math.isfinite(r["a"][-1]))

    def test_dense_a_max_sweep_never_leaks_overshoot(self):
        H0, omega_m = 70.0, 2.0
        a_turn = 2.0
        for frac in np.linspace(0.5, 1.5, 25):
            a_max = a_turn * float(frac)
            try:
                r = phys.integrate_evolution(
                    H0=H0, omega_m=omega_m, omega_r=0.0, omega_de=0.0,
                    a_i=1.0e-5, a_max=a_max, step_frac=0.005)
            except phys._Overshoot:
                self.fail(f"_Overshoot leaked to caller at a_max/a_turn="
                          f"{frac:.6f}")
            except (ValueError, RuntimeError):
                continue
            self.assertTrue(math.isfinite(r["a"][-1]))


class ContractingBranch(unittest.TestCase):
    def test_H_sign_flips_on_mirrored_branch(self):
        r = phys.integrate_evolution(H0=70.0, omega_m=1.5, omega_r=9.24e-5,
                                      omega_de=0.0, a_max=5.0,
                                      step_frac=0.005, continue_collapse=True)
        H = r["H_kms_mpc"]
        n = len(H)
        mid = n // 2
        # Just before turnaround H > 0 (expanding); just after, H < 0
        # (contracting). The exact midpoint (H=0) is the turnaround itself.
        self.assertGreater(H[mid - 2], 0.0)
        self.assertLess(H[mid + 2], 0.0)


class AMaxAndTMaxValidation(unittest.TestCase):
    def test_a_max_equal_one_reports_age(self):
        r = phys.integrate_evolution(H0=67.4, omega_m=0.315,
                                      omega_r=9.24e-5, a_max=1.0,
                                      step_frac=0.005)
        self.assertIsNotNone(r["summary"]["age_today_gyr"])

    def test_t_max_before_t_i_is_rejected(self):
        # Uses a positive t_max that genuinely falls before the model's
        # own start time t_i (not just a negative-number sanity check):
        # a Milne (empty, pure-curvature) universe started late, at
        # a_i=0.9, has a correspondingly large t_i; a small positive
        # t_max well below that must be rejected.
        with self.assertRaises(ValueError):
            phys.integrate_evolution(H0=70.0, omega_m=0.0, omega_r=0.0,
                                      omega_de=0.0, a_i=0.9, a_max=5.0,
                                      t_max_gyr=1.0e-6, step_frac=0.005)

    def test_step_frac_too_small_is_rejected_cleanly(self):
        # 1e-6 is the smallest step_frac this program accepts as valid at
        # all; combined with the very wide default a_i-to-a_max range,
        # that still implies far more RK4 steps than the internal safety
        # limit -- this must fail with a clear, immediate error (a
        # preflight ValueError estimating the step count) rather than
        # hang or silently truncate the integration.
        with self.assertRaises((ValueError, RuntimeError)):
            phys.integrate_evolution(H0=70.0, omega_m=0.3, a_i=1.0e-8,
                                      a_max=5.0, step_frac=1.0e-6)


class OmegaAndAgeScan(unittest.TestCase):
    def test_omega_fractions_sum_to_one(self):
        a = [1.0e-6, 1.0e-3, 0.5, 1.0, 3.0]
        om, orr, ok, ode = phys.omega_fractions(
            a, omega_m=0.3, omega_r=9.24e-5, omega_k=0.0,
            omega_de=0.7 - 9.24e-5, w0=-1.0, wa=0.0)
        total = om + orr + ok + ode
        for t in total:
            self.assertAlmostEqual(t, 1.0, places=8)

    def test_age_scan_labels_recollapse_within_horizon(self):
        kw = dict(H0=70.0, omega_m=0.3, omega_r=9.24e-5, omega_de=None,
                  w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=5.0, t_max=None,
                  step_frac=0.01, continue_collapse=False,
                  presets="EdS,lambdaCDM,closed,phantom",
                  scan_param="omega_m", scan_lo=0.5, scan_hi=2.0, scan_n=8,
                  force_flat=True, age_ref_gyr=13.0, no_plot=True)
        result = drv._run_age(kw, outdir=None, csvdir=None, dpi=150, lw=1.6)
        # At least one scanned point with omega_m > 1 (forced flat) should
        # be flagged as recollapsing within the finite look-ahead horizon.
        self.assertTrue(any(result["recollapsed"]))

    def test_age_ref_gyr_must_be_positive(self):
        with self.assertRaises(ValueError):
            drv.run(mode="age", age_ref_gyr=-1.0, no_plot=True,
                    csvdir="/tmp/_cosmo_test_csv_unused")


class CSVProvenance(unittest.TestCase):
    def test_evolve_csv_written_with_header_and_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            drv.run(mode="evolve", H0=67.4, omega_m=0.315, no_plot=True,
                    csvdir=tmp)
            files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0])) as fh:
                lines = fh.readlines()
            comment_lines = [l for l in lines if l.startswith("#")]
            data_lines = [l for l in lines if not l.startswith("#")]
            self.assertGreater(len(data_lines), 1)
            self.assertIn("t_Gyr", data_lines[0])
            self.assertIn("version", "".join(comment_lines))

    def test_evolve_csv_records_actual_stop_reason(self):
        # The help file states that stop_reason is written to CSV
        # provenance; this asserts the ACTUAL value is present, not
        # merely that a header/rows exist.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            drv.run(mode="evolve", H0=70.0, omega_m=2.0, omega_r=0.0,
                    omega_de=0.0, a_i=1.0e-5, a_max=2.0, no_plot=True,
                    csvdir=tmp)
            files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            with open(os.path.join(tmp, files[0])) as fh:
                text = fh.read()
            self.assertIn("stop_reason = turnaround", text)


class IndeterminateRootSearchSemantics(unittest.TestCase):
    """Audit 5 (Copilot P0-2/P1-2, Codex P1-2): the certified search must
    raise _IndeterminateRootSearch -- never silently return None ("no
    root") -- both when its evaluation budget runs out and when a
    sub-interval has been bisected down to float64's own subdivision
    limit without resolving either way. Both paths are forced directly
    here via temporary monkeypatches, restored in `finally`, rather than
    hoping to stumble onto a naturally occurring pathological case."""

    def test_budget_exhaustion_raises_indeterminate(self):
        # A genuine near-double-root case (root separation ~1e-9) needs
        # more than a handful of bisections to resolve; capping the
        # budget at 3 evaluations forces exhaustion long before the
        # depth/width floor (60 levels) could ever be reached.
        ode_crit = NearDoubleRootRegression._find_critical_omega_de()
        omega_de = ode_crit - 1.0e-9
        omega_m = NearDoubleRootRegression.OMEGA_M
        omega_k = 1.0 - omega_m - omega_de
        args = (H0_pgyr(70.0), omega_m, 0.0, omega_k, omega_de, -1.0, 0.0)
        orig_budget = phys._CERT_MAX_EVALUATIONS
        try:
            phys._CERT_MAX_EVALUATIONS = 3
            with self.assertRaises(phys._IndeterminateRootSearch) as ctx:
                phys._first_root_ahead(1.0e-8, 20.0, args)
            self.assertIn("search budget", str(ctx.exception))
        finally:
            phys._CERT_MAX_EVALUATIONS = orig_budget

    def test_depth_and_width_floor_raises_indeterminate(self):
        # Force every certification attempt to fail by making the
        # Lipschitz bound artificially enormous (so B is always deeply
        # negative), on an interval already close to a single ULP wide;
        # this must bottom out at the float64 subdivision limit (or the
        # depth cap) and raise, not silently return "safe".
        orig_bound = phys._e2_derivative_lipschitz_bound

        def huge_bound(*a, **k):
            return 1.0e18

        try:
            phys._e2_derivative_lipschitz_bound = huge_bound
            args = (H0_pgyr(70.0), 0.3, 9.24e-5,
                    1.0 - 0.3 - 9.24e-5 - 0.7, 0.7, -1.0, 0.0)
            with self.assertRaises(phys._IndeterminateRootSearch) as ctx:
                phys._first_root_ahead(1.0, 1.000001, args)
            self.assertIn("narrow", str(ctx.exception))
        finally:
            phys._e2_derivative_lipschitz_bound = orig_bound

    def test_indeterminate_propagates_through_evolve_not_silently(self):
        # With _first_root_ahead forced to always raise, an ordinary
        # closed-universe run whose proximity check fires (so the
        # certified search is actually invoked) must see the exception
        # propagate to the caller -- never a plausible-looking
        # completed result.
        orig = phys._first_root_ahead

        def always_raise(*a, **k):
            raise phys._IndeterminateRootSearch("forced for test")

        try:
            phys._first_root_ahead = always_raise
            with self.assertRaises(phys._IndeterminateRootSearch):
                phys.integrate_evolution(
                    H0=70.0, omega_m=1.5, omega_r=0.0, omega_de=0.0,
                    a_i=1.0e-8, a_max=5.0, step_frac=0.005)
        finally:
            phys._first_root_ahead = orig

    def test_indeterminate_recorded_not_treated_as_no_recollapse_in_age_scan(self):
        # The age scan isolates one bad point's exception per-point (so
        # one pathological parameter value does not abort the whole
        # scan), but it must record that point as a FAILURE -- never as
        # recollapsed=False/age=nan silently standing in for a confirmed
        # "no recollapse, ordinary point" result. With _first_root_ahead
        # forced to always raise, every point whose proximity check
        # fires must show up in the CSV's 'note' column instead of
        # blending in as an anonymous ordinary point.
        orig = phys._first_root_ahead

        def always_raise(*a, **k):
            raise phys._IndeterminateRootSearch("forced for test")

        try:
            phys._first_root_ahead = always_raise
            with tempfile.TemporaryDirectory() as tmp:
                kw = dict(H0=70.0, omega_m=0.3, omega_r=9.24e-5,
                          omega_de=None, w0=-1.0, wa=0.0, a_i=1.0e-8,
                          a_max=5.0, t_max=None, step_frac=0.01,
                          continue_collapse=False,
                          presets="EdS,lambdaCDM,closed,phantom",
                          scan_param="omega_m", scan_lo=1.5, scan_hi=3.0,
                          scan_n=4, force_flat=True, age_ref_gyr=13.0,
                          no_plot=True)
                # scan_lo/scan_hi > 1 with force_flat means every scanned
                # point recollapses, so the proximity check -- and hence
                # the forced-raising _first_root_ahead -- fires for all
                # of them; this must raise (all points failed) rather
                # than return a scan result with a full column of
                # ordinary-looking recollapsed=False/age=nan points.
                with self.assertRaises(RuntimeError):
                    drv._run_age(kw, outdir=None, csvdir=tmp, dpi=150, lw=1.6)
        finally:
            phys._first_root_ahead = orig


class LipschitzBoundPropertyTests(unittest.TestCase):
    """Copilot Audit 5 P1-3: randomized/property-based verification that
    _e2_derivative_lipschitz_bound is a genuine upper bound on
    |dE(a)^2/da| everywhere in [a_lo, a_hi] -- checked by dense sampling
    of the exact analytic derivative _dE2_da, across positive AND
    negative omega_de, nonzero wa, and a case constructed so the CPL
    shape's own interior extremum falls strictly inside the interval."""

    def test_bound_dominates_dense_sampling_across_random_parameters(self):
        rng = np.random.RandomState(20260823)
        n_trials = 150
        n_checked = 0
        for _ in range(n_trials):
            omega_m = rng.uniform(0.0, 2.0)
            omega_r = rng.uniform(0.0, 1.0e-3)
            omega_de = rng.uniform(-1.0, 2.0)
            omega_k = 1.0 - omega_m - omega_r - omega_de
            w0 = rng.uniform(-2.0, -0.2)
            wa = rng.uniform(-3.0, 3.0)
            a_lo = 10.0 ** rng.uniform(-3.0, 0.3)
            a_hi = a_lo * 10.0 ** rng.uniform(0.05, 1.0)
            try:
                bound = phys._e2_derivative_lipschitz_bound(
                    a_lo, a_hi, omega_m, omega_r, omega_k, omega_de, w0, wa)
                a_grid = np.linspace(a_lo, a_hi, 400)
                deriv_max = max(
                    abs(phys._dE2_da(float(aa), omega_m, omega_r, omega_k,
                                      omega_de, w0, wa))
                    for aa in a_grid)
            except ValueError:
                continue  # outside this program's supported policy range
            n_checked += 1
            self.assertGreaterEqual(
                bound, deriv_max * (1.0 - 1.0e-9),
                f"bound {bound:.6g} < sampled max |dE2/da| {deriv_max:.6g} "
                f"for om={omega_m:.4g}, or={omega_r:.4g}, ok={omega_k:.4g}, "
                f"ode={omega_de:.4g}, w0={w0:.4g}, wa={wa:.4g}, "
                f"a in [{a_lo:.4g},{a_hi:.4g}]")
        self.assertGreater(n_checked, n_trials // 2,
                            "too many random trials were rejected as "
                            "outside the policy range for this to be a "
                            "meaningful property check")

    def test_bound_dominates_when_cpl_extremum_is_interior(self):
        w0, wa = -0.5, 0.8
        n = -3.0 * (1.0 + w0 + wa)
        a_crit = -n / (3.0 * wa)
        self.assertGreater(a_crit, 0.0,
                            "test setup error: expected a positive "
                            "interior extremum")
        a_lo, a_hi = a_crit * 0.5, a_crit * 1.7
        omega_m, omega_r, omega_de = 0.3, 9.24e-5, 0.6
        omega_k = 1.0 - omega_m - omega_r - omega_de
        bound = phys._e2_derivative_lipschitz_bound(
            a_lo, a_hi, omega_m, omega_r, omega_k, omega_de, w0, wa)
        a_grid = np.linspace(a_lo, a_hi, 2000)
        deriv_max = max(
            abs(phys._dE2_da(float(aa), omega_m, omega_r, omega_k,
                              omega_de, w0, wa))
            for aa in a_grid)
        self.assertGreaterEqual(bound, deriv_max * (1.0 - 1.0e-9))


class RootBracketInvariant(unittest.TestCase):
    """Copilot Audit 5 P1-4: every bracket _first_root_ahead returns
    must satisfy E(a_lo)^2 >= 0 and E(a_hi)^2 < 0 -- directly asserted
    here rather than only inferred from downstream behavior."""

    def test_bracket_endpoints_satisfy_sign_condition(self):
        cases = [
            dict(omega_m=1.5, omega_r=0.0, omega_de=0.005, w0=-1.0, wa=0.0),
            dict(omega_m=2.0, omega_r=0.0, omega_de=0.0, w0=-1.0, wa=0.0),
            dict(omega_m=0.3, omega_r=9.24e-5, omega_de=0.6999076,
                 w0=-1.0, wa=0.0),
            dict(omega_m=1.2, omega_r=0.0, omega_de=0.001, w0=-1.2, wa=0.0),
        ]
        H0p = H0_pgyr(70.0)
        for kw in cases:
            omega_k = 1.0 - kw["omega_m"] - kw["omega_r"] - kw["omega_de"]
            args = (H0p, kw["omega_m"], kw["omega_r"], omega_k,
                    kw["omega_de"], kw["w0"], kw["wa"])
            with self.subTest(**kw):
                try:
                    bracket = phys._first_root_ahead(1.0e-6, 50.0, args)
                except phys._IndeterminateRootSearch:
                    continue
                if bracket is None:
                    continue
                lo, hi = bracket
                e2_lo = float(phys.E2(lo, kw["omega_m"], kw["omega_r"],
                                       omega_k, kw["omega_de"], kw["w0"],
                                       kw["wa"]))
                e2_hi = float(phys.E2(hi, kw["omega_m"], kw["omega_r"],
                                       omega_k, kw["omega_de"], kw["w0"],
                                       kw["wa"]))
                self.assertGreaterEqual(e2_lo, 0.0)
                self.assertLess(e2_hi, 0.0)


class BigRipPrecedenceRegression(unittest.TestCase):
    """Codex Audit 5 P0-1: a Big Rip must never be reported when the
    physical expanding branch actually recollapses first, even if that
    turnaround lies beyond the requested a_max/t_max -- 'turnaround is
    None' for the requested run must not be conflated with 'no root
    exists at all'. Uses Codex's own exact reproducer, with independent
    roots at a~3.113 and a~9.432."""

    def test_recollapse_beyond_a_max_suppresses_big_rip(self):
        r = phys.integrate_evolution(
            H0=70.0, omega_m=1.5, omega_r=0.0, omega_de=0.001, w0=-1.2,
            wa=0.0, a_i=1.0e-8, a_max=1.5, step_frac=0.005)
        s = r["summary"]
        self.assertIsNone(s["big_rip_gyr"])
        self.assertTrue(
            any("recollapses" in w for w in s["warnings"]),
            "expected a warning citing the found future recollapse")

    def test_flat_phantom_still_reports_genuine_big_rip(self):
        # Regression guard: the stricter recollapse-precedence check
        # must not break the ordinary case where the phantom term truly
        # does dominate forever (flat, generous a_max).
        r = phys.integrate_evolution(
            H0=70.0, omega_m=0.3, omega_r=9.24e-5,
            omega_de=0.7 - 9.24e-5, w0=-1.2, wa=0.0, a_i=1.0e-8,
            a_max=60.0, step_frac=0.005)
        s = r["summary"]
        self.assertIsNotNone(s["big_rip_gyr"])
        self.assertGreater(s["big_rip_gyr"], s["age_today_gyr"])


class CycloidInteriorAccuracy(unittest.TestCase):
    """Codex Audit 5 P0-2: the fine interior sampling generated across a
    turning-point handoff must be ACCURATE at multiple interior times
    approaching the singularity, not merely present (gap size alone is
    covered by NearDoubleRootGapCoverage/NearDoubleRootRegression
    below). Cross-checked against the exact closed-form cycloid for the
    matter-only closed universe (Omega_m0=2)."""

    OMEGA_M = 2.0
    H0 = 70.0

    @classmethod
    def _exact(cls, eta):
        om = cls.OMEGA_M
        a = om / (2.0 * (om - 1.0)) * (1.0 - math.cos(eta))
        h0_t = om / (2.0 * (om - 1.0) ** 1.5) * (eta - math.sin(eta))
        return a, h0_t / H0_pgyr(cls.H0)

    def test_interior_samples_match_exact_cycloid_at_several_times(self):
        r = phys.integrate_evolution(H0=self.H0, omega_m=self.OMEGA_M,
                                      omega_r=0.0, omega_de=0.0,
                                      a_i=1.0e-5, a_max=5.0,
                                      step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "turnaround")
        t_arr = np.asarray(r["t_gyr"])
        a_arr = np.asarray(r["a"])
        # eta values spanning well before turnaround (pi) up to just
        # short of it, where the interior time-quantile sampling is
        # exercised most heavily.
        for eta in (2.5, 2.8, 3.0, 3.1, 3.13, 3.14):
            with self.subTest(eta=eta):
                a_exact, t_exact = self._exact(eta)
                idx = int(np.argmin(np.abs(t_arr - t_exact)))
                self.assertLess(
                    abs(t_arr[idx] - t_exact), 0.1,
                    f"no returned sample within 0.1 Gyr of exact "
                    f"t={t_exact:.6g} Gyr (eta={eta})")
                rel_err = abs(a_arr[idx] - a_exact) / a_exact
                self.assertLess(
                    rel_err, 1.0e-3,
                    f"eta={eta}: nearest-sample a={a_arr[idx]:.8g} vs "
                    f"exact a={a_exact:.8g} (rel err {rel_err:.3g})")


class NearDoubleRootGapCoverage(unittest.TestCase):
    """Codex Audit 5 P0-2, gap-size aspect: for a TRUE near-double-root
    turnaround (not just the loitering/no-root case already covered by
    NearDoubleRootRegression.test_confirmed_no_root_reaches_a_max_
    without_large_gaps), the fine interior sampling across the handoff
    must leave no large unrepresented time gap -- this is the exact
    shape of case Codex's original finding demonstrated (a proactive
    handoff firing far from the true turnaround)."""

    def test_no_large_time_gap_near_true_double_root(self):
        omega_m = NearDoubleRootRegression.OMEGA_M
        ode_crit = NearDoubleRootRegression._find_critical_omega_de()
        omega_de = ode_crit - 1.0e-7  # just below critical: a real,
                                       # narrowly-separated root pair
        r = phys.integrate_evolution(
            H0=NearDoubleRootRegression.H0, omega_m=omega_m, omega_r=0.0,
            omega_de=omega_de, w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=20.0,
            step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "turnaround")
        t_arr = np.asarray(r["t_gyr"])
        max_gap = float(np.diff(t_arr).max())
        self.assertLess(
            max_gap, 20.0,
            f"max time gap {max_gap:.4g} Gyr across the turnaround "
            "handoff is too large; interior samples may not be covering "
            "the approach to the true near-double root")


class HardBoundInvariantSweep(unittest.TestCase):
    """A general sweep, beyond the single-family AMaxHardBoundInvariant
    case above, asserting max(a)<=a_max and max(t)<=t_max_gyr across a
    variety of configurations and scales -- the literal, tolerance-free
    contract Codex Audit 5 P1-1 identified as being at risk."""

    def test_a_and_t_never_exceed_requested_bounds(self):
        configs = [
            dict(H0=70.0, omega_m=0.3, omega_r=9.24e-5, omega_de=None,
                 w0=-1.0, wa=0.0, a_max=5.0, t_max=None),
            dict(H0=70.0, omega_m=1.5, omega_r=0.0, omega_de=0.0,
                 w0=-1.0, wa=0.0, a_max=3.0, t_max=None),
            dict(H0=70.0, omega_m=2.0, omega_r=0.0, omega_de=0.0,
                 w0=-1.0, wa=0.0, a_max=5.0, t_max=10.0),
            dict(H0=67.4, omega_m=0.315, omega_r=9.24e-5, omega_de=None,
                 w0=-1.2, wa=0.3, a_max=10.0, t_max=None),
            dict(H0=70.0, omega_m=1.5, omega_r=0.0, omega_de=0.005,
                 w0=-1.0, wa=0.0, a_max=20.0, t_max=None),
        ]
        for kw in configs:
            with self.subTest(**kw):
                r = phys.integrate_evolution(
                    H0=kw["H0"], omega_m=kw["omega_m"],
                    omega_r=kw["omega_r"], omega_de=kw["omega_de"],
                    w0=kw["w0"], wa=kw["wa"], a_i=1.0e-8,
                    a_max=kw["a_max"], t_max_gyr=kw["t_max"],
                    step_frac=0.005)
                self.assertLessEqual(max(r["a"]), kw["a_max"] + 1.0e-9)
                if kw["t_max"] is not None:
                    self.assertLessEqual(max(r["t_gyr"]),
                                          kw["t_max"] + 1.0e-9)


class ContinueCollapseTMaxTruncation(unittest.TestCase):
    """Gemini Audit 5 finding #2: mirroring the contracting branch for
    --continue_collapse must not silently extend the RETURNED trajectory
    past a --t_max that falls strictly on the CONTRACTING branch
    (t_turn < t_max_gyr < 2*t_turn), even though the forward pass itself
    correctly reports a genuine turnaround rather than a t_max stop."""

    H0 = 70.0
    OMEGA_M = 2.0

    @classmethod
    def _exact_cycloid(cls, eta):
        # Same closed-form cycloid as CycloidInteriorAccuracy, valid for
        # eta in [0, 2*pi] (turnaround at eta=pi, Big Crunch at
        # eta=2*pi by the same time-reversal symmetry as the mirrored
        # branch itself): a(eta) and t(eta) in Gyr, computed
        # independently of physics_cosmo.py's own quadrature/bisection.
        om = cls.OMEGA_M
        a = om / (2.0 * (om - 1.0)) * (1.0 - math.cos(eta))
        h0_t = om / (2.0 * (om - 1.0) ** 1.5) * (eta - math.sin(eta))
        return a, h0_t / H0_pgyr(cls.H0)

    @classmethod
    def _eta_at_time(cls, t_target, n_iter=200):
        # t(eta) is monotonically increasing on [pi, 2*pi] (the
        # contracting branch), so invert it by plain bisection -- an
        # independent check, never calling back into physics_cosmo.py.
        lo, hi = math.pi, 2.0 * math.pi
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if cls._exact_cycloid(mid)[1] < t_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def test_t_max_on_contracting_branch_truncates_the_mirror(self):
        # Copilot Audit 6 P0-1 / Codex Audit 6 P1-1 (independently):
        # an earlier version merely filtered the mirrored samples to
        # t <= t_max without inserting the actual state AT t_max, so
        # the returned trajectory silently stopped materially earlier
        # than the requested (and claimed) stop_reason="t_max" time.
        # This strengthens the original (upper-bound-only) assertion to
        # equality at t_max, the exact cycloid value of a there, and the
        # sign of H at that endpoint -- per both audits' explicit ask.
        t_turn = math.pi / H0_pgyr(self.H0)
        t_max = 1.5 * t_turn  # strictly between t_turn and 2*t_turn
        r = phys.integrate_evolution(
            H0=self.H0, omega_m=self.OMEGA_M, omega_r=0.0, omega_de=0.0,
            a_i=1.0e-5, a_max=5.0, t_max_gyr=t_max, step_frac=0.005,
            continue_collapse=True)
        s = r["summary"]
        t_arr = np.asarray(r["t_gyr"])
        a_arr = np.asarray(r["a"])
        H_arr = np.asarray(r["H_kms_mpc"])
        self.assertIsNotNone(s["turnaround"])
        self.assertEqual(s["stop_reason"], "t_max")
        # Hard bound: no returned sample may exceed t_max.
        self.assertLessEqual(float(t_arr.max()), t_max + 1.0e-9)
        # Equality, not merely non-exceedance: the final sample must
        # land exactly (to numerical tolerance) at t_max.
        self.assertAlmostEqual(float(t_arr[-1]), t_max, delta=1.0e-8)
        # The final scale factor must match the mirrored forward
        # solution -- the exact cycloid value at t_max, computed here
        # independently of physics_cosmo.py's own quadrature/bisection.
        eta_at_tmax = self._eta_at_time(t_max)
        a_exact, _t_exact = self._exact_cycloid(eta_at_tmax)
        self.assertAlmostEqual(float(a_arr[-1]), a_exact, delta=1.0e-6)
        # Strictly increasing time throughout, and monotone-decreasing a
        # over the final stretch of the mirrored (contracting) tail --
        # deliberately just the last handful of samples rather than the
        # whole array's second half, since the turning-point handoff's
        # own fine interior sampling can still be expanding-branch data
        # well past the midpoint of the array by index.
        self.assertTrue(np.all(np.diff(t_arr) > 0))
        self.assertTrue(np.all(np.diff(a_arr[-20:]) <= 0))
        # H < 0 at the returned endpoint: it is genuinely on the
        # contracting branch, not an expanding-branch value reattached
        # with the wrong sign.
        self.assertLess(float(H_arr[-1]), 0.0)
        # total_lifetime_gyr still reports the FULL mirrored lifetime;
        # only the RETURNED array is truncated at t_max.
        self.assertAlmostEqual(s["total_lifetime_gyr"], 2.0 * t_turn,
                                delta=1.0e-6)

    def test_t_max_at_several_fractions_between_turnaround_and_full_lifetime(self):
        # Additional fractions between t_turn and 2*t_turn (Copilot
        # Audit 6 P0-1's explicit ask), including one deliberately
        # between two ordinary mirrored samples and one very close to
        # the full lifetime.
        t_turn = math.pi / H0_pgyr(self.H0)
        for frac in (1.05, 1.3, 1.5, 1.7, 1.95, 1.999):
            with self.subTest(frac=frac):
                t_max = frac * t_turn
                r = phys.integrate_evolution(
                    H0=self.H0, omega_m=self.OMEGA_M, omega_r=0.0,
                    omega_de=0.0, a_i=1.0e-5, a_max=5.0,
                    t_max_gyr=t_max, step_frac=0.005,
                    continue_collapse=True)
                t_arr = np.asarray(r["t_gyr"])
                a_arr = np.asarray(r["a"])
                self.assertLessEqual(float(t_arr.max()), t_max + 1.0e-9)
                self.assertAlmostEqual(float(t_arr[-1]), t_max, delta=1.0e-7)
                eta_at_tmax = self._eta_at_time(t_max)
                a_exact, _ = self._exact_cycloid(eta_at_tmax)
                self.assertAlmostEqual(float(a_arr[-1]), a_exact,
                                        delta=1.0e-5)

    def test_t_max_past_full_mirror_reports_full_lifetime_untruncated(self):
        t_turn = math.pi / H0_pgyr(self.H0)
        r = phys.integrate_evolution(
            H0=self.H0, omega_m=self.OMEGA_M, omega_r=0.0, omega_de=0.0,
            a_i=1.0e-5, a_max=5.0, t_max_gyr=3.0 * t_turn,
            step_frac=0.005, continue_collapse=True)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "turnaround")
        self.assertAlmostEqual(float(max(r["t_gyr"])), 2.0 * t_turn,
                                delta=1.0e-5)


class BigBangConnectivity(unittest.TestCase):
    """Codex Audit 6 P0-1: E(a_i)^2 > 0 alone does not certify that a_i
    lies on a branch continuously connected back to the true Big Bang
    at a=0 -- there can be a hidden forbidden interval strictly between
    them, or E(a)^2 itself can tend to a NEGATIVE value as a->0 even
    though it is positive at a_i. Both reproducers below are Codex's
    own exact cases (a hidden two-root forbidden interval below a_i
    where matter still dominates asymptotically, and a CPL dark-energy
    term whose negative-coefficient leading power dominates negatively
    as a->0), plus a standard-connected-model control that must NOT
    raise."""

    def test_hidden_forbidden_interval_below_a_i_is_rejected(self):
        # Large negative curvature creates a genuine forbidden interval
        # (E(a)^2 < 0) between a~1e-6 and a~0.6, even though matter
        # dominates (positively) as a->0 and DE dominates (positively)
        # again near a_i=0.99: E(a_i)^2 > 0 by itself hides this.
        om_m, om_r = 1.0e-6, 0.0
        om_k = -0.5
        om_de = 1.0 - om_m - om_r - om_k
        with self.assertRaises(ValueError) as ctx:
            phys.integrate_evolution(
                H0=70.0, omega_m=om_m, omega_r=om_r, omega_de=om_de,
                w0=-1.0, wa=0.0, a_i=0.99, a_max=1.5, step_frac=0.01)
        self.assertIn("forbidden interval", str(ctx.exception))

    def test_negative_asymptotic_de_dominance_is_rejected(self):
        # w0=1, wa=0 gives a CPL leading power n_de=-6, more negative
        # than radiation's -4, so a small-magnitude negative omega_de
        # dominates E(a)^2 NEGATIVELY as a->0 even though radiation
        # keeps E(a_i)^2 > 0 at the requested a_i=1e-8.
        om_m, om_r = 0.3, 9.24e-5
        om_de, w0, wa = -1.0e-25, 1.0, 0.0
        with self.assertRaises(ValueError) as ctx:
            phys.integrate_evolution(
                H0=70.0, omega_m=om_m, omega_r=om_r, omega_de=om_de,
                w0=w0, wa=wa, a_i=1.0e-8, a_max=2.0, step_frac=0.01)
        self.assertIn("negative", str(ctx.exception))

    def test_standard_connected_model_is_not_rejected(self):
        # Control: an ordinary flat LambdaCDM-like model must integrate
        # cleanly -- the connectivity certification must not be a
        # false-positive trap for the common case.
        r = phys.integrate_evolution(
            H0=67.4, omega_m=0.315, omega_r=9.24e-5, omega_de=None,
            w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=2.0, step_frac=0.005)
        self.assertIsNotNone(r["summary"]["age_today_gyr"])

    def test_recollapsing_closed_model_is_not_rejected(self):
        # A genuinely closed, positive-curvature model (no hidden
        # forbidden interval, matter dominates as a->0) must not be
        # rejected by the connectivity check even though it recollapses
        # later in its own history.
        r = phys.integrate_evolution(
            H0=70.0, omega_m=2.0, omega_r=0.0, omega_de=0.0, w0=-1.0,
            wa=0.0, a_i=1.0e-5, a_max=5.0, step_frac=0.005)
        self.assertIsNotNone(r["summary"]["turnaround"])


class BigRipOverflowSafety(unittest.TestCase):
    """Codex Audit 6 P0-2: the Big-Rip dominance-scale computation must
    never overflow, even for w0 immediately below -1 (where the OLD
    1/p_phantom exponent diverges), and must not count non-negative
    matter/radiation as hazards that could cause a future recollapse --
    only negative curvature can. Exercises exactly the w0 regime Codex's
    reproducer targeted, across flat/open/closed curvature."""

    def test_w0_immediately_below_minus_one_never_overflows(self):
        w0_values = (-1.000001, -1.0 - 1.0e-12,
                     math.nextafter(-1.0, -math.inf))
        omega_k_values = (0.0, 0.01, -0.01)  # flat, open, closed
        for w0 in w0_values:
            for omega_k in omega_k_values:
                with self.subTest(w0=w0, omega_k=omega_k):
                    om_m, om_r = 0.3, 9.24e-5
                    om_de = 1.0 - om_m - om_r - omega_k
                    try:
                        r = phys.integrate_evolution(
                            H0=70.0, omega_m=om_m, omega_r=om_r,
                            omega_de=om_de, w0=w0, wa=0.0, a_i=1.0e-8,
                            a_max=5.0, step_frac=0.01)
                    except OverflowError:
                        self.fail(
                            f"OverflowError for w0={w0!r}, "
                            f"omega_k={omega_k!r}")
                    s = r["summary"]
                    self.assertIn(
                        s["fate_status"],
                        ("big_rip", "future_recollapse", "unresolved"))


class StructuredFateFields(unittest.TestCase):
    """Codex Audit 6 P1-2 / Copilot Audit 6 P2-2: the model's ultimate
    physical fate must be exposed as structured fields
    (fate_status/future_turnaround_a/fate_search_limit_a), not only as
    prose warnings that --compare mode and CSV consumers cannot parse."""

    def test_big_rip_fate_status(self):
        r = phys.integrate_evolution(
            H0=70.0, omega_m=0.3, omega_r=9.24e-5,
            omega_de=0.7 - 9.24e-5, w0=-1.2, wa=0.0, a_i=1.0e-8,
            a_max=60.0, step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["fate_status"], "big_rip")
        self.assertIsNotNone(s["big_rip_gyr"])

    def test_future_recollapse_fate_status(self):
        # Same closed-phantom-recollapse-beyond-a_max configuration as
        # BigRipPrecedenceRegression.test_recollapse_beyond_a_max_suppresses_big_rip
        # above (Codex Audit 5's exact reproducer, independent roots at
        # a~3.113 and a~9.432).
        r = phys.integrate_evolution(
            H0=70.0, omega_m=1.5, omega_r=0.0, omega_de=0.001, w0=-1.2,
            wa=0.0, a_i=1.0e-8, a_max=1.5, step_frac=0.005)
        s = r["summary"]
        self.assertIsNone(s["turnaround"])
        self.assertEqual(s["fate_status"], "future_recollapse")
        self.assertIsNotNone(s["future_turnaround_a"])

    def test_recollapse_fate_status(self):
        r = phys.integrate_evolution(
            H0=70.0, omega_m=2.0, omega_r=0.0, omega_de=0.0, w0=-1.0,
            wa=0.0, a_i=1.0e-5, a_max=5.0, step_frac=0.005)
        self.assertEqual(r["summary"]["fate_status"], "recollapse")

    def test_no_fate_question_is_none(self):
        r = phys.integrate_evolution(
            H0=67.4, omega_m=0.315, omega_r=9.24e-5, omega_de=None,
            w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=2.0, step_frac=0.005)
        self.assertIsNone(r["summary"]["fate_status"])


class ModelVersionBump(unittest.TestCase):
    """Copilot Audit 6 P1-3 / Codex Audit 6 P1-5: MODEL_VERSION must
    distinguish this materially changed build from the prior release
    that lacked the Audit-6 fixes."""

    def test_version_differs_from_prior_release(self):
        self.assertNotEqual(phys.MODEL_VERSION, "1.2.0")


class MatterRadiationEqualityReachability(unittest.TestCase):
    """Codex Audit 6 P1-3: a_eq_rm = omega_r/omega_m is a pure algebraic
    identity, independent of the integrated trajectory -- unlike
    a_eq_mde/a_accel (already gated by a_reached), it was reported even
    when the model recollapsed long before ever reaching that scale
    factor."""

    def test_a_eq_rm_none_when_recollapse_precedes_it(self):
        # Codex's exact reproducer: recollapses at a~1.005, far short
        # of a_eq_rm = omega_r/omega_m = 100.
        r = phys.integrate_evolution(
            H0=70.0, omega_m=1.0, omega_r=100.0, omega_de=0.0, w0=-1.0,
            wa=0.0, a_i=1.0e-8, a_max=5.0, step_frac=0.01)
        s = r["summary"]
        self.assertIsNotNone(s["turnaround"])
        self.assertIsNone(s["a_eq_rm"])

    def test_a_eq_rm_reported_when_reachable(self):
        r = phys.integrate_evolution(
            H0=67.4, omega_m=0.315, omega_r=9.24e-5, omega_de=None,
            w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=2.0, step_frac=0.005)
        s = r["summary"]
        self.assertIsNotNone(s["a_eq_rm"])
        self.assertAlmostEqual(s["a_eq_rm"], 9.24e-5 / 0.315, places=9)


class TotalLifetimeUnnesting(unittest.TestCase):
    """Gemini Audit 6: total_lifetime_gyr is knowable analytically the
    instant a genuine turnaround is found, independent of whether the
    optional --continue_collapse array-mirroring was requested."""

    def test_total_lifetime_reported_without_continue_collapse(self):
        r = phys.integrate_evolution(
            H0=70.0, omega_m=2.0, omega_r=0.0, omega_de=0.0, w0=-1.0,
            wa=0.0, a_i=1.0e-5, a_max=5.0, step_frac=0.005,
            continue_collapse=False)
        s = r["summary"]
        self.assertIsNotNone(s["turnaround"])
        self.assertIsNotNone(s["total_lifetime_gyr"])
        self.assertAlmostEqual(
            s["total_lifetime_gyr"], 2.0 * s["turnaround"]["t_turn_gyr"],
            places=9)


class NaNPolicyNearDoubleRoot(unittest.TestCase):
    """Codex Audit 6 P2-1: only a point where E(a)^2 <= 0 exactly may be
    masked to NaN in Omega_i(a) -- a small-but-strictly-positive E(a)^2
    near a loitering near-double-root dip has large-but-finite,
    physically meaningful Omega values that must NOT be masked."""

    def test_only_non_positive_e2_points_are_nan(self):
        # A loitering configuration with a near-double-root dip that
        # stays strictly positive throughout (no actual turnaround).
        om_m, om_r = 0.3, 9.24e-5
        om_k = -1.0e-6
        om_de = 1.0 - om_m - om_r - om_k
        r = phys.integrate_evolution(
            H0=70.0, omega_m=om_m, omega_r=om_r, omega_de=om_de,
            w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=5.0, step_frac=0.001)
        a_arr = np.asarray(r["a"])
        om_arr = np.asarray(r["Om"])
        e2_arr = phys.E2(a_arr, om_m, om_r, om_k, om_de, -1.0, 0.0)
        is_nan = np.isnan(om_arr)
        # NaN exactly where, and only where, E(a)^2 <= 0.
        self.assertTrue(np.array_equal(is_nan, e2_arr <= 0.0))
        # If this configuration has any strictly-positive-but-tiny
        # E(a)^2 point, it must be finite (not NaN) with a
        # correspondingly large Omega value -- the actual physical
        # scenario Codex Audit 6 P2-1 identified as wrongly masked.
        tiny_positive = (e2_arr > 0.0) & (e2_arr < 1.0e-9)
        if np.any(tiny_positive):
            self.assertTrue(np.all(np.isfinite(om_arr[tiny_positive])))


@unittest.skipIf(REPO_ROOT is None,
                 "main.py could not be located near this test file in any "
                 "supported layout (normal repo/tests/ or a flattened "
                 "upload directory); skipping CLI subprocess tests rather "
                 "than failing on what may be an upload artifact.")
class CLISmokeTest(unittest.TestCase):
    """A real CLI subprocess smoke test: --version, one successful run,
    and one deliberately invalid run, each checked end-to-end through
    main.py rather than only through the library API."""

    def test_version_flag(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--version"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0)
        self.assertIn(phys.MODEL_VERSION, result.stdout + result.stderr)

    def test_successful_run_exits_zero_and_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "main.py", "--mode", "evolve",
                 "--omega_m", "0.3", "--no_plot", "--csvdir", tmp],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                any(f.endswith(".csv") for f in os.listdir(tmp)),
                "expected a CSV file to have been written")

    def test_invalid_input_exits_nonzero_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "main.py", "--mode", "evolve",
                 "--omega_m", "0.3", "--w0", "1000.0", "--wa", "1000.0",
                 "--no_plot", "--csvdir", tmp],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Cosmology_expansion_simulator:", result.stderr)


class CSVFieldAssertions(unittest.TestCase):
    """Codex Audit 5 item 12: assert the exact model version and the
    newly added resolved-parameter provenance fields actually appear in
    the CSV, not merely that SOME header/rows exist (already covered by
    CSVProvenance above)."""

    def test_csv_provenance_contains_resolved_fields_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            drv.run(mode="evolve", H0=70.0, omega_m=1.5, omega_r=0.0,
                    omega_de=None, w0=-1.0, wa=0.0, a_i=1.0e-8,
                    a_max=5.0, t_max=None, step_frac=0.005,
                    continue_collapse=False, no_plot=True, csvdir=tmp)
            files = [f for f in os.listdir(tmp) if f.endswith(".csv")]
            self.assertEqual(len(files), 1)
            with open(os.path.join(tmp, files[0])) as fh:
                text = fh.read()
            self.assertIn(f"version {phys.MODEL_VERSION}", text)
            for key in ("omega_de0_resolved", "a_turn", "t_turn_gyr",
                        "fate_status", "n_forward_iterations", "n_output_samples"):
                self.assertIn(key, text)
            # This configuration genuinely recollapses, so a_turn must
            # be a real resolved number, not a None placeholder.
            self.assertNotIn("a_turn = None", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
