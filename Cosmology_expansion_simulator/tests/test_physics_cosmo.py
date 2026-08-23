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
  * Boundary coincidence: a_max requested one ULP below an exact
    analytic turning point must still report 'turnaround', not flip to
    'a_max' from floating-point noise in the independently bisected
    a_turn -- while an a_max genuinely, non-trivially short of the
    turning point must still be respected as the stopping point.
  * Extreme/pathological CPL (w0, wa) parameters, which must raise a
    clear domain error rather than silently overflow to NaN/inf and
    have that non-finite value mistaken for a physical turning point.
  * The H(t) sign flip on the mirrored contracting branch.
  * Input validation (t_max <= t_i, excessive step counts, non-positive
    age_ref_gyr) and CSV output (headers, rows, and provenance/result
    metadata).
  * That the module's internal _Overshoot control-flow exception never
    escapes the public API.

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
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_cosmo as phys
import driver_cosmo as drv


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


class BoundaryCoincidenceTieBreak(unittest.TestCase):
    """When a_max is requested to sit essentially exactly at a genuine
    turning point, floating-point noise in how a_turn is independently
    bisected must never flip the reported stop_reason away from
    'turnaround' -- the physical event is the same regardless of which
    side of a_turn a_max's last bit happens to round to."""

    def test_a_max_one_ulp_below_analytic_turnaround(self):
        # Pure matter+curvature (no radiation, no dark energy): the
        # turning point is the exact closed-form a_turn = Om/(Om-1),
        # from E(a)^2 = Om*a^-3 + (1-Om)*a^-2 = 0.
        for omega_m in (1.5, 2.5, 3.0, 7.0):
            a_turn_analytic = omega_m / (omega_m - 1.0)
            a_max = np.nextafter(a_turn_analytic, -np.inf)
            with self.subTest(omega_m=omega_m):
                r = phys.integrate_evolution(
                    H0=70.0, omega_m=omega_m, omega_r=0.0, omega_de=0.0,
                    w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=a_max,
                    step_frac=0.005)
                s = r["summary"]
                self.assertEqual(s["stop_reason"], "turnaround")
                self.assertAlmostEqual(r["a"][-1], a_turn_analytic,
                                        places=9)

    def test_a_max_meaningfully_below_turnaround_still_stops_at_a_max(self):
        # A guard against over-correction: a_max that is genuinely,
        # non-trivially short of the turning point (far looser than the
        # tie-break's relative tolerance) must still be respected as
        # the stopping point, not swallowed into "turnaround".
        omega_m = 3.0
        a_turn_analytic = omega_m / (omega_m - 1.0)
        a_max = a_turn_analytic * 0.999
        r = phys.integrate_evolution(
            H0=70.0, omega_m=omega_m, omega_r=0.0, omega_de=0.0,
            w0=-1.0, wa=0.0, a_i=1.0e-8, a_max=a_max, step_frac=0.005)
        s = r["summary"]
        self.assertEqual(s["stop_reason"], "a_max")
        self.assertAlmostEqual(r["a"][-1], a_max, places=9)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
