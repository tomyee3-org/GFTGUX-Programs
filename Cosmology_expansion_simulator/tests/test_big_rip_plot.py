"""Big Rip visibility without changing the finite numerical trajectory."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import physics_cosmo as phys
import plot_cosmo as viz
import driver_cosmo as drv


class BigRipPlot(unittest.TestCase):
    def test_pure_phantom_future_matches_exact_solution(self):
        # With no matter/radiation/curvature the remaining lifetime is
        # exactly a_end**(-m)/(m*H0); there is no finite Big Bang origin.
        result = phys.integrate_evolution(omega_m=0, omega_r=0, omega_de=1,
                                           w0=-1.2, a_i=0.5, a_max=70)
        future = viz._big_rip_continuation(result)
        self.assertIsNotNone(future)
        m = 0.3
        rate = phys.H0_per_Gyr(70)
        a_end = result["a"][-1]
        expected_remaining = a_end**(-m) / (m * rate)
        self.assertAlmostEqual(future["t_rip_gyr"] - result["t_gyr"][-1],
                               expected_remaining, places=9)
        time_left = future["t_rip_gyr"] - future["t_gyr"]
        np.testing.assert_allclose(time_left,
                                   future["a"]**(-m) / (m * rate), rtol=1e-10)
        np.testing.assert_allclose(future["H_kms_mpc"],
                                   70 * future["a"]**m, rtol=1e-10)
        np.testing.assert_allclose(future["q"], -1.3, rtol=1e-10)

    def test_requested_model_shows_visible_future_and_preserves_data(self):
        result = phys.integrate_evolution(w0=-1.2, a_max=70)
        original = copy.deepcopy(result)
        future = viz._big_rip_continuation(result)
        self.assertIsNotNone(future)
        self.assertGreater(future["a"][-1], 1e6 * result["a"][-1])
        self.assertTrue(np.all(np.diff(future["a"]) > 0))
        self.assertTrue(np.all(np.diff(future["t_gyr"]) > 0))
        self.assertTrue(np.all(future["t_gyr"] < future["t_rip_gyr"]))
        # The correction to the DE-only estimate must not be hidden:
        # matter makes the remaining lifetime a little shorter.
        self.assertLess(future["t_rip_gyr"], result["summary"]["big_rip_gyr"])
        captured = []
        with mock.patch.object(viz, "_save_and_show",
                               side_effect=lambda fig, *args: captured.append(fig)):
            viz.plot_evolve(result)
        fig = captured[0]
        try:
            for ax in (fig.axes[0], fig.axes[1], fig.axes[3]):
                continuations = [line for line in ax.lines
                                 if line.get_label() == "future continuation (plot only)"]
                self.assertEqual(len(continuations), 1)
                self.assertLess(ax.get_xlim()[0], future["t_rip_gyr"])
                self.assertGreater(ax.get_xlim()[1], future["t_rip_gyr"])
            self.assertTrue(any("infty" in text.get_text()
                                for text in fig.axes[0].texts))
            fig.canvas.draw()
        finally:
            plt.close(fig)
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original[key])
            else:
                self.assertEqual(value, original[key])

    def test_no_extension_for_other_fates_cpl_early_or_time_stop(self):
        for kw in ({}, {"w0": -1.2, "wa": 0.1},
                   {"omega_m": 1.5, "omega_r": 0, "omega_de": 0.001,
                    "w0": -1.2, "a_max": 1.05},
                   {"w0": -1.2, "a_max": 0.5},
                   {"w0": -1.2, "a_max": 70, "t_max_gyr": 30}):
            with self.subTest(kw=kw):
                result = phys.integrate_evolution(**kw)
                self.assertIsNone(viz._big_rip_continuation(result))

    def test_explicit_time_limit_disables_plot_extension_even_if_not_reached(self):
        # At a=70 the run ends around 54 Gyr, before this time limit.
        # The requested t_max must still suppress the automatic future.
        captured = []
        with mock.patch.object(viz, "_save_and_show",
                               side_effect=lambda fig, *args: captured.append(fig)):
            drv.run(mode="evolve", w0=-1.2, a_max=70, t_max=60)
        fig = captured[0]
        try:
            self.assertFalse(any(line.get_label() == "future continuation (plot only)"
                                 for ax in fig.axes for line in ax.lines))
            self.assertLess(fig.axes[0].get_xlim()[1], 60)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
