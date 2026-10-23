"""Regression coverage for the console and unavailable-age presentation."""
import contextlib
import io
import re
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import driver_cosmo as driver
import physics_cosmo as physics
import plot_cosmo as plots


class SummaryPresentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.normal = physics.integrate_evolution(H0=67.4, omega_m=0.315)
        cls.closed = physics.integrate_evolution(omega_m=2, omega_r=0,
                                               omega_de=0)
        cls.eternal = physics.integrate_evolution(omega_m=0, omega_r=0,
                                                omega_de=1, w0=-1)

    @staticmethod
    def summary_text(summary):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            driver._print_evolve_summary(summary)
        return stream.getvalue()

    def test_console_preserves_computed_age_and_turnaround_values(self):
        for result in (self.normal, self.closed, self.eternal):
            summary = result["summary"]
            text = self.summary_text(summary)
            if summary["age_today_gyr"] is not None:
                value = re.search(r"Age today \(a=1\)\s*:\s*(\S+)", text)[1]
                self.assertEqual(float(value), summary["age_today_gyr"])
            else:
                value = re.search(r"Elapsed a_i -> today\s*:\s*(\S+)", text)[1]
                self.assertEqual(float(value), summary["elapsed_ai_to_today_gyr"])
            if summary["turnaround"] is not None:
                a, t = re.search(r"a_turn=(\S+), t_turn=(\S+) Gyr", text).groups()
                self.assertEqual(float(a), summary["turnaround"]["a_turn"])
                self.assertEqual(float(t), summary["turnaround"]["t_turn_gyr"])
                lifetime = re.search(r"Big Bang to Big Crunch\):\s*(\S+)", text)[1]
                self.assertEqual(float(lifetime), summary["total_lifetime_gyr"])

    def test_fate_classification_is_separate_from_run_stop(self):
        expected = {"big_rip": "Big Rip ahead", "recollapse": "recollapse",
                    "future_recollapse": "future recollapse beyond this run",
                    "unresolved": "unresolved"}
        for status, phrase in expected.items():
            with self.subTest(status=status):
                summary = dict(self.normal["summary"], fate_status=status,
                               future_turnaround_a=8, stop_reason="a_max")
                text = self.summary_text(summary)
                self.assertRegex(text, rf"Model fate\s*:\s*{phrase}")
                self.assertRegex(text, r"Run stopped because\s*:\s*a_max")

    def test_unclassified_fate_does_not_certify_eternal_expansion(self):
        text = self.summary_text(self.normal["summary"])
        self.assertNotIn("Model fate", text)
        self.assertNotIn("expands forever", text)

    def test_comparison_distinguishes_past_eternal_and_unreached_today(self):
        truncated = physics.integrate_evolution(a_max=0.5)
        self.assertIsNone(truncated["summary"]["age_today_gyr"])
        self.assertGreater(self.eternal["a"][-1], 1)
        self.assertEqual(self.eternal["summary"]["past_status"], "past_eternal")
        captured = []
        with mock.patch.object(plots, "_save_and_show",
                               side_effect=lambda fig, *args: captured.append(fig)):
            plots.plot_compare(["normal", "de Sitter", "truncated"],
                               [self.normal, self.eternal, truncated])
        fig = captured[0]
        try:
            axis = fig.axes[2]
            labels = [text.get_text() for text in axis.texts]
            self.assertIn("past-eternal\n(no finite age)", labels)
            self.assertIn("$a=1$ not reached", labels)
            self.assertNotIn("never reaches", axis.get_ylabel())
            self.assertEqual(axis.patches[0].get_height(),
                             self.normal["summary"]["age_today_gyr"])
            fig.canvas.draw()
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
