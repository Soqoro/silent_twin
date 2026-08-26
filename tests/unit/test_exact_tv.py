from __future__ import annotations

from fractions import Fraction
import unittest

from silenttwin.metrics.exact_tv import (
    check_reachable_history_bisimulation,
    compare_enumerated_transcript_distributions,
    enumerate_action_sequences,
    exact_total_variation,
)
from silenttwin.worlds import build_world_pair


class ExactTranscriptDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = build_world_pair(
            "email",
            template_id="email.incident_escalation",
            pair_family="monitor_blind_spot",
            dataset_split="test",
            seed=23,
        )

    def test_total_variation_uses_exact_probability_mass(self) -> None:
        value = exact_total_variation(
            {"a": Fraction(1, 2), "b": Fraction(1, 2)},
            {"a": Fraction(1, 4), "b": Fraction(1, 4), "c": Fraction(1, 2)},
        )
        self.assertEqual(Fraction(1, 2), value)

    def test_exact_silenttwin_is_bisimilar_for_every_adaptive_probe_history(self) -> None:
        report = check_reachable_history_bisimulation(
            self.pair, "silenttwin", max_steps=32, seed=9
        )
        self.assertTrue(report.bisimilar)
        self.assertEqual(Fraction(0), report.maximum_adaptive_total_variation)
        # Public-state quotienting avoids enumerating 2**32 scripted paths.
        self.assertLessEqual(report.checked_public_states, 33)
        self.assertEqual(64, report.checked_transitions)

    def test_generic_failure_has_an_exact_adaptive_counterexample(self) -> None:
        report = check_reachable_history_bisimulation(
            self.pair, "generic_failure", max_steps=16
        )
        self.assertFalse(report.bisimilar)
        self.assertEqual(Fraction(1), report.maximum_adaptive_total_variation)
        self.assertIsNotNone(report.counterexample)
        self.assertEqual(1, len(report.counterexample.action_sequence))

    def test_randomized_feedback_uses_the_complete_channel_distribution(self) -> None:
        report = check_reachable_history_bisimulation(
            self.pair, "randomized_feedback", max_steps=1
        )
        self.assertEqual(
            "exact_randomized_binary_channel_dynamic_program", report.method
        )
        # Bernoulli(3/4) versus Bernoulli(1/4), not one seeded trajectory.
        self.assertEqual(Fraction(1, 2), report.maximum_adaptive_total_variation)
        self.assertFalse(report.bisimilar)

        uninformative = check_reachable_history_bisimulation(
            self.pair,
            "randomized_feedback",
            max_steps=16,
            runtime_kwargs={"truth_probability": 0.5},
        )
        self.assertEqual(Fraction(0), uninformative.maximum_adaptive_total_variation)
        self.assertTrue(uninformative.bisimilar)

    def test_finite_support_enumerator_is_not_a_single_trajectory_check(self) -> None:
        actions = tuple(action.action_id for action in self.pair.theta0.probes)
        histories = enumerate_action_sequences(actions, 2)
        self.assertEqual(7, len(histories))
        exact = compare_enumerated_transcript_distributions(
            self.pair, "silenttwin", histories, seeds=(1, 2)
        )
        leaking = compare_enumerated_transcript_distributions(
            self.pair, "generic_failure", histories, seeds=(1, 2)
        )
        self.assertEqual(Fraction(0), exact.total_variation)
        self.assertGreater(leaking.total_variation, 0)

    def test_enumerator_refuses_unbounded_history_materialization(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_histories"):
            enumerate_action_sequences(("a", "b"), 32, max_histories=1000)


if __name__ == "__main__":
    unittest.main()
