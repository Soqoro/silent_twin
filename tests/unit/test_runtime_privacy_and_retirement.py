from __future__ import annotations

import unittest

from silenttwin.attackers import BayesianAttacker, MockLLMAttacker
from silenttwin.experiments.trial_runner import TrialMode, TrialRunner, TrialSpec
from silenttwin.runtime import SessionRetiredError, make_runtime
from silenttwin.schemas import SessionState
from silenttwin.worlds import build_world_pair


def transcript_for(runtime_name: str, world) -> bytes:
    runtime = make_runtime(runtime_name, world=world, seed=123)
    session = runtime.start_session()
    runtime.run_actions(session, world.probes)
    return session.transcript.canonical_bytes()


class RuntimePrivacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pair = build_world_pair("email", seed=11)

    def test_verdict_correlated_generic_failure_is_distinguishable(self) -> None:
        left = transcript_for("generic_failure", self.pair.theta0)
        right = transcript_for("generic_failure", self.pair.theta1)
        self.assertNotEqual(left, right)

    def test_exact_silenttwin_is_byte_identical(self) -> None:
        left = transcript_for("silenttwin", self.pair.theta0)
        right = transcript_for("silenttwin", self.pair.theta1)
        self.assertEqual(left, right)

    def test_bayesian_attacker_uses_the_known_finite_state_likelihood(self) -> None:
        for state in ("theta0", "theta1"):
            with self.subTest(state=state):
                result = TrialRunner().run(
                    TrialSpec(
                        world_pair=self.pair,
                        actual_state=state,
                        runtime="generic_failure",
                        attacker=BayesianAttacker(seed=123),
                        query_budget=1,
                        mode=TrialMode.INFERENCE_ONLY,
                        seed=123,
                    )
                )
                self.assertEqual(state, result.prediction.prediction)
                self.assertGreater(result.prediction.posterior[state], 0.99)

        exact = TrialRunner().run(
            TrialSpec(
                world_pair=self.pair,
                actual_state="theta1",
                runtime="silenttwin",
                attacker=BayesianAttacker(seed=123),
                query_budget=1,
                mode=TrialMode.INFERENCE_ONLY,
                seed=123,
            )
        )
        self.assertEqual(
            {"theta0": 0.5, "theta1": 0.5}, dict(exact.prediction.posterior)
        )

    def test_mock_llm_agent_is_offline_and_deterministic(self) -> None:
        def run_once():
            return TrialRunner().run(
                TrialSpec(
                    world_pair=self.pair,
                    actual_state="theta1",
                    runtime="generic_failure",
                    attacker=MockLLMAttacker(seed=5),
                    query_budget=1,
                    mode=TrialMode.SINGLE_FINAL_EFFECT,
                    seed=5,
                )
            )

        first = run_once()
        second = run_once()
        self.assertEqual(first.prediction.prediction, second.prediction.prediction)
        self.assertEqual(first.selected_final_exploit, second.selected_final_exploit)

    def test_settlement_is_not_visible_after_silenttwin_retirement(self) -> None:
        runtime = make_runtime("silenttwin", world=self.pair.theta0)
        session = runtime.start_session()
        runtime.run_actions(session, self.pair.theta0.workflow_actions[:2])
        before = session.transcript.canonical_bytes()
        finalization = runtime.finalize(session)
        self.assertEqual(SessionState.COMMITTED, session.state)
        self.assertEqual(before, session.transcript.canonical_bytes())
        self.assertEqual(1, len(runtime.trusted_reporter.reports))
        self.assertEqual(
            finalization.trusted_report, runtime.trusted_reporter.reports[0]
        )
        self.assertNotIn(b"settlement_message\":{", before)

    def test_retired_context_rejects_every_channel(self) -> None:
        runtime = make_runtime("silenttwin", world=self.pair.theta0)
        session = runtime.start_session()
        session.tool_invoker = lambda name, args: (name, args)
        session.real_world_reader = lambda key: key
        session.retire()

        operations = (
            lambda: session.invoke_tool("tool", {}),
            lambda: session.receive_callback({"status": "done"}),
            lambda: session.read_real_world("balance"),
            lambda: session.receive_settlement({"status": "committed"}),
            lambda: session.retain_memory({"value": "trusted repair"}),
        )
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(SessionRetiredError):
                    operation()


if __name__ == "__main__":
    unittest.main()
