from __future__ import annotations

import json
import unittest

from silenttwin.schemas import AgentVisibleTranscript, VisibleEvent
from silenttwin.worlds import WorldPair, build_world_pair


class WorldPairTests(unittest.TestCase):
    def test_all_templates_have_identical_public_projections(self) -> None:
        for suite in ("email", "files", "calendar", "payments"):
            with self.subTest(suite=suite):
                pair = build_world_pair(suite, seed=7)
                self.assertEqual(
                    pair.theta0.public_projection(), pair.theta1.public_projection()
                )
                self.assertEqual(64, len(pair.public_instance_hash))
                self.assertNotIn(
                    "theta", json.dumps(pair.theta0.public_projection()).lower()
                )

    def test_pair_rejects_a_public_mismatch(self) -> None:
        pair = build_world_pair("email", seed=9)
        pair.theta1.public_task = "different public task"
        with self.assertRaisesRegex(ValueError, "identical initial public projections"):
            WorldPair(pair.theta0, pair.theta1)


class TranscriptBoundaryTests(unittest.TestCase):
    def test_private_state_cannot_enter_visible_transcript(self) -> None:
        with self.assertRaisesRegex(ValueError, "private field"):
            VisibleEvent(
                event_type="tool_result",
                sequence=0,
                value={"private_state": "theta0"},
            )

    def test_private_monitor_decision_cannot_be_nested(self) -> None:
        with self.assertRaisesRegex(ValueError, "private field"):
            VisibleEvent(
                event_type="tool_result",
                sequence=0,
                callback={"nested": {"monitor_decision": "allow"}},
            )

    def test_sealed_transcript_is_append_only(self) -> None:
        transcript = AgentVisibleTranscript()
        transcript.append(VisibleEvent(event_type="tool_result", sequence=0))
        transcript.seal()
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            transcript.append(VisibleEvent(event_type="tool_result", sequence=1))


if __name__ == "__main__":
    unittest.main()
