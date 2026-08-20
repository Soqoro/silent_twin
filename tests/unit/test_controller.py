from __future__ import annotations

import unittest

from silenttwin.runtime.controller import (
    AtomicityError,
    DependencyAwareController,
    InadmissibleEffectError,
)
from silenttwin.schemas import MonitorDecision, StagedEffect, TruePolicyLabel


def effect(
    effect_id: str,
    *,
    allowed: bool = True,
    data: tuple[str, ...] = (),
    control: tuple[str, ...] = (),
    group: str | None = None,
) -> StagedEffect:
    return StagedEffect(
        effect_id=effect_id,
        payload_digest=f"payload-{effect_id}",
        destination_digest=f"destination-{effect_id}",
        monitor_decision=(
            MonitorDecision.ALLOW if allowed else MonitorDecision.BLOCK
        ),
        true_policy_label=(
            TruePolicyLabel.PERMITTED if allowed else TruePolicyLabel.PROHIBITED
        ),
        policy_epoch=3,
        data_dependencies=data,
        control_dependencies=control,
        atomic_group=group,
    )


class ControllerTests(unittest.TestCase):
    def test_rejected_effect_cannot_commit(self) -> None:
        controller = DependencyAwareController(
            [effect("rejected", allowed=False)], expected_policy_epoch=3
        )
        with self.assertRaises(InadmissibleEffectError):
            controller.issue_commit_token("rejected")
        report = controller.commit_all()
        self.assertNotIn("rejected", report.committed_effect_ids)
        self.assertIn("rejected", report.rejected_effect_ids)

    def test_rejected_data_dependency_discards_descendant(self) -> None:
        controller = DependencyAwareController(
            [
                effect("root", allowed=False),
                effect("data-child", data=("root",)),
            ],
            expected_policy_epoch=3,
        )
        report = controller.commit_all()
        self.assertNotIn("data-child", report.committed_effect_ids)
        self.assertIn("data-child", report.discarded_effect_ids)

    def test_rejected_control_dependency_discards_descendant(self) -> None:
        controller = DependencyAwareController(
            [
                effect("root", allowed=False),
                effect("control-child", control=("root",)),
            ],
            expected_policy_epoch=3,
        )
        report = controller.commit_all()
        self.assertNotIn("control-child", report.committed_effect_ids)
        self.assertIn("control-child", report.discarded_effect_ids)

    def test_independent_approved_branches_commit(self) -> None:
        controller = DependencyAwareController(
            [
                effect("safe-summary"),
                effect("safe-audit"),
                effect("rejected", allowed=False),
            ],
            expected_policy_epoch=3,
        )
        report = controller.commit_all()
        self.assertEqual(
            {"safe-summary", "safe-audit"}, set(report.committed_effect_ids)
        )

    def test_atomic_groups_commit_all_or_none(self) -> None:
        safe = DependencyAwareController(
            [effect("a", group="bundle"), effect("b", group="bundle")],
            expected_policy_epoch=3,
        )
        safe_report = safe.commit_all()
        self.assertEqual({"a", "b"}, set(safe_report.committed_effect_ids))
        self.assertFalse(safe_report.atomicity_violations)

        mixed = DependencyAwareController(
            [
                effect("safe", group="mixed"),
                effect("rejected", allowed=False, group="mixed"),
            ],
            expected_policy_epoch=3,
        )
        mixed_report = mixed.commit_all()
        self.assertNotIn("safe", mixed_report.committed_effect_ids)
        self.assertNotIn("rejected", mixed_report.committed_effect_ids)
        self.assertFalse(mixed_report.atomicity_violations)

    def test_single_member_of_atomic_group_cannot_commit_directly(self) -> None:
        controller = DependencyAwareController(
            [effect("a", group="bundle"), effect("b", group="bundle")],
            expected_policy_epoch=3,
        )
        token = controller.issue_commit_token("a")
        with self.assertRaises(AtomicityError):
            controller.commit_effect("a", token)


if __name__ == "__main__":
    unittest.main()
