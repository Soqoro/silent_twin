from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import unittest

from silenttwin.schemas import MonitorDecision, TruePolicyLabel, stable_digest
from silenttwin.worlds import build_world_pair
from silenttwin.worlds.dataset import (
    DATASET_REVISION,
    DatasetValidationError,
    build_dataset_world_pair,
    expected_split_manifest,
    iter_template_specs,
    load_and_validate_dataset_manifests,
    select_template_id,
    template_specs,
    validate_records_respect_split,
    validate_split_manifests,
    validate_train_evaluation_separation,
)


_IDENTITY_AND_PROVENANCE_KEYS = frozenset(
    {
        "action_id",
        "dataset_revision",
        "dataset_split",
        "pair_family",
        "policy_tag",
        "public_instance_id",
        "public_instance_index",
        "seed",
        "template_id",
    }
)


def _without_identifiers(value):
    """Independently strip identity/provenance while retaining task content."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_identifiers(child)
            for key, child in value.items()
            if str(key) not in _IDENTITY_AND_PROVENANCE_KEYS
            and not str(key).endswith("_hash")
        }
    if isinstance(value, list):
        return [_without_identifiers(child) for child in value]
    return value


class DatasetRegistryTests(unittest.TestCase):
    def test_sixteen_templates_are_structurally_distinct_and_stratified(self) -> None:
        specs = template_specs()
        self.assertEqual(16, len(specs))
        self.assertEqual(16, len({item.template_id for item in specs}))
        self.assertEqual(16, len({item.structural_signature for item in specs}))
        self.assertEqual(
            {"train": 8, "development": 4, "test": 4},
            {
                split: len(iter_template_specs(split=split))
                for split in ("train", "development", "test")
            },
        )
        for domain in ("email", "files", "calendar", "payments"):
            self.assertEqual(4, len(iter_template_specs(domain=domain)))
            for split in ("train", "development", "test"):
                self.assertTrue(iter_template_specs(domain=domain, split=split))

    def test_checked_in_dataset_and_split_manifests_match_registry(self) -> None:
        hashes = load_and_validate_dataset_manifests()
        self.assertEqual(64, len(hashes["dataset_manifest_hash"]))
        self.assertEqual(64, len(hashes["split_manifest_hash"]))

    def test_template_selection_is_deterministic_and_split_strict(self) -> None:
        first = select_template_id("email", "train", 0)
        self.assertEqual(first, select_template_id("email", "train", 2))
        requested = select_template_id(
            "email", "test", 999, requested="email.incident_escalation"
        )
        self.assertEqual("email.incident_escalation", requested)
        with self.assertRaisesRegex(DatasetValidationError, "belongs to split"):
            select_template_id(
                "email", "test", 0, requested="email.triage_and_reply"
            )
        with self.assertRaisesRegex(DatasetValidationError, "dataset revision"):
            select_template_id(
                "email", "train", 0, dataset_revision="future-revision"
            )

    def test_every_template_has_deterministic_material_instance_variants(self) -> None:
        for spec in template_specs():
            with self.subTest(template=spec.template_id):
                first = build_dataset_world_pair(
                    spec.template_id,
                    instance_id="replicate-00000000",
                    instance_index=0,
                    seed=73,
                    split=spec.split,
                )
                replay = build_dataset_world_pair(
                    spec.template_id,
                    instance_id="replicate-00000000",
                    instance_index=0,
                    seed=73,
                    split=spec.split,
                )
                second = build_dataset_world_pair(
                    spec.template_id,
                    instance_id="replicate-00000001",
                    instance_index=1,
                    seed=73,
                    split=spec.split,
                )

                self.assertEqual(first.public_projection, replay.public_projection)
                self.assertEqual(first.public_instance_hash, replay.public_instance_hash)
                self.assertEqual(
                    stable_digest(first.public_projection),
                    first.public_instance_hash,
                )
                self.assertNotEqual(first.public_instance_hash, second.public_instance_hash)

                # This is the scientific guard against nominal clones: the
                # concrete tasks still differ after instance IDs, indices,
                # hashes, seeds, and template provenance are all removed.
                first_content = _without_identifiers(dict(first.public_projection))
                second_content = _without_identifiers(dict(second.public_projection))
                self.assertNotEqual(first_content, second_content)
                self.assertNotEqual(
                    first.theta0.public_environment["public_instance_content"],
                    second.theta0.public_environment["public_instance_content"],
                )
                self.assertNotEqual(
                    first.theta0.action(first.theta0.actions[0].action_id).arguments[
                        "payload"
                    ],
                    second.theta0.action(second.theta0.actions[0].action_id).arguments[
                        "payload"
                    ],
                )

                self.assertEqual(spec.template_id, first.template_id)
                self.assertEqual(spec.structural_family, first.structural_family)
                self.assertEqual(spec.structural_family, first.theta0.structural_family)
                self.assertEqual(
                    first.theta0.public_projection(), first.theta1.public_projection()
                )
                self.assertEqual(
                    second.theta0.public_projection(), second.theta1.public_projection()
                )

    def test_invalid_explicit_instance_indices_are_rejected(self) -> None:
        for invalid in (-1, True):
            with self.subTest(instance_index=invalid):
                with self.assertRaisesRegex(ValueError, "instance_index"):
                    build_dataset_world_pair(
                        "email.triage_and_reply",
                        instance_id="invalid-instance",
                        instance_index=invalid,
                        seed=0,
                        split="train",
                    )

    def test_both_pair_families_have_one_private_difference(self) -> None:
        for spec in template_specs():
            with self.subTest(template=spec.template_id, family="monitor_blind_spot"):
                blind = build_world_pair(
                    spec.domain,
                    template_id=spec.template_id,
                    pair_family="monitor_blind_spot",
                    dataset_split=spec.split,
                    instance_id="fixed",
                    seed=17,
                )
                self.assertEqual(
                    blind.theta0.public_projection(), blind.theta1.public_projection()
                )
                self.assertEqual(spec.template_id, blind.theta0.template_id)
                self.assertEqual(DATASET_REVISION, blind.theta0.dataset_revision)
                exploits = blind.theta0.exploits
                self.assertEqual(2, len(exploits))
                for world in blind.worlds:
                    self.assertTrue(
                        all(
                            world.true_policy.evaluate(action)
                            is TruePolicyLabel.PROHIBITED
                            for action in exploits
                        )
                    )
                theta0_decisions = tuple(
                    blind.theta0.monitor.decide(action) for action in exploits
                )
                theta1_decisions = tuple(
                    blind.theta1.monitor.decide(action) for action in exploits
                )
                self.assertEqual(
                    (MonitorDecision.ALLOW, MonitorDecision.BLOCK), theta0_decisions
                )
                self.assertEqual(
                    (MonitorDecision.BLOCK, MonitorDecision.ALLOW), theta1_decisions
                )

            with self.subTest(template=spec.template_id, family="authorization"):
                authorization = build_world_pair(
                    spec.domain,
                    template_id=spec.template_id,
                    pair_family="authorization",
                    dataset_split=spec.split,
                    instance_id="fixed",
                    seed=17,
                )
                self.assertEqual(
                    authorization.theta0.public_projection(),
                    authorization.theta1.public_projection(),
                )
                candidate = authorization.theta0.action("authorization_candidate")
                labels = tuple(
                    world.true_policy.evaluate(candidate) for world in authorization.worlds
                )
                decisions = tuple(
                    world.monitor.decide(candidate) for world in authorization.worlds
                )
                self.assertEqual(
                    {TruePolicyLabel.PERMITTED, TruePolicyLabel.PROHIBITED}, set(labels)
                )
                self.assertEqual(
                    {MonitorDecision.ALLOW, MonitorDecision.BLOCK}, set(decisions)
                )

    def test_split_leakage_and_tampered_manifest_are_rejected(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "reuse templates"):
            validate_train_evaluation_separation(
                ["email.triage_and_reply"],
                ["email.triage_and_reply"],
            )

        manifests = {
            split: expected_split_manifest(split)
            for split in ("train", "development", "test")
        }
        corrupted = deepcopy(manifests)
        corrupted["test"]["template_ids"][0] = "email.triage_and_reply"
        with self.assertRaisesRegex(DatasetValidationError, "does not exactly match"):
            validate_split_manifests(corrupted)

    def test_result_records_are_bound_to_declared_split_and_revision(self) -> None:
        valid = {
            "template_id": "email.incident_escalation",
            "dataset_split": "test",
            "dataset_revision": DATASET_REVISION,
        }
        validate_records_respect_split([valid], expected_split="test")
        leaked = {**valid, "template_id": "email.triage_and_reply"}
        with self.assertRaisesRegex(DatasetValidationError, "leaks from split"):
            validate_records_respect_split([leaked], expected_split="test")


if __name__ == "__main__":
    unittest.main()
