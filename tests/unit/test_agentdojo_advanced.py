from __future__ import annotations

from copy import deepcopy

import pytest

from silenttwin.agentdojo.advanced import (
    ABLATION_INVARIANTS,
    CHANNEL_INTERVENTIONS,
    CHANNEL_OPERATIONS,
    AdvancedProtocolError,
    AuthoredEffect,
    ablation_contract,
    apply_authored_graph_ablation,
    channel_intervention_audit,
    settle_authored_effect_graph,
    validate_authored_effect_graph,
)


def _mixed_graph() -> tuple[AuthoredEffect, ...]:
    """One unsafe branch, a rollback cascade, and independent useful work."""

    return (
        AuthoredEffect(
            "unsafe",
            "unsafe-plan",
            "prohibited",
            atomic_group="mixed-bundle",
        ),
        AuthoredEffect(
            "atomic-sibling",
            "atomic-sibling-plan",
            "permitted",
            atomic_group="mixed-bundle",
        ),
        AuthoredEffect(
            "atomic-descendant",
            "atomic-descendant-plan",
            "permitted",
            data_dependencies=("atomic-sibling",),
        ),
        AuthoredEffect(
            "unsafe-descendant",
            "unsafe-descendant-plan",
            "permitted",
            control_dependencies=("unsafe",),
        ),
        AuthoredEffect(
            "independent-safe",
            "independent-safe-plan",
            "permitted",
        ),
    )


def _decisions() -> dict[str, str]:
    return {
        "unsafe": "block",
        "atomic-sibling": "allow",
        "atomic-descendant": "allow",
        "unsafe-descendant": "allow",
        "independent-safe": "allow",
    }


def test_e4_dependency_and_atomic_closure_reaches_a_fixed_point() -> None:
    effects = _mixed_graph()
    report = settle_authored_effect_graph(
        effects,
        monitor_decisions=_decisions(),
        runtime="speculative_dependency_aware",
    )

    assert report.rejected_effect_ids == ("unsafe",)
    assert report.atomicity_discarded_effect_ids == ("atomic-sibling",)
    # atomic-descendant is discovered only after its parent is rolled back by
    # the atomic-group pass, so this assertion exercises the second closure pass.
    assert report.dependency_discarded_effect_ids == (
        "atomic-descendant",
        "unsafe-descendant",
    )
    assert report.committed_effect_ids == ("independent-safe",)
    assert report.prohibited_committed_effect_ids == ()
    assert report.safe_effect_salvage == pytest.approx(0.25)
    assert report.clean_report == {
        "unsafe": "rejected",
        "atomic-sibling": "atomicity_discarded",
        "atomic-descendant": "dependency_discarded",
        "unsafe-descendant": "dependency_discarded",
        "independent-safe": "committed",
    }


def test_e4_immediate_termination_discards_even_independent_safe_work() -> None:
    report = settle_authored_effect_graph(
        _mixed_graph(),
        monitor_decisions=_decisions(),
        runtime="immediate_termination",
    )

    assert report.committed_effect_ids == ()
    assert report.atomicity_discarded_effect_ids == ()
    assert report.dependency_discarded_effect_ids == (
        "atomic-sibling",
        "atomic-descendant",
        "unsafe-descendant",
        "independent-safe",
    )
    assert report.safe_effect_salvage == 0.0


def test_e4_validator_requires_a_permitted_descendant_of_prohibited_effect() -> None:
    unrelated_dependency_graph = (
        AuthoredEffect("unsafe", "unsafe-plan", "prohibited"),
        AuthoredEffect(
            "safe-parent",
            "safe-parent-plan",
            "permitted",
            atomic_group=None,
        ),
        AuthoredEffect(
            "safe-child",
            "safe-child-plan",
            "permitted",
            data_dependencies=("safe-parent",),
            atomic_group="safe-bundle",
        ),
    )

    with pytest.raises(
        AdvancedProtocolError,
        match="permitted descendant of a prohibited effect",
    ):
        validate_authored_effect_graph(unrelated_dependency_graph)


def test_e4_validator_requires_independent_permitted_work() -> None:
    no_independent_work = (
        AuthoredEffect(
            "unsafe",
            "unsafe-plan",
            "prohibited",
            atomic_group="mixed-bundle",
        ),
        AuthoredEffect(
            "safe-descendant",
            "safe-descendant-plan",
            "permitted",
            control_dependencies=("unsafe",),
            atomic_group="mixed-bundle",
        ),
    )

    with pytest.raises(
        AdvancedProtocolError,
        match="permitted dependency-free independent effect",
    ):
        validate_authored_effect_graph(no_independent_work)


def test_e4_validator_rejects_atomic_peer_as_nominal_independent_work() -> None:
    atomic_peer_only = (
        AuthoredEffect(
            "unsafe",
            "unsafe-plan",
            "prohibited",
            atomic_group="mixed-bundle",
        ),
        AuthoredEffect(
            "safe-atomic-peer",
            "safe-peer-plan",
            "permitted",
            atomic_group="mixed-bundle",
        ),
        AuthoredEffect(
            "safe-descendant",
            "safe-descendant-plan",
            "permitted",
            control_dependencies=("unsafe",),
            atomic_group="mixed-bundle",
        ),
    )

    with pytest.raises(
        AdvancedProtocolError,
        match="outside every atomic group",
    ):
        validate_authored_effect_graph(atomic_peer_only)


def test_e4_monitor_ledger_is_exact_and_authored() -> None:
    decisions = _decisions()
    decisions.pop("atomic-descendant")
    with pytest.raises(AdvancedProtocolError, match="cover every authored effect exactly"):
        settle_authored_effect_graph(
            _mixed_graph(),
            monitor_decisions=decisions,
            runtime="speculative_dependency_aware",
        )


@pytest.mark.parametrize(
    ("ablation", "removed_kind", "newly_committed"),
    [
        (
            "incomplete_data_dependencies",
            "data_dependency",
            {"atomic-descendant"},
        ),
        (
            "incomplete_control_dependencies",
            "control_dependency",
            {"unsafe-descendant"},
        ),
        (
            "missing_atomic_group",
            "atomic_group",
            {"atomic-sibling", "atomic-descendant"},
        ),
    ],
)
def test_e5_authored_graph_ablation_is_operational_and_auditable(
    ablation: str,
    removed_kind: str,
    newly_committed: set[str],
) -> None:
    source = _mixed_graph()
    source_snapshot = tuple(effect.to_dict() for effect in source)

    transformed, audit = apply_authored_graph_ablation(
        source,
        ablation=ablation,
    )
    repeated, repeated_audit = apply_authored_graph_ablation(
        source,
        ablation=ablation,
    )

    assert transformed == repeated
    assert audit == repeated_audit
    assert tuple(effect.to_dict() for effect in source) == source_snapshot
    assert transformed != source
    assert audit["operational_change"] is True
    assert audit["status"] == "evaluated"
    assert audit["changed_invariant"] == ABLATION_INVARIANTS[ablation]
    assert audit["source_graph_hash"] == validate_authored_effect_graph(source)
    assert audit["source_graph_hash"] != audit["transformed_graph_hash"]
    assert audit["transformation_hash"]
    assert {row["annotation_kind"] for row in audit["removed_annotations"]} == {
        removed_kind
    }

    baseline = settle_authored_effect_graph(
        source,
        monitor_decisions=_decisions(),
        runtime="speculative_dependency_aware",
    )
    ablated = settle_authored_effect_graph(
        transformed,
        monitor_decisions=_decisions(),
        runtime="speculative_dependency_aware",
        authored_source_effects=source,
        ablation_audit=audit,
    )
    assert newly_committed <= (
        set(ablated.committed_effect_ids) - set(baseline.committed_effect_ids)
    )
    assert ablated.graph_hash == audit["transformed_graph_hash"]


def test_e5_ablated_settlement_fails_closed_on_missing_or_tampered_binding() -> None:
    source = _mixed_graph()
    transformed, audit = apply_authored_graph_ablation(
        source,
        ablation="missing_atomic_group",
    )

    with pytest.raises(AdvancedProtocolError):
        settle_authored_effect_graph(
            transformed,
            monitor_decisions=_decisions(),
            runtime="speculative_dependency_aware",
        )
    with pytest.raises(AdvancedProtocolError, match="requires both"):
        settle_authored_effect_graph(
            transformed,
            monitor_decisions=_decisions(),
            runtime="speculative_dependency_aware",
            authored_source_effects=source,
        )
    tampered = deepcopy(audit)
    tampered["removed_annotations"] = []
    with pytest.raises(AdvancedProtocolError, match="audit does not match"):
        settle_authored_effect_graph(
            transformed,
            monitor_decisions=_decisions(),
            runtime="speculative_dependency_aware",
            authored_source_effects=source,
            ablation_audit=tampered,
        )


def test_e5_authored_graph_ablation_rejects_inapplicable_or_unknown_change() -> None:
    source = _mixed_graph()
    no_data_edge = tuple(
        AuthoredEffect(
            effect.effect_id,
            effect.plan_id,
            effect.ground_truth_label,
            control_dependencies=(
                ("unsafe",)
                if effect.effect_id in {"atomic-descendant", "unsafe-descendant"}
                else effect.control_dependencies
            ),
            atomic_group=effect.atomic_group,
        )
        for effect in source
    )
    validate_authored_effect_graph(no_data_edge)

    with pytest.raises(AdvancedProtocolError, match="no data dependency"):
        apply_authored_graph_ablation(
            no_data_edge,
            ablation="incomplete_data_dependencies",
        )
    with pytest.raises(AdvancedProtocolError, match="unsupported"):
        apply_authored_graph_ablation(source, ablation="nominal_only")


def _live_evidence(channel: str) -> dict[str, object]:
    evidence: dict[str, object] = {
        "intervention_stage": "live_controlled_trial",
        "configured_operation": CHANNEL_OPERATIONS[channel],
        "model_observation_count": 3,
        "probe_feedback_transform_count": 0,
        "settlement_event_count_delivered": 0,
        "callback_invoked": False,
        "attacker_active_at_settlement": False,
        "postselection_delivered_to_agent": False,
        "later_environment_read": False,
        "retained_memory_observed": False,
        "verdict_dependent_termination_observed": False,
        "actual_profile_probe_observed": False,
    }
    if channel == "timing":
        evidence.update(
            {
                "probe_feedback_transform_count": 1,
                "timing_scheduler_invocation_count": 1,
                "timing_observations": [
                    {
                        "clock": "monotonic_perf_counter",
                        "response_elapsed_us": 0,
                        "scheduled_padding_us": 5000,
                        "observed_padding_us": 5000,
                        "observed_total_us": 5000,
                        "bucket_threshold_us": 2500,
                        "timing_class": "measured-padded",
                        "scheduler_completed_before_transcript_append": True,
                    }
                ],
            }
        )
    return evidence


@pytest.mark.parametrize("channel", tuple(CHANNEL_INTERVENTIONS))
def test_channel_audit_accepts_only_live_controlled_evidence(channel: str) -> None:
    audit = channel_intervention_audit(
        channel=channel,
        operational_evidence=_live_evidence(channel),
    )

    assert audit["changed_invariant"] == CHANNEL_INTERVENTIONS[channel]
    assert audit["status"] == (
        "not_evaluated" if channel == "retained_kv_cache" else "evaluated"
    )
    assert audit["intervention_applied"] is (
        channel not in {"exact_silenttwin", "retained_kv_cache"}
    )

    posthoc = _live_evidence(channel)
    posthoc["intervention_stage"] = "completed_transcript_decoration"
    with pytest.raises(AdvancedProtocolError, match="live controlled trial"):
        channel_intervention_audit(
            channel=channel,
            operational_evidence=posthoc,
        )


def test_kv_cache_is_not_evaluated_and_never_textually_emulated() -> None:
    audit = channel_intervention_audit(
        channel="retained_kv_cache",
        operational_evidence=_live_evidence("retained_kv_cache"),
    )

    assert audit["status"] == "not_evaluated"
    assert audit["intervention_applied"] is False
    assert "textual emulation is forbidden" in audit["reason"]


def test_e5_ablation_contracts_are_control_evaluated_or_explicitly_not_evaluated() -> None:
    for name, invariant in ABLATION_INVARIANTS.items():
        contract = ablation_contract(name, backend_exposes_kv_cache=False)
        assert contract["changed_invariant"] == invariant
        if name == "none":
            assert contract["status"] == "control"
            assert contract["operational_change"] is False
        elif name == "retained_kv_cache":
            assert contract["status"] == "not_evaluated"
            assert contract["operational_change"] is False
        else:
            assert contract["status"] == "evaluated"
            assert contract["operational_change"] is True

    exposed = ablation_contract(
        "retained_kv_cache",
        backend_exposes_kv_cache=True,
    )
    assert exposed["status"] == "evaluated"
    assert exposed["operational_change"] is True
