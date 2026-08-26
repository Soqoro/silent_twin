"""Trusted construction helpers shared by the E1/E2 thin adapters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from silenttwin.attackers import make_attacker
from silenttwin.attackers.base import Attacker
from silenttwin.attackers.black_box_attacker import (
    BlackBoxAttacker,
    TrainingExample,
    TrainingProvenance,
)
from silenttwin.attackers.random_attacker import MajorityPriorAttacker
from silenttwin.config import ExperimentConfig, stable_hash
from silenttwin.experiments.feedback import FeedbackKind
from silenttwin.experiments.trial_runner import (
    TrialMode,
    TrialRunner,
    TrialSpec,
    counterbalanced_state_cell,
)
from silenttwin.worlds.dataset import (
    build_dataset_world_pair,
    iter_template_specs,
    select_template_id,
)
from silenttwin.worlds.paired_world import WorldPair


@dataclass(frozen=True, slots=True)
class TrialAssignment:
    sample_index: int
    public_instance_index: int
    assignment_cell: int
    template_id: str
    pair: WorldPair
    actual_state: str
    donor_state: str


@lru_cache(maxsize=8)
def _tier2_model_client(model_config: Any) -> Any:
    """Load and retain one local model per worker process/configuration."""

    from silenttwin.model_clients import LocalTransformersModelClient

    client = LocalTransformersModelClient(model_config)
    client.ensure_available()
    return client


def assignment_for_sample(
    config: ExperimentConfig, sample_index: int
) -> TrialAssignment:
    """Map a row to a balanced public-instance factorial assignment."""

    if sample_index < 0:
        raise ValueError("sample index must be non-negative")
    if config.experiment == "e2" or (
        config.experiment == "e1" and config.feedback_source == "shuffled"
    ):
        rows_per_instance = 4
        actual_state, donor_state = counterbalanced_state_cell(
            sample_index % rows_per_instance,
            config.seed,
        )
    else:
        rows_per_instance = 2
        state_index = (sample_index + config.seed) % rows_per_instance
        actual_state = f"theta{state_index}"
        donor_state = actual_state
    public_instance_index = sample_index // rows_per_instance
    template_id = select_template_id(
        config.world_suite,
        config.dataset_split,
        public_instance_index,
        requested=config.template_id,
        dataset_revision=config.dataset_revision,
    )
    pair = build_dataset_world_pair(
        template_id,
        pair_family=config.pair_family,
        instance_id=f"{config.seed}-{public_instance_index:08d}",
        instance_index=public_instance_index,
        seed=config.seed,
        dataset_revision=config.dataset_revision,
        split=config.dataset_split,
    )
    return TrialAssignment(
        sample_index=sample_index,
        public_instance_index=public_instance_index,
        assignment_cell=sample_index % rows_per_instance,
        template_id=template_id,
        pair=pair,
        actual_state=actual_state,
        donor_state=donor_state,
    )


@lru_cache(maxsize=64)
def _black_box_training_data(
    world_suite: str,
    runtime: str,
    query_budget: int,
    pair_family: str,
    dataset_revision: str,
    seed: int,
) -> tuple[tuple[TrainingExample, ...], TrainingProvenance]:
    """Generate labelled trajectories only from the declared training split."""

    runtime_hash = stable_hash(
        {
            "runtime": runtime,
            "query_budget": query_budget,
            "pair_family": pair_family,
            "dataset_revision": dataset_revision,
            "feedback_source": "genuine",
        }
    )
    specs = iter_template_specs(domain=world_suite, split="train")
    examples: list[TrainingExample] = []
    for template in specs:
        pair = build_dataset_world_pair(
            template.template_id,
            pair_family=pair_family,
            instance_id=f"black-box-training-{template.template_id}",
            seed=seed,
            dataset_revision=dataset_revision,
            split="train",
        )
        for state in ("theta0", "theta1"):
            # The training policy alternates public probes and has no access to
            # the label. Labels enter only after the trajectory is complete.
            training_attacker = MajorityPriorAttacker(seed=seed)
            result = TrialRunner().run(
                TrialSpec(
                    world_pair=pair,
                    actual_state=state,
                    runtime=runtime,
                    attacker=training_attacker,
                    query_budget=query_budget,
                    mode=TrialMode.INFERENCE_ONLY,
                    feedback_source=FeedbackKind.GENUINE,
                    seed=seed,
                    trial_id=f"black-box-training-{template.template_id}-{state}",
                    pair_family=pair_family,
                    template_id=template.template_id,
                    dataset_split="train",
                    dataset_revision=dataset_revision,
                )
            )
            examples.append(
                TrainingExample(
                    transcript=result.decision_transcript,
                    label=state,
                    template_id=template.template_id,
                    runtime_configuration_hash=runtime_hash,
                )
            )
    provenance = TrainingProvenance(
        dataset_revision=dataset_revision,
        split="train",
        template_ids=tuple(template.template_id for template in specs),
        runtime_configuration_hashes=(runtime_hash,),
    )
    return tuple(examples), provenance


def make_configured_attacker(
    config: ExperimentConfig, sample_index: int
) -> Attacker:
    rows_per_public_instance = (
        4
        if config.experiment == "e2"
        or (config.experiment == "e1" and config.feedback_source == "shuffled")
        else 2
    )
    public_instance_index = sample_index // rows_per_public_instance
    # Attacker randomness is a public-instance/replicate factor.  It must not
    # change across the trusted target/donor assignment rows, otherwise the RNG
    # seed itself becomes an indirect hidden-state signal.
    attacker_seed = int(
        stable_hash(
            [
                config.seed,
                public_instance_index,
                config.attacker,
                config.decoding_seed,
            ]
        )[:16],
        16,
    )
    if config.attacker == "black_box":
        examples, provenance = _black_box_training_data(
            config.world_suite,
            config.runtime,
            config.query_budget,
            config.pair_family,
            config.dataset_revision,
            config.seed,
        )
        return BlackBoxAttacker(seed=attacker_seed).fit(
            examples,
            provenance=provenance,
        )
    if config.attacker == "llm":
        from silenttwin.model_clients import LocalModelConfig

        model_config = LocalModelConfig(
            model_id=str(config.model_id),
            model_revision=config.model_revision or "main",
            model_cache_dir=config.model_cache_dir,
            dtype=config.dtype,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            decoding_seed=int(config.decoding_seed or 0),
            batch_size=config.batch_size,
        )
        client = _tier2_model_client(model_config)
        return make_attacker(
            "llm",
            seed=int(config.decoding_seed or 0),
            model_client=client,
            max_tokens=config.max_new_tokens,
        )
    return make_attacker(config.attacker, seed=attacker_seed)


def code_provenance(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "configuration_hash": config.configuration_hash,
        "analysis_revision": config.analysis_revision,
    }


__all__ = [
    "TrialAssignment",
    "assignment_for_sample",
    "code_provenance",
    "make_configured_attacker",
]
