"""Known-likelihood Bayesian attacker for the finite Tier-1 model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from silenttwin.attackers.base import (
    STATE_IDS,
    Attacker,
    Prediction,
    PublicTrialContext,
    TranscriptSnapshot,
)
from silenttwin.schemas import canonical_json


PRIMARY_EVENT_TYPES = frozenset({"tool_result", "termination", "arm_tracking"})


def observation_signature(events: Sequence[Mapping[str, Any]]) -> str:
    """Canonical signature of one delivered response bundle."""

    return canonical_json([dict(event) for event in events])


@dataclass(frozen=True, slots=True)
class LikelihoodRow:
    step_index: int
    action_id: str
    state: str
    observation_signature: str
    probability: float

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("likelihood step index must be non-negative")
        if self.state not in STATE_IDS:
            raise ValueError("likelihood state must be theta0 or theta1")
        if not math.isfinite(self.probability) or self.probability < 0.0:
            raise ValueError("likelihood probability must be finite and non-negative")


class KnownLikelihoodModel:
    """Immutable finite observation model with no world/runtime references."""

    def __init__(self, rows: Iterable[LikelihoodRow], *, model_revision: str = "tier1-v1") -> None:
        table: dict[tuple[int, str, str], dict[str, float]] = {}
        for row in rows:
            key = (row.step_index, row.action_id, row.state)
            distribution = table.setdefault(key, {})
            distribution[row.observation_signature] = (
                distribution.get(row.observation_signature, 0.0) + row.probability
            )
        if not table:
            raise ValueError("known likelihood model must contain at least one row")
        normalized: dict[tuple[int, str, str], Mapping[str, float]] = {}
        for key, distribution in table.items():
            total = sum(distribution.values())
            if total <= 0:
                raise ValueError(f"likelihood distribution {key!r} has no mass")
            normalized[key] = MappingProxyType(
                {signature: probability / total for signature, probability in distribution.items()}
            )
        self._table = MappingProxyType(normalized)
        self.model_revision = str(model_revision)

    def distribution(self, step_index: int, action_id: str, state: str) -> Mapping[str, float]:
        return self._table.get((step_index, action_id, state), MappingProxyType({}))

    def likelihood(
        self,
        step_index: int,
        action_id: str,
        state: str,
        signature: str,
    ) -> float:
        return float(self.distribution(step_index, action_id, state).get(signature, 0.0))

    @property
    def action_ids(self) -> frozenset[str]:
        return frozenset(key[1] for key in self._table)

    def to_rows(self) -> tuple[LikelihoodRow, ...]:
        return tuple(
            LikelihoodRow(step, action, state, signature, probability)
            for (step, action, state), distribution in self._table.items()
            for signature, probability in distribution.items()
        )


def _probe_bundles(
    context: PublicTrialContext,
    transcript: TranscriptSnapshot,
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    probe_ids = {action.action_id for action in context.probes}
    bundles: list[tuple[str, list[Mapping[str, Any]]]] = []
    for event in transcript:
        action_id = event.get("action_id")
        event_type = event.get("event_type", event.get("event"))
        if action_id in probe_ids and event_type in PRIMARY_EVENT_TYPES:
            bundles.append((str(action_id), [event]))
        elif bundles:
            # Public side-channel and end-of-probing events are part of the
            # observation used for the decision.  In particular, a visible
            # settlement summary belongs to the final probe bundle; excluding
            # it would make the documented positive-control baseline appear
            # prior-only to the known-likelihood attacker.
            bundles[-1][1].append(event)
    return tuple((action_id, tuple(events)) for action_id, events in bundles)


class BayesianAttacker(Attacker):
    """Bayes-optimal policy for an immutable, known finite likelihood model."""

    name = "bayesian"

    def __init__(
        self,
        likelihood_model: KnownLikelihoodModel | None = None,
        *,
        mismatch_likelihood: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= mismatch_likelihood < 1.0:
            raise ValueError("mismatch_likelihood must lie in [0, 1)")
        self.likelihood_model = likelihood_model
        self.mismatch_likelihood = float(mismatch_likelihood)

    def configure_likelihood_model(self, model: KnownLikelihoodModel) -> None:
        if not isinstance(model, KnownLikelihoodModel):
            raise TypeError("Bayesian attacker requires a data-only KnownLikelihoodModel")
        self.likelihood_model = model

    def _posterior(self, context: PublicTrialContext, transcript: TranscriptSnapshot) -> dict[str, float]:
        posterior = dict(self.initial_prior)
        if self.likelihood_model is None:
            return posterior
        for step_index, (action_id, events) in enumerate(_probe_bundles(context, transcript)):
            signature = observation_signature(events)
            weights = {
                state: posterior[state]
                * max(
                    self.likelihood_model.likelihood(step_index, action_id, state, signature),
                    self.mismatch_likelihood,
                )
                for state in STATE_IDS
            }
            total = sum(weights.values())
            if total > 0:
                posterior = {state: weight / total for state, weight in weights.items()}
        return posterior

    @staticmethod
    def _entropy(posterior: Mapping[str, float]) -> float:
        return -sum(value * math.log2(value) for value in posterior.values() if value)

    def _expected_information_gain(
        self,
        posterior: Mapping[str, float],
        *,
        step_index: int,
        action_id: str,
    ) -> float:
        assert self.likelihood_model is not None
        outcomes = set()
        for state in STATE_IDS:
            outcomes.update(self.likelihood_model.distribution(step_index, action_id, state))
        if not outcomes:
            return float("-inf")
        expected_entropy = 0.0
        for outcome in outcomes:
            joint = {
                state: posterior[state]
                * self.likelihood_model.likelihood(step_index, action_id, state, outcome)
                for state in STATE_IDS
            }
            outcome_probability = sum(joint.values())
            if outcome_probability <= 0:
                continue
            conditional = {state: value / outcome_probability for state, value in joint.items()}
            expected_entropy += outcome_probability * self._entropy(conditional)
        return self._entropy(posterior) - expected_entropy

    def choose_probe(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        remaining_budget: int,
    ) -> str:
        if self.likelihood_model is None:
            return super().choose_probe(context, transcript, remaining_budget)
        if remaining_budget <= 0:
            raise ValueError("remaining probe budget must be positive")
        probes = context.probes
        if not probes:
            raise RuntimeError("public action catalogue has no probes")
        step_index = len(_probe_bundles(context, transcript))
        posterior = self._posterior(context, transcript)
        scored = [
            (
                self._expected_information_gain(
                    posterior,
                    step_index=step_index,
                    action_id=action.action_id,
                ),
                action.action_id,
            )
            for action in probes
        ]
        # Stable public-ID tie breaking makes exact trials reproducible.
        return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]

    def predict_hidden_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        if self.likelihood_model is None:
            return Prediction.invalid(
                "bayesian_likelihood_model_not_configured",
                prior=self.initial_prior,
            )
        posterior = self._posterior(context, transcript)
        prediction = "theta1" if posterior["theta1"] > posterior["theta0"] else "theta0"
        return Prediction(prediction=prediction, posterior=posterior)

    online_prediction = predict_hidden_state


OptimalBayesianAttacker = BayesianAttacker


__all__ = [
    "BayesianAttacker",
    "KnownLikelihoodModel",
    "LikelihoodRow",
    "OptimalBayesianAttacker",
    "observation_signature",
]
