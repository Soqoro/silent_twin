"""Optimal deterministic posterior for the known Tier-1 finite-state model."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from silenttwin.attackers.base import Attacker
from silenttwin.runtime import RandomizedFeedbackRuntime, Runtime, SilentTwinRuntime, get_runtime
from silenttwin.schemas import AgentVisibleTranscript
from silenttwin.worlds import WorldPair


class BayesianAttacker(Attacker):
    name = "bayesian"

    def __init__(
        self,
        world_pair: WorldPair | None = None,
        runtime: Runtime | str | None = None,
        *,
        seed: int = 0,
        prior: Mapping[str, float] | None = None,
        mismatch_likelihood: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, **kwargs)
        self.world_pair = world_pair
        self.runtime = runtime
        self.initial_prior = dict(prior or {"theta0": 0.5, "theta1": 0.5})
        if set(self.initial_prior) != {"theta0", "theta1"}:
            raise ValueError("prior must contain exactly theta0 and theta1")
        if any(value < 0 for value in self.initial_prior.values()) or sum(self.initial_prior.values()) <= 0:
            raise ValueError("prior probabilities must be non-negative with positive mass")
        total = sum(self.initial_prior.values())
        self.initial_prior = {key: value / total for key, value in self.initial_prior.items()}
        if not 0 <= mismatch_likelihood < 1:
            raise ValueError("mismatch_likelihood must lie in [0, 1)")
        self.mismatch_likelihood = mismatch_likelihood
        self.reset()

    def reset(self) -> None:
        super().reset()
        prior = getattr(self, "initial_prior", {"theta0": 0.5, "theta1": 0.5})
        self.posterior = dict(prior)

    def _runtime_clone(self, runtime: Runtime | str) -> Runtime:
        if isinstance(runtime, str):
            return get_runtime(runtime, seed=self.seed)
        kwargs: dict[str, Any] = {"seed": runtime.seed}
        if isinstance(runtime, RandomizedFeedbackRuntime):
            kwargs["truth_probability"] = runtime.truth_probability
        if isinstance(runtime, SilentTwinRuntime):
            kwargs["ablations"] = runtime.ablations
        return get_runtime(runtime.name, **kwargs)

    def _predicted_transcript(
        self,
        world_pair: WorldPair,
        state: str,
        runtime: Runtime | str,
        actions: Iterable[str],
    ) -> str:
        model_runtime = self._runtime_clone(runtime)
        session = model_runtime.start_session(world_pair.world_for_state(state))
        for action in actions:
            if not session.active:
                break
            model_runtime.execute(session, action)
        return session.transcript.canonical_json()

    def posterior_from_transcript(
        self,
        world_pair: WorldPair,
        transcript: AgentVisibleTranscript,
        runtime: Runtime | str,
        actions: Iterable[str],
    ) -> dict[str, float]:
        observed = transcript.canonical_json()
        weights: dict[str, float] = {}
        for state in ("theta0", "theta1"):
            predicted = self._predicted_transcript(world_pair, state, runtime, actions)
            likelihood = 1.0 if predicted == observed else self.mismatch_likelihood
            weights[state] = self.initial_prior[state] * likelihood
        total = sum(weights.values())
        if total <= 0:
            return dict(self.initial_prior)
        return {state: value / total for state, value in weights.items()}

    update_posterior = posterior_from_transcript

    def _probe_information_score(self, world_pair: WorldPair, runtime: Runtime | str, action_id: str) -> int:
        left = self._predicted_transcript(world_pair, "theta0", runtime, (action_id,))
        right = self._predicted_transcript(world_pair, "theta1", runtime, (action_id,))
        return int(left != right)

    def choose_probe(
        self,
        world_pair: WorldPair,
        transcript: AgentVisibleTranscript,
        remaining_budget: int,
        **kwargs: Any,
    ):
        runtime = kwargs.get("runtime", "generic_failure")
        probes = world_pair.theta0.probes
        scores = [self._probe_information_score(world_pair, runtime, action.action_id) for action in probes]
        best = max(scores, default=0)
        candidates = [action for action, score in zip(probes, scores) if score == best]
        choice = candidates[self._probe_index % len(candidates)]
        self._probe_index += 1
        return choice

    def predict_state(
        self,
        transcript: AgentVisibleTranscript,
        *,
        world_pair: WorldPair | None = None,
        runtime: Runtime | str | None = None,
        actions: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        world_pair = world_pair or self.world_pair
        runtime = runtime or self.runtime
        if world_pair is None or runtime is None:
            # With no finite-state model context, retain the prior rather than
            # attempting to parse arbitrary transcript strings.
            self.posterior = dict(self.initial_prior)
        else:
            selected_actions = tuple(actions) if actions is not None else tuple(
                event.action_id
                for event in transcript
                if event.action_id is not None and event.event_type in {"tool_result", "termination", "arm_tracking"}
            )
            self.posterior = self.posterior_from_transcript(
                world_pair,
                transcript,
                runtime,
                selected_actions,
            )
        return self._posterior_argmax()


OptimalBayesianAttacker = BayesianAttacker

__all__ = ["BayesianAttacker", "OptimalBayesianAttacker"]
