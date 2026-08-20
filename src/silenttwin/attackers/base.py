"""Adaptive attacker interface and a trusted Tier-1 attack harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from silenttwin.runtime import Runtime, get_runtime
from silenttwin.schemas import AgentVisibleTranscript, RuntimeStep
from silenttwin.worlds import WorldPair, normalize_state


@dataclass(frozen=True)
class AttackRunResult:
    transcript: AgentVisibleTranscript
    probe_action_ids: tuple[str, ...]
    predicted_state: str
    posterior: Mapping[str, float]
    selected_exploit: str | None
    final_exploit_attempts: int
    probe_steps: tuple[RuntimeStep, ...]
    exploit_step: RuntimeStep | None
    session: Any

    @property
    def secret_prediction(self) -> str:
        return self.predicted_state


class Attacker(ABC):
    name = "base"

    def __init__(self, *, seed: int = 0, **_: Any) -> None:
        self.seed = int(seed)
        self.posterior: dict[str, float] = {"theta0": 0.5, "theta1": 0.5}
        self._probe_index = 0

    def reset(self) -> None:
        self.posterior = {"theta0": 0.5, "theta1": 0.5}
        self._probe_index = 0

    def choose_probe(
        self,
        world_pair: WorldPair,
        transcript: AgentVisibleTranscript,
        remaining_budget: int,
        **_: Any,
    ):
        probes = world_pair.theta0.probes
        if not probes:
            raise RuntimeError("world has no probe actions")
        choice = probes[self._probe_index % len(probes)]
        self._probe_index += 1
        return choice

    select_probe = choose_probe

    @abstractmethod
    def predict_state(
        self,
        transcript: AgentVisibleTranscript,
        *,
        world_pair: WorldPair | None = None,
        runtime: Runtime | str | None = None,
        actions: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError

    def infer_state(self, transcript: AgentVisibleTranscript, **kwargs: Any) -> str:
        return self.predict_state(transcript, **kwargs)

    def select_exploit(self, world_pair: WorldPair, predicted_state: str | None = None, **_: Any):
        prediction = normalize_state(predicted_state or self._posterior_argmax())
        target = "exploit_cloud_link" if prediction == "theta0" else "exploit_encoded_attachment"
        return world_pair.theta0.action(target)

    choose_exploit = select_exploit

    def _posterior_argmax(self) -> str:
        # Stable tie breaking is important for deterministic balanced trials.
        return "theta1" if self.posterior.get("theta1", 0.0) > self.posterior.get("theta0", 0.0) else "theta0"

    def run_attack(
        self,
        world_pair: WorldPair,
        private_state: str | int,
        runtime: Runtime | str,
        query_budget: int,
        *,
        final_exploit: bool = True,
    ) -> AttackRunResult:
        """Trusted harness; the attacker sees only the pair's public actions and transcript."""

        if query_budget < 0:
            raise ValueError("query_budget must be non-negative")
        selected_runtime = get_runtime(runtime, seed=self.seed) if isinstance(runtime, str) else runtime
        self.reset()
        # Selection of the actual world is evaluation-harness state and is not
        # passed into attacker callbacks.
        world = world_pair.world_for_state(private_state)
        session = selected_runtime.start_session(world)
        probe_ids: list[str] = []
        probe_steps: list[RuntimeStep] = []
        for query_index in range(query_budget):
            if not session.active:
                break
            probe = self.choose_probe(
                world_pair,
                session.transcript,
                query_budget - query_index,
                runtime=selected_runtime,
                previous_actions=tuple(probe_ids),
            )
            probe_ids.append(probe.action_id)
            probe_steps.append(selected_runtime.execute(session, probe))

        prediction = self.predict_state(
            session.transcript,
            world_pair=world_pair,
            runtime=selected_runtime,
            actions=tuple(probe_ids),
        )
        exploit_step = None
        selected_exploit = None
        exploit_attempts = 0
        if final_exploit:
            # The protocol provides exactly one final selection, independent of
            # query budget and condition.  A terminated session counts as the
            # one attempted selection even though it cannot invoke the tool.
            exploit_attempts = 1
            exploit = self.select_exploit(world_pair, prediction)
            selected_exploit = exploit.action_id
            if session.active:
                exploit_step = selected_runtime.execute(session, exploit)
        return AttackRunResult(
            transcript=session.transcript,
            probe_action_ids=tuple(probe_ids),
            predicted_state=prediction,
            posterior=dict(self.posterior),
            selected_exploit=selected_exploit,
            final_exploit_attempts=exploit_attempts,
            probe_steps=tuple(probe_steps),
            exploit_step=exploit_step,
            session=session,
        )

    attack = run_attack


BaseAttacker = Attacker


__all__ = ["AttackRunResult", "Attacker", "BaseAttacker"]
