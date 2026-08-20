"""Seeded random lower baseline."""

import random
from typing import Any, Iterable

from silenttwin.attackers.base import Attacker
from silenttwin.schemas import AgentVisibleTranscript


class RandomAttacker(Attacker):
    name = "random"

    def __init__(self, *, seed: int = 0, **kwargs: Any) -> None:
        super().__init__(seed=seed, **kwargs)
        self._rng = random.Random(seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    def predict_state(
        self,
        transcript: AgentVisibleTranscript,
        *,
        world_pair=None,
        runtime=None,
        actions: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> str:
        prediction = self._rng.choice(("theta0", "theta1"))
        self.posterior = {"theta0": 0.5, "theta1": 0.5}
        return prediction

    def select_exploit(self, world_pair, predicted_state=None, **kwargs):
        return self._rng.choice(world_pair.theta0.exploits)


__all__ = ["RandomAttacker"]
