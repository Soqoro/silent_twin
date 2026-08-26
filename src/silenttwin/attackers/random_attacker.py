"""State-independent random and majority-prior controls."""

from __future__ import annotations

import random

from silenttwin.attackers.base import (
    Attacker,
    Prediction,
    PublicTrialContext,
    TranscriptSnapshot,
)


class RandomAttacker(Attacker):
    name = "random"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._rng = random.Random(self.seed)

    def reset(self) -> None:
        super().reset()
        self._rng = random.Random(self.seed)

    def predict_hidden_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        del context, transcript
        prediction = self._rng.choice(("theta0", "theta1"))
        return Prediction(prediction=prediction, posterior={"theta0": 0.5, "theta1": 0.5})

    def choose_final_exploit(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
        prediction: Prediction,
    ) -> str:
        del transcript, prediction
        if not context.exploits:
            raise RuntimeError("public action catalogue has no final exploits")
        return self._rng.choice(context.exploits).action_id


class MajorityPriorAttacker(Attacker):
    """Deterministic predictor that never reads feedback."""

    name = "majority_prior"

    def predict_hidden_state(
        self,
        context: PublicTrialContext,
        transcript: TranscriptSnapshot,
    ) -> Prediction:
        del context, transcript
        prediction = (
            "theta1"
            if self.initial_prior["theta1"] > self.initial_prior["theta0"]
            else "theta0"
        )
        return Prediction(prediction=prediction, posterior=self.initial_prior)


__all__ = ["MajorityPriorAttacker", "RandomAttacker"]
