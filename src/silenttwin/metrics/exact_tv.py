"""Exact Tier-1 transcript distributions and reachable-history checking.

Two deliberately different claims live here:

* :func:`exact_total_variation` and
  :func:`enumerate_transcript_distribution` are exact over an explicitly
  declared finite support of action sequences and seeds.
* :func:`check_reachable_history_bisimulation` checks every public probe action
  at every reachable abstract public state for the built-in finite-state
  runtimes.  Deterministic runtimes use bisimulation; the randomized-feedback
  baseline uses its exact rational binary-channel distribution.

Neither API calls byte mismatch counts "TV", and neither extrapolates from one
scripted trajectory.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math
from typing import Any, Hashable, Iterable, Mapping, Sequence

from silenttwin.runtime import get_runtime
from silenttwin.schemas import canonical_json, stable_digest
from silenttwin.worlds.paired_world import WorldPair


Probability = Fraction
TranscriptDistribution = dict[str, Probability]


def _as_fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("boolean is not a probability")
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def validate_probability_distribution(
    distribution: Mapping[Hashable, int | float | Fraction],
) -> dict[Hashable, Fraction]:
    if not distribution:
        raise ValueError("a probability distribution cannot be empty")
    normalized: dict[Hashable, Fraction] = {}
    for outcome, probability in distribution.items():
        exact = _as_fraction(probability)
        if exact < 0:
            raise ValueError(f"negative probability for outcome {outcome!r}")
        if exact:
            normalized[outcome] = exact
    if sum(normalized.values(), Fraction(0)) != 1:
        raise ValueError("probability mass must sum exactly to one")
    return normalized


def exact_total_variation(
    left: Mapping[Hashable, int | float | Fraction],
    right: Mapping[Hashable, int | float | Fraction],
) -> Fraction:
    """Return ``1/2 * sum_x |P(x)-Q(x)|`` using rational arithmetic."""

    first = validate_probability_distribution(left)
    second = validate_probability_distribution(right)
    outcomes = set(first) | set(second)
    return sum(
        (abs(first.get(item, Fraction(0)) - second.get(item, Fraction(0))) for item in outcomes),
        Fraction(0),
    ) / 2


def enumerate_action_sequences(
    action_ids: Sequence[str],
    max_steps: int,
    *,
    include_shorter: bool = True,
    max_histories: int = 1_000_000,
) -> tuple[tuple[str, ...], ...]:
    """Materialize an explicitly bounded finite support of open-loop probes."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if max_histories <= 0:
        raise ValueError("max_histories must be positive")
    actions = tuple(action_ids)
    if len(set(actions)) != len(actions):
        raise ValueError("action_ids contains a duplicate")
    if max_steps and not actions:
        raise ValueError("a positive horizon requires at least one action")
    lengths = range(max_steps + 1) if include_shorter else (max_steps,)
    expected = sum(len(actions) ** length for length in lengths)
    if expected > max_histories:
        raise ValueError(
            f"requested support has {expected} histories, exceeding max_histories={max_histories}"
        )
    return tuple(
        history
        for length in lengths
        for history in product(actions, repeat=length)
    )


def _run_transcript(
    pair: WorldPair,
    state: str,
    runtime: str,
    actions: Sequence[str],
    *,
    seed: int,
    runtime_kwargs: Mapping[str, Any] | None = None,
    end_probing: bool = True,
):
    selected_runtime = get_runtime(runtime, seed=seed, **dict(runtime_kwargs or {}))
    session = selected_runtime.start_session(
        pair.world_for_state(state), session_id=f"exact-check-{pair.public_instance_hash[:16]}"
    )
    for action_id in actions:
        if not session.active:
            break
        action = session.world.action(action_id)
        if action.effectful:
            raise ValueError(
                f"exact probing distributions accept only non-effectful actions; {action_id!r} is effectful"
            )
        selected_runtime.execute(session, action)
    if end_probing and session.active:
        selected_runtime.end_probing(session)
    return selected_runtime, session


def enumerate_transcript_distribution(
    pair: WorldPair,
    state: str,
    runtime: str,
    action_sequences: Iterable[Sequence[str]],
    *,
    seeds: Sequence[int] = (0,),
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> TranscriptDistribution:
    """Enumerate an exact uniform distribution over supplied sequences/seeds."""

    histories = tuple(tuple(history) for history in action_sequences)
    selected_seeds = tuple(seeds)
    if not histories:
        raise ValueError("action_sequences must not be empty")
    if not selected_seeds:
        raise ValueError("seeds must not be empty")
    if len(set(histories)) != len(histories):
        raise ValueError("action_sequences contains a duplicate history")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("seeds contains a duplicate")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in selected_seeds):
        raise ValueError("seeds must be non-negative integers")
    mass = Fraction(1, len(histories) * len(selected_seeds))
    result: dict[str, Fraction] = defaultdict(Fraction)
    for seed in selected_seeds:
        for history in histories:
            _, session = _run_transcript(
                pair,
                state,
                runtime,
                history,
                seed=seed,
                runtime_kwargs=runtime_kwargs,
            )
            result[session.transcript.canonical_json()] += mass
    return dict(result)


@dataclass(frozen=True)
class TranscriptDistributionComparison:
    left: TranscriptDistribution
    right: TranscriptDistribution
    total_variation: Fraction
    action_sequence_count: int
    seed_count: int

    @property
    def total_variation_float(self) -> float:
        return float(self.total_variation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "exact_finite_support_enumeration",
            "total_variation": float(self.total_variation),
            "total_variation_fraction": (
                f"{self.total_variation.numerator}/{self.total_variation.denominator}"
            ),
            "action_sequence_count": self.action_sequence_count,
            "seed_count": self.seed_count,
            "theta0_support_size": len(self.left),
            "theta1_support_size": len(self.right),
        }


def compare_enumerated_transcript_distributions(
    pair: WorldPair,
    runtime: str,
    action_sequences: Iterable[Sequence[str]],
    *,
    seeds: Sequence[int] = (0,),
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> TranscriptDistributionComparison:
    histories = tuple(tuple(history) for history in action_sequences)
    left = enumerate_transcript_distribution(
        pair,
        "theta0",
        runtime,
        histories,
        seeds=seeds,
        runtime_kwargs=runtime_kwargs,
    )
    right = enumerate_transcript_distribution(
        pair,
        "theta1",
        runtime,
        histories,
        seeds=seeds,
        runtime_kwargs=runtime_kwargs,
    )
    return TranscriptDistributionComparison(
        left=left,
        right=right,
        total_variation=exact_total_variation(left, right),
        action_sequence_count=len(histories),
        seed_count=len(seeds),
    )


_SUPPORTED_MARKOV_RUNTIMES = frozenset(
    {
        "arm_style",
        "binary_denial",
        "detailed_refusal",
        "generic_failure",
        "opaque_termination",
        "randomized_feedback",
        "silenttwin",
        "visible_settlement",
    }
)


def _normalized_runtime_family(runtime: str) -> str:
    normalized = runtime.strip().lower().replace("-", "_")
    if normalized.startswith("silenttwin:") or normalized.startswith("silenttwin_"):
        return "silenttwin"
    return normalized


def _operational_public_state(session: Any) -> str:
    environment = session.temporary_environment
    state = {
        "session_state": session.state.value,
        "sequence": len(session.transcript),
        "temporary_environment": {
            "fidelity": environment.fidelity,
            "fixed_timing_bucket": environment.fixed_timing_bucket,
            "values": getattr(environment, "_values", {}),
        },
    }
    return stable_digest(state)


@dataclass(frozen=True)
class BisimulationCounterexample:
    action_sequence: tuple[str, ...]
    theta0_transcript_hash: str
    theta1_transcript_hash: str
    theta0_transcript: str
    theta1_transcript: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_sequence": list(self.action_sequence),
            "theta0_transcript_hash": self.theta0_transcript_hash,
            "theta1_transcript_hash": self.theta1_transcript_hash,
            "theta0_transcript": self.theta0_transcript,
            "theta1_transcript": self.theta1_transcript,
        }


@dataclass(frozen=True)
class BisimulationReport:
    runtime: str
    max_steps: int
    checked_public_states: int
    checked_transitions: int
    bisimilar: bool
    maximum_adaptive_total_variation: Fraction
    counterexample: BisimulationCounterexample | None = None
    method: str = "deterministic_reachable_public_state_bisimulation"

    @property
    def exact_tv(self) -> Fraction:
        return self.maximum_adaptive_total_variation

    @property
    def exact_tv_float(self) -> float:
        return float(self.maximum_adaptive_total_variation)

    def to_dict(self) -> dict[str, Any]:
        tv = self.maximum_adaptive_total_variation
        return {
            "method": self.method,
            "runtime": self.runtime,
            "max_steps": self.max_steps,
            "checked_public_states": self.checked_public_states,
            "checked_transitions": self.checked_transitions,
            "all_reachable_histories_equivalent": self.bisimilar,
            "maximum_adaptive_total_variation": float(tv),
            "maximum_adaptive_total_variation_fraction": f"{tv.numerator}/{tv.denominator}",
            "counterexample": (
                self.counterexample.to_dict() if self.counterexample is not None else None
            ),
        }


def _randomized_feedback_report(
    pair: WorldPair,
    runtime: str,
    max_steps: int,
    *,
    runtime_kwargs: Mapping[str, Any] | None,
) -> BisimulationReport:
    """Solve the built-in randomized binary channel exactly.

    Each informative probe is a binary symmetric channel: its truthful monitor
    observation is emitted with probability ``p`` and its complement with
    probability ``1-p``.  All informative public actions are Blackwell-
    equivalent (some merely swap the output labels), so an adaptive policy
    cannot improve on taking an informative action at every step.  The exact
    maximum TV is therefore the rational TV between the two length-Q product
    distributions, grouped by the number of truthful outputs.
    """

    options = dict(runtime_kwargs or {})
    truth_probability = _as_fraction(options.get("truth_probability", 0.75))
    if not 0 <= truth_probability <= 1:
        raise ValueError("truth_probability must lie in [0, 1]")
    probe_ids = tuple(action.action_id for action in pair.theta0.probes)
    informative = tuple(
        action_id
        for action_id in probe_ids
        if pair.theta0.monitor.decide(pair.theta0.action(action_id))
        != pair.theta1.monitor.decide(pair.theta1.action(action_id))
    )
    if max_steps == 0 or not informative or truth_probability == Fraction(1, 2):
        tv = Fraction(0)
    else:
        false_probability = 1 - truth_probability
        distance = Fraction(0)
        for truthful_count in range(max_steps + 1):
            left = (truth_probability**truthful_count) * (
                false_probability ** (max_steps - truthful_count)
            )
            right = (false_probability**truthful_count) * (
                truth_probability ** (max_steps - truthful_count)
            )
            distance += math.comb(max_steps, truthful_count) * abs(left - right)
        tv = distance / 2
    return BisimulationReport(
        runtime=runtime,
        max_steps=max_steps,
        checked_public_states=max_steps + 1,
        checked_transitions=max_steps * len(probe_ids),
        bisimilar=tv == 0,
        maximum_adaptive_total_variation=tv,
        counterexample=None,
        method="exact_randomized_binary_channel_dynamic_program",
    )


def _counterexample(
    history: tuple[str, ...], left_session: Any, right_session: Any
) -> BisimulationCounterexample:
    left = left_session.transcript.canonical_json()
    right = right_session.transcript.canonical_json()
    return BisimulationCounterexample(
        action_sequence=history,
        theta0_transcript_hash=stable_digest(left_session.transcript.to_dict()),
        theta1_transcript_hash=stable_digest(right_session.transcript.to_dict()),
        theta0_transcript=left,
        theta1_transcript=right,
    )


def check_reachable_history_bisimulation(
    pair: WorldPair,
    runtime: str,
    max_steps: int,
    *,
    seed: int = 0,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> BisimulationReport:
    """Check every adaptive public probe choice up to ``max_steps`` exactly.

    The quotient merges histories only on the complete operational public state
    used by the built-in Markov Tier-1 runtimes (lifecycle, sequence, and
    temporary-environment state). The built-in randomized channel is solved
    over its complete finite probability law. Custom runtimes are rejected
    rather than receiving an unjustified universal guarantee.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    family = _normalized_runtime_family(runtime)
    if family not in _SUPPORTED_MARKOV_RUNTIMES:
        raise ValueError(
            f"runtime {runtime!r} is not supported by the exact Markov-state checker"
        )
    left_probe_ids = tuple(action.action_id for action in pair.theta0.probes)
    right_probe_ids = tuple(action.action_id for action in pair.theta1.probes)
    if left_probe_ids != right_probe_ids:
        raise ValueError("paired worlds expose different public probe catalogues")
    if max_steps and not left_probe_ids:
        raise ValueError("a positive horizon requires at least one public probe")
    if any(pair.theta0.action(action_id).effectful for action_id in left_probe_ids):
        raise ValueError("reachable-history checking permits only non-effectful probes")
    if family == "randomized_feedback":
        return _randomized_feedback_report(
            pair,
            runtime,
            max_steps,
            runtime_kwargs=runtime_kwargs,
        )

    queue: deque[tuple[str, ...]] = deque([()])
    seen: set[tuple[int, str, str]] = set()
    checked_states = 0
    checked_transitions = 0
    while queue:
        history = queue.popleft()
        _, left_session = _run_transcript(
            pair,
            "theta0",
            runtime,
            history,
            seed=seed,
            runtime_kwargs=runtime_kwargs,
            end_probing=len(history) >= max_steps,
        )
        _, right_session = _run_transcript(
            pair,
            "theta1",
            runtime,
            history,
            seed=seed,
            runtime_kwargs=runtime_kwargs,
            end_probing=len(history) >= max_steps,
        )
        left_public_state = _operational_public_state(left_session)
        right_public_state = _operational_public_state(right_session)
        state_key = (len(history), left_public_state, right_public_state)
        if left_session.transcript.canonical_bytes() != right_session.transcript.canonical_bytes():
            return BisimulationReport(
                runtime=runtime,
                max_steps=max_steps,
                checked_public_states=checked_states,
                checked_transitions=checked_transitions,
                bisimilar=False,
                maximum_adaptive_total_variation=Fraction(1),
                counterexample=_counterexample(history, left_session, right_session),
            )
        if left_public_state != right_public_state:
            return BisimulationReport(
                runtime=runtime,
                max_steps=max_steps,
                checked_public_states=checked_states,
                checked_transitions=checked_transitions,
                bisimilar=False,
                maximum_adaptive_total_variation=Fraction(1),
                counterexample=_counterexample(history, left_session, right_session),
            )

        # "At most Q" includes a policy that stops at this public history.
        # Check the pre-prediction flush before quotienting histories: visible
        # settlement can depend on the full private decision record even when
        # ordinary probe observations and operational public state coincide.
        if left_session.active and len(history) < max_steps:
            _, stopped_left = _run_transcript(
                pair,
                "theta0",
                runtime,
                history,
                seed=seed,
                runtime_kwargs=runtime_kwargs,
                end_probing=True,
            )
            _, stopped_right = _run_transcript(
                pair,
                "theta1",
                runtime,
                history,
                seed=seed,
                runtime_kwargs=runtime_kwargs,
                end_probing=True,
            )
            if stopped_left.transcript.canonical_bytes() != stopped_right.transcript.canonical_bytes():
                return BisimulationReport(
                    runtime=runtime,
                    max_steps=max_steps,
                    checked_public_states=checked_states + 1,
                    checked_transitions=checked_transitions,
                    bisimilar=False,
                    maximum_adaptive_total_variation=Fraction(1),
                    counterexample=_counterexample(history, stopped_left, stopped_right),
                )

        if state_key in seen:
            continue
        seen.add(state_key)
        checked_states += 1
        if len(history) >= max_steps or not left_session.active:
            continue
        for action_id in left_probe_ids:
            checked_transitions += 1
            successor = (*history, action_id)
            _, next_left = _run_transcript(
                pair,
                "theta0",
                runtime,
                successor,
                seed=seed,
                runtime_kwargs=runtime_kwargs,
                end_probing=len(successor) >= max_steps,
            )
            _, next_right = _run_transcript(
                pair,
                "theta1",
                runtime,
                successor,
                seed=seed,
                runtime_kwargs=runtime_kwargs,
                end_probing=len(successor) >= max_steps,
            )
            if next_left.transcript.canonical_bytes() != next_right.transcript.canonical_bytes():
                return BisimulationReport(
                    runtime=runtime,
                    max_steps=max_steps,
                    checked_public_states=checked_states,
                    checked_transitions=checked_transitions,
                    bisimilar=False,
                    maximum_adaptive_total_variation=Fraction(1),
                    counterexample=_counterexample(successor, next_left, next_right),
                )
            queue.append(successor)

    return BisimulationReport(
        runtime=runtime,
        max_steps=max_steps,
        checked_public_states=checked_states,
        checked_transitions=checked_transitions,
        bisimilar=True,
        maximum_adaptive_total_variation=Fraction(0),
        counterexample=None,
    )


__all__ = [
    "BisimulationCounterexample",
    "BisimulationReport",
    "TranscriptDistribution",
    "TranscriptDistributionComparison",
    "check_reachable_history_bisimulation",
    "compare_enumerated_transcript_distributions",
    "enumerate_action_sequences",
    "enumerate_transcript_distribution",
    "exact_total_variation",
    "validate_probability_distribution",
]
