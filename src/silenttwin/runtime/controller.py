"""Dependency-aware, token-gated final effect controller."""

from __future__ import annotations

import hashlib
import hmac
from typing import Callable, Iterable, Mapping

from silenttwin.runtime.dependency_graph import DependencyCycleError, DependencyGraph
from silenttwin.schemas import CommitReport, CommitToken, EffectStatus, StagedEffect, canonical_json


class CommitError(RuntimeError):
    pass


class InadmissibleEffectError(CommitError):
    pass


class InvalidCommitTokenError(CommitError):
    pass


class AtomicityError(CommitError):
    pass


EffectExecutor = Callable[[StagedEffect], object]


class DependencyAwareController:
    """Commits only token-authorized effects with admissible closures.

    ``ignore_*`` switches exist solely for the named Experiment-5 degraded
    conditions.  Production/full SilentTwin uses their safe defaults.
    """

    def __init__(
        self,
        effects: Iterable[StagedEffect] = (),
        *,
        expected_policy_epoch: int | None = None,
        token_secret: bytes | str = b"silenttwin-tier1-controller-v1",
        executor: EffectExecutor | None = None,
        ignore_data_dependencies: bool = False,
        ignore_control_dependencies: bool = False,
        enforce_atomic_groups: bool = True,
    ) -> None:
        self.graph = DependencyGraph()
        self.expected_policy_epoch = expected_policy_epoch
        self._token_secret = token_secret.encode("utf-8") if isinstance(token_secret, str) else token_secret
        if not self._token_secret:
            raise ValueError("token_secret must not be empty")
        self.executor = executor
        self.ignore_data_dependencies = ignore_data_dependencies
        self.ignore_control_dependencies = ignore_control_dependencies
        self.enforce_atomic_groups = enforce_atomic_groups
        self.execution_results: dict[str, object] = {}
        for effect in effects:
            self.record(effect)

    @property
    def effects(self) -> Mapping[str, StagedEffect]:
        return self.graph.effects

    def record(self, effect: StagedEffect, admissible: bool | None = None) -> StagedEffect:
        if admissible is not None:
            effect.admissible = bool(admissible)
            effect.status = EffectStatus.ADMISSIBLE if effect.admissible else EffectStatus.REJECTED
        self.graph.add(effect)
        return effect

    stage = record
    add_effect = record

    def _basic_eligibility(self, effect_id: str) -> tuple[bool, str | None]:
        effect = self.graph.get(effect_id)
        if effect.admissible is not True:
            return False, "effect is not recorded as admissible"
        if effect.status in {EffectStatus.REJECTED, EffectStatus.DISCARDED}:
            return False, f"effect status is {effect.status.value}"
        if self.expected_policy_epoch is not None and effect.policy_epoch != self.expected_policy_epoch:
            return False, "policy epoch does not match controller epoch"
        try:
            closure = self.graph.closure(
                effect_id,
                include_data=not self.ignore_data_dependencies,
                include_control=not self.ignore_control_dependencies,
            )
        except DependencyCycleError as exc:
            return False, str(exc)
        if not closure.complete:
            missing = sorted(closure.missing_data | closure.missing_control)
            return False, f"missing dependencies: {', '.join(missing)}"
        for dependency_id in closure.effect_ids:
            dependency = self.graph.get(dependency_id)
            if dependency.admissible is not True or dependency.status in {
                EffectStatus.REJECTED,
                EffectStatus.DISCARDED,
            }:
                return False, f"dependency {dependency_id} is not admissible"
            if self.expected_policy_epoch is not None and dependency.policy_epoch != self.expected_policy_epoch:
                return False, f"dependency {dependency_id} has a mismatched policy epoch"
        return True, None

    def eligibility(self, effect_id: str) -> tuple[bool, str | None]:
        eligible, reason = self._basic_eligibility(effect_id)
        if not eligible or not self.enforce_atomic_groups:
            return eligible, reason
        effect = self.graph.get(effect_id)
        if effect.atomic_group:
            members = self.graph.atomic_groups()[effect.atomic_group]
            for member_id in members:
                member_eligible, member_reason = self._basic_eligibility(member_id)
                if not member_eligible:
                    return False, f"atomic peer {member_id} is ineligible: {member_reason}"
        return True, None

    def is_eligible(self, effect_id: str) -> bool:
        return self.eligibility(effect_id)[0]

    def _token_message(self, effect: StagedEffect) -> bytes:
        return canonical_json(
            {
                "effect_id": effect.effect_id,
                "policy_epoch": effect.policy_epoch,
                "payload_digest": effect.payload_digest,
                "destination_digest": effect.destination_digest,
                "data_dependencies": list(effect.data_dependencies),
                "control_dependencies": list(effect.control_dependencies),
                "atomic_group": effect.atomic_group,
                "admissible": effect.admissible,
                "monitor_decision": effect.monitor_decision.value,
            }
        ).encode("utf-8")

    def issue_commit_token(self, effect_id: str) -> CommitToken:
        eligible, reason = self.eligibility(effect_id)
        if not eligible:
            raise InadmissibleEffectError(f"cannot authorize {effect_id}: {reason}")
        effect = self.graph.get(effect_id)
        signature = hmac.new(self._token_secret, self._token_message(effect), hashlib.sha256).hexdigest()
        return CommitToken(
            effect_id=effect.effect_id,
            policy_epoch=effect.policy_epoch,
            payload_digest=effect.payload_digest,
            destination_digest=effect.destination_digest,
            signature=signature,
        )

    authorize = issue_commit_token

    def validate_commit_token(self, token: CommitToken, effect_id: str | None = None) -> bool:
        target = effect_id or token.effect_id
        if target not in self.graph:
            return False
        effect = self.graph.get(target)
        if (
            token.effect_id != effect.effect_id
            or token.policy_epoch != effect.policy_epoch
            or token.payload_digest != effect.payload_digest
            or token.destination_digest != effect.destination_digest
        ):
            return False
        expected = hmac.new(self._token_secret, self._token_message(effect), hashlib.sha256).hexdigest()
        return hmac.compare_digest(token.signature, expected)

    def _commit_one(self, effect_id: str, token: CommitToken) -> StagedEffect:
        eligible, reason = self.eligibility(effect_id)
        if not eligible:
            raise InadmissibleEffectError(f"cannot commit {effect_id}: {reason}")
        if not self.validate_commit_token(token, effect_id):
            raise InvalidCommitTokenError(f"invalid or mismatched commit token for {effect_id}")
        effect = self.graph.get(effect_id)
        if effect.status is EffectStatus.COMMITTED:
            return effect
        if self.executor is not None:
            self.execution_results[effect_id] = self.executor(effect)
        effect.status = EffectStatus.COMMITTED
        return effect

    def commit_effect(self, effect_id: str, token: CommitToken) -> StagedEffect:
        effect = self.graph.get(effect_id)
        if self.enforce_atomic_groups and effect.atomic_group:
            members = self.graph.atomic_groups()[effect.atomic_group]
            if len(members) > 1:
                raise AtomicityError(
                    f"effect {effect_id} belongs to atomic group {effect.atomic_group!r}; use commit_group"
                )
        return self._commit_one(effect_id, token)

    commit = commit_effect

    def commit_group(self, atomic_group: str, tokens: Mapping[str, CommitToken]) -> tuple[StagedEffect, ...]:
        try:
            members = self.graph.atomic_groups()[atomic_group]
        except KeyError as exc:
            raise KeyError(f"unknown atomic group: {atomic_group}") from exc
        # Validate everything before invoking an executor, preserving all-or-none
        # behavior for validation failures.
        for effect_id in members:
            eligible, reason = self.eligibility(effect_id)
            if not eligible:
                raise InadmissibleEffectError(f"cannot commit atomic group {atomic_group}: {reason}")
            token = tokens.get(effect_id)
            if token is None or not self.validate_commit_token(token, effect_id):
                raise InvalidCommitTokenError(
                    f"missing or invalid token for {effect_id} in atomic group {atomic_group}"
                )
        committed = []
        for effect_id in members:
            committed.append(self._commit_one(effect_id, tokens[effect_id]))
        return tuple(committed)

    def issue_commit_tokens(self) -> dict[str, CommitToken]:
        return {
            effect_id: self.issue_commit_token(effect_id)
            for effect_id in self.graph.topological_order(
                include_data=not self.ignore_data_dependencies,
                include_control=not self.ignore_control_dependencies,
            )
            if self.is_eligible(effect_id)
        }

    def commit_all(self, tokens: Mapping[str, CommitToken] | None = None) -> CommitReport:
        supplied = dict(tokens) if tokens is not None else self.issue_commit_tokens()
        dependency_violations: list[str] = []
        committed: list[str] = []
        rejected: list[str] = []
        discarded: list[str] = []

        for effect in self.graph:
            if effect.admissible is not True:
                rejected.append(effect.effect_id)

        groups_done: set[str] = set()
        order = self.graph.topological_order(
            include_data=not self.ignore_data_dependencies,
            include_control=not self.ignore_control_dependencies,
        )
        for effect_id in order:
            effect = self.graph.get(effect_id)
            eligible, reason = self.eligibility(effect_id)
            if not eligible:
                if effect.status is not EffectStatus.REJECTED:
                    effect.status = EffectStatus.DISCARDED
                discarded.append(effect_id)
                if reason and ("dependenc" in reason or "cycle" in reason):
                    dependency_violations.append(effect_id)
                continue
            if self.enforce_atomic_groups and effect.atomic_group:
                if effect.atomic_group in groups_done:
                    continue
                groups_done.add(effect.atomic_group)
                members = self.graph.atomic_groups()[effect.atomic_group]
                if not all(member in supplied for member in members):
                    for member in members:
                        target = self.graph.get(member)
                        if target.status is not EffectStatus.REJECTED:
                            target.status = EffectStatus.DISCARDED
                        if member not in discarded:
                            discarded.append(member)
                    continue
                try:
                    group_effects = self.commit_group(effect.atomic_group, supplied)
                except CommitError:
                    for member in members:
                        target = self.graph.get(member)
                        if target.status is not EffectStatus.REJECTED:
                            target.status = EffectStatus.DISCARDED
                        if member not in discarded:
                            discarded.append(member)
                else:
                    committed.extend(member.effect_id for member in group_effects)
                continue
            token = supplied.get(effect_id)
            if token is None:
                effect.status = EffectStatus.DISCARDED
                discarded.append(effect_id)
                continue
            try:
                self._commit_one(effect_id, token)
            except CommitError:
                effect.status = EffectStatus.DISCARDED
                discarded.append(effect_id)
            else:
                committed.append(effect_id)

        atomicity_violations: list[str] = []
        for group, members in self.graph.atomic_groups().items():
            count = sum(self.graph.get(member).status is EffectStatus.COMMITTED for member in members)
            if 0 < count < len(members):
                atomicity_violations.append(group)

        return CommitReport(
            committed_effect_ids=tuple(dict.fromkeys(committed)),
            discarded_effect_ids=tuple(dict.fromkeys(discarded)),
            rejected_effect_ids=tuple(dict.fromkeys(rejected)),
            prohibited_effect_ids=tuple(
                effect_id
                for effect_id in dict.fromkeys(committed)
                if self.graph.get(effect_id).is_prohibited
            ),
            dependency_violations=tuple(dict.fromkeys(dependency_violations)),
            atomicity_violations=tuple(atomicity_violations),
        )

    finalize = commit_all


FinalController = DependencyAwareController


__all__ = [
    "AtomicityError",
    "CommitError",
    "DependencyAwareController",
    "EffectExecutor",
    "FinalController",
    "InadmissibleEffectError",
    "InvalidCommitTokenError",
]
