"""Data/control dependency graph for staged effects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping

from silenttwin.schemas import StagedEffect


class DependencyError(ValueError):
    pass


class DependencyCycleError(DependencyError):
    pass


class MissingDependencyError(DependencyError):
    pass


@dataclass(frozen=True)
class DependencyClosure:
    effect_id: str
    effect_ids: frozenset[str]
    missing_data: frozenset[str] = frozenset()
    missing_control: frozenset[str] = frozenset()

    @property
    def complete(self) -> bool:
        return not self.missing_data and not self.missing_control


class DependencyGraph:
    """Registry and transitive-closure calculator for staged effects."""

    def __init__(self, effects: Iterable[StagedEffect] = ()) -> None:
        self._effects: dict[str, StagedEffect] = {}
        for effect in effects:
            self.add(effect)

    def add(self, effect: StagedEffect) -> None:
        if effect.effect_id in self._effects:
            raise ValueError(f"duplicate staged effect ID: {effect.effect_id}")
        self._effects[effect.effect_id] = effect

    add_effect = add

    def get(self, effect_id: str) -> StagedEffect:
        try:
            return self._effects[effect_id]
        except KeyError as exc:
            raise KeyError(f"unknown staged effect: {effect_id}") from exc

    @property
    def effects(self) -> Mapping[str, StagedEffect]:
        return dict(self._effects)

    def direct_dependencies(
        self,
        effect_id: str,
        *,
        include_data: bool = True,
        include_control: bool = True,
    ) -> tuple[str, ...]:
        effect = self.get(effect_id)
        dependencies: tuple[str, ...] = ()
        if include_data:
            dependencies += effect.data_dependencies
        if include_control:
            dependencies += effect.control_dependencies
        return tuple(dict.fromkeys(dependencies))

    def closure(
        self,
        effect_id: str,
        *,
        include_data: bool = True,
        include_control: bool = True,
    ) -> DependencyClosure:
        if effect_id not in self._effects:
            raise KeyError(f"unknown staged effect: {effect_id}")
        visited: set[str] = set()
        visiting: set[str] = set()
        missing_data: set[str] = set()
        missing_control: set[str] = set()

        def visit(current_id: str) -> None:
            if current_id in visiting:
                cycle = " -> ".join((*visiting, current_id))
                raise DependencyCycleError(f"dependency cycle involving {cycle}")
            if current_id in visited:
                return
            visiting.add(current_id)
            current = self._effects[current_id]
            dependency_sets = (
                ((current.data_dependencies if include_data else ()), missing_data),
                ((current.control_dependencies if include_control else ()), missing_control),
            )
            for dependencies, missing in dependency_sets:
                for dependency_id in dependencies:
                    if dependency_id not in self._effects:
                        missing.add(dependency_id)
                    else:
                        visit(dependency_id)
            visiting.remove(current_id)
            visited.add(current_id)

        visit(effect_id)
        return DependencyClosure(
            effect_id=effect_id,
            effect_ids=frozenset(visited),
            missing_data=frozenset(missing_data),
            missing_control=frozenset(missing_control),
        )

    dependency_closure = closure

    def descendants(
        self,
        effect_id: str,
        *,
        include_data: bool = True,
        include_control: bool = True,
    ) -> frozenset[str]:
        if effect_id not in self._effects:
            raise KeyError(f"unknown staged effect: {effect_id}")
        reverse: dict[str, set[str]] = defaultdict(set)
        for candidate in self._effects.values():
            for dependency in self.direct_dependencies(
                candidate.effect_id,
                include_data=include_data,
                include_control=include_control,
            ):
                reverse[dependency].add(candidate.effect_id)
        found: set[str] = set()
        frontier = list(reverse.get(effect_id, ()))
        while frontier:
            current = frontier.pop()
            if current in found:
                continue
            found.add(current)
            frontier.extend(reverse.get(current, ()))
        return frozenset(found)

    def atomic_groups(self) -> Mapping[str, tuple[str, ...]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for effect in self._effects.values():
            if effect.atomic_group:
                grouped[effect.atomic_group].append(effect.effect_id)
        return {key: tuple(values) for key, values in sorted(grouped.items())}

    def topological_order(
        self,
        *,
        include_data: bool = True,
        include_control: bool = True,
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(effect_id: str) -> None:
            if effect_id in visiting:
                raise DependencyCycleError(f"dependency cycle involving {effect_id}")
            if effect_id in seen:
                return
            visiting.add(effect_id)
            for dependency in self.direct_dependencies(
                effect_id,
                include_data=include_data,
                include_control=include_control,
            ):
                if dependency in self._effects:
                    visit(dependency)
            visiting.remove(effect_id)
            seen.add(effect_id)
            ordered.append(effect_id)

        for effect_id in self._effects:
            visit(effect_id)
        return tuple(ordered)

    def __contains__(self, effect_id: object) -> bool:
        return effect_id in self._effects

    def __len__(self) -> int:
        return len(self._effects)

    def __iter__(self) -> Iterator[StagedEffect]:
        return iter(self._effects.values())


__all__ = [
    "DependencyClosure",
    "DependencyCycleError",
    "DependencyError",
    "DependencyGraph",
    "MissingDependencyError",
]
