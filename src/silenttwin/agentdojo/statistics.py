"""Suite-stratified, structural-scenario clustered AgentDojo statistics."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from itertools import product
from typing import Any, Callable, Hashable, Mapping, Sequence, TypeVar

from .config import AGENTDOJO_SUITES, stable_hash


T = TypeVar("T")
DEFAULT_RESAMPLES = 5000
DEFAULT_SEED = 20260824
SUITE_WEIGHTING_RULES = ("equal_suite", "task_weighted")


class AgentDojoStatisticsError(ValueError):
    pass


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AgentDojoStatisticsError("cannot take a quantile of an empty sample")
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_cluster_strata(
    observations: Sequence[T],
    cluster_ids: Sequence[Hashable],
    suite_strata: Sequence[str],
) -> dict[str, dict[Hashable, list[T]]]:
    if not (len(observations) == len(cluster_ids) == len(suite_strata)):
        raise AgentDojoStatisticsError("observations, clusters, and suites differ in length")
    if not observations:
        raise AgentDojoStatisticsError("clustered inference requires observations")
    grouped: dict[str, dict[Hashable, list[T]]] = defaultdict(lambda: defaultdict(list))
    cluster_suite: dict[Hashable, str] = {}
    for observation, cluster_id, suite in zip(observations, cluster_ids, suite_strata):
        if suite not in AGENTDOJO_SUITES:
            raise AgentDojoStatisticsError(f"unknown AgentDojo suite {suite!r}")
        if cluster_id in cluster_suite and cluster_suite[cluster_id] != suite:
            raise AgentDojoStatisticsError("one structural cluster appears in multiple suites")
        cluster_suite[cluster_id] = suite
        grouped[suite][cluster_id].append(observation)
    return {suite: dict(clusters) for suite, clusters in grouped.items()}


def suite_stratified_cluster_bootstrap_ci(
    observations: Sequence[T],
    cluster_ids: Sequence[Hashable],
    suite_strata: Sequence[str],
    statistic: Callable[[Sequence[T]], float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    suite_weighting: str = "equal_suite",
) -> tuple[float, float]:
    """Resample complete structural scenarios within each suite.

    The number of clusters in each observed suite is held fixed. Repeated
    theta/donor, environment, injection, and decoding rows therefore move as
    one unit and cannot inflate ``n``.
    """

    if not 0 < confidence < 1 or resamples <= 0:
        raise AgentDojoStatisticsError("invalid bootstrap confidence/resample count")
    if suite_weighting not in SUITE_WEIGHTING_RULES:
        raise AgentDojoStatisticsError(
            f"suite_weighting must be one of {SUITE_WEIGHTING_RULES}"
        )
    grouped = _validate_cluster_strata(observations, cluster_ids, suite_strata)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sample: list[T] = []
        suite_estimates: list[float] = []
        for suite in AGENTDOJO_SUITES:
            clusters = grouped.get(suite)
            if not clusters:
                continue
            identifiers = tuple(clusters)
            suite_sample: list[T] = []
            for _ in identifiers:
                selected = identifiers[rng.randrange(len(identifiers))]
                suite_sample.extend(clusters[selected])
            sample.extend(suite_sample)
            if suite_weighting == "equal_suite":
                suite_estimates.append(float(statistic(suite_sample)))
        estimate = (
            sum(suite_estimates) / len(suite_estimates)
            if suite_weighting == "equal_suite"
            and suite_estimates
            and all(math.isfinite(item) for item in suite_estimates)
            else float(statistic(sample))
            if suite_weighting == "task_weighted"
            else float("nan")
        )
        if math.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        raise AgentDojoStatisticsError("statistic was undefined in every bootstrap resample")
    alpha = (1 - confidence) / 2
    return _quantile(estimates, alpha), _quantile(estimates, 1 - alpha)


def _binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return ``P[X <= successes]`` for a binomial random variable.

    The log-sum-exp implementation keeps the exact-bound inversion below
    dependency free without relying on factorial-sized intermediate floats.
    """

    if not 0 <= successes <= trials:
        raise AgentDojoStatisticsError("binomial successes must lie in [0, trials]")
    if not 0.0 <= probability <= 1.0:
        raise AgentDojoStatisticsError("binomial probability must lie in [0,1]")
    if successes == trials or probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0
    log_probability = math.log(probability)
    log_complement = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_probability
        + (trials - index) * log_complement
        for index in range(successes + 1)
    ]
    maximum = max(terms)
    if not math.isfinite(maximum):
        return 0.0
    return min(1.0, math.exp(maximum) * sum(math.exp(item - maximum) for item in terms))


def _clopper_pearson_one_sided_upper(
    successes: int, trials: int, *, alpha: float
) -> float:
    """Exact one-sided binomial upper confidence limit."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise AgentDojoStatisticsError("exact binomial bound requires valid counts")
    if not 0.0 < alpha < 1.0:
        raise AgentDojoStatisticsError("exact binomial alpha must lie in (0,1)")
    if successes == trials:
        return 1.0
    lower = successes / trials
    upper = 1.0
    # The binomial CDF is monotone decreasing in p.  Solve
    # P_p[X <= successes] = alpha by deterministic bisection.
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def suite_stratified_cluster_binary_upper_bound(
    observations: Sequence[int | bool | float],
    cluster_ids: Sequence[Hashable],
    suite_strata: Sequence[str],
    *,
    confidence: float = 0.95,
    suite_weighting: str = "equal_suite",
) -> dict[str, Any]:
    """Conservative finite-sample upper bound for a rare binary violation.

    A structural cluster is counted as a violation when *any* nested row has
    a positive outcome.  Per-suite one-sided Clopper--Pearson bounds use a
    Bonferroni allocation across observed suites; averaging those simultaneous
    bounds preserves the configured suite weighting.  Unlike a percentile
    bootstrap, the bound is strictly positive after finitely many all-zero
    observations, so a rare-event safety gate cannot pass merely because no
    event happened to be sampled.
    """

    if not 0.0 < confidence < 1.0:
        raise AgentDojoStatisticsError("upper-bound confidence must lie in (0,1)")
    if suite_weighting not in SUITE_WEIGHTING_RULES:
        raise AgentDojoStatisticsError(
            f"suite_weighting must be one of {SUITE_WEIGHTING_RULES}"
        )
    grouped = _validate_cluster_strata(observations, cluster_ids, suite_strata)
    observed_suites = [suite for suite in AGENTDOJO_SUITES if grouped.get(suite)]
    if not observed_suites:
        raise AgentDojoStatisticsError("upper bound requires observed suite strata")
    per_suite_alpha = (1.0 - confidence) / len(observed_suites)
    suite_bounds: dict[str, dict[str, Any]] = {}
    for suite in observed_suites:
        clusters = grouped[suite]
        indicators: list[int] = []
        for nested in clusters.values():
            numeric = []
            for value in nested:
                if isinstance(value, bool):
                    numeric.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    numeric.append(float(value))
                else:
                    raise AgentDojoStatisticsError(
                        "cluster binary bound received a non-finite observation"
                    )
            if any(not 0.0 <= value <= 1.0 for value in numeric):
                raise AgentDojoStatisticsError(
                    "cluster binary observations must lie in [0,1]"
                )
            indicators.append(int(any(value > 0.0 for value in numeric)))
        successes = sum(indicators)
        trials = len(indicators)
        suite_bounds[suite] = {
            "cluster_count": trials,
            "event_cluster_count": successes,
            "estimate": successes / trials,
            "ci_upper": _clopper_pearson_one_sided_upper(
                successes, trials, alpha=per_suite_alpha
            ),
            "simultaneous_one_sided_confidence": 1.0 - per_suite_alpha,
        }
    if suite_weighting == "equal_suite":
        estimate = sum(item["estimate"] for item in suite_bounds.values()) / len(
            suite_bounds
        )
        upper = sum(item["ci_upper"] for item in suite_bounds.values()) / len(
            suite_bounds
        )
    else:
        total = sum(item["cluster_count"] for item in suite_bounds.values())
        estimate = sum(
            item["estimate"] * item["cluster_count"]
            for item in suite_bounds.values()
        ) / total
        upper = sum(
            item["ci_upper"] * item["cluster_count"]
            for item in suite_bounds.values()
        ) / total
    return {
        "estimate": estimate,
        "ci_level": confidence,
        "ci_upper": upper,
        "ci_method": (
            "bonferroni_simultaneous_one_sided_clopper_pearson_"
            "structural_cluster_any_event"
        ),
        "suite_weighting": suite_weighting,
        "independent_unit": "structural_group_id",
        "independent_unit_count": sum(
            item["cluster_count"] for item in suite_bounds.values()
        ),
        "suite_bounds": suite_bounds,
    }


def collapse_repeated_measurements(
    records: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    condition_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Average stochastic/nested rows before primary inference."""

    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for index, record in enumerate(records):
        suite = record.get("agentdojo_suite")
        cluster = record.get("structural_group_id")
        if suite not in AGENTDOJO_SUITES or not isinstance(cluster, str) or not cluster:
            raise AgentDojoStatisticsError(f"record {index} lacks suite/structural group")
        raw = record.get(metric)
        if isinstance(raw, bool):
            value = float(raw)
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            value = float(raw)
        else:
            raise AgentDojoStatisticsError(f"record {index} has no numeric {metric}")
        condition = tuple(record.get(name) for name in condition_fields)
        key = (suite, cluster, *condition)
        grouped[key].append(value)
        metadata[key] = {
            "agentdojo_suite": suite,
            "structural_group_id": cluster,
            **{name: record.get(name) for name in condition_fields},
        }
    result = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        result.append(
            {
                **metadata[key],
                metric: sum(grouped[key]) / len(grouped[key]),
                "nested_row_count": len(grouped[key]),
            }
        )
    return result


def paired_scenario_contrast(
    target: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    contrast_id: str,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    suite_weighting: str = "equal_suite",
) -> dict[str, Any]:
    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for row in rows:
            key = (str(row["agentdojo_suite"]), str(row["structural_group_id"]))
            if key in result:
                raise AgentDojoStatisticsError("contrast rows were not collapsed by scenario")
            result[key] = float(row[metric])
        return result

    left = index(target)
    right = index(reference)
    if set(left) != set(right) or not left:
        raise AgentDojoStatisticsError("paired contrast requires identical scenario cohorts")
    keys = sorted(left)
    differences = [left[key] - right[key] for key in keys]
    clusters = [key[1] for key in keys]
    suites = [key[0] for key in keys]
    if suite_weighting not in SUITE_WEIGHTING_RULES:
        raise AgentDojoStatisticsError(
            f"suite_weighting must be one of {SUITE_WEIGHTING_RULES}"
        )
    suite_means = [
        sum(value for value, observed in zip(differences, suites) if observed == suite)
        / sum(observed == suite for observed in suites)
        for suite in AGENTDOJO_SUITES
        if suite in suites
    ]
    task_weighted_estimate = sum(differences) / len(differences)
    estimate = (
        sum(suite_means) / len(suite_means)
        if suite_weighting == "equal_suite"
        else task_weighted_estimate
    )
    lower, upper = suite_stratified_cluster_bootstrap_ci(
        differences,
        clusters,
        suites,
        lambda rows: sum(rows) / len(rows),
        confidence=confidence,
        resamples=resamples,
        seed=seed ^ int(stable_hash([contrast_id, metric])[:8], 16),
        suite_weighting=suite_weighting,
    )
    sensitivity_lower, sensitivity_upper = suite_stratified_cluster_bootstrap_ci(
        differences,
        clusters,
        suites,
        lambda rows: sum(rows) / len(rows),
        confidence=confidence,
        resamples=resamples,
        seed=seed ^ int(stable_hash([contrast_id, metric, "task_weighted"])[:8], 16),
        suite_weighting="task_weighted",
    )
    by_suite: dict[str, dict[str, Any]] = {}
    for suite in AGENTDOJO_SUITES:
        suite_values = [value for value, observed_suite in zip(differences, suites) if observed_suite == suite]
        if not suite_values:
            continue
        suite_clusters = [
            cluster for cluster, observed_suite in zip(clusters, suites) if observed_suite == suite
        ]
        suite_lower, suite_upper = suite_stratified_cluster_bootstrap_ci(
            suite_values,
            suite_clusters,
            [suite] * len(suite_values),
            lambda rows: sum(rows) / len(rows),
            confidence=confidence,
            resamples=resamples,
            seed=seed ^ int(stable_hash([contrast_id, metric, suite])[:8], 16),
        )
        by_suite[suite] = {
            "estimate": sum(suite_values) / len(suite_values),
            "ci_lower": suite_lower,
            "ci_upper": suite_upper,
            "independent_unit_count": len(suite_values),
        }
    return {
        "contrast_id": contrast_id,
        "metric": metric,
        "estimate": estimate,
        "ci_level": confidence,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_method": "suite_stratified_structural_scenario_cluster_bootstrap",
        "suite_weighting": suite_weighting,
        "task_weighted_sensitivity_estimate": task_weighted_estimate,
        "task_weighted_sensitivity_ci_lower": sensitivity_lower,
        "task_weighted_sensitivity_ci_upper": sensitivity_upper,
        "independent_unit": "structural_group_id",
        "independent_unit_count": len(keys),
        "suite_strata": by_suite,
        "task_level_paired_estimates": [
            {
                "agentdojo_suite": suite,
                "structural_group_id": group,
                "target": left[(suite, group)],
                "reference": right[(suite, group)],
                "difference": left[(suite, group)] - right[(suite, group)],
            }
            for suite, group in keys
        ],
        "paired_sign_flip_p_value": paired_sign_flip_p_value(
            differences,
            seed=seed ^ int(stable_hash([contrast_id, metric, "sign_flip"])[:8], 16),
        ),
    }


def paired_sign_flip_p_value(
    differences: Sequence[float],
    *,
    permutations: int = 10000,
    seed: int = DEFAULT_SEED,
    exact_limit: int = 18,
) -> float:
    """Two-sided paired randomization test on collapsed scenario differences."""

    values = tuple(float(item) for item in differences)
    if not values:
        raise AgentDojoStatisticsError("paired sign-flip test requires differences")
    observed = abs(sum(values) / len(values))
    if len(values) <= exact_limit:
        extreme = 0
        total = 0
        for signs in product((-1.0, 1.0), repeat=len(values)):
            estimate = abs(sum(sign * value for sign, value in zip(signs, values)) / len(values))
            extreme += estimate >= observed - 1e-15
            total += 1
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(permutations):
        estimate = abs(
            sum((-value if rng.randrange(2) else value) for value in values) / len(values)
        )
        extreme += estimate >= observed - 1e-15
    return (extreme + 1) / (permutations + 1)


def binary_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise AgentDojoStatisticsError("AUC labels/scores are empty or unmatched")
    positive = [float(score) for label, score in zip(labels, scores) if bool(label)]
    negative = [float(score) for label, score in zip(labels, scores) if not bool(label)]
    if not positive or not negative:
        return float("nan")
    wins = 0.0
    for pos in positive:
        for neg in negative:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return wins / (len(positive) * len(negative))


def clustered_auc(
    records: Sequence[Mapping[str, Any]],
    *,
    label_field: str,
    score_field: str,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    suite_weighting: str = "equal_suite",
) -> dict[str, Any]:
    observations: list[tuple[int, float]] = []
    clusters: list[str] = []
    suites: list[str] = []
    for row in records:
        observations.append((int(bool(row[label_field])), float(row[score_field])))
        clusters.append(str(row["structural_group_id"]))
        suites.append(str(row["agentdojo_suite"]))
    if suite_weighting not in SUITE_WEIGHTING_RULES:
        raise AgentDojoStatisticsError(
            f"suite_weighting must be one of {SUITE_WEIGHTING_RULES}"
        )
    task_weighted_estimate = binary_auc(
        [row[0] for row in observations], [row[1] for row in observations]
    )
    suite_estimates = []
    for suite in AGENTDOJO_SUITES:
        suite_rows = [
            row for row, observed_suite in zip(observations, suites) if observed_suite == suite
        ]
        if suite_rows:
            suite_estimates.append(
                binary_auc([row[0] for row in suite_rows], [row[1] for row in suite_rows])
            )
    estimate = (
        sum(suite_estimates) / len(suite_estimates)
        if suite_weighting == "equal_suite"
        and suite_estimates
        and all(math.isfinite(item) for item in suite_estimates)
        else task_weighted_estimate
        if suite_weighting == "task_weighted"
        else float("nan")
    )
    lower, upper = suite_stratified_cluster_bootstrap_ci(
        observations,
        clusters,
        suites,
        lambda rows: binary_auc([row[0] for row in rows], [row[1] for row in rows]),
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        suite_weighting=suite_weighting,
    )
    sensitivity_lower, sensitivity_upper = suite_stratified_cluster_bootstrap_ci(
        observations,
        clusters,
        suites,
        lambda rows: binary_auc([row[0] for row in rows], [row[1] for row in rows]),
        confidence=confidence,
        resamples=resamples,
        seed=seed ^ int(stable_hash(["auc", "task_weighted"])[:8], 16),
        suite_weighting="task_weighted",
    )
    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_level": confidence,
        "ci_method": "suite_stratified_structural_scenario_cluster_bootstrap",
        "suite_weighting": suite_weighting,
        "task_weighted_sensitivity_estimate": task_weighted_estimate,
        "task_weighted_sensitivity_ci_lower": sensitivity_lower,
        "task_weighted_sensitivity_ci_upper": sensitivity_upper,
    }


def attack_error_accounting(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float | int | None]:
    if not records:
        raise AgentDojoStatisticsError("attack accounting requires records")
    valid = [row for row in records if row.get("run_valid") is True]
    errors = len(records) - len(valid)
    valid_successes = sum(bool(row.get("attack_success")) for row in valid)
    # Fail closed for a safety claim: an invalid run cannot be used to lower ASR.
    conservative_successes = valid_successes + errors
    return {
        "row_count": len(records),
        "valid_run_count": len(valid),
        "error_count": errors,
        # JSON artifacts are emitted with ``allow_nan=False``.  ``None`` is
        # also the scientifically honest value when no valid denominator
        # exists; NaN would make an all-error cell impossible to publish.
        "valid_run_asr": valid_successes / len(valid) if valid else None,
        "conservative_asr": conservative_successes / len(records),
        "error_rate": errors / len(records),
    }


def tost_equivalence_from_interval(
    contrast: Mapping[str, Any], *, margin: float
) -> dict[str, Any]:
    if not 0 < margin < 1:
        raise AgentDojoStatisticsError("TOST margin must lie in (0,1)")
    confidence = float(contrast.get("ci_level", 0.0))
    lower = float(contrast["ci_lower"])
    upper = float(contrast["ci_upper"])
    passed = confidence == 0.90 and lower > -margin and upper < margin
    return {
        "method": "two_one_sided_tests_via_90pct_ci_inversion",
        "margin": margin,
        "ci_level": confidence,
        "ci_lower": lower,
        "ci_upper": upper,
        "equivalent": passed,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if any(not 0 <= float(value) <= 1 for value in p_values.values()):
        raise AgentDojoStatisticsError("Holm p-values must lie in [0,1]")
    ordered = sorted(p_values.items(), key=lambda item: (float(item[1]), item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "SUITE_WEIGHTING_RULES",
    "AgentDojoStatisticsError",
    "attack_error_accounting",
    "binary_auc",
    "clustered_auc",
    "collapse_repeated_measurements",
    "holm_adjust",
    "paired_sign_flip_p_value",
    "paired_scenario_contrast",
    "suite_stratified_cluster_bootstrap_ci",
    "suite_stratified_cluster_binary_upper_bound",
    "tost_equivalence_from_interval",
]
