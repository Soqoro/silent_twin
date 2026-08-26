"""Dependency-free metrics used by SilentTwin experiments."""

from .calibration import (
    binary_brier_score,
    binary_entropy,
    binary_log_loss,
    expected_calibration_error,
    invalid_output_rate,
    mean_entropy_reduction,
    posterior_entropy,
)
from .confidence_intervals import (
    bootstrap_ci,
    cluster_bootstrap_statistic_ci,
    paired_bootstrap_ci,
    paired_cluster_permutation_p_value,
    paired_task_cluster_bootstrap_ci,
    task_cluster_bootstrap_ci,
)
from .exact_tv import (
    check_reachable_history_bisimulation,
    compare_enumerated_transcript_distributions,
    exact_total_variation,
)
from .power import (
    find_required_sample_size,
    make_sample_size_freeze,
    paired_discordance_rate,
    simulate_paired_binary_power,
    validate_sample_size_freeze,
)
from .privacy import accuracy, accuracy_above_prior, binary_auc
from .safety import feedback_assisted_gain, prohibited_effect_rate
from .utility import salvage_rate

__all__ = [
    "accuracy",
    "accuracy_above_prior",
    "binary_auc",
    "binary_brier_score",
    "binary_entropy",
    "binary_log_loss",
    "bootstrap_ci",
    "check_reachable_history_bisimulation",
    "cluster_bootstrap_statistic_ci",
    "compare_enumerated_transcript_distributions",
    "exact_total_variation",
    "expected_calibration_error",
    "feedback_assisted_gain",
    "find_required_sample_size",
    "invalid_output_rate",
    "make_sample_size_freeze",
    "mean_entropy_reduction",
    "paired_bootstrap_ci",
    "paired_cluster_permutation_p_value",
    "paired_discordance_rate",
    "paired_task_cluster_bootstrap_ci",
    "posterior_entropy",
    "prohibited_effect_rate",
    "salvage_rate",
    "simulate_paired_binary_power",
    "task_cluster_bootstrap_ci",
    "validate_sample_size_freeze",
]
