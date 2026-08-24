from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any


Q_ERROR_FLOOR = 1.0
DEFAULT_EQUALITY_TOLERANCE = 1e-9


def q_error(estimate: float, actual: float) -> float:
    """Zero-aware Q-error using max(value, 1), while callers retain raw estimates."""
    estimate_value = _finite_nonnegative(estimate, "estimate")
    actual_value = _finite_nonnegative(actual, "actual")
    safe_estimate = max(estimate_value, Q_ERROR_FLOOR)
    safe_actual = max(actual_value, Q_ERROR_FLOOR)
    return max(safe_estimate / safe_actual, safe_actual / safe_estimate)


def signed_error_ratio(estimate: float, actual: float) -> float:
    """Directional estimate/actual ratio under the same explicit floor as Q-error."""
    estimate_value = _finite_nonnegative(estimate, "estimate")
    actual_value = _finite_nonnegative(actual, "actual")
    return max(estimate_value, Q_ERROR_FLOOR) / max(actual_value, Q_ERROR_FLOOR)


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    clean = finite_values(values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    ordered = sorted(clean)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def geometric_mean(values: Sequence[float]) -> float | None:
    clean = finite_values(values)
    if not clean:
        return None
    if any(value <= 0 for value in clean):
        raise ValueError("geometric mean requires strictly positive values")
    return math.exp(statistics.fmean(math.log(value) for value in clean))


def summarize(values: Sequence[float]) -> dict[str, float | int] | None:
    clean = finite_values(values)
    if not clean:
        return None
    thresholds = {
        "le1_5": sum(value <= 1.5 for value in clean),
        "le2": sum(value <= 2.0 for value in clean),
        "le5": sum(value <= 5.0 for value in clean),
        "gt10": sum(value > 10.0 for value in clean),
        "gt100": sum(value > 100.0 for value in clean),
    }
    result: dict[str, float | int] = {
        "count": len(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "geometric_mean": float(geometric_mean(clean)),
        "p75": float(percentile(clean, 0.75)),
        "p90": float(percentile(clean, 0.90)),
        "p95": float(percentile(clean, 0.95)),
        "p99": float(percentile(clean, 0.99)),
        "max": max(clean),
    }
    for name, count in thresholds.items():
        result[f"{name}_count"] = count
        result[f"{name}_fraction"] = count / len(clean)
        result[f"{name}_percent"] = count * 100.0 / len(clean)
    return result


def summarize_estimator(rows: Sequence[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    if not rows:
        return None
    q_errors = [float(row[f"{prefix}_q_error"]) for row in rows]
    ratios = [float(row[f"{prefix}_signed_error_ratio"]) for row in rows]
    summary = summarize(q_errors)
    assert summary is not None
    summary.update(
        {
            "median_signed_ratio": statistics.median(finite_values(ratios)),
            "overestimation_rate": sum(ratio > 1.0 + DEFAULT_EQUALITY_TOLERANCE for ratio in ratios)
            / len(ratios),
            "underestimation_rate": sum(ratio < 1.0 - DEFAULT_EQUALITY_TOLERANCE for ratio in ratios)
            / len(ratios),
        }
    )
    return summary


def paired_outcomes(
    left: Sequence[float],
    right: Sequence[float],
    *,
    tolerance: float = DEFAULT_EQUALITY_TOLERANCE,
) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired outcome inputs must have equal length")
    left_values, right_values = finite_values(left), finite_values(right)
    if len(left_values) != len(left) or len(right_values) != len(right):
        raise ValueError("paired outcome inputs contain non-finite values")
    counts = {"left_better": 0, "right_better": 0, "equal": 0}
    for left_value, right_value in zip(left_values, right_values):
        scale = max(abs(left_value), abs(right_value), 1.0)
        if abs(left_value - right_value) <= tolerance * scale:
            counts["equal"] += 1
        elif left_value < right_value:
            counts["left_better"] += 1
        else:
            counts["right_better"] += 1
    return counts


def paired_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    metric: Callable[[Sequence[float]], float],
    *,
    samples: int = 10_000,
    random_seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    return _bootstrap_ci(
        left,
        right,
        metric,
        clusters=None,
        samples=samples,
        random_seed=random_seed,
        confidence=confidence,
    )


def paired_cluster_bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    clusters: Sequence[Hashable],
    metric: Callable[[Sequence[float]], float],
    *,
    samples: int = 10_000,
    random_seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    return _bootstrap_ci(
        left,
        right,
        metric,
        clusters=clusters,
        samples=samples,
        random_seed=random_seed,
        confidence=confidence,
    )


def paired_mean_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    samples: int = 10_000,
    random_seed: int = 42,
) -> dict[str, float | int | None]:
    return paired_bootstrap_ci(
        left,
        right,
        statistics.fmean,
        samples=samples,
        random_seed=random_seed,
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid success/total counts")
    if total == 0:
        return {"successes": successes, "total": total, "rate": None, "ci_low": None, "ci_high": None}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
    }


def proportion_le2(values: Sequence[float]) -> float:
    clean = finite_values(values)
    return 0.0 if not clean else sum(value <= 2.0 for value in clean) / len(clean)


def finite_values(values: Sequence[float]) -> list[float]:
    output: list[float] = []
    for value in values:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Aggregate input contains a non-finite value: {value!r}")
        output.append(numeric)
    return output


def finite_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _bootstrap_ci(
    left: Sequence[float],
    right: Sequence[float],
    metric: Callable[[Sequence[float]], float],
    *,
    clusters: Sequence[Hashable] | None,
    samples: int,
    random_seed: int,
    confidence: float,
) -> dict[str, float | int | None]:
    if len(left) != len(right):
        raise ValueError("paired bootstrap inputs must have the same length")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    left_values, right_values = finite_values(left), finite_values(right)
    if not left_values:
        return {
            "sample_count": 0,
            "cluster_count": 0 if clusters is not None else None,
            "bootstrap_samples": samples,
            "random_seed": random_seed,
            "observed_delta": None,
            "ci_low": None,
            "ci_high": None,
            "confidence": confidence,
        }
    rng = random.Random(random_seed)
    observed_delta = float(metric(left_values) - metric(right_values))
    deltas: list[float] = []
    n = len(left_values)
    cluster_groups: dict[Hashable, list[int]] | None = None
    cluster_keys: list[Hashable] = []
    if clusters is not None:
        if len(clusters) != n:
            raise ValueError("cluster labels must have the same length as bootstrap inputs")
        cluster_groups = defaultdict(list)
        for index, cluster in enumerate(clusters):
            cluster_groups[cluster].append(index)
        cluster_keys = list(cluster_groups)
    for _ in range(samples):
        if cluster_groups is None:
            indices = [rng.randrange(n) for _ in range(n)]
        else:
            sampled_clusters = [rng.choice(cluster_keys) for _ in cluster_keys]
            indices = [index for cluster in sampled_clusters for index in cluster_groups[cluster]]
        left_sample = [left_values[index] for index in indices]
        right_sample = [right_values[index] for index in indices]
        delta = float(metric(left_sample) - metric(right_sample))
        if not math.isfinite(delta):
            raise ValueError("bootstrap metric produced a non-finite delta")
        deltas.append(delta)
    alpha = (1.0 - confidence) / 2.0
    return {
        "sample_count": n,
        "cluster_count": None if cluster_groups is None else len(cluster_groups),
        "bootstrap_samples": samples,
        "random_seed": random_seed,
        "observed_delta": observed_delta,
        "ci_low": percentile(deltas, alpha),
        "ci_high": percentile(deltas, 1.0 - alpha),
        "confidence": confidence,
    }


def _finite_nonnegative(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return numeric
