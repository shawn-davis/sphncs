"""Reusable distance metrics for SPHNCS object spaces."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from functools import lru_cache
from math import inf, isfinite
from numbers import Real

Metric = Callable[[object, object], float]
StringMetric = Callable[[str, str], float]


def lp_distance(left: Iterable[Real], right: Iterable[Real], p: Real = 2) -> float:
    """Return Minkowski Lp distance between two equally sized numeric vectors.

    ``p`` must be at least one. ``p=float('inf')`` returns the Chebyshev
    (L-infinity) distance. Inputs are materialized once so any finite iterable
    of real values is accepted.
    """
    if isinstance(p, bool) or not isinstance(p, Real) or p < 1:
        raise ValueError("p must be a real number greater than or equal to 1")
    left_values = tuple(left)
    right_values = tuple(right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have equal lengths")
    try:
        differences = [abs(float(a) - float(b)) for a, b in zip(left_values, right_values)]
    except (TypeError, ValueError) as exc:
        raise TypeError("vectors must contain real numeric values") from exc
    if not all(isfinite(value) for value in differences):
        raise ValueError("vectors must contain finite numeric values")
    if p == inf:
        return max(differences, default=0.0)
    return float(sum(value**p for value in differences) ** (1 / p))


def l1_distance(left: Iterable[Real], right: Iterable[Real]) -> float:
    """Return Manhattan (L1) distance between numeric vectors."""
    return lp_distance(left, right, p=1)


def l2_distance(left: Iterable[Real], right: Iterable[Real]) -> float:
    """Return Euclidean (L2) distance between numeric vectors."""
    return lp_distance(left, right, p=2)


@lru_cache(maxsize=32_768)
def _char_ngram_counts(value: str, ngram_size: int) -> Counter[str]:
    """Build a multiset of character shingles once per immutable input string."""
    return Counter(value[i : i + ngram_size] for i in range(max(1, len(value) - ngram_size + 1)))


def normalized_levenshtein(left: str, right: str) -> float:
    """Levenshtein edit distance normalized to the interval [0, 1]."""
    if left == right:
        return 0.0
    if not left or not right:
        return 1.0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, start=1):
        current = [i]
        for j, char_right in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_left != char_right)))
        previous = current
    return previous[-1] / len(left)


def char_ngram_jaccard(left: str, right: str, ngram_size: int = 3) -> float:
    """Multiset character n-gram Jaccard distance."""
    if ngram_size < 1:
        raise ValueError("ngram_size must be positive")
    if left == right:
        return 0.0
    grams_left = _char_ngram_counts(left, ngram_size)
    grams_right = _char_ngram_counts(right, ngram_size)
    union = sum((grams_left | grams_right).values())
    return 1.0 - sum((grams_left & grams_right).values()) / union if union else 0.0


METRICS: dict[str, Metric] = {
    "l1": l1_distance,
    "l2": l2_distance,
    "normalized_levenshtein": normalized_levenshtein,
    "char_ngram_jaccard": char_ngram_jaccard,
}


def resolve_metric(metric: str | Metric) -> Metric:
    if callable(metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError as exc:
        choices = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric {metric!r}; choose one of {choices}, or pass a callable.") from exc
