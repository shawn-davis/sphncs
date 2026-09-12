"""String distances suitable for FastMap's on-demand metric interface."""

from __future__ import annotations

from collections import Counter
from typing import Callable

StringMetric = Callable[[str, str], float]


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
    grams_left = Counter(left[i : i + ngram_size] for i in range(max(1, len(left) - ngram_size + 1)))
    grams_right = Counter(right[i : i + ngram_size] for i in range(max(1, len(right) - ngram_size + 1)))
    union = sum((grams_left | grams_right).values())
    return 1.0 - sum((grams_left & grams_right).values()) / union if union else 0.0


METRICS: dict[str, StringMetric] = {
    "normalized_levenshtein": normalized_levenshtein,
    "char_ngram_jaccard": char_ngram_jaccard,
}


def resolve_metric(metric: str | StringMetric) -> StringMetric:
    if callable(metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError as exc:
        choices = ", ".join(sorted(METRICS))
        raise ValueError(f"Unknown metric {metric!r}; choose one of {choices}, or pass a callable.") from exc
