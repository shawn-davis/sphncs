"""Scalar features used for optional first-stage partitioning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from math import log2

import numpy as np


def shannon_entropy(value: str) -> float:
    """Return character Shannon entropy in bits per character."""
    if not value:
        return 0.0
    total = len(value)
    return float(-sum((count / total) * log2(count / total) for count in Counter(value).values()))


def normalized_shannon_entropy(value: str) -> float:
    """Return Shannon entropy normalized by the string's observed alphabet.

    This is Pielou evenness: it measures how evenly a string uses its distinct
    characters, rather than rewarding a larger character alphabet or length.
    """
    alphabet_size = len(set(value))
    return shannon_entropy(value) / log2(alphabet_size) if alphabet_size > 1 else 0.0


PartitioningFeature = Callable[[object], float]


def partition_feature_values(
    values: Iterable[object], feature: str | PartitioningFeature,
) -> np.ndarray:
    """Evaluate a built-in string feature or a caller-supplied scalar feature."""
    values = list(values)
    if callable(feature):
        result = np.asarray([feature(value) for value in values], dtype=float)
        if not np.all(np.isfinite(result)):
            raise ValueError("partitioning_feature must produce finite numeric values")
        return result
    if feature == "length":
        return np.asarray([len(value) for value in values], dtype=float)
    if feature == "entropy":
        if any(not isinstance(value, str) for value in values):
            raise TypeError("entropy partitioning requires strings")
        return np.asarray([shannon_entropy(value) for value in values], dtype=float)
    if feature == "normalized_entropy":
        if any(not isinstance(value, str) for value in values):
            raise TypeError("normalized_entropy partitioning requires strings")
        return np.asarray([normalized_shannon_entropy(value) for value in values], dtype=float)
    raise ValueError(f"unsupported partitioning feature: {feature}")
