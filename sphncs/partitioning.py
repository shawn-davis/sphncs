"""Scalar string features used for optional first-stage partitioning."""

from __future__ import annotations

from collections import Counter
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


def partition_feature_values(values: list[str], feature: str) -> np.ndarray:
    """Compute the configured first-stage partition feature for strings."""
    if feature == "length":
        return np.asarray([len(value) for value in values], dtype=float)
    if feature == "entropy":
        return np.asarray([shannon_entropy(value) for value in values], dtype=float)
    if feature == "normalized_entropy":
        return np.asarray([normalized_shannon_entropy(value) for value in values], dtype=float)
    raise ValueError(f"unsupported partitioning feature: {feature}")
