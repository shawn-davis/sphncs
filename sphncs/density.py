"""One-dimensional KDE evaluation, extrema, labels, and representatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass
class DensityModel:
    grid: np.ndarray
    density: np.ndarray
    boundaries: np.ndarray
    modes: np.ndarray
    labels: np.ndarray
    representative_indices: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.boundaries, values, side="right").astype(int)


def fit_density(
    values: np.ndarray,
    *,
    bandwidth: str | float = "ISJ",
    grid_points: int = 1024,
    prominence: float | None = None,
    prominence_fraction: float = 0.05,
    min_samples: int = 3,
) -> DensityModel:
    """Fit FFTKDE and turn meaningful minima into interval labels."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("KDE requires at least one value")
    lower, upper = float(values.min()), float(values.max())
    if values.size < min_samples or np.isclose(lower, upper):
        grid = np.linspace(lower - 0.5, upper + 0.5 if upper >= lower else lower + 0.5, max(2, grid_points))
        density = np.zeros_like(grid)
        mode = np.array([float(values.mean())])
        labels = np.zeros(values.size, dtype=int)
        return DensityModel(grid, density, np.array([], dtype=float), mode, labels, np.array([int(np.argmin(abs(values - mode[0])))]) )

    padding = max((upper - lower) * 0.1, np.finfo(float).eps * 100)
    grid = np.linspace(lower - padding, upper + padding, grid_points)
    try:
        from KDEpy import FFTKDE
    except ImportError as exc:  # pragma: no cover - dependency declaration is authoritative
        raise ImportError("KDEpy is required; install sphncs with its runtime dependencies.") from exc
    try:
        density = np.asarray(FFTKDE(kernel="gaussian", bw=bandwidth).fit(values).evaluate(grid), dtype=float)
    except ValueError:
        # ISJ can fail to find a root for small, highly discrete partitions.
        # Keep an explicit user-selected bandwidth strict, but make the default
        # automatic choice robust by falling back to another KDEpy rule.
        if not isinstance(bandwidth, str) or bandwidth.upper() != "ISJ":
            raise
        density = np.asarray(FFTKDE(kernel="gaussian", bw="silverman").fit(values).evaluate(grid), dtype=float)
    scale = float(np.ptp(density))
    effective_prominence = (
        prominence
        if prominence is not None
        else max(scale * prominence_fraction, np.finfo(float).eps)
    )
    minima, _ = find_peaks(-density, prominence=effective_prominence)
    boundaries = grid[minima]
    labels = np.searchsorted(boundaries, values, side="right").astype(int)

    modes: list[float] = []
    representatives: list[int] = []
    for label in range(len(boundaries) + 1):
        left = -np.inf if label == 0 else boundaries[label - 1]
        right = np.inf if label == len(boundaries) else boundaries[label]
        mask = (grid > left) & (grid <= right)
        mode = float(grid[mask][np.argmax(density[mask])])
        members = np.flatnonzero(labels == label)
        if members.size:
            modes.append(mode)
            representatives.append(int(members[np.argmin(abs(values[members] - mode))]))
    return DensityModel(grid, density, boundaries, np.asarray(modes), labels, np.asarray(representatives, dtype=int))
