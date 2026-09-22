"""Adapter around fastmapy's batched, on-demand FastMap implementation."""

from __future__ import annotations

import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_fastmap():
    """Import the installed package or the checked-out project submodule."""
    try:
        from fastmap import Distance, FastMap
        return Distance, FastMap
    except ImportError:
        submodule = Path(__file__).resolve().parent.parent / "vendor" / "fastmapy"
        if submodule.is_dir():
            sys.path.insert(0, str(submodule))
            try:
                from fastmap import Distance, FastMap
                return Distance, FastMap
            except ImportError:
                pass
    raise ImportError(
        "fastmapy is required. Install FastMapy or initialize and install the "
        "project submodule with `git submodule update --init --recursive` and "
        "`python -m pip install -e vendor/fastmapy`."
    )


@dataclass
class _StoredPivot:
    """Pickle-safe representation of a fastmapy pivot."""

    left: object
    left_proj: np.ndarray
    right: object
    right_proj: np.ndarray
    distance: float


class _StoredFastMapModel:
    """A package-owned FastMap projection used when reloading a saved model."""

    def __init__(self, metric: Callable[[object, object], float], pivots: list[_StoredPivot], dim: int):
        self._metric = metric
        self._pivots = pivots
        self._dim = dim

    @classmethod
    def from_fastmap(cls, model, metric: Callable[[object, object], float]):
        try:
            pivots = [
                _StoredPivot(
                    pivot.left,
                    np.asarray(pivot.left_proj, dtype=float).copy(),
                    pivot.right,
                    np.asarray(pivot.right_proj, dtype=float).copy(),
                    float(pivot.distance),
                )
                for pivot in model._pivots
            ]
            return cls(metric, pivots, int(model._dim))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("unsupported fastmapy model state") from exc

    def _dist(self, left, left_proj, right, right_proj, index: int) -> float:
        squared = float(self._metric(left, right)) ** 2
        residual = float(np.sum((left_proj[:index] - right_proj[:index]) ** 2))
        return float(np.sqrt(max(squared - residual, 0.0)))

    def _projection(self, value: object, index: int) -> np.ndarray:
        projection = np.zeros(self._dim)
        for dimension in range(index):
            pivot = self._pivots[dimension]
            if pivot.distance == 0:
                continue
            left_distance = self._dist(pivot.left, pivot.left_proj, value, projection, index)
            right_distance = self._dist(pivot.right, pivot.right_proj, value, projection, index)
            projection[dimension] = (left_distance**2 + pivot.distance**2 - right_distance**2) / (2 * pivot.distance)
        return projection

    def transform(self, X: list[object]) -> list[np.ndarray]:
        return [self._projection(value, self._dim) for value in X]


class FastMapyEmbeddings:
    """A batch of independent one-dimensional fastmapy models.

    Each model gets a distinct pivot pair through ``FastMap.fit_many``. This
    intentionally differs from a single multi-dimensional FastMap embedding:
    sphncs uses every independent 1-D density partition as consensus evidence.
    """

    def __init__(
        self, n_embeddings: int, metric: Callable[[object, object], float], *,
        random_state: int | None = None, cores: int = 1, iters: int = 3,
        cache_distances: bool = True,
    ):
        if n_embeddings < 1:
            raise ValueError("n_embeddings must be at least 1")
        self.n_embeddings = n_embeddings
        self.metric = metric
        self.random_state = random_state
        self.cores = cores
        self.iters = iters
        self.cache_distances = cache_distances

    def fit(self, X: list[object]):
        if not X:
            raise ValueError("FastMap requires at least one string")
        self.X_ = list(X)
        if len(X) == 1:
            self.models_ = []
            self.coordinates_ = np.zeros((1, self.n_embeddings), dtype=float)
            return self
        Distance, FastMap = _load_fastmap()
        metric = self.metric

        class CallableDistance(Distance):
            @staticmethod
            def get_name():
                return getattr(metric, "__name__", "sphncs_metric")

            def calculate(self, left, right) -> float:
                value = float(metric(left, right))
                if value < 0:
                    raise ValueError("Metrics must return non-negative distances")
                return value

        count = min(self.n_embeddings, len(X))
        # fastmapy currently draws its pivot starts from Python's module-level
        # RNG. Restore it afterwards so fitting sphncs does not perturb callers.
        rng_state = random.getstate()
        try:
            if self.random_state is not None:
                random.seed(self.random_state)
            self.models_ = FastMap.fit_many(
                X,
                count=count,
                dim=1,
                distance=CallableDistance,
                cores=self.cores,
                iters=self.iters,
                cache_distances=self.cache_distances,
            )
        finally:
            random.setstate(rng_state)
        coordinates = np.column_stack([np.asarray(model.transform(X), dtype=float).reshape(-1) for model in self.models_])
        if count < self.n_embeddings:
            coordinates = np.pad(coordinates, ((0, 0), (0, self.n_embeddings - count)))
        self.coordinates_ = coordinates
        return self

    def fit_transform(self, X: list[object]) -> np.ndarray:
        return self.fit(X).coordinates_.copy()

    def transform(self, X: list[object]) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("FastMap must be fitted before transform")
        if not self.models_:
            return np.zeros((len(X), self.n_embeddings), dtype=float)
        coordinates = np.column_stack([np.asarray(model.transform(X), dtype=float).reshape(-1) for model in self.models_])
        if coordinates.shape[1] < self.n_embeddings:
            coordinates = np.pad(coordinates, ((0, 0), (0, self.n_embeddings - coordinates.shape[1])))
        return coordinates

    def __getstate__(self):
        """Replace fastmapy's non-pickleable local distance class on save."""
        state = self.__dict__.copy()
        if "models_" in state:
            state["models_"] = [
                model
                if isinstance(model, _StoredFastMapModel)
                else _StoredFastMapModel.from_fastmap(model, self.metric)
                for model in state["models_"]
            ]
        return state
