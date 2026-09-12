"""Adapter around fastmapy's batched, on-demand FastMap implementation."""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Callable

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


class FastMapyEmbeddings:
    """A batch of independent one-dimensional fastmapy models.

    Each model gets a distinct pivot pair through ``FastMap.fit_many``. This
    intentionally differs from a single multi-dimensional FastMap embedding:
    sphncs uses every independent 1-D density partition as consensus evidence.
    """

    def __init__(
        self, n_embeddings: int, metric: Callable[[str, str], float], *,
        random_state: int | None = None, cores: int = 1,
    ):
        if n_embeddings < 1:
            raise ValueError("n_embeddings must be at least 1")
        self.n_embeddings = n_embeddings
        self.metric = metric
        self.random_state = random_state
        self.cores = cores

    def fit(self, X: list[str]):
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
                    raise ValueError("String metrics must return non-negative distances")
                return value

        count = min(self.n_embeddings, len(X))
        # fastmapy currently draws its pivot starts from Python's module-level
        # RNG. Restore it afterwards so fitting sphncs does not perturb callers.
        rng_state = random.getstate()
        try:
            if self.random_state is not None:
                random.seed(self.random_state)
            self.models_ = FastMap.fit_many(X, count=count, dim=1, distance=CallableDistance, cores=self.cores)
        finally:
            random.setstate(rng_state)
        coordinates = np.column_stack([np.asarray(model.transform(X), dtype=float).reshape(-1) for model in self.models_])
        if count < self.n_embeddings:
            coordinates = np.pad(coordinates, ((0, 0), (0, self.n_embeddings - count)))
        self.coordinates_ = coordinates
        return self

    def fit_transform(self, X: list[str]) -> np.ndarray:
        return self.fit(X).coordinates_.copy()

    def transform(self, X: list[str]) -> np.ndarray:
        if not hasattr(self, "models_"):
            raise RuntimeError("FastMap must be fitted before transform")
        if not self.models_:
            return np.zeros((len(X), self.n_embeddings), dtype=float)
        coordinates = np.column_stack([np.asarray(model.transform(X), dtype=float).reshape(-1) for model in self.models_])
        if coordinates.shape[1] < self.n_embeddings:
            coordinates = np.pad(coordinates, ((0, 0), (0, self.n_embeddings - coordinates.shape[1])))
        return coordinates
