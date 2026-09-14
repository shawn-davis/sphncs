"""Log-specific sphncs API with conservative structural defaults."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from typing import Any

from .distances import char_ngram_jaccard
from .estimator import SphncsClusterer


class LogSPHNCS(SphncsClusterer):
    """Cluster log lines with filtering and exact-template compression enabled.

    This log-focused API keeps raw records for traceability while clustering their
    filtered forms. Exact filtered duplicates are embedded once and expanded
    before KDE, so duplicate frequency still contributes to the density model.
    """

    def __init__(self, *, log_filters: str | list[str] | None = "all", metric=None, **kwargs: Any):
        if metric is None:
            metric = partial(char_ngram_jaccard, ngram_size=4)
        defaults = {
            "partitioning": True,
            "partitioning_feature": "length",
            "clustering_mode": "spectral_consensus",
            "n_embeddings": 10,
            "deduplicate": True,
            "fastmap_distance_cache": True,
        }
        defaults.update(kwargs)
        super().__init__(metric=metric, log_filters=log_filters, **defaults)

    def fit(self, X: Iterable[str], y=None):
        """Fit the generic engine while retaining log-specific traceability."""
        result = super().fit(X, y)
        self.raw_strings_ = self.objects_
        self.strings_ = self.raw_strings_
        self.processed_strings_ = self.transformed_objects_
        self.partition_strings_ = self.partition_objects_
        self.length_strings_ = self.partition_objects_
        self.length_model_ = self.partition_model_ if self.partitioning_feature == "length" else None
        self.length_boundaries_ = self.partition_boundaries_
        self.raw_representatives_ = [self.raw_strings_[index] for index in self.representative_indices_]
        return result

    def get_raw_string(self, index: int) -> str:
        self._require_fitted()
        return self.raw_strings_[index]
