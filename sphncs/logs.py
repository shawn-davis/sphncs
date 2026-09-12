"""Log-specific sphncs API with conservative structural defaults."""

from __future__ import annotations

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
            "length_partitioning": True,
            "clustering_mode": "spectral_consensus",
            "n_embeddings": 10,
            "deduplicate_strings": True,
        }
        defaults.update(kwargs)
        super().__init__(metric=metric, log_filters=log_filters, **defaults)
