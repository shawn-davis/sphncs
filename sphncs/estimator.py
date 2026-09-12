"""The public sklearn-style sphncs estimator."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .density import DensityModel, fit_density
from .distances import StringMetric, resolve_metric
from .embedding import FastMapyEmbeddings
from .preprocessing import LogPreprocessor


@dataclass
class _Partition:
    interval_label: int
    indices: np.ndarray
    fastmap: FastMapyEmbeddings
    coordinates: np.ndarray
    density_models: list[DensityModel]
    labels: np.ndarray
    representatives: np.ndarray
    spectral_prototypes: np.ndarray | None = None


class SphncsClusterer:
    """Cluster strings using FastMap coordinates and one-dimensional KDE extrema.

    ``clustering_mode='single'`` is the base model: a one-dimensional FastMap
    embedding and density minima.  ``'spectral_consensus'`` embeds several FastMap
    dimensions, derives one KDE label per dimension, then clusters their shared
    label affinity graph.
    """

    def __init__(
        self,
        metric: str | StringMetric = "normalized_levenshtein",
        *,
        length_partitioning: bool = False,
        clustering_mode: str = "single",
        n_embeddings: int = 1,
        bandwidth: str | float = "ISJ",
        length_bandwidth: str | float | None = None,
        grid_points: int = 1024,
        length_grid_points: int = 512,
        extrema_prominence: float | None = None,
        extrema_prominence_fraction: float = 0.05,
        min_partition_size: int = 3,
        consensus_n_clusters: int | str = "auto",
        random_state: int | None = None,
        fastmap_iters: int = 3,
        fastmap_distance_cache: bool = True,
        log_filters: str | Iterable[str] | None = None,
        deduplicate_strings: bool = False,
        length_partitioning_before_filtering: bool = False,
    ):
        self.metric = metric
        self.length_partitioning = length_partitioning
        self.clustering_mode = clustering_mode
        self.n_embeddings = n_embeddings
        self.bandwidth = bandwidth
        self.length_bandwidth = length_bandwidth
        self.grid_points = grid_points
        self.length_grid_points = length_grid_points
        self.extrema_prominence = extrema_prominence
        self.extrema_prominence_fraction = extrema_prominence_fraction
        self.min_partition_size = min_partition_size
        self.consensus_n_clusters = consensus_n_clusters
        self.random_state = random_state
        self.fastmap_iters = fastmap_iters
        self.fastmap_distance_cache = fastmap_distance_cache
        self.log_filters = log_filters
        self.deduplicate_strings = deduplicate_strings
        self.length_partitioning_before_filtering = length_partitioning_before_filtering

    def fit(self, X: list[str], y=None):
        del y
        self._validate_parameters()
        self.raw_strings_ = self._validate_strings(X)
        # ``strings_`` remains an alias for backward compatibility. New code
        # should use the explicit raw/processed learned attributes instead.
        self.strings_ = self.raw_strings_
        self.preprocessor_ = LogPreprocessor(self.log_filters)
        self.processed_strings_ = self.preprocessor_.transform_many(self.raw_strings_)
        self.metric_ = resolve_metric(self.metric)
        self.length_strings_ = (
            self.raw_strings_ if self.length_partitioning_before_filtering else self.processed_strings_
        )
        lengths = np.asarray([len(value) for value in self.length_strings_], dtype=float)
        if self.length_partitioning:
            self.length_model_ = fit_density(
                lengths,
                bandwidth=self.length_bandwidth or self.bandwidth,
                grid_points=self.length_grid_points,
                prominence=self.extrema_prominence,
                prominence_fraction=self.extrema_prominence_fraction,
                min_samples=self.min_partition_size,
            )
            partition_labels = self.length_model_.labels
            self.length_boundaries_ = self.length_model_.boundaries
        else:
            self.length_model_ = None
            self.length_boundaries_ = np.array([], dtype=float)
            partition_labels = np.zeros(len(self.strings_), dtype=int)

        self.partitions_: list[_Partition] = []
        labels = np.empty(len(self.strings_), dtype=int)
        offset = 0
        for partition_label in np.unique(partition_labels):
            indices = np.flatnonzero(partition_labels == partition_label)
            partition = self._fit_partition(int(partition_label), indices)
            labels[indices] = partition.labels + offset
            offset += int(partition.labels.max()) + 1
            self.partitions_.append(partition)

        self.labels_ = labels
        self.n_clusters_ = offset
        self.embedding_ = self._training_embedding()
        self.representative_indices_ = self._global_representatives()
        self.representatives_ = [self.processed_strings_[index] for index in self.representative_indices_]
        self.raw_representatives_ = [self.raw_strings_[index] for index in self.representative_indices_]
        return self

    def fit_predict(self, X: list[str], y=None) -> np.ndarray:
        return self.fit(X, y).labels_.copy()

    def fit_transform(self, X: list[str], y=None) -> np.ndarray:
        return self.fit(X, y).embedding_.copy()

    def get_raw_string(self, index: int) -> str:
        """Return the original string aligned with a processed training value."""
        self._require_fitted()
        return self.raw_strings_[index]

    def transform(self, X: list[str]) -> np.ndarray:
        self._require_fitted()
        raw_strings = self._validate_strings(X)
        strings = self.preprocessor_.transform_many(raw_strings)
        result = np.zeros((len(strings), self._components), dtype=float)
        for label, indices in self._route(strings, raw_strings).items():
            result[indices] = self.partitions_[label].fastmap.transform([strings[i] for i in indices])
        return result

    def predict(self, X: list[str]) -> np.ndarray:
        self._require_fitted()
        raw_strings = self._validate_strings(X)
        strings = self.preprocessor_.transform_many(raw_strings)
        labels = np.empty(len(strings), dtype=int)
        offset = self._partition_offsets()
        for partition_number, indices in self._route(strings, raw_strings).items():
            partition = self.partitions_[partition_number]
            values = partition.fastmap.transform([strings[i] for i in indices])
            base_labels = np.column_stack([model.predict(values[:, dimension]) for dimension, model in enumerate(partition.density_models)])
            if self.clustering_mode == "single":
                local = base_labels[:, 0]
            else:
                local = self._predict_spectral(base_labels, partition.spectral_prototypes_)
            labels[indices] = local + offset[partition_number]
        return labels

    def _fit_partition(self, interval_label: int, indices: np.ndarray) -> _Partition:
        strings = [self.processed_strings_[index] for index in indices]
        if self.deduplicate_strings:
            unique_strings = list(dict.fromkeys(strings))
            unique_indices = {value: index for index, value in enumerate(unique_strings)}
            inverse = np.asarray([unique_indices[value] for value in strings], dtype=int)
        else:
            unique_strings = strings
            inverse = np.arange(len(strings))
        components = 1 if self.clustering_mode == "single" else self.n_embeddings
        fastmap = FastMapyEmbeddings(
            components,
            self.metric_,
            random_state=self.random_state,
            iters=self.fastmap_iters,
            cache_distances=self.fastmap_distance_cache,
        )
        coordinates = fastmap.fit_transform(unique_strings)[inverse]
        density_models = [
            fit_density(
                coordinates[:, dimension], bandwidth=self.bandwidth, grid_points=self.grid_points,
                prominence=self.extrema_prominence, min_samples=self.min_partition_size,
                prominence_fraction=self.extrema_prominence_fraction,
            )
            for dimension in range(components)
        ]
        base_labels = np.column_stack([model.labels for model in density_models])
        if self.clustering_mode == "single":
            labels = base_labels[:, 0]
            representatives = indices[density_models[0].representative_indices]
            prototypes = None
        else:
            labels = self._spectral_labels(base_labels)
            prototypes = self._spectral_prototypes(base_labels, labels)
            representatives = self._select_consensus_representatives(indices, labels, density_models)
        return _Partition(interval_label, indices, fastmap, coordinates, density_models, labels, representatives, prototypes)

    def _spectral_labels(self, base_labels: np.ndarray) -> np.ndarray:
        affinity = self._coassociation_graph(base_labels)
        n_samples = affinity.shape[0]
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=int)
        cluster_count = self._resolve_consensus_count(affinity, base_labels)
        if cluster_count == 1:
            return np.zeros(n_samples, dtype=int)
        component_count, component_labels = connected_components(affinity, directed=False)
        if cluster_count == component_count:
            # Spectral embedding warns (correctly) on a disconnected graph. When
            # its requested count is exactly the components, the components are
            # the unambiguous spectral-consensus result already.
            return self._coalesce_equivalent_labels(base_labels, component_labels)
        from sklearn.cluster import SpectralClustering

        model = SpectralClustering(
            n_clusters=cluster_count, affinity="precomputed", assign_labels="kmeans",
            random_state=self.random_state, n_init=10, n_jobs=1,
        )
        return self._coalesce_equivalent_labels(base_labels, model.fit_predict(affinity))

    @staticmethod
    def _coassociation_graph(base_labels: np.ndarray) -> sparse.csr_matrix:
        """Sparse, inverse-base-cluster-size weighted shared-label graph."""
        n_samples, dimensions = base_labels.shape
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for dimension in range(dimensions):
            for label in np.unique(base_labels[:, dimension]):
                members = np.flatnonzero(base_labels[:, dimension] == label)
                weight = 1.0 / len(members)
                rows.append(np.repeat(members, len(members)))
                cols.append(np.tile(members, len(members)))
                data.append(np.full(len(members) ** 2, weight))
        graph = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_samples, n_samples))
        return graph.tocsr()

    def _resolve_consensus_count(self, affinity: sparse.csr_matrix, base_labels: np.ndarray) -> int:
        n_samples = affinity.shape[0]
        if isinstance(self.consensus_n_clusters, int):
            if not 1 <= self.consensus_n_clusters <= n_samples:
                raise ValueError("consensus_n_clusters must be between 1 and the partition size")
            return self.consensus_n_clusters
        components, _ = connected_components(affinity, directed=False)
        if components > 1:
            return components
        # The one-dimensional KDE fits have already observed useful cluster
        # scales. Reduce their counts instead of treating the number of
        # embeddings as a requested number of consensus clusters.
        observed_counts = np.asarray(
            [len(np.unique(base_labels[:, dimension])) for dimension in range(base_labels.shape[1])]
        )
        strategy = "median" if self.consensus_n_clusters == "auto" else self.consensus_n_clusters
        reducers = {"min": np.min, "mean": np.mean, "median": np.median, "max": np.max}
        return int(np.clip(np.rint(reducers[strategy](observed_counts)), 1, n_samples))

    @staticmethod
    def _coalesce_equivalent_labels(base_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Keep nodes with identical co-association rows in one consensus cluster."""
        _, signatures = np.unique(base_labels, axis=0, return_inverse=True)
        result = labels.copy()
        for signature in np.unique(signatures):
            members = np.flatnonzero(signatures == signature)
            values, counts = np.unique(result[members], return_counts=True)
            # Deterministic tie-break: the smallest existing label wins.
            result[members] = values[np.flatnonzero(counts == counts.max())[0]]
        _, compact = np.unique(result, return_inverse=True)
        return compact

    @staticmethod
    def _spectral_prototypes(base_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
        prototype = np.zeros((labels.max() + 1, base_labels.shape[1], int(base_labels.max()) + 1), dtype=float)
        for cluster in range(prototype.shape[0]):
            members = base_labels[labels == cluster]
            for dimension in range(base_labels.shape[1]):
                values, counts = np.unique(members[:, dimension], return_counts=True)
                prototype[cluster, dimension, values] = counts / len(members)
        return prototype

    @staticmethod
    def _predict_spectral(base_labels: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(base_labels), len(prototypes)))
        for dimension in range(base_labels.shape[1]):
            scores += prototypes[:, dimension, base_labels[:, dimension]].T
        return scores.argmax(axis=1)

    def _select_consensus_representatives(self, indices: np.ndarray, labels: np.ndarray, models: list[DensityModel]) -> np.ndarray:
        candidates = np.unique(np.concatenate([model.representative_indices for model in models]))
        chosen: list[int] = []
        for cluster in range(labels.max() + 1):
            members = np.flatnonzero(labels == cluster)
            local_candidates = np.intersect1d(candidates, members)
            if not len(local_candidates):
                local_candidates = members
            scores = [np.mean([self.metric_(self.processed_strings_[indices[candidate]], self.processed_strings_[indices[member]]) for member in members]) for candidate in local_candidates]
            chosen.append(int(indices[local_candidates[int(np.argmin(scores))]]))
        return np.asarray(chosen, dtype=int)

    def _route(self, strings: list[str], raw_strings: list[str] | None = None) -> dict[int, np.ndarray]:
        if not self.length_partitioning:
            return {0: np.arange(len(strings))}
        if self.length_partitioning_before_filtering:
            if raw_strings is None:
                raise ValueError("raw_strings are required for raw-length partition routing")
            strings = raw_strings
        partition = np.searchsorted(self.length_boundaries_, [len(value) for value in strings], side="right")
        return {int(label): np.flatnonzero(partition == label) for label in np.unique(partition)}

    def _training_embedding(self) -> np.ndarray:
        result = np.zeros((len(self.strings_), self._components), dtype=float)
        for partition in self.partitions_:
            result[partition.indices] = partition.coordinates
        return result

    def _global_representatives(self) -> np.ndarray:
        representatives: list[int] = []
        for partition in self.partitions_:
            representatives.extend(partition.representatives.tolist())
        return np.asarray(representatives, dtype=int)

    def _partition_offsets(self) -> list[int]:
        offsets: list[int] = []
        offset = 0
        for partition in self.partitions_:
            offsets.append(offset)
            offset += int(partition.labels.max()) + 1
        return offsets

    @property
    def _components(self) -> int:
        return 1 if self.clustering_mode == "single" else self.n_embeddings

    def _validate_parameters(self) -> None:
        if self.clustering_mode not in {"single", "spectral_consensus"}:
            raise ValueError("clustering_mode must be 'single' or 'spectral_consensus'")
        if self.clustering_mode == "spectral_consensus" and self.n_embeddings < 2:
            raise ValueError("spectral_consensus requires n_embeddings >= 2")
        if isinstance(self.consensus_n_clusters, bool) or (
            not isinstance(self.consensus_n_clusters, (int, str))
        ):
            raise TypeError("consensus_n_clusters must be an integer or one of 'auto', 'min', 'mean', 'median', 'max'")
        if isinstance(self.consensus_n_clusters, str) and self.consensus_n_clusters not in {
            "auto", "min", "mean", "median", "max"
        }:
            raise ValueError("consensus_n_clusters must be an integer or one of 'auto', 'min', 'mean', 'median', 'max'")
        if self.grid_points < 2 or self.length_grid_points < 2:
            raise ValueError("grid point counts must be at least 2")
        if not isinstance(self.extrema_prominence_fraction, (int, float)) or isinstance(self.extrema_prominence_fraction, bool):
            raise TypeError("extrema_prominence_fraction must be a number")
        if not 0 < self.extrema_prominence_fraction <= 1:
            raise ValueError("extrema_prominence_fraction must be in (0, 1]")
        if not isinstance(self.fastmap_iters, int) or isinstance(self.fastmap_iters, bool) or self.fastmap_iters < 1:
            raise ValueError("fastmap_iters must be a positive integer")
        if not isinstance(self.fastmap_distance_cache, bool):
            raise TypeError("fastmap_distance_cache must be a boolean")
        if not isinstance(self.deduplicate_strings, bool):
            raise TypeError("deduplicate_strings must be a boolean")
        if not isinstance(self.length_partitioning_before_filtering, bool):
            raise TypeError("length_partitioning_before_filtering must be a boolean")

    @staticmethod
    def _validate_strings(X: list[str]) -> list[str]:
        values = list(X)
        if not values:
            raise ValueError("X must contain at least one string")
        if any(not isinstance(value, str) for value in values):
            raise TypeError("X must contain only strings")
        return values

    def _require_fitted(self) -> None:
        if not hasattr(self, "partitions_"):
            raise RuntimeError("SphncsClusterer must be fitted before calling transform or predict")
