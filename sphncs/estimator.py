"""The domain-neutral sklearn-style SPHNCS estimator."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .density import DensityModel, fit_density
from .distances import Metric, resolve_metric
from .embedding import FastMapyEmbeddings
from .partitioning import PartitioningFeature, partition_feature_values

ObjectTransformer = Callable[[object], object]


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
    """Similarity-Preserving Hierarchical Nonparametric Clustering System.

    SPHNCS clusters arbitrary objects for which ``metric(left, right)`` returns
    a non-negative distance. FastMap produces one or more on-demand 1-D
    projections, KDE minima form base clusters, and optional spectral consensus
    combines independent projections. A scalar ``partitioning_feature`` can
    optionally form a first, hard-routing hierarchy.
    """

    def __init__(
        self,
        metric: str | Metric | None = None,
        *,
        object_transformer: ObjectTransformer | None = None,
        partitioning: bool = False,
        partitioning_feature: str | PartitioningFeature | None = None,
        partitioning_before_transform: bool = False,
        clustering_mode: str = "single",
        n_embeddings: int = 1,
        bandwidth: str | float = "ISJ",
        partitioning_bandwidth: str | float | None = None,
        grid_points: int = 1024,
        partitioning_grid_points: int = 512,
        extrema_prominence: float | None = None,
        extrema_prominence_fraction: float = 0.05,
        min_partition_size: int = 3,
        consensus_n_clusters: int | str = "auto",
        random_state: int | None = None,
        fastmap_iters: int = 3,
        fastmap_distance_cache: bool = False,
        deduplicate: bool = False,
        # Compatibility aliases retained through the pre-1.0 API.
        length_partitioning: bool | None = None,
        length_bandwidth: str | float | None = None,
        length_grid_points: int | None = None,
        length_partitioning_before_filtering: bool | None = None,
        deduplicate_strings: bool | None = None,
        log_filters: str | Iterable[str] | None = None,
    ):
        if length_partitioning is not None:
            partitioning = length_partitioning
            if partitioning_feature is None:
                partitioning_feature = "length"
        if length_bandwidth is not None:
            partitioning_bandwidth = length_bandwidth
        if length_grid_points is not None:
            partitioning_grid_points = length_grid_points
        if length_partitioning_before_filtering is not None:
            partitioning_before_transform = length_partitioning_before_filtering
        if deduplicate_strings is not None:
            deduplicate = deduplicate_strings
        if log_filters is not None:
            from .preprocessing import LogPreprocessor
            if object_transformer is not None:
                raise ValueError("pass either object_transformer or log_filters, not both")
            self._legacy_log_preprocessor = LogPreprocessor(log_filters)
            object_transformer = self._legacy_log_preprocessor.transform
        else:
            self._legacy_log_preprocessor = None

        self.metric = metric
        self.object_transformer = object_transformer
        self.partitioning = partitioning
        self.partitioning_feature = partitioning_feature
        self.partitioning_before_transform = partitioning_before_transform
        self.clustering_mode = clustering_mode
        self.n_embeddings = n_embeddings
        self.bandwidth = bandwidth
        self.partitioning_bandwidth = partitioning_bandwidth
        self.grid_points = grid_points
        self.partitioning_grid_points = partitioning_grid_points
        self.extrema_prominence = extrema_prominence
        self.extrema_prominence_fraction = extrema_prominence_fraction
        self.min_partition_size = min_partition_size
        self.consensus_n_clusters = consensus_n_clusters
        self.random_state = random_state
        self.fastmap_iters = fastmap_iters
        self.fastmap_distance_cache = fastmap_distance_cache
        self.deduplicate = deduplicate
        self.log_filters = log_filters

        # Backward-compatible public constructor attributes.
        self.length_partitioning = partitioning
        self.length_bandwidth = partitioning_bandwidth
        self.length_grid_points = partitioning_grid_points
        self.length_partitioning_before_filtering = partitioning_before_transform
        self.deduplicate_strings = deduplicate

    def fit(self, X: Iterable[object], y=None):
        del y
        self._validate_parameters()
        self.objects_ = self._validate_objects(X)
        self.metric_ = self._resolve_metric()
        self.transformed_objects_ = self._transform_many(self.objects_)
        if self._legacy_log_preprocessor is not None:
            self.preprocessor_ = self._legacy_log_preprocessor
        self.partition_objects_ = self.objects_ if self.partitioning_before_transform else self.transformed_objects_
        if self.partitioning:
            self.partition_values_ = partition_feature_values(self.partition_objects_, self.partitioning_feature)
            self.partition_model_ = fit_density(
                self.partition_values_, bandwidth=self.partitioning_bandwidth or self.bandwidth,
                grid_points=self.partitioning_grid_points, prominence=self.extrema_prominence,
                prominence_fraction=self.extrema_prominence_fraction, min_samples=self.min_partition_size,
            )
            partition_labels = self.partition_model_.labels
            self.partition_boundaries_ = self.partition_model_.boundaries
        else:
            self.partition_values_ = np.array([], dtype=float)
            self.partition_model_ = None
            self.partition_boundaries_ = np.array([], dtype=float)
            partition_labels = np.zeros(len(self.objects_), dtype=int)

        self.partitions_: list[_Partition] = []
        labels = np.empty(len(self.objects_), dtype=int)
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
        self.representatives_ = [self.transformed_objects_[index] for index in self.representative_indices_]
        if self.log_filters is not None:
            # Compatibility surface for callers migrating to LogSPHNCS.
            self.raw_strings_ = self.objects_
            self.strings_ = self.raw_strings_
            self.processed_strings_ = self.transformed_objects_
            self.partition_strings_ = self.partition_objects_
            self.length_strings_ = self.partition_objects_
            self.length_model_ = self.partition_model_ if self.partitioning_feature == "length" else None
            self.length_boundaries_ = self.partition_boundaries_
            self.raw_representatives_ = [self.raw_strings_[index] for index in self.representative_indices_]
        return self

    def fit_predict(self, X: Iterable[object], y=None) -> np.ndarray:
        return self.fit(X, y).labels_.copy()

    def fit_transform(self, X: Iterable[object], y=None) -> np.ndarray:
        return self.fit(X, y).embedding_.copy()

    def get_object(self, index: int) -> object:
        """Return an original training object by its position-aligned index."""
        self._require_fitted()
        return self.objects_[index]

    def get_raw_string(self, index: int) -> str:
        """Deprecated log compatibility alias; use ``get_object`` instead."""
        self._require_fitted()
        value = self.objects_[index]
        if not isinstance(value, str):
            raise TypeError("get_raw_string is available only for string inputs")
        return value

    def transform(self, X: Iterable[object]) -> np.ndarray:
        self._require_fitted()
        objects = self._validate_objects(X)
        transformed = self._transform_many(objects)
        result = np.zeros((len(objects), self._components), dtype=float)
        for label, indices in self._route(transformed, objects).items():
            result[indices] = self.partitions_[label].fastmap.transform([transformed[index] for index in indices])
        return result

    def predict(self, X: Iterable[object]) -> np.ndarray:
        self._require_fitted()
        objects = self._validate_objects(X)
        transformed = self._transform_many(objects)
        labels = np.empty(len(objects), dtype=int)
        offset = self._partition_offsets()
        for partition_number, indices in self._route(transformed, objects).items():
            partition = self.partitions_[partition_number]
            values = partition.fastmap.transform([transformed[index] for index in indices])
            base_labels = np.column_stack([model.predict(values[:, dimension]) for dimension, model in enumerate(partition.density_models)])
            local = base_labels[:, 0] if self.clustering_mode == "single" else self._predict_spectral(base_labels, partition.spectral_prototypes_)
            labels[indices] = local + offset[partition_number]
        return labels

    def _fit_partition(self, interval_label: int, indices: np.ndarray) -> _Partition:
        objects = [self.transformed_objects_[index] for index in indices]
        unique_objects, inverse = self._deduplicate(objects)
        components = 1 if self.clustering_mode == "single" else self.n_embeddings
        fastmap = FastMapyEmbeddings(
            components, self.metric_, random_state=self.random_state, iters=self.fastmap_iters,
            cache_distances=self.fastmap_distance_cache,
        )
        coordinates = fastmap.fit_transform(unique_objects)[inverse]
        density_models = [
            fit_density(coordinates[:, dimension], bandwidth=self.bandwidth, grid_points=self.grid_points,
                        prominence=self.extrema_prominence, min_samples=self.min_partition_size,
                        prominence_fraction=self.extrema_prominence_fraction)
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

    def _deduplicate(self, objects: list[object]) -> tuple[list[object], np.ndarray]:
        if not self.deduplicate:
            return objects, np.arange(len(objects))
        unique: list[object] = []
        inverse: list[int] = []
        for value in objects:
            try:
                index = next(index for index, known in enumerate(unique) if value == known)
            except StopIteration:
                index = len(unique)
                unique.append(value)
            inverse.append(index)
        return unique, np.asarray(inverse, dtype=int)

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
            return self._coalesce_equivalent_labels(base_labels, component_labels)
        from sklearn.cluster import SpectralClustering
        model = SpectralClustering(n_clusters=cluster_count, affinity="precomputed", assign_labels="kmeans", random_state=self.random_state, n_init=10, n_jobs=1)
        return self._coalesce_equivalent_labels(base_labels, model.fit_predict(affinity))

    @staticmethod
    def _coassociation_graph(base_labels: np.ndarray) -> sparse.csr_matrix:
        n_samples, dimensions = base_labels.shape
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for dimension in range(dimensions):
            for label in np.unique(base_labels[:, dimension]):
                members = np.flatnonzero(base_labels[:, dimension] == label)
                rows.append(np.repeat(members, len(members)))
                cols.append(np.tile(members, len(members)))
                data.append(np.full(len(members) ** 2, 1.0 / len(members)))
        return sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_samples, n_samples)).tocsr()

    def _resolve_consensus_count(self, affinity: sparse.csr_matrix, base_labels: np.ndarray) -> int:
        n_samples = affinity.shape[0]
        if isinstance(self.consensus_n_clusters, int):
            if not 1 <= self.consensus_n_clusters <= n_samples:
                raise ValueError("consensus_n_clusters must be between 1 and the partition size")
            return self.consensus_n_clusters
        components, _ = connected_components(affinity, directed=False)
        if components > 1:
            return components
        observed = np.asarray([len(np.unique(base_labels[:, dimension])) for dimension in range(base_labels.shape[1])])
        reducers = {"auto": np.median, "min": np.min, "mean": np.mean, "median": np.median, "max": np.max}
        return int(np.clip(np.rint(reducers[self.consensus_n_clusters](observed)), 1, n_samples))

    @staticmethod
    def _coalesce_equivalent_labels(base_labels: np.ndarray, labels: np.ndarray) -> np.ndarray:
        _, signatures = np.unique(base_labels, axis=0, return_inverse=True)
        result = labels.copy()
        for signature in np.unique(signatures):
            members = np.flatnonzero(signatures == signature)
            values, counts = np.unique(result[members], return_counts=True)
            result[members] = values[np.flatnonzero(counts == counts.max())[0]]
        return np.unique(result, return_inverse=True)[1]

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
            scores = [np.mean([self.metric_(self.transformed_objects_[indices[candidate]], self.transformed_objects_[indices[member]]) for member in members]) for candidate in local_candidates]
            chosen.append(int(indices[local_candidates[int(np.argmin(scores))]]))
        return np.asarray(chosen, dtype=int)

    def _route(self, transformed: list[object], original: list[object]) -> dict[int, np.ndarray]:
        if not self.partitioning:
            return {0: np.arange(len(transformed))}
        values = partition_feature_values(original if self.partitioning_before_transform else transformed, self.partitioning_feature)
        partition = np.searchsorted(self.partition_boundaries_, values, side="right")
        return {int(label): np.flatnonzero(partition == label) for label in np.unique(partition)}

    def _training_embedding(self) -> np.ndarray:
        result = np.zeros((len(self.objects_), self._components), dtype=float)
        for partition in self.partitions_:
            result[partition.indices] = partition.coordinates
        return result

    def _global_representatives(self) -> np.ndarray:
        return np.asarray([index for partition in self.partitions_ for index in partition.representatives], dtype=int)

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

    def _resolve_metric(self) -> Metric:
        if self.metric is None:
            raise ValueError("metric is required; pass a distance callable or a named built-in string metric")
        return resolve_metric(self.metric)

    def _transform_many(self, objects: list[object]) -> list[object]:
        if self.object_transformer is None:
            return list(objects)
        return [self.object_transformer(value) for value in objects]

    def _validate_parameters(self) -> None:
        if self.clustering_mode not in {"single", "spectral_consensus"}:
            raise ValueError("clustering_mode must be 'single' or 'spectral_consensus'")
        if self.clustering_mode == "spectral_consensus" and self.n_embeddings < 2:
            raise ValueError("spectral_consensus requires n_embeddings >= 2")
        if self.partitioning and self.partitioning_feature is None:
            raise ValueError("partitioning_feature is required when partitioning=True")
        if self.partitioning_feature is not None and not (callable(self.partitioning_feature) or self.partitioning_feature in {"length", "entropy", "normalized_entropy"}):
            raise ValueError("partitioning_feature must be a callable, 'length', 'entropy', or 'normalized_entropy'")
        if self.grid_points < 2 or self.partitioning_grid_points < 2:
            raise ValueError("grid point counts must be at least 2")
        if not isinstance(self.extrema_prominence_fraction, (int, float)) or isinstance(self.extrema_prominence_fraction, bool) or not 0 < self.extrema_prominence_fraction <= 1:
            raise ValueError("extrema_prominence_fraction must be in (0, 1]")
        if not isinstance(self.fastmap_iters, int) or isinstance(self.fastmap_iters, bool) or self.fastmap_iters < 1:
            raise ValueError("fastmap_iters must be a positive integer")
        if not isinstance(self.fastmap_distance_cache, bool) or not isinstance(self.deduplicate, bool):
            raise TypeError("fastmap_distance_cache and deduplicate must be booleans")

    @staticmethod
    def _validate_objects(X: Iterable[object]) -> list[object]:
        values = list(X)
        if not values:
            raise ValueError("X must contain at least one object")
        return values

    def _require_fitted(self) -> None:
        if not hasattr(self, "partitions_"):
            raise RuntimeError("SphncsClusterer must be fitted before calling transform or predict")
