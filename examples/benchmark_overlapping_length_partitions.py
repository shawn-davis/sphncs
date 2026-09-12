"""Prototype overlapping length partitions with exact-template reconciliation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from compare_log_cluster_fitness import fitness, pairwise_distances
from sphncs import SphncsClusterer
from sphncs.density import fit_density
from sphncs.distances import char_ngram_jaccard


def compact_labels(labels: np.ndarray) -> np.ndarray:
    """Merge labels connected by an exact filtered template and compact them."""
    parent = list(range(int(labels.max()) + 1))

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    return parent, find, union


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with aligned raw and processed logs")
    parser.add_argument("--overlap", type=float, default=12.0, help="characters included across each length boundary")
    parser.add_argument("--prominence-fraction", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.overlap < 0:
        parser.error("--overlap must be non-negative")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    raw, processed = source["raw"], source["processed"]
    lengths = np.asarray([len(value) for value in raw], dtype=float)
    length_model = fit_density(
        lengths,
        prominence_fraction=args.prominence_fraction,
        grid_points=512,
        min_samples=3,
    )
    primary = length_model.labels
    labels = np.empty(len(raw), dtype=int)
    offset = 0
    local_runs = []
    metric = partial(char_ngram_jaccard, ngram_size=4)
    started = perf_counter()
    for partition in np.unique(primary):
        core = np.flatnonzero(primary == partition)
        near_boundary = np.zeros(len(raw), dtype=bool)
        if partition > 0:
            near_boundary |= np.abs(lengths - length_model.boundaries[partition - 1]) <= args.overlap
        if partition < len(length_model.boundaries):
            near_boundary |= np.abs(lengths - length_model.boundaries[partition]) <= args.overlap
        training_indices = np.flatnonzero((primary == partition) | near_boundary)
        local = SphncsClusterer(
            metric=metric,
            clustering_mode="spectral_consensus",
            n_embeddings=10,
            consensus_n_clusters="max",
            extrema_prominence_fraction=args.prominence_fraction,
            deduplicate_strings=True,
            random_state=7,
        ).fit([processed[index] for index in training_indices])
        local_position = {int(index): position for position, index in enumerate(training_indices)}
        core_local_labels = local.labels_[[local_position[int(index)] for index in core]]
        local_to_global = {int(label): offset + position for position, label in enumerate(np.unique(core_local_labels))}
        labels[core] = [local_to_global[int(label)] for label in core_local_labels]
        offset += len(local_to_global)
        local_runs.append(
            {
                "partition": int(partition),
                "core_records": len(core),
                "training_records": len(training_indices),
                "core_clusters": len(local_to_global),
            }
        )

    parent, find, union = compact_labels(labels)
    template_labels: dict[str, set[int]] = defaultdict(set)
    for value, label in zip(processed, labels, strict=True):
        template_labels[value].add(int(label))
    exact_merges = 0
    for values in template_labels.values():
        first, *rest = values
        for label in rest:
            if find(first) != find(label):
                union(first, label)
                exact_merges += 1
    merged = np.asarray([find(int(label)) for label in labels])
    _, labels = np.unique(merged, return_inverse=True)
    fit_seconds = perf_counter() - started

    started = perf_counter()
    distances = pairwise_distances(processed, 4)
    output = {
        "method": "Overlapping raw-length partitions + exact-template reconciliation",
        "overlap_characters": args.overlap,
        "extrema_prominence_fraction": args.prominence_fraction,
        "fit_seconds": fit_seconds,
        "fitness_matrix_seconds": perf_counter() - started,
        "length_partitions": len(np.unique(primary)),
        "local_runs": local_runs,
        "exact_cross_partition_merges": exact_merges,
        "fitness": fitness([str(label) for label in labels], distances),
        "labels": [int(label) for label in labels],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "labels"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
