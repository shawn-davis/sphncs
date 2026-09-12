"""Benchmark spectral-consensus sphncs on the first Windows LogHub records."""

from __future__ import annotations

import json
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

import sphncs.estimator as estimator_module
from sphncs import SphncsClusterer
from sphncs.distances import char_ngram_jaccard
from sphncs.embedding import FastMapyEmbeddings
from sphncs.preprocessing import LogPreprocessor

try:  # Supports both ``python examples/...`` and module-style invocation.
    from .windows_jaccard_4gram import DEFAULT_URL, read_first_log_records
except ImportError:  # pragma: no cover - direct-script path
    from windows_jaccard_4gram import DEFAULT_URL, read_first_log_records


EMBEDDING_COUNTS = (2, 5, 10, 15, 20)


@contextmanager
def fit_timers():
    """Measure the major stages inside one estimator fit without changing it."""
    timings = {"fastmap_seconds": 0.0, "density_seconds": 0.0, "spectral_seconds": 0.0}
    original_embedding = FastMapyEmbeddings.fit_transform
    original_density = estimator_module.fit_density
    original_spectral = SphncsClusterer._spectral_labels

    def timed_embedding(self, *args, **kwargs):
        started = perf_counter()
        try:
            return original_embedding(self, *args, **kwargs)
        finally:
            timings["fastmap_seconds"] += perf_counter() - started

    def timed_density(*args, **kwargs):
        started = perf_counter()
        try:
            return original_density(*args, **kwargs)
        finally:
            timings["density_seconds"] += perf_counter() - started

    def timed_spectral(self, *args, **kwargs):
        started = perf_counter()
        try:
            return original_spectral(self, *args, **kwargs)
        finally:
            timings["spectral_seconds"] += perf_counter() - started

    FastMapyEmbeddings.fit_transform = timed_embedding
    estimator_module.fit_density = timed_density
    SphncsClusterer._spectral_labels = timed_spectral
    try:
        yield timings
    finally:
        FastMapyEmbeddings.fit_transform = original_embedding
        estimator_module.fit_density = original_density
        SphncsClusterer._spectral_labels = original_spectral


def cluster_tightness(model: SphncsClusterer, values: list[str], metric) -> tuple[dict, list[dict]]:
    """Return overall and per-cluster representative-distance summaries."""
    all_distances: list[float] = []
    clusters: list[dict] = []
    for label, representative_index in enumerate(model.representative_indices_):
        indices = np.flatnonzero(model.labels_ == label)
        distances = np.sort(np.asarray([metric(values[index], values[representative_index]) for index in indices]))
        all_distances.extend(distances.tolist())
        clusters.append(
            {
                "label": label,
                "size": int(len(indices)),
                "representative_index": int(representative_index),
                "representative": values[representative_index],
                "mean_distance": float(distances.mean()),
                "median_distance": float(np.median(distances)),
                "p95_distance": float(np.quantile(distances, 0.95, method="nearest")),
                "max_distance": float(distances.max()),
            }
        )
    distances = np.sort(np.asarray(all_distances))
    return (
        {
            "mean_distance": float(distances.mean()),
            "median_distance": float(np.median(distances)),
            "p95_distance": float(np.quantile(distances, 0.95, method="nearest")),
            "max_distance": float(distances.max()),
            "within_0_25": float(np.mean(distances <= 0.25)),
        },
        clusters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-counts", type=int, nargs="+", default=EMBEDDING_COUNTS)
    parser.add_argument("--fastmap-iters", type=int, default=3)
    parser.add_argument("--no-fastmap-distance-cache", action="store_true")
    parser.add_argument(
        "--log-filter",
        dest="log_filters",
        action="append",
        default=[],
        help="Preprocessing filter to apply; repeat for each selection, or use --log-filter all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/windows-jaccard-4gram-length-spectral-benchmark.json"),
    )
    parser.add_argument("--quiet", action="store_true", help="write the JSON artifact without printing it")
    args = parser.parse_args()
    if any(count < 2 for count in args.embedding_counts):
        parser.error("--embedding-counts values must be at least 2")
    started = perf_counter()
    records, archive_member = read_first_log_records(DEFAULT_URL, 1000)
    data_load_seconds = perf_counter() - started
    metric = partial(char_ngram_jaccard, ngram_size=4)
    results: list[dict] = []

    for n_embeddings in args.embedding_counts:
        model = SphncsClusterer(
            metric=metric,
            length_partitioning=True,
            clustering_mode="spectral_consensus",
            n_embeddings=n_embeddings,
            random_state=7,
            fastmap_iters=args.fastmap_iters,
            fastmap_distance_cache=not args.no_fastmap_distance_cache,
            log_filters=args.log_filters,
        )
        with fit_timers() as timings:
            started = perf_counter()
            model.fit(records)
            fit_seconds = perf_counter() - started
        raw_tightness, raw_clusters = cluster_tightness(model, model.raw_strings_, metric)
        processed_tightness, processed_clusters = cluster_tightness(model, model.processed_strings_, metric)
        timings["other_seconds"] = max(
            0.0, fit_seconds - sum(timings.values())
        )
        results.append(
            {
                "n_embeddings": n_embeddings,
                "fit_seconds": fit_seconds,
                **timings,
                "n_length_partitions": len(model.partitions_),
                "length_boundaries": model.length_boundaries_.tolist(),
                "n_clusters": int(model.n_clusters_),
                "tightness": raw_tightness,
                "processed_tightness": processed_tightness,
                "clusters": raw_clusters,
                "processed_clusters": processed_clusters,
            }
        )

    output = {
        "source_url": DEFAULT_URL,
        "archive_member": archive_member,
        "records": len(records),
        "data_load_seconds": data_load_seconds,
        "metric": "char_ngram_jaccard",
        "ngram_size": 4,
        "length_partitioning": True,
        "clustering_mode": "spectral_consensus",
        "random_state": 7,
        "fastmap_iters": args.fastmap_iters,
        "fastmap_distance_cache": not args.no_fastmap_distance_cache,
        "log_filters": list(LogPreprocessor(args.log_filters).filters),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
