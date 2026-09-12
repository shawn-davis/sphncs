"""Compare log-clustering methods with distance-based fitness measures.

Every measure is calculated in the filtered-string space using multiset
four-character-shingle Jaccard distance.  This makes the comparison fair to
methods that retain raw input only for traceability.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import silhouette_score

from sphncs import LogSPHNCS, SphncsClusterer
from sphncs.distances import char_ngram_jaccard


def pairwise_distances(values: list[str], ngram_size: int) -> np.ndarray:
    """Return the exact symmetric distance matrix for the evaluation metric.

    Filtering leaves many exact templates.  Calculate each unique-template pair
    once, then expand it to record level so duplicates retain their true weight.
    """
    metric = partial(char_ngram_jaccard, ngram_size=ngram_size)
    unique_values = list(dict.fromkeys(values))
    lookup = {value: index for index, value in enumerate(unique_values)}
    inverse = np.asarray([lookup[value] for value in values])
    unique_distances = np.zeros((len(unique_values), len(unique_values)), dtype=np.float32)
    for right in range(1, len(unique_values)):
        for left in range(right):
            unique_distances[left, right] = unique_distances[right, left] = metric(unique_values[left], unique_values[right])
    return unique_distances[np.ix_(inverse, inverse)]


def fitness(labels: list[str], distances: np.ndarray) -> dict[str, float | int]:
    """Calculate cohesion, separation, and membership-distribution measures."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    members = list(groups.values())
    sizes = np.asarray([len(indices) for indices in members], dtype=float)
    probabilities = sizes / sizes.sum()

    medoids: list[int] = []
    scatter: list[float] = []
    representative_distances: list[np.ndarray] = []
    for indices in members:
        block = distances[np.ix_(indices, indices)]
        medoid_at = int(np.argmin(block.mean(axis=1)))
        medoids.append(indices[medoid_at])
        scatter.append(float(block[medoid_at].mean()))
        representative_distances.append(block[medoid_at])
    medoid_distances = distances[np.ix_(medoids, medoids)]
    scatter_array = np.asarray(scatter)
    all_representative_distances = np.concatenate(representative_distances)
    numerator = scatter_array[:, None] + scatter_array[None, :]
    non_diagonal = ~np.eye(len(medoid_distances), dtype=bool)
    zero_medoid_pairs = int(np.count_nonzero((medoid_distances == 0) & non_diagonal) // 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = numerator / medoid_distances
    # Distinct clusters with identical medoids have no separating distance. The
    # Davies--Bouldin criterion is unbounded in this case, including 0 / 0.
    ratios[(medoid_distances == 0) & non_diagonal] = np.inf
    np.fill_diagonal(ratios, -np.inf)
    db_value = float(np.mean(np.max(ratios, axis=1)))

    return {
        "n_clusters": len(members),
        "singleton_clusters": int(np.sum(sizes == 1)),
        "silhouette": float(silhouette_score(distances, labels, metric="precomputed")),
        # A Davies--Bouldin analogue using string medoids rather than vector centroids.
        "medoid_davies_bouldin": None if np.isinf(db_value) else db_value,
        "zero_medoid_separation_pairs": zero_medoid_pairs,
        "mean_medoid_scatter": float(np.mean(scatter_array)),
        "median_medoid_distance": float(np.median(all_representative_distances)),
        "p95_medoid_distance": float(np.quantile(all_representative_distances, 0.95, method="nearest")),
        "within_0_25": float(np.mean(all_representative_distances <= 0.25)),
        "largest_cluster_share": float(sizes.max() / sizes.sum()),
        "effective_cluster_count": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "size_coefficient_of_variation": float(sizes.std() / sizes.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with raw and processed log arrays")
    parser.add_argument("baselines", type=Path, help="JSON output from benchmark_template_parsers.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-length-partitioning", action="store_true", help="partition on raw lengths before filtering")
    parser.add_argument("--no-length-partitioning", action="store_true", help="disable first-stage length partitioning")
    parser.add_argument("--ngram-size", type=int, default=4, help="character-shingle size for LogSPHNCS and evaluation")
    parser.add_argument(
        "--consensus-k",
        choices=("min", "mean", "median", "max"),
        help="reduce per-embedding KDE cluster counts to the spectral target",
    )
    parser.add_argument(
        "--prominence-fraction",
        type=float,
        default=0.05,
        help="fraction of each KDE's density range required for an extrema",
    )
    parser.add_argument(
        "--ensemble-ngrams",
        type=int,
        nargs="+",
        help="ensemble independently fitted LogSPHNCS runs with these n-gram sizes",
    )
    args = parser.parse_args()
    if args.raw_length_partitioning and args.no_length_partitioning:
        parser.error("--raw-length-partitioning requires length partitioning")
    if args.ngram_size < 1:
        parser.error("--ngram-size must be positive")
    if args.ensemble_ngrams and any(size < 1 for size in args.ensemble_ngrams):
        parser.error("--ensemble-ngrams values must be positive")
    if not 0 < args.prominence_fraction <= 1:
        parser.error("--prominence-fraction must be in (0, 1]")

    inputs = json.loads(args.input.read_text(encoding="utf-8"))
    raw = inputs["raw"]
    processed = inputs["processed"]
    options = {
        "random_state": 7,
        "length_partitioning": not args.no_length_partitioning,
        "length_partitioning_before_filtering": args.raw_length_partitioning,
        "extrema_prominence_fraction": args.prominence_fraction,
    }
    if args.consensus_k:
        options["consensus_n_clusters"] = args.consensus_k
    ensemble_runs = []
    if args.ensemble_ngrams:
        started = perf_counter()
        models = []
        for ngram_size in args.ensemble_ngrams:
            run_started = perf_counter()
            model = LogSPHNCS(metric=partial(char_ngram_jaccard, ngram_size=ngram_size), **options).fit(raw)
            models.append(model)
            ensemble_runs.append(
                {
                    "ngram_size": ngram_size,
                    "fit_seconds": perf_counter() - run_started,
                    "n_clusters": int(model.n_clusters_),
                }
            )
        if any(model.processed_strings_ != processed for model in models):
            raise RuntimeError("LogSPHNCS filtering differs from the benchmark input")
        voter = SphncsClusterer(clustering_mode="spectral_consensus", n_embeddings=len(models), random_state=7)
        labels = voter._spectral_labels(np.column_stack([model.labels_ for model in models]))
        fit_seconds = perf_counter() - started
        method = "LogSPHNCS ensemble (" + "/".join(f"{size}-gram" for size in args.ensemble_ngrams) + ")"
        matrix_started = perf_counter()
        distances = np.mean(
            [pairwise_distances(processed, ngram_size) for ngram_size in args.ensemble_ngrams], axis=0
        )
        matrix_seconds = perf_counter() - matrix_started
        space = "mean multiset character Jaccard distance across " + ", ".join(f"{size}-grams" for size in args.ensemble_ngrams)
    else:
        started = perf_counter()
        model = LogSPHNCS(metric=partial(char_ngram_jaccard, ngram_size=args.ngram_size), **options).fit(raw)
        fit_seconds = perf_counter() - started
        if model.processed_strings_ != processed:
            raise RuntimeError("LogSPHNCS filtering differs from the benchmark input")
        labels = model.labels_
        method = (
            f"LogSPHNCS (10D, Jaccard {args.ngram_size}-gram; no length partitions)"
            if args.no_length_partitioning
            else f"LogSPHNCS (10D, Jaccard {args.ngram_size}-gram; raw-length partitions)"
            if args.raw_length_partitioning
            else f"LogSPHNCS (10D, Jaccard {args.ngram_size}-gram)"
        )
        matrix_started = perf_counter()
        distances = pairwise_distances(processed, args.ngram_size)
        matrix_seconds = perf_counter() - matrix_started
        space = f"multiset character {args.ngram_size}-gram Jaccard distance"

    results = [
        {
            "method": method,
            "fit_seconds": fit_seconds,
            "labels": [str(label) for label in labels],
            "fitness": fitness([str(label) for label in labels], distances),
        }
    ]
    for result in json.loads(args.baselines.read_text(encoding="utf-8"))["results"]:
        if "labels" not in result:
            continue
        results.append({"method": result["method"], "parse_seconds": result["parse_seconds"], "labels": result["labels"], "fitness": fitness(result["labels"], distances)})
    output = {
        "records": len(processed),
        "space": "all preprocessed log filters; " + space,
        "length_partitioning_space": (
            "disabled" if args.no_length_partitioning else "raw strings" if args.raw_length_partitioning else "filtered strings"
        ),
        "consensus_cluster_count": args.consensus_k or "auto (median)",
        "extrema_prominence_fraction": args.prominence_fraction,
        "ensemble_runs": ensemble_runs,
        "pairwise_matrix_seconds": matrix_seconds,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
