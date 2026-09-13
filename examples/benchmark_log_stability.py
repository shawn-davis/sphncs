"""Measure sample-size and arrival-order stability of LogSPHNCS and Drain3."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import adjusted_rand_score

from benchmark_template_parsers import drain3
from sphncs import LogSPHNCS


def ordered_labels(labels: np.ndarray | list[str], order: np.ndarray) -> list[str]:
    """Put labels from a permuted fit back in original prefix order."""
    result = np.empty(len(order), dtype=object)
    result[order] = labels
    return result.tolist()


def pairwise_ari(runs: list[dict]) -> list[float]:
    return [
        float(adjusted_rand_score(left["labels"], right["labels"]))
        for left, right in combinations(runs, 2)
    ]


def summarize(runs: list[dict], largest_labels: list[str] | None) -> dict:
    aris = pairwise_ari(runs)
    canonical = next(run for run in runs if run["order"] == "original")
    result = {
        "mean_runtime_seconds": float(np.mean([run["runtime_seconds"] for run in runs])),
        "cluster_count_min": min(run["n_clusters"] for run in runs),
        "cluster_count_max": max(run["n_clusters"] for run in runs),
        "arrival_order_ari_mean": float(np.mean(aris)),
        "arrival_order_ari_min": float(np.min(aris)),
    }
    if largest_labels is not None:
        result["prefix_ari_to_largest"] = float(adjusted_rand_score(canonical["labels"], largest_labels[: len(canonical["labels"])]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw/processed benchmark JSON")
    parser.add_argument("--sizes", type=int, nargs="+", default=(1_000, 5_000, 10_000))
    parser.add_argument("--shuffle-seeds", type=int, nargs="*", default=(11, 29))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    raw = source["raw"]
    processed = source["processed"]
    if sorted(args.sizes) != list(args.sizes) or any(size < 2 or size > len(raw) for size in args.sizes):
        parser.error("--sizes must be ascending and lie within the input")

    methods: dict[str, dict[int, list[dict]]] = {"LogSPHNCS": {}, "Drain3": {}}
    orders = [("original", None)] + [(f"shuffle-{seed}", seed) for seed in args.shuffle_seeds]
    for size in args.sizes:
        indices = np.arange(size)
        for order_name, seed in orders:
            order = indices if seed is None else np.random.default_rng(seed).permutation(indices)

            started = perf_counter()
            model = LogSPHNCS(
                random_state=7,
                extrema_prominence_fraction=0.01,
                consensus_n_clusters="max",
            ).fit([raw[index] for index in order])
            methods["LogSPHNCS"].setdefault(size, []).append(
                {
                    "order": order_name,
                    "runtime_seconds": perf_counter() - started,
                    "n_clusters": int(model.n_clusters_),
                    "labels": ordered_labels(model.labels_, order),
                }
            )

            started = perf_counter()
            labels, _, seconds = drain3([processed[index] for index in order])
            methods["Drain3"].setdefault(size, []).append(
                {
                    "order": order_name,
                    "runtime_seconds": seconds if seconds else perf_counter() - started,
                    "n_clusters": len(set(labels)),
                    "labels": ordered_labels(labels, order),
                }
            )
            print(f"completed size={size:,}, order={order_name}", flush=True)

    output = {
        "records_available": len(raw),
        "sizes": args.sizes,
        "arrival_orders": [name for name, _ in orders],
        "configuration": {
            "logsphncs": "10D, 4-gram Jaccard, all preprocessing filters, filtered-length partitions, 1% prominence, max consensus k",
            "drain3": "preprocessed strings, parser masking disabled",
        },
        "runs": methods,
        "summary": {},
    }
    largest = args.sizes[-1]
    for method, by_size in methods.items():
        largest_labels = next(run["labels"] for run in by_size[largest] if run["order"] == "original")
        output["summary"][method] = {
            str(size): summarize(runs, None if size == largest else largest_labels)
            for size, runs in by_size.items()
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
