"""Score a saved LogSPHNCS ensemble in its mean constituent-metric space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from compare_log_cluster_fitness import fitness, pairwise_distances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing processed log strings")
    parser.add_argument("ensemble", type=Path, help="output of build_log_sphncs_ensemble.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))["processed"]
    ensemble = json.loads(args.ensemble.read_text(encoding="utf-8"))
    started = perf_counter()
    distances = np.mean([pairwise_distances(records, size) for size in ensemble["ngrams"]], axis=0)
    output = {
        **ensemble,
        "space": "mean multiset Jaccard distance across " + ", ".join(f"{size}-grams" for size in ensemble["ngrams"]),
        "pairwise_matrix_seconds": perf_counter() - started,
        "fitness": fitness([str(label) for label in ensemble["labels"]], distances),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "labels"}, indent=2))


if __name__ == "__main__":
    main()
