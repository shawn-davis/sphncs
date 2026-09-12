"""Score 10D spectral sphncs using normalized Levenshtein on filtered inputs."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from rapidfuzz.distance import Levenshtein

from sphncs import SphncsClusterer
from sphncs.distances import char_ngram_jaccard


def accelerated_normalized_levenshtein(left: str, right: str) -> float:
    """Equivalent normalized distance, implemented in RapidFuzz's C extension."""
    return float(Levenshtein.normalized_distance(left, right))


def tightness(records, labels, metric):
    groups = defaultdict(list)
    for index, label in enumerate(labels):
        groups[int(label)].append(index)
    distances = []
    for indices in groups.values():
        scores = [np.mean([metric(records[left], records[right]) for right in indices]) for left in indices]
        representative = indices[int(np.argmin(scores))]
        distances.extend(metric(records[index], records[representative]) for index in indices)
    values = np.asarray(distances)
    return {
        "mean_distance": float(values.mean()),
        "median_distance": float(np.median(values)),
        "p95_distance": float(np.quantile(values, 0.95, method="nearest")),
        "within_0_25": float(np.mean(values <= 0.25)),
    }


def main():
    input_path = Path("/private/tmp/windows-first1000-all-filtered.json")
    output_path = Path("outputs/windows-first1000-all-filtered-sphncs-10d-levenshtein-medoid.json")
    records = json.loads(input_path.read_text(encoding="utf-8"))["processed"]
    model = SphncsClusterer(
        metric=accelerated_normalized_levenshtein,
        length_partitioning=True,
        clustering_mode="spectral_consensus",
        n_embeddings=10,
        random_state=7,
    )
    started = perf_counter()
    model.fit(records)
    output = {
        "method": "sphncs 10D spectral (normalized Levenshtein)",
        "fit_seconds": perf_counter() - started,
        "n_clusters": int(model.n_clusters_),
        "singleton_clusters": int(sum((model.labels_ == label).sum() == 1 for label in range(model.n_clusters_))),
        "jaccard_tightness": tightness(records, model.labels_, partial(char_ngram_jaccard, ngram_size=4)),
        "levenshtein_tightness": tightness(records, model.labels_, accelerated_normalized_levenshtein),
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
