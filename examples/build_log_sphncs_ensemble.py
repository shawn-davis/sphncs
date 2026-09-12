"""Fit several LogSPHNCS metrics and save their spectral-consensus labels."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from sphncs import LogSPHNCS, SphncsClusterer
from sphncs.distances import char_ngram_jaccard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--ngrams", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(size < 1 for size in args.ngrams):
        parser.error("--ngrams values must be positive")

    raw = json.loads(args.input.read_text(encoding="utf-8"))["raw"]
    started = perf_counter()
    models = []
    runs = []
    for size in args.ngrams:
        run_started = perf_counter()
        model = LogSPHNCS(metric=partial(char_ngram_jaccard, ngram_size=size), random_state=7).fit(raw)
        models.append(model)
        runs.append({"ngram_size": size, "fit_seconds": perf_counter() - run_started, "n_clusters": int(model.n_clusters_)})
    voter = SphncsClusterer(clustering_mode="spectral_consensus", n_embeddings=len(models), random_state=7)
    labels = voter._spectral_labels(np.column_stack([model.labels_ for model in models]))
    output = {
        "method": "LogSPHNCS ensemble (" + "/".join(f"{size}-gram" for size in args.ngrams) + ")",
        "fit_seconds": perf_counter() - started,
        "ngrams": args.ngrams,
        "runs": runs,
        "labels": [int(label) for label in labels],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "labels"}, indent=2))


if __name__ == "__main__":
    main()
