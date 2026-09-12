"""Run LogSPHNCS-style clustering on Drain3-mined templates, not regex masks."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from sphncs import SphncsClusterer
from sphncs.distances import char_ngram_jaccard

from compare_log_cluster_fitness import fitness, pairwise_distances


def drain_templates(records: list[str]) -> tuple[list[str], int, float]:
    """Map every record to its final Drain3 template with no extra regex masks."""
    config = TemplateMinerConfig()
    config.masking_instructions = []
    miner = TemplateMiner(persistence_handler=None, config=config)
    started = perf_counter()
    cluster_ids = [miner.add_log_message(record)["cluster_id"] for record in records]
    # Drain3 generalizes templates online. Replacing every provisional template
    # with its final cluster template gives preprocessing a stable vocabulary.
    templates = [miner.drain.id_to_cluster[cluster_id].get_template() for cluster_id in cluster_ids]
    return templates, len(miner.drain.id_to_cluster), perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing raw log lines")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))["raw"]
    templates, final_drain3_clusters, drain_seconds = drain_templates(raw)
    metric = partial(char_ngram_jaccard, ngram_size=4)
    started = perf_counter()
    model = SphncsClusterer(
        metric=metric,
        length_partitioning=True,
        clustering_mode="spectral_consensus",
        n_embeddings=10,
        deduplicate_strings=True,
        random_state=7,
    ).fit(templates)
    sphncs_seconds = perf_counter() - started

    started = perf_counter()
    distances = pairwise_distances(templates, 4)
    fitness_seconds = perf_counter() - started
    output = {
        "method": "Drain3-template preprocessing + SPHNCS (10D, Jaccard 4-gram)",
        "records": len(raw),
        "drain3_preprocess_seconds": drain_seconds,
        "sphncs_fit_seconds": sphncs_seconds,
        "fitness_matrix_seconds": fitness_seconds,
        "n_drain3_templates": len(set(templates)),
        "n_final_drain3_clusters": final_drain3_clusters,
        "n_clusters": int(model.n_clusters_),
        "singleton_clusters": int(sum((model.labels_ == label).sum() == 1 for label in range(model.n_clusters_))),
        "space": "online Drain3 templates; multiset character 4-gram Jaccard distance",
        "fitness": fitness([str(label) for label in model.labels_], distances),
        "labels": [int(label) for label in model.labels_],
        "representatives": [
            {
                "label": int(label),
                "template": model.representatives_[label],
                "raw_representative": raw[index],
                "representative_index": int(index),
            }
            for label, index in enumerate(model.representative_indices_)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in {"labels", "representatives"}}, indent=2))


if __name__ == "__main__":
    main()
