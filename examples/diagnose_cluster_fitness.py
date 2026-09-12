"""Explain medoid scatter and Davies--Bouldin differences between two labelings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from compare_log_cluster_fitness import pairwise_distances


def diagnose(name: str, labels: list[str], values: list[str], distances: np.ndarray) -> dict:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    cluster_labels = list(groups)
    members = list(groups.values())
    medoids: list[int] = []
    scatter: list[float] = []
    weighted_distances: list[np.ndarray] = []
    for indices in members:
        block = distances[np.ix_(indices, indices)]
        medoid_at = int(np.argmin(block.mean(axis=1)))
        medoids.append(indices[medoid_at])
        scatter.append(float(block[medoid_at].mean()))
        weighted_distances.append(block[medoid_at])

    medoid_distances = distances[np.ix_(medoids, medoids)]
    scatter_array = np.asarray(scatter)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = (scatter_array[:, None] + scatter_array[None, :]) / medoid_distances
    np.fill_diagonal(ratios, -np.inf)
    worst_neighbor = np.argmax(ratios, axis=1)
    clusters = []
    for index, label in enumerate(cluster_labels):
        neighbor = int(worst_neighbor[index])
        clusters.append(
            {
                "label": label,
                "size": len(members[index]),
                "scatter": scatter[index],
                "medoid_index": medoids[index],
                "medoid": values[medoids[index]],
                "worst_neighbor_label": cluster_labels[neighbor],
                "worst_neighbor_size": len(members[neighbor]),
                "worst_neighbor_medoid": values[medoids[neighbor]],
                "intermedoid_distance": float(medoid_distances[index, neighbor]),
                "db_ratio": float(ratios[index, neighbor]),
            }
        )
    clusters.sort(key=lambda cluster: cluster["db_ratio"], reverse=True)
    return {
        "method": name,
        "n_clusters": len(clusters),
        "unweighted_mean_medoid_scatter": float(scatter_array.mean()),
        "record_weighted_mean_medoid_distance": float(np.concatenate(weighted_distances).mean()),
        "medoid_davies_bouldin": float(np.mean(np.max(ratios, axis=1))),
        "clusters_by_db_contribution": clusters,
        "clusters_by_scatter": sorted(clusters, key=lambda cluster: cluster["scatter"], reverse=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with processed log lines")
    parser.add_argument("sphncs", type=Path, help="1%% prominence fitness output")
    parser.add_argument("baselines", type=Path, help="baseline parser fitness output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values = json.loads(args.input.read_text(encoding="utf-8"))["processed"]
    sphncs = json.loads(args.sphncs.read_text(encoding="utf-8"))["results"][0]
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))["results"]
    drain3 = next(result for result in baselines if result["method"] == "Drain3")
    distances = pairwise_distances(values, 4)
    output = {
        "space": "all regex-filtered logs; multiset character 4-gram Jaccard distance",
        "methods": [
            diagnose("SPHNCS 1% prominence", sphncs["labels"], values, distances),
            diagnose("Drain3", drain3["labels"], values, distances),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for method in output["methods"]:
        print(
            f"{method['method']}: DB={method['medoid_davies_bouldin']:.3f}, "
            f"unweighted scatter={method['unweighted_mean_medoid_scatter']:.3f}, "
            f"weighted scatter={method['record_weighted_mean_medoid_distance']:.3f}"
        )
        for cluster in method["clusters_by_db_contribution"][:5]:
            print(
                f"  {cluster['label']} (n={cluster['size']}) -> {cluster['worst_neighbor_label']} "
                f"(n={cluster['worst_neighbor_size']}): ratio={cluster['db_ratio']:.3f}, "
                f"scatter={cluster['scatter']:.3f}, intermedoid={cluster['intermedoid_distance']:.3f}"
            )


if __name__ == "__main__":
    main()
