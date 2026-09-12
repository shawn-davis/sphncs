"""Benchmark Drain3 and LogPAI template parsers on preprocessed log strings.

Run this script with an environment containing ``drain3`` and ``logparser3``.
The input JSON must contain a ``processed`` array; all parsers receive those
exact strings with their own masking/preprocessing disabled.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np


@lru_cache(maxsize=100_000)
def _grams(value: str) -> Counter[str]:
    return Counter(value[index : index + 4] for index in range(max(1, len(value) - 3)))


def jaccard(left: str, right: str) -> float:
    if left == right:
        return 0.0
    union = sum((_grams(left) | _grams(right)).values())
    return 1.0 - sum((_grams(left) & _grams(right)).values()) / union if union else 0.0


def summarize(name: str, labels: list[str], values: list[str], seconds: float, templates: dict[str, str]) -> dict:
    members: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    distances: list[float] = []
    representatives: list[dict] = []
    for label, indices in members.items():
        unique_indices: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            unique_indices[values[index]].append(index)
        unique_values = list(unique_indices)
        scores = [
            sum(len(unique_indices[right]) * jaccard(left, right) for right in unique_values)
            for left in unique_values
        ]
        representative = unique_indices[unique_values[int(np.argmin(scores))]][0]
        for value, cluster_indices in unique_indices.items():
            distances.extend([jaccard(value, values[representative])] * len(cluster_indices))
        representatives.append(
            {
                "label": label,
                "size": len(indices),
                "representative_index": representative,
                "representative": values[representative],
                "template": templates.get(label),
            }
        )
    values_array = np.asarray(distances)
    return {
        "method": name,
        "parse_seconds": seconds,
        "n_clusters": len(members),
        "singleton_clusters": sum(len(indices) == 1 for indices in members.values()),
        "labels": labels,
        "tightness": {
            "mean_distance": float(values_array.mean()),
            "median_distance": float(np.median(values_array)),
            "p95_distance": float(np.quantile(values_array, 0.95, method="nearest")),
            "within_0_25": float(np.mean(values_array <= 0.25)),
        },
        "representatives": representatives,
    }


def drain3(values: list[str]) -> tuple[list[str], dict[str, str], float]:
    from drain3 import TemplateMiner
    from drain3.template_miner_config import TemplateMinerConfig

    config = TemplateMinerConfig()
    config.masking_instructions = []
    miner = TemplateMiner(persistence_handler=None, config=config)
    started = perf_counter()
    labels: list[str] = []
    templates: dict[str, str] = {}
    for value in values:
        result = miner.add_log_message(value)
        label = str(result["cluster_id"])
        labels.append(label)
        templates[label] = result["template_mined"]
    return labels, templates, perf_counter() - started


def logpai_parser(name: str, values: list[str], workspace: Path) -> tuple[list[str], dict[str, str], float]:
    import pandas as pd

    input_name = "filtered.log"
    input_path = workspace / input_name
    input_path.write_text("\n".join(values) + "\n", encoding="utf-8")
    output = workspace / name
    if name == "Spell":
        from logparser.Spell import LogParser

        parser = LogParser(indir=str(workspace), outdir=str(output), log_format="<Content>", tau=0.5, rex=[])
    elif name == "IPLoM":
        from logparser.IPLoM import LogParser

        parser = LogParser(log_format="<Content>", indir=str(workspace), outdir=str(output), maxEventLen=200, step2Support=0, CT=0.35, lowerBound=0.25, upperBound=0.9, rex=[])
    elif name == "LogCluster":
        from logparser.LogCluster import LogParser

        parser = LogParser(str(workspace), "<Content>", str(output), rsupport=1, rex=[])
    else:  # pragma: no cover
        raise ValueError(name)
    started = perf_counter()
    parser.parse(input_name)
    seconds = perf_counter() - started
    structured = pd.read_csv(output / (input_name + "_structured.csv"))
    labels = structured["EventId"].astype(str).tolist()
    templates = dict(zip(structured["EventId"].astype(str), structured["EventTemplate"].astype(str)))
    return labels, templates, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("Drain3", "Spell", "IPLoM", "LogCluster"), default=("Drain3", "Spell", "IPLoM", "LogCluster"))
    args = parser.parse_args()
    values = json.loads(args.input.read_text(encoding="utf-8"))["processed"]
    results = []
    if "Drain3" in args.methods:
        labels, templates, seconds = drain3(values)
        results.append(summarize("Drain3", labels, values, seconds, templates))
    with tempfile.TemporaryDirectory(prefix="sphncs-logparser-") as directory:
        workspace = Path(directory)
        for name in ("Spell", "IPLoM", "LogCluster"):
            if name not in args.methods:
                continue
            try:
                labels, templates, seconds = logpai_parser(name, values, workspace)
                results.append(summarize(name, labels, values, seconds, templates))
            except Exception as error:  # Record unsupported parser failures without hiding other baselines.
                results.append({"method": name, "error": f"{type(error).__name__}: {error}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": len(values), "metric": "char_ngram_jaccard", "ngram_size": 4, "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
