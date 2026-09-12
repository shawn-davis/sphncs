"""Prototype a blocked Dirichlet-process refinement seeded by SPHNCS labels.

Each distinct filtered template is a block whose multiplicity contributes to a
character-4-gram Dirichlet-multinomial likelihood.  Blocks may move between any
tables, so this refinement can correct hard length-partition boundaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import gammaln

from compare_log_cluster_fitness import fitness, pairwise_distances


def grams(value: str, size: int = 4) -> Counter[str]:
    return Counter(value[index : index + size] for index in range(max(1, len(value) - size + 1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with processed log strings")
    parser.add_argument("seed_labels", type=Path, help="SPHNCS fitness output containing record labels")
    parser.add_argument("--alpha", type=float, default=1.0, help="CRP concentration")
    parser.add_argument("--prior-mass", type=float, default=10.0, help="total symmetric Dirichlet prior mass")
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument(
        "--candidate-count",
        type=int,
        help="restrict each update to this many nearest active table medoids",
    )
    parser.add_argument(
        "--cohesion-gate",
        action="store_true",
        help="accept a proposed move only when local Jaccard-medoid cost does not increase",
    )
    db_gates = parser.add_mutually_exclusive_group()
    db_gates.add_argument(
        "--db-gate",
        action="store_true",
        help="reject moves that worsen the worst affected medoid Davies--Bouldin ratio",
    )
    db_gates.add_argument(
        "--global-db-gate",
        action="store_true",
        help="reject moves that worsen the full medoid Davies--Bouldin objective",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.alpha <= 0 or args.prior_mass <= 0 or args.sweeps < 1 or args.candidate_count is not None and args.candidate_count < 1:
        parser.error("alpha, prior mass, and sweeps must be positive")

    values = json.loads(args.input.read_text(encoding="utf-8"))["processed"]
    seed_labels = json.loads(args.seed_labels.read_text(encoding="utf-8"))["results"][0]["labels"]
    by_template: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        by_template[value].append(index)
    templates = list(by_template)
    members = list(by_template.values())
    weights = np.asarray([len(indices) for indices in members], dtype=int)

    vocabulary = {gram for template in templates for gram in grams(template)}
    vocabulary_index = {gram: index for index, gram in enumerate(vocabulary)}
    vectors: list[tuple[np.ndarray, np.ndarray]] = []
    for template in templates:
        counts = grams(template)
        indices = np.fromiter((vocabulary_index[gram] for gram in counts), dtype=int)
        data = np.fromiter((count for count in counts.values()), dtype=float)
        vectors.append((indices, data))
    eta = args.prior_mass / len(vocabulary)
    template_distances = (
        pairwise_distances(templates, 4)
        if args.candidate_count or args.cohesion_gate or args.db_gate or args.global_db_gate
        else None
    )

    template_labels = []
    split_seed_templates = 0
    for indices in members:
        counts = Counter(seed_labels[index] for index in indices)
        if len(counts) > 1:
            split_seed_templates += 1
        template_labels.append(int(counts.most_common(1)[0][0]))
    assignments = np.asarray(template_labels, dtype=int)
    next_label = int(assignments.max()) + 1
    cluster_counts: dict[int, np.ndarray] = {}
    cluster_tokens: dict[int, float] = defaultdict(float)
    cluster_records: dict[int, int] = defaultdict(int)
    table_items: dict[int, set[int]] = defaultdict(set)
    for item, label in enumerate(assignments):
        if label not in cluster_counts:
            cluster_counts[label] = np.zeros(len(vocabulary), dtype=float)
        indices, data = vectors[item]
        cluster_counts[label][indices] += data * weights[item]
        cluster_tokens[label] += float(data.sum() * weights[item])
        cluster_records[label] += int(weights[item])
        table_items[label].add(item)

    def predictive(label: int | None, item: int) -> float:
        indices, data = vectors[item]
        data = data * weights[item]
        if label is None:
            counts, total = 0.0, 0.0
        else:
            counts, total = cluster_counts[label][indices], cluster_tokens[label]
        return float(
            np.sum(gammaln(counts + data + eta) - gammaln(counts + eta))
            + gammaln(total + len(vocabulary) * eta)
            - gammaln(total + data.sum() + len(vocabulary) * eta)
        )

    rng = np.random.default_rng(args.seed)
    history = []
    proposed_moves = 0
    rejected_moves = 0
    db_rejected_moves = 0
    global_db_rejected_moves = 0

    def cohesion_cost(items: set[int]) -> float:
        if not items:
            return 0.0
        item_array = np.fromiter(items, dtype=int)
        block = template_distances[np.ix_(item_array, item_array)]
        item_weights = weights[item_array]
        medoid = int(np.argmin(block @ item_weights))
        return float(np.dot(item_weights, block[:, medoid]))

    def table_profile(items: set[int]) -> tuple[int, float] | None:
        if not items:
            return None
        item_array = np.fromiter(items, dtype=int)
        block = template_distances[np.ix_(item_array, item_array)]
        item_weights = weights[item_array]
        medoid = int(item_array[int(np.argmin(block @ item_weights))])
        return medoid, float(np.dot(item_weights, template_distances[item_array, medoid]) / item_weights.sum())

    def affected_db(
        replacements: dict[int, tuple[int, float] | None],
        profiles: dict[int, tuple[int, float]],
    ) -> float:
        combined = {label: profile for label, profile in profiles.items() if label not in replacements}
        combined.update({label: profile for label, profile in replacements.items() if profile is not None})
        ratios = []
        for label, profile in replacements.items():
            if profile is None:
                continue
            medoid, scatter = profile
            for other_label, (other_medoid, other_scatter) in combined.items():
                if other_label == label:
                    continue
                separation = float(template_distances[medoid, other_medoid])
                ratios.append(np.inf if separation == 0 else (scatter + other_scatter) / separation)
        return max(ratios, default=0.0)

    def global_db(profiles: dict[int, tuple[int, float]]) -> float:
        """Return the exact medoid Davies--Bouldin objective for profiles."""
        if len(profiles) < 2:
            return 0.0
        ordered = list(profiles.values())
        medoid_indices = np.asarray([profile[0] for profile in ordered], dtype=int)
        scatter = np.asarray([profile[1] for profile in ordered], dtype=float)
        separation = template_distances[np.ix_(medoid_indices, medoid_indices)]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = (scatter[:, None] + scatter[None, :]) / separation
        off_diagonal = ~np.eye(len(ordered), dtype=bool)
        ratios[(separation == 0) & off_diagonal] = np.inf
        np.fill_diagonal(ratios, -np.inf)
        return float(np.mean(np.max(ratios, axis=1)))

    def current_profiles(
        replacements: dict[int, set[int] | None],
    ) -> dict[int, tuple[int, float]]:
        """Profile the whole current partition after replacing selected tables."""
        all_labels = set(table_items) | set(replacements)
        result = {}
        for label in all_labels:
            items = replacements[label] if label in replacements else table_items[label]
            if items:
                result[label] = table_profile(items)
        return result

    started = perf_counter()
    for sweep in range(args.sweeps):
        medoids: dict[int, int] = {}
        profiles: dict[int, tuple[int, float]] = {}
        if args.candidate_count or args.db_gate:
            for label, items in table_items.items():
                profile = table_profile(items)
                if profile is not None:
                    profiles[label] = profile
                    medoids[label] = profile[0]
        for item in rng.permutation(len(templates)):
            old = int(assignments[item])
            old_before = set(table_items[old])
            indices, data = vectors[item]
            cluster_counts[old][indices] -= data * weights[item]
            cluster_tokens[old] -= float(data.sum() * weights[item])
            cluster_records[old] -= int(weights[item])
            table_items[old].remove(item)
            if cluster_records[old] == 0:
                del cluster_counts[old], cluster_tokens[old], cluster_records[old]

            active = list(cluster_counts)
            if args.candidate_count:
                candidate_labels = [
                    label
                    for _, label in sorted(
                        (float(template_distances[item, medoids[label]]), label)
                        for label in active
                        if label in medoids
                    )[: args.candidate_count]
                ]
                if old in active and old not in candidate_labels:
                    candidate_labels.append(old)
                active = candidate_labels
            scores = np.asarray(
                [np.log(cluster_records[label]) + predictive(label, item) for label in active]
                + [np.log(args.alpha) + predictive(None, item)]
            )
            scores -= scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            selected = int(rng.choice(len(probabilities), p=probabilities))
            new = None if selected == len(active) else active[selected]
            if new != old:
                proposed_moves += 1
            if args.cohesion_gate and new != old:
                destination_before = set() if new is None else set(table_items[new])
                before_cost = cohesion_cost(old_before) + cohesion_cost(destination_before)
                after_cost = cohesion_cost(table_items[old]) + cohesion_cost(destination_before | {item})
                if after_cost > before_cost + 1e-12:
                    new = old
                    rejected_moves += 1
            if args.db_gate and new != old:
                destination_before = set() if new is None else set(table_items[new])
                destination_label = next_label if new is None else new
                before = {old: table_profile(old_before)}
                if new is not None:
                    before[new] = table_profile(destination_before)
                after = {old: table_profile(table_items[old])}
                after[destination_label] = table_profile(destination_before | {item})
                if affected_db(after, profiles) > affected_db(before, profiles) + 1e-12:
                    new = old
                    rejected_moves += 1
                    db_rejected_moves += 1
            if args.global_db_gate and new != old:
                destination_before = set() if new is None else set(table_items[new])
                destination_label = next_label if new is None else new
                before_profiles = current_profiles(
                    {old: old_before, **({new: destination_before} if new is not None else {})}
                )
                after_profiles = current_profiles(
                    {old: set(table_items[old]), destination_label: destination_before | {item}}
                )
                if global_db(after_profiles) > global_db(before_profiles) + 1e-12:
                    new = old
                    rejected_moves += 1
                    global_db_rejected_moves += 1
            if new is None:
                new = next_label
                next_label += 1
            if new not in cluster_counts:
                cluster_counts[new] = np.zeros(len(vocabulary), dtype=float)
                cluster_tokens[new] = 0.0
                cluster_records[new] = 0
            cluster_counts[new][indices] += data * weights[item]
            cluster_tokens[new] += float(data.sum() * weights[item])
            cluster_records[new] += int(weights[item])
            table_items[new].add(item)
            if old != new and not table_items[old]:
                del table_items[old]
            assignments[item] = new
        history.append({"sweep": sweep + 1, "n_clusters": len(cluster_counts)})

    refinement_seconds = perf_counter() - started

    record_labels = np.empty(len(values), dtype=int)
    for item, indices in enumerate(members):
        record_labels[indices] = assignments[item]
    _, record_labels = np.unique(record_labels, return_inverse=True)
    started = perf_counter()
    distances = pairwise_distances(values, 4)
    output = {
        "method": "SPHNCS-seeded blocked Dirichlet-process 4-gram refinement",
        "alpha": args.alpha,
        "dirichlet_prior_mass": args.prior_mass,
        "sweeps": args.sweeps,
        "candidate_count": args.candidate_count,
        "cohesion_gate": args.cohesion_gate,
        "db_gate": args.db_gate,
        "global_db_gate": args.global_db_gate,
        "proposed_moves": proposed_moves,
        "rejected_moves": rejected_moves,
        "db_rejected_moves": db_rejected_moves,
        "global_db_rejected_moves": global_db_rejected_moves,
        "random_seed": args.seed,
        "templates": len(templates),
        "vocabulary_size": len(vocabulary),
        "initial_clusters": len(set(seed_labels)),
        "seed_templates_split_across_clusters": split_seed_templates,
        "refinement_seconds": refinement_seconds,
        "fitness_matrix_seconds": perf_counter() - started,
        "cluster_history": history,
        "fitness": fitness([str(label) for label in record_labels], distances),
        "labels": [int(label) for label in record_labels],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "labels"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
