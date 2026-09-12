"""Plot first-stage length KDEs before and after log-field filtering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sphncs.density import fit_density
from sphncs.preprocessing import LogPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing the raw Windows log lines")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text(encoding="utf-8"))["raw"]
    filtered = LogPreprocessor("all").transform_many(raw)
    spaces = [("Raw lengths — partition before filtering", raw), ("Filtered lengths — current default", filtered)]

    figure, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    for axis, (title, values) in zip(axes, spaces, strict=True):
        lengths = np.asarray([len(value) for value in values], dtype=float)
        model = fit_density(lengths, bandwidth="ISJ", grid_points=512, min_samples=3)
        cluster_sizes = [int(np.sum(model.labels == label)) for label in np.unique(model.labels)]
        axis.hist(lengths, bins="auto", density=True, color="#a9c5e3", edgecolor="white", alpha=0.8, label="Observed length distribution")
        axis.plot(model.grid, model.density, color="#174a7e", linewidth=2.4, label="ISJ KDE")
        for boundary in model.boundaries:
            axis.axvline(boundary, color="#c53b32", linewidth=1.6, linestyle="--")
        boundary_label = "no cuts" if not len(model.boundaries) else f"{len(model.boundaries)} KDE cut points"
        axis.set(
            title=f"{title}: {len(cluster_sizes)} partitions ({boundary_label})",
            xlabel="String length (characters)",
            ylabel="Density",
        )
        axis.text(
            0.99,
            0.95,
            "Partition sizes: " + ", ".join(map(str, sorted(cluster_sizes, reverse=True))),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#a8a8a8", "alpha": 0.9, "pad": 5},
        )
        axis.grid(alpha=0.18)
        axis.legend(loc="upper left")

    figure.suptitle("Windows first 1,000 logs: length-partitioning KDEs", fontsize=15, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
