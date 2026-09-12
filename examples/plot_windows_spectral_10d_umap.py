"""Plot a UMAP view of the 10-D spectral-consensus FastMap embedding."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from sphncs import SphncsClusterer
from sphncs.distances import char_ngram_jaccard

try:  # Supports both ``python examples/...`` and module-style invocation.
    from .windows_jaccard_4gram import DEFAULT_URL, read_first_log_records
except ImportError:  # pragma: no cover - direct-script path
    from windows_jaccard_4gram import DEFAULT_URL, read_first_log_records


def main() -> None:
    records, archive_member = read_first_log_records(DEFAULT_URL, 1000)
    metric = partial(char_ngram_jaccard, ngram_size=4)
    model = SphncsClusterer(
        metric=metric,
        length_partitioning=True,
        clustering_mode="spectral_consensus",
        n_embeddings=10,
        random_state=7,
    ).fit(records)
    projection = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=7,
    ).fit_transform(model.embedding_)

    figure, axes = plt.subplots(figsize=(11, 8), constrained_layout=True)
    scatter = axes.scatter(
        projection[:, 0],
        projection[:, 1],
        c=model.labels_,
        cmap=plt.get_cmap("turbo", model.n_clusters_),
        s=18,
        alpha=0.85,
        linewidths=0,
    )
    colorbar = figure.colorbar(scatter, ax=axes, pad=0.02)
    colorbar.set_label("Spectral-consensus cluster label")
    axes.set(
        title=(
            "Windows LogHub: UMAP of 10-D FastMap embedding\n"
            "4-gram Jaccard · length partitioning · spectral consensus"
        ),
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )
    axes.grid(alpha=0.15)

    path = Path("outputs/windows-jaccard-4gram-length-spectral-10d-umap.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    print(
        f"saved {path} ({len(records)} records, {model.embedding_.shape[1]} FastMap dimensions, "
        f"{model.n_clusters_} clusters; source member: {archive_member})"
    )


if __name__ == "__main__":
    main()
