import numpy as np
import pytest

from sphncs import SphncsClusterer


def test_single_mode_fits_and_predicts():
    pytest.importorskip("KDEpy")
    strings = ["order-001", "order-002", "invoice-900", "invoice-901"]
    model = SphncsClusterer(random_state=0, grid_points=128).fit(strings)
    assert model.embedding_.shape == (4, 1)
    assert len(model.labels_) == 4
    assert model.predict(strings).shape == (4,)


def test_spectral_mode_fits():
    pytest.importorskip("KDEpy")
    strings = ["abc001", "abc002", "abc003", "xyz100", "xyz101", "xyz102"]
    model = SphncsClusterer(
        clustering_mode="spectral_consensus", n_embeddings=2,
        consensus_n_clusters=2, random_state=0, grid_points=128,
    ).fit(strings)
    assert model.embedding_.shape == (6, 2)
    assert len(np.unique(model.labels_)) == 2


def test_spectral_auto_count_uses_observed_kde_cluster_counts():
    model = SphncsClusterer(clustering_mode="spectral_consensus", n_embeddings=10)
    # A connected co-association graph whose ten input clusterings contain one,
    # two, and three observed clusters: median count is two.
    base_labels = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 0],
            [0, 0, 0, 1, 1, 1, 2, 2, 0, 0],
        ]
    )
    affinity = model._coassociation_graph(base_labels)
    assert model._resolve_consensus_count(affinity, base_labels) == 2


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [("min", 2), ("mean", 3), ("median", 3), ("max", 4)],
)
def test_spectral_cluster_count_reducers_use_per_embedding_kde_counts(strategy, expected):
    base_labels = np.array([[0, 0, 0], [0, 1, 1], [1, 1, 2], [1, 2, 3]])
    model = SphncsClusterer(clustering_mode="spectral_consensus", n_embeddings=3, consensus_n_clusters=strategy)
    assert model._resolve_consensus_count(model._coassociation_graph(base_labels), base_labels) == expected


def test_spectral_consensus_keeps_equivalent_base_labels_together():
    model = SphncsClusterer(clustering_mode="spectral_consensus", n_embeddings=2, random_state=0)
    base_labels = np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 0]])
    labels = model._spectral_labels(base_labels)
    assert labels[0] == labels[1]


def test_density_falls_back_when_isj_cannot_fit_a_discrete_partition(monkeypatch):
    pytest.importorskip("KDEpy")
    from sphncs import density

    calls = []

    class FailingISJ:
        def __init__(self, *, kernel, bw):
            calls.append(bw)
            self.bw = bw

        def fit(self, values):
            return self

        def evaluate(self, grid):
            if self.bw == "ISJ":
                raise ValueError("no root")
            return np.ones(len(grid))

    monkeypatch.setattr("KDEpy.FFTKDE", FailingISJ)
    result = density.fit_density(np.array([0.0, 1.0, 2.0]), bandwidth="ISJ", grid_points=16)
    assert calls == ["ISJ", "silverman"]
    assert result.labels.shape == (3,)


@pytest.mark.parametrize("fraction", [0, -0.01, 1.01])
def test_extrema_prominence_fraction_must_be_a_unit_fraction(fraction):
    model = SphncsClusterer(extrema_prominence_fraction=fraction)
    with pytest.raises(ValueError, match="extrema_prominence_fraction"):
        model._validate_parameters()
