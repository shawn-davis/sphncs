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
