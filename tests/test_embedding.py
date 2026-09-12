import numpy as np

from sphncs.distances import normalized_levenshtein
from sphncs.embedding import FastMapyEmbeddings


def test_fastmapy_batched_embeddings_fit_and_transform_without_distance_matrix():
    strings = ["abc", "abd", "xyz"]
    model = FastMapyEmbeddings(2, normalized_levenshtein)
    coordinates = model.fit_transform(strings)
    assert coordinates.shape == (3, 2)
    assert np.allclose(coordinates, model.transform(strings))
