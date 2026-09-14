from math import inf

import pytest

from sphncs.distances import (
    char_ngram_jaccard,
    l1_distance,
    l2_distance,
    lp_distance,
    normalized_levenshtein,
    resolve_metric,
)


def test_normalized_levenshtein():
    assert normalized_levenshtein("kitten", "sitting") == 3 / 7
    assert normalized_levenshtein("same", "same") == 0


def test_ngram_jaccard_is_bounded():
    assert char_ngram_jaccard("abcdef", "abcdef") == 0
    assert 0 < char_ngram_jaccard("abcdef", "uvwxyz") <= 1


def test_l1_l2_and_general_lp_distances():
    left = [1, -2, 3]
    right = [4, 2, -1]
    assert l1_distance(left, right) == 11
    assert l2_distance(left, right) == pytest.approx(41**0.5)
    assert lp_distance(left, right, p=3) == pytest.approx(155 ** (1 / 3))
    assert lp_distance(left, right, p=inf) == 4
    assert resolve_metric("l1") is l1_distance
    assert resolve_metric("l2") is l2_distance


@pytest.mark.parametrize("left,right,p,error", [
    ([1], [1, 2], 2, "equal lengths"),
    ([1], [2], 0.5, "greater than or equal"),
    ([float("nan")], [1], 2, "finite"),
])
def test_lp_distance_validates_inputs(left, right, p, error):
    with pytest.raises(ValueError, match=error):
        lp_distance(left, right, p=p)
