from sphncs.distances import char_ngram_jaccard, normalized_levenshtein


def test_normalized_levenshtein():
    assert normalized_levenshtein("kitten", "sitting") == 3 / 7
    assert normalized_levenshtein("same", "same") == 0


def test_ngram_jaccard_is_bounded():
    assert char_ngram_jaccard("abcdef", "abcdef") == 0
    assert 0 < char_ngram_jaccard("abcdef", "uvwxyz") <= 1
