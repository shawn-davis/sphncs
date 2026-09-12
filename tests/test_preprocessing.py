import pytest

from sphncs import LogPreprocessor, LogSPHNCS, SphncsClusterer


def test_log_preprocessor_removes_metadata_and_masks_selected_values():
    preprocessor = LogPreprocessor(
        ["timestamp", "severity", "uuid", "ip", "hex", "number", "path", "quoted", "identifier"]
    )
    value = (
        '2026-09-12 09:10:11, INFO pid=42 request "open" from 192.168.1.4 '
        'failed at C:\\Logs\\app.log with 0x80004005 id 17 '
        'trace 123e4567-e89b-12d3-a456-426614174000'
    )

    assert preprocessor.transform(value) == (
        "<#T> <#S> <#D> request <#Q> from <#I> failed at <#P> with <#H> "
        "id <#N> <#S> <#U>"
    )


def test_variable_and_all_shortcuts_expand_to_documented_filters():
    assert LogPreprocessor("variable").transform("pid=7 10.0.0.1 99") == "<#D> <#I> <#N>"
    assert LogPreprocessor("all").transform("2026-01-02 03:04:05 INFO pid=7") == (
        "<#T> <#S> <#D>"
    )


def test_number_filter_normalizes_decimal_and_bare_hexadecimal_log_fields():
    preprocessor = LogPreprocessor(["hex", "number"])
    assert preprocessor.transform("code 0000000e sequence 00000007 HRESULT 0x80004005") == (
        "code <#N> sequence <#N> HRESULT <#H>"
    )
    assert preprocessor.transform("component 31bf3856ad364e35") == "component 31bf3856ad364e35"


def test_log_preprocessor_rejects_unknown_filters():
    with pytest.raises(ValueError, match="Unknown log filter"):
        LogPreprocessor(["timestamp", "bogus"])


def test_log_preprocessor_without_filters_preserves_the_original_value():
    value = " 2026-09-12 09:10:11 INFO id=42 "
    assert LogPreprocessor().transform(value) == value


def test_clusterer_defaults_to_filtered_representatives_and_retains_raw_strings():
    pytest.importorskip("KDEpy")
    strings = [
        "2026-09-12 09:10:11 INFO event id=101 completed",
        "2026-09-12 09:11:12 WARNING event id=202 completed",
        "2026-09-12 09:12:13 ERROR connection id=303 failed",
    ]
    model = SphncsClusterer(
        metric="char_ngram_jaccard",
        log_filters=["timestamp", "severity", "number"],
        random_state=0,
        bandwidth=0.1,
        grid_points=128,
    ).fit(strings)

    assert model.processed_strings_ == [
        "<#T> <#S> event id=<#N> completed",
        "<#T> <#S> event id=<#N> completed",
        "<#T> <#S> connection id=<#N> failed",
    ]
    assert model.raw_strings_ == strings
    assert model.raw_representatives_ == [strings[index] for index in model.representative_indices_]
    assert model.representatives_ == [model.processed_strings_[index] for index in model.representative_indices_]
    assert model.get_raw_string(1) == strings[1]
    assert model.predict(["2027-01-01 12:00:00 DEBUG event id=999 completed"]).shape == (1,)


def test_length_partitioning_can_use_raw_lengths_before_log_filtering():
    pytest.importorskip("KDEpy")
    strings = [
        "2026-09-12 09:10:11 INFO event id=101 completed",
        "2026-09-12 09:11:12 WARNING event id=202 completed",
        "2026-09-12 09:12:13 ERROR event id=303 completed",
    ]
    model = SphncsClusterer(
        metric="char_ngram_jaccard",
        length_partitioning=True,
        length_partitioning_before_filtering=True,
        log_filters=["timestamp", "severity", "number"],
        random_state=0,
        bandwidth=0.1,
        grid_points=128,
    ).fit(strings)

    assert model.length_strings_ == strings
    assert model.processed_strings_[0] != strings[0]
    assert model.predict(["2027-01-01 12:00:00 INFO event id=404 completed"]).shape == (1,)


def test_log_sphncs_compresses_exact_filtered_templates_before_fastmap():
    pytest.importorskip("KDEpy")
    strings = [
        "2026-09-12 09:10:11 INFO event id=101 completed",
        "2026-09-12 09:11:12 INFO event id=202 completed",
        "2026-09-12 09:12:13 INFO event id=303 completed",
    ]
    model = LogSPHNCS(bandwidth=0.1, grid_points=128, random_state=0).fit(strings)

    assert model.processed_strings_ == ["<#T> <#S> event id=<#N> completed"] * 3
    assert model.embedding_.shape == (3, 10)
    assert model.partitions_[0].fastmap.X_ == [model.processed_strings_[0]]
    assert model.raw_strings_ == strings
