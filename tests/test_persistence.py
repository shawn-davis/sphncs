import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from sphncs import LogSPHNCS, ModelPersistenceError, SphncsClusterer


def test_saved_generic_model_round_trips_predictions_and_embeddings(tmp_path):
    pytest.importorskip("KDEpy")
    values = ["order-001", "order-002", "invoice-900", "invoice-901"]
    model = SphncsClusterer(metric="normalized_levenshtein", grid_points=128, random_state=0).fit(values)
    path = model.save(tmp_path / "generic.sphncs")

    restored = SphncsClusterer.load(path)

    assert path.exists()
    assert np.allclose(restored.transform(values), model.transform(values))
    assert restored.predict(values).tolist() == model.predict(values).tolist()
    assert restored.representatives_ == model.representatives_
    with ZipFile(path) as archive:
        assert any(name.startswith("fastmaps/") for name in archive.namelist())


def test_saved_log_model_round_trips_subclass_and_filtered_predictions(tmp_path):
    pytest.importorskip("KDEpy")
    values = [
        "2026-01-01 INFO order id=001 completed",
        "2026-01-02 INFO order id=002 completed",
        "2026-01-03 ERROR invoice id=900 failed",
    ]
    model = LogSPHNCS(grid_points=128, random_state=0).fit(values)

    restored = LogSPHNCS.load(model.save(tmp_path / "logs.sphncs"))

    assert isinstance(restored, LogSPHNCS)
    assert restored.processed_strings_ == model.processed_strings_
    assert restored.predict(values).tolist() == model.predict(values).tolist()


def test_saving_unfitted_model_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="fitted"):
        SphncsClusterer(metric="normalized_levenshtein").save(tmp_path / "model.sphncs")


def test_load_rejects_checksum_mismatch(tmp_path):
    pytest.importorskip("KDEpy")
    model = SphncsClusterer(metric="normalized_levenshtein", grid_points=128).fit(["a", "b"])
    path = model.save(tmp_path / "model.sphncs")
    with ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    contents["estimator.pkl"] += b"tampered"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in contents.items():
            archive.writestr(name, payload)

    with pytest.raises(ModelPersistenceError, match="integrity"):
        SphncsClusterer.load(path)


def test_load_rejects_unsupported_format_version(tmp_path):
    path = tmp_path / "bad-version.sphncs"
    manifest = {"format": "sphncs-model", "format_version": 99, "payload": "model.pkl", "model_class": "x", "sha256": "0" * 64}
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("model.pkl", b"")

    with pytest.raises(ModelPersistenceError, match="unsupported model format version"):
        SphncsClusterer.load(path)
