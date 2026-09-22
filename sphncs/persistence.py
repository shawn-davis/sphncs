"""Versioned persistence for fitted sphncs estimators.

SPHNCS stores estimator state in a validated archive and delegates fitted
FastMap projections to FastMapy's native, versioned save/load API. Archives
contain pickle data and must only be loaded from trusted sources.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .estimator import SphncsClusterer

PERSISTENCE_FORMAT = "sphncs-model"
PERSISTENCE_VERSION = 2
_MANIFEST_NAME = "manifest.json"
_ESTIMATOR_NAME = "estimator.pkl"
_T = TypeVar("_T", bound="SphncsClusterer")


class ModelPersistenceError(ValueError):
    """Raised when a model archive is malformed, incompatible, or unusable."""


def save_model(model: SphncsClusterer, path: str | os.PathLike[str]) -> Path:
    """Atomically save a fitted model to a versioned ``.sphncs`` archive."""
    model._require_fitted()
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"model path is a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.TemporaryDirectory(dir=target.parent) as staging_directory:
            staging = Path(staging_directory)
            estimator = _copy_without_fastmaps(model)
            try:
                estimator_payload = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
            except (pickle.PicklingError, TypeError, AttributeError) as exc:
                raise ModelPersistenceError(
                    "model is not serializable; metrics, transformers, and partitioning "
                    "features must be importable functions or otherwise pickleable objects"
                ) from exc
            try:
                fastmap_entries = _save_fastmaps(model, staging)
            except Exception as exc:
                raise ModelPersistenceError(
                    "FastMapy could not persist the fitted projection state"
                ) from exc
            artifacts = {_ESTIMATOR_NAME: _checksum(estimator_payload)}
            artifacts.update({entry: _checksum((staging / entry).read_bytes()) for group in fastmap_entries for entry in group})
            manifest = {
                "format": PERSISTENCE_FORMAT,
                "format_version": PERSISTENCE_VERSION,
                "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
                "estimator": _ESTIMATOR_NAME,
                "fastmaps": fastmap_entries,
                "sha256": artifacts,
            }
            with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False) as temporary:
                temporary_name = temporary.name
            with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(_MANIFEST_NAME, json.dumps(manifest, sort_keys=True, separators=(",", ":")))
                archive.writestr(_ESTIMATOR_NAME, estimator_payload)
                for group in fastmap_entries:
                    for entry in group:
                        archive.write(staging / entry, entry)
        with open(temporary_name, "rb") as temporary:
            os.fsync(temporary.fileno())
        os.replace(temporary_name, target)
    except OSError as exc:
        raise ModelPersistenceError(f"could not save model to {target}: {exc}") from exc
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return target


def load_model(path: str | os.PathLike[str], expected_type: type[_T]) -> _T:
    """Load a trusted model archive created by :func:`save_model`."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            manifest = _read_manifest(archive)
            artifact_names = [_ESTIMATOR_NAME, *[name for group in manifest["fastmaps"] for name in group]]
            members = archive.namelist()
            if set(members) != {_MANIFEST_NAME, *artifact_names} or len(members) != len(artifact_names) + 1:
                raise ModelPersistenceError("model archive has unexpected contents")
            artifacts = {name: archive.read(name) for name in artifact_names}
    except FileNotFoundError:
        raise
    except zipfile.BadZipFile as exc:
        raise ModelPersistenceError("model archive is not a valid ZIP file") from exc
    except OSError as exc:
        raise ModelPersistenceError(f"could not read model archive {source}: {exc}") from exc
    for name, payload in artifacts.items():
        if _checksum(payload) != manifest["sha256"][name]:
            raise ModelPersistenceError(f"model archive failed the integrity check for {name}")
    try:
        model = pickle.loads(artifacts[_ESTIMATOR_NAME])
    except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError, TypeError) as exc:
        raise ModelPersistenceError("estimator state could not be deserialized") from exc
    if not isinstance(model, expected_type):
        raise ModelPersistenceError(f"model archive contains {type(model).__name__}, not {expected_type.__name__}")
    if f"{type(model).__module__}.{type(model).__qualname__}" != manifest["model_class"]:
        raise ModelPersistenceError("model archive class does not match its manifest")
    if len(model.partitions_) != len(manifest["fastmaps"]):
        raise ModelPersistenceError("model archive has inconsistent partition state")
    try:
        with tempfile.TemporaryDirectory(dir=source.parent) as staging_directory:
            staging = Path(staging_directory)
            for partition, entries in zip(model.partitions_, manifest["fastmaps"]):
                paths = []
                for entry in entries:
                    output = staging / entry
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(artifacts[entry])
                    paths.append(output)
                partition.fastmap.load_models(paths)
    except Exception as exc:
        raise ModelPersistenceError("FastMapy projection state could not be restored") from exc
    model._require_fitted()
    return model


def _copy_without_fastmaps(model: SphncsClusterer) -> SphncsClusterer:
    copy_model = copy.copy(model)
    copy_model.partitions_ = []
    for partition in model.partitions_:
        copied_partition = copy.copy(partition)
        copied_fastmap = copy.copy(partition.fastmap)
        copied_fastmap.models_ = []
        copied_partition.fastmap = copied_fastmap
        copy_model.partitions_.append(copied_partition)
    return copy_model


def _save_fastmaps(model: SphncsClusterer, staging: Path) -> list[list[str]]:
    entries: list[list[str]] = []
    for partition_index, partition in enumerate(model.partitions_):
        paths = partition.fastmap.save_models(staging / "fastmaps" / str(partition_index))
        entries.append([str(path.relative_to(staging)) for path in paths])
    return entries


def _read_manifest(archive: zipfile.ZipFile) -> dict:
    if archive.namelist().count(_MANIFEST_NAME) != 1:
        raise ModelPersistenceError("model archive has an invalid manifest")
    try:
        manifest = json.loads(archive.read(_MANIFEST_NAME))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelPersistenceError("model archive has an invalid manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != PERSISTENCE_FORMAT:
        raise ModelPersistenceError("model archive is not a sphncs model")
    if manifest.get("format_version") != PERSISTENCE_VERSION:
        raise ModelPersistenceError(f"unsupported model format version {manifest.get('format_version')!r}")
    if manifest.get("estimator") != _ESTIMATOR_NAME or not isinstance(manifest.get("model_class"), str):
        raise ModelPersistenceError("model archive has an invalid manifest")
    fastmaps = manifest.get("fastmaps")
    if not isinstance(fastmaps, list) or not all(isinstance(group, list) for group in fastmaps):
        raise ModelPersistenceError("model archive has invalid FastMap entries")
    names = [name for group in fastmaps for name in group]
    if not all(isinstance(name, str) and name.startswith("fastmaps/") for name in names) or len(set(names)) != len(names):
        raise ModelPersistenceError("model archive has invalid FastMap entries")
    checksums = manifest.get("sha256")
    required = {_ESTIMATOR_NAME, *names}
    if not isinstance(checksums, dict) or set(checksums) != required or not all(_is_checksum(checksums[name]) for name in required):
        raise ModelPersistenceError("model archive has invalid checksums")
    return manifest


def _checksum(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_checksum(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
