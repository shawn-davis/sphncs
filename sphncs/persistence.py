"""Versioned, integrity-checked persistence for fitted sphncs estimators.

The archive contains a small JSON manifest and one Python pickle payload.  Pickle
is required because sphncs deliberately accepts arbitrary Python objects and
distance callables.  Consequently, archives must only be loaded from trusted
sources.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import TYPE_CHECKING, TypeVar
import zipfile

if TYPE_CHECKING:
    from .estimator import SphncsClusterer


PERSISTENCE_FORMAT = "sphncs-model"
PERSISTENCE_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_PAYLOAD_NAME = "model.pkl"
_T = TypeVar("_T", bound="SphncsClusterer")


class ModelPersistenceError(ValueError):
    """Raised when a model archive is malformed, incompatible, or unsafe to use."""


def save_model(model: "SphncsClusterer", path: str | os.PathLike[str]) -> Path:
    """Atomically save a fitted model to a versioned ``.sphncs`` archive.

    The archive can contain Python objects and callables supplied to the
    estimator.  Treat it like a pickle file: only load archives you trust.
    """
    model._require_fitted()
    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"model path is a directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ModelPersistenceError(
            "model is not serializable; metrics, transformers, and partitioning "
            "features must be importable functions or otherwise pickleable objects"
        ) from exc

    manifest = {
        "format": PERSISTENCE_FORMAT,
        "format_version": PERSISTENCE_VERSION,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "payload": _PAYLOAD_NAME,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as temporary:
            temporary_name = temporary.name
        with zipfile.ZipFile(temporary_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST_NAME, json.dumps(manifest, sort_keys=True, separators=(",", ":")))
            archive.writestr(_PAYLOAD_NAME, payload)
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
    """Load and validate a trusted model archive created by :func:`save_model`."""
    source = Path(path)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            members = archive.namelist()
            if (
                members.count(_MANIFEST_NAME) != 1
                or members.count(_PAYLOAD_NAME) != 1
                or len(members) != 2
            ):
                raise ModelPersistenceError("model archive has unexpected contents")
            try:
                manifest = json.loads(archive.read(_MANIFEST_NAME))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ModelPersistenceError("model archive has an invalid manifest") from exc
            _validate_manifest(manifest)
            payload = archive.read(_PAYLOAD_NAME)
    except FileNotFoundError:
        raise
    except zipfile.BadZipFile as exc:
        raise ModelPersistenceError("model archive is not a valid ZIP file") from exc
    except OSError as exc:
        raise ModelPersistenceError(f"could not read model archive {source}: {exc}") from exc

    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ModelPersistenceError("model archive failed its integrity check")
    try:
        model = pickle.loads(payload)
    except (pickle.UnpicklingError, AttributeError, EOFError, ImportError, IndexError, TypeError) as exc:
        raise ModelPersistenceError("model archive could not be deserialized") from exc
    if not isinstance(model, expected_type):
        raise ModelPersistenceError(
            f"model archive contains {type(model).__name__}, not {expected_type.__name__}"
        )
    if f"{type(model).__module__}.{type(model).__qualname__}" != manifest["model_class"]:
        raise ModelPersistenceError("model archive class does not match its manifest")
    model._require_fitted()
    return model


def _validate_manifest(manifest: object) -> None:
    if not isinstance(manifest, dict):
        raise ModelPersistenceError("model archive has an invalid manifest")
    if manifest.get("format") != PERSISTENCE_FORMAT:
        raise ModelPersistenceError("model archive is not a sphncs model")
    if manifest.get("format_version") != PERSISTENCE_VERSION:
        raise ModelPersistenceError(
            f"unsupported model format version {manifest.get('format_version')!r}"
        )
    if manifest.get("payload") != _PAYLOAD_NAME:
        raise ModelPersistenceError("model archive has an invalid payload declaration")
    if not isinstance(manifest.get("model_class"), str):
        raise ModelPersistenceError("model archive has an invalid model class")
    checksum = manifest.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ModelPersistenceError("model archive has an invalid checksum")
    try:
        int(checksum, 16)
    except ValueError as exc:
        raise ModelPersistenceError("model archive has an invalid checksum") from exc
