"""Crash-safe atomic disk persistence for snapshots and static historical SQL bundles."""
from __future__ import annotations

import io
import logging
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

from .state import RecommenderSnapshotV4, StaticHistoryBundle

logger = logging.getLogger(__name__)


class _CrossPackageUnpickler(pickle.Unpickler):
    """Resolve classes seamlessly whether saved under `src.*` or `etapa_4.src.*`."""

    def find_class(self, module: str, name: str) -> Any:
        target_pkg = __package__ or "src"
        if module.startswith("etapa_4.src.") and target_pkg == "src":
            module = "src." + module[len("etapa_4.src.") :]
        elif module.startswith("src.") and target_pkg == "etapa_4.src":
            module = "etapa_4.src." + module[len("src.") :]
        return super().find_class(module, name)


def _atomic_dump(path: Path, obj: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        last_perm_err: OSError | None = None
        for attempt in range(12):
            try:
                os.replace(tmp_name, path)
                return True
            except PermissionError as exc:
                last_perm_err = exc
                time.sleep(0.04 * (attempt + 1))
        logger.warning("Skipping disk cache replace on %s (file locked, e.g. OneDrive): %s", path.name, last_perm_err)
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return False
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _safe_load(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        raw = path.read_bytes()
        return _CrossPackageUnpickler(io.BytesIO(raw)).load()
    except Exception as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path.name, exc)
        return None


class DiskCache:
    """Manages atomic snapshot and static history persistence under etapa_4/cache/."""

    SNAPSHOT_FILENAME = "snapshot_v4_latest.pkl"
    HISTORY_FILENAME = "static_history_v4.pkl"

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @property
    def snapshot_path(self) -> Path:
        return self.cache_dir / self.SNAPSHOT_FILENAME

    @property
    def history_path(self) -> Path:
        return self.cache_dir / self.HISTORY_FILENAME

    def save_snapshot(self, snapshot: RecommenderSnapshotV4) -> Path:
        _atomic_dump(self.snapshot_path, snapshot)
        return self.snapshot_path

    def load_snapshot(self) -> RecommenderSnapshotV4 | None:
        obj = _safe_load(self.snapshot_path)
        return obj if isinstance(obj, RecommenderSnapshotV4) else None

    def save_static_history(self, bundle: StaticHistoryBundle) -> Path:
        _atomic_dump(self.history_path, bundle)
        return self.history_path

    def load_static_history(self) -> StaticHistoryBundle | None:
        obj = _safe_load(self.history_path)
        return obj if isinstance(obj, StaticHistoryBundle) else None
