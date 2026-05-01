"""Thread-safe lazy load of the trained joblib bundle (invalidates on file mtime change)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import joblib


class BundleStore:
    __slots__ = ("_path", "_lock", "_cache", "_mtime")

    def __init__(self, artifact_path: Path) -> None:
        self._path = artifact_path
        self._lock = threading.Lock()
        self._cache: dict[str, Any] | None = None
        self._mtime: float | None = None

    @property
    def path(self) -> Path:
        return self._path

    def get(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._path.is_file():
                self._cache = None
                self._mtime = None
                return None
            mt = self._path.stat().st_mtime
            if self._cache is not None and self._mtime == mt:
                return self._cache
            self._cache = joblib.load(self._path)
            self._mtime = mt
            return self._cache
