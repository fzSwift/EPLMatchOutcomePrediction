"""Environment-driven settings (single place for paths and CORS)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _parse_origins(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return [
            "http://127.0.0.1:8000",
            "http://localhost:8000",
            "http://127.0.0.1:8501",
            "http://localhost:8501",
        ]
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    web_dir: Path
    artifact_path: Path
    cors_origins: list[str]
    cors_wildcard: bool
    env: str

    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    artifact = Path(
        os.environ.get("EPL_ARTIFACT_PATH", str(root / "artifacts" / "bundle.joblib"))
    )
    wildcard = os.environ.get("EPL_CORS_WILDCARD", "").lower() in ("1", "true", "yes")
    origins = (
        ["*"]
        if wildcard
        else _parse_origins(os.environ.get("EPL_CORS_ORIGINS"))
    )
    return Settings(
        project_root=root,
        web_dir=root / "web",
        artifact_path=artifact,
        cors_origins=origins,
        cors_wildcard=wildcard,
        env=os.environ.get("EPL_ENV", "development"),
    )
