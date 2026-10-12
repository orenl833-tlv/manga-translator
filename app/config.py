"""
Central config, read from environment variables (see .env.example).

Kept deliberately simple — this is a single-user, self-hosted tool, not a
multi-tenant service, so there's no settings database, no per-user config,
just process env vars.
"""
from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so `python -m app.main` works without extra deps.

    Only sets a variable if it isn't already set in the real environment,
    so real env vars (e.g. from Docker) always win over the .env file.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings:
    # Which LLM does the translation step — "anthropic" or "openai". See
    # app/pipeline/translate.py for how this is dispatched.
    translation_provider: str = os.environ.get("TRANSLATION_PROVIDER", "anthropic")

    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    anthropic_effort: str = os.environ.get("ANTHROPIC_EFFORT", "medium")

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    openai_model: str = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    data_dir: Path = Path(os.environ.get("DATA_DIR", "./data")).resolve()

    max_image_long_edge: int = int(os.environ.get("MAX_IMAGE_LONG_EDGE", "1600"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
