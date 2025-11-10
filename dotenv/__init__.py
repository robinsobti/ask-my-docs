from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _parse_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    return key.strip(), value.strip().strip('"').strip("'")


def load_dotenv(dotenv_path: str | None = None, *, override: bool = False) -> bool:
    """
    Minimal drop-in replacement for python-dotenv's load_dotenv.
    Reads KEY=VALUE pairs from the given file (defaults to .env) and
    injects them into os.environ.
    """
    path = Path(dotenv_path or ".env")
    if not path.exists():
        return False

    updated = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = _parse_line(raw_line)
            if not parsed:
                continue
            key, value = parsed
            if key in os.environ and not override:
                continue
            os.environ[key] = value
            updated = True
    return updated
