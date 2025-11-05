"""Convenience helpers for accessing the dental metadata schema."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from src.dental.metadata_schema import apply_dental_defaults, load_dental_metadata_schema, validate_dental_metadata


@lru_cache(maxsize=1)
def get_dental_schema() -> Dict[str, Any]:
    """Return the dental metadata schema loaded from ``configs/dental_metadata.yaml``.

    The result is cached so repeated calls avoid hitting the filesystem on every document.
    """

    return load_dental_metadata_schema()


def normalize_document_metadata(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """Apply defaults and schema validation to a single dental metadata record.

    Parameters
    ----------
    raw_record:
        A mapping that should include the required dental schema fields such as
        ``doc_id``, ``title``, ``source_url``, ``content_type``, and ``retrieved_at``.

    Returns
    -------
    Dict[str, Any]
        A normalized copy of the metadata with defaults applied and fields validated.
    """

    defaults_applied = apply_dental_defaults(raw_record)
    validated = validate_dental_metadata(defaults_applied)
    return validated
