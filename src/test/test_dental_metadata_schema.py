from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dental.metadata_schema import apply_dental_defaults, validate_dental_metadata


def _make_base_record() -> dict[str, object]:
    return {
        "doc_id": "health-info/oral-hygiene",
        "title": "Oral Hygiene",
        "source_url": "https://www.nidcr.nih.gov/health-info/oral-hygiene",
        "content_type": "html_article",
        "retrieved_at": "2025-11-03T10:00:00Z",
    }


def test_apply_dental_defaults_injects_optional_defaults() -> None:
    record = _make_base_record()
    defaults_applied = apply_dental_defaults(record)

    assert defaults_applied["publisher"] == "National Institute of Dental and Craniofacial Research"
    assert defaults_applied["license"] == "Public domain (US federal government)"
    assert defaults_applied["language"] == "en"


def test_validate_dental_metadata_happy_path_returns_normalized_record() -> None:
    record = _make_base_record()

    normalized = validate_dental_metadata(record)

    assert normalized["doc_id"] == "health-info/oral-hygiene"
    assert normalized["publisher"] == "National Institute of Dental and Craniofacial Research"
    assert normalized["license"] == "Public domain (US federal government)"
    assert normalized["language"] == "en"
    assert normalized["retrieved_at"] == "2025-11-03T10:00:00Z"


def test_validate_dental_metadata_missing_required_field_raises() -> None:
    record = _make_base_record()
    record.pop("title")

    with pytest.raises(ValueError):
        validate_dental_metadata(record)


def test_validate_dental_metadata_rejects_invalid_enum() -> None:
    record = _make_base_record()
    record["content_type"] = "unknown"

    with pytest.raises(ValueError):
        validate_dental_metadata(record)
