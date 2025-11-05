"""Helpers for working with the dental corpus metadata schema."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

import yaml

from src.dental import DENTAL_NAMESPACE

SCHEMA_FILENAME = "dental_metadata.yaml"
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs" / SCHEMA_FILENAME

REQUIRED_KEYS = ["schema_name", "namespace", "version", "fields"]
ALLOWED_LANGUAGE_CODES = {"en"}
ALLOWED_URL_SCHEMES = {"http", "https"}
DOC_ID_DISALLOWED_WHITESPACE = {" ", "\t", "\n", "\r"}


def get_schema_path() -> Path:
    """Return the absolute path to the dental metadata schema file."""
    return SCHEMA_PATH


def load_dental_metadata_schema() -> dict[str, Any]:
    """Load and parse the dental metadata schema."""
    schema_path = get_schema_path()
    if not schema_path.exists():
        raise FileNotFoundError(f"Dental metadata schema not found at {schema_path}")

    try:
        raw_yaml = schema_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Failed to read dental metadata schema at {schema_path}") from exc

    try:
        schema = yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML schema at {schema_path}") from exc

    if not isinstance(schema, dict):
        raise TypeError(f"Dental metadata schema at {schema_path} must be a mapping.")

    missing = [key for key in REQUIRED_KEYS if key not in schema]
    if missing:
        raise ValueError(f"Dental metadata schema missing required keys: {', '.join(missing)}")

    if schema["namespace"] != DENTAL_NAMESPACE:
        raise ValueError(
            f"Dental metadata schema namespace mismatch: expected '{DENTAL_NAMESPACE}', found '{schema['namespace']}'."
        )

    fields = schema["fields"]
    if not isinstance(fields, list):
        raise TypeError("Dental metadata schema 'fields' must be a list of field definitions.")

    for idx, field in enumerate(fields):
        if not isinstance(field, dict):
            raise TypeError(f"Dental metadata schema field #{idx} must be a mapping.")
        if "name" not in field or "type" not in field:
            raise ValueError(f"Dental metadata schema field #{idx} must include 'name' and 'type' keys.")

    return schema
    

def apply_dental_defaults(record: Dict[str, Any]) -> Dict[str, Any]:
    """Populate default values defined in the metadata schema onto ``record``.

    Suggested approach:
    1. Load the schema (reuse ``load_dental_metadata_schema``).
    2. Iterate through each field definition and check for ``default`` values.
    3. For optional fields that have defaults, set them when the key is missing or falsy (decide the exact rule and document it).
    4. Return a new dictionary (do not mutate the input in-place) to simplify reasoning in tests.
    """

    schema = load_dental_metadata_schema()
    fields = schema.get("fields", [])

    result = dict(record)
    for field in fields:
        if not isinstance(field, dict):
            continue

        name = field.get("name")
        if not name:
            continue

        if "default" not in field:
            continue

        default_value = field["default"]
        if name not in result or result[name] is None:
            result[name] = deepcopy(default_value)

    return result


def validate_dental_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a metadata payload against the schema definition."""
    if not isinstance(record, dict):
        raise TypeError("Dental metadata record must be a mapping.")

    schema = load_dental_metadata_schema()
    record_with_defaults = apply_dental_defaults(record)

    fields = schema.get("fields", [])
    field_lookup = {field["name"]: field for field in fields if isinstance(field, dict) and field.get("name")}

    normalized: Dict[str, Any] = dict(record_with_defaults)

    for field_name, field_def in field_lookup.items():
        field_type = field_def.get("type", "string")
        required = bool(field_def.get("required", False))
        value = record_with_defaults.get(field_name)

        if value is None:
            if required:
                raise ValueError(f"Dental metadata '{field_name}' is required but missing.")
            continue

        if isinstance(value, str):
            stripped = value.strip()
            if required and not stripped:
                raise ValueError(f"Dental metadata '{field_name}' cannot be blank.")
            value = stripped if stripped != value else value

        if field_type == "string":
            if not isinstance(value, str):
                raise TypeError(f"Dental metadata '{field_name}' must be a string.")
            normalized[field_name] = value
        elif field_type == "url":
            if not isinstance(value, str):
                raise TypeError(f"Dental metadata '{field_name}' must be a string URL.")
            parsed = urlparse(value)
            if parsed.scheme not in ALLOWED_URL_SCHEMES or not parsed.netloc:
                raise ValueError(f"Dental metadata '{field_name}' must be an http(s) URL.")
            if not parsed.netloc.endswith("nidcr.nih.gov"):
                raise ValueError(f"Dental metadata '{field_name}' must originate from nidcr.nih.gov.")
            normalized[field_name] = value
        elif field_type == "enum":
            allowed_values = field_def.get("allowed_values")
            if not isinstance(allowed_values, list) or not allowed_values:
                raise ValueError(f"Dental metadata schema for '{field_name}' is missing allowed values.")
            if value not in allowed_values:
                raise ValueError(f"Dental metadata '{field_name}' must be one of {allowed_values}; received '{value}'.")
            normalized[field_name] = value
        elif field_type == "datetime":
            if isinstance(value, datetime):
                normalized[field_name] = value.isoformat()
            elif isinstance(value, str):
                iso_candidate = value.strip()
                if iso_candidate.endswith("Z"):
                    iso_candidate = iso_candidate[:-1] + "+00:00"
                try:
                    datetime.fromisoformat(iso_candidate)
                except ValueError as exc:
                    raise ValueError(f"Dental metadata '{field_name}' must be ISO-8601 formatted.") from exc
                normalized[field_name] = value.strip()
            else:
                raise TypeError(f"Dental metadata '{field_name}' must be an ISO-8601 datetime string.")
        elif field_type == "list[string]":
            if isinstance(value, tuple):
                value = list(value)
            if not isinstance(value, list):
                raise TypeError(f"Dental metadata '{field_name}' must be a list of strings.")
            cleaned_list = []
            for idx, item in enumerate(value):
                if not isinstance(item, str):
                    raise TypeError(f"Dental metadata '{field_name}' entry #{idx} must be a string.")
                trimmed = item.strip()
                if not trimmed:
                    raise ValueError(f"Dental metadata '{field_name}' entry #{idx} cannot be blank.")
                cleaned_list.append(trimmed)
            normalized[field_name] = cleaned_list
        elif field_type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"Dental metadata '{field_name}' must be an integer.")
            normalized[field_name] = value
        else:
            raise ValueError(f"Dental metadata field '{field_name}' has unsupported type '{field_type}'.")

    doc_id = normalized.get("doc_id")
    if doc_id is not None:
        if any(ws in doc_id for ws in DOC_ID_DISALLOWED_WHITESPACE):
            raise ValueError("Dental metadata 'doc_id' must not contain whitespace.")

    language = normalized.get("language")
    if language is not None:
        if not isinstance(language, str):
            raise TypeError("Dental metadata 'language' must be a string.")
        language_code = language.lower()
        if language_code not in ALLOWED_LANGUAGE_CODES:
            raise ValueError(f"Dental metadata 'language' must be one of {sorted(ALLOWED_LANGUAGE_CODES)}.")
        normalized["language"] = language_code

    page_number = normalized.get("page_number")
    if page_number is not None:
        if page_number <= 0:
            raise ValueError("Dental metadata 'page_number' must be a positive integer.")

    return normalized
