"""Shared Pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST_PATH = PROJECT_ROOT / "sample_request.json"


@pytest.fixture
def valid_payload() -> dict:
    """Return the repository's valid prediction request."""

    return json.loads(SAMPLE_REQUEST_PATH.read_text(encoding="utf-8"))
