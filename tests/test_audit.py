import sys
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.audit.logger import log_query_event, get_audit_log


@pytest.fixture(autouse=True)
def temp_audit_file(tmp_path):
    """Redirect the audit log file to a temp directory for each test."""
    temp_file = tmp_path / "audit_log.jsonl"
    with patch("src.audit.logger._AUDIT_LOG_FILE", temp_file):
        yield temp_file


def _make_result(**overrides):
    base = {
        "draft": "Test draft response.",
        "citations": ["[1] doc.pdf"],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "brief",
    }
    base.update(overrides)
    return base


def test_log_query_event_writes_record(temp_audit_file):
    """log_query_event writes a valid JSON record to the audit log."""
    result = _make_result()
    record = log_query_event("How to install?", result, source_channel="manual", duration_ms=150)

    assert record["query"] == "How to install?"
    assert record["detected_language"] == "en"
    assert record["source_channel"] == "manual"
    assert record["duration_ms"] == 150
    assert record["draft_length"] == len("Test draft response.")
    assert record["citation_count"] == 1
    assert record["needs_escalation"] is False

    # Verify file
    lines = temp_audit_file.read_text().strip().split("\n")
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["query"] == "How to install?"


def test_log_multiple_events(temp_audit_file):
    """Multiple log events append to the same file."""
    log_query_event("q1", _make_result(), "manual")
    log_query_event("q2", _make_result(), "zendesk")
    log_query_event("q3", _make_result(), "email")

    lines = temp_audit_file.read_text().strip().split("\n")
    assert len(lines) == 3


def test_log_escalation_fields(temp_audit_file):
    """Escalation info is captured correctly."""
    result = _make_result(escalation={"needs_escalation": True, "reason": "Billing dispute"})
    record = log_query_event("refund please", result, "zendesk")

    assert record["needs_escalation"] is True
    assert record["escalation_reason"] == "Billing dispute"


def test_get_audit_log_empty(temp_audit_file):
    """Returns empty list when no audit entries exist."""
    result = get_audit_log()
    assert result == []


def test_get_audit_log_recent_first(temp_audit_file):
    """Returns most recent entries first."""
    log_query_event("first", _make_result(), "manual")
    log_query_event("second", _make_result(), "manual")

    entries = get_audit_log()
    assert len(entries) == 2
    assert entries[0]["query"] == "second"
    assert entries[1]["query"] == "first"


def test_get_audit_log_limit(temp_audit_file):
    """Limit parameter caps returned entries."""
    for i in range(5):
        log_query_event(f"q{i}", _make_result(), "manual")

    entries = get_audit_log(limit=2)
    assert len(entries) == 2


def test_get_audit_log_offset(temp_audit_file):
    """Offset parameter skips entries."""
    for i in range(5):
        log_query_event(f"q{i}", _make_result(), "manual")

    entries = get_audit_log(limit=2, offset=2)
    assert len(entries) == 2
    # Most recent first, then offset by 2
    assert entries[0]["query"] == "q2"
    assert entries[1]["query"] == "q1"
