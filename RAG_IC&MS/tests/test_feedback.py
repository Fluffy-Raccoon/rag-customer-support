import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.feedback.learning_loop import (
    capture_approved_response,
    embed_approved_response,
    get_approved_responses,
    _APPROVED_RESPONSES_FILE,
)


@pytest.fixture(autouse=True)
def temp_data_dir(tmp_path):
    """Redirect the JSONL file to a temp directory for each test."""
    temp_file = tmp_path / "approved_responses.jsonl"
    with patch("src.feedback.learning_loop._APPROVED_RESPONSES_FILE", temp_file):
        yield temp_file


@pytest.fixture
def mock_embedder():
    with patch("src.feedback.learning_loop.get_embeddings") as mock:
        mock.return_value = [[0.1] * 3072]
        yield mock


@pytest.fixture
def mock_upsert():
    with patch("src.feedback.learning_loop.upsert_vectors") as mock:
        yield mock


def test_capture_writes_jsonl(temp_data_dir, mock_embedder, mock_upsert):
    """capture_approved_response writes a valid JSONL record."""
    record = capture_approved_response(
        original_query="How to install?",
        draft_response="Draft here",
        final_response="Final answer here",
        agent_edits="minor fix",
        ticket_id=42,
    )

    assert record["original_query"] == "How to install?"
    assert record["final_response"] == "Final answer here"
    assert record["ticket_id"] == 42
    assert "id" in record
    assert "approved_at" in record

    # Verify file contents
    lines = temp_data_dir.read_text().strip().split("\n")
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["original_query"] == "How to install?"


def test_capture_appends_multiple(temp_data_dir, mock_embedder, mock_upsert):
    """Multiple captures append to the same file."""
    capture_approved_response("q1", "d1", "f1")
    capture_approved_response("q2", "d2", "f2")

    lines = temp_data_dir.read_text().strip().split("\n")
    assert len(lines) == 2


def test_embed_calls_pinecone_with_correct_metadata(mock_embedder, mock_upsert):
    """embed_approved_response upserts with source_type='approved_response'."""
    record = {
        "id": "test123",
        "original_query": "How to reset?",
        "final_response": "Go to settings...",
        "approved_at": "2026-01-01T00:00:00",
        "ticket_id": None,
    }

    embed_approved_response(record)

    mock_embedder.assert_called_once()
    mock_upsert.assert_called_once()

    # Check metadata
    call_args = mock_upsert.call_args
    ids = call_args[0][0]
    metadatas = call_args[0][2]
    assert ids[0] == "approved_test123"
    assert metadatas[0]["source_type"] == "approved_response"
    assert metadatas[0]["approval_date"] == "2026-01-01T00:00:00"


def test_get_approved_responses_empty(temp_data_dir):
    """Returns empty list when no approved responses exist."""
    result = get_approved_responses()
    assert result == []


def test_get_approved_responses_returns_recent_first(temp_data_dir, mock_embedder, mock_upsert):
    """Returns most recent approved responses first."""
    capture_approved_response("first", "d1", "f1")
    capture_approved_response("second", "d2", "f2")

    results = get_approved_responses(limit=10)
    assert len(results) == 2
    assert results[0]["original_query"] == "second"
    assert results[1]["original_query"] == "first"


def test_get_approved_responses_respects_limit(temp_data_dir, mock_embedder, mock_upsert):
    """Limit parameter caps number of returned records."""
    for i in range(5):
        capture_approved_response(f"q{i}", f"d{i}", f"f{i}")

    results = get_approved_responses(limit=2)
    assert len(results) == 2


def test_capture_triggers_embedding(temp_data_dir, mock_embedder, mock_upsert):
    """capture_approved_response also embeds the response."""
    capture_approved_response("query", "draft", "final")

    mock_embedder.assert_called_once()
    mock_upsert.assert_called_once()

    # Verify the training text format
    embed_call = mock_embedder.call_args[0][0]
    assert "Customer Question: query" in embed_call[0]
    assert "Approved Response: final" in embed_call[0]
