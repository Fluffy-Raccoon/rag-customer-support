import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.parsers.zendesk_parser import parse_zendesk_csv


def test_parse_zendesk_csv():
    """Zendesk CSV parser produces Q&A pair documents."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "subject", "description", "agent_response", "status"])
        writer.writerow(["1", "Reset password", "I need to reset my password", "Go to settings > reset", "solved"])
        writer.writerow(["2", "Billing issue", "Charged twice", "We will refund", "solved"])
        writer.writerow(["3", "No response", "Hello", "", "open"])  # no agent response
        tmp_path = f.name

    try:
        docs = parse_zendesk_csv(tmp_path)
        assert len(docs) == 2  # third row skipped (no agent response)
        assert "Reset password" in docs[0]
        assert "Go to settings > reset" in docs[0]
        assert "Billing issue" in docs[1]
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_parse_zendesk_csv_missing_columns():
    """Raises ValueError if required columns are missing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticket_id", "subject"])
        writer.writerow(["1", "Test"])
        tmp_path = f.name

    try:
        import pytest
        with pytest.raises(ValueError, match="missing required columns"):
            parse_zendesk_csv(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
