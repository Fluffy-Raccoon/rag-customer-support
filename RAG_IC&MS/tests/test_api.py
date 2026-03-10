import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


# Mock settings before importing the app
@pytest.fixture(autouse=True)
def mock_settings():
    mock = MagicMock()
    mock.openai_api_key = "test-key"
    mock.openai_embedding_model = "text-embedding-3-large"
    mock.openai_chat_model = "gpt-4-turbo"
    mock.pinecone_api_key = "test-key"
    mock.pinecone_index_name = "test-index"
    mock.log_level = "WARNING"
    mock.chunk_size = 512
    mock.chunk_overlap = 50
    mock.retrieval_top_k = 10
    mock.zendesk_subdomain = None
    mock.zendesk_email = None
    mock.zendesk_api_token = None
    mock.imap_server = None
    mock.imap_user = None
    mock.imap_password = None
    mock.imap_trigger_folder = "Generate Draft"
    mock.api_key = None
    with patch("config.settings.get_settings", return_value=mock):
        with patch("config.get_settings", return_value=mock):
            yield mock


@pytest.fixture
def client(mock_settings):
    from src.api.main import app
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_empty_body(client):
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 400


def test_query_endpoint(client):
    mock_result = {
        "draft": "Here is your answer.",
        "citations": ["[1] manual.pdf - Section 1"],
        "escalation": {"needs_escalation": False, "reason": "No escalation needed"},
        "detected_language": "en",
        "complexity": "brief",
    }
    with patch("src.api.main.process_customer_query", return_value=mock_result):
        response = client.post("/query", json={"query": "How do I reset my password?"})
    assert response.status_code == 200
    data = response.json()
    assert data["draft"] == "Here is your answer."
    assert data["detected_language"] == "en"
    assert data["escalation"]["needs_escalation"] is False
    assert data["complexity"] == "brief"


def test_zendesk_endpoint_not_configured(client):
    """Returns 400 when Zendesk is not configured."""
    response = client.post("/zendesk/generate-draft", json={"ticket_id": 123})
    assert response.status_code == 400
    assert "not configured" in response.json()["detail"].lower()


def test_email_process_not_configured(client):
    """Returns 400 when IMAP is not configured."""
    response = client.post("/email/process")
    assert response.status_code == 400
    assert "not configured" in response.json()["detail"].lower()


def test_email_start_monitor_not_configured(client):
    """Returns 400 when IMAP is not configured."""
    response = client.post("/email/start-monitor", json={"poll_interval": 30})
    assert response.status_code == 400


def test_zendesk_endpoint_with_mock(client):
    """Returns draft when Zendesk is configured and mocked."""
    mock_result = {
        "draft": "Zendesk draft",
        "citations": ["[1] doc.pdf"],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "brief",
    }
    with patch("src.api.main.zendesk_generate_draft", return_value=mock_result):
        response = client.post("/zendesk/generate-draft", json={"ticket_id": 456})
    assert response.status_code == 200
    assert response.json()["draft"] == "Zendesk draft"
    assert response.json()["ticket_id"] == 456


def test_email_process_with_mock(client):
    """Returns processed count when IMAP is configured and mocked."""
    mock_results = [
        {"msg_id": "1", "subject": "Help", "sender": "a@b.com", "result": {"detected_language": "en"}},
    ]
    with patch("src.api.main.process_trigger_folder", return_value=mock_results):
        response = client.post("/email/process")
    assert response.status_code == 200
    assert response.json()["processed"] == 1


def test_email_stop_monitor_when_not_running(client):
    with patch("src.api.main.is_monitor_running", return_value=False):
        response = client.post("/email/stop-monitor")
    assert response.status_code == 200
    assert response.json()["status"] == "not_running"


def test_feedback_approve_endpoint(client):
    """Captures an approved response and returns its ID."""
    mock_record = {"id": "abc123", "original_query": "test"}
    with patch("src.api.main.capture_approved_response", return_value=mock_record):
        response = client.post("/feedback/approve", json={
            "original_query": "How to install?",
            "draft_response": "Draft text",
            "final_response": "Final text",
            "agent_edits": "minor tweak",
        })
    assert response.status_code == 200
    assert response.json()["status"] == "captured"
    assert response.json()["id"] == "abc123"


def test_feedback_list_endpoint(client):
    """Returns approved responses list."""
    mock_records = [{"id": "abc", "original_query": "test"}]
    with patch("src.api.main.get_approved_responses", return_value=mock_records):
        response = client.get("/feedback/approved")
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_audit_log_endpoint(client):
    """Returns audit log entries."""
    mock_entries = [{"timestamp": "2026-01-01T00:00:00", "query": "test"}]
    with patch("src.api.main.get_audit_log", return_value=mock_entries):
        response = client.get("/audit/log")
    assert response.status_code == 200
    assert response.json()["count"] == 1


def test_api_key_auth_rejects_wrong_key(mock_settings):
    """Returns 401 when API key is configured but wrong key provided."""
    mock_settings.api_key = "correct-key"
    from src.api.main import app
    auth_client = TestClient(app)
    mock_result = {
        "draft": "answer",
        "citations": [],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "brief",
    }
    with patch("src.api.main.get_settings", return_value=mock_settings):
        with patch("src.api.main.process_customer_query", return_value=mock_result):
            response = auth_client.post(
                "/query",
                json={"query": "test"},
                headers={"X-API-Key": "wrong-key"},
            )
    assert response.status_code == 401


def test_api_key_auth_accepts_correct_key(mock_settings):
    """Returns 200 when correct API key is provided."""
    mock_settings.api_key = "correct-key"
    from src.api.main import app
    auth_client = TestClient(app)
    mock_result = {
        "draft": "answer",
        "citations": [],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "brief",
    }
    with patch("src.api.main.get_settings", return_value=mock_settings):
        with patch("src.api.main.process_customer_query", return_value=mock_result):
            response = auth_client.post(
                "/query",
                json={"query": "test"},
                headers={"X-API-Key": "correct-key"},
            )
    assert response.status_code == 200


def test_health_no_auth_required(mock_settings):
    """Health endpoint works even with API key configured."""
    mock_settings.api_key = "some-key"
    from src.api.main import app
    auth_client = TestClient(app)
    response = auth_client.get("/health")
    assert response.status_code == 200
