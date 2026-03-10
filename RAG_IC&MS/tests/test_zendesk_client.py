import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.integrations.zendesk_client import (
    format_internal_note,
    get_latest_customer_message,
    _get_zendesk_config,
    zendesk_generate_draft,
    ZendeskConfigError,
)


@pytest.fixture
def mock_zendesk_settings():
    mock = MagicMock()
    mock.zendesk_subdomain = "testcompany"
    mock.zendesk_email = "support@testcompany.com"
    mock.zendesk_api_token = "test-token-123"
    mock.openai_api_key = "test"
    mock.openai_embedding_model = "text-embedding-3-large"
    mock.openai_chat_model = "gpt-4-turbo"
    mock.pinecone_api_key = "test"
    mock.pinecone_index_name = "test"
    mock.retrieval_top_k = 10
    mock.chunk_size = 512
    mock.chunk_overlap = 50
    mock.log_level = "WARNING"
    return mock


def test_zendesk_config_error_when_not_configured():
    """Raises ZendeskConfigError when credentials are missing."""
    mock = MagicMock()
    mock.zendesk_subdomain = None
    mock.zendesk_email = None
    mock.zendesk_api_token = None
    with patch("src.integrations.zendesk_client.get_settings", return_value=mock):
        with pytest.raises(ZendeskConfigError):
            _get_zendesk_config()


def test_zendesk_config_returns_correct_base_url(mock_zendesk_settings):
    with patch("src.integrations.zendesk_client.get_settings", return_value=mock_zendesk_settings):
        base_url, headers = _get_zendesk_config()
        assert base_url == "https://testcompany.zendesk.com"
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Basic ")


def test_format_internal_note_no_escalation():
    result = {
        "draft": "Here is your answer.",
        "citations": ["[1] manual.pdf - Setup"],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "moderate",
    }
    note = format_internal_note(result)
    assert "AI-GENERATED DRAFT" in note
    assert "Here is your answer." in note
    assert "[1] manual.pdf - Setup" in note
    assert "No escalation needed" in note
    assert "Language detected: EN" in note
    assert "Complexity: moderate" in note


def test_format_internal_note_with_escalation():
    result = {
        "draft": "Draft text",
        "citations": [],
        "escalation": {"needs_escalation": True, "reason": "Billing dispute"},
        "detected_language": "de",
        "complexity": "brief",
    }
    note = format_internal_note(result)
    assert "ESCALATE" in note
    assert "Billing dispute" in note
    assert "DE" in note


@patch("src.integrations.zendesk_client.get_ticket_comments")
def test_get_latest_customer_message_filters_enduser(mock_comments, mock_zendesk_settings):
    mock_comments.return_value = [
        {"body": "Customer question", "author": {"role": "end-user"}, "via": {"channel": "email"}},
        {"body": "Agent reply", "author": {"role": "agent"}, "via": {"channel": "web"}},
        {"body": "Customer follow-up", "author": {"role": "end-user"}, "via": {"channel": "email"}},
    ]
    with patch("src.integrations.zendesk_client.get_settings", return_value=mock_zendesk_settings):
        msg = get_latest_customer_message(123)
    assert msg == "Customer follow-up"


@patch("src.integrations.zendesk_client.get_ticket_comments")
def test_get_latest_customer_message_fallback_to_first(mock_comments, mock_zendesk_settings):
    """Falls back to first comment if no end-user comments found."""
    mock_comments.return_value = [
        {"body": "Initial description", "author": {"role": "admin"}, "via": {"channel": "web"}},
    ]
    with patch("src.integrations.zendesk_client.get_settings", return_value=mock_zendesk_settings):
        msg = get_latest_customer_message(123)
    assert msg == "Initial description"


@patch("src.integrations.zendesk_client.post_internal_note")
@patch("src.integrations.zendesk_client.process_customer_query")
@patch("src.integrations.zendesk_client.get_latest_customer_message")
def test_zendesk_generate_draft_full_flow(mock_get_msg, mock_query, mock_post, mock_zendesk_settings):
    mock_get_msg.return_value = "How do I reset my password?"
    mock_query.return_value = {
        "draft": "Go to settings...",
        "citations": ["[1] faq.pdf"],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "brief",
    }
    mock_post.return_value = {"ticket": {"id": 123}}

    with patch("src.integrations.zendesk_client.get_settings", return_value=mock_zendesk_settings):
        result = zendesk_generate_draft(123)

    mock_get_msg.assert_called_once_with(123)
    mock_query.assert_called_once_with("How do I reset my password?")
    mock_post.assert_called_once()
    assert result["draft"] == "Go to settings..."


@patch("src.integrations.zendesk_client.httpx")
def test_zendesk_api_error_handling(mock_httpx, mock_zendesk_settings):
    """HTTP errors from Zendesk API are raised."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_httpx.get.return_value = mock_response

    with patch("src.integrations.zendesk_client.get_settings", return_value=mock_zendesk_settings):
        with pytest.raises(Exception, match="404"):
            from src.integrations.zendesk_client import get_ticket
            get_ticket(99999)
