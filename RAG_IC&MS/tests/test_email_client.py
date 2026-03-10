import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from email.mime.text import MIMEText

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.integrations.email_client import (
    extract_email_body,
    mark_as_processed,
    check_trigger_folder,
    process_trigger_folder,
    start_email_monitor,
    stop_email_monitor,
    is_monitor_running,
    IMAPConfigError,
)


@pytest.fixture
def mock_imap_settings():
    mock = MagicMock()
    mock.imap_server = "mail.test.com"
    mock.imap_user = "test@test.com"
    mock.imap_password = "testpass"
    mock.imap_trigger_folder = "Generate Draft"
    mock.openai_api_key = "test"
    mock.openai_embedding_model = "text-embedding-3-large"
    mock.openai_chat_model = "gpt-4-turbo"
    mock.pinecone_api_key = "test"
    mock.pinecone_index_name = "test"
    mock.retrieval_top_k = 10
    mock.chunk_size = 512
    mock.chunk_overlap = 50
    mock.log_level = "WARNING"
    mock.zendesk_subdomain = None
    mock.zendesk_email = None
    mock.zendesk_api_token = None
    return mock


def _make_email_bytes(subject="Test Subject", sender="customer@test.com", body="Hello, I need help"):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = "support@test.com"
    return msg.as_bytes()


def test_extract_email_body_plain_text():
    raw = _make_email_bytes(body="Please help me with installation")
    body = extract_email_body(raw)
    assert "Please help me with installation" in body


def test_extract_email_body_empty():
    raw = _make_email_bytes(body="")
    body = extract_email_body(raw)
    assert body == ""


def test_imap_config_error_when_not_configured():
    mock = MagicMock()
    mock.imap_server = None
    mock.imap_user = None
    mock.imap_password = None
    with patch("src.integrations.email_client.get_settings", return_value=mock):
        with pytest.raises(IMAPConfigError):
            from src.integrations.email_client import connect_imap
            connect_imap()


def test_mark_as_processed():
    mock_mail = MagicMock()
    mark_as_processed(mock_mail, "1")
    mock_mail.store.assert_called_once_with(b"1", "+FLAGS", "\\Seen")


@patch("src.integrations.email_client.connect_imap")
def test_check_trigger_folder(mock_connect, mock_imap_settings):
    raw_email = _make_email_bytes(subject="Help needed", sender="user@test.com", body="I need help")

    mock_mail = MagicMock()
    mock_connect.return_value = mock_mail
    mock_mail.select.return_value = ("OK", [b"1"])
    mock_mail.search.return_value = ("OK", [b"1"])
    mock_mail.fetch.return_value = ("OK", [(b"1", raw_email)])

    with patch("src.integrations.email_client.get_settings", return_value=mock_imap_settings):
        results = check_trigger_folder()

    assert len(results) == 1
    assert results[0]["subject"] == "Help needed"
    assert results[0]["sender"] == "user@test.com"
    assert "I need help" in results[0]["body"]


@patch("src.integrations.email_client.connect_imap")
@patch("src.integrations.email_client.process_customer_query")
def test_process_trigger_folder(mock_query, mock_connect, mock_imap_settings):
    raw_email = _make_email_bytes(subject="Install question", sender="user@test.com", body="How to install?")

    mock_mail = MagicMock()
    mock_connect.return_value = mock_mail
    mock_mail.select.return_value = ("OK", [b"1"])
    mock_mail.search.return_value = ("OK", [b"1"])
    mock_mail.fetch.return_value = ("OK", [(b"1", raw_email)])

    mock_query.return_value = {
        "draft": "Here's how to install...",
        "citations": ["[1] guide.pdf"],
        "escalation": {"needs_escalation": False, "reason": ""},
        "detected_language": "en",
        "complexity": "moderate",
    }

    with patch("src.integrations.email_client.get_settings", return_value=mock_imap_settings):
        results = process_trigger_folder()

    assert len(results) == 1
    assert results[0]["subject"] == "Install question"
    assert results[0]["result"]["draft"] == "Here's how to install..."
    # Verify draft was saved and email marked as processed
    mock_mail.append.assert_called_once()
    mock_mail.store.assert_called_once()


@patch("src.integrations.email_client.connect_imap")
def test_process_trigger_folder_empty(mock_connect, mock_imap_settings):
    mock_mail = MagicMock()
    mock_connect.return_value = mock_mail
    mock_mail.select.return_value = ("OK", [b"0"])
    mock_mail.search.return_value = ("OK", [b""])

    with patch("src.integrations.email_client.get_settings", return_value=mock_imap_settings):
        results = process_trigger_folder()

    assert results == []


def test_monitor_lifecycle(mock_imap_settings):
    """Test start/stop of email monitor."""
    with patch("src.integrations.email_client.get_settings", return_value=mock_imap_settings):
        with patch("src.integrations.email_client.process_trigger_folder", return_value=[]):
            assert not is_monitor_running()

            start_email_monitor(poll_interval=1)
            assert is_monitor_running()

            stop_email_monitor()
            import time
            time.sleep(0.5)
            assert not is_monitor_running()
