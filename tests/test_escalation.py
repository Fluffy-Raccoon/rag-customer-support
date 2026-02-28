import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.escalation_classifier import (
    check_escalation_need,
    _keyword_pre_screen,
)


def test_keyword_refund():
    matched, reason = _keyword_pre_screen("I want a refund for my order")
    assert matched is True
    assert "refund" in reason.lower()


def test_keyword_data_breach():
    matched, reason = _keyword_pre_screen("There's been a data breach in our system")
    assert matched is True
    assert "data breach" in reason.lower()


def test_keyword_speak_to_manager():
    matched, reason = _keyword_pre_screen("I want to speak to a manager immediately")
    assert matched is True
    assert "manager" in reason.lower()


def test_keyword_billing_dispute():
    matched, reason = _keyword_pre_screen("I have a billing dispute about my invoice")
    assert matched is True


def test_keyword_german_erstattung():
    matched, reason = _keyword_pre_screen("Ich möchte eine Erstattung beantragen")
    assert matched is True


def test_keyword_french_remboursement():
    matched, reason = _keyword_pre_screen("Je voudrais un remboursement")
    assert matched is True


def test_no_keyword_match():
    matched, reason = _keyword_pre_screen("How do I reset my password?")
    assert matched is False
    assert reason == ""


def test_check_escalation_keyword_hit_skips_llm():
    """When keyword matches, LLM should NOT be called."""
    with patch("src.generation.escalation_classifier._llm_classify") as mock_llm:
        result = check_escalation_need(
            query="I want a refund",
            context="some context",
            draft="some draft",
        )
        mock_llm.assert_not_called()
        assert result["needs_escalation"] is True


@patch("src.generation.escalation_classifier.get_settings")
@patch("src.generation.escalation_classifier.OpenAI")
def test_check_escalation_no_keyword_uses_llm(mock_openai_cls, mock_settings):
    """When no keyword match, LLM is called."""
    settings = MagicMock()
    settings.openai_api_key = "test"
    settings.openai_chat_model = "gpt-4-turbo"
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"needs_escalation": false, "reason": "Standard query"}'
    mock_client.chat.completions.create.return_value = mock_response

    result = check_escalation_need(
        query="How do I reset my password?",
        context="password reset docs",
        draft="Go to settings",
    )
    assert result["needs_escalation"] is False
    mock_client.chat.completions.create.assert_called_once()


@patch("src.generation.escalation_classifier.get_settings")
@patch("src.generation.escalation_classifier.OpenAI")
def test_llm_failure_returns_safe_default(mock_openai_cls, mock_settings):
    """If LLM call fails, return safe default (no escalation)."""
    settings = MagicMock()
    settings.openai_api_key = "test"
    settings.openai_chat_model = "gpt-4-turbo"
    mock_settings.return_value = settings

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.side_effect = Exception("API error")

    result = check_escalation_need(
        query="Tell me about your products",
        context="",
        draft="",
    )
    assert result["needs_escalation"] is False
    assert "unavailable" in result["reason"].lower()
