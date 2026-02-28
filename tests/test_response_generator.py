import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.response_generator import (
    process_customer_query,
    _build_context,
    _extract_citations,
    RESPONSE_PROMPT,
)
from src.retrieval.pinecone_client import SearchResult


@patch("src.generation.response_generator.get_settings")
@patch("src.generation.response_generator.get_embeddings")
@patch("src.generation.response_generator.pinecone_query")
@patch("src.generation.response_generator.check_escalation_need")
@patch("src.generation.response_generator.detect_language")
@patch("src.generation.response_generator.OpenAI")
def test_process_customer_query_returns_all_fields(
    mock_openai_cls, mock_detect, mock_escalation, mock_pinecone, mock_embed, mock_settings
):
    """process_customer_query returns draft, citations, escalation, language, and complexity."""
    # Setup mocks
    settings = MagicMock()
    settings.openai_api_key = "test"
    settings.openai_chat_model = "gpt-4-turbo"
    settings.retrieval_top_k = 10
    mock_settings.return_value = settings

    mock_detect.return_value = "de"
    mock_embed.return_value = [[0.1] * 3072]

    mock_pinecone.return_value = [
        SearchResult(id="1", score=0.9, metadata={"source_file": "doc.pdf", "section": "Intro", "text": "Test content"})
    ]

    mock_escalation.return_value = {"needs_escalation": False, "reason": "OK"}

    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Draft response"
    mock_client.chat.completions.create.return_value = mock_response

    result = process_customer_query("Wie installiere ich die Software?")

    assert "draft" in result
    assert "citations" in result
    assert "escalation" in result
    assert "detected_language" in result
    assert "complexity" in result
    assert result["detected_language"] == "de"
    assert result["complexity"] in ("brief", "moderate", "detailed")


def test_prompt_contains_multilingual_instructions():
    """The prompt template includes multilingual synthesis instructions."""
    assert "English, German, or French" in RESPONSE_PROMPT
    assert "Translate and synthesize" in RESPONSE_PROMPT


def test_prompt_contains_complexity_placeholder():
    """The prompt template includes a complexity guidance placeholder."""
    assert "{complexity_guidance}" in RESPONSE_PROMPT


def test_build_context():
    results = [
        SearchResult(id="1", score=0.9, metadata={"source_file": "doc.pdf", "section": "Setup", "text": "Install steps"}),
        SearchResult(id="2", score=0.8, metadata={"source_file": "faq.pdf", "section": "", "text": "FAQ content"}),
    ]
    context = _build_context(results)
    assert "doc.pdf" in context
    assert "Section: Setup" in context
    assert "Install steps" in context
    assert "faq.pdf" in context


def test_extract_citations_deduplicates():
    results = [
        SearchResult(id="1", score=0.9, metadata={"source_file": "doc.pdf", "section": "A"}),
        SearchResult(id="2", score=0.8, metadata={"source_file": "doc.pdf", "section": "A"}),
        SearchResult(id="3", score=0.7, metadata={"source_file": "faq.pdf", "section": "B"}),
    ]
    citations = _extract_citations(results)
    assert len(citations) == 2  # deduplicated
    assert "doc.pdf" in citations[0]
    assert "faq.pdf" in citations[1]
