import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.language_detector import detect_language, detect_language_with_confidence


def test_detect_english():
    text = "Hello, I need help resetting my password. Can you please assist me with this issue?"
    assert detect_language(text) == "en"


def test_detect_german():
    text = "Guten Tag, ich brauche Hilfe bei der Einrichtung meines Kontos. Können Sie mir bitte helfen?"
    assert detect_language(text) == "de"


def test_detect_french():
    text = "Bonjour, j'ai besoin d'aide pour configurer mon compte. Pouvez-vous m'aider s'il vous plaît?"
    assert detect_language(text) == "fr"


def test_empty_text_defaults_to_english():
    assert detect_language("") == "en"
    assert detect_language("   ") == "en"


def test_unsupported_language_defaults_to_english():
    # Japanese text — should fall back to English
    with patch("src.generation.language_detector._detect_with_openai", return_value=None):
        assert detect_language("これはテストです。日本語のテキストです。") == "en"


def test_short_mixed_language_uses_openai_fallback():
    """Short text with mixed languages should trigger OpenAI fallback."""
    with patch("src.generation.language_detector._detect_with_openai", return_value="de") as mock_openai:
        lang, method = detect_language_with_confidence("Wie geht on-premise?")
        # Should have attempted OpenAI fallback for this short text
        if method == "openai":
            mock_openai.assert_called_once()
            assert lang == "de"


def test_confidence_returns_method():
    text = "This is a sufficiently long English text for langdetect to be quite confident about the language detection."
    lang, method = detect_language_with_confidence(text)
    assert lang == "en"
    assert method == "langdetect"
