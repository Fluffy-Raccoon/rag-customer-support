import logging

from langdetect import detect, detect_langs, LangDetectException
from openai import OpenAI

from config import get_settings

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"en", "de", "fr"}

# Short texts (< this many chars) get OpenAI fallback if langdetect is uncertain
_SHORT_TEXT_THRESHOLD = 50

# Minimum langdetect probability to trust the result
_CONFIDENCE_THRESHOLD = 0.7


def detect_language(text: str) -> str:
    """Detect the language of the text. Returns ISO 639-1 code, defaults to 'en'.

    Uses langdetect as primary detector with OpenAI fallback for short or
    ambiguous text.
    """
    lang, _ = detect_language_with_confidence(text)
    return lang


def detect_language_with_confidence(text: str) -> tuple[str, str]:
    """Detect language and return (lang_code, method).

    method is 'langdetect' or 'openai' indicating which detector was used.
    """
    text = text.strip()
    if not text:
        return "en", "default"

    # Try langdetect first
    try:
        results = detect_langs(text)
        best = results[0]
        lang = best.lang
        prob = best.prob

        if lang in SUPPORTED_LANGUAGES and (prob >= _CONFIDENCE_THRESHOLD or len(text) >= _SHORT_TEXT_THRESHOLD):
            return lang, "langdetect"

        # Low confidence or short text — try OpenAI fallback
        if len(text) < _SHORT_TEXT_THRESHOLD or prob < _CONFIDENCE_THRESHOLD:
            openai_lang = _detect_with_openai(text)
            if openai_lang:
                return openai_lang, "openai"

        # Fall back to langdetect result if it's supported
        if lang in SUPPORTED_LANGUAGES:
            return lang, "langdetect"

        return "en", "default"

    except LangDetectException:
        # langdetect failed entirely — try OpenAI
        openai_lang = _detect_with_openai(text)
        if openai_lang:
            return openai_lang, "openai"
        return "en", "default"


def _detect_with_openai(text: str) -> str | None:
    """Use OpenAI to detect language. Returns lang code or None on failure."""
    try:
        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{
                "role": "user",
                "content": (
                    f"What language is this text written in? "
                    f"Reply with ONLY the ISO 639-1 code (en, de, or fr). "
                    f"If unsure, reply 'en'.\n\nText: {text[:200]}"
                ),
            }],
            temperature=0,
            max_tokens=5,
        )
        lang = response.choices[0].message.content.strip().lower()
        return lang if lang in SUPPORTED_LANGUAGES else None
    except Exception:
        logger.warning("OpenAI language detection fallback failed")
        return None
