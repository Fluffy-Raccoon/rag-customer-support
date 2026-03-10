import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.complexity_analyzer import analyze_complexity


def test_brief_simple_question():
    assert analyze_complexity("What is TYREX?") == "brief"


def test_brief_short_query():
    assert analyze_complexity("Where is the FAQ?") == "brief"


def test_moderate_technical_question():
    result = analyze_complexity("How do I configure the proxy settings for the management server?")
    assert result in ("moderate", "detailed")


def test_moderate_medium_length():
    result = analyze_complexity(
        "I'm having trouble connecting to the server. "
        "The connection keeps timing out after a few seconds."
    )
    assert result in ("moderate", "detailed")


def test_detailed_step_by_step():
    assert analyze_complexity("Please explain step by step how to install the software") == "detailed"


def test_detailed_multiple_questions():
    query = (
        "I need help with several things. "
        "How do I install the software? "
        "What are the hardware requirements? "
        "How do I configure the network?"
    )
    assert analyze_complexity(query) == "detailed"


def test_detailed_multiple_technical_terms():
    query = "How do I configure SSL certificates and setup the proxy for the database?"
    assert analyze_complexity(query) == "detailed"


def test_empty_query():
    assert analyze_complexity("") == "brief"


def test_german_step_by_step():
    assert analyze_complexity("Erklären Sie Schritt für Schritt die Installation") == "detailed"
