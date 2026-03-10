import re

# Technical terms that suggest a more detailed response is needed
_TECHNICAL_TERMS = {
    "configure", "konfigurieren", "configurer",
    "install", "installieren", "installer",
    "setup", "einrichten", "deploy", "bereitstellen",
    "troubleshoot", "debug", "fehler",
    "migrate", "migrieren", "upgrade",
    "integrate", "integrieren", "integration",
    "certificate", "zertifikat", "certificat",
    "proxy", "firewall", "ssl", "https", "api",
    "database", "datenbank", "backup",
    "permission", "berechtigung", "authentication",
}

# Phrases that indicate multi-step or detailed requests
_DETAILED_PHRASES = [
    r"step[- ]by[- ]step",
    r"schritt für schritt",
    r"étape par étape",
    r"how do i .+ and .+",
    r"wie kann ich .+ und .+",
    r"explain .+ in detail",
    r"im detail",
    r"en détail",
]

_DETAILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _DETAILED_PHRASES]


def analyze_complexity(query_text: str) -> str:
    """Analyze query complexity for response length calibration.

    Returns: "brief", "moderate", or "detailed"
    """
    text = query_text.strip()
    if not text:
        return "brief"

    words = text.split()
    word_count = len(words)
    question_marks = text.count("?")
    lower_text = text.lower()
    lower_words = set(lower_text.split())

    # Check for explicit detail request patterns
    for pattern in _DETAILED_PATTERNS:
        if pattern.search(text):
            return "detailed"

    # Multiple questions → detailed
    if question_marks >= 3:
        return "detailed"

    # Long query with multiple questions → detailed
    if word_count > 60 and question_marks >= 2:
        return "detailed"

    # Count technical terms
    tech_count = len(lower_words & _TECHNICAL_TERMS)

    # Short, single question, no technical terms → brief
    if word_count < 15 and question_marks <= 1 and tech_count == 0:
        return "brief"

    # Multiple technical terms or moderately long → moderate/detailed
    if tech_count >= 3 or (word_count > 50 and tech_count >= 1):
        return "detailed"

    if tech_count >= 1 or word_count >= 15 or question_marks >= 2:
        return "moderate"

    return "brief"
