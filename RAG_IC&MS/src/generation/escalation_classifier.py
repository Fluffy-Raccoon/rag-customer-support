import json
import logging
import re

from openai import OpenAI

from config import get_settings

logger = logging.getLogger(__name__)

# Keyword triggers from spec Section 3.3
ESCALATION_TRIGGERS = [
    "billing dispute",
    "refund request",
    "refund",
    "legal threat",
    "security incident",
    "data breach",
    "account compromise",
    "executive escalation",
    "speak to a manager",
    "speak to manager",
    "regulatory compliance",
    "conflicting information",
    "requires system access",
    "custom development request",
    # German equivalents
    "rechnungsstreit",
    "erstattung",
    "rückerstattung",
    "sicherheitsvorfall",
    "datenpanne",
    "kontokompromittierung",
    "eskalation",
    "vorgesetzten sprechen",
    # French equivalents
    "litige de facturation",
    "remboursement",
    "incident de sécurité",
    "violation de données",
    "parler au responsable",
]

_TRIGGER_PATTERNS = [re.compile(re.escape(t), re.IGNORECASE) for t in ESCALATION_TRIGGERS]

ESCALATION_PROMPT = """Analyze if this support query requires human escalation beyond a standard response.

Query: {query}
Available context: {context}
Draft response: {draft}

Escalate if:
- Query involves billing disputes, refunds, legal, security, or compliance
- Documentation is insufficient or conflicting
- Request requires system access or policy decisions
- Customer explicitly requests manager/escalation

Return ONLY valid JSON: {{"needs_escalation": true/false, "reason": "..."}}"""


def check_escalation_need(
    query: str,
    context: str,
    draft: str,
) -> dict:
    """Check if a query needs escalation. Uses keyword pre-screening first,
    then falls back to LLM for non-obvious cases."""

    # Fast keyword pre-screen
    matched, reason = _keyword_pre_screen(query)
    if matched:
        logger.info("Escalation triggered by keyword: %s", reason)
        return {"needs_escalation": True, "reason": reason}

    # LLM-based classification for non-obvious cases
    return _llm_classify(query, context, draft)


def _keyword_pre_screen(query: str) -> tuple[bool, str]:
    """Fast check against escalation trigger keywords.

    Returns (matched, reason).
    """
    for pattern in _TRIGGER_PATTERNS:
        match = pattern.search(query)
        if match:
            return True, f"Query contains escalation trigger: '{match.group()}'"
    return False, ""


def _llm_classify(query: str, context: str, draft: str) -> dict:
    """Use the LLM to classify whether a query needs escalation."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)

    prompt = ESCALATION_PROMPT.format(
        query=query,
        context=context[:2000],
        draft=draft[:1000],
    )

    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except (json.JSONDecodeError, Exception):
        logger.exception("Escalation classification failed")
        return {"needs_escalation": False, "reason": "Classification unavailable"}
