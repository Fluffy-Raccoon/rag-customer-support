import base64
import logging

import httpx

from config import get_settings
from src.generation.response_generator import process_customer_query

logger = logging.getLogger(__name__)


class ZendeskConfigError(Exception):
    pass


def _get_zendesk_config() -> tuple[str, dict]:
    """Return (base_url, auth_headers). Raises ZendeskConfigError if not configured."""
    settings = get_settings()
    if not all([settings.zendesk_subdomain, settings.zendesk_email, settings.zendesk_api_token]):
        raise ZendeskConfigError("Zendesk integration is not configured")

    base_url = f"https://{settings.zendesk_subdomain}.zendesk.com"
    credentials = f"{settings.zendesk_email}/token:{settings.zendesk_api_token}"
    encoded = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded}",
        "Content-Type": "application/json",
    }
    return base_url, headers


def get_ticket(ticket_id: int) -> dict:
    """Fetch a ticket from Zendesk."""
    base_url, headers = _get_zendesk_config()
    response = httpx.get(f"{base_url}/api/v2/tickets/{ticket_id}.json", headers=headers)
    response.raise_for_status()
    return response.json()["ticket"]


def get_ticket_comments(ticket_id: int) -> list[dict]:
    """Fetch all comments for a ticket."""
    base_url, headers = _get_zendesk_config()
    response = httpx.get(f"{base_url}/api/v2/tickets/{ticket_id}/comments.json", headers=headers)
    response.raise_for_status()
    return response.json()["comments"]


def get_latest_customer_message(ticket_id: int) -> str:
    """Extract the latest end-user (non-agent) comment body from a ticket."""
    comments = get_ticket_comments(ticket_id)

    # Filter to end-user comments and get the latest
    customer_comments = [
        c for c in comments
        if c.get("author", {}).get("role") == "end-user"
        or c.get("via", {}).get("channel") == "email"  # fallback: email-originated
    ]

    if not customer_comments:
        # Fallback: use the ticket description (first comment)
        if comments:
            return comments[0].get("body", "")
        raise ValueError(f"No comments found for ticket {ticket_id}")

    return customer_comments[-1].get("body", "")


def post_internal_note(ticket_id: int, body: str) -> dict:
    """Post an internal (non-public) note to a Zendesk ticket."""
    base_url, headers = _get_zendesk_config()
    payload = {
        "ticket": {
            "comment": {
                "body": body,
                "public": False,
            }
        }
    }
    response = httpx.put(
        f"{base_url}/api/v2/tickets/{ticket_id}.json",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()
    return response.json()


def format_internal_note(result: dict) -> str:
    """Format a RAG result into a structured Zendesk internal note."""
    citations = "\n".join(result.get("citations", []))

    escalation = result.get("escalation", {})
    if escalation.get("needs_escalation"):
        esc_text = f"!! ESCALATE: {escalation.get('reason', 'Unknown reason')}"
    else:
        esc_text = "No escalation needed"

    lang = result.get("detected_language", "en").upper()
    complexity = result.get("complexity", "moderate")

    return (
        "AI-GENERATED DRAFT (Review before sending)\n"
        "============================================\n\n"
        f"{result.get('draft', '')}\n\n"
        "SOURCES:\n"
        f"{citations}\n\n"
        f"ESCALATION: {esc_text}\n\n"
        f"Language detected: {lang} | Complexity: {complexity}"
    )


def zendesk_generate_draft(ticket_id: int) -> dict:
    """Full Zendesk draft workflow: fetch → RAG query → format → post note."""
    # 1. Get latest customer message
    customer_message = get_latest_customer_message(ticket_id)
    logger.info("Zendesk ticket %d: customer message length=%d", ticket_id, len(customer_message))

    # 2. Generate draft via RAG pipeline
    result = process_customer_query(customer_message)

    # 3. Format as internal note
    note_body = format_internal_note(result)

    # 4. Post to Zendesk
    post_internal_note(ticket_id, note_body)
    logger.info("Zendesk ticket %d: internal note posted", ticket_id)

    return result
