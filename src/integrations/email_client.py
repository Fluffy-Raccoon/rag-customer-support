import email as email_lib
import imaplib
import logging
import threading
import time
from email import policy
from email.mime.text import MIMEText
from datetime import datetime, timezone

from config import get_settings
from src.generation.response_generator import process_customer_query

logger = logging.getLogger(__name__)

_monitor_thread: threading.Thread | None = None
_monitor_stop_event = threading.Event()


class IMAPConfigError(Exception):
    pass


def _check_imap_config():
    settings = get_settings()
    if not all([settings.imap_server, settings.imap_user, settings.imap_password]):
        raise IMAPConfigError("IMAP integration is not configured")


def connect_imap() -> imaplib.IMAP4_SSL:
    """Connect and authenticate to the IMAP server."""
    _check_imap_config()
    settings = get_settings()
    mail = imaplib.IMAP4_SSL(settings.imap_server)
    mail.login(settings.imap_user, settings.imap_password)
    return mail


def extract_email_body(raw_bytes: bytes) -> str:
    """Parse RFC822 email bytes and extract the plain text body."""
    msg = email_lib.message_from_bytes(raw_bytes, policy=policy.default)

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
    else:
        if msg.get_content_type() == "text/plain":
            body = msg.get_content()

    return body.strip() if body else ""


def check_trigger_folder() -> list[dict]:
    """Check the trigger folder for unread emails.

    Returns a list of dicts with: msg_id, subject, sender, body.
    """
    settings = get_settings()
    mail = connect_imap()

    try:
        status, _ = mail.select(f'"{settings.imap_trigger_folder}"')
        if status != "OK":
            logger.warning("Could not select folder: %s", settings.imap_trigger_folder)
            return []

        _, message_ids = mail.search(None, "UNSEEN")
        if not message_ids[0]:
            return []

        results = []
        for msg_id in message_ids[0].split():
            _, data = mail.fetch(msg_id, "(RFC822)")
            raw_bytes = data[0][1]

            msg = email_lib.message_from_bytes(raw_bytes, policy=policy.default)
            body = extract_email_body(raw_bytes)

            results.append({
                "msg_id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                "subject": msg.get("Subject", ""),
                "sender": msg.get("From", ""),
                "body": body,
            })

        return results
    finally:
        try:
            mail.close()
            mail.logout()
        except Exception:
            pass


def mark_as_processed(mail: imaplib.IMAP4_SSL, msg_id: str) -> None:
    """Mark an email as seen/processed."""
    mail.store(msg_id.encode() if isinstance(msg_id, str) else msg_id, "+FLAGS", "\\Seen")


def save_draft_reply(mail: imaplib.IMAP4_SSL, original_subject: str, original_sender: str, draft_text: str) -> None:
    """Save a formatted draft reply to the Drafts folder via IMAP APPEND."""
    reply_msg = MIMEText(draft_text, "plain", "utf-8")
    reply_msg["Subject"] = f"Re: {original_subject}"
    reply_msg["To"] = original_sender
    reply_msg["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")

    mail.append(
        "Drafts",
        "\\Draft",
        None,
        reply_msg.as_bytes(),
    )


def process_trigger_folder() -> list[dict]:
    """Process all unread emails in the trigger folder.

    For each email: generate RAG draft → save draft reply → mark processed.
    Returns list of results.
    """
    settings = get_settings()
    _check_imap_config()

    mail = connect_imap()
    results = []

    try:
        status, _ = mail.select(f'"{settings.imap_trigger_folder}"')
        if status != "OK":
            logger.warning("Could not select folder: %s", settings.imap_trigger_folder)
            return []

        _, message_ids = mail.search(None, "UNSEEN")
        if not message_ids[0]:
            logger.info("No unread emails in trigger folder")
            return []

        for msg_id in message_ids[0].split():
            msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

            try:
                _, data = mail.fetch(msg_id, "(RFC822)")
                raw_bytes = data[0][1]

                msg = email_lib.message_from_bytes(raw_bytes, policy=policy.default)
                body = extract_email_body(raw_bytes)
                subject = msg.get("Subject", "")
                sender = msg.get("From", "")

                if not body:
                    logger.warning("Empty body for email %s, skipping", msg_id_str)
                    mark_as_processed(mail, msg_id_str)
                    continue

                # Generate draft
                logger.info("Processing email %s: %s", msg_id_str, subject)
                result = process_customer_query(body)

                # Format draft reply
                draft_body = _format_draft_reply(result)

                # Save to Drafts
                save_draft_reply(mail, subject, sender, draft_body)

                # Mark as processed
                mark_as_processed(mail, msg_id_str)

                results.append({
                    "msg_id": msg_id_str,
                    "subject": subject,
                    "sender": sender,
                    "result": result,
                })
                logger.info("Draft saved for email %s", msg_id_str)

            except Exception:
                logger.exception("Failed to process email %s", msg_id_str)

        return results
    finally:
        try:
            mail.close()
            mail.logout()
        except Exception:
            pass


def _format_draft_reply(result: dict) -> str:
    """Format a RAG result into a draft email reply."""
    citations = "\n".join(result.get("citations", []))

    escalation = result.get("escalation", {})
    if escalation.get("needs_escalation"):
        esc_note = f"\n[INTERNAL NOTE - ESCALATE: {escalation.get('reason', '')}]\n"
    else:
        esc_note = ""

    return (
        f"{result.get('draft', '')}\n"
        f"{esc_note}\n"
        "---\n"
        f"Sources: {citations}\n"
        f"Language: {result.get('detected_language', 'en').upper()} | "
        f"Complexity: {result.get('complexity', 'moderate')}\n"
        "[AI-generated draft — review before sending]"
    )


def start_email_monitor(poll_interval: int = 30) -> None:
    """Start background email monitoring in a separate thread."""
    global _monitor_thread

    _check_imap_config()

    if _monitor_thread and _monitor_thread.is_alive():
        logger.warning("Email monitor is already running")
        return

    _monitor_stop_event.clear()

    def _monitor_loop():
        logger.info("Email monitor started (poll interval: %ds)", poll_interval)
        while not _monitor_stop_event.is_set():
            try:
                results = process_trigger_folder()
                if results:
                    logger.info("Processed %d emails", len(results))
            except Exception:
                logger.exception("Error in email monitor loop")
            _monitor_stop_event.wait(timeout=poll_interval)
        logger.info("Email monitor stopped")

    _monitor_thread = threading.Thread(target=_monitor_loop, daemon=True, name="email-monitor")
    _monitor_thread.start()


def stop_email_monitor() -> None:
    """Stop the background email monitor."""
    global _monitor_thread
    _monitor_stop_event.set()
    if _monitor_thread and _monitor_thread.is_alive():
        _monitor_thread.join(timeout=5)
    _monitor_thread = None
    logger.info("Email monitor stop requested")


def is_monitor_running() -> bool:
    return _monitor_thread is not None and _monitor_thread.is_alive()
