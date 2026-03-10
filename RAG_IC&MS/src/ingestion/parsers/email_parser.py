import email
import logging
from email import policy
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_email(file_path: str) -> str:
    """Parse .eml or .msg files and extract subject + body."""
    path = Path(file_path)
    if path.suffix.lower() == ".msg":
        return _parse_msg(file_path)
    return _parse_eml(file_path)


def _parse_eml(file_path: str) -> str:
    """Parse a .eml file using the stdlib email module."""
    with open(file_path, "rb") as f:
        msg = email.message_from_binary_file(f, policy=policy.default)

    subject = msg.get("Subject", "")
    sender = msg.get("From", "")

    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_content()
                break
    else:
        body = msg.get_content()

    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if sender:
        parts.append(f"From: {sender}")
    if body:
        parts.append(f"\n{body}")

    return "\n".join(parts)


def _parse_msg(file_path: str) -> str:
    """Parse a .msg file using extract-msg."""
    try:
        import extract_msg

        msg = extract_msg.Message(file_path)
        parts = []
        if msg.subject:
            parts.append(f"Subject: {msg.subject}")
        if msg.sender:
            parts.append(f"From: {msg.sender}")
        if msg.body:
            parts.append(f"\n{msg.body}")
        msg.close()
        return "\n".join(parts)
    except ImportError:
        logger.error("extract-msg not installed — cannot parse .msg files")
        raise
