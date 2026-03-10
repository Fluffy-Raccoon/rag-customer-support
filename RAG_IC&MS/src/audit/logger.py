"""Structured audit logging for all generated responses."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_AUDIT_LOG_FILE = Path("data/audit_log.jsonl")


def _ensure_data_dir():
    _AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_query_event(
    query: str,
    result: dict,
    source_channel: str,
    duration_ms: int = 0,
) -> dict:
    """Append a structured audit record for a generated response.

    Returns the logged record.
    """
    _ensure_data_dir()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "detected_language": result.get("detected_language", ""),
        "complexity": result.get("complexity", ""),
        "needs_escalation": result.get("escalation", {}).get("needs_escalation", False),
        "escalation_reason": result.get("escalation", {}).get("reason", ""),
        "draft_length": len(result.get("draft", "")),
        "citation_count": len(result.get("citations", [])),
        "source_channel": source_channel,
        "duration_ms": duration_ms,
    }

    with open(_AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.debug("Audit logged query via %s channel", source_channel)
    return record


def get_audit_log(limit: int = 100, offset: int = 0) -> list[dict]:
    """Read audit log entries, most recent first."""
    if not _AUDIT_LOG_FILE.exists():
        return []

    records = []
    with open(_AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Most recent first
    records.reverse()

    return records[offset : offset + limit]
