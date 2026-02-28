"""Feedback learning loop: capture approved responses, embed, and index for retrieval boosting."""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import get_settings
from src.ingestion.embedder import get_embeddings
from src.retrieval.pinecone_client import upsert_vectors

logger = logging.getLogger(__name__)

_APPROVED_RESPONSES_FILE = Path("data/approved_responses.jsonl")


def _ensure_data_dir():
    _APPROVED_RESPONSES_FILE.parent.mkdir(parents=True, exist_ok=True)


def capture_approved_response(
    original_query: str,
    draft_response: str,
    final_response: str,
    agent_edits: str = "",
    ticket_id: int | None = None,
) -> dict:
    """Store an approved Q&A pair and embed it for future retrieval.

    Returns the stored record.
    """
    _ensure_data_dir()

    record = {
        "id": uuid.uuid4().hex[:12],
        "original_query": original_query,
        "draft_response": draft_response,
        "final_response": final_response,
        "agent_edits": agent_edits,
        "ticket_id": ticket_id,
        "approved_at": datetime.now(timezone.utc).isoformat(),
    }

    # Append to JSONL store
    with open(_APPROVED_RESPONSES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info("Captured approved response %s", record["id"])

    # Embed and index
    embed_approved_response(record)

    return record


def embed_approved_response(record: dict) -> None:
    """Create a training document from an approved response and index in Pinecone."""
    training_text = (
        f"Customer Question: {record['original_query']}\n\n"
        f"Approved Response: {record['final_response']}"
    )

    embeddings = get_embeddings([training_text])

    vector_id = f"approved_{record['id']}"
    metadata = {
        "text": training_text,
        "source_type": "approved_response",
        "source_file": "approved_responses",
        "document_type": "approved_response",
        "approval_date": record["approved_at"],
        "original_query": record["original_query"],
        "ticket_id": str(record.get("ticket_id") or ""),
    }

    upsert_vectors([vector_id], embeddings, [metadata])
    logger.info("Embedded approved response %s", record["id"])


def get_approved_responses(limit: int = 50) -> list[dict]:
    """Read recent approved responses from the JSONL store."""
    if not _APPROVED_RESPONSES_FILE.exists():
        return []

    records = []
    with open(_APPROVED_RESPONSES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Return most recent first
    records.reverse()
    return records[:limit]


def reprocess_approved_responses() -> int:
    """Re-embed all stored approved responses. Useful after model changes.

    Returns count of reprocessed responses.
    """
    records = get_approved_responses(limit=0)  # 0 means no limit since we reverse and slice
    if not records:
        return 0

    # Read all records (not limited)
    all_records = []
    with open(_APPROVED_RESPONSES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    for record in all_records:
        embed_approved_response(record)

    logger.info("Reprocessed %d approved responses", len(all_records))
    return len(all_records)
