import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel

from config import get_settings
from src.generation.response_generator import process_customer_query
from src.ingestion.pipeline import ingest_document, ingest_directory
from src.integrations.zendesk_client import (
    zendesk_generate_draft,
    format_internal_note,
    ZendeskConfigError,
)
from src.integrations.email_client import (
    process_trigger_folder,
    start_email_monitor,
    stop_email_monitor,
    is_monitor_running,
    IMAPConfigError,
)
from src.feedback.learning_loop import (
    capture_approved_response,
    get_approved_responses,
)
from src.audit.logger import log_query_event, get_audit_log

# Configure logging
settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Customer Support RAG",
    description="RAG-based customer support draft response system",
    version="0.2.0",
)


# --- API Key Authentication ---


def verify_api_key(x_api_key: str | None = Header(None)):
    """Verify API key if configured. Skipped when api_key setting is None."""
    configured_key = get_settings().api_key
    if configured_key is None:
        return  # Auth disabled
    if x_api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# --- Request / Response Models ---


class QueryRequest(BaseModel):
    query: str


class EscalationInfo(BaseModel):
    needs_escalation: bool
    reason: str


class QueryResponse(BaseModel):
    draft: str
    citations: list[str]
    escalation: EscalationInfo
    detected_language: str
    complexity: str


class ZendeskDraftRequest(BaseModel):
    ticket_id: int


class EmailMonitorRequest(BaseModel):
    poll_interval: int = 30


class IngestDirectoryRequest(BaseModel):
    directory_path: str
    full_reindex: bool = False


class IngestResponse(BaseModel):
    chunks_indexed: int
    message: str


class FeedbackApproveRequest(BaseModel):
    original_query: str
    draft_response: str
    final_response: str
    agent_edits: str = ""
    ticket_id: int | None = None


class AuditLogRequest(BaseModel):
    limit: int = 100
    offset: int = 0


# --- Endpoints ---


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query_endpoint(request: QueryRequest):
    """Process a customer query and return a draft response with citations."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()
    result = process_customer_query(request.query)
    duration_ms = int((time.time() - start) * 1000)

    log_query_event(request.query, result, source_channel="manual", duration_ms=duration_ms)

    return QueryResponse(
        draft=result["draft"],
        citations=result["citations"],
        escalation=EscalationInfo(**result["escalation"]),
        detected_language=result["detected_language"],
        complexity=result["complexity"],
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
def ingest_file(file: UploadFile = File(...), full_reindex: bool = False):
    """Ingest a single document file (PDF, DOCX, EML, MSG, CSV)."""
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if not suffix:
        raise HTTPException(status_code=400, detail="File must have an extension")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        count = ingest_document(tmp_path, full_reindex=full_reindex)
        return IngestResponse(
            chunks_indexed=count,
            message=f"Successfully ingested {file.filename} ({count} chunks)",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ingest/directory", response_model=IngestResponse, dependencies=[Depends(verify_api_key)])
def ingest_dir(request: IngestDirectoryRequest):
    """Ingest all supported documents from a directory."""
    dir_path = Path(request.directory_path)
    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {request.directory_path}")

    count = ingest_directory(str(dir_path), full_reindex=request.full_reindex)
    return IngestResponse(
        chunks_indexed=count,
        message=f"Ingested {count} chunks from {request.directory_path}",
    )


# --- Zendesk Integration ---


@app.post("/zendesk/generate-draft", dependencies=[Depends(verify_api_key)])
def zendesk_draft_endpoint(request: ZendeskDraftRequest):
    """Generate a draft response for a Zendesk ticket and post as internal note."""
    try:
        start = time.time()
        result = zendesk_generate_draft(request.ticket_id)
        duration_ms = int((time.time() - start) * 1000)

        log_query_event(
            query=f"[Zendesk ticket #{request.ticket_id}]",
            result=result,
            source_channel="zendesk",
            duration_ms=duration_ms,
        )

        return {
            "ticket_id": request.ticket_id,
            "draft": result["draft"],
            "citations": result["citations"],
            "escalation": result["escalation"],
            "detected_language": result["detected_language"],
            "complexity": result["complexity"],
        }
    except ZendeskConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Email Integration ---


@app.post("/email/process", dependencies=[Depends(verify_api_key)])
def email_process_endpoint():
    """Manually process all unread emails in the trigger folder."""
    try:
        start = time.time()
        results = process_trigger_folder()
        duration_ms = int((time.time() - start) * 1000)

        for r in results:
            log_query_event(
                query=f"[Email] {r['subject']}",
                result=r["result"],
                source_channel="email",
                duration_ms=duration_ms,
            )

        return {
            "processed": len(results),
            "emails": [
                {
                    "msg_id": r["msg_id"],
                    "subject": r["subject"],
                    "detected_language": r["result"]["detected_language"],
                }
                for r in results
            ],
        }
    except IMAPConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/email/start-monitor", dependencies=[Depends(verify_api_key)])
def email_start_monitor_endpoint(request: EmailMonitorRequest):
    """Start background email monitoring."""
    try:
        if is_monitor_running():
            return {"status": "already_running"}
        start_email_monitor(poll_interval=request.poll_interval)
        return {"status": "started", "poll_interval": request.poll_interval}
    except IMAPConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/email/stop-monitor", dependencies=[Depends(verify_api_key)])
def email_stop_monitor_endpoint():
    """Stop background email monitoring."""
    if not is_monitor_running():
        return {"status": "not_running"}
    stop_email_monitor()
    return {"status": "stopped"}


# --- Feedback Learning Loop ---


@app.post("/feedback/approve", dependencies=[Depends(verify_api_key)])
def feedback_approve_endpoint(request: FeedbackApproveRequest):
    """Capture an approved response and embed it for future retrieval boosting."""
    record = capture_approved_response(
        original_query=request.original_query,
        draft_response=request.draft_response,
        final_response=request.final_response,
        agent_edits=request.agent_edits,
        ticket_id=request.ticket_id,
    )
    return {"status": "captured", "id": record["id"]}


@app.get("/feedback/approved", dependencies=[Depends(verify_api_key)])
def feedback_list_endpoint(limit: int = 50):
    """List recent approved responses."""
    records = get_approved_responses(limit=limit)
    return {"count": len(records), "responses": records}


# --- Audit Log ---


@app.get("/audit/log", dependencies=[Depends(verify_api_key)])
def audit_log_endpoint(limit: int = 100, offset: int = 0):
    """Read recent audit log entries."""
    entries = get_audit_log(limit=limit, offset=offset)
    return {"count": len(entries), "entries": entries}
