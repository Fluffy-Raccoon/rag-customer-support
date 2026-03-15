# RAG Customer Support System

A Retrieval-Augmented Generation system that drafts customer support responses from company documentation. Supports multilingual queries (EN/DE/FR), integrates with Zendesk and email (IMAP), and learns from agent-approved responses over time.

## Quickstart

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) (`pip install poetry`)
- OpenAI API key (embeddings)
- Anthropic API key (response generation)
- Pinecone API key + index

### Setup

```bash
# Install Poetry (if not already installed)
pip3 install poetry

# Install dependencies
python3 -m poetry install

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys (see Configuration below)

# Run tests to verify installation
python3 -m pytest tests/ -v

# Start the server
python3 -m uvicorn src.api.main:app --reload
```

> **Tip:** If `poetry` or other CLI tools give "command not found" after install, use `python3 -m <tool>` instead (e.g., `python3 -m poetry install`). This bypasses PATH issues.

The API is now available at `http://localhost:8000`. Interactive docs at `/docs`.

### Ingest documents and query

```bash
# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/manual.pdf"

# Ingest an entire directory
curl -X POST http://localhost:8000/ingest/directory \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "data/"}'

# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I install the software on-premise?"}'
```

---

## Architecture

```
Document Ingestion          Query Pipeline              Feedback Loop
─────────────────          ──────────────              ─────────────
PDF/DOCX/Email/CSV         Customer Query              Agent approves draft
        │                        │                           │
   Parse + Clean           Detect Language (EN/DE/FR)   Store Q&A pair
        │                        │                           │
   Chunk (512 tokens)      Embed Query (OpenAI)        Embed + index
        │                        │                           │
   Embed (OpenAI)          Pinecone Search             Boost in reranker
        │                        │                      (1.3x weight)
   Upsert to Pinecone     Rerank + Build Context
                                 │
                           Generate Draft (Claude)
                                 │
                           Citations + Escalation Check
                                 │
                     ┌───────────┴───────────┐
                     │                       │
               Zendesk Note           Email Draft
```

### Supported document formats

| Format | Notes |
|--------|-------|
| PDF | PyMuPDF extraction, OCR fallback for scanned pages |
| DOCX | Paragraphs + tables |
| DOC | Must be pre-converted to DOCX |
| EML/MSG | Email body, subject, sender |
| CSV | Zendesk ticket exports (subject, description, agent_response) |

---

## Configuration

All settings are managed via environment variables (`.env` file). Required settings must be set for the server to start. Optional settings enable additional features when provided.

### Required

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (used for embeddings) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (used for response generation) |
| `PINECONE_API_KEY` | — | Pinecone API key |
| `PINECONE_INDEX_NAME` | `customer-support-rag` | Pinecone index (3072-dim, cosine) |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model |
| `ANTHROPIC_CHAT_MODEL` | `claude-sonnet-4-20250514` | Chat model for response generation |

### Optional

| Variable                  | Default          | Description                                        |
| ------------------------- | ---------------- | -------------------------------------------------- |
| `API_KEY`                 | *(disabled)*     | When set, all endpoints require `X-API-Key` header |
| `ZENDESK_SUBDOMAIN`       | *(disabled)*     | Zendesk subdomain (e.g. `mycompany`)               |
| `ZENDESK_EMAIL`           | *(disabled)*     | Zendesk agent email for API auth                   |
| `ZENDESK_API_TOKEN`       | *(disabled)*     | Zendesk API token                                  |
| `IMAP_SERVER`             | *(disabled)*     | IMAP server hostname                               |
| `IMAP_USER`               | *(disabled)*     | IMAP login user                                    |
| `IMAP_PASSWORD`           | *(disabled)*     | IMAP password                                      |
| [[`IMAP_TRIGGER_FOLDER`]] | `Generate Draft` | IMAP folder to monitor                             |
| `LOG_LEVEL`               | `INFO`           | Logging level                                      |
| `CHUNK_SIZE`              | `512`            | Tokens per chunk                                   |
| `CHUNK_OVERLAP`           | `50`             | Token overlap between chunks                       |
| `RETRIEVAL_TOP_K`         | `10`             | Documents retrieved per query                      |

---

## API Reference

All mutation endpoints support optional API key authentication via the `X-API-Key` header when `API_KEY` is configured. The `/health` endpoint is always unauthenticated.

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/query` | Process a customer query, returns draft + citations + escalation |
| `POST` | `/ingest` | Upload and ingest a single document (multipart form) |
| `POST` | `/ingest/directory` | Ingest all supported files in a directory |

### Zendesk

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/zendesk/generate-draft` | Generate draft for a ticket and post as internal note |

**Request:** `{"ticket_id": 123}`

### Email

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/email/process` | Process all unread emails in trigger folder |
| `POST` | `/email/start-monitor` | Start background email polling |
| `POST` | `/email/stop-monitor` | Stop background email polling |

### Feedback

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/feedback/approve` | Capture an agent-approved response |
| `GET` | `/feedback/approved?limit=50` | List recent approved responses |

**Approve request:**
```json
{
  "original_query": "How do I reset my password?",
  "draft_response": "AI-generated draft...",
  "final_response": "Agent-edited final version...",
  "agent_edits": "Clarified step 3",
  "ticket_id": 456
}
```

### Audit

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/audit/log?limit=100&offset=0` | Query audit trail of all generated responses |

### Response format (POST /query)

```json
{
  "draft": "Here is how to reset your password...",
  "citations": ["[1] FAQ.pdf - Password Reset"],
  "escalation": {
    "needs_escalation": false,
    "reason": "No escalation needed"
  },
  "detected_language": "en",
  "complexity": "brief"
}
```

---

## Testing & Debugging UI

A Streamlit-based testing interface for running queries, inspecting all pipeline stages, and calibrating RAG behavior:

```bash
# Launch the test UI (opens in browser)
python3 -m streamlit run tools/test_ui.py
```

> **Note:** If `streamlit` is installed but `streamlit run` gives "command not found", use `python3 -m streamlit run` instead. This happens when the `streamlit` binary isn't on your shell PATH.

Features:

- Text area for queries (no JSON escaping needed)
- **Draft Response** tab — rendered draft + citations
- **Retrieval Debug** tab — sortable tables of raw vs reranked chunks with scores
- **Context & Prompt** tab — see the exact prompt sent to Claude
- **Pipeline Metadata** tab — language, complexity, escalation, settings
- Query history with JSON export

---

## CLI Scripts

```bash
# Test a query manually
python3 scripts/test_query.py --query "Wie installiere ich die Software?"

# Test multilingual detection (EN/DE/FR)
python3 scripts/test_multilingual.py

# Import Zendesk CSV history
python3 scripts/import_zendesk_csv.py --csv data/zendesk_export.csv

# Re-index documents (skips unchanged files)
python3 scripts/reindex.py --documents data/

# Re-index with full refresh
python3 scripts/reindex.py --documents data/ --full-reindex

# Run email monitor as standalone daemon
python3 scripts/email_monitor.py --poll-interval 30
```

---

## Integrations

### Zendesk

The system integrates with Zendesk in two ways:

1. **API endpoint** — `POST /zendesk/generate-draft` fetches the latest customer message from a ticket, generates a draft, and posts it as an internal note.

2. **Sidebar widget** — A Zendesk app (`zendesk-app/`) adds a "Generate Draft" button to the ticket sidebar. Install it via the Zendesk Apps framework.

   To configure, update `API_BASE_URL` in `zendesk-app/src/app.js` to point to your server.

### Email (IMAP)

The email integration monitors a designated IMAP folder for unread emails:

1. Agent moves a customer email to the "Generate Draft" folder in their mail client
2. The system picks up the email, generates a draft response
3. The draft is saved to the Drafts folder as a reply
4. The original email is marked as read

Start monitoring via `POST /email/start-monitor` or `python3 scripts/email_monitor.py`.

---

## Feedback Learning Loop

When an agent reviews, edits, and approves a draft response, capture it via `POST /feedback/approve`. The system:

1. Stores the Q&A pair in `data/approved_responses.jsonl`
2. Embeds the approved response and indexes it in Pinecone with `source_type: "approved_response"`
3. On future queries, the reranker boosts matching approved responses by 1.3x

This creates a continuous improvement cycle — the more responses agents approve, the better future drafts become.

---

## Deployment

### Docker

```bash
# Build and run
docker compose up --build

# Or build manually
docker build -t rag-support .
docker run -p 8000:8000 --env-file .env -v ./data:/app/data rag-support
```

The `data/` directory is mounted as a volume to persist:
- `approved_responses.jsonl` — feedback loop data
- `audit_log.jsonl` — query audit trail
- `.indexed_versions.json` — document version tracking

### Pinecone index setup

Create a Pinecone serverless index before first use:

- **Dimension:** 3072 (text-embedding-3-large)
- **Metric:** cosine
- **Cloud/Region:** aws / us-east-1

---

## Testing

```bash
# Run all tests (83 tests)
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_feedback.py -v

# Run with coverage
python3 -m pytest tests/ --cov=src
```

### Test files

| File | Coverage |
|------|----------|
| `test_api.py` | API endpoints, auth, integrations |
| `test_feedback.py` | Feedback capture, embedding, storage |
| `test_audit.py` | Audit logging, retrieval |
| `test_chunker.py` | Text chunking, metadata |
| `test_parsers.py` | Document parsers (CSV) |
| `test_email_parser_real.py` | Email parser against real `.eml` test data |
| `test_response_generator.py` | RAG pipeline, prompts, citations |
| `test_language_detector.py` | EN/DE/FR detection, OpenAI fallback |
| `test_complexity_analyzer.py` | Query complexity classification |
| `test_escalation.py` | Keyword + LLM escalation detection |
| `test_zendesk_client.py` | Zendesk API client |
| `test_email_client.py` | IMAP integration, monitor lifecycle |

---

## Project Structure

```
├── config/
│   └── settings.py              # Pydantic settings (env vars)
├── src/
│   ├── api/main.py              # FastAPI app + all endpoints
│   ├── ingestion/
│   │   ├── parsers/             # PDF, DOCX, Email, CSV parsers
│   │   ├── chunker.py           # Token-based chunking
│   │   ├── embedder.py          # OpenAI embedding wrapper
│   │   └── pipeline.py          # Ingestion orchestration + version tracking
│   ├── retrieval/
│   │   ├── pinecone_client.py   # Vector DB operations
│   │   └── reranker.py          # Result reranking + approved response boost
│   ├── generation/
│   │   ├── response_generator.py  # Full RAG query pipeline
│   │   ├── language_detector.py   # EN/DE/FR detection
│   │   ├── complexity_analyzer.py # Brief/moderate/detailed classification
│   │   └── escalation_classifier.py # Keyword + LLM escalation check
│   ├── integrations/
│   │   ├── zendesk_client.py    # Zendesk API (tickets, notes)
│   │   └── email_client.py      # IMAP monitoring + draft generation
│   ├── feedback/
│   │   └── learning_loop.py     # Approved response capture + embedding
│   └── audit/
│       └── logger.py            # Structured query audit logging
├── scripts/                     # CLI tools (query, reindex, monitor)
├── tools/
│   └── test_ui.py              # Streamlit testing UI for RAG calibration
├── zendesk-app/                 # Zendesk sidebar widget
├── tests/                       # 83 unit tests
├── data/                        # Documents + persistent data files
├── Dockerfile                   # Container definition
└── docker-compose.yml           # Container orchestration
```
