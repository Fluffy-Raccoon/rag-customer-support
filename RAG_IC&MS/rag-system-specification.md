# Technical Specification: RAG-Based Customer Support System

## Project Overview

Build a Retrieval-Augmented Generation (RAG) system that ingests company documentation and historical support interactions to generate draft responses for customer inquiries. The system assists human agents by providing contextual, source-cited response drafts while learning from approved responses over time.

---

## 1. System Architecture

### 1.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT INGESTION PIPELINE                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   PDF    │  │   Word   │  │  Email   │  │  Zendesk │  │ Approved │  │
│  │  Parser  │  │  Parser  │  │  Parser  │  │   CSV    │  │ Responses│  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       └──────────────┴──────────────┴──────────────┴──────────────┘      │
│                                    │                                     │
│                          ┌─────────▼─────────┐                          │
│                          │  Text Processor   │                          │
│                          │  (Chunking, Clean)│                          │
│                          └─────────┬─────────┘                          │
│                                    │                                     │
│                          ┌─────────▼─────────┐                          │
│                          │  OpenAI Embedding │                          │
│                          │     Service       │                          │
│                          └─────────┬─────────┘                          │
│                                    │                                     │
│                          ┌─────────▼─────────┐                          │
│                          │  Pinecone Vector  │                          │
│                          │     Database      │                          │
│                          └───────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        QUERY & RESPONSE PIPELINE                         │
│                                                                          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │    Email     │      │   Zendesk    │      │   Manual     │          │
│  │  (eM Client) │      │   Trigger    │      │   Trigger    │          │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘          │
│         └────────────────────┬────────────────────────┘                  │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │ Language Detector │                                │
│                    │   (EN/DE/FR)      │                                │
│                    └─────────┬─────────┘                                │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │  Query Embedding  │                                │
│                    │     (OpenAI)      │                                │
│                    └─────────┬─────────┘                                │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │  Pinecone Search  │                                │
│                    │  (Cross-lingual)  │                                │
│                    └─────────┬─────────┘                                │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │   LLM Response    │                                │
│                    │   Generation      │                                │
│                    │   (OpenAI GPT-4)  │                                │
│                    └─────────┬─────────┘                                │
│                              │                                           │
│                    ┌─────────▼─────────┐                                │
│                    │  Draft Output +   │                                │
│                    │  Source Citations │                                │
│                    └─────────┬─────────┘                                │
│                              │                                           │
│         ┌────────────────────┴────────────────────┐                     │
│         │                                         │                     │
│  ┌──────▼──────┐                         ┌───────▼───────┐             │
│  │   Zendesk   │                         │     Email     │             │
│  │Internal Note│                         │    Draft      │             │
│  └─────────────┘                         └───────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        FEEDBACK LEARNING LOOP                            │
│                                                                          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │   Human      │      │   Approved   │      │  Re-embed &  │          │
│  │   Review     │─────▶│   Response   │─────▶│  Index       │          │
│  │   & Edit     │      │   Storage    │      │  (Pinecone)  │          │
│  └──────────────┘      └──────────────┘      └──────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Component           | Technology                      | Purpose                                                |
| ------------------- | ------------------------------- | ------------------------------------------------------ |
| Vector Database     | Pinecone                        | Store and retrieve document embeddings                 |
| Embeddings          | OpenAI `text-embedding-3-large` | Generate semantic embeddings for documents and queries |
| LLM                 | OpenAI GPT-4 or GPT-4-turbo     | Generate response drafts                               |
| Language Detection  | `langdetect` or OpenAI          | Auto-detect query language                             |
| Document Parsing    | See Section 2.2                 | Extract text from various formats                      |
| Backend Framework   | Python (FastAPI or Flask)       | API layer and orchestration                            |
| Zendesk Integration | Zendesk API                     | Read tickets, write internal notes                     |
| Email Integration   | eM Client plugin or IMAP        | Manual trigger workflow                                |

---

## 2. Document Ingestion Pipeline

### 2.1 Supported Document Types

| Format | Parser | Notes |
|--------|--------|-------|
| PDF | `PyMuPDF` (fitz) or `pdfplumber` | Handle scanned PDFs with OCR fallback using `pytesseract` |
| Word (.docx) | `python-docx` | Extract text, tables, headers |
| Word (.doc) | Convert via LibreOffice, then parse as .docx | Legacy format support |
| Email (.eml/.msg) | `email` stdlib or `extract-msg` | Parse body, attachments |
| Zendesk CSV | `pandas` | Historical ticket export |

### 2.2 Text Processing Pipeline

```python
# Pseudocode for ingestion pipeline
def ingest_document(file_path, doc_type):
    # 1. Parse document
    raw_text = parse_document(file_path, doc_type)
    
    # 2. Clean text
    cleaned_text = clean_text(raw_text)
    
    # 3. Chunk text (with overlap for context continuity)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=512,      # tokens
        overlap=50           # tokens
    )
    
    # 4. Generate metadata for each chunk
    for chunk in chunks:
        chunk.metadata = {
            "source_file": file_path,
            "section": extract_section_header(chunk),
            "document_type": doc_type,
            "language": detect_language(chunk.text),
            "ingestion_date": datetime.now().isoformat(),
            "version": get_document_version(file_path)
        }
    
    # 5. Generate embeddings
    embeddings = openai.embeddings.create(
        model="text-embedding-3-large",
        input=[chunk.text for chunk in chunks]
    )
    
    # 6. Upsert to Pinecone
    pinecone_index.upsert(vectors=zip(chunk_ids, embeddings, metadatas))
```

### 2.3 Chunking Strategy

- **Chunk size:** 512 tokens (optimal for retrieval precision)
- **Overlap:** 50 tokens (maintains context across chunk boundaries)
- **Boundary awareness:** Prefer splitting at paragraph/section breaks
- **Metadata preservation:** Each chunk retains reference to source document and section

### 2.4 Zendesk Historical Data Import

```python
# CSV columns expected from Zendesk export:
# ticket_id, created_at, subject, description, agent_response, status, tags

def import_zendesk_csv(csv_path):
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Create Q&A pair document
        qa_text = f"""
        Customer Question: {row['subject']}
        {row['description']}
        
        Agent Response: {row['agent_response']}
        """
        
        # Process and embed
        ingest_document(qa_text, doc_type="zendesk_history")
```

---

## 3. Query & Response Pipeline

### 3.1 Query Processing Flow

```python
def process_customer_query(query_text, source_channel):
    # 1. Detect language
    detected_language = detect_language(query_text)  # EN, DE, or FR
    
    # 2. Analyze query complexity (for response length calibration)
    complexity = analyze_complexity(query_text)
    # Returns: "brief", "moderate", "detailed"
    
    # 3. Generate query embedding
    query_embedding = openai.embeddings.create(
        model="text-embedding-3-large",
        input=query_text
    ).data[0].embedding
    
    # 4. Retrieve relevant chunks from Pinecone
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=10,
        include_metadata=True
    )
    
    # 5. Build context from retrieved chunks
    context = build_context(results)
    
    # 6. Generate response draft
    response = generate_response(
        query=query_text,
        context=context,
        language=detected_language,
        complexity=complexity
    )
    
    # 7. Format output with citations
    formatted_response = format_with_citations(response, results)
    
    # 8. Check for escalation need
    escalation_check = check_escalation_need(query_text, context, response)
    
    return {
        "draft": formatted_response,
        "citations": extract_citations(results),
        "escalation": escalation_check,
        "detected_language": detected_language
    }
```

### 3.2 LLM Prompt Template

```python
RESPONSE_GENERATION_PROMPT = """
You are a customer support assistant drafting responses for a human agent to review.

INSTRUCTIONS:
- Respond in {detected_language} (the same language as the customer's query)
- Match the response length to the query complexity:
  - Brief questions → Concise, direct answers (1-2 paragraphs)
  - Technical/complex questions → Detailed, step-by-step responses
- Tone: Friendly, respectful, proactive, and professional
- If the provided context does not fully answer the question, acknowledge what you can answer and flag what needs clarification or escalation
- Always cite your sources using [Source: document_name, section] format
- If the query requires technical expertise, access to internal systems, or policy decisions beyond documentation, recommend escalation

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER QUERY:
{query}

DRAFT RESPONSE:
"""
```

### 3.3 Escalation Detection

The system should flag for escalation when:

```python
ESCALATION_TRIGGERS = [
    "billing dispute",
    "refund request over threshold",
    "legal threat",
    "security incident",
    "data breach",
    "account compromise",
    "executive escalation request",
    "regulatory compliance",
    "unable to find relevant documentation",
    "conflicting information in sources",
    "requires system access or changes",
    "custom development request"
]

def check_escalation_need(query, context, response):
    # LLM-based classification
    prompt = f"""
    Analyze if this support query requires human escalation beyond a standard response.
    
    Query: {query}
    Available context: {context}
    Draft response: {response}
    
    Escalate if:
    - Query involves billing disputes, refunds, legal, security, or compliance
    - Documentation is insufficient or conflicting
    - Request requires system access or policy decisions
    - Customer explicitly requests manager/escalation
    
    Return JSON: {{"needs_escalation": bool, "reason": str}}
    """
    return llm_classify(prompt)
```

### 3.4 Response Output Format

```
DRAFT RESPONSE
─────────────────────────────────────────────────────────────
[Generated response text in detected language]

REFERENCES
─────────────────────────────────────────────────────────────
[1] Product Manual v2.3 - Section 4.2: Installation
[2] FAQ Document - Troubleshooting Connection Issues
[3] Previous Ticket #12345 - Similar Issue Resolution

ESCALATION RECOMMENDATION
─────────────────────────────────────────────────────────────
⚠️ ESCALATE: This query involves a billing dispute requiring
   finance team review.
   
   OR
   
✓ No escalation needed
```

---

## 4. Multilingual Support

### 4.1 Language Detection

```python
from langdetect import detect

def detect_language(text):
    lang = detect(text)
    if lang in ['en', 'de', 'fr']:
        return lang
    return 'en'  # Default fallback
```

### 4.2 Cross-Lingual Retrieval Strategy

OpenAI's `text-embedding-3-large` supports multilingual embeddings, enabling cross-lingual semantic search. This means:

- A German query can retrieve relevant English documentation
- Retrieved content is translated in the response generation step

```python
def generate_response_multilingual(query, context, target_language):
    # Context may contain mixed languages
    # LLM translates and synthesizes in target language
    
    prompt = f"""
    Generate a response in {target_language}.
    
    The context below may be in English, German, or French.
    Translate and synthesize relevant information into your response.
    
    Context: {context}
    Query: {query}
    """
    return llm_generate(prompt)
```

---

## 5. Feedback Learning Loop

### 5.1 Approved Response Capture

When an agent approves and sends a response (with or without edits):

```python
def capture_approved_response(original_query, draft_response, final_response, agent_edits):
    # Store the approved Q&A pair
    approved_data = {
        "query": original_query,
        "draft": draft_response,
        "final": final_response,
        "edits_made": agent_edits,
        "approved_at": datetime.now().isoformat(),
        "agent_id": current_agent_id
    }
    
    # Save to approved responses database
    save_to_approved_db(approved_data)
    
    # Create new training document
    training_doc = f"""
    Customer Question: {original_query}
    
    Approved Response: {final_response}
    """
    
    # Embed and index in Pinecone with special metadata
    embed_and_index(
        text=training_doc,
        metadata={
            "source_type": "approved_response",
            "approval_date": datetime.now().isoformat(),
            "original_ticket_id": ticket_id
        }
    )
```

### 5.2 Weighting Approved Responses

Approved responses should be weighted higher in retrieval:

```python
def retrieve_with_weighting(query_embedding, top_k=10):
    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k * 2,  # Fetch more to allow reranking
        include_metadata=True
    )
    
    # Boost approved responses
    for result in results:
        if result.metadata.get("source_type") == "approved_response":
            result.score *= 1.3  # 30% boost
    
    # Re-sort and return top_k
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]
```

---

## 6. Integration Specifications

### 6.1 Zendesk Integration

**API Requirements:**
- Zendesk API token with read/write permissions
- Subdomain configuration

**Workflow:**

```python
# Trigger: Agent clicks "Generate Draft" button in Zendesk
# Implementation: Zendesk app (sidebar widget) or webhook

def zendesk_generate_draft(ticket_id):
    # 1. Fetch ticket details
    ticket = zendesk_api.tickets.show(ticket_id)
    
    # 2. Extract customer message (latest comment)
    customer_message = get_latest_customer_message(ticket)
    
    # 3. Generate draft
    result = process_customer_query(
        query_text=customer_message,
        source_channel="zendesk"
    )
    
    # 4. Post as internal note
    zendesk_api.tickets.update(
        ticket_id,
        comment={
            "body": format_internal_note(result),
            "public": False  # Internal note
        }
    )
    
def format_internal_note(result):
    return f"""
    🤖 AI-GENERATED DRAFT (Review before sending)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    {result['draft']}
    
    📚 SOURCES:
    {chr(10).join(result['citations'])}
    
    {format_escalation(result['escalation'])}
    
    Language detected: {result['detected_language'].upper()}
    """
```

**Zendesk App Structure:**
```
zendesk-app/
├── manifest.json
├── assets/
│   └── iframe.html
├── src/
│   └── app.js
└── translations/
    └── en.json
```

### 6.2 Email Integration (eM Client)

**Option A: IMAP Monitoring (Recommended for local deployment)**

```python
# Monitor inbox, manual trigger via folder move or flag

import imaplib
import email

def check_for_trigger(imap_config):
    # Connect to mailbox
    mail = imaplib.IMAP4_SSL(imap_config['server'])
    mail.login(imap_config['user'], imap_config['password'])
    
    # Check "Generate Draft" folder (user moves emails here)
    mail.select('"Generate Draft"')
    
    _, messages = mail.search(None, 'UNSEEN')
    
    for msg_id in messages[0].split():
        # Fetch and process email
        _, data = mail.fetch(msg_id, '(RFC822)')
        email_message = email.message_from_bytes(data[0][1])
        
        # Generate draft
        result = process_customer_query(
            query_text=extract_email_body(email_message),
            source_channel="email"
        )
        
        # Save draft to "Drafts" folder or display in UI
        save_draft(email_message, result)
```

**Option B: eM Client Rule + Local Script**

Configure eM Client rule:
1. When email moved to "Generate Draft" folder
2. Execute script: `python generate_draft.py --email-file %file%`

---

## 7. Manual Re-indexing Workflow

### 7.1 Document Update Process

```python
# CLI tool for manual re-indexing

def reindex_documents(document_paths, full_reindex=False):
    if full_reindex:
        # Clear existing vectors for these documents
        clear_document_vectors(document_paths)
    
    for path in document_paths:
        print(f"Processing: {path}")
        
        # Detect document type
        doc_type = detect_document_type(path)
        
        # Run ingestion pipeline
        ingest_document(path, doc_type)
        
        print(f"✓ Indexed: {path}")
    
    print(f"\nReindexing complete. {len(document_paths)} documents processed.")

# Usage:
# python reindex.py --documents /path/to/updated/docs --full-reindex
```

### 7.2 Version Tracking

```python
# Track document versions to avoid duplicate indexing

def get_document_version(file_path):
    # Use file hash + modification time
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    mod_time = os.path.getmtime(file_path)
    
    return f"{file_hash}_{mod_time}"

def is_document_changed(file_path, indexed_versions):
    current_version = get_document_version(file_path)
    return current_version != indexed_versions.get(file_path)
```

---

## 8. Configuration

### 8.1 Environment Variables

```bash
# .env file

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_CHAT_MODEL=gpt-4-turbo

# Pinecone
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=customer-support-rag

# Zendesk
ZENDESK_SUBDOMAIN=yourcompany
ZENDESK_EMAIL=support@yourcompany.com
ZENDESK_API_TOKEN=...

# Email (IMAP)
IMAP_SERVER=mail.yourcompany.com
IMAP_USER=support@yourcompany.com
IMAP_PASSWORD=...
IMAP_TRIGGER_FOLDER=Generate Draft

# System
LOG_LEVEL=INFO
CHUNK_SIZE=512
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=10
```

### 8.2 Pinecone Index Configuration

```python
# Index setup
pinecone.create_index(
    name="customer-support-rag",
    dimension=3072,  # text-embedding-3-large dimension
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

---

## 9. Project Structure

```
customer-support-rag/
├── README.md
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── parsers/
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── email_parser.py
│   │   │   └── zendesk_parser.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pinecone_client.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── response_generator.py
│   │   ├── language_detector.py
│   │   └── escalation_classifier.py
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── zendesk_client.py
│   │   └── email_client.py
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── learning_loop.py
│   └── api/
│       ├── __init__.py
│       └── main.py  # FastAPI app
├── scripts/
│   ├── reindex.py
│   ├── import_zendesk_csv.py
│   └── test_query.py
├── zendesk-app/
│   ├── manifest.json
│   └── src/
├── tests/
│   └── ...
└── docs/
    └── ...
```

---

## 10. Implementation Phases

### Phase 1: Core RAG Pipeline (Week 1-2)
- [ ] Set up Pinecone index
- [ ] Implement document parsers (PDF, DOCX, Email)
- [ ] Build chunking and embedding pipeline
- [ ] Create basic query-response flow
- [ ] Implement citation extraction

### Phase 2: Multilingual & Quality (Week 3)
- [ ] Add language detection
- [ ] Implement cross-lingual retrieval
- [ ] Add response length calibration
- [ ] Build escalation detection
- [ ] Test with sample documents in EN/DE/FR

### Phase 3: Integrations (Week 4)
- [ ] Build Zendesk app/integration
- [ ] Implement internal note posting
- [ ] Set up email monitoring (eM Client)
- [ ] Create manual trigger workflows

### Phase 4: Feedback Loop (Week 5)
- [ ] Implement approved response capture
- [ ] Build re-embedding pipeline for approved responses
- [ ] Add response weighting logic
- [ ] Create admin dashboard for monitoring

### Phase 5: Testing & Deployment (Week 6)
- [ ] End-to-end testing with real documents
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deployment to local infrastructure
- [ ] Agent training

---

## 11. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Draft acceptance rate | >70% | % of drafts sent with minor or no edits |
| Response time reduction | 50% | Average time to first response |
| Citation accuracy | >95% | Spot checks on source validity |
| Language detection accuracy | >98% | Automated testing |
| Escalation precision | >90% | False positive/negative tracking |

---

## 12. Open Questions for Infrastructure Team

Before development begins, clarify:

1. **Server specifications:** What hardware is available for local deployment? (CPU/GPU, RAM, storage)
2. **Network access:** Can the local server reach OpenAI and Pinecone APIs?
3. **Authentication:** How will agents authenticate to the system?
4. **Backup:** What backup procedures exist for the vector database?
5. **Monitoring:** What monitoring/alerting infrastructure is available?
6. **eM Client version:** Which version is deployed? (Affects integration approach)

---

## 13. Security Considerations

- All API keys stored in environment variables, never in code
- Zendesk API uses token-based authentication
- Local deployment keeps customer data on-premises
- Pinecone data encrypted at rest and in transit
- Audit logging for all generated responses
- Agent authentication required for all triggers

---

*Document Version: 1.0*  
*Last Updated: February 2026*  
*Ready for Claude Code Implementation*
