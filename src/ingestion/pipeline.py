import json
import logging
import uuid
from pathlib import Path

from config import get_settings
from src.ingestion.chunker import build_chunks_with_metadata, get_document_version
from src.ingestion.embedder import get_embeddings
from src.ingestion.parsers.docx_parser import parse_docx
from src.ingestion.parsers.email_parser import parse_email
from src.ingestion.parsers.pdf_parser import parse_pdf
from src.ingestion.parsers.zendesk_parser import parse_zendesk_csv
from src.retrieval.pinecone_client import delete_by_source, upsert_vectors

logger = logging.getLogger(__name__)

# Extension → (parser function, document_type)
_PARSER_MAP = {
    ".pdf": (parse_pdf, "pdf"),
    ".docx": (parse_docx, "docx"),
    ".doc": (parse_docx, "doc"),  # assumes pre-converted to docx
    ".eml": (parse_email, "email"),
    ".msg": (parse_email, "email"),
}

# Persistent version tracking
_VERSIONS_FILE = Path("data/.indexed_versions.json")


def _load_indexed_versions() -> dict[str, str]:
    """Load indexed versions from persistent JSON file."""
    if _VERSIONS_FILE.exists():
        try:
            return json.loads(_VERSIONS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read version file, starting fresh")
    return {}


def _save_indexed_versions(versions: dict[str, str]) -> None:
    """Persist indexed versions to JSON file."""
    _VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _VERSIONS_FILE.write_text(json.dumps(versions, indent=2), encoding="utf-8")


_indexed_versions: dict[str, str] = _load_indexed_versions()


def is_document_changed(file_path: str) -> bool:
    """Check if a document has changed since last indexing."""
    current_version = get_document_version(file_path)
    return current_version != _indexed_versions.get(file_path)


def detect_document_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in _PARSER_MAP:
        return _PARSER_MAP[suffix][1]
    if suffix == ".csv":
        return "zendesk_csv"
    raise ValueError(f"Unsupported file type: {suffix}")


def ingest_document(
    file_path: str,
    full_reindex: bool = False,
) -> int:
    """Ingest a single document: parse → chunk → embed → upsert.

    Returns the number of chunks indexed.
    """
    settings = get_settings()
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Version check — skip if unchanged
    if not full_reindex and path.is_file():
        current_version = get_document_version(file_path)
        if _indexed_versions.get(file_path) == current_version:
            logger.info("Skipping unchanged file: %s", file_path)
            return 0

    doc_type = detect_document_type(file_path)

    # Handle Zendesk CSV separately (multiple documents per file)
    if doc_type == "zendesk_csv":
        return _ingest_zendesk_csv(file_path, settings)

    # Parse
    suffix = path.suffix.lower()
    parser_fn, _ = _PARSER_MAP[suffix]
    raw_text = parser_fn(file_path)

    if not raw_text.strip():
        logger.warning("No text extracted from: %s", file_path)
        return 0

    # Chunk
    chunks = build_chunks_with_metadata(
        text=raw_text,
        source_file=file_path,
        document_type=doc_type,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        return 0

    if full_reindex:
        delete_by_source(file_path)

    # Embed
    texts = [c.text for c in chunks]
    embeddings = get_embeddings(texts)

    # Build IDs and metadata (include text in metadata for retrieval)
    ids = [f"{path.stem}_{uuid.uuid4().hex[:8]}" for _ in chunks]
    metadatas = []
    for chunk in chunks:
        meta = {**chunk.metadata, "text": chunk.text}
        metadatas.append(meta)

    # Upsert
    upsert_vectors(ids, embeddings, metadatas)

    # Update version cache and persist
    if path.is_file():
        _indexed_versions[file_path] = get_document_version(file_path)
        _save_indexed_versions(_indexed_versions)

    logger.info("Indexed %d chunks from %s", len(chunks), file_path)
    return len(chunks)


def _ingest_zendesk_csv(csv_path: str, settings) -> int:
    """Ingest all Q&A pairs from a Zendesk CSV export."""
    documents = parse_zendesk_csv(csv_path)
    if not documents:
        return 0

    all_chunks = []
    for doc_text in documents:
        chunks = build_chunks_with_metadata(
            text=doc_text,
            source_file=csv_path,
            document_type="zendesk_history",
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    texts = [c.text for c in all_chunks]
    embeddings = get_embeddings(texts)

    ids = [f"zendesk_{uuid.uuid4().hex[:8]}" for _ in all_chunks]
    metadatas = [{**c.metadata, "text": c.text} for c in all_chunks]

    upsert_vectors(ids, embeddings, metadatas)
    logger.info("Indexed %d chunks from Zendesk CSV %s", len(all_chunks), csv_path)
    return len(all_chunks)


def ingest_directory(dir_path: str, full_reindex: bool = False) -> int:
    """Ingest all supported documents from a directory."""
    supported = set(_PARSER_MAP.keys()) | {".csv"}
    total = 0

    for path in sorted(Path(dir_path).rglob("*")):
        if path.suffix.lower() in supported:
            try:
                count = ingest_document(str(path), full_reindex=full_reindex)
                total += count
            except Exception:
                logger.exception("Failed to ingest: %s", path)

    logger.info("Directory ingestion complete: %d total chunks from %s", total, dir_path)
    return total
