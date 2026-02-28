import hashlib
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import tiktoken
from langdetect import detect, LangDetectException

ENCODING = tiktoken.get_encoding("cl100k_base")

# Regex for splitting at paragraph boundaries
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[str]:
    """Split text into token-based chunks with overlap.

    Prefers splitting at paragraph boundaries when possible.
    """
    paragraphs = PARAGRAPH_SPLIT.split(text.strip())
    tokens: list[int] = []
    # Track paragraph boundary positions (in token indices)
    boundaries: list[int] = []

    for para in paragraphs:
        para_tokens = ENCODING.encode(para)
        boundaries.append(len(tokens))
        tokens.extend(para_tokens)

    if not tokens:
        return []

    boundary_set = set(boundaries)
    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size

        if end >= len(tokens):
            chunk_tokens = tokens[start:]
            chunk_text_str = ENCODING.decode(chunk_tokens).strip()
            if chunk_text_str:
                chunks.append(chunk_text_str)
            break

        # Try to find a paragraph boundary near the end for a clean split
        best_boundary = None
        for b in sorted(boundary_set):
            if start + chunk_size // 2 <= b <= end:
                best_boundary = b

        split_at = best_boundary if best_boundary else end

        chunk_tokens = tokens[start:split_at]
        chunk_text_str = ENCODING.decode(chunk_tokens).strip()
        if chunk_text_str:
            chunks.append(chunk_text_str)

        start = split_at - overlap

    return chunks


def build_chunks_with_metadata(
    text: str,
    source_file: str,
    document_type: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Chunk text and attach metadata to each chunk."""
    raw_chunks = chunk_text(text, chunk_size, overlap)
    lang = _detect_language(text[:500])
    version = get_document_version(source_file) if os.path.isfile(source_file) else None

    result: list[Chunk] = []
    for i, chunk_str in enumerate(raw_chunks):
        section = _extract_section_header(chunk_str)
        metadata = {
            "source_file": source_file,
            "section": section,
            "document_type": document_type,
            "language": lang,
            "ingestion_date": datetime.now(timezone.utc).isoformat(),
            "chunk_index": i,
        }
        if version:
            metadata["version"] = version
        result.append(Chunk(text=chunk_str, metadata=metadata))

    return result


def get_document_version(file_path: str) -> str:
    """Generate a version string from file hash + modification time."""
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    mod_time = os.path.getmtime(file_path)
    return f"{file_hash}_{mod_time}"


def _detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in ("en", "de", "fr") else "en"
    except LangDetectException:
        return "en"


def _extract_section_header(text: str) -> str:
    """Try to extract a section header from the first line of a chunk."""
    first_line = text.split("\n", 1)[0].strip()
    # Heuristic: short lines that look like headers
    if len(first_line) < 120 and not first_line.endswith("."):
        return first_line
    return ""
