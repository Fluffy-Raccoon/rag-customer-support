import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_text, build_chunks_with_metadata

import tiktoken

ENCODING = tiktoken.get_encoding("cl100k_base")


def test_chunk_text_basic():
    """Chunks a simple text into expected number of pieces."""
    # Create text that is ~1200 tokens (should produce ~3 chunks at 512 tokens)
    text = "This is a test sentence. " * 300
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) >= 2
    for chunk in chunks:
        tokens = ENCODING.encode(chunk)
        # Allow some slack for boundary-aware splitting
        assert len(tokens) <= 600


def test_chunk_text_short():
    """Short text that fits in one chunk."""
    text = "Hello, this is a short text."
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty():
    """Empty text returns no chunks."""
    assert chunk_text("", chunk_size=512, overlap=50) == []
    assert chunk_text("   ", chunk_size=512, overlap=50) == []


def test_chunk_text_respects_paragraphs():
    """Chunker should prefer paragraph boundaries."""
    para1 = "First paragraph. " * 100
    para2 = "Second paragraph. " * 100
    text = f"{para1}\n\n{para2}"
    chunks = chunk_text(text, chunk_size=512, overlap=50)
    assert len(chunks) >= 2


def test_build_chunks_with_metadata():
    """Metadata is attached correctly to each chunk."""
    text = "Test content for metadata. " * 50
    chunks = build_chunks_with_metadata(
        text=text,
        source_file="test.pdf",
        document_type="pdf",
        chunk_size=512,
        overlap=50,
    )
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["source_file"] == "test.pdf"
        assert chunk.metadata["document_type"] == "pdf"
        assert chunk.metadata["language"] in ("en", "de", "fr")
        assert "ingestion_date" in chunk.metadata
        assert "chunk_index" in chunk.metadata
