import logging
from dataclasses import dataclass

from pinecone import Pinecone

from config import get_settings

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 100


@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
    text: str = ""


def _get_index():
    settings = get_settings()
    pc = Pinecone(api_key=settings.pinecone_api_key)
    return pc.Index(settings.pinecone_index_name)


def upsert_vectors(
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """Batch upsert vectors with metadata to Pinecone."""
    index = _get_index()
    vectors = [
        {"id": id_, "values": emb, "metadata": meta}
        for id_, emb, meta in zip(ids, embeddings, metadatas)
    ]

    for i in range(0, len(vectors), _UPSERT_BATCH):
        batch = vectors[i : i + _UPSERT_BATCH]
        index.upsert(vectors=batch)
        logger.info("Upserted batch %d-%d of %d vectors", i, i + len(batch), len(vectors))


def query(
    embedding: list[float],
    top_k: int = 10,
    filter_: dict | None = None,
) -> list[SearchResult]:
    """Query Pinecone for similar vectors."""
    index = _get_index()
    params = {
        "vector": embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if filter_:
        params["filter"] = filter_

    response = index.query(**params)

    results = []
    for match in response.matches:
        results.append(
            SearchResult(
                id=match.id,
                score=match.score,
                metadata=match.metadata or {},
                text=match.metadata.get("text", "") if match.metadata else "",
            )
        )
    return results


def delete_by_source(source_file: str) -> None:
    """Delete all vectors associated with a source file."""
    index = _get_index()
    # Pinecone serverless supports delete by metadata filter
    index.delete(filter={"source_file": {"$eq": source_file}})
    logger.info("Deleted vectors for source: %s", source_file)
