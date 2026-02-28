from src.retrieval.pinecone_client import SearchResult

# Boost factor for approved responses (prep for Phase 4 feedback loop)
APPROVED_RESPONSE_BOOST = 1.3


def rerank(results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
    """Rerank search results, boosting approved responses."""
    for result in results:
        if result.metadata.get("source_type") == "approved_response":
            result.score *= APPROVED_RESPONSE_BOOST

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]
