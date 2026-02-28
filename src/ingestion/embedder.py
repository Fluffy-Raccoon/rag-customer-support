import logging

from openai import OpenAI

from config import get_settings

logger = logging.getLogger(__name__)

# OpenAI allows max 2048 texts per embedding call
_BATCH_SIZE = 2048


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info("Embedded batch %d-%d of %d texts", i, i + len(batch), len(texts))

    return all_embeddings
