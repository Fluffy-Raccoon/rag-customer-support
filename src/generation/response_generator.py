import logging

from openai import OpenAI

from config import get_settings
from src.generation.complexity_analyzer import analyze_complexity
from src.generation.escalation_classifier import check_escalation_need
from src.generation.language_detector import detect_language
from src.ingestion.embedder import get_embeddings
from src.retrieval.pinecone_client import SearchResult, query as pinecone_query
from src.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

RESPONSE_PROMPT = """You are a customer support assistant drafting responses for a human agent to review.

INSTRUCTIONS:
- Respond in {detected_language} (the same language as the customer's query)
- The context below may be in English, German, or French. Translate and synthesize relevant information into your response in {detected_language}.
- Response length: {complexity_guidance}
- Tone: Friendly, respectful, proactive, and professional
- If the provided context does not fully answer the question, acknowledge what you can answer and flag what needs clarification or escalation
- Always cite your sources using [Source: document_name, section] format
- If the query requires technical expertise, access to internal systems, or policy decisions beyond documentation, recommend escalation

CONTEXT FROM KNOWLEDGE BASE:
{context}

CUSTOMER QUERY:
{query}

DRAFT RESPONSE:"""

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
}

COMPLEXITY_GUIDANCE = {
    "brief": "Keep your response to 1-2 short paragraphs. Be concise and direct.",
    "moderate": "Provide a clear, structured response of 2-3 paragraphs.",
    "detailed": "Provide a detailed, step-by-step response with thorough explanations.",
}


def process_customer_query(query_text: str) -> dict:
    """Full query pipeline: detect language → analyze complexity → embed → retrieve → generate → format."""
    settings = get_settings()

    # 1. Detect language
    detected_lang = detect_language(query_text)
    logger.info("Detected language: %s", detected_lang)

    # 2. Analyze query complexity
    complexity = analyze_complexity(query_text)
    logger.info("Query complexity: %s", complexity)

    # 3. Embed query
    query_embedding = get_embeddings([query_text])[0]

    # 4. Retrieve from Pinecone (fetch extra for reranking)
    raw_results = pinecone_query(
        embedding=query_embedding,
        top_k=settings.retrieval_top_k * 2,
    )

    # 5. Rerank
    results = rerank(raw_results, top_k=settings.retrieval_top_k)

    # 6. Build context
    context = _build_context(results)

    # 7. Generate response
    draft = _generate_response(query_text, context, detected_lang, complexity, settings)

    # 8. Check escalation
    escalation = check_escalation_need(query_text, context, draft)

    # 9. Format citations
    citations = _extract_citations(results)

    return {
        "draft": draft,
        "citations": citations,
        "escalation": escalation,
        "detected_language": detected_lang,
        "complexity": complexity,
    }


def _build_context(results: list[SearchResult]) -> str:
    """Build a context string from search results."""
    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source_file", "Unknown")
        section = result.metadata.get("section", "")
        text = result.metadata.get("text", result.text)
        header = f"[{i}] Source: {source}"
        if section:
            header += f", Section: {section}"
        context_parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(context_parts)


def _generate_response(
    query_text: str,
    context: str,
    detected_lang: str,
    complexity: str,
    settings,
) -> str:
    """Call the LLM to generate a draft response."""
    client = OpenAI(api_key=settings.openai_api_key)
    lang_name = LANGUAGE_NAMES.get(detected_lang, "English")
    guidance = COMPLEXITY_GUIDANCE.get(complexity, COMPLEXITY_GUIDANCE["moderate"])

    prompt = RESPONSE_PROMPT.format(
        detected_language=lang_name,
        complexity_guidance=guidance,
        context=context,
        query=query_text,
    )

    response = client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )

    return response.choices[0].message.content.strip()


def _extract_citations(results: list[SearchResult]) -> list[str]:
    """Extract formatted citation strings from search results."""
    citations = []
    seen = set()
    for i, result in enumerate(results, 1):
        source = result.metadata.get("source_file", "Unknown")
        section = result.metadata.get("section", "")
        key = (source, section)
        if key in seen:
            continue
        seen.add(key)
        cite = f"[{len(citations) + 1}] {source}"
        if section:
            cite += f" - {section}"
        citations.append(cite)
    return citations
