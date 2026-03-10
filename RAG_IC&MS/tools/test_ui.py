"""
Streamlit testing UI for RAG pipeline calibration.

Run from project root:
    streamlit run tools/test_ui.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_settings
from src.generation.response_generator import process_customer_query_debug

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Test UI", page_icon="?", layout="wide")
st.title("RAG Pipeline Test UI")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Sidebar: settings & history
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings (read-only)")
    try:
        settings = get_settings()
        st.text(f"Chat model:     {settings.anthropic_chat_model}")
        st.text(f"Embedding model: {settings.openai_embedding_model}")
        st.text(f"Top-K:           {settings.retrieval_top_k}")
        st.text(f"Chunk size:      {settings.chunk_size}")
        st.text(f"Chunk overlap:   {settings.chunk_overlap}")
    except Exception as e:
        st.error(f"Could not load settings: {e}")

    st.divider()
    st.header("Query History")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history)):
            label = entry["query"][:60] + ("..." if len(entry["query"]) > 60 else "")
            if st.button(label, key=f"history_{i}"):
                st.session_state.selected_history = len(st.session_state.history) - 1 - i

        st.divider()
        history_json = json.dumps(st.session_state.history, indent=2, default=str)
        st.download_button(
            "Export All History (JSON)",
            data=history_json,
            file_name=f"rag_test_history_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
        )
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No queries yet.")

# ---------------------------------------------------------------------------
# Main: query input
# ---------------------------------------------------------------------------
query = st.text_area(
    "Customer query",
    height=150,
    placeholder="Paste a customer email or question here...",
)

run_clicked = st.button("Run Query", type="primary", disabled=not query.strip())

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
result = None

if run_clicked and query.strip():
    with st.spinner("Running pipeline..."):
        start = time.time()
        try:
            result = process_customer_query_debug(query.strip())
            duration = time.time() - start
            result["_duration_s"] = round(duration, 2)
            result["_timestamp"] = datetime.now().isoformat()
            result["query"] = query.strip()

            st.session_state.history.append(result)
            st.session_state.selected_history = len(st.session_state.history) - 1
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

# Show result from history if clicked
if result is None and "selected_history" in st.session_state:
    idx = st.session_state.selected_history
    if 0 <= idx < len(st.session_state.history):
        result = st.session_state.history[idx]

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if result:
    st.divider()

    # Quick stats bar
    cols = st.columns(4)
    cols[0].metric("Language", result.get("detected_language", "?").upper())
    cols[1].metric("Complexity", result.get("complexity", "?").title())
    esc = result.get("escalation", {})
    cols[2].metric("Escalation", "Yes" if esc.get("needs_escalation") else "No")
    cols[3].metric("Duration", f"{result.get('_duration_s', '?')}s")

    # Tabs
    tab_draft, tab_retrieval, tab_prompt, tab_meta = st.tabs([
        "Draft Response",
        "Retrieval Debug",
        "Context & Prompt",
        "Pipeline Metadata",
    ])

    # --- Tab 1: Draft Response ---
    with tab_draft:
        st.markdown(result["draft"])
        st.divider()
        st.subheader("Citations")
        for cite in result.get("citations", []):
            st.text(cite)
        if esc.get("needs_escalation"):
            st.warning(f"Escalation recommended: {esc.get('reason', '')}")

    # --- Tab 2: Retrieval Debug ---
    with tab_retrieval:
        debug = result.get("debug", {})

        st.subheader("Raw Pinecone Results")
        raw = debug.get("raw_results", [])
        if raw:
            df_raw = pd.DataFrame(raw)
            df_raw["text_preview"] = df_raw["text"].str[:200]
            st.dataframe(
                df_raw[["score", "source", "section", "text_preview", "id"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No raw results returned.")

        st.subheader("After Reranking")
        reranked = debug.get("reranked_results", [])
        if reranked:
            df_reranked = pd.DataFrame(reranked)
            df_reranked["text_preview"] = df_reranked["text"].str[:200]
            st.dataframe(
                df_reranked[["score", "source", "section", "text_preview", "id"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No reranked results.")

        # Show full chunk text in expanders
        if reranked:
            st.subheader("Full Chunk Text (Reranked)")
            for i, chunk in enumerate(reranked):
                with st.expander(f"[{i+1}] {chunk.get('source', '?')} — score: {chunk.get('score', 0):.4f}"):
                    st.text(chunk.get("text", ""))

    # --- Tab 3: Context & Prompt ---
    with tab_prompt:
        st.subheader("Context Sent to LLM")
        with st.expander("Show context", expanded=False):
            st.code(debug.get("context_sent_to_llm", ""), language=None)

        st.subheader("Full Prompt Sent to Claude")
        with st.expander("Show prompt", expanded=False):
            st.code(debug.get("full_prompt", ""), language=None)

    # --- Tab 4: Pipeline Metadata ---
    with tab_meta:
        st.subheader("Query Info")
        st.json({
            "query": result.get("query", ""),
            "detected_language": result.get("detected_language", ""),
            "complexity": result.get("complexity", ""),
            "timestamp": result.get("_timestamp", ""),
            "duration_s": result.get("_duration_s", ""),
        })
        st.subheader("Escalation")
        st.json(result.get("escalation", {}))
        st.subheader("Settings Used")
        st.json(debug.get("settings_used", {}))

    # Export single result
    st.divider()
    result_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        "Export This Result (JSON)",
        data=result_json,
        file_name=f"rag_result_{datetime.now():%Y%m%d_%H%M%S}.json",
        mime="application/json",
    )
