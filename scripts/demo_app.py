from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when launched via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_HYBRID_ALPHA,
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_RETRIEVAL_TOP_K,
)

st.set_page_config(page_title="Ask-My-Docs", layout="wide")

try:
    import src.retriever as retriever  # noqa: F401
except ImportError as exc:  # pragma: no cover - surfaced in UI instead
    st.error(f"Failed to import retriever module: {exc}")
    st.stop()

try:
    import src.generator as generator  # type: ignore[attr-defined]  # noqa: F401
except ImportError:
    generator = None

try:
    import src.obs as obs  # type: ignore[attr-defined]  # noqa: F401
except ImportError:
    obs = None

if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
    st.session_state["log_path"] = None

st.title("Ask-My-Docs")
st.caption("RAG with hybrid retrieval and sources.")

mode_options = ["hybrid", "vector", "bm25"]
default_mode = DEFAULT_RETRIEVAL_MODE if DEFAULT_RETRIEVAL_MODE in mode_options else "hybrid"
default_mode_idx = mode_options.index(default_mode)

default_top_k = min(max(DEFAULT_RETRIEVAL_TOP_K, 1), 20)
default_alpha = min(max(float(DEFAULT_HYBRID_ALPHA), 0.0), 1.0)
default_rerank_depth = min(max(DEFAULT_RERANK_DEPTH, 10), 100)

if st.session_state["run_id"] is None and obs is not None:
    try:
        run_identifier, run_path = obs.init_run_log()
    except Exception as exc:  # pragma: no cover - surfaced to UI
        st.warning(f"Failed to initialize run log: {exc}")
    else:
        st.session_state["run_id"] = run_identifier
        st.session_state["log_path"] = run_path

with st.sidebar:
    st.header("Settings")
    retrieval_mode = st.selectbox("Mode", options=mode_options, index=default_mode_idx)
    top_k = st.slider("k (top docs)", min_value=1, max_value=20, value=default_top_k, step=1)
    hybrid_alpha = st.slider(
        "alpha (hybrid weight)",
        min_value=0.0,
        max_value=1.0,
        value=default_alpha,
        step=0.05,
    )
    if retrieval_mode != "hybrid":
        st.caption("Alpha ignored unless hybrid.")
    rerank_depth = st.slider("rerank_depth", min_value=10, max_value=100, value=default_rerank_depth, step=5)
    generator_model = st.text_input("Generator model", value=DEFAULT_GENERATOR_MODEL)
    show_sources = st.slider("Show sources", min_value=1, max_value=5, value=3, step=1)
    st.caption("Logs written to runs/<timestamp>/queries.jsonl")
    run_id = st.session_state.get("run_id")
    if run_id:
        st.caption(f"Run id: {run_id}")
    elif obs is None:
        st.caption("Run logging unavailable: obs module missing.")

settings = {
    "mode": retrieval_mode,
    "top_k": top_k,
    "alpha": hybrid_alpha,
    "rerank_depth": rerank_depth,
    "generator_model": generator_model,
    "show_sources": show_sources,
}

st.session_state["settings"] = settings

query_params = st.query_params
print(f"query_params: {query_params}")
raw_query = query_params.get("q", "")
print(f"raw_query: {raw_query}")
if isinstance(raw_query, list):
    initial_question = raw_query[0] if raw_query else ""
else:
    initial_question = raw_query or ""

def _sync_query_param() -> None:
    value = st.session_state.get("question", "").strip()
    params = dict(st.query_params)
    if value:
        params["q"] = value
    else:
        params.pop("q", None)
    st.query_params = params

question = st.text_input(
    "Ask a question about your documents",
    value=initial_question,
    key="question",
    on_change=_sync_query_param,
)
ask_disabled = not question.strip()
ask_clicked = st.button("Ask", disabled=ask_disabled, type="primary")
