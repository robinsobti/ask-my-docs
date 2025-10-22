from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Ensure project root is importable when launched via `streamlit run`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_EMBEDDER_MODEL,
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
    from src.embedder import Embedder  # type: ignore[attr-defined]  # noqa: F401
except ImportError:
    Embedder = None

try:
    import src.generator as generator  # type: ignore[attr-defined]  # noqa: F401
except ImportError:
    generator = None

import src.obs as obs  # type: ignore[attr-defined]  # noqa: F401
from src.obs import new_ctx, timer, append_query_log

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
    if st.session_state["run_id"]:
        st.caption(f"Logs written to runs/{st.session_state['run_id']}/queries.jsonl")
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
raw_query = query_params.get("q", "")
if isinstance(raw_query, list):
    raw_query = raw_query[0] if raw_query else ""
else:
    raw_query = raw_query or ""

if "question" not in st.session_state:
    st.session_state["question"] = raw_query


def _sync_query_params_from_question() -> None:
    """Keep the URL's `q` query parameter aligned with the current question."""
    question_value = st.session_state.get("question", "").strip()

    current_param = st.query_params.get("q", "")
    if isinstance(current_param, list):
        current_param = current_param[0] if current_param else ""
    else:
        current_param = current_param or ""

    if question_value:
        if current_param != question_value:
            st.query_params["q"] = question_value
    elif "q" in st.query_params:
        del st.query_params["q"]


def question_text_changed() -> None:
    _sync_query_params_from_question()


question = st.text_input(
    "Ask a question about your documents",
    value=st.session_state.get("question", ""),
    key="question",
    on_change=question_text_changed,
)

_sync_query_params_from_question()

ask_disabled = not question.strip()


def ask_button_clicked() -> None:
    """Placeholder callback to preserve button on_click wiring."""
    params = {
        "mode": retrieval_mode,
        "k": top_k,
        "alpha": hybrid_alpha,
        "generator_model": generator_model,
        "embed_model": DEFAULT_EMBEDDER_MODEL
    }
    ctx: Dict[str, Any] = new_ctx(st.session_state["run_id"], st.session_state.get("question", "").strip(), params)
    with timer("retrieve", ctx):
        results = retriever.retrieve(query=question, mode=retrieval_mode, k=top_k, alpha=hybrid_alpha, embedder=Embedder())
        results_to_show = results[:show_sources]
        prompt = generator.build_prompt(query=question, docs = results)
        #TODO fix arguments
        generator.generate_answer(prompt = prompt, model = generator_model, max_tokens=500)
    append_query_log(Path(st.session_state["log_path"]), record=ctx)

ask_clicked = st.button("Ask", disabled=ask_disabled, type="primary", on_click=ask_button_clicked)
