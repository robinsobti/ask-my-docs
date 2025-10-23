from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

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
    PRICE_COMPLETION_PER_1K,
    PRICE_PROMPT_PER_1K
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
from src.obs import append_query_log, new_ctx, record_error, timer

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
    """Handle Ask button presses and log instrumentation."""
    if generator is None:
        st.error("Generator module is unavailable.")
        return
    if retrieval_mode in ("vector", "hybrid") and Embedder is None:
        st.error("Embedder module is required for the selected retrieval mode.")
        return
    if st.session_state.get("run_id") is None or st.session_state.get("log_path") is None:
        st.error("Run logging is not initialized.")
        return

    question_text = st.session_state.get("question", "").strip()
    if not question_text:
        st.warning(f"Question cannot be blank: {exc}")
        return
    
    params = {
        "mode": retrieval_mode,
        "k": top_k,
        "alpha": hybrid_alpha,
        "generator_model": generator_model,
        "embed_model": DEFAULT_EMBEDDER_MODEL,
        "rerank_depth": rerank_depth,
    }
    ctx: Dict[str, Any] = new_ctx(st.session_state["run_id"], question_text, params)

    embedder_instance = None
    results = []
    reranked_results = []
    prompt = ""
    answer_text = ""
    token_usage: Dict[str, Any] | None = None

    try:
        with timer("retrieve", ctx):
            if retrieval_mode in ("vector", "hybrid") and Embedder is not None:
                embedder_instance = Embedder(model_name=DEFAULT_EMBEDDER_MODEL)
            results = retriever.retrieve(
                query=question_text,
                mode=retrieval_mode,
                k=top_k,
                alpha=hybrid_alpha,
                embedder=embedder_instance,
            )
        ctx["top_docs"] = results[:show_sources]

        with timer("rerank", ctx):
            reranked_results = results  # Stub: no reranking implemented yet.
        ctx["reranked_docs"] = reranked_results[:show_sources]

        with timer("prompt_build", ctx):
            prompt = generator.build_prompt(query=question_text, docs=reranked_results)
        ctx["prompt_chars"] = len(prompt)

        with timer("llm_generate", ctx):
            answer_text, token_usage = generator.generate_answer(
                prompt=prompt,
                model=generator_model,
                max_tokens=500,
            )
        ctx["answer"] = answer_text
        ctx["answer_chars"] = len(answer_text)
        ctx["token_usage"] = token_usage

        cost_usd = None
        if token_usage is not None and (PRICE_PROMPT_PER_1K is not None or PRICE_COMPLETION_PER_1K is not None):
            prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
            cost = 0.0
            if PRICE_PROMPT_PER_1K is not None:
                cost += (prompt_tokens / 1000.0) * PRICE_PROMPT_PER_1K
            if PRICE_COMPLETION_PER_1K is not None:
                cost += (completion_tokens / 1000.0) * PRICE_COMPLETION_PER_1K
            cost_usd = round(cost, 6)
            ctx["cost_usd"] = cost_usd
        else:
            ctx["cost_usd"] = None

        with timer("render", ctx):
            st.session_state["last_results"] = reranked_results[:show_sources]
            st.session_state["last_answer"] = answer_text
            st.session_state["last_token_usage"] = token_usage
            st.session_state["last_cost_usd"] = ctx.get("cost_usd")
    except Exception as exc:  # pragma: no cover - surfaced in UI bv
        record_error(ctx, exc)
        stage_times = {
            name: stage_info["elapsed_ms"]
            for name, stage_info in ctx.get("stages", {}).items()
        }
        ctx["stage_times_ms"] = stage_times
        ctx["latency_ms"] = round(sum(stage_times.values()), 3) if stage_times else 0.0
        append_query_log(Path(st.session_state["log_path"]), record=ctx)
        st.error(f"Failed to process request: {exc}")
        return

    stage_times = {
        name: stage_info["elapsed_ms"]
        for name, stage_info in ctx.get("stages", {}).items()
    }
    ctx["stage_times_ms"] = stage_times
    ctx["latency_ms"] = round(sum(stage_times.values()), 3) if stage_times else 0.0

    append_query_log(Path(st.session_state["log_path"]), record=ctx)

ask_clicked = st.button("Ask", disabled=ask_disabled, type="primary", on_click=ask_button_clicked)
