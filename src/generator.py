from __future__ import annotations

import textwrap
from typing import Any, Dict, Iterable, List, Tuple
from openai import OpenAI
from .config import OPENAI_API_KEY, OPENAI_MAX_TOKENS, OPENAI_TEMPERATURE, OPENAI_TOP_P, DEFAULT_GENERATOR_MODEL

openai_client = None
DEFAULT_MAX_TOKENS = 512    

def build_prompt(query: str, docs: Iterable[Dict[str, Any]]) -> str:
    """
    Construct a simple prompt that includes the question and numbered context passages.
    """
    context_sections: List[str] = []
    for index, doc in enumerate(docs, start=1):
        title = (doc.get("title") or "").strip() or "Untitled"
        source = (doc.get("source") or "").strip() or "unknown-source"
        text = (doc.get("text") or "").strip()
        excerpt = textwrap.shorten(text, width=600, placeholder="…") if text else ""
        section_lines = [
            f"[{index}] Title: {title}",
            f"Source: {source}",
        ]
        if excerpt:
            section_lines.append(excerpt)
        context_sections.append("\n".join(section_lines))

    context_block = "\n\n".join(context_sections) if context_sections else "No supporting documents were retrieved."
    prompt = (
        "You are a retrieval-augmented assistant. Use the numbered context passages to answer the question.\n"
        "Refer to sources inline using the notation [number]. If the context does not contain the answer, say so.\n"
        f"Question: {query.strip()}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )
    return prompt


def _extract_question(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.lower().startswith("question:"):
            return line.split(":", 1)[1].strip()
    return ""


def _extract_passages(prompt: str) -> List[Tuple[str, str]]:
    passages: List[Tuple[str, str]] = []
    in_context = False
    buffer: List[str] = []
    current_idx: str | None = None

    for line in prompt.splitlines():
        if not in_context:
            if line.strip().lower() == "context:":
                in_context = True
            continue

        stripped = line.strip()
        if not stripped:
            if buffer and current_idx is not None:
                passages.append((current_idx, "\n".join(buffer).strip()))
                buffer = []
                current_idx = None
            continue

        if stripped.startswith("[") and "]" in stripped.split()[0]:
            if buffer and current_idx is not None:
                passages.append((current_idx, "\n".join(buffer).strip()))
                buffer = []
            marker = stripped.split("]", 1)[0]
            current_idx = marker.lstrip("[").rstrip("]")
            remainder = stripped[len(marker) + 1 :].strip()
            if remainder:
                buffer.append(remainder)
        else:
            buffer.append(stripped)

    if buffer and current_idx is not None:
        passages.append((current_idx, "\n".join(buffer).strip()))

    return passages


def generate_answer_mock(prompt: str, *, model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Tuple[str, Dict[str, int | str]]:
    """
    Produce a placeholder answer using the prompt contents.
    """
    question = _extract_question(prompt)
    passages = _extract_passages(prompt)

    if not passages:
        answer_text = "I could not find supporting documents to answer that question."
    else:
        first_idx, first_passage = passages[0]
        summary = textwrap.shorten(first_passage.replace("\n", " "), width=max(80, min(240, max_tokens)), placeholder="…")
        if question:
            answer_text = f"Based on the context, {summary} [source {first_idx}]."
        else:
            answer_text = f"{summary} [source {first_idx}]."

    prompt_tokens = len(prompt.split())
    completion_tokens = len(answer_text.split())
    token_usage = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "max_tokens": max_tokens,
    }
    return answer_text, token_usage

openai_client = None
def _get_client() -> OpenAI:
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return openai_client

def generate_answer(prompt: str, *, model: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> Tuple[str, Dict[str, int | str]]:
    print(f"Prompt: {prompt}")
    client = _get_client()
    response = None
    try:
        response = client.responses.create(
        model=model or DEFAULT_GENERATOR_MODEL,
            input=prompt,
            temperature=OPENAI_TEMPERATURE,
            top_p=OPENAI_TOP_P,
            max_output_tokens=max_tokens or OPENAI_MAX_TOKENS
        )
    except Exception as exc:
        print(f"Error while calling OpenAI: {exc}")
        return "Unable to generate an answer. Please retry.", {
            "model": model or DEFAULT_GENERATOR_MODEL,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_tokens": max_tokens
        }
    
    print(f"Response: {response.output_text}")
    answer_text = response.output_text.strip() if getattr(response, "output_text", "") else ""
    if not answer_text:
        raise RuntimeError("LLM response did not contain any text.")
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", 0) or 0
    completion_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens
    token_usage: Dict[str, int | str] = {
        "model": response.model if hasattr(response, "model") else model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "max_tokens": max_tokens,
    }
    return answer_text, token_usage

if __name__ == "__main__":
    try:
        raise SystemExit(generate_answer(prompt="", model=""))
    finally:
        print("Done")

