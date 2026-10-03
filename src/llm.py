"""
Step 18 — LLM generation + output checkpoint (multi-provider).

Tries providers in order: Groq (fast, default) -> Gemini -> retrieved-chunk.
If a provider has no key or errors, it falls through to the next, so the
product never hard-fails. Order configurable via LLM_PROVIDER (groq|gemini|auto).

Grounded + cited (answers only from numbered context, tags each sentence with
its source), abstains when nothing relevant is retrieved, and caches identical
(query, context) pairs for instant repeats. Model names are env vars because
providers deprecate model IDs regularly.
"""

import os
import re
import hashlib

from src.features import structural_features

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto").lower()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
MAX_OUTPUT_TOKENS = 600

RELEVANCE_FLOOR = 0.05
NO_ANSWER = "I don't have that information in this document."

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the question using ONLY the "
    "numbered context passages provided. Rules:\n"
    "1. After each sentence in your answer, cite the passage(s) it came from, "
    "like [1] or [2][3].\n"
    "2. If the context does not contain the answer, reply exactly: "
    f"\"{NO_ANSWER}\" — do not guess or use outside knowledge.\n"
    "3. Be concise and direct. Do not repeat the question."
)

_CACHE = {}
_CACHE_MAX = 256


def _cache_key(query, context_chunks):
    h = hashlib.sha256()
    h.update(query.strip().lower().encode())
    for c in context_chunks:
        h.update(b"|"); h.update(c.get("text", "").encode())
    return h.hexdigest()


def _looks_relevant(context_chunks: list) -> bool:
    if not context_chunks:
        return False
    top = context_chunks[0]
    score = top.get("score")
    if score is None:
        score = top.get("rerank_score", 0.0)
    return score >= RELEVANCE_FLOOR


def _build_context(context_chunks):
    lines, mapping = [], {}
    for i, c in enumerate(context_chunks, start=1):
        lines.append(f"[{i}] {c['text']}"); mapping[i] = c
    return "\n\n".join(lines), mapping


def _resolve_citations(text, mapping):
    out = []
    for m in sorted({int(x) for x in re.findall(r"\[(\d+)\]", text)}):
        c = mapping.get(m)
        if c:
            out.append({"marker": m, "chunk_id": c.get("doc_id"), "text": c["text"]})
    return out


def _chunk_fallback(context_chunks, note):
    if not _looks_relevant(context_chunks):
        return {"text": NO_ANSWER, "source": "fallback", "note": note, "citations": []}
    top = context_chunks[0]
    return {"text": top["text"], "source": "fallback", "note": note,
            "citations": [{"marker": 1, "chunk_id": top.get("doc_id"), "text": top["text"]}]}


def _groq_key():
    return os.environ.get("GROQ_API_KEY")


def _gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _call_groq(query, context_text, mapping):
    from openai import OpenAI
    client = OpenAI(api_key=_groq_key(), base_url=GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}],
    )
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty response from Groq")
    return {"text": text, "source": "groq", "note": None, "citations": _resolve_citations(text, mapping)}


def _call_gemini(query, context_text, mapping, pdf_bytes=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=_gemini_key())
    if pdf_bytes:
        contents = [types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n(Source document, cite as [1].)\n\nQuestion: {query}")]
    else:
        contents = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nQuestion: {query}"
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS))
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("empty response from Gemini")
    return {"text": text, "source": "gemini", "note": None, "citations": _resolve_citations(text, mapping)}


def _provider_order():
    if LLM_PROVIDER == "groq":
        return ["groq"]
    if LLM_PROVIDER == "gemini":
        return ["gemini"]
    return ["groq", "gemini"]


def generate_answer(query: str, context_chunks: list, pdf_bytes: bytes = None) -> dict:
    key = _cache_key(query, context_chunks)
    if key in _CACHE:
        cached = dict(_CACHE[key])
        cached["note"] = (cached.get("note") or "") + " [cached]"
        return cached

    context_text, mapping = _build_context(context_chunks)
    errors = []
    for provider in _provider_order():
        try:
            if provider == "groq":
                if not _groq_key():
                    errors.append("groq: no GROQ_API_KEY"); continue
                result = _call_groq(query, context_text, mapping)
            elif provider == "gemini":
                if not _gemini_key():
                    errors.append("gemini: no GEMINI_API_KEY"); continue
                result = _call_gemini(query, context_text, mapping, pdf_bytes=pdf_bytes)
            else:
                continue
            if len(_CACHE) < _CACHE_MAX:
                _CACHE[key] = dict(result)
            return result
        except Exception as e:
            errors.append(f"{provider}: {type(e).__name__}: {e}"); continue

    note = "No LLM available (" + "; ".join(errors) + ")" if errors else "No LLM configured."
    return _chunk_fallback(context_chunks, note)


def check_output(text: str) -> dict:
    feats = structural_features(text)
    triggers = []
    if feats["pii_hits"] > 0:
        triggers.append(f"{int(feats['pii_hits'])} PII-like pattern(s) in generated output")
    if feats["injection_kw"] > 0:
        triggers.append(f"{int(feats['injection_kw'])} injection-pattern keyword(s) in generated output")
    return {"flagged": bool(triggers), "triggers": triggers or ["no issues detected"]}


if __name__ == "__main__":
    ctx = [{"text": "The insured's name is Uttrakhand Jal Vidyut Nigam Ltd.", "doc_id": "d1", "score": 0.9}]
    print("provider order:", _provider_order())
    print(generate_answer("What is the insured's name?", ctx))