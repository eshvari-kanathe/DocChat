"""
LLM Generator: supports multiple providers via a unified OpenAI-compatible interface.

Supported providers:
  • groq     — Free tier, OpenAI-compatible, fast (llama-3.3-70b-versatile)
  • openai   — GPT-4o-mini / GPT-4o
  • gemini   — Google Gemini via openai-compatibility shim (gemini-1.5-flash)
  • ollama   — Fully local (no internet needed), requires Ollama running locally

Provider selection priority:
  1. Explicit `provider` argument to generate_answer()
  2. LLM_PROVIDER env variable / config
  3. Whichever key is present: GROQ_API_KEY → OPENAI_API_KEY → GEMINI_API_KEY

Response structure:
    {
        "answer":        str,
        "citations":     list[{"source": str, "page_number": int|None}],
        "model":         str,
        "input_tokens":  int,
        "output_tokens": int,
        "raw":           str,
    }
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

Citation = dict[str, Any]
GeneratorResponse = dict[str, Any]

# ─── Provider Configs ────────────────────────────────────────────────────────

PROVIDER_CONFIGS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "label": "Groq (Free)",
    },
    "openai": {
        "base_url": None,  # default OpenAI base
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "label": "OpenAI",
    },
    "gemini": {
        # Google Gemini exposes an OpenAI-compatible endpoint
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "key_env": "GEMINI_API_KEY",
        "label": "Google Gemini (Free)",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3",
        "key_env": None,        # no key required
        "label": "Ollama (Local)",
    },
}


# ─── Citation Parser ─────────────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^\],]+?)(?:,|,\s*)(?:Page|p\.?)\s*(\d+|\?)\]",
    re.IGNORECASE,
)

# Matches format: * filename (Page X) or * filename (p. X)
_LIST_CITATION_PATTERN = re.compile(
    r"\*\s*([^\n\(\*]+?)\s*\(\s*(?:Page|p\.?)\s*(\d+|\?)\s*\)",
    re.IGNORECASE,
)


def _parse_citations(
    answer_text: str,
    context_chunks: list[dict],
) -> list[Citation]:
    """
    Extract citations from the answer text.
    Supports:
      1. New list format: * filename (Page X)
      2. Old inline format: [Source: filename, Page X]
      3. Fallback: first retrieved context chunk.
    """
    citations: list[Citation] = []
    seen: set[tuple] = set()

    # Try list format first
    for source_name, page_str in _LIST_CITATION_PATTERN.findall(answer_text):
        source_name = source_name.strip()
        page_num = int(page_str) if page_str.isdigit() else None
        key = (source_name.lower(), page_num)
        if key not in seen:
            seen.add(key)
            citations.append({"source": source_name, "page_number": page_num})

    # Try old format next
    for source_name, page_str in _CITATION_PATTERN.findall(answer_text):
        source_name = source_name.strip()
        page_num = int(page_str) if page_str.isdigit() else None
        key = (source_name.lower(), page_num)
        if key not in seen:
            seen.add(key)
            citations.append({"source": source_name, "page_number": page_num})

    # Fallback: if no citations found but context chunks exist, use the top chunk
    if not citations and context_chunks and "I couldn't find information" not in answer_text:
        top_meta = context_chunks[0].get("metadata", {})
        fallback_source = top_meta.get("source", "Unknown")
        fallback_page = top_meta.get("page_number")
        key = (fallback_source.lower(), fallback_page)
        if key not in seen:
            citations.append({"source": fallback_source, "page_number": fallback_page})

    return citations


# ─── Provider Resolution ─────────────────────────────────────────────────────

def resolve_provider(provider: str | None = None) -> tuple[str, str, str | None, str]:
    """
    Determine which provider to use and return (provider, model, api_key, base_url).

    Auto-detection order (when provider=None):
      GROQ_API_KEY set  → groq
      OPENAI_API_KEY    → openai
      GEMINI_API_KEY    → gemini
      else              → ollama (local, no key)
    """
    # Use explicit or config value
    chosen = provider or getattr(config, "LLM_PROVIDER", None) or os.getenv("LLM_PROVIDER", "")

    if not chosen:
        # Auto-detect from available keys
        if os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", ""):
            chosen = "groq"
        elif os.getenv("OPENAI_API_KEY") or getattr(config, "OPENAI_API_KEY", ""):
            chosen = "openai"
        elif os.getenv("GEMINI_API_KEY") or getattr(config, "GEMINI_API_KEY", ""):
            chosen = "gemini"
        else:
            chosen = "ollama"

    if chosen not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider '{chosen}'. "
            f"Valid options: {list(PROVIDER_CONFIGS.keys())}"
        )

    cfg = PROVIDER_CONFIGS[chosen]
    key_env = cfg["key_env"]

    # Resolve API key
    api_key: str | None = None
    if key_env:
        api_key = (
            os.getenv(key_env, "")
            or getattr(config, key_env, "")
            or ""
        ) or None

    base_url = cfg["base_url"]
    default_model = cfg["default_model"]

    return chosen, default_model, api_key, base_url


# ─── Generator ───────────────────────────────────────────────────────────────

def generate_answer(
    messages: list[dict],
    context_chunks: list[dict],
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> GeneratorResponse:
    """
    Generate a grounded answer using the configured LLM provider.

    Args:
        messages:       Full message list from build_rag_prompt().
        context_chunks: Retrieved chunks (used for citation fallback).
        provider:       "groq" | "openai" | "gemini" | "ollama" | None (auto-detect).
        model:          Override the provider's default model.
        temperature:    Override config.LLM_TEMPERATURE.
        max_tokens:     Override config.MAX_TOKENS.
        api_key:        Override the provider's API key.

    Returns:
        GeneratorResponse dict.
    """
    try:
        from openai import OpenAI, APIError, RateLimitError, AuthenticationError
    except ImportError:
        raise ImportError("openai SDK is required: pip install openai")

    # Resolve provider settings
    chosen_provider, default_model, resolved_key, base_url = resolve_provider(provider)

    _api_key = api_key or resolved_key
    _model = model or default_model
    _temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    _max_tokens = max_tokens or config.MAX_TOKENS

    # Ollama doesn't need an API key
    if chosen_provider == "ollama":
        _api_key = _api_key or "ollama"  # SDK requires non-empty string

    if not _api_key and chosen_provider != "ollama":
        key_env = PROVIDER_CONFIGS[chosen_provider]["key_env"]
        raise ValueError(
            f"No API key found for provider '{chosen_provider}'. "
            f"Set {key_env} in your .env file or enter it in the sidebar."
        )

    # Build OpenAI-compatible client
    client_kwargs: dict[str, Any] = {"api_key": _api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    logger.info(f"Calling [{chosen_provider}] model={_model}")

    try:
        response = client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=_temperature,
            max_tokens=_max_tokens,
        )
    except AuthenticationError as e:
        raise ValueError(
            f"Authentication failed for '{chosen_provider}' — check your API key: {e}"
        ) from e
    except RateLimitError as e:
        raise RuntimeError(
            f"Rate limit hit on '{chosen_provider}'. "
            "Switch provider or wait a moment. Error: {e}"
        ) from e
    except APIError as e:
        raise RuntimeError(f"API error from '{chosen_provider}': {e}") from e

    raw_answer = response.choices[0].message.content or ""
    usage = response.usage
    citations = _parse_citations(raw_answer, context_chunks)

    # Clean the answer text so it doesn't double-print the sources list in the UI
    clean_answer = raw_answer
    if "### Sources" in clean_answer:
        clean_answer = clean_answer.split("### Sources")[0]

    # Standardize fallback response and clear citations
    if "currently available documents" in clean_answer.lower() or "upload the relevant" in clean_answer.lower() or "couldn't find information" in clean_answer.lower():
        clean_answer = (
            "I couldn't find information related to your question in the currently available documents.\n\n"
            "Please upload the relevant PDF, TXT, or DOCX file(s), and I'll be happy to search through them and provide an answer based on the uploaded content."
        )
        citations = []

    logger.info(
        f"[{chosen_provider}/{response.model}] "
        f"{usage.prompt_tokens if usage else '?'} in / "
        f"{usage.completion_tokens if usage else '?'} out | "
        f"{len(citations)} citation(s)"
    )

    return {
        "answer": clean_answer.strip(),
        "citations": citations,
        "model": f"{chosen_provider}/{response.model}",
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "raw": raw_answer,
        "provider": chosen_provider,
    }
