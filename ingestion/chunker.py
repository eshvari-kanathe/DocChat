"""
Chunking strategies for document text.

Two strategies are implemented and can be compared side-by-side:
  A) fixed  — fixed character window with token-overlap (fast, predictable size)
  B) semantic — paragraph/sentence boundary splitting (context-preserving)

Each chunk is a Document-like dict enriched with:
    chunk_id       - short UUID for vector store keying
    chunk_index    - sequential index within the parent document
    chunk_strategy - "fixed" | "semantic"
    + all original doc metadata (source, page_number, etc.)
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Literal

logger = logging.getLogger(__name__)

Document = dict[str, Any]
Chunk = dict[str, Any]
ChunkStrategy = Literal["fixed", "semantic"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _new_chunk_id() -> str:
    return str(uuid.uuid4()).replace("-", "")[:12]


def _clone_meta(doc: Document) -> dict[str, Any]:
    """Copy all metadata fields except 'text'."""
    return {k: v for k, v in doc.items() if k != "text"}


# ─── Strategy A: Fixed-size with overlap ─────────────────────────────────────

def fixed_size_chunker(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split document text into fixed-size chunks (measured in approximate tokens)
    with a sliding overlap window to preserve context across chunk boundaries.

    1 token ≈ 4 characters (conservative estimate).

    Args:
        doc:        Document dict with 'text' and metadata.
        chunk_size: Target chunk size in tokens.
        overlap:    Number of overlapping tokens between consecutive chunks.
    """
    text = doc["text"]
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    step = char_size - char_overlap

    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + char_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(
                {
                    **_clone_meta(doc),
                    "text": chunk_text,
                    "chunk_id": _new_chunk_id(),
                    "chunk_index": chunk_index,
                    "chunk_strategy": "fixed",
                    "chunk_size_cfg": chunk_size,
                    "overlap_cfg": overlap,
                }
            )
            chunk_index += 1

        start += step

    return chunks


# ─── Strategy B: Semantic / paragraph-based ───────────────────────────────────

def semantic_chunker(
    doc: Document,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 600,
) -> list[Chunk]:
    """
    Split document text on natural paragraph and sentence boundaries.

    Algorithm:
      1. Split on double-newlines (paragraph separators).
      2. Accumulate paragraphs into a buffer until max_chunk_tokens is reached.
      3. If a single paragraph exceeds max_chunk_tokens, split it further on
         sentence boundaries ('.', '!', '?').

    Args:
        doc:              Document dict.
        min_chunk_tokens: Minimum tokens before a chunk can be flushed.
        max_chunk_tokens: Maximum tokens in any single chunk.
    """
    text = doc["text"]

    # Step 1 — paragraph split
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Step 2 — further split very long paragraphs on sentence boundaries
    def sentence_split(s: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", s)
        return [p.strip() for p in parts if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        if len(para) // 4 > max_chunk_tokens:
            units.extend(sentence_split(para))
        else:
            units.append(para)

    # Step 3 — group units into chunks
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens, chunk_index
        if not buffer:
            return
        chunk_text = "\n\n".join(buffer).strip()
        if chunk_text and (buffer_tokens >= min_chunk_tokens):
            chunks.append(
                {
                    **_clone_meta(doc),
                    "text": chunk_text,
                    "chunk_id": _new_chunk_id(),
                    "chunk_index": chunk_index,
                    "chunk_strategy": "semantic",
                    "min_chunk_tokens_cfg": min_chunk_tokens,
                    "max_chunk_tokens_cfg": max_chunk_tokens,
                }
            )
            chunk_index += 1
        buffer.clear()
        buffer_tokens = 0

    for unit in units:
        unit_tokens = len(unit) // 4
        if buffer and buffer_tokens + unit_tokens > max_chunk_tokens:
            flush()
        buffer.append(unit)
        buffer_tokens += unit_tokens

    flush()  # flush remaining content

    # Edge case: if chunker produced nothing (very short doc), keep as one chunk
    if not chunks and text.strip():
        chunks.append(
            {
                **_clone_meta(doc),
                "text": text.strip(),
                "chunk_id": _new_chunk_id(),
                "chunk_index": 0,
                "chunk_strategy": "semantic",
                "min_chunk_tokens_cfg": min_chunk_tokens,
                "max_chunk_tokens_cfg": max_chunk_tokens,
            }
        )

    return chunks


# ─── Public API ──────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    strategy: ChunkStrategy = "fixed",
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 600,
) -> list[Chunk]:
    """
    Chunk a list of Document dicts using the specified strategy.

    Args:
        docs:             Documents from the loader/triage pipeline.
        strategy:         "fixed" or "semantic".
        chunk_size:       (fixed) target tokens per chunk.
        overlap:          (fixed) token overlap between chunks.
        min_chunk_tokens: (semantic) minimum tokens per chunk.
        max_chunk_tokens: (semantic) maximum tokens per chunk.

    Returns:
        Flat list of all Chunk dicts.
    """
    all_chunks: list[Chunk] = []

    for doc in docs:
        if strategy == "fixed":
            chunks = fixed_size_chunker(doc, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "semantic":
            chunks = semantic_chunker(
                doc,
                min_chunk_tokens=min_chunk_tokens,
                max_chunk_tokens=max_chunk_tokens,
            )
        else:
            raise ValueError(f"Unknown chunking strategy: '{strategy}'. Use 'fixed' or 'semantic'.")

        all_chunks.extend(chunks)

    logger.info(
        f"Chunking [{strategy}]: {len(docs)} document segments → {len(all_chunks)} chunks"
    )
    return all_chunks


def compare_strategies(
    docs: list[Document],
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_tokens: int = 50,
    max_chunk_tokens: int = 600,
) -> dict[str, list[Chunk]]:
    """
    Run both chunking strategies and return results for comparison.
    Useful for the evaluation module.
    """
    return {
        "fixed": chunk_documents(
            docs, strategy="fixed", chunk_size=chunk_size, overlap=overlap
        ),
        "semantic": chunk_documents(
            docs,
            strategy="semantic",
            min_chunk_tokens=min_chunk_tokens,
            max_chunk_tokens=max_chunk_tokens,
        ),
    }
