"""
Retriever: embeds a natural language query and fetches the top-k
most relevant document chunks from the vector store.

Supports:
  • Standard top-k cosine similarity retrieval
  • Optional minimum score threshold to filter low-quality matches
  • Optional ChromaDB metadata filtering (e.g., filter by source file)
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]


def retrieve(
    query: str,
    store,
    embed_fn: Callable[[str], Any],
    top_k: int = 5,
    score_threshold: float = 0.1,
    metadata_filter: dict | None = None,
) -> list[SearchResult]:
    """
    Embed the query and retrieve the top-k most relevant chunks.

    Args:
        query:            Natural language question from the user.
        store:            ChromaStore instance.
        embed_fn:         Callable that takes a string and returns a numpy array.
        top_k:            Maximum number of chunks to retrieve.
        score_threshold:  Minimum cosine similarity score (0–1). Chunks below
                          this threshold are filtered out.
        metadata_filter:  Optional ChromaDB 'where' clause dict, e.g.:
                          {"source": "policy.pdf"}

    Returns:
        List of SearchResult dicts, sorted by descending similarity score.
        Returns [] if the store is empty or no results pass the threshold.
    """
    if store.count == 0:
        logger.warning("Retriever: vector store is empty — no documents indexed yet.")
        return []

    # Embed the query
    query_embedding = embed_fn(query).tolist()

    # Search
    results = store.similarity_search(
        query_embedding=query_embedding,
        top_k=top_k,
        where=metadata_filter,
    )

    # Apply score threshold
    filtered = [r for r in results if r["score"] >= score_threshold]

    if not filtered and results:
        logger.info(
            f"Retriever: all {len(results)} result(s) below threshold "
            f"({score_threshold}). Top score was {results[0]['score']:.3f}"
        )

    logger.info(
        f"Retriever: '{query[:70]}' → {len(filtered)} chunk(s) "
        f"(top score: {filtered[0]['score']:.3f})" if filtered else
        f"Retriever: '{query[:70]}' → 0 chunks retrieved"
    )

    return filtered


def format_context_for_display(results: list[SearchResult]) -> str:
    """
    Format retrieved chunks as a human-readable string for debugging.
    """
    if not results:
        return "(no context retrieved)"

    lines = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        lines.append(
            f"[{i}] {meta.get('source', 'unknown')} | "
            f"p.{meta.get('page_number', '?')} | "
            f"score={r['score']:.3f}\n"
            f"{r['text'][:200]}..."
        )
    return "\n\n".join(lines)
