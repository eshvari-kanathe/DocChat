"""
Embedding module: wraps SentenceTransformers for local, free, high-quality embeddings.

Default model: all-MiniLM-L6-v2
  • 384-dimensional dense vectors
  • Fast (~14k sentences/sec on CPU)
  • Excellent semantic similarity performance

Embeddings are L2-normalized so cosine similarity = dot product, which is
what ChromaDB uses internally with its "cosine" HNSW space.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Module-level model cache (one per model name)
_MODEL_CACHE: dict[str, Any] = {}


def get_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load (or return cached) SentenceTransformer model.

    The first call downloads the model weights (~90 MB for MiniLM).
    Subsequent calls return the cached instance instantly.
    """
    global _MODEL_CACHE

    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model '{model_name}'...")
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        logger.info(f"Embedding model '{model_name}' loaded successfully.")

    return _MODEL_CACHE[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of text strings in batches.

    Args:
        texts:         List of strings to embed.
        model_name:    HuggingFace model identifier.
        batch_size:    Sentences per encoding batch.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized.
    """
    if not texts:
        return np.array([])

    model = get_model(model_name)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize → cosine sim = dot product
        convert_to_numpy=True,
    )

    logger.info(f"Embedded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


def embed_query(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Embed a single query string (no progress bar).

    Returns:
        1-D numpy array of shape (embedding_dim,).
    """
    return embed_texts([query], model_name=model_name, show_progress=False)[0]


def embed_chunks(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """
    Embed a list of Chunk dicts (extracting 'text' field for encoding).

    Returns:
        (chunks, embeddings_array)
        The same chunks list is returned for convenient pairing.
    """
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        show_progress=show_progress,
    )
    return chunks, embeddings
