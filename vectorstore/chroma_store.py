"""
ChromaDB vector store: persistent, file-backed storage with cosine similarity search.

Architecture:
  • One ChromaDB PersistentClient per project (stored in ./chroma_db/)
  • One Collection per document set (default: "docchat")
  • Chunk IDs are used as document IDs for idempotent upserts
  • All metadata is stored flat (ChromaDB requirement: str/int/float/bool values only)

Similarity Score Interpretation:
  ChromaDB with "cosine" HNSW space returns distances in range [0, 2]:
    distance = 0   → identical vectors (similarity = 1.0)
    distance = 1   → orthogonal vectors (similarity = 0.0)
    distance = 2   → opposite vectors  (similarity = -1.0)
  Conversion: similarity = 1.0 - distance  (clamped to [0, 1])
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SearchResult = dict[str, Any]
Chunk = dict[str, Any]


class ChromaStore:
    """
    Manages a single ChromaDB collection for document chunk storage and retrieval.

    Usage:
        store = ChromaStore(collection_name="docchat", persist_dir="./chroma_db")
        store.upsert_chunks(chunks, embeddings.tolist())
        results = store.similarity_search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "docchat",
        persist_dir: str = "./chroma_db",
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("chromadb is required: pip install chromadb")

        self.collection_name = collection_name
        self.persist_dir = str(Path(persist_dir).resolve())

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"ChromaStore ready: collection='{collection_name}', "
            f"path='{self.persist_dir}', "
            f"docs={self._collection.count()}"
        )

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Total number of chunks stored in the collection."""
        return self._collection.count()

    # ─── Write Operations ────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert chunks with precomputed embeddings into the collection.

        Upsert semantics: existing chunk_ids are updated; new ones are inserted.
        This makes repeated ingestion of the same file idempotent.

        Args:
            chunks:     List of Chunk dicts (must have 'chunk_id' and 'text').
            embeddings: Parallel list of embedding vectors (float lists).
        """
        if not chunks:
            logger.warning("upsert_chunks called with empty chunk list — skipping.")
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [_flatten_metadata(c) for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            f"Upserted {len(chunks)} chunk(s) into '{self.collection_name}' "
            f"(total: {self._collection.count()})"
        )

    def delete_by_source(self, source: str) -> None:
        """Remove all chunks originating from a specific filename."""
        self._collection.delete(where={"source": source})
        logger.info(f"Deleted all chunks for source: '{source}'")

    def reset(self) -> None:
        """Delete and recreate the collection (clears all data)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Collection '{self.collection_name}' has been reset.")

    # ─── Read Operations ─────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Find the top-k most similar chunks using cosine similarity.

        Args:
            query_embedding: Embedding vector for the query.
            top_k:           Number of results to return.
            where:           Optional ChromaDB metadata filter dict.

        Returns:
            List of SearchResult dicts sorted by descending similarity score.
            Each result: {"id", "text", "metadata", "score"}
        """
        n = self._collection.count()
        if n == 0:
            logger.warning("similarity_search: collection is empty")
            return []

        actual_k = min(top_k, n)
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        search_results: list[SearchResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # cosine space: distance ∈ [0, 2], similarity = 1 - distance (clamped)
            similarity = max(0.0, min(1.0, 1.0 - distance))
            search_results.append(
                {
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(similarity, 4),
                }
            )

        # Already sorted by distance (ascending), so similarity is descending
        return search_results

    def list_sources(self) -> list[str]:
        """Return sorted list of unique source filenames in the collection."""
        if self._collection.count() == 0:
            return []
        results = self._collection.get(include=["metadatas"])
        sources = sorted({m.get("source", "") for m in results["metadatas"]})
        return [s for s in sources if s]

    def get_all_metadata(self) -> list[dict]:
        """Return metadata for all chunks (without embeddings or text)."""
        if self._collection.count() == 0:
            return []
        return self._collection.get(include=["metadatas"])["metadatas"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _flatten_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """
    Extract and flatten chunk metadata into ChromaDB-compatible types.
    ChromaDB requires all metadata values to be str, int, float, or bool.
    """
    return {
        "source": str(chunk.get("source", "")),
        "source_path": str(chunk.get("source_path", "")),
        "page_number": int(chunk.get("page_number", 0)),
        "total_pages": int(chunk.get("total_pages", 0)),
        "file_type": str(chunk.get("file_type", "")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "chunk_strategy": str(chunk.get("chunk_strategy", "fixed")),
    }
