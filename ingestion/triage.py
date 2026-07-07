"""
Document triage: filters noisy, duplicate, or irrelevant documents
before chunking and indexing.

Triage rules (applied in order):
  1. Blocked filename patterns  (e.g. temp Office files, .DS_Store)
  2. Minimum token count        (very short pages are usually headers/blank)
  3. Content-hash deduplication (SHA-256 of stripped text)
"""
from __future__ import annotations

import fnmatch
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

Document = dict[str, Any]
RejectionLog = list[dict[str, str]]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _token_estimate(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters (OpenAI-style approximation)."""
    return len(text) // 4


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _is_blocked(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)


# ─── Public API ──────────────────────────────────────────────────────────────

def triage_documents(
    docs: list[Document],
    min_tokens: int | None = None,
    blocked_patterns: list[str] | None = None,
) -> tuple[list[Document], RejectionLog]:
    """
    Filter documents by quality and uniqueness.

    Args:
        docs:             List of Document dicts from the loader.
        min_tokens:       Minimum token count to accept a document page.
        blocked_patterns: Glob patterns matched against the source filename.

    Returns:
        (accepted_docs, rejection_log)
        rejection_log contains dicts with keys: source, page_number, reason.
    """
    min_tokens = min_tokens if min_tokens is not None else config.MIN_DOC_TOKENS
    blocked_patterns = blocked_patterns or config.BLOCKED_PATTERNS

    seen_hashes: set[str] = set()
    accepted: list[Document] = []
    rejected: RejectionLog = []

    for doc in docs:
        source = doc.get("source", "unknown")
        page = doc.get("page_number", "?")
        text = doc.get("text", "")

        # Rule 1 — Blocked filename patterns
        if _is_blocked(source, blocked_patterns):
            reason = "blocked_filename_pattern"
            rejected.append({"source": source, "page_number": str(page), "reason": reason})
            logger.debug(f"REJECT [{reason}]: {source}")
            continue

        # Rule 2 — Minimum token count
        tokens = _token_estimate(text)
        if tokens < min_tokens:
            reason = f"too_short ({tokens} tokens < {min_tokens} required)"
            rejected.append({"source": source, "page_number": str(page), "reason": reason})
            logger.debug(f"REJECT [{reason}]: {source} p.{page}")
            continue

        # Rule 3 — Duplicate content (hash of stripped text)
        content_hash = _sha256(text.strip())
        if content_hash in seen_hashes:
            reason = "duplicate_content"
            rejected.append({"source": source, "page_number": str(page), "reason": reason})
            logger.debug(f"REJECT [{reason}]: {source} p.{page}")
            continue

        seen_hashes.add(content_hash)
        accepted.append(doc)

    logger.info(
        f"Triage: {len(accepted)} accepted, {len(rejected)} rejected "
        f"(out of {len(docs)} document segments)"
    )
    return accepted, rejected


def print_rejection_report(rejection_log: RejectionLog) -> None:
    """Pretty-print the triage rejection report."""
    if not rejection_log:
        print("No documents were rejected.")
        return
    print(f"\n{'─'*60}")
    print(f"  Rejection Report ({len(rejection_log)} items)")
    print(f"{'─'*60}")
    for entry in rejection_log:
        print(
            f"  • {entry['source']} (p.{entry['page_number']}) → {entry['reason']}"
        )
    print(f"{'─'*60}\n")
