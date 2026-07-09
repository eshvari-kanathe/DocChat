from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

Document = dict[str, Any]


# ─── Individual loaders ───────────────────────────────────────────────────────

def load_pdf(file_path: Path) -> list[Document]:
    try:
        import fitz  
    except ImportError:
        raise ImportError("PyMuPDF is required: pip install PyMuPDF")

    docs: list[Document] = []
    try:
        pdf = fitz.open(str(file_path))
        total = len(pdf)
        for idx in range(total):
            page = pdf[idx]
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    {
                        "text": text,
                        "source": file_path.name,
                        "source_path": str(file_path.resolve()),
                        "page_number": idx + 1,
                        "total_pages": total,
                        "file_type": "pdf",
                    }
                )
        pdf.close()
    except Exception as exc:
        logger.error(f"PDF load failed [{file_path}]: {exc}")
        raise

    logger.info(f"  PDF '{file_path.name}': {len(docs)} pages extracted")
    return docs


def load_docx(file_path: Path) -> list[Document]:
    """
    Extract paragraphs from a DOCX file.
    DOCX has no native page boundary info, so the whole doc = page 1.
    """
    try:
        from docx import Document as DocxDoc
    except ImportError:
        raise ImportError("python-docx is required: pip install python-docx")

    docs: list[Document] = []
    try:
        doc = DocxDoc(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        full_text = "\n\n".join(paragraphs)
        if full_text:
            docs.append(
                {
                    "text": full_text,
                    "source": file_path.name,
                    "source_path": str(file_path.resolve()),
                    "page_number": 1,
                    "total_pages": 1,
                    "file_type": "docx",
                }
            )
    except Exception as exc:
        logger.error(f"DOCX load failed [{file_path}]: {exc}")
        raise

    logger.info(f"  DOCX '{file_path.name}': {len(docs)} segment(s) extracted")
    return docs


def load_txt(file_path: Path) -> list[Document]:
    """Load a plain-text file as a single document."""
    docs: list[Document] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs.append(
                {
                    "text": text,
                    "source": file_path.name,
                    "source_path": str(file_path.resolve()),
                    "page_number": 1,
                    "total_pages": 1,
                    "file_type": "txt",
                }
            )
    except Exception as exc:
        logger.error(f"TXT load failed [{file_path}]: {exc}")
        raise

    logger.info(f"  TXT '{file_path.name}': loaded")
    return docs


# ─── Dispatch table ──────────────────────────────────────────────────────────

_LOADERS: dict[str, Any] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".doc": load_docx,
    ".txt": load_txt,
}

SUPPORTED_EXTENSIONS: list[str] = list(_LOADERS.keys())


# ─── Public API ───────────────────────────────────────────────────────────────

def load_document(file_path: str | Path) -> list[Document]:
    """
    Load a single document file and return page-level Document dicts.

    Raises:
        FileNotFoundError: if file does not exist.
        ValueError: if file extension is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info(f"Loading [{ext.upper()}]: {path.name}")
    return loader(path)


def load_documents(file_paths: list[str | Path]) -> list[Document]:
    """
    Load multiple document files, skipping any that fail.

    Returns combined list of all successfully extracted Documents.
    """
    all_docs: list[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(load_document(fp))
        except Exception as exc:
            logger.warning(f"Skipping '{fp}': {exc}")

    logger.info(
        f"Loaded {len(all_docs)} document segments from {len(file_paths)} file(s)"
    )
    return all_docs
