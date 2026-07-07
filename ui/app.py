"""
DocChat — Streamlit Application
================================
Main entrypoint for the RAG-based Document Q&A system.

Run with:
    streamlit run ui/app.py

Features:
  • Multi-file upload (PDF, DOCX, TXT)
  • Configurable chunking strategy, size, and overlap via sidebar
  • ChromaDB vector store with real-time indexing progress
  • Grounded answer generation with OpenAI GPT-4o-mini
  • Citation cards linking answers to source documents + page numbers
  • Conversation memory for multi-turn Q&A
  • Dark, premium UI with animations
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# ── Bootstrap path so imports work whether run from root or ui/ ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from embeddings.embedder import embed_chunks, embed_query, get_model
from ingestion.chunker import chunk_documents
from ingestion.loader import load_document
from ingestion.triage import triage_documents
from rag.generator import generate_answer
from rag.prompt_templates import (
    EMPTY_STORE_RESPONSE,
    NO_CONTEXT_RESPONSE,
    build_rag_prompt,
)
from rag.retriever import retrieve
from ui.components import (
    inject_css,
    render_assistant_message,
    render_hero,
    render_sidebar_section,
    render_source_badge,
    render_status_bar,
    render_thinking,
    render_user_message,
)
from vectorstore.chroma_store import ChromaStore

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("docchat.app")


# ─── Page Config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="DocChat — RAG Document Q&A",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


# ─── Session State Initialization ─────────────────────────────────────────────

def _init_session() -> None:
    """Initialize all session state variables on first run."""
    defaults: dict = {
        "messages": [],             # list of {"role": str, "content": str, "citations": list, "meta": dict}
        "store": None,              # ChromaStore instance
        "indexed_sources": [],      # list of filenames successfully indexed
        "chunk_count": 0,           # total chunks in store
        "processing": False,        # True while indexing
        "api_key_valid": False,     # True if API key check passed
        "conversation_history": [], # for multi-turn prompt injection
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ─── Store Initializer ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_store(collection_name: str, persist_dir: str) -> ChromaStore:
    """Cached ChromaStore — persists across reruns."""
    return ChromaStore(collection_name=collection_name, persist_dir=persist_dir)


def ensure_store() -> ChromaStore:
    if st.session_state.store is None:
        st.session_state.store = get_store(
            collection_name=config.COLLECTION_NAME,
            persist_dir=str(ROOT / "chroma_db"),
        )
        # Sync indexed sources from existing store
        existing = st.session_state.store.list_sources()
        if existing:
            st.session_state.indexed_sources = existing
            st.session_state.chunk_count = st.session_state.store.count
    return st.session_state.store


# ─── Document Indexing ────────────────────────────────────────────────────────

def index_file(
    file_bytes: bytes,
    filename: str,
    strategy: str,
    chunk_size: int,
    overlap: int,
    store: ChromaStore,
    progress_bar,
    status_text,
) -> tuple[int, list[str]]:
    """
    Run the full ingestion pipeline for a single uploaded file.

    Returns:
        (chunks_added, warnings)
    """
    warnings: list[str] = []

    # Save upload to a temp file (PyMuPDF needs a real path for PDFs)
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        # 1. Load
        status_text.markdown(f"📖 Loading `{filename}`...")
        progress_bar.progress(10)
        docs = load_document(tmp_path)
        if not docs:
            warnings.append(f"No text extracted from {filename}")
            return 0, warnings

        # 2. Triage
        status_text.markdown(f"🔍 Triaging `{filename}`...")
        progress_bar.progress(25)
        accepted, rejected = triage_documents(docs)
        if rejected:
            warnings.append(
                f"{len(rejected)} page(s) from '{filename}' rejected "
                f"(duplicates / too short)"
            )
        if not accepted:
            warnings.append(f"All pages from '{filename}' rejected by triage")
            return 0, warnings

        # Patch source to original filename (temp file has random name)
        for doc in accepted:
            doc["source"] = filename
            doc["source_path"] = filename

        # 3. Chunk
        status_text.markdown(f"✂️ Chunking `{filename}` [{strategy}]...")
        progress_bar.progress(45)
        chunks = chunk_documents(
            accepted,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        if not chunks:
            warnings.append(f"No chunks produced from '{filename}'")
            return 0, warnings

        # 4. Embed
        status_text.markdown(f"🧬 Embedding {len(chunks)} chunks...")
        progress_bar.progress(65)
        _, embeddings = embed_chunks(
            chunks,
            model_name=config.EMBEDDING_MODEL,
            show_progress=False,
        )

        # 5. Store
        status_text.markdown(f"💾 Storing in vector database...")
        progress_bar.progress(88)
        store.upsert_chunks(chunks, embeddings.tolist())
        progress_bar.progress(100)

        return len(chunks), warnings

    finally:
        tmp_path.unlink(missing_ok=True)


# ─── API Key Validation ───────────────────────────────────────────────────────

def _check_api_key(provider: str, key: str) -> bool:
    """Validate API key format depending on the provider."""
    if provider == "ollama":
        return True
    if not key:
        return False
    if provider == "groq":
        return key.startswith("gsk_") and len(key) > 20
    if provider == "openai":
        return key.startswith("sk-") and len(key) > 20
    if provider == "gemini":
        return len(key) > 20
    return True


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.html(
        """
        <div style="text-align:center; padding:0.5rem 0 1rem;">
          <span style="font-size:2rem;">🗂️</span>
          <div style="font-size:1.1rem; font-weight:700; color:#e8edf8; margin-top:0.3rem;">
            DocChat
          </div>
          <div style="font-size:0.75rem; color:#8899b8;">RAG Document Q&amp;A</div>
        </div>
        """
    )

    # ── LLM Provider Selection ──
    render_sidebar_section("🤖 LLM Provider")
    from rag.generator import PROVIDER_CONFIGS, resolve_provider

    # Get configured default or use auto-detected
    detected_provider, _, _, _ = resolve_provider()
    provider_options = list(PROVIDER_CONFIGS.keys())
    
    selected_provider = st.selectbox(
        "Select Provider",
        options=provider_options,
        index=provider_options.index(detected_provider) if detected_provider in provider_options else 0,
        format_func=lambda x: PROVIDER_CONFIGS[x]["label"],
        label_visibility="collapsed",
    )
    
    # Save selection to session state and config/env
    os.environ["LLM_PROVIDER"] = selected_provider
    config.LLM_PROVIDER = selected_provider

    # ── Validate API Key silently (Hidden from UI) ──
    cfg = PROVIDER_CONFIGS[selected_provider]
    key_env = cfg["key_env"]

    if key_env:
        default_val = getattr(config, key_env, "") or os.getenv(key_env, "")
        if default_val:
            os.environ[key_env] = default_val
            setattr(config, key_env, default_val)
        st.session_state.api_key_valid = _check_api_key(selected_provider, default_val)
    else:
        # Ollama requires no key
        st.session_state.api_key_valid = True

    # ── Upload ──
    render_sidebar_section("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT",
    )

    # ── Default Settings (Hidden from UI) ──
    strategy = "fixed"
    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    top_k = config.TOP_K
    score_threshold = 0.10

    # ── Index Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    index_clicked = st.button(
        "🚀 Index Documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    # ── Clear ──
    col1, col2 = st.columns(2)
    with col1:
        clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)
    with col2:
        clear_docs = st.button("🧹 Clear Docs", use_container_width=True)

    # ── Indexed Sources ──
    store = ensure_store()
    sources = store.list_sources()
    if sources:
        render_sidebar_section(f"📚 Indexed ({len(sources)} files)")
        for src in sources:
            ext = Path(src).suffix.lstrip(".").lower()
            render_source_badge(src, ext)


# ─── Handle Actions ────────────────────────────────────────────────────────────

if clear_chat:
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.rerun()

if clear_docs:
    ensure_store().reset()
    st.session_state.store = None
    st.session_state.indexed_sources = []
    st.session_state.chunk_count = 0
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.success("Vector store cleared!")
    st.rerun()

if index_clicked and uploaded_files:
    store = ensure_store()
    total_new_chunks = 0
    all_warnings: list[str] = []

    index_container = st.empty()
    with index_container.container():
        st.markdown("### 📥 Indexing Documents")
        progress_bar = st.progress(0)
        status_text = st.empty()

        for uf in uploaded_files:
            if uf.name in store.list_sources():
                st.info(f"⏭️ '{uf.name}' already indexed — skipping")
                continue

            file_bytes = uf.read()
            try:
                n_chunks, warnings = index_file(
                    file_bytes=file_bytes,
                    filename=uf.name,
                    strategy=strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    store=store,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )
                total_new_chunks += n_chunks
                all_warnings.extend(warnings)
                if n_chunks > 0 and uf.name not in st.session_state.indexed_sources:
                    st.session_state.indexed_sources.append(uf.name)
            except Exception as exc:
                st.error(f"❌ Failed to index '{uf.name}': {exc}")
                logger.exception(f"Indexing failed for {uf.name}")

        st.session_state.chunk_count = store.count
        status_text.empty()
        progress_bar.empty()

        if total_new_chunks > 0:
            st.success(
                f"✅ Indexed {len(uploaded_files)} file(s) → "
                f"{total_new_chunks} new chunks stored"
            )
        for w in all_warnings:
            st.warning(f"⚠️ {w}")

    import time
    time.sleep(2)
    index_container.empty()
    st.rerun()


# ─── Main Chat Interface ───────────────────────────────────────────────────────

render_hero()

# Status bar (removed from UI)
store = ensure_store()


# Render conversation history
if st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_message(msg["content"])
        else:
            render_assistant_message(
                content=msg["content"],
                citations=msg.get("citations"),
                meta=msg.get("meta"),
            )
else:
    # Empty state
    st.html(
        """
        <div style="
          text-align: center;
          padding: 3rem 1rem;
          color: #4a5a7a;
        ">
          <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
      
         
          <div style="margin-top: 1.5rem; display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap;">
            <span style="padding: 0.4rem 0.9rem; background: #141c2e; border: 1px solid #1e2c4a; border-radius: 20px; font-size: 0.8rem; color: #8899b8;">
              💡 "What is the remote work policy?"
            </span>
            <span style="padding: 0.4rem 0.9rem; background: #141c2e; border: 1px solid #1e2c4a; border-radius: 20px; font-size: 0.8rem; color: #8899b8;">
              💡 "How do I report a security incident?"
            </span>
          </div>
        </div>
        """
    )


# ─── Chat Input ──────────────────────────────────────────────────────────────

query = st.chat_input(
    placeholder="Ask a question about your documents...",
    disabled=st.session_state.processing,
)

if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    render_user_message(query)

    # Guard: no documents
    if store.count == 0:
        resp_content = EMPTY_STORE_RESPONSE
        st.session_state.messages.append(
            {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
        )
        render_assistant_message(resp_content)

    # Guard: no API key
    elif not st.session_state.api_key_valid:
        selected_provider_label = PROVIDER_CONFIGS.get(selected_provider, {}).get("label", "Selected provider")
        resp_content = (
            f"⚠️ **{selected_provider_label} API key not configured.** "
            f"Please enter your API key in the sidebar to generate answers. "
            f"You can still index documents without a key."
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
        )
        render_assistant_message(resp_content)

    else:
        # Show thinking indicator
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            render_thinking()

        st.session_state.processing = True
        try:
            # 1. Retrieve
            context_chunks = retrieve(
                query=query,
                store=store,
                embed_fn=embed_query,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            if not context_chunks:
                resp_content = NO_CONTEXT_RESPONSE
                citations = []
                meta = {}
            else:
                # 2. Build prompt (with conversation history for multi-turn)
                # Keep last 6 messages (3 turns) for context window efficiency
                history_for_prompt = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.conversation_history[-6:]
                ]
                messages = build_rag_prompt(
                    query=query,
                    context_chunks=context_chunks,
                    conversation_history=history_for_prompt,
                )

                # 3. Generate
                gen_resp = generate_answer(messages, context_chunks)
                resp_content = gen_resp["answer"]
                citations = gen_resp["citations"]
                meta = {
                    "model": gen_resp["model"],
                    "input_tokens": gen_resp["input_tokens"],
                    "output_tokens": gen_resp["output_tokens"],
                    "n_chunks": len(context_chunks),
                }

            # Update conversation history for next turn
            st.session_state.conversation_history.append(
                {"role": "user", "content": query}
            )
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": resp_content}
            )

            # Append to messages
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": resp_content,
                    "citations": citations if context_chunks else [],
                    "meta": meta if context_chunks else {},
                }
            )

        except ValueError as exc:
            # API key / config error
            resp_content = f"⚠️ Configuration error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
        except RuntimeError as exc:
            # OpenAI API error
            resp_content = f"❌ API error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
        except Exception as exc:
            logger.exception("Unexpected error during generation")
            resp_content = f"❌ Unexpected error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
        finally:
            st.session_state.processing = False
            thinking_placeholder.empty()

        # Re-render the final answer
        last_msg = st.session_state.messages[-1]
        render_assistant_message(
            content=last_msg["content"],
            citations=last_msg.get("citations"),
            meta=last_msg.get("meta"),
        )



