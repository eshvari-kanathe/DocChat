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
import database
from embeddings.embedder import embed_chunks, embed_query, get_model
from ingestion.chunker import chunk_documents
from ingestion.loader import load_document
from ingestion.triage import triage_documents
from rag.generator import generate_answer, condense_query
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
        "logged_in": False,         # True if user is logged in
        "username": None,           # username of logged in user
        "auth_action": "login",     # "login" or "register"
        "active_chat_id": None,     # UUID of the currently open chat
        "rename_target": None,      # chat_id being renamed
        "current_query": None,      # query in-flight during processing
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# ─── Chat Loading Helper ──────────────────────────────────────────────────────

def _load_chat(chat_id: str) -> None:
    """Load a chat's messages into session state."""
    raw_msgs = database.get_chat_messages(chat_id)
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = []
    st.session_state.conversation_history = []
    for msg in raw_msgs:
        content = msg["content"]
        if msg["role"] == "assistant":
            clean = content
            citations = []
            if "### Sources" in clean:
                parts = clean.split("### Sources")
                clean = parts[0].strip()
                for line in parts[1].strip().split("\n"):
                    if line.strip().startswith("*"):
                        citations.append({"source": line.replace("*", "").strip(), "page_number": None})
            st.session_state.messages.append({"role": "assistant", "content": clean, "citations": citations, "meta": {}})
        else:
            st.session_state.messages.append({"role": "user", "content": content})
        st.session_state.conversation_history.append({"role": msg["role"], "content": content})


# ─── User Authentication Gate ────────────────────────────────────────────────
if not st.session_state.logged_in:
    # Render premium styled auth header
    st.html(
        f"""
        <div class="auth-header">
          <div class="auth-logo">🗂️</div>
          <div class="auth-title">DocChat</div>
          <div class="auth-subtitle">{"Login to your document assistant" if st.session_state.auth_action == "login" else "Create your account"}</div>
        </div>
        """
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form(key="auth_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            if st.session_state.auth_action == "login":
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                if submit:
                    if database.authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        # Load most recent chat or create a new one
                        chats = database.get_user_chats(username)
                        if chats:
                            _load_chat(chats[0]["chat_id"])
                        else:
                            new_id = database.create_chat(username)
                            st.session_state.active_chat_id = new_id
                            st.session_state.messages = []
                            st.session_state.conversation_history = []
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            else:
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                if submit:
                    success, msg = database.register_user(username, password)
                    if success:
                        st.success(msg)
                        st.session_state.auth_action = "login"
                        import time
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)

        # Toggle between login and register
        if st.session_state.auth_action == "login":
            if st.button("Don't have an account? Sign up", use_container_width=True):
                st.session_state.auth_action = "register"
                st.rerun()
        else:
            if st.button("Already have an account? Sign in", use_container_width=True):
                st.session_state.auth_action = "login"
                st.rerun()

    st.stop()



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
        status_text.markdown(f"Chunking `{filename}` [{strategy}]...")
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
        status_text.markdown(f" Storing in vector database...")
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

    # ── User Account & Logout ──
    st.markdown(f"👤 **Logged in as:** `{st.session_state.username}`")
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.active_chat_id = None
        st.session_state.rename_target = None
        st.rerun()

    st.markdown("---")

    # ── LLM Provider Configuration (Hidden from UI) ──
    from rag.generator import PROVIDER_CONFIGS, resolve_provider

    detected_provider, _, _, _ = resolve_provider()
    selected_provider = detected_provider
    os.environ["LLM_PROVIDER"] = selected_provider
    config.LLM_PROVIDER = selected_provider

    cfg = PROVIDER_CONFIGS[selected_provider]
    key_env = cfg["key_env"]
    if key_env:
        default_val = getattr(config, key_env, "") or os.getenv(key_env, "")
        if default_val:
            os.environ[key_env] = default_val
            setattr(config, key_env, default_val)
        st.session_state.api_key_valid = _check_api_key(selected_provider, default_val)
    else:
        st.session_state.api_key_valid = True

    # ── NEW CHAT button ──
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        new_id = database.create_chat(st.session_state.username)
        st.session_state.active_chat_id = new_id
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.rename_target = None
        st.rerun()

    # ── Chat List ──
    render_sidebar_section("💬 Conversations")
    user_chats = database.get_user_chats(st.session_state.username)

    for chat in user_chats:
        cid = chat["chat_id"]
        is_active = cid == st.session_state.active_chat_id
        title = chat["title"] or "New Chat"
        short_title = title[:32] + "…" if len(title) > 32 else title

        # ── Inline rename form for this chat ──
        if st.session_state.rename_target == cid:
            with st.form(key=f"rename_{cid}", clear_on_submit=True):
                new_name = st.text_input("Rename chat", value=title, label_visibility="collapsed")
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.form_submit_button("✅ Save", use_container_width=True):
                        database.rename_chat(cid, new_name)
                        st.session_state.rename_target = None
                        st.rerun()
                with c2:
                    if st.form_submit_button("✖ Cancel", use_container_width=True):
                        st.session_state.rename_target = None
                        st.rerun()
        else:
            col_title, col_rename, col_delete = st.columns([5, 1, 1])
            with col_title:
                label = f"**{short_title}**" if is_active else short_title
                if st.button(label, key=f"load_{cid}", use_container_width=True):
                    _load_chat(cid)
                    st.session_state.rename_target = None
                    st.rerun()
            with col_rename:
                if st.button("✏️", key=f"rename_btn_{cid}", help="Rename", type="tertiary"):
                    st.session_state.rename_target = cid
                    st.rerun()
            with col_delete:
                if st.button("🗑️", key=f"delete_{cid}", help="Delete", type="tertiary"):
                    database.delete_chat(cid)
                    # If we deleted the active chat, open or create another
                    if cid == st.session_state.active_chat_id:
                        remaining = database.get_user_chats(st.session_state.username)
                        if remaining:
                            _load_chat(remaining[0]["chat_id"])
                        else:
                            nid = database.create_chat(st.session_state.username)
                            st.session_state.active_chat_id = nid
                            st.session_state.messages = []
                            st.session_state.conversation_history = []
                    st.rerun()

    st.markdown("---")

    # ── Upload ──
    render_sidebar_section("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported: PDF, DOCX, TXT",
    )

    # ── Chunking Strategy ──
    render_sidebar_section("⚙️ Chunking Strategy")
    strategy = st.selectbox(
        "Strategy",
        options=["fixed", "semantic"],
        format_func=lambda x: "Fixed (Default)" if x == "fixed" else "Paragraph (Semantic)",
        label_visibility="collapsed"
    )

    chunk_size = config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP
    top_k = config.TOP_K
    score_threshold = 0.10

    # ── Index Button ──
    st.markdown("<br>", unsafe_allow_html=True)
    index_clicked = st.button(
        "📥 Index Documents",
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
        render_sidebar_section(f"📄 Indexed ({len(sources)} files)")
        for src in sources:
            ext = Path(src).suffix.lstrip(".").lower()
            render_source_badge(src, ext)



# ─── Handle Actions ────────────────────────────────────────────────────────────

if clear_chat:
    cid = st.session_state.active_chat_id
    if cid:
        database.clear_chat_messages(cid)
    st.session_state.messages = []
    st.session_state.conversation_history = []
    st.rerun()

if clear_docs:
    ensure_store().reset()
    st.session_state.store = None
    st.session_state.indexed_sources = []
    st.session_state.chunk_count = 0
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
          <div style="font-size: 1rem; color: #8899b8;">Start a new conversation or select a previous chat from the sidebar.</div>
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

if query and not st.session_state.processing:
    # Ensure we have an active chat
    if not st.session_state.active_chat_id:
        new_id = database.create_chat(st.session_state.username)
        st.session_state.active_chat_id = new_id

    st.session_state.processing = True
    st.session_state.current_query = query
    st.session_state.messages.append({"role": "user", "content": query})

    # Save user message
    database.save_message(st.session_state.active_chat_id, "user", query)

    # Auto-set chat title from first user message
    chat_msgs = database.get_chat_messages(st.session_state.active_chat_id)
    user_msgs = [m for m in chat_msgs if m["role"] == "user"]
    if len(user_msgs) == 1:
        auto_title = query[:60] + ("…" if len(query) > 60 else "")
        database.update_chat_title(st.session_state.active_chat_id, auto_title)

    st.rerun()

if st.session_state.processing and st.session_state.get("current_query"):
    query = st.session_state.current_query
    active_cid = st.session_state.active_chat_id

    # Guard: no documents
    if store.count == 0:
        resp_content = EMPTY_STORE_RESPONSE
        st.session_state.messages.append(
            {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
        )
        database.save_message(active_cid, "assistant", resp_content)
        st.session_state.processing = False
        st.session_state.current_query = None
        st.rerun()

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
        database.save_message(active_cid, "assistant", resp_content)
        st.session_state.processing = False
        st.session_state.current_query = None
        st.rerun()

    else:
        thinking_placeholder = st.empty()
        with thinking_placeholder:
            render_thinking()

        try:
            # 1. Build conversation context from THIS chat only
            history_for_prompt = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.conversation_history[-6:]
            ]

            # 2. Condense follow-up query
            condensed_query_text = condense_query(
                query=query,
                conversation_history=history_for_prompt,
            )

            # 3. Retrieve
            context_chunks = retrieve(
                query=condensed_query_text,
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
                # 4. Build prompt
                messages = build_rag_prompt(
                    query=condensed_query_text,
                    context_chunks=context_chunks,
                    conversation_history=history_for_prompt,
                )

                # 5. Generate
                gen_resp = generate_answer(messages, context_chunks)
                resp_content = gen_resp["answer"]
                citations = gen_resp["citations"]
                meta = {
                    "model": gen_resp["model"],
                    "input_tokens": gen_resp["input_tokens"],
                    "output_tokens": gen_resp["output_tokens"],
                    "n_chunks": len(context_chunks),
                }

            # Update session conversation history (current chat only)
            st.session_state.conversation_history.append({"role": "user", "content": query})
            st.session_state.conversation_history.append({"role": "assistant", "content": resp_content})

            # Append to UI messages
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": resp_content,
                    "citations": citations if context_chunks else [],
                    "meta": meta if context_chunks else {},
                }
            )

            # Persist to DB with optional source block
            db_content = resp_content
            if citations:
                db_content += "\n\n### Sources\n" + "\n".join([f"* {c['source']}" for c in citations])
            database.save_message(active_cid, "assistant", db_content)

        except ValueError as exc:
            resp_content = f"⚠️ Configuration error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
            database.save_message(active_cid, "assistant", resp_content)
        except RuntimeError as exc:
            resp_content = f"❌ API error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
            database.save_message(active_cid, "assistant", resp_content)
        except Exception as exc:
            logger.exception("Unexpected error during generation")
            resp_content = f"❌ Unexpected error: {exc}"
            st.session_state.messages.append(
                {"role": "assistant", "content": resp_content, "citations": [], "meta": {}}
            )
            database.save_message(active_cid, "assistant", resp_content)
        finally:
            st.session_state.processing = False
            st.session_state.current_query = None
            thinking_placeholder.empty()
            st.rerun()


if index_clicked and uploaded_files:
    store = ensure_store()
    total_new_chunks = 0
    all_warnings: list[str] = []

    index_container = st.empty()
    with index_container.container():
        st.markdown("###  Indexing Documents")
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


