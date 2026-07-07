"""
Reusable Streamlit UI components for DocChat.
Provides helper functions for rendering chat messages, citation cards,
source badges, metrics panels, and styled containers.
"""
from __future__ import annotations

import streamlit as st


# ─── CSS Injection ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ──────────────────────────────────────── */
:root {
  --bg-primary:      #090d18;
  --bg-secondary:    #0f1629;
  --bg-card:         #141c2e;
  --bg-card-hover:   #1a2440;
  --border:          #1e2c4a;
  --border-accent:   #2a3f6e;
  --accent-purple:   #7c6aff;
  --accent-cyan:     #00d4ff;
  --accent-green:    #00e5b0;
  --accent-pink:     #ff6b9d;
  --text-primary:    #e8edf8;
  --text-secondary:  #8899b8;
  --text-muted:      #4a5a7a;
  --gradient-hero:   linear-gradient(135deg, #7c6aff 0%, #00d4ff 100%);
  --gradient-card:   linear-gradient(135deg, #141c2e 0%, #0f1629 100%);
  --shadow-glow:     0 0 30px rgba(124,106,255,0.15);
  --shadow-card:     0 4px 24px rgba(0,0,0,0.4);
  --radius-lg:       16px;
  --radius-md:       10px;
  --radius-sm:       6px;
}

/* ── Global Reset ────────────────────────────────────────── */
* { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background-color: var(--bg-primary) !important;
  color: var(--text-primary) !important;
}

.stApp {
  background: var(--bg-primary) !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-purple); }

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container {
  padding: 1.5rem 1rem !important;
}

/* ── Main Content ────────────────────────────────────────── */
.main .block-container {
  max-width: 900px;
  padding: 2rem 2rem 4rem !important;
}

/* ── Hero Header ─────────────────────────────────────────── */
.docchat-hero {
  background: linear-gradient(135deg, #7c6aff22 0%, #00d4ff18 100%);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-lg);
  padding: 2rem 2.5rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.docchat-hero::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -20%;
  width: 400px;
  height: 400px;
  background: radial-gradient(circle, #7c6aff15 0%, transparent 60%);
  pointer-events: none;
}
.docchat-hero h1 {
  font-size: 2rem;
  font-weight: 700;
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.4rem 0;
}
.docchat-hero p {
  color: var(--text-secondary);
  margin: 0;
  font-size: 0.95rem;
}

/* ── Chat Messages ───────────────────────────────────────── */
.chat-container {
  display: flex;
  flex-direction: column;
  gap: 1.2rem;
  margin-bottom: 1.5rem;
}

.chat-message {
  display: flex;
  gap: 0.8rem;
  align-items: flex-start;
  animation: fadeSlideIn 0.3s ease-out;
}
@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}

.chat-message.user { flex-direction: row-reverse; }

.chat-avatar {
  width: 38px;
  height: 38px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.1rem;
  flex-shrink: 0;
  border: 2px solid var(--border-accent);
}
.chat-avatar.user-avatar {
  background: linear-gradient(135deg, #7c6aff, #a855f7);
}
.chat-avatar.bot-avatar {
  background: linear-gradient(135deg, #00d4ff, #00e5b0);
}

.chat-bubble {
  max-width: 80%;
  padding: 1rem 1.2rem;
  border-radius: var(--radius-lg);
  font-size: 0.93rem;
  line-height: 1.65;
  position: relative;
}
.chat-bubble.user-bubble {
  background: linear-gradient(135deg, #7c6aff, #a855f7);
  color: #ffffff;
  border-bottom-right-radius: var(--radius-sm);
  margin-left: auto;
}
.chat-bubble.bot-bubble {
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-primary);
  border-bottom-left-radius: var(--radius-sm);
}
.chat-bubble.bot-bubble:hover {
  border-color: var(--border-accent);
  box-shadow: var(--shadow-card);
}

/* ── Citation Cards ──────────────────────────────────────── */
.citation-section {
  margin-top: 0.8rem;
  padding-top: 0.8rem;
  border-top: 1px solid var(--border);
}
.citation-label {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}
.citation-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}
.citation-card {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.3rem 0.7rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border-accent);
  border-left: 3px solid var(--accent-purple);
  border-radius: var(--radius-sm);
  font-size: 0.78rem;
  color: var(--text-secondary);
  cursor: default;
  transition: all 0.2s ease;
}
.citation-card:hover {
  background: var(--bg-card-hover);
  color: var(--text-primary);
  border-left-color: var(--accent-cyan);
  transform: translateY(-1px);
}
.citation-icon { color: var(--accent-purple); font-size: 0.8rem; }
.citation-page {
  background: var(--border-accent);
  color: var(--accent-cyan);
  padding: 0 0.3rem;
  border-radius: 3px;
  font-size: 0.7rem;
  font-family: 'JetBrains Mono', monospace;
}

/* ── Source Badges in Sidebar ────────────────────────────── */
.source-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.7rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 0.4rem;
  font-size: 0.8rem;
  color: var(--text-secondary);
  transition: all 0.2s ease;
}
.source-badge:hover {
  border-color: var(--accent-purple);
  color: var(--text-primary);
}
.source-icon { color: var(--accent-green); }

/* ── Metric Cards ────────────────────────────────────────── */
.metrics-row {
  display: flex;
  gap: 0.8rem;
  margin-top: 0.6rem;
  flex-wrap: wrap;
}
.metric-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  padding: 0.2rem 0.6rem;
  border-radius: 20px;
  font-size: 0.73rem;
  font-family: 'JetBrains Mono', monospace;
  border: 1px solid;
}
.metric-chip.tokens {
  background: #7c6aff18;
  border-color: #7c6aff55;
  color: var(--accent-purple);
}
.metric-chip.model {
  background: #00d4ff18;
  border-color: #00d4ff55;
  color: var(--accent-cyan);
}
.metric-chip.chunks {
  background: #00e5b018;
  border-color: #00e5b055;
  color: var(--accent-green);
}

/* ── Input Area ──────────────────────────────────────────── */
.stChatInput, .stChatInputContainer {
  border-color: var(--border-accent) !important;
}
.stChatInput > div {
  background: var(--bg-card) !important;
  border-radius: var(--radius-md) !important;
  border: 1px solid var(--border-accent) !important;
}
.stChatInput textarea {
  background: transparent !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── Streamlit Overrides ─────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, #7c6aff, #a855f7) !important;
  color: white !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  transition: all 0.2s ease !important;
}
.stButton > button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 15px rgba(124,106,255,0.4) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-accent) !important;
  color: var(--text-secondary) !important;
}
.stSelectbox > div > div {
  background: var(--bg-card) !important;
  border-color: var(--border-accent) !important;
  color: var(--text-primary) !important;
}
.stSlider { padding: 0 !important; }
[data-testid="stSlider"] > div > div > div {
  background: var(--gradient-hero) !important;
}
.stFileUploader {
  background: var(--bg-card) !important;
  border: 1px dashed var(--border-accent) !important;
  border-radius: var(--radius-md) !important;
}
.stFileUploader:hover {
  border-color: var(--accent-purple) !important;
}
.stExpander {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
div[data-testid="stMarkdownContainer"] p {
  color: var(--text-primary) !important;
}
.stAlert {
  border-radius: var(--radius-md) !important;
}
hr {
  border-color: var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Thinking indicator ──────────────────────────────────── */
.thinking-dots span {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent-purple);
  margin: 0 2px;
  animation: bounce 1.2s infinite ease-in-out;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; background: var(--accent-cyan); }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; background: var(--accent-green); }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.8); opacity: 0.6; }
  40% { transform: scale(1.2); opacity: 1; }
}

/* ── Status Bar ──────────────────────────────────────────── */
.status-bar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.8rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  font-size: 0.8rem;
  color: var(--text-secondary);
  margin-bottom: 1rem;
}
.status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.green { background: var(--accent-green); box-shadow: 0 0 6px var(--accent-green); }
.status-dot.yellow { background: #f5c842; box-shadow: 0 0 6px #f5c842; }
.status-dot.red { background: var(--accent-pink); box-shadow: 0 0 6px var(--accent-pink); }

/* ── Sidebar Section Headers ─────────────────────────────── */
.sidebar-section-header {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 1.2rem 0 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}
</style>
"""


# ─── Component Functions ──────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject global custom CSS into the Streamlit app."""
    st.html(CUSTOM_CSS)


def render_hero() -> None:
    """Render the DocChat hero header banner."""
    st.html(
        """
        <div class="docchat-hero">
          <h1>🗂️ DocChat</h1>
          <p>
            RAG-powered document intelligence — upload files, ask questions,
            get grounded answers with source citations.
          </p>
        </div>
        """
    )


def render_user_message(content: str) -> None:
    """Render a user chat bubble."""
    safe = content.replace("<", "&lt;").replace(">", "&gt;")
    st.html(
        f"""
        <div class="chat-message user">
          <div class="chat-avatar user-avatar">👤</div>
          <div class="chat-bubble user-bubble">{safe}</div>
        </div>
        """
    )


def markdown_to_html(md_text: str) -> str:
    """
    Parse a subset of Markdown to HTML:
      - Headings: ###, ##, # (rendered as styled divs to prevent Streamlit anchors)
      - Bold: **text**
      - Bullet lists: * or - (rendered as divs to avoid list resets)
      - Numbered lists: 1. (rendered as divs to avoid list resets)
      - Line breaks and paragraphs
    """
    import re
    import html

    lines = md_text.split("\n")
    processed_lines = []

    for line in lines:
        stripped = line.strip()

        # Headings
        if stripped.startswith("### "):
            title = stripped[4:]
            processed_lines.append(
                f"<div style='margin: 1rem 0 0.5rem 0; color: var(--accent-cyan); "
                f"font-weight: 700; font-size: 1.05rem; text-transform: uppercase; "
                f"letter-spacing: 0.05em;'>{html.escape(title)}</div>"
            )
            continue
        elif stripped.startswith("## "):
            title = stripped[3:]
            processed_lines.append(
                f"<div style='margin: 1.2rem 0 0.6rem 0; color: var(--accent-purple); "
                f"font-weight: 700; font-size: 1.2rem;'>{html.escape(title)}</div>"
            )
            continue
        elif stripped.startswith("# "):
            title = stripped[2:]
            processed_lines.append(
                f"<div style='margin: 1.4rem 0 0.8rem 0; color: var(--accent-purple); "
                f"font-weight: 700; font-size: 1.35rem;'>{html.escape(title)}</div>"
            )
            continue

        # Bullet list items
        ul_match = re.match(r"^[\*\-\+]\s+(.*)$", stripped)
        if ul_match:
            item_text = ul_match.group(1)
            parsed_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(item_text))
            processed_lines.append(
                f"<div style='margin: 0.3rem 0 0.3rem 0.8rem; display: flex; align-items: flex-start;'>"
                f"<span style='color: var(--accent-purple); margin-right: 0.5rem;'>•</span>"
                f"<span style='color: var(--text-primary);'>{parsed_text}</span>"
                f"</div>"
            )
            continue

        # Numbered list items
        ol_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ol_match:
            num = ol_match.group(1)
            item_text = ol_match.group(2)
            parsed_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(item_text))
            processed_lines.append(
                f"<div style='margin: 0.3rem 0 0.3rem 0.8rem; display: flex; align-items: flex-start;'>"
                f"<span style='color: var(--accent-cyan); font-weight: 600; margin-right: 0.4rem;'>{num}.</span>"
                f"<span style='color: var(--text-primary);'>{parsed_text}</span>"
                f"</div>"
            )
            continue

        # Blank/empty lines
        if not stripped:
            processed_lines.append("<div style='height: 0.3rem;'></div>")
            continue

        # Normal text paragraph
        parsed_text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html.escape(line))
        processed_lines.append(
            f"<div style='margin-bottom: 0.4rem; color: var(--text-primary); "
            f"line-height: 1.6;'>{parsed_text}</div>"
        )

    return "".join(processed_lines)
def render_assistant_message(
    content: str,
    citations: list[dict] | None = None,
    meta: dict | None = None,
) -> None:
    """
    Render an assistant chat bubble with optional citation cards and token metrics.

    Args:
        content:   Answer text (markdown formatted).
        citations: List of {"source": str, "page_number": int|None} dicts.
        meta:      Dict with optional keys: model, input_tokens, output_tokens, n_chunks.
    """
    import html

    safe_content = markdown_to_html(content)

    citation_html = ""
    if citations:
        cards = ""
        for c in citations:
            src = html.escape(c.get("source") or "Unknown")
            page = c.get("page_number")
            page_badge = f'<span class="citation-page">p.{page}</span>' if page else ""
            cards += (
                f'<div class="citation-card">'
                f'<span class="citation-icon">📄</span>'
                f'{src}{page_badge}'
                f"</div>"
            )
        citation_html = (
            '<div class="citation-section">'
            '<div class="citation-label">📎 Sources</div>'
            f'<div class="citation-cards">{cards}</div>'
            "</div>"
        )

    metric_html = ""

    st.html(
        f"""
        <div class="chat-message bot">
          <div class="chat-avatar bot-avatar">🤖</div>
          <div class="chat-bubble bot-bubble">
            {safe_content}
            {citation_html}
            {metric_html}
          </div>
        </div>
        """
    )


def render_thinking() -> None:
    """Render an animated 'thinking' indicator."""
    st.html(
        """
        <div class="chat-message bot">
          <div class="chat-avatar bot-avatar">🤖</div>
          <div class="chat-bubble bot-bubble">
            <div class="thinking-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
        """
    )


def render_status_bar(doc_count: int, chunk_count: int, api_ok: bool) -> None:
    """Render a compact status indicator bar."""
    dot_color = "green" if (api_ok and doc_count > 0) else ("yellow" if doc_count > 0 else "red")
    status_text = (
        f"{doc_count} source(s) indexed · {chunk_count} chunks in store"
        if doc_count > 0
        else "No documents indexed — upload files to begin"
    )
    api_indicator = "✅ API connected" if api_ok else "⚠️ No API key"
    st.html(
        f"""
        <div class="status-bar">
          <div class="status-dot {dot_color}"></div>
          <span>{status_text}</span>
          <span style="margin-left:auto; font-size:0.75rem;">{api_indicator}</span>
        </div>
        """
    )


def render_source_badge(filename: str, file_type: str = "") -> None:
    """Render a source file badge in the sidebar."""
    icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(file_type.lower(), "📁")
    safe = filename.replace("<", "&lt;")
    st.html(
        f"""
        <div class="source-badge">
          <span class="source-icon">{icon}</span>
          <span>{safe}</span>
        </div>
        """
    )


def render_sidebar_section(label: str) -> None:
    """Render a styled sidebar section header."""
    st.html(f'<div class="sidebar-section-header">{label}</div>')
