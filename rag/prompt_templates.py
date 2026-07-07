"""
Prompt templates for the DocChat RAG pipeline.

Design principles:
  1. STRICT grounding — the model must only use provided context.
  2. EXPLICIT "I don't know" fallback to prevent hallucinations.
  3. STRUCTURED citations in [Source: filename, Page X] format.
  4. CONVERSATION HISTORY support for multi-turn awareness.
"""
from __future__ import annotations

from typing import Any

SearchResult = dict[str, Any]

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are DocChat, a precise and trustworthy document assistant.

Your task is to answer questions based STRICTLY on the provided CONTEXT excerpts from internal documents. You do not have access to the internet or any knowledge beyond what is in the CONTEXT below.

## Mandatory Formatting Rules

Every response must be formatted using the following markdown template exactly (do not output any text before "### Answer"):

### Answer

<your natural, conversational answer text here>

### Sources

* <document_name> (Page <number>)

## Mandatory Response Rules

1. **Groundedness**: Answer ONLY using information from the CONTEXT. Do not use prior knowledge, guesses, or external information. Do not fabricate document names, page numbers, or statistics.
2. **Uncertainty**: If the answer is not found in the context, or if the context is insufficient or ambiguous, you MUST ignore the template above and respond with exactly:
"I couldn't find this information in the uploaded documents. Please upload a relevant document or try rephrasing your question."
3. **No Inline Citations**: Never include inline citations like "[Source: X]" or references to source names/pages within the body of your answer. Mention document names and page numbers ONLY in the "### Sources" section at the very bottom.
4. **Natural & Conversational**: Provide a natural, conversational response. Do not repeat the user's question back to them.
5. **Synthesis**: If multiple documents or excerpts support the answer, merge the information into a single, well-written response.
6. **Bullet Points**: Use bullet points (using '*') when listing multiple items.
7. **Bold Highlighting**: Highlight all important values (numbers, dates, policies, limits, percentages) in **bold**.
8. **Conciseness**: Keep answers concise (3–6 sentences) unless the user explicitly asks for details.
9. **Formatting**: Preserve formatting such as tables, lists, and numbered steps whenever appropriate.
"""


# ─── Prompt Builder ──────────────────────────────────────────────────────────

def build_rag_prompt(
    query: str,
    context_chunks: list[SearchResult],
    conversation_history: list[dict] | None = None,
) -> list[dict]:
    """
    Assemble the full message list for the OpenAI chat completions API.

    Args:
        query:                Current user question.
        context_chunks:       Top-k SearchResult dicts from the retriever.
        conversation_history: Previous {role, content} messages for multi-turn
                              context. Should NOT include the current query.

    Returns:
        List of message dicts ready for client.chat.completions.create().
    """
    # Build the context block with numbered citations
    context_parts: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown Document")
        page = meta.get("page_number", "?")
        score = chunk.get("score", 0.0)
        chunk_text = chunk.get("text", "").strip()

        context_parts.append(
            f"--- Context Excerpt {i} ---\n"
            f"Source: {source} | Page: {page} | Relevance Score: {score:.2f}\n\n"
            f"{chunk_text}"
        )

    context_block = "\n\n".join(context_parts) if context_parts else "(No relevant context found)"

    # Assemble messages
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject previous conversation turns for multi-turn memory
    if conversation_history:
        messages.extend(conversation_history)

    # Current user message with injected context
    user_content = (
        f"## Retrieved Context\n\n"
        f"{context_block}\n\n"
        f"{'─' * 60}\n\n"
        f"## Question\n\n"
        f"{query}\n\n"
        f"Please answer the question based solely on the context above. "
        f"Do not include any inline citations or mention document names/page numbers in the answer body. "
        f"Format the response using the ### Answer and ### Sources format."
    )

    messages.append({"role": "user", "content": user_content})
    return messages


# ─── Fallback / Error Templates ──────────────────────────────────────────────

NO_CONTEXT_RESPONSE = "I couldn't find this information in the uploaded documents. Please upload a relevant document or try rephrasing your question."

EMPTY_STORE_RESPONSE = "No documents have been indexed yet. Please upload one or more PDF, DOCX, or TXT files using the sidebar, then ask your question."

