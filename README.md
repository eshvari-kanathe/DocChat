# 🗂️ DocChat — RAG-Based Document Q&A System

> **Retrieval-Augmented Generation** for private document corpora.  
> Upload PDFs, DOCX, or TXT files and get grounded, cited answers powered by
> SentenceTransformers + ChromaDB + GPT-4o-mini.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           DocChat Pipeline                            │
│                                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│   │  Upload  │──▶ │  Triage  │──▶ │  Chunk   │──▶ │   Embed      │   │
│   │  UI      │    │ (dedup,  │    │ (fixed / │    │ (all-MiniLM  │   │
│   │ (Streamlit)   │  filter) │    │ semantic)│    │  -L6-v2)     │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘   │
│                                                           │           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────▼───────┐   │
│   │  Answer  │◀── │   LLM    │◀── │  Prompt  │◀── │  ChromaDB    │   │
│   │ + Cite   │    │(GPT-4o-  │    │ Template │    │ (cosine top-k│   │
│   │  Cards   │    │  mini)   │    │(grounded)│    │  retrieval)  │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
docchat/
│
├── config.py                     # Central config (env vars + defaults)
├── requirements.txt              # All Python dependencies
├── .env.example                  # API key template → copy to .env
│
├── ingestion/
│   ├── loader.py                 # PDF (PyMuPDF), DOCX, TXT extraction
│   ├── triage.py                 # Dedup + noise filtering
│   └── chunker.py                # Fixed-size & semantic chunking
│
├── embeddings/
│   └── embedder.py               # SentenceTransformers (all-MiniLM-L6-v2)
│
├── vectorstore/
│   └── chroma_store.py           # ChromaDB CRUD + cosine similarity search
│
├── rag/
│   ├── retriever.py              # Top-k retrieval with score threshold
│   ├── prompt_templates.py       # Grounded system prompt + builder
│   └── generator.py              # OpenAI chat completions + citation parser
│
├── evaluation/
│   ├── test_set.json             # 15 sample Q&A pairs
│   └── evaluator.py              # Precision@k, Recall@k, LLM-as-judge
│
└── ui/
    ├── components.py             # Reusable styled Streamlit components
    └── app.py                    # Main Streamlit application
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- An OpenAI API key (`gpt-4o-mini` or higher)

### 2. Clone & Install

```bash
# Navigate to the project directory
cd docchat

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY=sk-...
```

Or enter it directly in the Streamlit sidebar (session-only, not saved to disk).

### 4. Run the App

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage Guide

### Indexing Documents

1. Open the sidebar → **Upload Documents**
2. Drag and drop or select PDF / DOCX / TXT files
3. Choose your **chunking strategy**:
   - **Fixed-size with overlap** — predictable, fast, good baseline
   - **Semantic (paragraph)** — context-preserving, better for structured docs
4. Adjust **Chunk Size** and **Overlap** sliders
5. Click **🚀 Index Documents**

### Asking Questions

1. Type your question in the chat input at the bottom
2. DocChat retrieves the most relevant chunks (configurable top-k)
3. GPT-4o-mini generates a grounded answer using **only** the retrieved context
4. Citations appear as cards under each answer: `📄 policy.pdf | p.3`
5. Ask follow-up questions — conversation history is maintained across turns

### When There's No Answer

If the answer isn't in the indexed documents, DocChat responds:
> *"I don't know based on the provided documents."*

This prevents hallucinations.

---

## Configuration Reference

All settings can be overridden via `.env`:

| Variable          | Default              | Description                          |
|-------------------|----------------------|--------------------------------------|
| `OPENAI_API_KEY`  | *(required)*         | OpenAI API key                       |
| `OPENAI_MODEL`    | `gpt-4o-mini`        | Chat completion model                |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`   | SentenceTransformers model           |
| `CHUNK_SIZE`      | `512`                | Tokens per chunk (fixed strategy)    |
| `CHUNK_OVERLAP`   | `64`                 | Overlap tokens (fixed strategy)      |
| `TOP_K`           | `5`                  | Chunks retrieved per query           |
| `COLLECTION_NAME` | `docchat`            | ChromaDB collection name             |
| `LLM_TEMPERATURE` | `0.1`                | LLM creativity (lower = more factual)|
| `MAX_TOKENS`      | `1024`               | Max output tokens                    |

---

## Running the Evaluation Suite

First, index documents that match the topics in `evaluation/test_set.json`, then:

```bash
python -m evaluation.evaluator \
    --test-set evaluation/test_set.json \
    --output evaluation/results.csv \
    --top-k 5
```

**Metrics reported:**

| Metric           | Description                                               |
|------------------|-----------------------------------------------------------|
| Precision@k      | Fraction of top-k chunks containing relevant keywords    |
| Recall@k         | Fraction of expected keywords found in top-k chunks      |
| Top Cosine Score | Highest similarity score among retrieved chunks          |
| LLM Judge Score  | 1–5 quality score from GPT-4o-mini evaluating the answer |
| Latency          | Retrieval and generation time in milliseconds            |

**Tuning recommendations:**

| Setting       | Effect of increasing                              |
|---------------|---------------------------------------------------|
| Chunk Size    | More context per chunk; may dilute relevance      |
| Overlap       | Less information loss at boundaries; more chunks  |
| Top-K         | Higher recall; more context tokens consumed       |
| Score Threshold | Higher precision; may miss relevant content     |

---

## Supported File Types

| Extension | Parser    | Page-accurate |
|-----------|-----------|---------------|
| `.pdf`    | PyMuPDF   | ✅ Yes        |
| `.docx`   | python-docx | ❌ (whole doc)|
| `.doc`    | python-docx | ❌ (whole doc)|
| `.txt`    | Built-in  | ❌ (whole doc)|

---

## Security

- API keys are **never written to disk** by the UI — stored only in the session
- Use `.env` locally; use environment secrets in production deployments
- ChromaDB is stored locally at `./chroma_db/` — keep this directory private

---

## Dependencies

| Library               | Version  | Purpose                        |
|-----------------------|----------|--------------------------------|
| `openai`              | ≥1.30    | LLM generation                 |
| `sentence-transformers` | ≥3.0  | Local embeddings               |
| `chromadb`            | ≥0.5     | Vector storage & search        |
| `PyMuPDF`             | ≥1.24    | PDF text extraction            |
| `python-docx`         | ≥1.1     | DOCX parsing                   |
| `streamlit`           | ≥1.36    | Web UI                         |
| `pandas`              | ≥2.2     | Evaluation data handling       |

---

## License

MIT License — see `LICENSE` for details.
