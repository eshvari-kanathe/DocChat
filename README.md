# 🗂️ DocChat — RAG-Based Document Q&A System

> **Retrieval-Augmented Generation** for private document corpora.  
> Upload PDFs, DOCX, or TXT files and get grounded, cited answers powered by
> SentenceTransformers + ChromaDB + Multi-Provider LLMs (Groq, Gemini, OpenAI, Ollama).

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           DocChat Pipeline                            │
│                                                                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│   │  Upload  │──▶ │  Triage  │──▶ │  Chunk   │──▶ │   Embed      │   │
│   │  UI      │    │ (dedup,  │    │ (fixed)  │    │ (all-MiniLM  │   │
│   │ (Streamlit)   │  filter) │    │          │    │  -L6-v2)     │   │
│   └──────────┘    └──────────┘    └──────────┘    └──────┬───────┘   │
│                                                           │           │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────▼───────┐   │
│   │  Answer  │◀── │   LLM    │◀── │  Prompt  │◀── │  ChromaDB    │   │
│   │ + Cite   │    │ (Groq/   │    │ Template │    │ (cosine top-k│   │
│   │  Badges  │    │  Gemini) │    │(grounded)│    │  retrieval)  │   │
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
│   ├── loaders.py                # PDF (PyMuPDF), DOCX, TXT extraction
│   ├── triage.py                 # Dedup + noise filtering
│   └── chunker.py                # Fixed-size & semantic chunking engines
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
│   └── generator.py              # Multi-provider client wrapper + citation parser
│
├── evaluation/
│   ├── test_set.json             # 15 sample Q&A pairs
│   └── evaluator.py              # Precision@k, Recall@k, LLM-as-judge (with backoff retries)
│
└── ui/
    ├── components.py             # Reusable styled Streamlit components
    └── app.py                    # Main Streamlit application
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- A Groq API key (Free, recommended), Google Gemini key, or OpenAI API key.

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

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and configure your credentials. By default, it is configured for Groq:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```

### 4. Run the App

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage Guide

### Indexing Documents

1. Drag and drop or select PDF / DOCX / TXT files under **📤 Upload Documents**.
2. Select your preferred **Chunking Strategy** (Fixed or Paragraph).
3. Click **🚀 Index Documents**.
   * Note: The backend automatically skips duplicate files using SHA-256 signatures, chunks files using optimal defaults (512 token size with 64 overlap), and saves them to local ChromaDB.

### Asking Questions

1. Type your question in the chat input at the bottom.
2. DocChat retrieves the most relevant paragraphs from ChromaDB.
3. The resolved LLM generates a grounded answer using **only** the retrieved context.
4. Citations appear as badges under the assistant bubble: `📎 Sources | 📄 policy.pdf p.3`
5. Ask follow-up questions — conversation history is automatically maintained across turns.

### When There's No Answer

If the answer isn't in the indexed documents, DocChat responds:
> *"I couldn't find information related to your question in the currently available documents.*
>
> *Please upload the relevant PDF, TXT, or DOCX file(s), and I'll be happy to search through them and provide an answer based on the uploaded content."*

This prevents hallucinations and guides the user to upload missing documents.

---

## Configuration Reference

All settings can be overridden via `.env`:

| Variable          | Default                | Description                          |
|-------------------|------------------------|--------------------------------------|
| `LLM_PROVIDER`    | `groq`                 | Active provider (`groq`, `gemini`, `openai`, `ollama`) |
| `GROQ_API_KEY`    | *(optional)*           | Groq API key                         |
| `GROQ_MODEL`      | `llama-3.1-8b-instant` | Groq chat completion model           |
| `OPENAI_API_KEY`  | *(optional)*           | OpenAI API key                       |
| `OPENAI_MODEL`    | `gpt-4o-mini`          | OpenAI chat completion model         |
| `GEMINI_API_KEY`  | *(optional)*           | Google Gemini API key                |
| `GEMINI_MODEL`    | `gemini-2.0-flash`     | Gemini chat completion model         |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama API endpoint       |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2`     | SentenceTransformers model           |
| `CHUNK_SIZE`      | `512`                  | Tokens per chunk (fixed strategy)    |
| `CHUNK_OVERLAP`   | `64`                   | Overlap tokens (fixed strategy)      |
| `TOP_K`           | `5`                    | Chunks retrieved per query           |
| `COLLECTION_NAME` | `docchat`              | ChromaDB collection name             |
| `LLM_TEMPERATURE` | `0.1`                  | LLM creativity (lower = more factual)|
| `MAX_TOKENS`      | `1024`                 | Max output tokens                    |

---

## Running the Evaluation Suite

First, generate the dummy documents using our test helper (which creates matching docs for the test set):
```bash
python generate_dummy_pdfs.py
```
Open your Streamlit UI, upload all PDFs from the newly created `dummy_docs/` folder, and index them. Then run the evaluator:

```bash
python -m evaluation.evaluator
```

**Metrics reported:**

| Metric           | Description                                               |
|------------------|-----------------------------------------------------------|
| Precision@k      | Fraction of top-k chunks containing relevant keywords    |
| Recall@k         | Fraction of expected keywords found in top-k chunks      |
| Top Cosine Score | Highest similarity score among retrieved chunks          |
| LLM Judge Score  | 1–5 quality score from active LLM evaluating the answer   |
| Latency          | Retrieval and generation time in milliseconds            |

*Note: The evaluator has built-in 20s pauses and auto-retries when hitting rate limits, which are common on free API tiers.*

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

- API keys are **never stored in UI code**; they must be set locally inside the private `.env` file.
- ChromaDB is stored locally at `./chroma_db/` — keep this directory private.

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
| `fpdf2`               | ≥2.8     | Dummy test data PDF generation |

---

## License

MIT License — see `LICENSE` for details.
