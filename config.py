"""
Central configuration for DocChat.
All settings are read from environment variables (via .env) with safe defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ─── LLM Provider ────────────────────────────────────────────────────────────
# Options: groq | openai | gemini | ollama
# Leave blank to auto-detect from whichever API key is present.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")

# ─── Groq (FREE — recommended) ───────────────────────────────────────────────
# Get a free key at: https://console.groq.com
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ─── Google Gemini (FREE tier) ────────────────────────────────────────────────
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ─── Ollama (Local, no key needed) ───────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ─── Shared LLM Settings ─────────────────────────────────────────────────────
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))        # tokens (approx)
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))   # tokens (approx)
MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "30"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "docchat")

# ─── Triage ───────────────────────────────────────────────────────────────────
MIN_DOC_TOKENS: int = int(os.getenv("MIN_DOC_TOKENS", "20"))
BLOCKED_PATTERNS: list[str] = [
    "~$*",          # Office temp files
    ".DS_Store",    # macOS metadata
    "Thumbs.db",    # Windows thumbnails
    "desktop.ini",  # Windows folder settings
    "*.tmp",
    "*.log",
]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
