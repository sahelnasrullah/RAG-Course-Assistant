"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", ROOT / "index"))
COLLECTION = os.getenv("RAG_COLLECTION", "course_docs")

# Embeddings
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/e5-base-v2")
# e5 models expect "query: " / "passage: " prefixes. For bge use
# QUERY_PREFIX="Represent this sentence for searching relevant passages: " and DOC_PREFIX="".
QUERY_PREFIX = os.getenv("RAG_QUERY_PREFIX", "query: ")
DOC_PREFIX = os.getenv("RAG_DOC_PREFIX", "passage: ")

# Chunking (characters, roughly 4 chars per token)
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1200))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))

# Retrieval
TOP_K = int(os.getenv("RAG_TOP_K", 4))
RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL_MODE", "hybrid")  # dense | bm25 | hybrid

# Generation
# backend "hf" runs the model with transformers (CPU locally, GPU on Colab);
# backend "ollama" talks to a local Ollama server.
LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "hf")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")  # Colab GPU: Qwen/Qwen2.5-7B-Instruct
OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "http://localhost:11434")
MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", 300))
LLM_DTYPE = os.getenv("RAG_LLM_DTYPE", "auto")  # auto | float32 | bfloat16 | float16

# Evaluation judge (only used by eval/run_eval.py judge; needs GEMINI_API_KEY).
JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "gemini-3.5-flash-lite")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); never overrides real env vars."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
