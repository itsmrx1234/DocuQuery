"""Environment-driven settings for the local RAG pipeline."""
from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    relevance_threshold: float | None = (
        float(os.environ["RELEVANCE_THRESHOLD"])
        if os.getenv("RELEVANCE_THRESHOLD", "").strip()
        else None
    )
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))


settings = Settings()
