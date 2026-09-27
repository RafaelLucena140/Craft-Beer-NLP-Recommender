"""Runtime configuration shared by the web and CLI entry points."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    raw_data_path: Path = Path(os.getenv("BEER_RAW_DATA_PATH", "data/raw/beers_dataset.csv"))
    prepared_data_path: Path = Path(os.getenv("BEER_DATA_PATH", "data/raw/beers_cleaned.csv"))
    sample_size: int = int(os.getenv("BEER_SAMPLE_SIZE", "5000"))
    sample_seed: int = int(os.getenv("BEER_SAMPLE_SEED", "42"))
    ingestion_batch_size: int = int(os.getenv("INGESTION_BATCH_SIZE", "256"))
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost")
    chroma_port: int = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "craft_beers")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    retrieval_candidates: int = int(os.getenv("RETRIEVAL_CANDIDATES", "30"))
    minimum_rating_reviews: int = int(os.getenv("MINIMUM_RATING_REVIEWS", "5"))
    high_rating_threshold: float = float(os.getenv("HIGH_RATING_THRESHOLD", "4.0"))


settings = Settings()
