"""Read provider configuration without opening connections or creating indexes."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .errors import PipelineError


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.5-flash-lite"
    index_name: str = "vendor-matching"
    namespace: str = "vendors"
    cloud: str = "aws"
    region: str = "us-east-1"
    dense_model: str = "llama-text-embed-v2"
    sparse_model: str = "pinecone-sparse-english-v0"
    dense_dimension: int = 1024
    rerank_model: str = "bge-reranker-v2-m3"
    top_k: int = 25
    top_n: int = 10
    timeout_seconds: int = 60

    @classmethod
    def from_env(cls):
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise PipelineError("Install the pipeline requirements before using matching.", 503) from exc
        root = Path(__file__).resolve().parent
        load_dotenv(root / ".env", override=False)
        load_dotenv(root.parent / ".env", override=False)
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            index_name=os.getenv("PINECONE_INDEX_NAME", "vendor-matching"),
            namespace=os.getenv("PINECONE_NAMESPACE", "vendors"),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )
