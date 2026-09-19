"""Hosted dense and sparse embeddings shared by import and retrieval."""

from __future__ import annotations

import math
from threading import Lock
from typing import Any

from .errors import PipelineError


class EmbeddingService:
    def __init__(self, settings):
        self.settings = settings
        self._client = None
        self._lock = Lock()

    @property
    def client(self):
        # Importing the application must not require provider credentials.
        with self._lock:
            if self._client is None:
                if not self.settings.pinecone_api_key:
                    raise PipelineError("Set PINECONE_API_KEY to use vendor matching.", status_code=503)
                try:
                    from pinecone import Pinecone

                    self._client = Pinecone(api_key=self.settings.pinecone_api_key, timeout=self.settings.timeout_seconds)
                except ImportError as exc:
                    raise PipelineError("Install the pipeline requirements to use Pinecone.", status_code=503) from exc
            return self._client

    def embed(self, texts: list[str], input_type: str = "passage") -> list[dict[str, Any]]:
        if input_type not in {"passage", "query"}:
            raise PipelineError("Embedding input_type must be passage or query.", status_code=422)
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise PipelineError("Embedding text cannot be empty.", status_code=422)
        result = []
        try:
            # The sparse model accepts fewer inputs per call than the dense model.
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                dense = self.client.inference.embed(
                    model=self.settings.dense_model,
                    inputs=batch,
                    parameters={"input_type": input_type, "dimension": self.settings.dense_dimension, "truncate": "NONE"},
                )
                sparse = self.client.inference.embed(
                    model=self.settings.sparse_model,
                    inputs=batch,
                    parameters={"input_type": input_type, "truncate": "NONE"},
                )
                dense_rows, sparse_rows = list(dense), list(sparse)
                if len(dense_rows) != len(batch) or len(sparse_rows) != len(batch):
                    raise PipelineError("Pinecone returned an incomplete embedding batch.")
                for dense_row, sparse_row in zip(dense_rows, sparse_rows):
                    values = list(dense_row["values"])
                    indices = list(sparse_row["sparse_indices"])
                    weights = list(sparse_row["sparse_values"])
                    if len(values) != self.settings.dense_dimension or len(indices) != len(weights):
                        raise PipelineError("Pinecone returned embeddings with an unexpected shape.")
                    if not all(math.isfinite(value) for value in values + weights):
                        raise PipelineError("Pinecone returned invalid embedding values.")
                    result.append({"dense": values, "sparse": {"indices": indices, "values": weights}})
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError("Pinecone embedding failed. Check the API key, model access, and service availability.") from exc
        return result

    def close(self) -> None:
        """Release connections after a request without initializing a client."""
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None
