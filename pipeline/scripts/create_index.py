"""Explicit one-time index setup. Never invoked by application startup."""

from pathlib import Path
import sys

# Permit both python -m pipeline.scripts.create_index and direct invocation.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.config import Settings
from pipeline.embedding import EmbeddingService


def main():
    from pinecone import SchemaBuilder

    settings = Settings.from_env()
    embeddings = EmbeddingService(settings)
    try:
        client = embeddings.client
        if client.indexes.exists(settings.index_name):
            print(f"Index {settings.index_name!r} already exists; no changes made.")
            return
        schema = (
            SchemaBuilder()
            .add_dense_vector_field("vendor_dense", dimension=settings.dense_dimension, metric="cosine")
            .add_sparse_vector_field("vendor_sparse")
            .add_string_field("chunk_text", full_text_search={"language": "en"})
            .build()
        )
        client.indexes.create(
            name=settings.index_name,
            schema=schema,
            deployment={"deployment_type": "managed", "cloud": settings.cloud, "region": settings.region},
        )
        print(f"Index {settings.index_name!r} creation requested. Import vendors after the index is ready.")
    finally:
        embeddings.close()


if __name__ == "__main__":
    main()
