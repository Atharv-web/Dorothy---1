"""Document-aware chunks without loading an embedding model or tokenizer."""

import re

from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker


def clean_text(text: str) -> str:
    """Keep the ingestion layer's ampersand cleanup for older stored outputs."""
    return re.sub(r"&amp(?:;|(?![a-zA-Z0-9]))", "and", text, flags=re.IGNORECASE)


def split_text(text: str, max_characters: int = 1500) -> list[str]:
    """Prefer paragraph/word boundaries; retain every character across pieces."""
    if max_characters < 1:
        raise ValueError("max_characters must be positive.")
    pieces = []
    remaining = text
    while len(remaining) > max_characters:
        window = remaining[:max_characters]
        boundary = window.rfind("\n") + 1
        if boundary < max_characters // 2:
            boundary = window.rfind(" ") + 1
        if boundary < max_characters // 2:
            boundary = max_characters
        pieces.append(remaining[:boundary])
        remaining = remaining[boundary:]
    if remaining:
        pieces.append(remaining)
    return pieces


def chunk_document(document, file_id: str, filename: str,
                   max_characters: int = 1500) -> list[dict]:
    """Preserve headings and item references while bounding each chunk's text.

    Pages identify the source items of the parent structural chunk. A split
    chunk can therefore cite several pages; these are not exact text offsets.
    """
    if max_characters < 1:
        raise ValueError("max_characters must be positive.")
    records = []
    chunker = HierarchicalChunker(always_emit_headings=True)
    for structural_index, chunk in enumerate(chunker.chunk(dl_doc=document)):
        text = clean_text(chunk.text)
        if not text.strip():
            continue
        items = chunk.meta.doc_items
        pages = sorted({provenance.page_no for item in items for provenance in item.prov})
        references = list(dict.fromkeys(item.self_ref for item in items))
        headings = [clean_text(heading) for heading in (chunk.meta.headings or [])]
        for part_index, piece in enumerate(split_text(text, max_characters)):
            records.append({
                "chunk_id": f"{file_id}:{structural_index}:{part_index}",
                "text": piece,
                "file_id": file_id,
                "filename": filename,
                "pages": pages,
                "headings": headings,
                "references": references,
                "structural_index": structural_index,
                "part_index": part_index,
            })
    return records
