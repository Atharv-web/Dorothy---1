"""Read completed ingestion outputs; never create another document converter."""

import json
import logging
import re
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from .chunking import chunk_document, clean_text
from .errors import ClarificationNeeded, PipelineError

logger = logging.getLogger(__name__)
MAX_DOCUMENT_CHARACTERS = 120_000
MAX_QUERY_CHARACTERS = 10_000
_ID = re.compile(r"[a-f0-9]{32}")


def _output_path(data_dir: Path, batch_id: str, file_id: str) -> Path:
    """Only internally generated UUIDs can address the ingestion output tree."""
    if not _ID.fullmatch(batch_id) or not _ID.fullmatch(file_id):
        raise PipelineError("The stored document identifier is invalid.", 409)
    root = Path(data_dir).resolve()
    output = root / batch_id / "output"
    candidate = (output / f"{file_id}.json").resolve()
    if not candidate.is_relative_to(root) or candidate.parent != output:
        raise PipelineError("The stored document path is invalid.", 409)
    return candidate


def parse_input(job: dict | None, data_dir: Path, query_text: str | None,
                file_ids: list[str] | None = None) -> dict:
    """Keep the chat query and complete document text as separate inputs.

    Failed, skipped, partial, and unreadable outputs are explicitly reported.
    A selected batch without readable text requires clarification even if the
    user also supplied a query, rather than silently ignoring their documents.
    """
    query = query_text.strip() if query_text else None
    if query and len(query) > MAX_QUERY_CHARACTERS:
        raise PipelineError("The query exceeds 10,000 characters. Shorten it and try again.", 413)
    sources = []
    warnings = []
    blocks = []
    total_characters = 0
    if file_ids is not None and not job:
        raise PipelineError("Select an ingestion batch before selecting document IDs.", 422)
    if job:
        if job["status"] in {"queued", "processing"}:
            raise PipelineError("This batch is still being processed. Wait for ingestion to finish.", 409)
        results = job.get("results", [])
        if file_ids is not None:
            if not file_ids:
                raise ClarificationNeeded("Select at least one processed document.")
            known_ids = {row.get("id") for row in results if row.get("id")}
            if set(file_ids) - known_ids:
                raise PipelineError("One or more selected document IDs do not belong to this batch.", 404)
            selected = set(file_ids)
        else:
            selected = None
        if job["status"] not in {"completed", "partial"}:
            warnings.append(f"Batch status is {job['status']}; only its readable completed outputs are used.")
        if job.get("error"):
            warnings.append(str(job["error"]))
        for row in results:
            file_id = row.get("id")
            filename = str(row.get("name") or file_id or "Unnamed document")
            status = row.get("status", "failed")
            # Failed/skipped archive entries often have no ID. Surface them even
            # when individual successful documents have been selected.
            if status not in {"completed", "partial"}:
                if selected is None or not file_id or file_id in selected:
                    warnings.append(f"{filename}: {status}. {row.get('message') or 'No extracted text is available.'}")
                continue
            if selected is not None and file_id not in selected:
                continue
            if status == "partial":
                warnings.append(f"{filename}: extraction is partial. {row.get('message') or 'Some content may be missing.'}")
            if not file_id:
                warnings.append(f"{filename}: no stored document ID is available.")
                continue
            path = _output_path(data_dir, str(job["id"]), str(file_id))
            try:
                document = DoclingDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
                text = clean_text(document.export_to_markdown()).strip()
            except (OSError, UnicodeError, ValueError, TypeError):
                logger.exception("Could not read ingestion output %s/%s", job["id"], file_id)
                warnings.append(f"{filename}: its extracted document could not be read. Upload it again.")
                continue
            if not text:
                warnings.append(f"{filename}: no usable extracted text was found.")
                continue
            block = f"[DOCUMENT: {filename}; FILE_ID: {file_id}]\n{text}\n[/DOCUMENT]"
            total_characters += len(block) + (2 if blocks else 0)
            if total_characters > MAX_DOCUMENT_CHARACTERS:
                raise PipelineError(
                    "The selected documents exceed 120,000 characters. Select fewer documents or split the batch.",
                    413,
                )
            # Structural chunks are metadata for downstream consumers. Full
            # Markdown above remains the extraction input so no text is lost
            # when the chunker's table serialization differs from Markdown.
            chunks = chunk_document(document, str(file_id), filename)
            pages = sorted({page for chunk in chunks for page in chunk["pages"]})
            sources.append({
                "filename": filename, "file_id": file_id, "status": status,
                "pages": pages, "chunks": chunks,
            })
            blocks.append(block)
        if not blocks:
            message = "No readable text was found in the selected documents. Upload a readable document or submit only a query."
            if warnings:
                message += " " + " ".join(warnings[:5])
            raise ClarificationNeeded(message)
    if not blocks and not query:
        raise ClarificationNeeded("Upload a requirement document or describe what you need to source.")
    return {"doc_text": "\n\n".join(blocks) or None, "query_text": query or None,
            "sources": sources, "warnings": warnings}
