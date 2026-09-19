"""API routes connecting saved ingestion batches to vendor matching."""

import asyncio
import logging
from time import monotonic

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .errors import ClarificationNeeded, PipelineError
from .schema import MatchRequest, VendorImportRequest

logger = logging.getLogger(__name__)


def create_router(data_dir, enqueue_upload):
    router = APIRouter(prefix="/api/pipeline", tags=["Vendor matching"])

    async def match_saved(request, payload):
        from .parse_input import parse_input
        from .recommendation_engine import RecommendationEngine

        job = None
        if payload.job_id:
            job = await run_in_threadpool(request.app.state.store.get, payload.job_id)
            if job is None:
                raise PipelineError("This batch does not exist.", 404)
        normalized = await run_in_threadpool(
            parse_input, job, data_dir, payload.query_text, payload.file_ids,
        )
        settings = Settings.from_env()
        return await run_in_threadpool(RecommendationEngine(settings).match, **normalized)

    async def handle_match(request, payload):
        try:
            return await match_saved(request, payload)
        except ClarificationNeeded as exc:
            return JSONResponse({"type": "clarification", "message": str(exc)}, status_code=422)
        except PipelineError as exc:
            return JSONResponse({"type": "error", "message": str(exc)}, status_code=exc.status_code)
        except Exception:
            logger.exception("Vendor matching failed")
            return JSONResponse({"type": "error", "message": "Vendor matching failed. Check the server logs."}, status_code=500)

    @router.post("/match")
    async def match(request: Request, payload: MatchRequest):
        """Match chat text, a saved document batch, or both."""
        return await handle_match(request, payload)

    @router.post("/match-upload")
    async def match_upload(request: Request):
        """Accept files/doc_file plus optional query_text using the existing upload worker."""
        # Cache the same form object consumed and closed by enqueue_upload.
        form = await request.form(max_files=500, max_fields=5)
        query_text = form.get("query_text")
        if query_text is not None and not isinstance(query_text, str):
            await form.close()
            return JSONResponse({"message": "query_text must be text."}, status_code=422)
        if query_text and len(query_text) > 10_000:
            await form.close()
            return JSONResponse({"message": "query_text must be at most 10,000 characters."}, status_code=422)
        try:
            queued = await enqueue_upload(request)
        finally:
            await form.close()
        job_id = queued.headers["location"].rsplit("/", 1)[-1]
        payload = MatchRequest(job_id=job_id, query_text=query_text)
        deadline = monotonic() + 30
        while monotonic() < deadline:
            job = await run_in_threadpool(request.app.state.store.get, job_id)
            if job and job["status"] not in {"queued", "processing"}:
                return await handle_match(request, payload)
            await asyncio.sleep(0.5)
        return JSONResponse({
            "type": "processing", "job_id": job_id,
            "message": "Document processing is still running. Submit the follow-up request when it completes.",
            "status_url": f"/api/pipeline/batches/{job_id}",
            "next_request": {"method": "POST", "url": "/api/pipeline/match", "body": payload.model_dump(exclude_none=True)},
        }, status_code=202)

    @router.get("/batches/{job_id}")
    async def batch_status(request: Request, job_id: str):
        job = await run_in_threadpool(request.app.state.store.get, job_id)
        if job is None:
            return JSONResponse({"message": "This batch does not exist."}, status_code=404)
        return {"job_id": job["id"], "status": job["status"], "results": job["results"], "error": job["error"]}

    @router.post("/vendors/import")
    async def import_vendors(payload: VendorImportRequest):
        """Import vendor profiles separately from requirement documents."""
        try:
            from .embedding import EmbeddingService
            from .retrieval_engine import RetrievalEngine

            settings = Settings.from_env()
            embeddings = EmbeddingService(settings)
            try:
                engine = RetrievalEngine(settings, embeddings)
                return await run_in_threadpool(
                    engine.import_vendors, [vendor.model_dump() for vendor in payload.vendors],
                )
            finally:
                await run_in_threadpool(embeddings.close)
        except PipelineError as exc:
            return JSONResponse({"type": "error", "message": str(exc)}, status_code=exc.status_code)
        except Exception:
            logger.exception("Vendor import failed")
            return JSONResponse({"type": "error", "message": "Vendor import failed. Check the server logs."}, status_code=500)

    return router
