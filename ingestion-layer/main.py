"""Run with: python main.py. Use one server process with the embedded worker."""

import logging
import os
import shutil
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from ingestion import Ingestor, MAX_FILES, MAX_FILE_BYTES, MAX_UPLOAD_BYTES, SUPPORTED, safe_name
from storage import Store, new_id

BASE = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
PREVIEW_CHARACTERS = 20_000


class UploadLimit:
    """Bound the request stream before multipart parsing can fill temporary storage."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > MAX_UPLOAD_BYTES + 2 * 1024 * 1024:
                raise HTTPException(413, "Upload exceeds 200 MiB. Split it into smaller batches.")
            return message

        await self.app(scope, limited_receive, send)


def create_app(data_dir=None, ingestor=None):
    data_dir = Path(data_dir or os.environ.get("INGESTION_DATA_DIR", BASE / "data")).resolve()
    templates = Jinja2Templates(directory=str(BASE / "ui"))
    processor = ingestor or Ingestor()

    @asynccontextmanager
    async def lifespan(app):
        store = Store(data_dir)
        store.recover()
        app.state.store = store
        app.state.ready = threading.Event()
        app.state.startup_error = None
        stop = threading.Event()

        def work():
            try:
                logger.info("Loading ingestion models using %s", sys.executable)
                processor.warmup()
                app.state.ready.set()
            except Exception:
                logger.exception("Ingestion model initialization failed")
                app.state.startup_error = "Document processing could not start. Check the server logs, then restart the application."
                return
            while not stop.is_set():
                try:
                    job = store.claim()
                    if not job:
                        stop.wait(0.5)
                        continue
                    try:
                        results = processor.process(
                            job, data_dir / job["id"],
                            lambda rows: store.update(job["id"], "processing", rows),
                        )
                        statuses = {r["status"] for r in results}
                        if statuses == {"completed"}:
                            status = "completed"
                        elif statuses & {"completed", "partial"}:
                            status = "partial"
                        elif statuses == {"skipped"}:
                            status = "skipped"
                        else:
                            status = "failed"
                        store.update(job["id"], status, results)
                    except Exception:
                        logger.exception("Job failed: %s", job["id"])
                        saved = store.get(job["id"])
                        store.update(job["id"], "failed", saved["results"], "Processing stopped unexpectedly. Check the server logs.")
                except Exception:
                    logger.exception("Worker error")
                    stop.wait(1)

        worker = threading.Thread(target=work, name="ingestion", daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            await run_in_threadpool(worker.join)

    app = FastAPI(title="Dorothy · Ingestion", lifespan=lifespan)
    app.add_middleware(UploadLimit)
    app.mount("/static", StaticFiles(directory=BASE / "ui"), name="static")

    def render(request, job=None, error=None, status_code=200, preview_id=None):
        ready = app.state.ready.is_set()
        startup_error = app.state.startup_error
        busy = bool(job and job["status"] in {"queued", "processing"})
        preview = None
        if job:
            outputs = [row for row in job["results"]
                       if row.get("id") and row["status"] in {"completed", "partial"}]
            selected = next((row for row in outputs if row["id"] == preview_id), None) if preview_id else next(iter(outputs), None)
            if preview_id and selected is None:
                raise HTTPException(404, "Preview not found.")
            if selected:
                preview = {"item": selected, "text": "", "truncated": False, "error": None}
                path = data_dir / job["id"] / "output" / f'{selected["id"]}.md'
                try:
                    with path.open(encoding="utf-8") as file:
                        text = file.read(PREVIEW_CHARACTERS + 1)
                    preview["text"] = text[:PREVIEW_CHARACTERS]
                    preview["truncated"] = len(text) > PREVIEW_CHARACTERS
                except (OSError, UnicodeError):
                    logger.exception("Could not read preview for batch %s", job["id"])
                    preview["error"] = "This preview is unavailable. Try the download or check the server logs."
        # A selected preview stays still while the user reads; only status views refresh.
        refresh = not error and not preview_id and not startup_error and (not ready or busy)
        return templates.TemplateResponse(
            request=request, name="index.html",
            context={"jobs": app.state.store.recent(), "job": job, "error": error,
                     "ready": ready, "startup_error": startup_error, "busy": busy,
                     "refresh": refresh, "preview": preview, "preview_selected": bool(preview_id),
                     "accept": ",".join(sorted(SUPPORTED))}, status_code=status_code,
        )

    @app.exception_handler(StarletteHTTPException)
    async def error_page(request, exc):
        return render(request, error=str(exc.detail), status_code=exc.status_code)

    @app.get("/")
    def index(request: Request):
        return render(request)

    @app.get("/ready")
    def readiness():
        ready = app.state.ready.is_set()
        state = "ready" if ready else "failed" if app.state.startup_error else "loading"
        return JSONResponse({"status": state}, status_code=200 if ready else 503)

    @app.post("/upload")
    async def upload(request: Request):
        if not app.state.ready.is_set():
            raise HTTPException(503, app.state.startup_error or "The workspace is still preparing. Please wait until uploads are enabled.")
        job_id = new_id()
        root = data_dir / job_id
        inputs = []
        total = 0
        queued = False
        try:
            async with request.form(max_files=MAX_FILES, max_fields=5) as form:
                files = [value for value in form.getlist("files")
                         if isinstance(value, UploadFile) and value.filename]
                if not files:
                    raise HTTPException(400, "Choose at least one file or a folder first.")
                (root / "source").mkdir(parents=True)
                for file in files:
                    try:
                        name = safe_name(file.filename)
                    except ValueError as exc:
                        raise HTTPException(400, str(exc)) from exc
                    suffix = Path(name).suffix.lower()
                    # Unsupported folder contents are recorded as skipped by the worker.
                    stored = new_id() + (suffix if suffix in SUPPORTED else ".unsupported")
                    size = 0
                    with (root / "source" / stored).open("wb") as destination:
                        while chunk := await file.read(1024 * 1024):
                            size += len(chunk)
                            total += len(chunk)
                            if size > MAX_FILE_BYTES or total > MAX_UPLOAD_BYTES:
                                raise HTTPException(413, "Limit exceeded: 50 MiB per file, 200 MiB per batch.")
                            await run_in_threadpool(destination.write, chunk)
                    inputs.append({"name": name, "stored": stored, "size": size})
            # Close multipart files before making this job visible to the worker.
            app.state.store.create(job_id, inputs)
            queued = True
        finally:
            # Only remove this request's fresh UUID directory on failed admission.
            if not queued and root.is_dir():
                try:
                    shutil.rmtree(root)
                except OSError:
                    logger.exception("Could not clean up rejected upload %s", job_id)
        return RedirectResponse(f"/jobs/{job_id}", status_code=303)

    @app.get("/jobs/{job_id}")
    def job_page(request: Request, job_id: str, preview: str | None = None):
        job = app.state.store.get(job_id)
        if not job:
            raise HTTPException(404, "This batch does not exist.")
        return render(request, job=job, preview_id=preview)

    @app.get("/jobs/{job_id}/files/{file_id}/{kind}")
    def download(job_id: str, file_id: str, kind: str):
        job = app.state.store.get(job_id)
        result = next((r for r in job["results"] if r["id"] == file_id), None) if job else None
        if kind not in {"md", "json"} or not result or result["status"] not in {"completed", "partial"}:
            raise HTTPException(404, "Output not found.")
        path = data_dir / job_id / "output" / f"{file_id}.{kind}"
        if not path.is_file():
            raise HTTPException(404, "Output file is missing.")
        name = Path(result["name"]).stem + "." + kind
        return FileResponse(path, filename=name, media_type="application/octet-stream")

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=8000)
