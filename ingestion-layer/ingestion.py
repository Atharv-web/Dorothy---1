"""File preparation and Docling conversion, independent of the web interface."""

import logging
import re
import shutil
import stat
import zipfile
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import rarfile
from defusedxml import ElementTree

from storage import new_id

DOCUMENTS = {".pdf", ".docx", ".xlsx", ".pptx", ".txt"}
IMAGES = {".jpg", ".jpeg", ".png", ".svg", ".webp"}
ARCHIVES = {".zip", ".rar"}
SUPPORTED = DOCUMENTS | IMAGES | ARCHIVES
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_EXPANDED_BYTES = 500 * 1024 * 1024
MAX_FILES = 500
MAX_DEPTH = 3
logger = logging.getLogger(__name__)


def clean_extracted_text(text: str) -> str:
    """Replace ampersand entities, including ones missing a semicolon."""
    return re.sub(r"&amp(?:;|(?![a-zA-Z0-9]))", "and", text, flags=re.IGNORECASE)


def clean_document_text(document):
    """Clean extracted text fields, retaining original text and source metadata."""
    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean_extracted_text(item) if key == "text" and isinstance(item, str)
                else clean(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    return type(document).model_validate(clean(document.model_dump()))


def safe_name(name: str) -> str:
    """Keep relative source paths as metadata, never as disk destinations."""
    name = name.replace("\\", "/")
    parts = PurePosixPath(name).parts
    if (not parts or name.startswith("/") or any(p in {".", ".."} for p in parts)
            or any(ord(c) < 32 or c == ":" for c in name) or len(name) > 1000):
        raise ValueError("Unsafe or invalid file path.")
    return name


@dataclass
class Budget:
    files: int = 0
    size: int = 0


class ArchiveLimitError(ValueError):
    """Stop archive expansion for the batch once its shared budget is exhausted."""


def extract_member(archive, member, destination: Path, budget: Budget):
    """Copy one member with shared limits; retain only a complete source file."""
    try:
        with archive.open(member) as source, destination.open("wb") as target:
            size = 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                budget.size += len(chunk)
                if budget.size > MAX_EXPANDED_BYTES:
                    raise ArchiveLimitError("Expanded batch exceeds 500 MiB.")
                if size > MAX_FILE_BYTES:
                    raise ValueError("Extracted file exceeds 50 MiB.")
                target.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise


class Ingestor:
    def __init__(self):
        self._converter = None
        self._device = None
        # Windows installers often do not add these tools to PATH.
        for tool, candidate in [("UNRAR_TOOL", "C:/Program Files/WinRAR/UnRAR.exe"),
                                ("SEVENZIP_TOOL", "C:/Program Files/7-Zip/7z.exe")]:
            if not shutil.which(getattr(rarfile, tool)) and Path(candidate).is_file():
                setattr(rarfile, tool, candidate)

    @property
    def converter(self):
        if self._converter is None:
            import torch
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
            from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            options = PdfPipelineOptions(
                accelerator_options=AcceleratorOptions(device=self._device),
                ocr_options=RapidOcrOptions(
                    backend="torch" if self._device == "cuda" else "onnxruntime",
                ),
            )
            self._converter = DocumentConverter(allowed_formats=[
                InputFormat.PDF, InputFormat.DOCX,
                InputFormat.XLSX, InputFormat.PPTX, InputFormat.IMAGE,
            ], format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=options),
            })
            logger.info("Document processing device: %s", self._device)
        return self._converter

    def _fallback_to_cpu(self):
        """Keep this worker on CPU after a failed GPU attempt."""
        import gc
        import torch

        self._converter = None
        self._device = "cpu"
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            logger.warning("Could not clear GPU cache", exc_info=True)

    def _initialize_pipelines(self):
        converter = self.converter
        for input_format in converter.allowed_formats:
            converter.initialize_pipeline(input_format)

    def warmup(self):
        """Load reusable pipelines before the worker accepts any documents."""
        started = perf_counter()
        try:
            self._initialize_pipelines()
        except Exception:
            if self._device != "cuda":
                raise
            logger.warning("GPU initialization failed; retrying on CPU", exc_info=True)
            self._fallback_to_cpu()
            self._initialize_pipelines()
        logger.info("Ingestion models ready; startup loading took %.2f seconds", perf_counter() - started)

    def _convert_document(self, source):
        from docling.datamodel.base_models import ConversionStatus

        def run():
            return self.converter.convert(
                source, raises_on_error=False,
                max_num_pages=500, max_file_size=MAX_FILE_BYTES,
            )

        try:
            result = run()
        except Exception:
            if self._device != "cuda" or source.suffix.lower() not in IMAGES | {".pdf"}:
                raise
            logger.warning("GPU conversion failed; retrying on CPU", exc_info=True)
        else:
            if (self._device != "cuda"
                    or source.suffix.lower() not in IMAGES | {".pdf"}
                    or result.status in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}):
                return result
            logger.warning("GPU conversion returned %s; retrying on CPU", result.status)
        self._fallback_to_cpu()
        return run()

    def convert(self, source: Path, output: Path):
        started = perf_counter()
        extension = source.suffix.lower()
        partial = False
        if extension == ".txt":
            from docling_core.types.doc import DocItemLabel, DoclingDocument

            raw = source.read_bytes()
            encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError as exc:
                raise ValueError("Save this text file as UTF-8 or UTF-16 and try again.") from exc
            if "\x00" in text:
                raise ValueError("This file contains null bytes. Upload plain UTF-8 or UTF-16 text.")
            # Text is already extracted; keep it literal in the shared document model.
            document = DoclingDocument(name=source.stem)
            if text:
                document.add_text(label=DocItemLabel.TEXT, text=text)
        else:
            from docling.datamodel.base_models import ConversionStatus

            if extension == ".svg":
                source = self.prepare_svg(source)
            result = self._convert_document(source)
            if result.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
                raise ValueError("Docling could not read this file. It may be damaged, encrypted, or mislabeled.")
            document = result.document
            partial = result.status == ConversionStatus.PARTIAL_SUCCESS
        document = clean_document_text(document)
        markdown = output.with_suffix(".md")
        structured = output.with_suffix(".json")
        try:
            document.save_as_markdown(markdown)
            document.save_as_json(structured)
        except Exception:
            # Do not leave one downloadable-looking output from an incomplete export.
            markdown.unlink(missing_ok=True)
            structured.unlink(missing_ok=True)
            raise
        logger.info("Extraction and export finished for %s in %.2f seconds", source.name, perf_counter() - started)
        return "partial" if partial else "completed", (
            "Some content could not be extracted. Review the output." if partial else ""
        )

    @staticmethod
    def prepare_svg(source):
        import resvg_py

        text = source.read_text(encoding="utf-8-sig")
        root = ElementTree.fromstring(text)
        if root.tag not in {"svg", "{http://www.w3.org/2000/svg}svg"}:
            raise ValueError("The file does not contain an SVG image.")
        # Do not allow SVGs to read local files or fetch external resources.
        for node in root.iter():
            if node.tag.rsplit("}", 1)[-1] in {"script", "foreignObject", "image", "feImage", "style"}:
                raise ValueError("SVG must be self-contained, without scripts, images, or style blocks.")
            for key, value in node.attrib.items():
                if key.rsplit("}", 1)[-1] == "href" and not value.startswith("#"):
                    raise ValueError("SVG external references are not supported.")
                # Local gradients and clipping paths are safe; external CSS URLs are not.
                local_refs_removed = re.sub(r"url\(\s*(['\"]?)#[\w.-]+\1\s*\)", "", value, flags=re.I)
                if re.search(r"url\s*\(|@import|\\", local_refs_removed, re.I):
                    raise ValueError("SVG CSS resource references are not supported.")
        target = source.with_suffix(".png")
        target.write_bytes(resvg_py.svg_to_bytes(svg_string=text, width=2000, height=2000))
        return target

    def process(self, job, root, publish):
        results = []
        budget = Budget()
        output = root / "output"
        output.mkdir(exist_ok=True)

        def record(name, status, message="", file_id=None):
            return {"name": name, "status": status, "message": message, "id": file_id}

        def failure(name, exc):
            if isinstance(exc, rarfile.RarCannotExec):
                message = "RAR needs an extraction tool. Install unrar or 7-Zip and add it to PATH."
            elif isinstance(exc, (rarfile.PasswordRequired, rarfile.RarWrongPassword)):
                message = "Password-protected archives are not supported. Upload an unencrypted copy."
            else:
                message = str(exc)[:400] or "This file could not be processed."
            return record(name, "failed", message)

        def ingest(path, name, depth=0):
            extension = path.suffix.lower()
            if extension not in SUPPORTED:
                yield record(name, "skipped", "Unsupported file type.")
                return
            try:
                if extension in ARCHIVES:
                    if depth >= MAX_DEPTH:
                        raise ValueError("Archive nesting exceeds the limit of 3 levels.")
                    if budget.files >= MAX_FILES or budget.size >= MAX_EXPANDED_BYTES:
                        raise ArchiveLimitError("Archive expansion limit reached for this batch.")
                    archive_type = zipfile.ZipFile if extension == ".zip" else rarfile.RarFile
                    with archive_type(path) as archive:
                        members = archive.infolist()
                        has_files = False
                        for member in members:
                            budget.files += 1
                            if budget.files > MAX_FILES:
                                raise ArchiveLimitError("Batch exceeds 500 archive entries.")
                            member_name = f"{name}/{member.filename}"
                            try:
                                member_name = f"{name}/{safe_name(member.filename)}"
                                is_zip = extension == ".zip"
                                link = (stat.S_ISLNK(member.external_attr >> 16) if is_zip
                                        else member.is_symlink() or bool(getattr(member, "file_redir", None)))
                                if link:
                                    raise ValueError("Archive links are not supported.")
                                is_directory = member.is_dir() if is_zip else member.isdir()
                                if is_directory:
                                    continue
                                has_files = True
                                suffix = PurePosixPath(member_name).suffix.lower()
                                if suffix not in SUPPORTED:
                                    yield record(member_name, "skipped", "Unsupported file type.")
                                    continue
                                encrypted = bool(member.flag_bits & 1) if is_zip else member.needs_password()
                                if encrypted:
                                    raise ValueError("Password-protected archives are not supported. Upload an unencrypted copy.")
                                if member.file_size > MAX_FILE_BYTES:
                                    raise ValueError("File exceeds 50 MiB.")
                                if budget.size + member.file_size > MAX_EXPANDED_BYTES:
                                    raise ArchiveLimitError("Expanded batch exceeds 500 MiB.")
                                extracted = root / "source" / (new_id() + suffix)
                                extract_member(archive, member, extracted, budget)
                                yield from ingest(extracted, member_name, depth + 1)
                            except ArchiveLimitError:
                                raise
                            except Exception as exc:
                                has_files = True
                                yield failure(member_name, exc)
                        if not has_files:
                            yield record(name, "skipped", "Archive contains no files.")
                    return
                file_id = new_id()
                status, message = self.convert(path, output / file_id)
                item = record(name, status, message, file_id)
                item["source"] = path.relative_to(root).as_posix()
                yield item
            except ArchiveLimitError:
                raise
            except Exception as exc:
                logger.exception("Ingestion failed for %s", name)
                yield failure(name, exc)

        for source in job["inputs"]:
            try:
                for item in ingest(root / "source" / source["stored"], source["name"]):
                    results.append(item)
                    # Storage errors must fail the job, not be reported as bad documents.
                    publish(results)
            except ArchiveLimitError as exc:
                budget.files = MAX_FILES
                results.append(failure(source["name"], exc))
                publish(results)
        return results
