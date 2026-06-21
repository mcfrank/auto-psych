"""
Load reference material from project references/ (PDFs, .md, .txt) for use by the theorist.
"""

import logging
from pathlib import Path
from typing import Optional

from src.runtime.config import references_dir

logger = logging.getLogger(__name__)

# Cap total reference text to avoid blowing context
MAX_REFERENCE_CHARS = 80_000


def load_references(project_id: str) -> str:
    """
    Load all reference content from references_dir: extract text from PDFs,
    read .md and .txt as UTF-8. Returns a single string with sections per file,
    truncated to MAX_REFERENCE_CHARS if needed.
    """
    ref_dir = references_dir(project_id)
    if not ref_dir.exists():
        return ""

    chunks: list[str] = []
    total = 0

    # Order: PDFs first, then .md, then .txt (arbitrary but deterministic)
    pdfs = sorted(ref_dir.glob("*.pdf"))
    mds = sorted(ref_dir.glob("*.md"))
    txts = sorted(ref_dir.glob("*.txt"))

    for path in pdfs + mds + txts:
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            header = f"## From: {path.name}\n\n"
            chunk = header + text.strip()
            if total + len(chunk) > MAX_REFERENCE_CHARS:
                remaining = max(0, MAX_REFERENCE_CHARS - total - len(header) - 100)
                if remaining > 0:
                    chunk = header + text[:remaining] + "\n\n[... truncated]\n"
                    chunks.append(chunk)
                    total += len(chunk)
                break
            chunks.append(chunk)
            total += len(chunk)
        except Exception:
            # A corrupt/unreadable reference file is skipped, but log it — a
            # silently dropped reference vanishes from the theorist's context.
            logger.warning("skipping unreadable reference %s", path, exc_info=True)
            continue

    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        # pypdf is a declared dependency; a missing import means a broken env, not
        # "this PDF has no text". Fail loudly rather than silently dropping every
        # PDF reference from the theorist's context.
        raise RuntimeError(
            f"cannot extract text from {path}: pypdf is not installed "
            "(it is a project dependency — run `uv sync`)"
        ) from exc
    reader = PdfReader(path)
    parts = []
    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                parts.append(text)
        except Exception:
            # A single unextractable page should not lose the whole PDF, but the
            # partial loss must be visible rather than silent.
            logger.warning(
                "skipping unextractable page %d of %s", page_num, path, exc_info=True
            )
            continue
    return "\n".join(parts)
