"""
Load reference material from project references/ (PDFs, .md, .txt) for use by the theorist.
"""

from pathlib import Path
from typing import Optional

from src.config import references_dir

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
            continue

    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                parts.append(text)
        except Exception:
            continue
    return "\n".join(parts)
