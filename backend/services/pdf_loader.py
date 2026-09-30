"""Page-aware PDF extraction using PyMuPDF."""
from pathlib import Path

import fitz


def extract_pages(pdf_path: Path) -> list[dict[str, object]]:
    """Extract non-empty text from each page, using one-based page numbers."""
    try:
        with fitz.open(pdf_path) as document:
            if document.page_count == 0:
                raise ValueError("The PDF has no pages.")
            pages = [
                {"page": number, "text": page.get_text("text").strip()}
                for number, page in enumerate(document, start=1)
                if page.get_text("text").strip()
            ]
    except (fitz.FileDataError, RuntimeError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable PDF.") from exc
    if not pages:
        raise ValueError("No extractable text was found in this PDF.")
    return pages
