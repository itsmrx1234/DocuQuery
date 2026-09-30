"""Simple, explicit character-based chunking with overlap."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Return non-empty overlapping chunks while preserving the source page."""
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    cleaned = text.strip()
    step = chunk_size - overlap
    return [cleaned[start : start + chunk_size] for start in range(0, len(cleaned), step) if cleaned[start : start + chunk_size].strip()]
