from backend.services.chunker import chunk_text


def test_chunking_generates_overlapping_chunks():
    chunks = chunk_text("abcdefghij", chunk_size=6, overlap=2)
    assert chunks == ["abcdef", "efghij", "ij"]


def test_chunking_handles_empty_input():
    assert chunk_text("   ", chunk_size=10, overlap=2) == []
