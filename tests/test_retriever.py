from backend.services.retriever import build_prompt, retrieve


class FakeEmbeddings:
    def encode(self, texts):
        return [[0.0, 1.0]]


class FakeStore:
    chunk_metadata = [
        {"filename": "facts.pdf", "page": 1, "text": "Python is a programming language."},
        {"filename": "facts.pdf", "page": 2, "text": "Paris is the capital of France."},
    ]

    def search(self, embedding, k):
        return [[0.1, 0.4]], [[1, 0]]


def test_retrieval_maps_faiss_positions_to_paris_chunk():
    chunks = retrieve(FakeStore(), FakeEmbeddings(), "What is the capital of France?", 2, None)
    assert "Paris" in chunks[0]["text"]
    assert "[Source: facts.pdf, Page 2]" in build_prompt("What is the capital?", chunks)
