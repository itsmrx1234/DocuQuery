import pytest

pytest.importorskip("fastapi")
pytest.importorskip("faiss")
fitz = pytest.importorskip("fitz")
pytest.importorskip("sentence_transformers")
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.vector_store import VectorStore


class FakeEmbeddings:
    def encode(self, texts):
        import numpy as np
        return np.array([[float(len(text)), 1.0] for text in texts], dtype="float32")


@pytest.fixture()
def client(tmp_path):
    app.state.vector_store = VectorStore(tmp_path)
    app.state.embedding_service = FakeEmbeddings()
    return TestClient(app)


def pdf_bytes():
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Paris is the capital of France.")
    return document.tobytes()


def test_upload_and_invalid_pdf(client):
    response = client.post("/documents/upload", files={"file": ("facts.pdf", pdf_bytes(), "application/pdf")})
    assert response.status_code == 201
    assert client.get("/documents").json()[0]["filename"] == "facts.pdf"
    assert client.post("/documents/upload", files={"file": ("nope.txt", b"no", "text/plain")}).status_code == 400


def test_empty_question_and_no_documents(client):
    assert client.post("/query", json={"question": "   "}).status_code == 422
    assert client.post("/query", json={"question": "Where is Paris?"}).status_code == 400


def test_ollama_unavailable_returns_helpful_error(client, monkeypatch):
    client.post("/documents/upload", files={"file": ("facts.pdf", pdf_bytes(), "application/pdf")})
    monkeypatch.setattr("backend.api.query.generate", lambda *args: (_ for _ in ()).throw(__import__("backend.services.ollama", fromlist=["OllamaError"]).OllamaError("Ollama is unavailable.")))
    assert client.post("/query", json={"question": "What is France's capital?"}).status_code == 503
