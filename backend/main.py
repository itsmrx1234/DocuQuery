from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import documents, query
from backend.config import settings
from backend.services.embeddings import EmbeddingService
from backend.services.vector_store import VectorStore

app = FastAPI(title="DocuQuery", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.state.embedding_service = EmbeddingService(settings.embedding_model)
app.state.vector_store = VectorStore(settings.data_dir)
app.include_router(documents.router)
app.include_router(query.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
