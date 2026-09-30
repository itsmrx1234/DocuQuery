"""Question endpoint: embed -> FAISS -> context -> local Ollama."""
from fastapi import APIRouter, HTTPException, Request

from backend.config import settings
from backend.models.schemas import QueryRequest, QueryResponse, SourceResponse
from backend.services.ollama import OllamaError, generate
from backend.services.retriever import NO_CONTEXT_ANSWER, build_prompt, retrieve

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest, request: Request) -> QueryResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    try:
        chunks = retrieve(request.app.state.vector_store, request.app.state.embedding_service, question, settings.top_k, settings.relevance_threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not search document embeddings: {exc}") from exc
    if not chunks:
        return QueryResponse(answer=NO_CONTEXT_ANSWER, sources=[])
    try:
        answer = generate(settings.ollama_base_url, settings.ollama_model, build_prompt(question, chunks))
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    sources = [SourceResponse(filename=chunk["filename"], page=chunk["page"], distance=chunk["distance"]) for chunk in chunks]
    return QueryResponse(answer=answer, sources=sources)
