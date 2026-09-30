"""Document upload, list, and deletion endpoints."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from backend.config import settings
from backend.models.schemas import DocumentResponse
from backend.services.chunker import chunk_text
from backend.services.pdf_loader import extract_pages

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)) -> DocumentResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    embedding_service, vector_store = request.app.state.embedding_service, request.app.state.vector_store
    suffix = Path(file.filename).suffix
    try:
        with NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(await file.read())
            temp_path = Path(temporary.name)
        pages = extract_pages(temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()

    document_id = uuid4().hex
    metadata = []
    for page in pages:
        for text in chunk_text(str(page["text"]), settings.chunk_size, settings.chunk_overlap):
            metadata.append({"chunk_id": uuid4().hex, "document_id": document_id, "filename": file.filename, "page": page["page"], "text": text})
    try:
        embeddings = embedding_service.encode([chunk["text"] for chunk in metadata])
        for item, embedding in zip(metadata, embeddings):
            item["embedding"] = embedding.tolist()
        vector_store.add(embeddings, metadata)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not create document embeddings: {exc}") from exc
    return DocumentResponse(document_id=document_id, filename=file.filename)


@router.get("", response_model=list[DocumentResponse])
def list_documents(request: Request) -> list[DocumentResponse]:
    seen = set()
    return [DocumentResponse(document_id=item["document_id"], filename=item["filename"]) for item in request.app.state.vector_store.chunk_metadata if not (item["document_id"] in seen or seen.add(item["document_id"]))]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: str, request: Request) -> None:
    if not request.app.state.vector_store.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
