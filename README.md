# DocuQuery

DocuQuery is a local, full-stack document question-answering application. Upload PDFs, then ask questions grounded in their text. It uses no paid LLM or embedding APIs: embeddings run through Sentence Transformers and generation runs through a local Ollama model.

## What is RAG?

Retrieval-Augmented Generation (RAG) retrieves relevant external knowledge before an LLM writes an answer. Rather than asking the LLM to rely on its own general knowledge, DocuQuery gives it the most relevant PDF chunks as context and instructs it to use only that context.

## How the pipeline works

### Indexing phase

```text
PDF
 ↓
Text Extraction (PyMuPDF, page by page)
 ↓
Chunking (with overlap)
 ↓
Embedding (all-MiniLM-L6-v2)
 ↓
FAISS
```

Each chunk keeps its document ID, filename, page number, text, and chunk ID. FAISS stores only vectors, so `metadata.json` remains aligned with FAISS positions. The index and metadata are persisted in `data/` and document deletion rebuilds the index to keep them synchronized.

### Query phase

```text
Question
 ↓
Query Embedding
 ↓
FAISS Similarity Search
 ↓
Top-K Chunks
 ↓
Prompt
 ↓
Ollama LLM
 ↓
Answer + sources
```

The **embedding model** converts text into numerical vectors. The **vector index** (FAISS `IndexFlatL2`) stores those vectors. **Similarity search** returns the closest document vectors to the question vector (smaller L2 distance is closer). The **LLM** is Ollama's configurable local generation model, which turns the retrieved context into a readable answer. Using the same embedding model for documents and questions places both in the same vector space, making distance comparisons meaningful.

## Run locally

1. Create a Python environment and install backend dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy configuration and install the local LLM:
   ```bash
   cp .env.example .env
   ollama pull llama3.2
   ```
3. Start the backend from the repository root:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. Start the React/Vite frontend in another terminal:
   ```bash
   cd frontend && npm install && npm run dev
   ```

Open the address printed by Vite (normally `http://localhost:5173`). Set `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, chunking values, `TOP_K`, or an optional `RELEVANCE_THRESHOLD` in `.env`. A blank relevance threshold disables filtering; otherwise it is the maximum allowed L2 distance.

## API

- `POST /documents/upload` — multipart `file` PDF upload.
- `GET /documents` — list uploaded documents.
- `DELETE /documents/{document_id}` — remove a document and rebuild FAISS.
- `POST /query` — accepts `{ "question": "..." }` and returns an answer and page-aware sources.

## Tests

```bash
pytest
```
