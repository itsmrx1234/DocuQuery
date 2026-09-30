"""Explicit FAISS retrieval and prompt construction."""

NO_CONTEXT_ANSWER = "I couldn't find enough relevant information in the uploaded documents to answer this question."


def retrieve(vector_store, embedding_service, question: str, top_k: int, threshold: float | None) -> list[dict]:
    query_embedding = embedding_service.encode([question])
    distances, indices = vector_store.search(query_embedding, top_k)
    results = []
    for distance, index in zip(distances[0], indices[0]):
        if index < 0 or (threshold is not None and float(distance) > threshold):
            continue
        metadata = vector_store.chunk_metadata[int(index)]
        results.append({**metadata, "distance": float(distance)})
    return results


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {chunk['filename']}, Page {chunk['page']}]\n{chunk['text']}" for chunk in chunks
    )
    return f"""You are DocuQuery, a document question-answering assistant.

Answer the user's question using ONLY the provided context. Do not use outside knowledge.
If the answer cannot be found in the provided context, say that the information is not available in the uploaded documents.

Context:

{context}

Question:
{question}

Answer:
"""
