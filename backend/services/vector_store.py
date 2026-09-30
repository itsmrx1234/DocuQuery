"""FAISS index plus its deliberately parallel metadata mapping."""
import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.index_path = data_dir / "index.faiss"
        self.metadata_path = data_dir / "metadata.json"
        self.index: faiss.Index | None = None
        self.chunk_metadata: list[dict] = []
        self.load()

    def load(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata_path.exists():
            self.chunk_metadata = json.loads(self.metadata_path.read_text())
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.index is not None and self.index.ntotal != len(self.chunk_metadata):
            raise RuntimeError("Stored FAISS index and metadata are out of sync.")

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(embeddings) != len(metadata):
            raise ValueError("Every embedding must have matching chunk metadata.")
        if not len(embeddings):
            return
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        if self.index.d != embeddings.shape[1]:
            raise ValueError("Embedding dimensions do not match the existing index.")
        self.index.add(np.ascontiguousarray(embeddings, dtype="float32"))
        self.chunk_metadata.extend(metadata)
        self.persist()

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("No documents have been uploaded yet.")
        return self.index.search(np.ascontiguousarray(query_embedding, dtype="float32"), min(k, self.index.ntotal))

    def delete_document(self, document_id: str) -> bool:
        retained = [item for item in self.chunk_metadata if item["document_id"] != document_id]
        if len(retained) == len(self.chunk_metadata):
            return False
        self.chunk_metadata = retained
        self.rebuild_index()
        return True

    def rebuild_index(self) -> None:
        self.index = None
        if self.chunk_metadata:
            vectors = np.array([item["embedding"] for item in self.chunk_metadata], dtype="float32")
            self.index = faiss.IndexFlatL2(vectors.shape[1])
            self.index.add(vectors)
        self.persist()

    def persist(self) -> None:
        serializable = self.chunk_metadata
        self.metadata_path.write_text(json.dumps(serializable, indent=2))
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        elif self.index_path.exists():
            self.index_path.unlink()
