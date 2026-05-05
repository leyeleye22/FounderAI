from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas.common import SourceChunk
from app.schemas.knowledge import KnowledgeChunk
from app.services.retrieval.base import BaseRetriever


class ChromaStore(BaseRetriever):
    def __init__(self, *, persist_directory: Path, collection_name: str, embedding_model: str) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = None
        self._collection = None

    def is_available(self) -> bool:
        try:
            self._ensure_collection()
            return True
        except Exception:
            return False

    def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            return
        collection = self._ensure_collection()
        ids = [item.chunk_id for item in chunks]
        documents = [item.content for item in chunks]
        metadatas = [self._metadata(item) for item in chunks]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def count(self) -> int:
        collection = self._ensure_collection()
        return int(collection.count())

    def search(self, *, query: str, module: str, limit: int = 4) -> list[SourceChunk]:
        collection = self._ensure_collection()
        where = {"module": module} if module else None
        result = collection.query(
            query_texts=[query],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        items: list[SourceChunk] = []
        for document, metadata in zip(documents, metadatas, strict=False):
            metadata = metadata or {}
            items.append(
                SourceChunk(
                    title=str(metadata.get("title") or "Knowledge"),
                    excerpt=str(document),
                    source_type=str(metadata.get("source_type") or "knowledge"),
                )
            )
        return items

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection

        self.persist_directory.mkdir(parents=True, exist_ok=True)

        import chromadb
        from chromadb.utils import embedding_functions

        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))

        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function,
        )
        return self._collection

    @staticmethod
    def _metadata(chunk: KnowledgeChunk) -> dict[str, Any]:
        return {
            "document_id": chunk.document_id,
            "module": chunk.module,
            "field": chunk.field,
            "title": chunk.title,
            "language": chunk.language,
            "source_type": chunk.source_type,
            "tags": ", ".join(chunk.tags),
        }
