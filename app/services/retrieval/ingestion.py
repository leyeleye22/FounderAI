from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.services.retrieval.chroma_store import ChromaStore
from app.services.retrieval.chunker import KnowledgeChunker
from app.services.retrieval.document_loader import KnowledgeDocumentLoader


@dataclass(slots=True)
class IngestionReport:
    documents: int
    chunks: int
    collection: str


class KnowledgeIngestionService:
    def __init__(self, *, knowledge_dir: Path, store: ChromaStore) -> None:
        self.loader = KnowledgeDocumentLoader(knowledge_dir)
        self.chunker = KnowledgeChunker()
        self.store = store

    def ingest(self) -> IngestionReport:
        documents = self.loader.load()
        chunks = self.chunker.chunk(documents)
        self.store.upsert(chunks)
        return IngestionReport(
            documents=len(documents),
            chunks=len(chunks),
            collection=self.store.collection_name,
        )
