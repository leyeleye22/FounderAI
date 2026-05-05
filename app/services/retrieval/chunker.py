from __future__ import annotations

import hashlib

from app.schemas.knowledge import KnowledgeChunk, KnowledgeDocument


class KnowledgeChunker:
    def chunk(self, documents: list[KnowledgeDocument]) -> list[KnowledgeChunk]:
        chunks: list[KnowledgeChunk] = []
        for document in documents:
            content = document.content.strip()
            if not content:
                continue
            for index, part in enumerate(self._split_content(content)):
                chunk_id = self._chunk_id(document.document_id, index, part)
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        module=document.module,
                        field=document.field,
                        title=document.title,
                        content=part,
                        language=document.language,
                        source_type=document.source_type,
                        tags=document.tags,
                    )
                )
        return chunks

    @staticmethod
    def _split_content(content: str, max_chars: int = 600) -> list[str]:
        normalized = " ".join(content.split())
        if len(normalized) <= max_chars:
            return [normalized]

        parts: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + max_chars, len(normalized))
            parts.append(normalized[start:end].strip())
            start = end
        return parts

    @staticmethod
    def _chunk_id(document_id: str, index: int, content: str) -> str:
        digest = hashlib.sha1(f"{document_id}:{index}:{content}".encode("utf-8")).hexdigest()[:12]
        return f"{document_id}:{index}:{digest}"
