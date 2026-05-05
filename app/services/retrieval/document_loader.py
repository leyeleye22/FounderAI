from __future__ import annotations

import json
from pathlib import Path

from app.schemas.knowledge import KnowledgeDocument


class KnowledgeDocumentLoader:
    def __init__(self, knowledge_dir: Path) -> None:
        self.knowledge_dir = knowledge_dir

    def load(self) -> list[KnowledgeDocument]:
        documents: list[KnowledgeDocument] = []
        if not self.knowledge_dir.exists():
            return documents

        for path in self.knowledge_dir.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                for item in payload:
                    documents.append(KnowledgeDocument.model_validate(item))
            elif isinstance(payload, dict):
                documents.append(KnowledgeDocument.model_validate(payload))
        return documents
