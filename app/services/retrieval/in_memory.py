import json
from pathlib import Path

from app.schemas.common import SourceChunk
from app.services.retrieval.base import BaseRetriever


MODULE_SEARCH_GROUPS: dict[str, list[str]] = {
    "problem-statement": ["problem_statement", "problem_validation"],
    "problem_statement": ["problem_statement", "problem_validation"],
    "problem-validation": ["problem_validation", "problem_statement"],
    "problem_validation": ["problem_validation", "problem_statement"],
    "research": ["problem_validation", "problem_statement"],
    "icp": ["icp"],
    "business": ["business_model"],
    "business-model": ["business_model"],
    "business_model": ["business_model"],
    "competitive-landscape": ["competitive_landscape"],
    "competitive_landscape": ["competitive_landscape"],
    "market-sizing": ["market_sizing"],
    "market_sizing": ["market_sizing"],
    "product": ["business_model"],
    "gtm": ["gtm"],
    "journey": ["user_journey"],
    "user-journey": ["user_journey"],
    "user_journey": ["user_journey"],
    "roi": ["roi"],
    "workshop": ["business_model"],
    "sprints": ["sprints"],
    "sprint": ["sprints"],
    "gamma": ["business_model"],
    "interview": ["interviews"],
    "interviews": ["interviews"],
}


class InMemoryRetriever(BaseRetriever):
    def __init__(self) -> None:
        self._knowledge: dict[str, list[SourceChunk]] = {}
        self._load_all_knowledge()

    def _load_all_knowledge(self) -> None:
        knowledge_dir = Path(__file__).parent.parent.parent.parent / "knowledge"
        if not knowledge_dir.exists():
            self._load_fallback()
            return

        loaded = False
        for json_file in knowledge_dir.rglob("*.json"):
            module_key = json_file.parent.name
            chunks = self._load_json_file(json_file)
            if chunks:
                if module_key not in self._knowledge:
                    self._knowledge[module_key] = []
                self._knowledge[module_key].extend(chunks)
                loaded = True

        if not loaded:
            self._load_fallback()

    def _load_json_file(self, json_file: Path) -> list[SourceChunk]:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunks = []
            for doc in data:
                chunks.append(
                    SourceChunk(
                        title=doc.get("title", "Unknown"),
                        excerpt=doc.get("content", "")[:500],
                        source_type=doc.get("source_type", "rule"),
                    )
                )
            return chunks
        except Exception:
            return []

    def _load_fallback(self) -> None:
        self._knowledge = {
            "problem_statement": [
                SourceChunk(
                    title="Probleme clair",
                    excerpt="Un bon probleme nomme une cible precise, une douleur concrete et un contexte observable.",
                    source_type="rule",
                ),
                SourceChunk(
                    title="Anti-pattern solution",
                    excerpt="Si la phrase commence par une solution ou un produit, le probleme est probablement mal formule.",
                    source_type="rule",
                ),
                SourceChunk(
                    title="Preuve terrain",
                    excerpt="Une bonne redaction du probleme s'appuie idealement sur une preuve, une citation ou un comportement observe.",
                    source_type="rule",
                ),
            ],
        }

    def search(self, *, query: str, module: str, limit: int = 4) -> list[SourceChunk]:
        chunks: list[SourceChunk] = []
        for group_key in MODULE_SEARCH_GROUPS.get(module, [module]):
            chunks.extend(self._knowledge.get(group_key, []))
            if len(chunks) >= limit:
                break
        return chunks[:limit]

    def count(self) -> int:
        return sum(len(items) for items in self._knowledge.values())
