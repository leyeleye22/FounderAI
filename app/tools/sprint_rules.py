from app.schemas.common import SourceChunk
from app.services.retrieval.base import BaseRetriever


class SprintRulesTool:
    name = "search_sprint_rules"

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def run(self, *, query: str) -> list[SourceChunk]:
        return self.retriever.search(query=query, module="sprints")

