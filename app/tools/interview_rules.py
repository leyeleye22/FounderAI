from app.schemas.common import SourceChunk
from app.services.retrieval.base import BaseRetriever


class InterviewRulesTool:
    name = "search_interview_rules"

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def run(self, *, query: str) -> list[SourceChunk]:
        return self.retriever.search(query=query, module="interview")

