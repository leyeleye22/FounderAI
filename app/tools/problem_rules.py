from app.schemas.common import SourceChunk
from app.services.retrieval.base import BaseRetriever


class ProblemRulesTool:
    name = "search_problem_rules"

    def __init__(self, retriever: BaseRetriever) -> None:
        self.retriever = retriever

    def run(self, *, statement: str) -> list[SourceChunk]:
        return self.retriever.search(query=statement, module="problem_statement")

