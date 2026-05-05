from functools import lru_cache
from pathlib import Path

from app.agents.conversational import ConversationalAgent
from app.agents.interview_analyst import InterviewAnalystAgent
from app.agents.orchestrator import FounderOrchestrator
from app.agents.problem_challenger import ProblemChallengerAgent
from app.agents.sprint_planner import SprintPlannerAgent
from app.core.settings import get_settings
from app.services.founderpath.client import FounderPathClient
from app.services.retrieval.chroma_store import ChromaStore
from app.services.retrieval.in_memory import InMemoryRetriever
from app.services.retrieval.base import BaseRetriever
from app.tools.interview_reader import InterviewReaderTool
from app.tools.interview_rules import InterviewRulesTool
from app.tools.problem_rules import ProblemRulesTool
from app.tools.project_snapshot import ProjectSnapshotTool
from app.tools.sprint_rules import SprintRulesTool


@lru_cache(maxsize=1)
def get_problem_challenger() -> ProblemChallengerAgent:
    founderpath_client = FounderPathClient()
    retriever = get_retriever()

    return ProblemChallengerAgent(
        project_snapshot_tool=ProjectSnapshotTool(founderpath_client),
        problem_rules_tool=ProblemRulesTool(retriever),
    )


@lru_cache(maxsize=1)
def get_orchestrator() -> FounderOrchestrator:
    return FounderOrchestrator()


@lru_cache(maxsize=1)
def get_interview_analyst() -> InterviewAnalystAgent:
    founderpath_client = FounderPathClient()
    retriever = get_retriever()
    return InterviewAnalystAgent(
        project_snapshot_tool=ProjectSnapshotTool(founderpath_client),
        interview_reader_tool=InterviewReaderTool(founderpath_client),
        interview_rules_tool=InterviewRulesTool(retriever),
    )


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    settings = get_settings()
    if settings.force_in_memory_retrieval:
        return InMemoryRetriever()

    chroma_store = ChromaStore(
        persist_directory=Path(settings.rag_dir),
        collection_name=settings.rag_collection,
        embedding_model=settings.embedding_model,
    )
    if chroma_store.is_available() and chroma_store.count() > 0:
        return chroma_store
    return InMemoryRetriever()


@lru_cache(maxsize=1)
def get_sprint_planner() -> SprintPlannerAgent:
    founderpath_client = FounderPathClient()
    retriever = get_retriever()
    return SprintPlannerAgent(
        project_snapshot_tool=ProjectSnapshotTool(founderpath_client),
        sprint_rules_tool=SprintRulesTool(retriever),
    )


@lru_cache(maxsize=1)
def get_conversational_agent() -> ConversationalAgent:
    founderpath_client = FounderPathClient()
    retriever = get_retriever()
    return ConversationalAgent(
        project_snapshot_tool=ProjectSnapshotTool(founderpath_client),
        retriever=retriever,
    )
