from fastapi import APIRouter

from app.registry import get_interview_analyst, get_orchestrator, get_problem_challenger
from app.schemas.agent_catalog import AgentCatalogResponse
from app.schemas.interview import InterviewAnalysisRequest, InterviewAnalysisResponse
from app.schemas.problem import ProblemChallengeRequest, ProblemChallengeResponse
from app.schemas.sprint import SprintPlanRequest, SprintPlanResponse
from app.registry import get_sprint_planner


router = APIRouter()


@router.get("/catalog", response_model=AgentCatalogResponse)
def agent_catalog() -> AgentCatalogResponse:
    return get_orchestrator().catalog()


@router.post("/problem/challenge", response_model=ProblemChallengeResponse)
def challenge_problem(payload: ProblemChallengeRequest) -> ProblemChallengeResponse:
    agent = get_problem_challenger()
    return agent.run(payload)


@router.post("/interview/analyze", response_model=InterviewAnalysisResponse)
def analyze_interview(payload: InterviewAnalysisRequest) -> InterviewAnalysisResponse:
    agent = get_interview_analyst()
    return agent.run(payload)


@router.post("/sprint/plan", response_model=SprintPlanResponse)
def plan_sprint(payload: SprintPlanRequest) -> SprintPlanResponse:
    agent = get_sprint_planner()
    return agent.run(payload)
