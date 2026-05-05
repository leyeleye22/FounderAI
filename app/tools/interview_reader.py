from app.domain.project_context import InterviewSnapshot
from app.services.founderpath.client import FounderPathClient


class InterviewReaderTool:
    name = "read_interview"

    def __init__(self, client: FounderPathClient) -> None:
        self.client = client

    def run(
        self,
        *,
        workspace_id: str | None,
        project_id: str | None,
        interview_id: str | None,
    ) -> InterviewSnapshot | None:
        return self.client.get_interview(
            workspace_id=workspace_id,
            project_id=project_id,
            interview_id=interview_id,
        )

