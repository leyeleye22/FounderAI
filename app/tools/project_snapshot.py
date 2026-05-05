from app.domain.project_context import ProjectSnapshot
from app.services.founderpath.client import FounderPathClient


class ProjectSnapshotTool:
    name = "get_project_snapshot"

    def __init__(self, client: FounderPathClient) -> None:
        self.client = client

    def run(self, *, workspace_id: str | None, project_id: str | None) -> ProjectSnapshot:
        return self.client.get_project_snapshot(workspace_id=workspace_id, project_id=project_id)

