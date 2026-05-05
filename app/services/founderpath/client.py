from __future__ import annotations

import httpx

from app.core.settings import get_settings
from app.domain.project_context import InterviewSnapshot, ModuleSnapshot, ProjectSnapshot


class FounderPathClient:
    """Adapter for FounderPath project data."""

    def get_project_snapshot(self, *, workspace_id: str | None, project_id: str | None) -> ProjectSnapshot:
        if not project_id:
            return ProjectSnapshot(workspace_id=workspace_id, project_id=project_id, project_name="Local project", modules=[])

        settings = get_settings()
        if not settings.founderpath_api_token:
            return ProjectSnapshot(
                workspace_id=workspace_id,
                project_id=project_id,
                project_name="Local project",
                modules=[],
            )

        resolved_workspace_id = workspace_id or self._get_current_workspace_id()
        if not resolved_workspace_id:
            return ProjectSnapshot(workspace_id=workspace_id, project_id=project_id, project_name="Local project", modules=[])

        project_payload = self._get_json(f"/workspaces/{resolved_workspace_id}/projects/{project_id}")
        problem_payload = self._get_json(f"/workspaces/{resolved_workspace_id}/projects/{project_id}/problem")
        interviews_payload = self._get_json(f"/workspaces/{resolved_workspace_id}/projects/{project_id}/interviews")

        project_data = self._unwrap(project_payload)
        interviews_data = self._unwrap_list(interviews_payload)
        step_map = self._extract_step_map(project_data)
        canvas_map = self._extract_canvas_map(project_data)

        modules: list[ModuleSnapshot] = []

        problem_block = problem_payload if isinstance(problem_payload, dict) else {}
        problem_data = problem_block.get("problem") if isinstance(problem_block.get("problem"), dict) else {}
        if (not problem_data or not self._problem_has_content(problem_data)) and step_map.get("define-problem"):
            problem_data = {
                "problemStatement": step_map.get("define-problem"),
                "who": step_map.get("target-user", ""),
            }
        if problem_data:
            modules.append(
                ModuleSnapshot(
                    module="problem_statement",
                    title="Problem statement",
                    summary=self._summarize_problem(problem_data),
                    raw=problem_data,
                )
            )

        if interviews_data:
            modules.append(
                ModuleSnapshot(
                    module="interview",
                    title="Customer interviews",
                    summary=self._summarize_interviews(interviews_data),
                    raw={"items": interviews_data},
                )
            )

        self._append_canvas_modules(modules, canvas_map)

        return ProjectSnapshot(
            workspace_id=resolved_workspace_id,
            project_id=str(project_data.get("uuid") or project_data.get("id") or project_id),
            project_name=str(project_data.get("name") or "Local project"),
            modules=modules,
        )

    def get_interview(
        self,
        *,
        workspace_id: str | None,
        project_id: str | None,
        interview_id: str | None,
    ) -> InterviewSnapshot | None:
        if not project_id or not interview_id:
            return None

        settings = get_settings()
        if not settings.founderpath_api_token:
            return None

        resolved_workspace_id = workspace_id or self._get_current_workspace_id()
        if not resolved_workspace_id:
            return None

        payload = self._get_json(f"/workspaces/{resolved_workspace_id}/projects/{project_id}/interviews/{interview_id}")
        data = self._unwrap(payload)
        if not data:
            return None

        willingness_score = data.get("willingness_score")
        try:
            normalized_score = float(willingness_score) if willingness_score is not None else None
        except (TypeError, ValueError):
            normalized_score = None

        return InterviewSnapshot(
            interview_id=str(data.get("id") or interview_id),
            interview_type=self._as_text(data.get("interview_type")),
            status=self._as_text(data.get("status")),
            contact_name=self._as_text(data.get("contact_name")),
            contact_role=self._as_text(data.get("contact_role")),
            contact_company=self._as_text(data.get("contact_company")),
            research_objectives=self._as_text(data.get("research_objectives")),
            hypotheses=self._as_text(data.get("hypotheses")),
            transcription=self._as_text(data.get("transcription")),
            key_evidence=self._as_text(data.get("key_evidence")),
            notes=self._as_text(data.get("notes")),
            result_signal=self._as_text(data.get("result_signal")),
            willingness_score=normalized_score,
            next_steps=self._as_text(data.get("next_steps")),
        )

    def _headers(self) -> dict[str, str]:
        settings = get_settings()
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.founderpath_api_token}",
        }

    def _get_json(self, path: str) -> dict | list | None:
        settings = get_settings()
        url = f"{settings.founderpath_api_base_url.rstrip('/')}{path}"
        try:
            with httpx.Client(timeout=settings.founderpath_request_timeout_seconds, headers=self._headers()) as client:
                response = client.get(url)
                response.raise_for_status()
                return response.json()
        except Exception:
            return None

    def _get_current_workspace_id(self) -> str | None:
        payload = self._get_json("/workspaces/current")
        data = self._unwrap(payload)
        workspace_id = data.get("uuid") or data.get("id")
        return str(workspace_id) if workspace_id else None

    @staticmethod
    def _unwrap(payload: dict | list | None) -> dict[str, object]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _unwrap_list(payload: dict | list | None) -> list[dict[str, object]]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return [item for item in payload["data"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    @staticmethod
    def _summarize_problem(problem_data: dict[str, object]) -> str:
        parts = [
            str(problem_data.get("who") or "").strip(),
            str(problem_data.get("problemStatement") or "").strip(),
            str(problem_data.get("when") or "").strip(),
        ]
        return " | ".join([part for part in parts if part])

    @staticmethod
    def _summarize_interviews(interviews: list[dict[str, object]]) -> str:
        completed = [
            item for item in interviews
            if str(item.get("status") or "").lower() == "completed"
        ]
        top_contacts = [
            str(item.get("contact_name") or "").strip()
            for item in interviews[:3]
            if str(item.get("contact_name") or "").strip()
        ]
        summary = f"{len(interviews)} interviews"
        if completed:
            summary += f", {len(completed)} completed"
        if top_contacts:
            summary += f", contacts: {', '.join(top_contacts)}"
        return summary

    @staticmethod
    def _as_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _extract_step_map(project_data: dict[str, object]) -> dict[str, str]:
        items = project_data.get("steps")
        result: dict[str, str] = {}
        if not isinstance(items, list):
            return result
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "").strip()
            value = str(item.get("value") or "").strip()
            if key and value:
                result[key] = value
        return result

    @staticmethod
    def _extract_canvas_map(project_data: dict[str, object]) -> dict[str, str]:
        items = project_data.get("canvases")
        result: dict[str, str] = {}
        if not isinstance(items, list):
            return result
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "").strip()
            value = str(item.get("value") or "").strip()
            if key and value:
                result[key] = value
        return result

    @staticmethod
    def _append_canvas_modules(modules: list[ModuleSnapshot], canvas_map: dict[str, str]) -> None:
        mapping = {
            "business-model-canvas": ("business", "Business model canvas"),
            "bmc": ("business", "Business model canvas"),
            "go-to-market": ("gtm", "Go to market"),
            "market_sizing": ("market_sizing", "Market sizing"),
            "tam": ("market_sizing", "Market sizing"),
            "problem-validation": ("problem_validation", "Problem validation"),
            "problem_validation": ("problem_validation", "Problem validation"),
            "icp": ("icp", "ICP"),
        }

        existing = {module.module for module in modules}
        for canvas_key, value in canvas_map.items():
            if canvas_key not in mapping or not value:
                continue
            module_key, title = mapping[canvas_key]
            if module_key in existing:
                continue
            modules.append(
                ModuleSnapshot(
                    module=module_key,
                    title=title,
                    summary=value[:280],
                    raw={"value": value, "canvas_key": canvas_key},
                )
            )
            existing.add(module_key)

    @staticmethod
    def _problem_has_content(problem_data: dict[str, object]) -> bool:
        for value in problem_data.values():
            if str(value or "").strip():
                return True
        return False
