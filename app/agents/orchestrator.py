from app.domain.module_catalog import MODULE_CATALOG
from app.schemas.agent_catalog import AgentCapabilityItem, AgentCatalogResponse


class FounderOrchestrator:
    agent_name = "founder_copilot"

    def catalog(self) -> AgentCatalogResponse:
        return AgentCatalogResponse(
            orchestrator=self.agent_name,
            modules=[
                AgentCapabilityItem(
                    module_key=item.module_key,
                    label=item.label,
                    capabilities=list(item.capabilities),
                )
                for item in MODULE_CATALOG
            ],
        )
