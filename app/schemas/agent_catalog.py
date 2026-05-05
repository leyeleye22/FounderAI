from pydantic import BaseModel


class AgentCapabilityItem(BaseModel):
    module_key: str
    label: str
    capabilities: list[str]


class AgentCatalogResponse(BaseModel):
    orchestrator: str
    modules: list[AgentCapabilityItem]
