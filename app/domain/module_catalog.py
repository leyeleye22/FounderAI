from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModuleCapability:
    module_key: str
    label: str
    capabilities: tuple[str, ...]


MODULE_CATALOG: tuple[ModuleCapability, ...] = (
    ModuleCapability("problem_statement", "Problem", ("challenge", "rewrite", "extract_gaps")),
    ModuleCapability("problem_validation", "Validation", ("review", "summarize", "extract_proofs")),
    ModuleCapability("research", "Research", ("summarize", "extract_patterns", "build_evidence")),
    ModuleCapability("icp", "ICP", ("propose", "sharpen", "compare_segments")),
    ModuleCapability("business", "BMC", ("propose", "review", "challenge")),
    ModuleCapability("competitive_landscape", "Competitors", ("analyze", "compare", "differentiate")),
    ModuleCapability("market_sizing", "Market Size", ("estimate", "explain_assumptions", "challenge_inputs")),
    ModuleCapability("product", "Product", ("summarize_metrics", "interpret", "flag_risks")),
    ModuleCapability("gtm", "GTM", ("propose", "prioritize_channels", "visualize")),
    ModuleCapability("journey", "Journey", ("summarize", "find_gaps", "improve_flow")),
    ModuleCapability("interview", "Interviews", ("summarize", "extract_insights", "cluster_objections")),
    ModuleCapability("roi", "ROI", ("estimate", "explain", "stress_test")),
    ModuleCapability("workshop", "Workshop", ("organize", "summarize", "prepare_next_actions")),
    ModuleCapability("sprints", "Sprints", ("generate", "refine", "sequence_tasks")),
    ModuleCapability("gamma", "Pitch", ("outline", "structure", "condense_story")),
)

