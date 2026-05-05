# FounderAI Implementation Plan

## Mission

Build a local-first AI copilot for FounderPath that can:

- challenge a problem statement
- analyze interviews
- propose BMC blocks
- estimate TAM / SAM / SOM
- review competitors
- generate sprints
- help with ROI and GTM

## Product constraints

- The assistant must work across the full project journey.
- It must read project context before suggesting content.
- It must output structured JSON for every screen.
- It must never save silently without an explicit validation step.

## Delivery order

### Phase 1

- scaffold API service
- load local Qwen3 model configuration
- define tool contracts
- create first heuristic + model-ready `problem_challenger`

### Phase 2

- add RAG ingestion contracts
- add embedding and reranking adapters
- connect to FounderPath project context APIs

### Phase 3

- add `interview_analyst`
- add `sprint_planner`
- add `bmc_assistant`

### Phase 4

- add `competitor_analyst`
- add `market_size_assistant`
- add `roi_assistant`

### Phase 5

- feedback logging
- acceptance / rejection capture
- evaluation dataset
- future fine-tuning preparation

## Current MVP

The first executable MVP is a FastAPI service with:

- `GET /health`
- `POST /agents/problem/challenge`

The response is a strict JSON contract suitable for FounderPath UI.

