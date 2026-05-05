# FounderAI Architecture

FounderAI is the AI copilot layer for FounderPath.

It is designed around four principles:

1. One orchestrator, many tools.
2. Retrieval before generation.
3. Structured outputs before free-form prose.
4. Human validation before persistence.

## Core layers

### 1. API layer

FastAPI routes expose health, chat, and agent actions.

### 2. Agent layer

The orchestrator receives the user intent, selects the right tool chain,
and returns a typed response.

### 3. Tool layer

Tools read and write FounderPath project data:

- project snapshot
- module answers
- interviews
- sprints
- ROI
- competitive landscape
- notes and files

### 4. RAG layer

The retrieval stack injects:

- field instructions
- good and bad examples
- formulas
- challenge prompts
- past validated outputs

### 5. Model layer

Qwen3-4B is the generation model.
Embeddings and reranking are separate services.

## First execution target

The first production-grade agent is `problem_challenger`.

It should:

- diagnose clarity issues
- identify missing proof
- ask short challenge questions
- propose a rewritten version
- return strict JSON

## Next agents

- interview_analyst
- sprint_planner
- bmc_assistant
- competitor_analyst
- market_size_assistant
- roi_assistant

