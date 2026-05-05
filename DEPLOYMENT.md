# FounderAI Deployment and CI/CD Plan

## Goal

Deploy FounderAI as a stable API service while keeping trained model updates releasable without rebuilding the whole application around local machine paths.

## Recommended Architecture

Use three distinct layers:

1. Front and proxy on Vercel
2. Laravel API on Hostinger
3. FounderAI as a separate Python service

Runtime flow:

`Browser -> Next proxy on Vercel -> FounderAI public URL -> Laravel API + LLM backend`

## Why separate the AI service

FounderAI is not just a thin endpoint:

- it is a FastAPI application
- it contains retrieval logic
- it can point to local or remote model assets
- it may evolve independently from the Next frontend

That makes a container-based deployment model a better fit than a serverless-only setup.

## Deployment Model

### Application release

The FounderAI API is deployed as a Docker image:

- built from `Dockerfile`
- pushed to `ghcr.io/<owner>/founderai`
- deployed by your runtime platform from that image

### Model release

The trained model update is released separately as a versioned bundle:

- produced by `scripts/build_model_bundle.py`
- uploaded by `.github/workflows/model-bundle.yml`
- includes training data metadata, adapter folders when present, and config files

This separation matters because:

- app code changes should not require rebuilding giant model artifacts
- model updates should not require changing application code
- rollback becomes easier because app image version and model bundle version can move independently

## Environments

Use at least:

- `local`
- `staging`
- `production`

## Required Environment Variables

### FounderAI service

- `FOUNDER_AI_ENV`
- `FOUNDER_AI_HOST`
- `FOUNDER_AI_PORT`
- `FOUNDER_AI_MODEL_DIR`
- `FOUNDER_AI_RAG_DIR`
- `FOUNDERPATH_API_BASE_URL`
- `FOUNDERPATH_API_TOKEN`
- `FOUNDER_AI_USE_LOCAL_HEURISTICS`
- `LLM_API_BASE_URL`
- `LLM_MODEL_NAME`
- `USE_FINETUNED_MODEL`
- `FINETUNED_MODEL_PATH`
- `LORA_ADAPTER_PATH`

### Frontend on Vercel

- `FOUNDER_AI_BASE_URL`
- `APP_API_BASE_URL`

Important:

- production should not rely on `127.0.0.1`
- production should not rely on local Windows file paths

## CI Pipeline

Workflow: `.github/workflows/ci.yml`

Checks:

- install Python dependencies
- compile Python files
- run `pytest`
- validate API contract through route and chat tests

This protects the frontend contract for:

- `/health`
- `/agents/catalog`
- `/agents/problem/challenge`
- `/agents/interview/analyze`
- `/agents/sprint/plan`
- `/agents/chat`

## Container Release Pipeline

Workflow: `.github/workflows/container-release.yml`

Trigger:

- push to `main`
- version tags like `v1.0.0`
- manual dispatch

Output:

- Docker image pushed to GHCR
- immutable image tags based on commit SHA
- `latest` tag for the default branch

## Model Bundle Pipeline

Workflow: `.github/workflows/model-bundle.yml`

Trigger:

- changes in training or adapter folders on `main`
- manual dispatch with optional bundle version

Output:

- versioned manifest
- `.tar.gz` bundle in `dist/model-bundles`
- uploaded workflow artifact

## Recommended Rollout Strategy

### API rollout

1. Merge to `main`
2. CI must pass
3. Container image is built and pushed
4. Deploy staging from the new image
5. Run smoke tests
6. Promote to production

### Model rollout

1. Train or update adapter
2. Commit training metadata and packaging logic as needed
3. Run `model-bundle` workflow
4. Publish or store the produced bundle
5. Mount or copy bundle to the deployment target
6. Update:
   - `USE_FINETUNED_MODEL=true`
   - `FINETUNED_MODEL_PATH`
   - `LORA_ADAPTER_PATH`
7. Restart FounderAI
8. Run smoke tests against `/agents/chat`

## Recommended Smoke Tests

At minimum after deployment:

- `GET /health`
- `GET /agents/catalog`
- `POST /agents/chat` with a valid payload from the frontend contract

## Rollback Strategy

Keep rollback simple:

- revert to previous Docker image tag for application issues
- revert to previous model bundle version for model-quality regressions

This is exactly why app image and model bundle should be versioned independently.

## Practical V1 Recommendation

For V1, use this stack:

- Front proxy: Vercel
- Laravel: Hostinger
- FounderAI API: container host or VPS
- Model runtime: remote OpenAI-compatible endpoint first, local fine-tuned bundle later

This gives you a clean path:

- ship API now
- train and improve the model later
- deploy model updates without redesigning the system
